"""Persistencia SQLite. Las salidas crudas se guardan siempre.

Esa regla —guardar el texto entero de cada invocación, aunque abulte— es la que
sostiene media arquitectura: permite re-juzgar con otro juez, recalibrar un
umbral o cambiar una rúbrica **sin volver a invocar ningún modelo**. Sin ella,
cada cambio de criterio costaría otra tirada completa y el proyecto no sería
practicable.

Por eso `judgement` es una tabla aparte de `invocation` y no una columna suya.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 4

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

-- Una tirada: una suite contra uno o varios modelos, con su manifiesto.
CREATE TABLE IF NOT EXISTS run (
    run_id        TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    suite_id      TEXT NOT NULL,
    suite_hash    TEXT NOT NULL,
    label         TEXT,
    manifest_json TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'running'
);

-- Cada invocación, con su salida CRUDA y su huella de ejecución.
CREATE TABLE IF NOT EXISTS invocation (
    invocation_id  TEXT PRIMARY KEY,
    run_id         TEXT NOT NULL REFERENCES run(run_id),
    case_id        TEXT NOT NULL,
    model_key      TEXT NOT NULL,
    repetition     INTEGER NOT NULL,
    variant_id     TEXT,
    broker_task_id TEXT,
    ok             INTEGER NOT NULL,
    output_text    TEXT NOT NULL DEFAULT '',
    tool_calls_json TEXT,
    tokens_input   INTEGER NOT NULL DEFAULT 0,
    tokens_output  INTEGER NOT NULL DEFAULT 0,
    cost_usd       REAL NOT NULL DEFAULT 0,
    latency_ms     REAL NOT NULL DEFAULT 0,
    model_served   TEXT,
    fallback_used  INTEGER NOT NULL DEFAULT 0,
    fingerprint_json TEXT,
    applied_params_json TEXT,
    -- Evidencia del contrato 2.10 del Broker (Client_API.md, 8.1/8.3/8.4/8.5).
    -- Se guarda por invocación porque es por invocación como se demuestra: una
    -- tirada que no puede enseñar esto no puede sostener lo que afirma.
    prompt_compression TEXT,  -- 8.5: la poda que se aplicó DE VERDAD
    content_source TEXT,      -- 8.1: de qué campo salió el texto
    deliverable_sha256 TEXT,  -- 8.3: sha256 del artefacto `final: true`
    auxiliary_invocations INTEGER NOT NULL DEFAULT 0,  -- 8.4: quién más lo vio
    auxiliary_roles_json TEXT,
    error_code     TEXT,
    error_message  TEXT,
    created_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_invocation_run ON invocation(run_id, case_id, model_key);

-- Agregación del piso 1: una fila por caso, modelo y métrica. Es la
-- observación que entra en los contrastes, no la invocación.
CREATE TABLE IF NOT EXISTS case_result (
    run_id        TEXT NOT NULL REFERENCES run(run_id),
    case_id       TEXT NOT NULL,
    model_key     TEXT NOT NULL,
    metric        TEXT NOT NULL,
    case_score    REAL,
    case_stability REAL,
    repetitions   INTEGER NOT NULL DEFAULT 0,
    failed_repetitions INTEGER NOT NULL DEFAULT 0,
    is_unstable   INTEGER NOT NULL DEFAULT 0,
    is_not_applicable INTEGER NOT NULL DEFAULT 0,
    not_applicable_reason TEXT,
    values_json   TEXT,
    PRIMARY KEY (run_id, case_id, model_key, metric)
);

-- Juicios, desacoplados de la invocación para poder reejecutarlos.
CREATE TABLE IF NOT EXISTS judgement (
    judgement_id   TEXT PRIMARY KEY,
    run_id         TEXT NOT NULL,
    case_id        TEXT NOT NULL,
    left_invocation  TEXT,
    right_invocation TEXT,
    judge_instrument TEXT NOT NULL,
    rubric_instrument TEXT,
    presented_order  TEXT NOT NULL,
    verdict          TEXT NOT NULL,
    reason           TEXT,
    reruns_judgement TEXT,
    created_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_judgement_run ON judgement(run_id, case_id);

-- Métricas del propio juez. Se publican; no se suponen.
CREATE TABLE IF NOT EXISTS judge_metrics (
    run_id            TEXT NOT NULL,
    judge_instrument  TEXT NOT NULL,
    position_bias     REAL,
    judge_self_consistency REAL,
    judge_human_agreement  REAL,
    human_human_agreement  REAL,
    kappa_kind        TEXT,
    calibrated        INTEGER NOT NULL DEFAULT 0,
    detail_json       TEXT,
    PRIMARY KEY (run_id, judge_instrument)
);

-- Comparaciones ya calculadas, con los dos niveles en columnas distintas.
CREATE TABLE IF NOT EXISTS comparison (
    baseline_run_id TEXT NOT NULL,
    candidate_run_id TEXT NOT NULL,
    case_id          TEXT NOT NULL,
    metric           TEXT NOT NULL,
    change_status    TEXT NOT NULL,
    change_direction TEXT NOT NULL,
    evidence_kind    TEXT,
    evidence_confidence REAL,
    baseline_score   REAL,
    candidate_score  REAL,
    effect_size      REAL,
    ci_low           REAL,
    ci_high          REAL,
    p_value          REAL,
    p_value_corrected REAL,
    practical_significance TEXT,
    comparability    TEXT NOT NULL,
    taxonomy_json    TEXT,
    notes_json       TEXT,
    PRIMARY KEY (baseline_run_id, candidate_run_id, case_id, metric)
);
"""


SCHEMA += """
CREATE TABLE IF NOT EXISTS comparison_review (
    review_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    baseline_run_id TEXT NOT NULL,
    candidate_run_id TEXT NOT NULL,
    baseline_model_key TEXT NOT NULL,
    candidate_model_key TEXT NOT NULL,
    review_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_comparison_review_runs
    ON comparison_review(baseline_run_id, candidate_run_id);
CREATE TABLE IF NOT EXISTS calibration_run (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    use_case TEXT NOT NULL,
    status TEXT NOT NULL,
    expired INTEGER NOT NULL DEFAULT 0,
    report_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS calibration_case (
    run_id TEXT NOT NULL REFERENCES calibration_run(run_id),
    case_id TEXT NOT NULL,
    partition_name TEXT NOT NULL,
    case_json TEXT NOT NULL,
    teacher_json TEXT NOT NULL,
    learner_json TEXT NOT NULL,
    PRIMARY KEY (run_id, case_id)
);
"""


class Database:
    """Envoltorio fino sobre sqlite3. Sin ORM: el esquema es el contrato."""

    def __init__(self, path: str | Path = "state/model_drift.sqlite") -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path))
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self._migrate()

    #: Columnas añadidas después de la versión 1 del esquema. `CREATE TABLE IF
    #: NOT EXISTS` no toca una tabla que ya existe, así que una base creada
    #: antes se quedaría sin ellas y las tiradas viejas seguirían abriéndose
    #: pero fallarían al insertar. Se añaden aquí, que es aditivo y sin pérdida.
    _COLUMNAS_NUEVAS: tuple[tuple[str, str, str], ...] = (
        ("invocation", "prompt_compression", "TEXT"),
        ("invocation", "content_source", "TEXT"),
        ("invocation", "deliverable_sha256", "TEXT"),
        ("invocation", "auxiliary_invocations", "INTEGER NOT NULL DEFAULT 0"),
        ("invocation", "auxiliary_roles_json", "TEXT"),
    )

    def _migrate(self) -> None:
        with self.connection:
            self.connection.executescript(SCHEMA)
            for tabla, columna, tipo in self._COLUMNAS_NUEVAS:
                existentes = {
                    fila["name"]
                    for fila in self.connection.execute(f"PRAGMA table_info({tabla})").fetchall()
                }
                if columna not in existentes:
                    self.connection.execute(f"ALTER TABLE {tabla} ADD COLUMN {columna} {tipo}")
            fila = self.connection.execute("SELECT version FROM schema_version").fetchone()
            if fila is None:
                self.connection.execute("INSERT INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,))
            elif fila["version"] != SCHEMA_VERSION:
                self.connection.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------- escritura

    def insert(self, table: str, values: dict[str, Any]) -> None:
        columnas = ", ".join(values)
        marcas = ", ".join("?" for _ in values)
        with self.connection:
            self.connection.execute(
                f"INSERT OR REPLACE INTO {table} ({columnas}) VALUES ({marcas})",
                tuple(_adapt(value) for value in values.values()),
            )

    def insert_many(self, table: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        columnas = ", ".join(rows[0])
        marcas = ", ".join("?" for _ in rows[0])
        with self.connection:
            self.connection.executemany(
                f"INSERT OR REPLACE INTO {table} ({columnas}) VALUES ({marcas})",
                [tuple(_adapt(value) for value in fila.values()) for fila in rows],
            )

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        with self.connection:
            return self.connection.execute(sql, params)

    # ------------------------------------------------------------- lectura

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        return list(self.connection.execute(sql, params).fetchall())

    def query_one(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        return self.connection.execute(sql, params).fetchone()

    def runs(self, limit: int = 50) -> list[sqlite3.Row]:
        return self.query("SELECT * FROM run ORDER BY created_at DESC LIMIT ?", (limit,))


def _adapt(value: Any) -> Any:
    """Convierte tipos de Python a algo que SQLite acepte."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value
