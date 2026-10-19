"""Regresión de la auditoría humana del 15-oct-2026 (motor y CLI).

Cada prueba nombra la incidencia que protege. Las de la ventana viven en
`test_gui.py`, junto a sus fixtures de Tk.
"""
from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from model_drift import cli
from model_drift.broker_contracts import ModelReference
from model_drift.cases import load_suite
from model_drift.comparison import compare_runs
from model_drift.config import AppConfig
from model_drift.db import Database
from model_drift.domain import ChangeStatus
from model_drift.fingerprint import ExecutionFingerprint, FingerprintBuilder
from model_drift.runner import RunConfig, Runner
from tests.broker_fake import FakeBroker

VIEJO = ModelReference(provider="ollama", deployment="local", model="qwen3.6:35b")
NUEVO = ModelReference(provider="ollama", deployment="local", model="qwen3.8:27b")

SUITE = {
    "id": "qa-mini",
    "cases": [
        {"id": f"c{i}", "prompt": f"pregunta {i}", "assertions": [{"type": "json_valid"}]} for i in range(1, 4)
    ],
}


@pytest.fixture()
def ruta_suite(tmp_path: Path) -> Path:
    destino = tmp_path / "suite.yaml"
    destino.write_text(yaml.safe_dump(SUITE, allow_unicode=True), encoding="utf-8")
    return destino


def _broker(**opciones: object) -> FakeBroker:
    broker = FakeBroker(**opciones)  # type: ignore[arg-type]
    broker.add_model(VIEJO.key)
    broker.add_model(NUEVO.key)
    broker.respond_fixed(VIEJO.key, '{"ok": true}')
    broker.respond_fixed(NUEVO.key, "ya no devuelvo JSON")
    return broker


def _tirada(broker: FakeBroker, ruta: Path, db: Database, modelos: list[ModelReference], reps: int = 6):  # type: ignore[no-untyped-def]
    with broker.client() as cliente:
        return Runner(cliente, load_suite(ruta), db=db, config=RunConfig(repetitions=reps)).run(modelos)


def _huella(modelo: str, *, tokenizer: str, runtime_version: str = "0.40.3") -> ExecutionFingerprint:
    """La huella real que dio el Broker 2.11 para los dos qwen de la auditoría."""
    return (
        FingerprintBuilder()
        .declared("provider", "ollama")
        .declared("deployment", "local")
        .declared("model", modelo)
        .verified("digest", f"digest-de-{modelo}")
        .verified("family", "qwen35moe" if "35b" in modelo else "qwen35")
        .verified("tokenizer", tokenizer)
        .verified("chat_template", "b507b9c2f6ca642bffcd06665ea7c91f")
        .verified("runtime", "ollama")
        .verified("runtime_version", runtime_version)
        .build()
    )


# --------------------------------------------------------------------- INC-01


def test_inc01_dos_modelos_con_tokenizer_distinto_si_se_comparan(ruta_suite: Path, tmp_path: Path) -> None:
    """El caso de uso insignia: antes salía siempre «NO comparables: tokenizer»."""
    with Database(tmp_path / "md.sqlite") as db:
        base = _tirada(_broker(), ruta_suite, db, [VIEJO])
        cand = _tirada(_broker(), ruta_suite, db, [NUEVO])
    base.manifest.models[0].fingerprint = _huella(VIEJO.model, tokenizer="0edfc799")
    cand.manifest.models[0].fingerprint = _huella(NUEVO.model, tokenizer="e1ef6a85")

    informe = compare_runs(base.results[VIEJO.key], cand.results[NUEVO.key], suite=load_suite(ruta_suite))
    assert informe.comparability.comparable, informe.comparability.headline()
    # Y la estadística corre: el modelo nuevo dejó de devolver JSON.
    assert any(item.change_status is ChangeStatus.cambio_significativo for item in informe.outcomes)


def test_inc01_el_runtime_sigue_teniendo_que_coincidir(ruta_suite: Path, tmp_path: Path) -> None:
    """Excluir lo del modelo no puede abrir la puerta a cambiar el instrumento."""
    with Database(tmp_path / "md.sqlite") as db:
        base = _tirada(_broker(), ruta_suite, db, [VIEJO])
        cand = _tirada(_broker(), ruta_suite, db, [NUEVO])
    base.manifest.models[0].fingerprint = _huella(VIEJO.model, tokenizer="a")
    cand.manifest.models[0].fingerprint = _huella(NUEVO.model, tokenizer="b", runtime_version="0.41.0")

    informe = compare_runs(base.results[VIEJO.key], cand.results[NUEVO.key])
    assert not informe.comparability.comparable
    assert "runtime_version" in informe.comparability.headline()
    assert "tokenizer" not in informe.comparability.headline()


def test_inc01_con_el_mismo_modelo_un_tokenizer_distinto_bloquea(ruta_suite: Path, tmp_path: Path) -> None:
    """«Mismo model_id, distinta ejecución»: aquí sí es la señal que se busca."""
    with Database(tmp_path / "md.sqlite") as db:
        base = _tirada(_broker(), ruta_suite, db, [VIEJO])
        cand = _tirada(_broker(), ruta_suite, db, [VIEJO])
    base.manifest.models[0].fingerprint = _huella(VIEJO.model, tokenizer="a")
    cand.manifest.models[0].fingerprint = _huella(VIEJO.model, tokenizer="b")

    informe = compare_runs(base.results[VIEJO.key], cand.results[VIEJO.key])
    assert not informe.comparability.comparable
    assert "tokenizer" in informe.comparability.headline()


# --------------------------------------------------------------------- INC-02


def test_inc02_token_rechazado_aborta_y_marca_la_tirada_fallida(ruta_suite: Path, tmp_path: Path) -> None:
    with Database(tmp_path / "md.sqlite") as db:
        salida = _tirada(_broker(reject_admin_token=True), ruta_suite, db, [VIEJO, NUEVO])
        fila = db.query_one("SELECT status, manifest_json FROM run WHERE run_id = ?", (salida.run_id,))

    # Al primer rechazo se para: el resto fallaría igual.
    assert salida.invocations == 1
    assert salida.status == "failed"
    assert salida.aborted_reason and "credencial" in salida.aborted_reason
    assert salida.errors == {"AUTH: ADMIN_AUTH_REQUIRED": 1}
    assert salida.failure_summary() == "1/1 invocaciones sin salida (AUTH: ADMIN_AUTH_REQUIRED ×1)"
    assert fila is not None and fila["status"] == "failed"
    assert "ADMIN_AUTH_REQUIRED" in fila["manifest_json"], "el motivo queda en las notas del manifiesto"


def test_inc02_sin_ninguna_salida_es_fallida_sin_abortar(ruta_suite: Path, tmp_path: Path) -> None:
    """Un fallo que no es de credencial no corta la tirada, pero tampoco es éxito."""
    broker = _broker()
    broker.respond_fixed(VIEJO.key, "")
    with Database(tmp_path / "md.sqlite") as db:
        salida = _tirada(broker, ruta_suite, db, [VIEJO], reps=5)
    assert salida.invocations == 15 and salida.failed_invocations == 15
    assert salida.status == "failed" and salida.aborted_reason is None
    # Respuestas vacías idénticas no son «determinismo observado».
    assert not any("Determinismo observado: sí" in linea for linea in salida.manifest.reproducibility.declare())


def test_inc02_una_tirada_sana_no_cambia(ruta_suite: Path, tmp_path: Path) -> None:
    with Database(tmp_path / "md.sqlite") as db:
        salida = _tirada(_broker(), ruta_suite, db, [VIEJO], reps=5)
    assert salida.status == "completed" and salida.failure_summary() is None


# ---------------------------------------------------------------- INC-03/08


@pytest.mark.parametrize("repeticiones", [0, -1])
def test_inc03_repeticiones_menores_que_uno_se_rechazan(repeticiones: int) -> None:
    with pytest.raises(ValueError, match="repeticiones"):
        RunConfig(repetitions=repeticiones)


def test_inc08_presupuesto_negativo_se_rechaza() -> None:
    with pytest.raises(ValueError, match="presupuesto"):
        RunConfig(repetitions=5, budget_usd=-1)


# ------------------------------------------------------------------ CLI


@pytest.fixture()
def cli_entorno(tmp_path: Path, ruta_suite: Path, monkeypatch):  # type: ignore[no-untyped-def]
    ruta_config = tmp_path / "model_drift.yaml"
    AppConfig(database=str(tmp_path / "md.sqlite"), repetitions=5, accept_learning_contamination=True).save(ruta_config)
    opciones: dict[str, object] = {}
    monkeypatch.setattr(cli, "_client", lambda config: _broker(**opciones).client())
    return ruta_config, ruta_suite, opciones


def _ejecutar(ruta_config: Path, ruta_suite: Path, *extra: str) -> int:
    return cli.main(["--config", str(ruta_config), "ejecutar", str(ruta_suite), *extra])


def _run_id(texto: str) -> str:
    encontrado = re.search(r"run-\d{8}-\d{6}-[0-9a-f]{6}", texto)
    assert encontrado, texto
    return encontrado.group(0)


def test_inc02_cli_token_rechazado_sale_con_error(cli_entorno, capsys) -> None:  # type: ignore[no-untyped-def]
    ruta_config, ruta_suite, opciones = cli_entorno
    opciones["reject_admin_token"] = True
    assert _ejecutar(ruta_config, ruta_suite, "--modelo", VIEJO.key) == cli.EXIT_ERROR
    salida = capsys.readouterr()
    assert "FALLOS" in salida.out and "ADMIN_AUTH_REQUIRED" in salida.out
    assert "Determinismo observado: sí" not in salida.out
    assert "no midió el modelo" in salida.err

    run_id = _run_id(salida.out)
    cli.main(["--config", str(ruta_config), "tiradas"])
    assert "[failed]" in capsys.readouterr().out

    # Y no se deja comparar como si hubiera medido algo.
    _ejecutar(ruta_config, ruta_suite, "--modelo", VIEJO.key)  # sigue rechazada
    capsys.readouterr()
    assert cli.main(["--config", str(ruta_config), "comparar", run_id, run_id]) == cli.EXIT_ERROR
    assert "falló entera" in capsys.readouterr().err


def test_inc03_cli_repeticiones_cero_es_error_claro(cli_entorno, capsys) -> None:  # type: ignore[no-untyped-def]
    ruta_config, ruta_suite, _ = cli_entorno
    assert _ejecutar(ruta_config, ruta_suite, "--modelo", VIEJO.key, "--repeticiones", "0") == cli.EXIT_ERROR
    assert "repeticiones deben ser 1 o más" in capsys.readouterr().err
    with Database(Path(ruta_config).parent / "md.sqlite") as db:
        assert db.runs() == [], "no se guarda ninguna tirada vacía"


def test_inc06_cli_tirada_multimodelo_exige_elegir_modelo(cli_entorno, capsys) -> None:  # type: ignore[no-untyped-def]
    ruta_config, ruta_suite, _ = cli_entorno
    _ejecutar(ruta_config, ruta_suite, "--modelo", VIEJO.key, "--modelo", NUEVO.key)
    multi = _run_id(capsys.readouterr().out)
    _ejecutar(ruta_config, ruta_suite, "--modelo", VIEJO.key)
    solo = _run_id(capsys.readouterr().out)

    assert cli.main(["--config", str(ruta_config), "comparar", solo, multi]) == cli.EXIT_ERROR
    error = capsys.readouterr().err
    assert "varios modelos" in error and VIEJO.key in error and NUEVO.key in error

    codigo = cli.main(
        [
            "--config", str(ruta_config), "comparar", solo, multi,
            "--modelo-candidato", NUEVO.key, "--suite", str(ruta_suite),
        ]
    )
    assert codigo == cli.EXIT_OK
    assert "NO comparables" not in capsys.readouterr().out, "INC-01: dos qwen distintos sí se comparan"


def test_inc04_tirada_sin_resultados_da_un_mensaje_que_se_entiende() -> None:
    with pytest.raises(ValueError, match="no tiene resultados comparables"):
        cli._elegir_modelo("run-x", {}, None)


def test_inc07_las_fechas_se_enseñan_en_hora_local() -> None:
    esperado = datetime(2026, 10, 15, 12, 44, 28, tzinfo=UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    assert cli.fecha_local("2026-10-15T12:44:28.581090+00:00") == esperado
    assert cli.fecha_local("no-es-fecha") == "no-es-fecha"


def test_cli_presupuesto_cero_es_un_tope_no_un_olvido(cli_entorno, capsys) -> None:  # type: ignore[no-untyped-def]
    """`--presupuesto 0` se ignoraba (`0 or config`), igual que `--repeticiones 0`."""
    ruta_config, ruta_suite, _ = cli_entorno
    assert _ejecutar(ruta_config, ruta_suite, "--modelo", VIEJO.key, "--presupuesto", "0") == cli.EXIT_ERROR
    salida = capsys.readouterr().out
    assert "presupuesto agotado" in salida and "invocaciones 0" in salida
    # Hallado en la prueba en vivo: sin invocaciones decía «Semilla fijada: sí».
    assert "Semilla fijada: no" in salida
