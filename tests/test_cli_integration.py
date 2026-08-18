"""La CLI de extremo a extremo: ejecutar dos tiradas y compararlas.

Es el camino que va a recorrer una persona de verdad, así que se prueba entero
contra el Broker simulado: `ejecutar`, `tiradas`, `comparar` y el informe HTML.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from model_drift import cli
from model_drift.config import AppConfig
from tests.broker_fake import FakeBroker

VIEJO = "ollama/local/qwen3:6b"
NUEVO = "ollama/local/qwen3:8b"

SUITE = {
    "id": "cli-demo",
    "cases": [
        {"id": f"c{i}", "prompt": f"pregunta {i}", "assertions": [{"type": "json_valid"}]}
        for i in range(1, 5)
    ],
}


@pytest.fixture()
def entorno(tmp_path: Path, monkeypatch):  # type: ignore[no-untyped-def]
    """Configuración y suite en disco, con el Broker simulado inyectado."""
    ruta_suite = tmp_path / "suite.yaml"
    ruta_suite.write_text(yaml.safe_dump(SUITE, allow_unicode=True), encoding="utf-8")
    ruta_config = tmp_path / "model_drift.yaml"
    AppConfig(
        database=str(tmp_path / "md.sqlite"),
        # Con 3 repeticiones el contraste exacto no puede bajar de p=0,1 y ningún
        # caso llega a decidirse. Cinco es el mínimo que permite concluir algo.
        repetitions=5,
        accept_learning_contamination=True,
    ).save(ruta_config)

    estado: dict[str, FakeBroker] = {}

    def _fabricar(config: AppConfig) -> object:
        broker = FakeBroker()
        broker.add_model(VIEJO)
        broker.add_model(NUEVO)
        broker.respond_fixed(VIEJO, '{"ok": true}')
        # El modelo nuevo deja de devolver JSON: degradación clara y con oráculo.
        broker.respond_fixed(NUEVO, "Claro, aquí va la respuesta en prosa.")
        estado["broker"] = broker
        return broker.client()

    monkeypatch.setattr(cli, "_client", _fabricar)
    return ruta_config, ruta_suite, tmp_path


def _run_id(salida: str) -> str:
    match = re.search(r"(run-\d{8}-\d{6}-[0-9a-f]{6})", salida)
    assert match, f"no se encontró el identificador de tirada en:\n{salida}"
    return match.group(1)


def test_camino_completo_de_la_cli(entorno, capsys) -> None:  # type: ignore[no-untyped-def]
    ruta_config, ruta_suite, tmp_path = entorno

    assert cli.main(["--config", str(ruta_config), "ejecutar", str(ruta_suite), "--modelo", VIEJO]) == 0
    base = _run_id(capsys.readouterr().out)

    assert cli.main(["--config", str(ruta_config), "ejecutar", str(ruta_suite), "--modelo", NUEVO]) == 0
    salida_cand = capsys.readouterr().out
    candidata = _run_id(salida_cand)
    # Contra un Broker al día no falta ninguna garantía, y la tirada lo dice
    # sin pasarse: semilla fijada, pero reproducibilidad exacta sin demostrar.
    assert "Garantías NO disponibles" not in salida_cand
    assert "Semilla fijada: sí" in salida_cand
    assert "Reproducibilidad exacta: NO demostrada" in salida_cand

    assert cli.main(["--config", str(ruta_config), "tiradas"]) == 0
    listado = capsys.readouterr().out
    assert base in listado and candidata in listado

    informe = tmp_path / "informe.html"
    codigo = cli.main(
        [
            "--config",
            str(ruta_config),
            "comparar",
            base,
            candidata,
            "--suite",
            str(ruta_suite),
            "--informe",
            str(informe),
        ]
    )
    texto = capsys.readouterr().out
    assert "Veredicto" in texto
    assert informe.is_file()
    html = informe.read_text(encoding="utf-8")
    assert "regresión" in html, "hay oráculo determinista: la dirección sí se puede afirmar"
    # Sin política declarada nada bloquea: una métrica que nadie declaró no
    # puede cortar una entrega.
    assert codigo == 0


def test_una_politica_bloqueante_devuelve_codigo_2(entorno, capsys) -> None:  # type: ignore[no-untyped-def]
    """Así una tirada puede cortar una entrega desde CI."""
    ruta_config, ruta_suite, tmp_path = entorno

    cli.main(["--config", str(ruta_config), "ejecutar", str(ruta_suite), "--modelo", VIEJO])
    base = _run_id(capsys.readouterr().out)
    cli.main(["--config", str(ruta_config), "ejecutar", str(ruta_suite), "--modelo", NUEVO])
    candidata = _run_id(capsys.readouterr().out)

    politica = tmp_path / "politica.yaml"
    politica.write_text(
        yaml.safe_dump({"blocking": {"pass_rate": {"max_drop_pp": 0, "minimum_practical_effect_pp": 1}}}),
        encoding="utf-8",
    )

    codigo = cli.main(
        [
            "--config",
            str(ruta_config),
            "comparar",
            base,
            candidata,
            "--suite",
            str(ruta_suite),
            "--politica",
            str(politica),
        ]
    )
    # 3, no 2: argparse ya usa el 2 para los errores de uso.
    assert codigo == cli.EXIT_BLOCKED == 3
    assert "Bloqueado" in capsys.readouterr().out


def test_comparar_una_tirada_que_no_existe(entorno, capsys) -> None:  # type: ignore[no-untyped-def]
    ruta_config, _, _ = entorno
    with pytest.raises(SystemExit):
        cli.main(["--config", str(ruta_config), "comparar", "run-inventada", "run-tampoco"])


def test_listado_de_modelos(entorno, capsys) -> None:  # type: ignore[no-untyped-def]
    ruta_config, _, _ = entorno
    assert cli.main(["--config", str(ruta_config), "modelos"]) == 0
    salida = capsys.readouterr().out
    assert VIEJO in salida and NUEVO in salida
    # El triestado se enseña: '?' no es lo mismo que 'no'.
    assert "?" in salida
    assert "no aplicable" in salida


def test_el_presupuesto_interrumpe_la_tirada(entorno, capsys) -> None:  # type: ignore[no-untyped-def]
    ruta_config, ruta_suite, _ = entorno
    codigo = cli.main(
        [
            "--config",
            str(ruta_config),
            "ejecutar",
            str(ruta_suite),
            "--modelo",
            VIEJO,
            "--presupuesto",
            "0.0000001",
        ]
    )
    assert codigo == 1
    assert "INTERRUMPIDA" in capsys.readouterr().out


def test_contra_un_broker_antiguo_la_cli_se_niega_a_contaminar(tmp_path: Path, capsys, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """El aviso solo debe saltar si de verdad va a contaminar.

    Antes se disparaba siempre, porque la bandera nacía optimista y nadie la
    comprobaba. Ahora se sondea el contrato del Broker y el aviso queda para
    los que aún no tienen el Cambio 4.
    """
    ruta_suite = tmp_path / "suite.yaml"
    ruta_suite.write_text(yaml.safe_dump(SUITE, allow_unicode=True), encoding="utf-8")
    ruta_config = tmp_path / "model_drift.yaml"
    AppConfig(database=str(tmp_path / "md.sqlite"), repetitions=5).save(ruta_config)

    def _antiguo(config: AppConfig) -> object:
        broker = FakeBroker()
        broker.supports_learning_exclusion = False
        broker.add_model(VIEJO)
        broker.respond_fixed(VIEJO, '{"ok": true}')
        return broker.client()

    monkeypatch.setattr(cli, "_client", _antiguo)
    codigo = cli.main(["--config", str(ruta_config), "ejecutar", str(ruta_suite), "--modelo", VIEJO])
    assert codigo == 1
    error = capsys.readouterr().err
    assert "estadísticas del router adaptativo" in error
    assert "--acepto-contaminar" in error
