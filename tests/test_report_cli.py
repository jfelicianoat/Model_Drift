"""Pruebas del informe y de la línea de comandos.

El informe es donde la honestidad del sistema se ve o no se ve, así que aquí se
comprueba lo que **no** debe aparecer: las palabras «mejora» y «regresión»
sobre un caso sin evidencia evaluativa.
"""
from __future__ import annotations

from pathlib import Path

from model_drift.cli import main
from model_drift.comparison import ComparisonReport
from model_drift.config import AppConfig
from model_drift.domain import (
    ChangeDirection,
    ChangeStatus,
    ComparisonOutcome,
    EvidenceKind,
    PracticalSignificance,
    RunVerdict,
)
from model_drift.judge import JudgeMetrics
from model_drift.manifest import ComparabilityIssue, ComparabilityReport, RunManifest
from model_drift.policy import RunVerdictReport
from model_drift.report import direction_label, render_comparison, render_text_summary, write_comparison


def _informe(*, comparable: bool = True) -> ComparisonReport:
    return ComparisonReport(
        baseline_run_id="run-a",
        candidate_run_id="run-b",
        comparability=ComparabilityReport(
            comparable=comparable,
            issues=[] if comparable else [ComparabilityIssue(kind="huella", detail="difiere chat_template")],
        ),
        outcomes=[
            ComparisonOutcome(
                case_id="con-oraculo",
                metric="pass_rate",
                change_status=ChangeStatus.cambio_significativo,
                change_direction=ChangeDirection.regresion,
                evidence_kind=EvidenceKind.oraculo_determinista,
                evidence_confidence=1.0,
                baseline_score=1.0,
                candidate_score=0.0,
                effect_size=-1.0,
                p_value_corrected=0.001,
                practical_significance=PracticalSignificance.detectado_relevante,
                change_taxonomy=["formato"],
            ),
            ComparisonOutcome(
                case_id="capturado-sin-evidencia",
                metric="output_match",
                change_status=ChangeStatus.cambio_significativo,
                baseline_score=1.0,
                candidate_score=0.0,
                effect_size=-1.0,
                notes=["sin evidencia evaluativa: la salida histórica no es verdad de referencia"],
            ),
        ],
        verdict=RunVerdictReport(verdict=RunVerdict.blocked, blocking_exceeded=["pass_rate"]),
        n_cases=2,
        n_compared=2,
        minimum_detectable_effect=0.35,
    )


# --------------------------------------------------------------------------
# La regla del vocabulario
# --------------------------------------------------------------------------


def test_sin_evidencia_el_informe_no_escribe_regresion_ni_mejora() -> None:
    outcome = ComparisonOutcome(
        case_id="c", metric="output_match", change_status=ChangeStatus.cambio_significativo
    )
    etiqueta = direction_label(outcome)
    assert etiqueta == "cambio sin dirección evaluable"
    assert "regres" not in etiqueta
    assert "mejor" not in etiqueta


def test_el_html_marca_el_caso_con_oraculo_y_no_el_capturado() -> None:
    html = render_comparison(_informe())
    assert "regresión" in html, "el caso con oráculo sí lleva dirección"
    assert "cambio sin dirección evaluable" in html
    # El caso sin evidencia aparece, pero sin veredicto de calidad.
    fila_capturada = html.split("capturado-sin-evidencia")[1][:400]
    assert "regresión" not in fila_capturada
    assert "mejora" not in fila_capturada


def test_la_cabecera_de_comparabilidad_va_antes_que_el_veredicto() -> None:
    html = render_comparison(_informe(comparable=False))
    posicion_aviso = html.index("no son comparables")
    posicion_veredicto = html.index("Veredicto")
    assert posicion_aviso < posicion_veredicto
    assert "pudo cambiarla el instrumento de medida" in html


def test_el_informe_declara_potencia_y_garantias_ausentes() -> None:
    manifiesto = RunManifest(run_id="run-b")
    manifiesto.declare_missing("el Broker no admite generation.seed")
    html = render_comparison(_informe(), manifest=manifiesto)
    assert "Garantías no disponibles" in html
    assert "generation.seed" in html
    assert "menor degradación detectable" in html
    assert "Reproducibilidad exacta: NO demostrada" in html


def test_el_informe_del_juez_distingue_no_medido_de_cero() -> None:
    """Un cero es un dato; «no medida» es una ignorancia. No se colapsan."""
    metricas = JudgeMetrics(position_bias=0.0, judge_self_consistency=None)
    html = render_comparison(_informe(), judge_metrics=metricas)
    assert "0.000" in html
    assert "no medida" in html


def test_el_resumen_de_terminal_separa_lo_accionable_de_lo_sin_direccion() -> None:
    texto = render_text_summary(_informe())
    assert "Regresiones relevantes" in texto
    assert "con-oraculo" in texto
    assert "SIN dirección evaluable" in texto
    assert "capturado-sin-evidencia" in texto


def test_escritura_del_informe_a_disco(tmp_path: Path) -> None:
    destino = write_comparison(_informe(), tmp_path / "sub" / "informe.html")
    assert destino.is_file()
    assert "<h1>" in destino.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------------


def test_la_configuracion_exige_identidad_completa_del_modelo() -> None:
    config = AppConfig()
    referencia = config.model_reference("ollama/local/qwen3:8b")
    assert referencia.key == "ollama/local/qwen3:8b"
    for malo in ("qwen3:8b", "ollama/qwen3:8b", "a//b"):
        try:
            config.model_reference(malo)
        except ValueError as error:
            assert "proveedor/despliegue/modelo" in str(error)
        else:  # pragma: no cover - solo salta si se rompe la validación
            raise AssertionError(f"{malo!r} debería haberse rechazado")


def test_el_token_del_broker_nunca_sale_del_fichero(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """El Broker rota su token en cada arranque: guardarlo en disco sobra y caduca."""
    ruta = tmp_path / "model_drift.yaml"
    AppConfig(broker_url="http://x:1").save(ruta)
    contenido = ruta.read_text(encoding="utf-8")
    assert "token" not in contenido

    monkeypatch.setenv("MODEL_DRIFT_BROKER_TOKEN", "secreto")
    cargada = AppConfig.load(ruta)
    assert cargada.broker_token == "secreto"
    assert cargada.broker_config().admin_token == "secreto"


def test_configuracion_ausente_usa_los_valores_por_defecto(tmp_path: Path) -> None:
    config = AppConfig.load(tmp_path / "no-existe.yaml")
    assert config.repetitions == 3
    assert config.accept_learning_contamination is False


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_potencia_declara_lo_que_no_puede_detectar(capsys) -> None:  # type: ignore[no-untyped-def]
    assert main(["potencia", "--casos", "12"]) == 0
    salida = capsys.readouterr().out
    assert "Curva de potencia" in salida
    assert "falta de potencia" in salida


def test_tiradas_sin_nada_guardado(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    config = tmp_path / "model_drift.yaml"
    AppConfig(database=str(tmp_path / "md.sqlite")).save(config)
    assert main(["--config", str(config), "tiradas"]) == 0
    assert "No hay tiradas" in capsys.readouterr().out


def test_un_error_no_vuelca_una_traza_al_usuario(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    config = tmp_path / "model_drift.yaml"
    AppConfig(database=str(tmp_path / "md.sqlite")).save(config)
    codigo = main(["--config", str(config), "ejecutar", str(tmp_path / "no-existe.yaml"), "--modelo", "a/b/c"])
    assert codigo == 1
    assert "Traceback" not in capsys.readouterr().err
