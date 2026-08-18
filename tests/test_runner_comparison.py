"""Prueba de extremo a extremo: ejecutar dos tiradas y compararlas.

Cubre los criterios de aceptación de las fases 1 y 2 que no son estadísticos
puros, y en particular los casos límite del plan que se resuelven aquí:

- **CASO 1** — la línea base histórica era incorrecta: la app no puede llamar
  regresión a la diferencia.
- **CASO 4** — mismo `model_id` con distinta plantilla de chat: la huella lo
  detecta y la comparación no atribuye nada al modelo.
- **CASO 5** — misma semilla, salidas distintas: se declara sin fingir.
- **CASO 8** — mejora la exactitud y empeoran latencia y coste: `trade_off`.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from model_drift.broker_contracts import ModelReference
from model_drift.cases import load_suite
from model_drift.comparison import compare_runs
from model_drift.db import Database
from model_drift.domain import ChangeDirection, ChangeStatus, RunVerdict
from model_drift.manifest import Determinism, ExactReproducibility, InstrumentRef
from model_drift.policy import RegressionPolicy
from model_drift.runner import METRIC_PASS, RunConfig, Runner
from tests.broker_fake import FakeBroker

VIEJO = ModelReference(provider="ollama", deployment="local", model="qwen3:6b")
NUEVO = ModelReference(provider="ollama", deployment="local", model="qwen3:8b")

SUITE = {
    "id": "demo-facturacion",
    "version": "1",
    "case_pass_rule": "todas",
    "cases": [
        {
            "id": "json-01",
            "prompt": "Extrae el total de la factura.",
            "assertions": [{"type": "json_valid"}],
        },
        {
            "id": "json-02",
            "prompt": "Devuelve el importe en JSON.",
            "assertions": [{"type": "json_valid"}],
        },
        {
            "id": "json-03",
            "prompt": "Dame el neto en JSON.",
            "assertions": [{"type": "json_valid"}],
        },
        {
            # Caso CAPTURADO sin aserciones: su salida histórica es línea base
            # de cambio, jamás verdad de referencia.
            "id": "capturado-01",
            "provenance": "capturado",
            "prompt": "¿Cuál es la capital de Australia?",
            "baseline_output": "La capital de Australia es Sídney.",
        },
        {
            "id": "tools-01",
            "prompt": "Consulta el saldo del cliente 42.",
            "required_capabilities": ["function_calling"],
            "assertions": [{"type": "tool_call", "name": "consultar_saldo"}],
        },
    ],
}


@pytest.fixture()
def suite_path(tmp_path: Path) -> Path:
    destino = tmp_path / "suite.yaml"
    destino.write_text(yaml.safe_dump(SUITE, allow_unicode=True), encoding="utf-8")
    return destino


def _broker(json_ok_ratio: str = "siempre") -> FakeBroker:
    """Broker con dos modelos; el nuevo puede dejar de devolver JSON."""
    broker = FakeBroker()
    # Sin `function_calling`: el caso de herramientas debe quedar not_applicable.
    broker.add_model(VIEJO.key, features={"tools": {"function_calling": "unsupported"}})
    broker.add_model(NUEVO.key, features={"tools": {"function_calling": "unsupported"}})
    broker.respond_fixed(VIEJO.key, '{"total": 100}')
    if json_ok_ratio == "siempre":
        broker.respond_fixed(NUEVO.key, '{"total": 100}')
    else:
        broker.respond_fixed(NUEVO.key, "El total asciende a cien euros.")
    return broker


def _run(
    broker: FakeBroker,
    suite_path: Path,
    model: ModelReference,
    db: Database,
    *,
    repeticiones_override: int = 4,
    **kwargs: object,
):
    suite = load_suite(suite_path)
    with broker.client() as cliente:
        config = RunConfig(repetitions=repeticiones_override, **kwargs)  # type: ignore[arg-type]
        runner = Runner(cliente, suite, db=db, config=config)
        return runner.run([model]), suite


# --------------------------------------------------------------------------
# Fase 1
# --------------------------------------------------------------------------


def test_una_capacidad_ausente_es_no_aplica_y_no_suspenso(suite_path: Path, tmp_path: Path) -> None:
    """Criterio de aceptación b de la Fase 1."""
    with Database(tmp_path / "md.sqlite") as db:
        salida, _ = _run(_broker(), suite_path, VIEJO, db)
    resultado = salida.results[VIEJO.key].outcomes[("tools-01", METRIC_PASS)]
    assert resultado.is_not_applicable is True
    assert "function_calling" in (resultado.not_applicable_reason or "")
    assert resultado.case_score is None
    # Y no entra en ningún contraste.
    assert resultado.is_usable is False


def test_la_salida_cruda_se_guarda_siempre(suite_path: Path, tmp_path: Path) -> None:
    """Es lo que permite re-juzgar sin volver a invocar."""
    with Database(tmp_path / "md.sqlite") as db:
        salida, _ = _run(_broker(), suite_path, VIEJO, db)
        filas = db.query("SELECT output_text FROM invocation WHERE run_id = ?", (salida.run_id,))
    assert filas, "no se persistió ninguna invocación"
    assert all(fila["output_text"] for fila in filas)


def test_con_un_broker_al_dia_la_semilla_se_fija_pero_no_basta(suite_path: Path, tmp_path: Path) -> None:
    """CASO 5: fijar la semilla es control, no reproducibilidad demostrada."""
    with Database(tmp_path / "md.sqlite") as db:
        salida, _ = _run(_broker(), suite_path, VIEJO, db)
    reproducibilidad = salida.manifest.reproducibility
    assert reproducibilidad.seed_fijada is True
    assert reproducibilidad.determinismo_solicitado is True
    # Y aun así: nadie ha reproducido ninguna tirada anterior, así que la
    # reproducibilidad exacta sigue SIN demostrar. Es la distinción del plan
    # §10.2 y la que impide confundir un mecanismo de control con una garantía.
    assert reproducibilidad.reproducibilidad_exacta is ExactReproducibility.no_demostrada
    assert salida.manifest.missing_guarantees == []


def test_con_un_broker_antiguo_se_declara_lo_que_falta(suite_path: Path, tmp_path: Path) -> None:
    """Contra un Broker sin los cambios: nada se finge, todo se declara."""
    broker = _broker()
    broker.supports_seed = False
    broker.supports_learning_exclusion = False
    for entrada in broker.catalog:
        entrada["execution_fingerprint"] = {}

    with Database(tmp_path / "md.sqlite") as db:
        salida, _ = _run(broker, suite_path, VIEJO, db)
    reproducibilidad = salida.manifest.reproducibility
    assert reproducibilidad.seed_fijada is False
    # Sin control de semilla la reproducibilidad exacta no es que esté sin
    # demostrar: es que no está disponible en este entorno. La distinción
    # importa porque una es una tarea pendiente y la otra un límite.
    assert reproducibilidad.reproducibilidad_exacta is ExactReproducibility.no_disponible
    faltantes = " · ".join(salida.manifest.missing_guarantees)
    assert "reproducible" in faltantes.lower()
    assert "identidad técnica" in faltantes
    assert "estadísticas del router" in faltantes


def test_el_determinismo_se_mide_no_se_supone(suite_path: Path, tmp_path: Path) -> None:
    """CASO 5, segunda mitad: repeticiones distintas -> determinismo observado = no."""
    broker = _broker()
    broker.respond(VIEJO.key, lambda prompt, model, seen: f'{{"total": {100 + seen}}}')
    with Database(tmp_path / "md.sqlite") as db:
        salida, _ = _run(broker, suite_path, VIEJO, db)
    assert salida.manifest.reproducibility.determinismo_observado is Determinism.no

    broker_estable = _broker()
    with Database(tmp_path / "md2.sqlite") as db:
        salida_estable, _ = _run(broker_estable, suite_path, VIEJO, db)
    assert salida_estable.manifest.reproducibility.determinismo_observado is Determinism.si


# --------------------------------------------------------------------------
# Fase 2
# --------------------------------------------------------------------------


def test_una_degradacion_con_oraculo_si_es_regresion(suite_path: Path, tmp_path: Path) -> None:
    with Database(tmp_path / "md.sqlite") as db:
        base, suite = _run(_broker("siempre"), suite_path, VIEJO, db)
        cand, _ = _run(_broker("nunca"), suite_path, NUEVO, db)

    informe = compare_runs(
        base.results[VIEJO.key],
        cand.results[NUEVO.key],
        suite=suite,
        policy=RegressionPolicy.from_mapping({"blocking": {"pass_rate": {"max_drop_pp": 0}}}),
    )
    json_cases = [
        item for item in informe.outcomes if item.metric == METRIC_PASS and item.case_id.startswith("json-")
    ]
    assert json_cases, "no se compararon los casos de JSON"
    assert all(item.change_status is ChangeStatus.cambio_significativo for item in json_cases)
    assert all(item.change_direction is ChangeDirection.regresion for item in json_cases)
    assert all(item.evidence_kind is not None for item in json_cases)
    # Y la taxonomía dice EN QUÉ cambió, no solo que cambió.
    assert any("formato" in item.change_taxonomy for item in json_cases)


def test_caso_1_una_linea_base_incorrecta_no_produce_regresion(suite_path: Path, tmp_path: Path) -> None:
    """CASO 1 del plan, el más peligroso de todos.

    El caso capturado no tiene aserciones: su salida histórica es línea base de
    cambio, no verdad. Aunque la versión nueva conteste algo completamente
    distinto —y de hecho correcto—, la app **no puede** llamarlo regresión.
    """
    broker_base = _broker()
    broker_base.respond_fixed(VIEJO.key, "La capital de Australia es Sídney.")
    broker_nuevo = _broker()
    broker_nuevo.respond_fixed(NUEVO.key, "La capital de Australia es Canberra.")

    with Database(tmp_path / "md.sqlite") as db:
        base, suite = _run(broker_base, suite_path, VIEJO, db)
        cand, _ = _run(broker_nuevo, suite_path, NUEVO, db)

    informe = compare_runs(base.results[VIEJO.key], cand.results[NUEVO.key], suite=suite)
    capturados = [
        item for item in informe.content_outcomes() if item.case_id == "capturado-01"
    ]
    assert capturados, "el caso capturado no se comparó"
    for item in capturados:
        assert item.change_direction is ChangeDirection.sin_direccion
        assert item.evidence_kind is None
        assert not item.is_regression
    # Y sí detecta el cambio: lo que no hace es ponerle signo.
    assert any(item.change_status is ChangeStatus.cambio_significativo for item in capturados)


def test_caso_4_la_huella_detecta_que_la_ejecucion_no_fue_identica(
    suite_path: Path, tmp_path: Path
) -> None:
    """Mismo `model_id`, distinta plantilla de chat: no es comparable."""
    with Database(tmp_path / "md.sqlite") as db:
        base, suite = _run(_broker(), suite_path, VIEJO, db)
        cand, _ = _run(_broker(), suite_path, VIEJO, db)

    # Se simula lo que dará el Cambio 2 del contrato: la capa B ya informada.
    from model_drift.fingerprint import FingerprintBuilder

    for salida, plantilla in ((base, "PLANTILLA-A"), (cand, "PLANTILLA-B")):
        modelo = salida.manifest.models[0]
        modelo.fingerprint = (
            FingerprintBuilder()
            .declared("provider", "ollama")
            .declared("deployment", "local")
            .declared("model", "qwen3:6b")
            .verified("digest", "sha256:mismo-binario")
            .hashed("chat_template", plantilla)
            .build()
        )

    informe = compare_runs(base.results[VIEJO.key], cand.results[VIEJO.key], suite=suite)
    assert informe.comparability.comparable is False
    assert informe.comparability.fingerprint_divergent
    assert "chat_template" in informe.comparability.headline()
    # Y ningún caso recibe dirección: la diferencia no es atribuible al modelo.
    assert all(item.change_status is ChangeStatus.indeterminado for item in informe.outcomes)
    assert all(item.change_direction is ChangeDirection.sin_direccion for item in informe.outcomes)


def test_un_instrumento_distinto_invalida_la_comparacion(suite_path: Path, tmp_path: Path) -> None:
    """El evaluador forma parte del instrumento de medida (plan §11)."""
    with Database(tmp_path / "md.sqlite") as db:
        base, suite = _run(_broker(), suite_path, VIEJO, db)
        cand, _ = _run(_broker(), suite_path, VIEJO, db)

    cand.manifest.instruments.append(InstrumentRef(id="embedder", kind="embedding", version="v2"))
    informe = compare_runs(base.results[VIEJO.key], cand.results[VIEJO.key], suite=suite)
    assert informe.comparability.comparable is False
    assert informe.comparability.instrument_changed
    assert all(item.change_status is ChangeStatus.indeterminado for item in informe.outcomes)


def test_control_negativo_dos_tiradas_iguales_no_producen_regresiones(
    suite_path: Path, tmp_path: Path
) -> None:
    """El instrumento no puede inventar deriva donde no la hay."""
    with Database(tmp_path / "md.sqlite") as db:
        base, suite = _run(_broker(), suite_path, VIEJO, db)
        cand, _ = _run(_broker(), suite_path, VIEJO, db)

    informe = compare_runs(base.results[VIEJO.key], cand.results[VIEJO.key], suite=suite)
    assert informe.comparability.comparable is True
    assert informe.regressions() == []
    assert informe.verdict is not None
    assert informe.verdict.verdict is RunVerdict.pass_


def test_caso_8_mejora_una_metrica_y_empeoran_otras_da_trade_off(
    suite_path: Path, tmp_path: Path
) -> None:
    """No se inventa un número que zanje el dilema."""
    from model_drift.domain import CaseOutcome

    # La base falla el JSON (es el modelo "nuevo" del broker degradado) y la
    # candidata lo acierta: así la exactitud MEJORA en la candidata.
    with Database(tmp_path / "md.sqlite") as db:
        base, suite = _run(_broker("nunca"), suite_path, NUEVO, db)
        cand, _ = _run(_broker("siempre"), suite_path, VIEJO, db)

    # La candidata acierta más pero tarda y cuesta bastante más.
    resultados_cand = cand.results[VIEJO.key]
    for case_id in ("json-01", "json-02", "json-03"):
        for metrica, valor in (("latency_ms", 900.0), ("cost_usd", 0.02)):
            resultados_cand.outcomes[(case_id, metrica)] = CaseOutcome(
                case_id=case_id,
                model_key=VIEJO.key,
                metric=metrica,
                case_score=valor,
                case_stability=1.0,
                repetitions=4,
                values=[valor] * 4,
            )
    resultados_base = base.results[NUEVO.key]
    for case_id in ("json-01", "json-02", "json-03"):
        for metrica, valor in (("latency_ms", 300.0), ("cost_usd", 0.005)):
            resultados_base.outcomes[(case_id, metrica)] = CaseOutcome(
                case_id=case_id,
                model_key=NUEVO.key,
                metric=metrica,
                case_score=valor,
                case_stability=1.0,
                repetitions=4,
                values=[valor] * 4,
            )

    politica = RegressionPolicy.from_mapping(
        {
            "blocking": {"pass_rate": {"max_drop_pp": 0, "minimum_practical_effect_pp": 1}},
            "warning": {
                "latency_ms": {
                    "max_increase_pct": 15,
                    "higher_is_better": False,
                    "minimum_practical_effect": 50.0,
                },
                "cost_usd": {
                    "max_increase_pct": 10,
                    "higher_is_better": False,
                    "minimum_practical_effect": 0.001,
                },
            },
        }
    )
    informe = compare_runs(resultados_base, resultados_cand, suite=suite, policy=politica)
    assert informe.verdict is not None
    assert informe.verdict.verdict is RunVerdict.trade_off_detected
    assert "pass_rate" in informe.verdict.improved
    assert {"latency_ms", "cost_usd"} <= set(informe.verdict.worsened)
    assert "compensación" in informe.verdict.summary().lower()


def test_el_informe_declara_su_propia_potencia(suite_path: Path, tmp_path: Path) -> None:
    """Un «sin cambio observable» sobre 3 casos no puede leerse como garantía."""
    with Database(tmp_path / "md.sqlite") as db:
        base, suite = _run(_broker(), suite_path, VIEJO, db)
        cand, _ = _run(_broker(), suite_path, VIEJO, db)

    informe = compare_runs(base.results[VIEJO.key], cand.results[VIEJO.key], suite=suite)
    frase = informe.power_statement()
    assert "casos comparados" in frase
    assert "la ausencia de señal no es evidencia de ausencia" in frase


def test_un_caso_sin_potencia_no_se_informa_como_sin_cambio(suite_path: Path, tmp_path: Path) -> None:
    """Un cambio del 100 % al 0 % con 3 repeticiones NO es «sin cambio observable».

    El contraste exacto no puede bajar de p = 0,1 con esas repeticiones, así que
    lo honesto es decir que no se puede decidir. Lo cómodo —informarlo como
    ausencia de cambio— sería justo el error que este proyecto existe para
    evitar, cometido por el propio instrumento.
    """
    with Database(tmp_path / "md.sqlite") as db:
        base, suite = _run(_broker("siempre"), suite_path, VIEJO, db, repeticiones_override=3)
        cand, _ = _run(_broker("nunca"), suite_path, NUEVO, db, repeticiones_override=3)

    informe = compare_runs(base.results[VIEJO.key], cand.results[NUEVO.key], suite=suite)
    json_cases = [
        item for item in informe.outcomes if item.metric == METRIC_PASS and item.case_id.startswith("json-")
    ]
    assert json_cases
    assert all(item.change_status is ChangeStatus.indeterminado for item in json_cases)
    assert all(item.change_direction is ChangeDirection.sin_direccion for item in json_cases)
    assert any("no bastan para decidir" in nota for item in json_cases for nota in item.notes)


def test_la_reproducibilidad_sale_de_seed_status_no_de_lo_pedido(
    suite_path: Path, tmp_path: Path
) -> None:
    """Pedir la semilla no es tenerla.

    El Broker no promete que el modelo la honre: solo declara si la envió. Si
    el endpoint no la admite (`seed_status: unsupported`), la tirada NO tuvo
    control de determinismo por mucho que se pidiera, y el manifiesto tiene que
    decir eso.
    """
    broker = _broker()
    broker.seed_status = "unsupported"
    with Database(tmp_path / "md.sqlite") as db:
        salida, _ = _run(broker, suite_path, VIEJO, db)

    assert salida.manifest.reproducibility.seed_fijada is False
    assert any(
        "no envió la semilla" in aviso or "no admite semilla" in aviso
        for aviso in salida.manifest.missing_guarantees
    )


def test_una_latencia_que_baja_es_mejora_aunque_nadie_la_declare() -> None:
    """El fallo que solo salió con una tirada real contra el Broker.

    Sin política, `MetricRule` heredaba `higher_is_better=True` y una latencia
    de 5.280 ms que bajaba a 5.030 ms se informaba como REGRESIÓN. Invertir el
    signo de la conclusión sobre latencia, coste y verbosidad no es un detalle
    cosmético: es decir lo contrario de lo que pasó.
    """
    from model_drift.domain import METRIC_COST, METRIC_LATENCY, METRIC_PASS, METRIC_TOKENS
    from model_drift.policy import RegressionPolicy

    politica = RegressionPolicy()
    for metrica in (METRIC_LATENCY, METRIC_COST, METRIC_TOKENS):
        assert politica.rule_for(metrica).higher_is_better is False, metrica
        # Bajar es mejorar: no puede contar como incumplimiento.
        assert politica.rule_for(metrica).worsens(-100.0) is False
        assert politica.rule_for(metrica).worsens(+100.0) is True
    # Una tasa de acierto sigue mejorando al subir.
    assert politica.rule_for(METRIC_PASS).higher_is_better is True
    assert politica.rule_for(METRIC_PASS).worsens(-0.1) is True


def test_una_latencia_que_baja_no_aparece_como_regresion(suite_path: Path, tmp_path: Path) -> None:
    from model_drift.domain import METRIC_LATENCY
    from model_drift.domain import ChangeDirection as Dir

    with Database(tmp_path / "md.sqlite") as db:
        base, suite = _run(_broker(), suite_path, VIEJO, db)
        cand, _ = _run(_broker(), suite_path, VIEJO, db)

    from model_drift.domain import CaseOutcome

    for resultados, valor in ((base.results[VIEJO.key], 5280.0), (cand.results[VIEJO.key], 5030.0)):
        clave = "json-01"
        resultados.outcomes[(clave, METRIC_LATENCY)] = CaseOutcome(
            case_id=clave,
            model_key=VIEJO.key,
            metric=METRIC_LATENCY,
            case_score=valor,
            case_stability=1.0,
            repetitions=5,
            values=[valor] * 5,
        )

    informe = compare_runs(base.results[VIEJO.key], cand.results[VIEJO.key], suite=suite)
    latencias = [item for item in informe.outcomes if item.metric == METRIC_LATENCY]
    assert latencias
    assert all(item.change_direction is not Dir.regresion for item in latencias)
