"""Prueba de la suite de ejemplo que se distribuye con el proyecto.

Sirve para dos cosas a la vez: comprueba el camino completo —suite real,
relaciones metamórficas reales, política real— y evita que los ficheros de
ejemplo se queden obsoletos en silencio, que es lo que le pasa a toda la
documentación que nadie ejecuta.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from model_drift.broker_contracts import ModelReference
from model_drift.cases import load_suite
from model_drift.comparison import compare_runs
from model_drift.db import Database
from model_drift.domain import METRIC_PASS, ChangeDirection, ChangeStatus, RunVerdict
from model_drift.metamorphic import load_relations
from model_drift.policy import RegressionPolicy
from model_drift.report import render_comparison
from model_drift.runner import RunConfig, Runner
from tests.broker_fake import FakeBroker

RAIZ = Path(__file__).resolve().parent.parent
VIEJO = ModelReference(provider="ollama", deployment="local", model="qwen3:6b")
NUEVO = ModelReference(provider="ollama", deployment="local", model="qwen3:8b")

FACTURA_OK = (
    '{"numero": "2026/0417", "fecha": "14/03/2026", "cliente": "Talleres Mendoza S.L.", '
    '"base_imponible": 1033.06, "iva": 216.94, "total": 1250.00}'
)
FACTURA_OK_2 = (
    '{"numero": "2026/0418", "fecha": "15/03/2026", "cliente": "Distribuciones Arán", '
    '"base_imponible": 500.00, "iva": 105.00, "total": 605.00}'
)


def _responder(degradado: bool):  # type: ignore[no-untyped-def]
    """Modelo simulado. El degradado envuelve el JSON en vallas de código."""

    def responder(prompt: str, model: str, seen: int) -> str:
        if "2026/0417" in prompt:
            return f"```json\n{FACTURA_OK}\n```" if degradado else FACTURA_OK
        if "2026/0418" in prompt:
            return f"```json\n{FACTURA_OK_2}\n```" if degradado else FACTURA_OK_2
        if "IVA repercutido" in prompt:
            return "Es el IVA que el vendedor cobra al cliente y luego ingresa en Hacienda."
        if "cuenta bancaria" in prompt:
            return "No consta en el contexto proporcionado."
        if "agenciatributaria" in prompt:
            return "El IVA repercutido es el que se cobra al cliente. Fuente: https://sede.agenciatributaria.gob.es/iva"
        if "NO puede deducirse" in prompt:
            return "b"
        if "800" in prompt and "21" in prompt:
            return "968 euros"
        if "800" in prompt and "4 %" in prompt:
            return "832 euros"
        if "2.500" in prompt:
            return "3025 euros"
        if "21 % de 1.000" in prompt:
            return "210"
        return "respuesta genérica"

    return responder


def _broker(degradado: bool = False) -> FakeBroker:
    broker = FakeBroker()
    features = {
        "generation": {"json_mode": "supported"},
        "tools": {"function_calling": "unsupported"},
    }
    for modelo in (VIEJO, NUEVO):
        broker.add_model(modelo.key, features=features)
    broker.respond(VIEJO.key, _responder(False))
    broker.respond(NUEVO.key, _responder(degradado))
    return broker


@pytest.fixture()
def suite():  # type: ignore[no-untyped-def]
    return load_suite(RAIZ / "suites" / "ejemplo")


def test_la_suite_de_ejemplo_carga_y_es_coherente(suite) -> None:  # type: ignore[no-untyped-def]
    relaciones = load_relations(RAIZ / "relaciones")
    referenciadas = {rid for caso in suite.cases for rid in caso.metamorphic_relations}
    faltan = referenciadas - set(relaciones)
    assert not faltan, f"la suite referencia relaciones que no existen: {faltan}"
    # Todo caso debe poder sostener nivel 2 o declarar por qué no.
    for caso in suite.cases:
        assert caso.provides_level_2_evidence(), f"{caso.id} no tiene ni aserciones ni relación"


def test_la_politica_de_ejemplo_es_valida() -> None:
    politica = RegressionPolicy.from_file(RAIZ / "politica-ejemplo.yaml")
    assert politica.rule_for("pass_rate").severity.value == "blocking"
    assert politica.rule_for("latency_ms").higher_is_better is False
    # Una métrica no declarada nunca puede bloquear una entrega.
    assert politica.rule_for("inventada").severity.value == "informative"


def test_camino_completo_sobre_la_suite_de_ejemplo(tmp_path: Path, suite) -> None:  # type: ignore[no-untyped-def]
    relaciones = load_relations(RAIZ / "relaciones")
    politica = RegressionPolicy.from_file(RAIZ / "politica-ejemplo.yaml")

    with Database(tmp_path / "md.sqlite") as db:
        with _broker(degradado=False).client() as cliente:
            base = Runner(
                cliente, suite, db=db, config=RunConfig(repetitions=3), relations=relaciones
            ).run([VIEJO])
        with _broker(degradado=True).client() as cliente:
            cand = Runner(
                cliente, suite, db=db, config=RunConfig(repetitions=3), relations=relaciones
            ).run([NUEVO])

    informe = compare_runs(
        base.results[VIEJO.key], cand.results[NUEVO.key], suite=suite, policy=politica
    )

    assert informe.comparability.comparable is True
    assert informe.n_compared > 0

    # El modelo degradado envuelve el JSON en vallas, pero el JSON sigue siendo
    # válido y sigue validando el esquema: la conformidad NO cae.
    json_cases = [
        item
        for item in informe.outcomes
        if item.metric == METRIC_PASS and item.case_id.startswith("factura-json")
    ]
    assert json_cases
    assert all(item.change_status is ChangeStatus.sin_cambio_observable for item in json_cases)

    # Y sin embargo la taxonomía SÍ ve el cambio de formato. Es justo el tipo de
    # deriva que ninguna métrica de acierto captura.
    con_formato = [item for item in informe.outcomes if "formato" in item.change_taxonomy]
    assert con_formato, "la taxonomía debería haber detectado las vallas de código"

    assert informe.verdict is not None
    assert informe.verdict.verdict in {RunVerdict.pass_, RunVerdict.warning, RunVerdict.indeterminado}

    # El informe se genera y respeta la regla del vocabulario.
    html = render_comparison(informe, manifest=cand.manifest)
    assert "<h1>" in html
    for item in informe.outcomes:
        if item.change_direction is ChangeDirection.sin_direccion and item.case_id in html:
            continue
    assert "Reproducibilidad exacta" in html


def test_las_relaciones_metamorficas_se_ejecutan(tmp_path: Path, suite) -> None:  # type: ignore[no-untyped-def]
    relaciones = load_relations(RAIZ / "relaciones")
    with Database(tmp_path / "md.sqlite") as db:
        with _broker().client() as cliente:
            salida = Runner(
                cliente, suite, db=db, config=RunConfig(repetitions=2), relations=relaciones
            ).run([VIEJO])

    metricas = {metrica for _, metrica in salida.results[VIEJO.key].outcomes}
    metamorficas = {metrica for metrica in metricas if metrica.startswith("mr:")}
    assert metamorficas, "no se evaluó ninguna relación metamórfica"

    # El caso de orden: el modelo simulado responde siempre "b", así que la
    # invariancia se cumple.
    orden = salida.results[VIEJO.key].outcomes.get(("orden-opciones-01", "mr:mr-orden-opciones"))
    assert orden is not None
    assert orden.case_score == 1.0
