"""Pruebas del instrumento de embeddings y del manifiesto."""
from __future__ import annotations

from pathlib import Path

import pytest

from model_drift.broker_contracts import ModelReference
from model_drift.embeddings import (
    BrokerEmbedder,
    CalibrationPoint,
    StaticEmbedder,
    calibrate_threshold,
)
from model_drift.fingerprint import FingerprintBuilder
from model_drift.manifest import (
    Determinism,
    ExactReproducibility,
    InstrumentRef,
    ModelUnderTest,
    ReproducibilityFlags,
    RunManifest,
    compare_manifests,
)
from tests.broker_fake import FakeBroker

EMB = ModelReference(provider="ollama", deployment="local", model="bge-m3")


# --------------------------------------------------------------------------
# Embeddings como instrumento
# --------------------------------------------------------------------------


def test_el_embedder_del_broker_cachea_y_se_identifica() -> None:
    broker = FakeBroker()
    broker.add_model(EMB.key, capabilities=("embedding",))
    with broker.client() as cliente:
        embedder = BrokerEmbedder(client=cliente, model=EMB)
        primero = embedder.embed(["hola", "adiós"])
        segundo = embedder.embed(["hola"])
    assert len(primero) == 2
    assert segundo[0] == primero[0]
    assert embedder.identity == EMB.key
    assert embedder.instrument().kind == "embedding"
    # "hola" se pidió una sola vez pese a aparecer en dos llamadas.
    prompts = [peticion["content"]["prompt"] for peticion in broker.requests]
    assert prompts.count("hola") == 1


def test_si_el_embedder_falla_no_se_inventa_una_similitud() -> None:
    broker = FakeBroker()
    broker.add_model(EMB.key, capabilities=("embedding",))
    broker.fail_prompts["texto"] = "PROVIDER_UNAVAILABLE"
    with broker.client() as cliente:
        embedder = BrokerEmbedder(client=cliente, model=EMB)
        assert embedder.embed(["texto"]) == []
        assert embedder.failures == 1


def test_la_calibracion_elige_el_umbral_sobre_datos_etiquetados() -> None:
    """Un umbral sin procedencia es una opinión con decimales."""
    embedder = StaticEmbedder(
        vectors={
            "a1": [1.0, 0.0],
            "a2": [0.99, 0.14],
            "b1": [1.0, 0.0],
            "b2": [0.0, 1.0],
            "c1": [1.0, 0.0],
            "c2": [0.95, 0.31],
            "d1": [1.0, 0.0],
            "d2": [-1.0, 0.0],
        }
    )
    puntos = [
        CalibrationPoint("a1", "a2", True),
        CalibrationPoint("c1", "c2", True),
        CalibrationPoint("b1", "b2", False),
        CalibrationPoint("d1", "d2", False),
    ]
    calibracion = calibrate_threshold(puntos, embedder)
    assert 0.0 < calibracion.threshold < 1.0
    assert calibracion.f1 == 1.0
    assert calibracion.n_points == 4
    assert calibracion.criterion == "f1"
    assert calibracion.embedder_identity.startswith("pruebas:")


def test_la_calibracion_se_guarda_con_su_procedencia(tmp_path: Path) -> None:
    embedder = StaticEmbedder(vectors={"x": [1.0, 0.0], "y": [1.0, 0.0]})
    calibracion = calibrate_threshold([CalibrationPoint("x", "y", True)], embedder)
    destino = calibracion.save(tmp_path / "calib" / "umbral.json")
    contenido = destino.read_text(encoding="utf-8")
    assert "threshold" in contenido and "criterion" in contenido and "embedder_identity" in contenido


def test_no_se_puede_calibrar_sin_datos() -> None:
    with pytest.raises(ValueError, match="sin pares etiquetados"):
        calibrate_threshold([], StaticEmbedder(vectors={}))


# --------------------------------------------------------------------------
# Manifiesto y comparabilidad
# --------------------------------------------------------------------------


def _manifiesto(run_id: str, **kwargs: object) -> RunManifest:
    base: dict[str, object] = {
        "run_id": run_id,
        "suite_hash": "abc123",
        "instruments": [InstrumentRef(id="suite", kind="suite", version="1", hash="abc123")],
    }
    base.update(kwargs)
    return RunManifest.model_validate(base)


def test_dos_manifiestos_iguales_son_comparables() -> None:
    informe = compare_manifests(_manifiesto("a"), _manifiesto("b"))
    assert informe.comparable is True
    assert "mismo instrumento" in informe.headline()


def test_una_suite_distinta_rompe_la_comparabilidad() -> None:
    informe = compare_manifests(_manifiesto("a"), _manifiesto("b", suite_hash="otro"))
    assert informe.comparable is False
    assert informe.instrument_changed is True


def test_un_fixture_cambiado_rompe_la_comparabilidad() -> None:
    izquierda = _manifiesto("a", fixtures={"fixture://e.json": "h1"})
    derecha = _manifiesto("b", fixtures={"fixture://e.json": "h2"})
    informe = compare_manifests(izquierda, derecha)
    assert informe.comparable is False
    assert any(problema.kind == "fixture" for problema in informe.issues)


def test_un_instrumento_solo_en_un_lado_tambien_rompe() -> None:
    derecha = _manifiesto("b")
    derecha.instruments.append(InstrumentRef(id="embedder", kind="embedding", version="v1"))
    informe = compare_manifests(_manifiesto("a"), derecha)
    assert informe.comparable is False


def test_la_politica_cambiada_rompe_la_comparabilidad() -> None:
    informe = compare_manifests(_manifiesto("a", policy_hash="p1"), _manifiesto("b", policy_hash="p2"))
    assert informe.comparable is False


def test_comparando_dos_modelos_distintos_su_identidad_no_cuenta_como_divergencia() -> None:
    """El modelo ES la variable del experimento: por eso se excluye del cotejo."""
    izquierda = _manifiesto("a")
    derecha = _manifiesto("b")
    izquierda.models.append(
        ModelUnderTest(
            key="ollama/local/qwen3:6b",
            provider="ollama",
            deployment="local",
            model="qwen3:6b",
            fingerprint=FingerprintBuilder()
            .declared("model", "qwen3:6b")
            .verified("digest", "sha256:viejo")
            .hashed("chat_template", "PLANTILLA")
            .build(),
        )
    )
    derecha.models.append(
        ModelUnderTest(
            key="ollama/local/qwen3:8b",
            provider="ollama",
            deployment="local",
            model="qwen3:8b",
            fingerprint=FingerprintBuilder()
            .declared("model", "qwen3:8b")
            .verified("digest", "sha256:nuevo")
            .hashed("chat_template", "PLANTILLA")
            .build(),
        )
    )
    informe = compare_manifests(
        izquierda,
        derecha,
        baseline_model_key="ollama/local/qwen3:6b",
        candidate_model_key="ollama/local/qwen3:8b",
    )
    assert informe.comparable is True


def test_la_firma_de_instrumentos_cambia_con_los_fixtures() -> None:
    sin_fixtures = _manifiesto("a")
    con_fixtures = _manifiesto("a", fixtures={"fixture://x": "h"})
    assert sin_fixtures.instrument_signature() != con_fixtures.instrument_signature()


def test_las_cuatro_banderas_se_declaran_sin_afirmar_de_mas() -> None:
    lineas = ReproducibilityFlags().declare()
    assert any("Semilla fijada: no" in linea for linea in lineas)
    assert any("no evaluado" in linea for linea in lineas)
    assert any("NO demostrada" in linea for linea in lineas)

    completas = ReproducibilityFlags(
        seed_fijada=True,
        determinismo_solicitado=True,
        determinismo_observado=Determinism.no,
        reproducibilidad_exacta=ExactReproducibility.no_disponible,
    ).declare()
    assert any("es lo normal y no invalida la medición" in linea for linea in completas)
    assert any("no disponible en este entorno" in linea for linea in completas)


def test_el_manifiesto_no_duplica_garantias_ausentes() -> None:
    manifiesto = _manifiesto("a")
    manifiesto.declare_missing("sin semilla")
    manifiesto.declare_missing("sin semilla")
    assert manifiesto.missing_guarantees == ["sin semilla"]
    assert "sin semilla" in manifiesto.to_json()


def test_busqueda_de_modelo_por_clave() -> None:
    manifiesto = _manifiesto("a")
    manifiesto.models.append(
        ModelUnderTest(key="k", provider="p", deployment="d", model="m")
    )
    assert manifiesto.model_by_key("k") is not None
    assert manifiesto.model_by_key("otro") is None
    assert manifiesto.models[0].declared_identity == "p/d/m"
