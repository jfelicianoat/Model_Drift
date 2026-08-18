"""Pruebas del juez: anonimización, tres métricas separadas y puerta dura."""
from __future__ import annotations

import json

from model_drift.broker_contracts import ModelReference
from model_drift.judge import (
    Judge,
    JudgeConfig,
    JudgeMetrics,
    Verdict,
    cohen_kappa,
    raw_agreement,
)
from tests.broker_fake import FakeBroker

JUEZ = ModelReference(provider="ollama", deployment="local", model="juez-grande")


def _judge(responder) -> tuple[Judge, FakeBroker]:
    broker = FakeBroker()
    broker.add_model(JUEZ.key)
    broker.respond(JUEZ.key, responder)
    cliente = broker.client()
    return Judge(cliente, JudgeConfig(model=JUEZ)), broker


def _veredicto(valor: str, motivo: str = "porque sí") -> str:
    return json.dumps({"veredicto": valor, "motivo": motivo})


# --------------------------------------------------------------------------
# Anonimización
# --------------------------------------------------------------------------


def test_el_juez_nunca_ve_nombres_de_modelo(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """CASO 10 del plan: la arquitectura lo impide, no la buena voluntad."""
    juez, broker = _judge(lambda prompt, model, seen: _veredicto("A"))
    juez.judge_pair(
        "caso-1",
        "¿Cuál es la capital de Australia?",
        "Respuesta de qwen3:6b",
        "Respuesta de qwen3:8b",
    )
    prompts = [peticion["content"]["prompt"] for peticion in broker.requests]
    assert prompts, "el juez no llegó a invocar"
    for prompt in prompts:
        assert "base" not in prompt.lower().split("### ")[0]
        assert "candidata" not in prompt.lower()
        assert "antigua" not in prompt.lower()
        assert "### Respuesta A" in prompt and "### Respuesta B" in prompt


def test_el_juez_pide_veredicto_estructurado() -> None:
    juez, broker = _judge(lambda prompt, model, seen: _veredicto("empate"))
    juez.judge_pair("c", "pregunta", "izquierda", "derecha")
    assert broker.requests[0]["output"]["format"] == "json"
    assert broker.requests[0]["output"]["json_schema"]["required"] == ["veredicto", "motivo"]


def test_siempre_se_ejecutan_los_dos_ordenes() -> None:
    juez, broker = _judge(lambda prompt, model, seen: _veredicto("A"))
    juez.judge_pair("c", "pregunta", "izquierda", "derecha")
    # Se cuentan prompts distintos y no peticiones: la primera se reintenta
    # porque el Broker simulado rechaza `generation.seed`, igual que el real.
    prompts = {peticion["content"]["prompt"] for peticion in broker.requests}
    assert len(prompts) == 2
    izquierda_primero = "### Respuesta A\nizquierda"
    derecha_primero = "### Respuesta A\nderecha"
    assert any(izquierda_primero in prompt for prompt in prompts)
    assert any(derecha_primero in prompt for prompt in prompts)


# --------------------------------------------------------------------------
# Sesgo de posición
# --------------------------------------------------------------------------


def test_un_juez_que_siempre_dice_A_es_puro_sesgo_de_posicion() -> None:
    """Decir siempre «A» es no mirar el contenido: se invierte en los dos órdenes."""
    juez, _ = _judge(lambda prompt, model, seen: _veredicto("A"))
    juicios = [juez.judge_pair(f"c{i}", "pregunta", f"izq{i}", f"der{i}") for i in range(10)]
    assert juez.measure_position_bias(juicios) == 1.0
    assert all(not juicio.consistent for juicio in juicios)
    assert all(juicio.verdict is Verdict.invalido for juicio in juicios)


def test_un_juez_que_mira_el_contenido_no_tiene_sesgo_de_posicion() -> None:
    def responder(prompt: str, model: str, seen: int) -> str:
        # Prefiere la respuesta que contiene "buena", esté donde esté.
        bloque_a = prompt.split("### Respuesta A")[1].split("### Respuesta B")[0]
        return _veredicto("A" if "buena" in bloque_a else "B")

    juez, _ = _judge(responder)
    juicios = [juez.judge_pair(f"c{i}", "pregunta", "respuesta buena", "respuesta mala") for i in range(6)]
    assert juez.measure_position_bias(juicios) == 0.0
    assert all(juicio.consistent for juicio in juicios)
    assert all(juicio.verdict is Verdict.a_mejor for juicio in juicios)
    assert all(juicio.favors_candidate is False for juicio in juicios)


# --------------------------------------------------------------------------
# Autoconsistencia: distinta del sesgo de posición
# --------------------------------------------------------------------------


def test_caso_6_un_juez_inestable_a_temperatura_cero_se_detecta() -> None:
    """`temperature = 0` reduce la varianza; no demuestra determinismo."""
    ciclo = ["A", "B", "empate"]
    juez, _ = _judge(lambda prompt, model, seen: _veredicto(ciclo[seen % 3]))
    muestra = [("c1", "pregunta", "izquierda", "derecha")]
    consistencia = juez.measure_self_consistency(muestra, repetitions=3)
    assert consistencia == 0.0
    assert juez.metrics.judge_self_consistency == 0.0


def test_un_juez_estable_puntua_uno_en_autoconsistencia() -> None:
    juez, _ = _judge(lambda prompt, model, seen: _veredicto("A"))
    consistencia = juez.measure_self_consistency([("c1", "p", "i", "d")], repetitions=4)
    assert consistencia == 1.0


# --------------------------------------------------------------------------
# Concordancia
# --------------------------------------------------------------------------


def test_kappa_perfecto_y_kappa_de_azar() -> None:
    etiquetas = ["A_mejor", "B_mejor", "empate", "A_mejor", "B_mejor", "empate"]
    assert cohen_kappa(etiquetas, list(etiquetas)) == 1.0
    opuestas = ["B_mejor", "A_mejor", "empate", "B_mejor", "A_mejor", "empate"]
    assert cohen_kappa(etiquetas, opuestas) < 0.5


def test_el_kappa_ponderado_castiga_menos_los_errores_pequenos() -> None:
    """Confundir empate con A no es lo mismo que confundir A con B."""
    humano = ["A_mejor", "A_mejor", "empate", "empate", "B_mejor", "B_mejor"]
    cerca = ["A_mejor", "empate", "empate", "A_mejor", "B_mejor", "B_mejor"]
    lejos = ["A_mejor", "B_mejor", "empate", "B_mejor", "B_mejor", "B_mejor"]
    assert cohen_kappa(humano, cerca, weighted=True) > cohen_kappa(humano, lejos, weighted=True)


def test_acuerdo_bruto() -> None:
    assert raw_agreement(["a", "b", "c"], ["a", "b", "x"]) == 2 / 3
    assert raw_agreement([], []) == 0.0


def test_la_segunda_persona_permite_interpretar_la_discrepancia() -> None:
    juez, _ = _judge(lambda prompt, model, seen: _veredicto("A"))
    juez_labels = ["A_mejor", "A_mejor", "empate", "B_mejor"]
    humano = ["A_mejor", "empate", "empate", "B_mejor"]
    segunda = ["A_mejor", "A_mejor", "empate", "B_mejor"]

    sin_segunda = juez.measure_human_agreement(juez_labels, humano)
    assert sin_segunda.human_human_agreement is None
    assert "segunda persona" in sin_segunda.interpretation()

    con_segunda = juez.measure_human_agreement(juez_labels, humano, second_annotator=segunda)
    assert con_segunda.human_human_agreement is not None
    assert con_segunda.n_second_annotator == 4
    assert "instrumento" in con_segunda.interpretation() or "juez" in con_segunda.interpretation()


# --------------------------------------------------------------------------
# La puerta dura
# --------------------------------------------------------------------------


def test_un_juez_sin_calibrar_no_puede_dar_direccion() -> None:
    config = JudgeConfig(model=JUEZ)
    calibrado, motivos = JudgeMetrics().is_calibrated(config)
    assert calibrado is False
    assert any("concordancia" in motivo for motivo in motivos)
    assert any("autoconsistencia" in motivo for motivo in motivos)


def test_un_juez_calibrado_pasa_la_puerta() -> None:
    config = JudgeConfig(model=JUEZ)
    metricas = JudgeMetrics(
        position_bias=0.05,
        judge_self_consistency=0.95,
        judge_human_agreement=0.72,
        human_human_agreement=0.70,
    )
    calibrado, motivos = metricas.is_calibrated(config)
    assert calibrado is True
    assert motivos == []
    assert "tanto o más" in metricas.interpretation()


def test_demasiado_sesgo_de_posicion_descalifica_al_juez() -> None:
    config = JudgeConfig(model=JUEZ)
    metricas = JudgeMetrics(
        position_bias=0.4,
        judge_self_consistency=0.99,
        judge_human_agreement=0.9,
    )
    calibrado, motivos = metricas.is_calibrated(config)
    assert calibrado is False
    assert any("sesgo de posición" in motivo for motivo in motivos)


def test_una_salida_ilegible_del_juez_es_invalida_no_empate() -> None:
    """Convertir un fallo del instrumento en un dato sería fabricar evidencia."""
    juez, _ = _judge(lambda prompt, model, seen: "no pienso contestarte en JSON")
    juicio = juez.judge_pair("c", "pregunta", "izquierda", "derecha")
    assert juicio.verdict is Verdict.invalido
    assert juicio.verdict_ab is Verdict.invalido
