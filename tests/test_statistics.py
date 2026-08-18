"""Pruebas de la estadística: unidad de análisis, contrastes y potencia.

Aquí viven los dos criterios de aceptación de la Fase 2 del plan, que
sustituyen al absoluto imposible de la revisión 1 («dos tiradas del mismo
modelo no producen ninguna regresión significativa»):

- **control negativo** — la tasa de falsas alarmas debe ser compatible con el
  nivel de error configurado, no cero;
- **control positivo** — una degradación sintética conocida debe detectarse con
  una potencia mínima predefinida.
"""
from __future__ import annotations

import random

import pytest

from model_drift.domain import PracticalSignificance
from model_drift.statistics import (
    aggregate_binary,
    aggregate_continuous,
    benjamini_hochberg,
    binarize,
    classify_practical,
    cluster_bootstrap,
    estimate_power,
    fisher_exact_2x2,
    mcnemar_exact,
    minimum_attainable_p,
    minimum_detectable_effect,
    power_curve,
)

# --------------------------------------------------------------------------
# Piso 1: agregación dentro del caso
# --------------------------------------------------------------------------


def test_agregacion_binaria_calcula_tasa_y_estabilidad() -> None:
    agregado = aggregate_binary([True, True, True, False])
    assert agregado.score == 0.75
    assert agregado.stability == 0.75
    assert agregado.repetitions == 4


def test_un_caso_repartido_al_cincuenta_por_ciento_es_maximamente_inestable() -> None:
    """Un caso que dice que sí la mitad de las veces mide el azar, no el modelo."""
    agregado = aggregate_binary([True, False, True, False])
    assert agregado.stability == 0.5


def test_agregacion_continua_usa_mediana_y_penaliza_la_dispersion() -> None:
    estable = aggregate_continuous([100.0, 101.0, 99.0])
    volatil = aggregate_continuous([10.0, 500.0, 90.0])
    assert estable.score == 100.0
    assert estable.stability > volatil.stability


def test_binarizacion_todas_es_mas_estricta_que_mayoria() -> None:
    assert binarize(0.9, "todas") is False
    assert binarize(0.9, "mayoria") is True
    assert binarize(1.0, "todas") is True


# --------------------------------------------------------------------------
# Contrastes
# --------------------------------------------------------------------------


def test_mcnemar_sin_discordantes_no_detecta_nada() -> None:
    """Los casos que se comportan igual en ambos lados no llevan información."""
    base = [True] * 20
    resultado = mcnemar_exact(base, list(base))
    assert resultado.discordant == 0
    assert resultado.p_value == 1.0
    assert resultado.effect == 0.0


def test_mcnemar_detecta_una_degradacion_sistematica() -> None:
    base = [True] * 30
    candidata = [False] * 12 + [True] * 18
    resultado = mcnemar_exact(base, candidata)
    assert resultado.b == 12 and resultado.c == 0
    assert resultado.p_value < 0.01
    assert resultado.effect < 0


def test_mcnemar_es_simetrico_en_la_direccion() -> None:
    base = [True] * 10 + [False] * 10
    candidata = [False] * 10 + [True] * 10
    hacia_abajo = mcnemar_exact(base, candidata)
    hacia_arriba = mcnemar_exact(candidata, base)
    assert hacia_abajo.p_value == hacia_arriba.p_value
    assert hacia_abajo.effect == -hacia_arriba.effect


def test_fisher_exacto_coincide_con_valores_conocidos() -> None:
    # Tabla clásica de la "dama que cata el té": 3/4 aciertos contra 1/4.
    assert abs(fisher_exact_2x2(3, 1, 1, 3) - 0.4857142857) < 1e-6
    # Separación total con muestras razonables: muy improbable por azar.
    assert fisher_exact_2x2(10, 0, 0, 10) < 0.001
    # Tablas idénticas: nada que detectar.
    assert fisher_exact_2x2(5, 5, 5, 5) == 1.0


def test_bootstrap_remuestrea_casos_y_cubre_el_cero_si_no_hay_efecto() -> None:
    base = [0.8] * 40
    candidata = [0.8] * 40
    resultado = cluster_bootstrap(base, candidata, iterations=500)
    assert resultado.effect == 0.0
    assert resultado.ci_low <= 0.0 <= resultado.ci_high


def test_bootstrap_excluye_el_cero_ante_una_caida_clara() -> None:
    rng = random.Random(7)
    base = [rng.uniform(0.85, 0.95) for _ in range(40)]
    candidata = [valor - 0.25 for valor in base]
    resultado = cluster_bootstrap(base, candidata, iterations=800)
    assert resultado.effect < -0.2
    assert resultado.excludes_zero


def test_el_bootstrap_de_dos_niveles_no_estrecha_el_intervalo() -> None:
    """Propagar la variabilidad intra-caso no puede dar más confianza, sino menos.

    Es la comprobación de que el remuestreo en dos niveles hace lo que dice: si
    saliera un intervalo más estrecho, estaría tratando las repeticiones como
    información independiente, que es justo lo que se quiere evitar.
    """
    rng = random.Random(11)
    base = [0.6] * 30
    candidata = [0.5] * 30
    reps_base = [[float(rng.random() < 0.6) for _ in range(10)] for _ in range(30)]
    reps_cand = [[float(rng.random() < 0.5) for _ in range(10)] for _ in range(30)]
    simple = cluster_bootstrap(base, candidata, iterations=600)
    dos_niveles = cluster_bootstrap(
        base, candidata, iterations=600, baseline_reps=reps_base, candidate_reps=reps_cand
    )
    ancho_simple = simple.ci_high - simple.ci_low
    ancho_dos = dos_niveles.ci_high - dos_niveles.ci_low
    assert ancho_dos >= ancho_simple


# --------------------------------------------------------------------------
# Comparaciones múltiples
# --------------------------------------------------------------------------


def test_benjamini_hochberg_es_monotono_y_no_supera_uno() -> None:
    p_values = [0.001, 0.008, 0.039, 0.041, 0.9]
    ajustados, rechazos = benjamini_hochberg(p_values, fdr=0.05)
    assert all(0.0 <= p <= 1.0 for p in ajustados)
    ordenados = [ajustados[i] for i in sorted(range(len(p_values)), key=lambda k: p_values[k])]
    assert ordenados == sorted(ordenados)
    assert rechazos[0] is True
    assert rechazos[-1] is False


def test_benjamini_hochberg_controla_la_avalancha_de_falsos_positivos() -> None:
    """Cuatrocientos contrastes de puro ruido no deben producir descubrimientos."""
    rng = random.Random(3)
    p_values = [rng.random() for _ in range(400)]
    _, rechazos = benjamini_hochberg(p_values, fdr=0.05)
    assert sum(rechazos) <= 5


def test_benjamini_hochberg_con_lista_vacia() -> None:
    assert benjamini_hochberg([]) == ([], [])


# --------------------------------------------------------------------------
# Relevancia práctica
# --------------------------------------------------------------------------


def test_significativo_pero_por_debajo_del_umbral_es_irrelevante() -> None:
    """CASO 3 del plan: p pequeño con efecto trivial no es una alarma."""
    resultado = classify_practical(-0.003, significant=True, minimum_practical_effect=0.02)
    assert resultado is PracticalSignificance.detectado_irrelevante


def test_relevante_sin_significacion_pide_mas_potencia() -> None:
    resultado = classify_practical(-0.15, significant=False, minimum_practical_effect=0.02)
    assert resultado is PracticalSignificance.relevante_no_concluyente


def test_significativo_y_relevante() -> None:
    resultado = classify_practical(-0.15, significant=True, minimum_practical_effect=0.02)
    assert resultado is PracticalSignificance.detectado_relevante


def test_ni_significativo_ni_relevante() -> None:
    resultado = classify_practical(-0.001, significant=False, minimum_practical_effect=0.02)
    assert resultado is PracticalSignificance.no_detectado


# --------------------------------------------------------------------------
# Fase 2: control negativo y control positivo
# --------------------------------------------------------------------------


def test_control_negativo_tasa_de_falsas_alarmas_compatible_con_alpha() -> None:
    """Criterio de aceptación 1 de la Fase 2.

    Con la misma configuración a ambos lados no puede exigirse CERO falsas
    alarmas —sería exigir un test sin potencia—, sino una tasa compatible con
    el alfa configurado. El contraste exacto de McNemar es conservador con
    pocos discordantes, así que la tasa observada queda por debajo de alfa,
    nunca por encima.
    """
    estimacion = estimate_power(
        n_cases=50, baseline_rate=0.9, difference=0.0, alpha=0.05, iterations=600, seed=101
    )
    assert estimacion.power <= 0.05 + 0.02, f"tasa de falsas alarmas {estimacion.power:.3f} por encima de alfa"


def test_control_negativo_efectos_centrados_en_cero() -> None:
    """Criterio de aceptación 1 de la Fase 2, segunda mitad."""
    rng = random.Random(5)
    efectos: list[float] = []
    for _ in range(40):
        base = [rng.random() < 0.9 for _ in range(50)]
        cand = [rng.random() < 0.9 for _ in range(50)]
        efectos.append(mcnemar_exact(base, cand).effect)
    media = sum(efectos) / len(efectos)
    assert abs(media) < 0.02, f"los efectos del control negativo no están centrados en cero: {media:.4f}"


def test_control_positivo_alcanza_la_potencia_minima() -> None:
    """Criterio de aceptación 2 de la Fase 2: una degradación conocida se ve."""
    estimacion = estimate_power(
        n_cases=50, baseline_rate=0.9, difference=-0.30, alpha=0.05, iterations=400, seed=202
    )
    assert estimacion.power >= 0.8, f"potencia insuficiente ante una caída del 30 %: {estimacion.power:.2f}"


def test_la_potencia_crece_con_la_magnitud_de_la_degradacion() -> None:
    curva = power_curve(n_cases=60, baseline_rate=0.9, iterations=250, seed=303)
    potencias = [punto.power for punto in curva]
    assert potencias == sorted(potencias), f"la curva de potencia no es monótona: {potencias}"


def test_el_efecto_minimo_detectable_mejora_con_mas_casos() -> None:
    """Un «sin cambio observable» sobre 12 casos no dice lo mismo que sobre 300."""
    pocos = minimum_detectable_effect(n_cases=15, iterations=150)
    muchos = minimum_detectable_effect(n_cases=200, iterations=150)
    assert pocos is not None and muchos is not None
    assert muchos < pocos


def test_el_efecto_minimo_detectable_sin_casos_es_desconocido() -> None:
    assert minimum_detectable_effect(n_cases=0) is None


# --------------------------------------------------------------------------
# El suelo de los contrastes exactos
# --------------------------------------------------------------------------


def test_con_pocas_repeticiones_el_contraste_tiene_un_suelo() -> None:
    """Con 3 repeticiones a cada lado, ninguna degradación baja de p = 0,1.

    Es el hecho que obliga al motor a devolver `indeterminado` en vez de
    `sin_cambio_observable`: presentar un cambio total como ausencia de cambio
    sería un falso negativo por construcción, no una conclusión.
    """
    assert minimum_attainable_p(3, 3, binary=True) == pytest.approx(0.1)
    # Con cinco ya se puede concluir con alfa = 0,05.
    assert minimum_attainable_p(5, 5, binary=True) < 0.05
    # Y el caso extremo real lo confirma: 3/3 contra 0/3 no llega.
    assert mcnemar_exact([True] * 3, [False] * 3).p_value > 0.05


def test_el_suelo_tambien_existe_en_el_contraste_de_permutacion() -> None:
    assert minimum_attainable_p(3, 3, binary=False) == pytest.approx(0.1)
    assert minimum_attainable_p(6, 6, binary=False) < 0.05
    assert minimum_attainable_p(0, 5, binary=True) == 1.0
