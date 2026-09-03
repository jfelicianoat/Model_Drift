"""Agregación de las repeticiones de un caso a una observación.

La unidad de la comparación es el caso, nunca la repetición: aquí es donde
N medidas se convierten en una, con su estabilidad medida al lado.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, median


@dataclass(frozen=True)
class CaseAggregate:
    """Resumen de las N repeticiones de un caso. Una sola observación."""

    score: float | None
    stability: float
    repetitions: int
    values: tuple[float, ...] = ()

    @property
    def is_empty(self) -> bool:
        return self.repetitions == 0 or self.score is None


def aggregate_binary(values: list[bool]) -> CaseAggregate:
    """Agrega repeticiones binarias en tasa de acierto y acuerdo.

    `stability` es la proporción del valor mayoritario: 1.0 si todas las
    repeticiones coinciden, 0.5 si están repartidas al 50 %. Un caso que dice
    que sí la mitad de las veces no mide el modelo, mide el azar.
    """
    if not values:
        return CaseAggregate(score=None, stability=0.0, repetitions=0)
    aciertos = sum(1 for value in values if value)
    tasa = aciertos / len(values)
    estabilidad = max(tasa, 1.0 - tasa)
    return CaseAggregate(
        score=tasa,
        stability=estabilidad,
        repetitions=len(values),
        values=tuple(1.0 if value else 0.0 for value in values),
    )


def aggregate_continuous(values: list[float], *, use_median: bool = True) -> CaseAggregate:
    """Agrega repeticiones continuas (latencia, tokens, similitud).

    `stability` = 1 - coeficiente de variación acotado a [0, 1]. Una latencia
    que oscila un 5 % es estable; una que oscila un 200 % no mide nada.
    """
    if not values:
        return CaseAggregate(score=None, stability=0.0, repetitions=0)
    centro = median(values) if use_median else mean(values)
    if len(values) == 1:
        return CaseAggregate(score=centro, stability=1.0, repetitions=1, values=(values[0],))
    promedio = mean(values)
    varianza = sum((value - promedio) ** 2 for value in values) / (len(values) - 1)
    desviacion = math.sqrt(varianza)
    if promedio == 0.0:
        estabilidad = 1.0 if desviacion == 0.0 else 0.0
    else:
        estabilidad = max(0.0, 1.0 - desviacion / abs(promedio))
    return CaseAggregate(score=centro, stability=min(1.0, estabilidad), repetitions=len(values), values=tuple(values))


def binarize(score: float, rule: str) -> bool:
    """Convierte `case_score` en un estado binario según la regla de la suite.

    `todas` es el defecto para oráculos deterministas: un caso que falla 3 de
    10 veces no está aprobado. `mayoria` pierde información a propósito y la
    suite tiene que declararlo.
    """
    if rule == "mayoria":
        return score > 0.5
    return score >= 1.0
