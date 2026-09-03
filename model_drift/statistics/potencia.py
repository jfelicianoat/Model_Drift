"""Potencia estadística: qué habría podido detectarse con este número de casos.

Sin esto, «no se observó cambio» es una frase vacía. Con esto, el informe
puede decir qué degradación mínima habría visto y, por tanto, qué no puede
descartar.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from model_drift.statistics.contrastes import mcnemar_exact


@dataclass
class PowerEstimate:
    """Potencia estimada por simulación para una configuración concreta."""

    n_cases: int
    baseline_rate: float
    difference: float
    alpha: float
    power: float
    iterations: int
    detected: int = field(default=0)


def estimate_power(
    *,
    n_cases: int,
    baseline_rate: float,
    difference: float,
    alpha: float = 0.05,
    iterations: int = 400,
    seed: int = 20260815,
) -> PowerEstimate:
    """Simula pares y estima con qué frecuencia se detecta una diferencia dada.

    Sirve para las dos mitades del criterio de aceptación de la Fase 2:

    - con `difference = 0` estima la **tasa de falsas alarmas**, que debe
      quedar en el entorno de `alpha`;
    - con una degradación conocida estima la **potencia de detección**.

    Modelo de simulación: cada caso pasa en la base con probabilidad
    `baseline_rate` y en la candidata con `baseline_rate + difference`, de forma
    independiente entre configuraciones. Es deliberadamente conservador —los
    casos reales están correlacionados entre configuraciones, lo que reduce la
    discordancia y aumenta la potencia—, así que la potencia real será igual o
    mayor que la estimada aquí.
    """
    rng = random.Random(seed)
    rate_candidate = min(1.0, max(0.0, baseline_rate + difference))
    detectados = 0
    for _ in range(iterations):
        base = [rng.random() < baseline_rate for _ in range(n_cases)]
        cand = [rng.random() < rate_candidate for _ in range(n_cases)]
        if mcnemar_exact(base, cand).p_value <= alpha:
            detectados += 1
    return PowerEstimate(
        n_cases=n_cases,
        baseline_rate=baseline_rate,
        difference=difference,
        alpha=alpha,
        power=detectados / iterations if iterations else 0.0,
        iterations=iterations,
        detected=detectados,
    )


def minimum_detectable_effect(
    *,
    n_cases: int,
    baseline_rate: float = 0.9,
    alpha: float = 0.05,
    target_power: float = 0.8,
    iterations: int = 300,
    seed: int = 20260815,
) -> float | None:
    """Menor degradación que esta configuración detecta con la potencia pedida.

    Es el número que hace honesto un «sin cambio observable»: no es lo mismo
    sobre 12 casos que sobre 300, y el informe no debe permitir confundirlos.
    Devuelve `None` si ni siquiera una caída total alcanza la potencia objetivo.
    """
    if n_cases <= 0:
        return None
    paso = 0.01
    diferencia = paso
    while diferencia <= baseline_rate:
        estimate = estimate_power(
            n_cases=n_cases,
            baseline_rate=baseline_rate,
            difference=-diferencia,
            alpha=alpha,
            iterations=iterations,
            seed=seed,
        )
        if estimate.power >= target_power:
            return round(diferencia, 4)
        diferencia += paso
    return None


def power_curve(
    *,
    n_cases: int,
    baseline_rate: float = 0.9,
    differences: tuple[float, ...] = (-0.02, -0.05, -0.10, -0.20, -0.30),
    alpha: float = 0.05,
    iterations: int = 300,
    seed: int = 20260815,
) -> list[PowerEstimate]:
    """Curva de potencia por magnitud. Va al informe de la Fase 2."""
    return [
        estimate_power(
            n_cases=n_cases,
            baseline_rate=baseline_rate,
            difference=difference,
            alpha=alpha,
            iterations=iterations,
            seed=seed,
        )
        for difference in differences
    ]
