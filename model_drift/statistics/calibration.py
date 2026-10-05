"""Intervalos exactos Clopper-Pearson, sin aproximación normal ni dependencias."""

from __future__ import annotations

import math


def _beta_fraction(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1, a - 1
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, 501):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = c if abs(c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = c if abs(c) > 1e-300 else 1e-300
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            return h
    raise ArithmeticError("no converge la beta incompleta")


def _beta_cdf(x: float, a: float, b: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _beta_fraction(a, b, x) / a
    return 1.0 - bt * _beta_fraction(b, a, 1.0 - x) / b


def _beta_quantile(q: float, a: float, b: float) -> float:
    low, high = 0.0, 1.0
    for _ in range(65):
        mid = (low + high) / 2
        if _beta_cdf(mid, a, b) < q:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def clopper_pearson(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total < 0 or not 0 <= successes <= total or not 0 < confidence < 1:
        raise ValueError("conteos o confianza inválidos")
    if total == 0:
        return 0.0, 1.0
    alpha = (1.0 - confidence) / 2.0
    low = _beta_quantile(alpha, successes, total - successes + 1) if successes else 0.0
    high = _beta_quantile(1.0 - alpha, successes + 1, total - successes) if successes < total else 1.0
    return low, high
