"""Corrección por comparaciones múltiples (Benjamini-Hochberg).

La familia de contrastes se declara antes de mirar los datos: todos los
casos de una métrica. Corregir sobre una familia elegida después de ver los
resultados es elegir el resultado.
"""
from __future__ import annotations


def benjamini_hochberg(p_values: list[float], *, fdr: float = 0.05) -> tuple[list[float], list[bool]]:
    """Corrección de Benjamini-Hochberg. Devuelve (p ajustados, rechazos).

    Cincuenta prompts por ocho métricas son cuatrocientos contrastes: sin
    corrección, con α = 0,05, salen veinte "regresiones" de puro azar. Se
    controla la tasa de falsos descubrimientos en vez de la familiar, porque en
    detección de deriva importa más no ahogarse en falsas alarmas que blindar
    contra un único falso positivo.

    La familia de contrastes se declara ANTES de mirar los resultados; este
    módulo solo la corrige.
    """
    m = len(p_values)
    if m == 0:
        return [], []
    indexados = sorted(enumerate(p_values), key=lambda par: par[1])
    ajustados = [0.0] * m
    minimo = 1.0
    # Recorrido descendente para imponer monotonía: un p ajustado nunca puede
    # ser mayor que el de un contraste con p mayor.
    for rango in range(m, 0, -1):
        indice, p = indexados[rango - 1]
        candidato = min(1.0, p * m / rango)
        minimo = min(minimo, candidato)
        ajustados[indice] = minimo
    rechazos = [ajustado <= fdr for ajustado in ajustados]
    return ajustados, rechazos
