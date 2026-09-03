"""Relevancia práctica: significativo no es lo mismo que importante."""
from __future__ import annotations

from model_drift.domain import PracticalSignificance


def classify_practical(
    effect: float,
    *,
    significant: bool,
    minimum_practical_effect: float,
) -> PracticalSignificance:
    """Cruza significación estadística con relevancia práctica (plan §13.1).

    Con suficientes casos, una diferencia de 0,3 puntos porcentuales sale
    significativa y no le importa a nadie. `detectado_irrelevante` es un
    resultado legítimo y frecuente, y el informe debe decirlo con ese nombre en
    vez de alarmar.
    """
    # El umbral práctico se aplica a la MAGNITUD del cambio, en cualquier
    # dirección: una mejora grande también es relevante para quien decide si
    # adopta la versión nueva. La dirección la aporta `higher_is_better` a
    # quien construye el veredicto (policy.py), no esta clasificación.
    magnitud = abs(effect)
    supera_umbral = magnitud >= minimum_practical_effect
    if significant and supera_umbral:
        return PracticalSignificance.detectado_relevante
    if significant:
        return PracticalSignificance.detectado_irrelevante
    if supera_umbral:
        return PracticalSignificance.relevante_no_concluyente
    return PracticalSignificance.no_detectado
