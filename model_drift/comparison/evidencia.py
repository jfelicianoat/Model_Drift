"""Qué evidencia evaluativa aporta un caso, y de qué tipo.

Es la puerta del nivel 2: sin evidencia no hay dirección, y sin dirección
el informe tiene prohibido decir «mejora» o «regresión» para ese caso. Las
métricas operativas (latencia, coste, tokens) son la excepción: su
dirección está definida de antemano.
"""
from __future__ import annotations

from dataclasses import dataclass

from model_drift.cases import Case
from model_drift.domain import (
    EXTERNAL_DEFINED_METRICS,
    METRIC_COST,
    METRIC_LATENCY,
    METRIC_TOKENS,
    EvidenceKind,
)
from model_drift.policy import RegressionPolicy

#: Métricas operativas: tienen dirección definida de antemano y por tanto
#: aportan evidencia de nivel 2 por sí mismas, sin oráculo ni juez.
OPERATIONAL_METRICS = frozenset(
    {METRIC_LATENCY, METRIC_COST, METRIC_TOKENS, "latency_p95", "tokens_output"}
    | EXTERNAL_DEFINED_METRICS
)


@dataclass
class CaseEvidence:
    """Qué evidencia evaluativa hay disponible para un caso concreto.

    Es la pieza que impide que una diferencia contra una línea base histórica
    se convierta sola en regresión.
    """

    kind: EvidenceKind | None = None
    confidence: float | None = None
    #: Si la métrica sube cuando el comportamiento mejora.
    higher_is_better: bool = True
    detail: str = ""


def evidence_for_case(
    case: Case | None,
    metric: str,
    *,
    policy: RegressionPolicy | None = None,
    judge_calibrated: bool = False,
    has_judge_verdict: bool = False,
) -> CaseEvidence:
    """Determina la evidencia disponible. Sin evidencia no habrá dirección.

    El orden refleja la escalera del plan §8.1: lo determinista manda sobre lo
    metamórfico, y lo metamórfico sobre la opinión de un modelo.
    """
    if metric in OPERATIONAL_METRICS:
        regla = policy.rule_for(metric) if policy else None
        return CaseEvidence(
            kind=EvidenceKind.metrica_definida,
            confidence=1.0,
            higher_is_better=regla.higher_is_better if regla else False,
            detail="métrica operativa con dirección declarada",
        )
    if case is None:
        return CaseEvidence(detail="el caso no está en la suite actual")
    if case.assertions:
        return CaseEvidence(
            kind=EvidenceKind.oraculo_determinista,
            confidence=1.0,
            detail=f"{len(case.assertions)} aserciones deterministas",
        )
    if case.metamorphic_relations:
        return CaseEvidence(
            kind=EvidenceKind.relacion_metamorfica,
            confidence=0.9,
            detail="relación metamórfica formal",
        )
    if has_judge_verdict and judge_calibrated:
        return CaseEvidence(kind=EvidenceKind.juez_calibrado, confidence=0.7, detail="juez calibrado")
    if case.human_label is not None:
        return CaseEvidence(kind=EvidenceKind.evaluacion_humana, confidence=1.0, detail="etiqueta humana")
    # Caso capturado sin nada añadido en la promoción: su salida histórica es
    # línea base de cambio, no verdad. Solo puede llegar al nivel 1.
    return CaseEvidence(
        detail=(
            "sin evidencia evaluativa: la salida histórica es línea base de cambio, "
            "no verdad de referencia"
        )
    )
