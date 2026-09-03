"""Nivel 1 (¿cambió?) y nivel 2 (¿en qué dirección?), caso a caso.

El contraste dentro del caso mira sus repeticiones; la corrección por
comparaciones múltiples se aplica fuera, sobre la familia completa de la
métrica. Un caso sin potencia suficiente no se llama «sin cambio»: se llama
indeterminado.
"""
from __future__ import annotations

from model_drift.comparison.evidencia import CaseEvidence
from model_drift.domain import (
    CaseOutcome,
    ChangeDirection,
    ChangeStatus,
    EvidenceKind,
    is_binary_metric,
)
from model_drift.statistics import (
    fisher_exact_2x2,
    minimum_attainable_p,
    permutation_test,
)


def _per_case_p_value(metric: str, baseline: CaseOutcome, candidate: CaseOutcome) -> float:
    """Contraste dentro del caso: ¿cambió el comportamiento de ESTE caso?

    Legítimo porque, con la configuración fija, las repeticiones de un mismo
    caso sí son extracciones independientes de la distribución de ese caso. Lo
    que no lo sería —y lo impide el piso 2— es juntar las repeticiones de todos
    los casos como si fueran unidades independientes.

    El contraste depende del tipo de métrica: exacto de Fisher sobre aciertos y
    fallos para las binarias, permutación sobre las repeticiones para las
    continuas. Contar «aciertos» de una latencia no significa nada.
    """
    if baseline.case_score is None or candidate.case_score is None:
        return 1.0
    if baseline.measurement_deterministic and candidate.measurement_deterministic:
        return 0.0 if baseline.case_score != candidate.case_score else 1.0
    n_base, n_cand = baseline.repetitions, candidate.repetitions
    if n_base == 0 or n_cand == 0:
        return 1.0

    if is_binary_metric(metric):
        aciertos_base = min(n_base, max(0, round(baseline.case_score * n_base)))
        aciertos_cand = min(n_cand, max(0, round(candidate.case_score * n_cand)))
        return fisher_exact_2x2(
            aciertos_base,
            n_base - aciertos_base,
            aciertos_cand,
            n_cand - aciertos_cand,
        )
    return permutation_test(list(baseline.values), list(candidate.values))


def _is_underpowered(metric: str, baseline: CaseOutcome, candidate: CaseOutcome, fdr: float) -> bool:
    """Si este caso no puede alcanzar significación ni en el mejor de los casos.

    Con pocas repeticiones el contraste exacto tiene un suelo por encima del
    nivel exigido: da igual lo brutal que sea la degradación, el p-valor no
    puede bajar. Reconocerlo evita el peor error posible aquí, que es informar
    un cambio total como «sin cambio observable».
    """
    if baseline.measurement_deterministic and candidate.measurement_deterministic:
        return False
    return minimum_attainable_p(
        baseline.repetitions, candidate.repetitions, binary=is_binary_metric(metric)
    ) > fdr


def _status_for_case(
    baseline: CaseOutcome | None,
    candidate: CaseOutcome | None,
    *,
    significant: bool,
    comparable: bool,
    underpowered: bool = False,
) -> tuple[ChangeStatus, list[str]]:
    """Nivel 1. No juzga calidad; solo dice si algo cambió y si es interpretable."""
    notas: list[str] = []
    if not comparable:
        return ChangeStatus.indeterminado, ["el instrumento de evaluación cambió entre las dos tiradas"]
    if baseline is None or candidate is None:
        return ChangeStatus.indeterminado, ["el caso no está presente en las dos tiradas"]
    if baseline.is_not_applicable or candidate.is_not_applicable:
        motivo = candidate.not_applicable_reason or baseline.not_applicable_reason or "capacidad ausente"
        return ChangeStatus.indeterminado, [f"no aplica: {motivo}"]

    # Fallo de EJECUCIÓN, no de calidad: la candidata dejó de producir salida
    # evaluable donde la base sí la producía.
    base_produce = baseline.repetitions > baseline.failed_repetitions
    cand_produce = candidate.repetitions > candidate.failed_repetitions
    if base_produce and not cand_produce:
        return ChangeStatus.fallo_nuevo, ["dejó de producir salida evaluable"]

    if baseline.is_unstable or candidate.is_unstable:
        cual = "la base" if baseline.is_unstable else "la candidata"
        notas.append(f"caso inestable en {cual}: su propio ruido tapa cualquier señal")
        return ChangeStatus.indeterminado, notas

    if significant:
        return ChangeStatus.cambio_significativo, notas

    difieren = baseline.case_score != candidate.case_score
    if underpowered and difieren:
        # Ni «cambió» ni «no cambió»: con estas repeticiones no se puede saber.
        # Presentarlo como ausencia de cambio sería la mentira más cómoda.
        notas.append(
            f"{baseline.repetitions} y {candidate.repetitions} repeticiones no bastan para decidir "
            "este caso: el contraste exacto no puede bajar del nivel exigido por muy grande "
            "que sea la diferencia. Sube las repeticiones."
        )
        return ChangeStatus.indeterminado, notas
    return ChangeStatus.sin_cambio_observable, notas


def _direction_for_case(
    effect: float,
    evidence: CaseEvidence,
    status: ChangeStatus,
) -> tuple[ChangeDirection, EvidenceKind | None, float | None]:
    """Nivel 2. Devuelve `sin_direccion` siempre que falte evidencia.

    Este es el guardián de la regla de la línea base: sin `evidence.kind` no
    sale ninguna dirección, por mucho que el efecto sea grande y claro.
    """
    if evidence.kind is None:
        return ChangeDirection.sin_direccion, None, None
    if status not in {ChangeStatus.cambio_significativo, ChangeStatus.fallo_nuevo}:
        return ChangeDirection.sin_direccion, None, None
    if effect == 0.0:
        return ChangeDirection.cambio_neutro, evidence.kind, evidence.confidence
    empeora = effect < 0 if evidence.higher_is_better else effect > 0
    direccion = ChangeDirection.regresion if empeora else ChangeDirection.mejora
    return direccion, evidence.kind, evidence.confidence
