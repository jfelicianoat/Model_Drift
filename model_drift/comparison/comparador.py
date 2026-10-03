"""`compare_runs`: el recorrido completo, de dos tiradas a un informe.

La función se lee como el orden de operaciones del plan: comparabilidad,
cambio, dirección, agregación entre casos y política. Cada paso está en su
propio método para que se pueda leer uno sin leer los otros; antes era un solo
bucle de doscientas líneas donde el orden se adivinaba por la sangría.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from model_drift.broker_client import BrokerClient
from model_drift.cases import Suite
from model_drift.comparison.contraste import (
    _direction_for_case,
    _is_underpowered,
    _per_case_p_value,
    _status_for_case,
)
from model_drift.comparison.evidencia import evidence_for_case
from model_drift.comparison.resultados import ComparisonReport, RunResults
from model_drift.domain import (
    CaseOutcome,
    ChangeDirection,
    ChangeStatus,
    Comparability,
    ComparisonOutcome,
)
from model_drift.judge import Judge
from model_drift.manifest import ComparabilityReport, compare_manifests
from model_drift.policy import RegressionPolicy, evaluate_policy
from model_drift.statistics import (
    benjamini_hochberg,
    binarize,
    classify_practical,
    cluster_bootstrap,
    mcnemar_exact,
    minimum_detectable_effect,
)
from model_drift.system1 import review_semantic_pairs
from model_drift.system1_contracts import System1Config
from model_drift.taxonomy import classify_change

#: Un caso crudo antes de corregir por comparaciones múltiples.
_Crudo = tuple[str, "CaseOutcome | None", "CaseOutcome | None", float]


@dataclass
class _Recuento:
    """Cuentas de la comparación, todas en la misma unidad: casos.

    Se llevan aparte porque se acumulan a lo largo de todas las métricas y el
    informe las necesita al final. Contarlas en unidades distintas es lo que
    produce cabeceras como «4 casos, 16 indeterminados».
    """

    indeterminados: set[str] = field(default_factory=set)
    no_aplica: set[str] = field(default_factory=set)
    inestables: set[str] = field(default_factory=set)
    comparados: int = 0

    def anotar(self, case_id: str, estado: ChangeStatus, base: CaseOutcome | None, cand: CaseOutcome | None) -> None:
        if estado is ChangeStatus.indeterminado:
            self.indeterminados.add(case_id)
        if (base and base.is_not_applicable) or (cand and cand.is_not_applicable):
            self.no_aplica.add(case_id)
        if (base and base.is_unstable) or (cand and cand.is_unstable):
            self.inestables.add(case_id)


@dataclass
class _Contexto:
    """Lo que no cambia mientras se recorren métricas y casos."""

    baseline: RunResults
    candidate: RunResults
    suite: Suite | None
    policy: RegressionPolicy
    comparabilidad: ComparabilityReport
    judged: set[str]
    judge_calibrated: bool

    @property
    def comparable(self) -> bool:
        return self.comparabilidad.comparable


def compare_runs(
    baseline: RunResults,
    candidate: RunResults,
    *,
    suite: Suite | None = None,
    policy: RegressionPolicy | None = None,
    judge_calibrated: bool = False,
    judged_cases: set[str] | None = None,
    system1: System1Config | None = None,
    broker: BrokerClient | None = None,
    judge: Judge | None = None,
) -> ComparisonReport:
    """Compara dos tiradas y produce el informe completo."""
    contexto = _Contexto(
        baseline=baseline,
        candidate=candidate,
        suite=suite,
        policy=policy or RegressionPolicy(),
        comparabilidad=compare_manifests(
            baseline.manifest,
            candidate.manifest,
            baseline_model_key=baseline.model_key,
            candidate_model_key=candidate.model_key,
        ),
        judged=judged_cases or set(),
        judge_calibrated=judge_calibrated,
    )
    recuento = _Recuento()

    outcomes: list[ComparisonOutcome] = []
    por_metrica: dict[str, list[ComparisonOutcome]] = {}
    for metrica in sorted(baseline.metrics() | candidate.metrics()):
        resultados_metrica = _comparar_metrica(metrica, contexto, recuento)
        outcomes.extend(resultados_metrica)
        por_metrica[metrica] = resultados_metrica

    report = _informe(outcomes, por_metrica, contexto, recuento)
    if system1 is not None and system1.enabled:
        report.semantic_review = review_semantic_pairs(
            report, baseline, candidate, suite=suite, policy=contexto.policy,
            config=system1, client=broker, judge=judge,
        )
    return report


# ------------------------------------------------------------------ métricas


def _comparar_metrica(metrica: str, contexto: _Contexto, recuento: _Recuento) -> list[ComparisonOutcome]:
    """Todos los casos de una métrica, corregidos sobre su propia familia."""
    crudos = _crudos_de_metrica(metrica, contexto)
    # La familia de contrastes se declara antes de mirar: todos los casos de
    # esta métrica. La corrección se aplica sobre esa familia completa.
    ajustados, rechazos = benjamini_hochberg([p for *_, p in crudos], fdr=contexto.policy.fdr)

    resultados: list[ComparisonOutcome] = []
    for (case_id, base, cand, p), p_ajustado, rechazo in zip(crudos, ajustados, rechazos, strict=True):
        outcome = _outcome_de_caso(
            case_id, base, cand, metrica, p, p_ajustado, rechazo=rechazo, contexto=contexto
        )
        resultados.append(outcome)
        recuento.anotar(case_id, outcome.change_status, base, cand)

    _agregar_entre_casos(metrica, crudos, resultados, contexto, recuento)
    return resultados


def _crudos_de_metrica(metrica: str, contexto: _Contexto) -> list[_Crudo]:
    """Empareja los casos de las dos tiradas y calcula su p sin corregir."""
    case_ids = sorted(contexto.baseline.case_ids(metrica) | contexto.candidate.case_ids(metrica))
    crudos: list[_Crudo] = []
    for case_id in case_ids:
        base = contexto.baseline.outcomes.get((case_id, metrica))
        cand = contexto.candidate.outcomes.get((case_id, metrica))
        p = _per_case_p_value(metrica, base, cand) if base and cand else 1.0
        crudos.append((case_id, base, cand, p))
    return crudos


# ---------------------------------------------------------------------- caso


def _outcome_de_caso(
    case_id: str,
    base: CaseOutcome | None,
    cand: CaseOutcome | None,
    metrica: str,
    p: float,
    p_ajustado: float,
    *,
    rechazo: bool,
    contexto: _Contexto,
) -> ComparisonOutcome:
    """Nivel 1 y nivel 2 de un caso concreto, con sus notas."""
    case = contexto.suite.case_by_id(case_id) if contexto.suite else None
    evidencia = evidence_for_case(
        case,
        metrica,
        policy=contexto.policy,
        judge_calibrated=contexto.judge_calibrated,
        has_judge_verdict=case_id in contexto.judged,
    )

    sin_potencia = (
        _is_underpowered(metrica, base, cand, contexto.policy.fdr)
        if base is not None and cand is not None
        else False
    )
    estado, notas = _status_for_case(
        base, cand, significant=rechazo, comparable=contexto.comparable, underpowered=sin_potencia
    )
    if not contexto.comparable:
        notas.extend(f"{issue.kind}: {issue.detail}" for issue in contexto.comparabilidad.issues)

    efecto = None
    if base is not None and cand is not None and base.case_score is not None and cand.case_score is not None:
        efecto = cand.case_score - base.case_score

    direccion, tipo_evidencia, confianza = (
        _direction_for_case(efecto, evidencia, estado)
        if efecto is not None
        else (ChangeDirection.sin_direccion, None, None)
    )
    sin_evidencia = evidencia.kind is None and estado is ChangeStatus.cambio_significativo
    if direccion is ChangeDirection.sin_direccion and sin_evidencia:
        notas.append(evidencia.detail)

    return ComparisonOutcome(
        case_id=case_id,
        metric=metrica,
        change_status=estado,
        change_direction=direccion,
        evidence_kind=tipo_evidencia,
        evidence_confidence=confianza,
        baseline_score=base.case_score if base else None,
        candidate_score=cand.case_score if cand else None,
        effect_size=efecto,
        p_value=p,
        p_value_corrected=p_ajustado,
        practical_significance=classify_practical(
            efecto or 0.0,
            significant=rechazo,
            minimum_practical_effect=contexto.policy.minimum_practical_effect(metrica),
        ),
        comparability=_comparabilidad_de_caso(contexto),
        change_taxonomy=_taxonomia(case_id, base, cand, contexto),
        notes=notas,
    )


def _comparabilidad_de_caso(contexto: _Contexto) -> Comparability:
    if contexto.comparable:
        return Comparability.ok
    return (
        Comparability.huella_divergente
        if contexto.comparabilidad.fingerprint_divergent
        else Comparability.instrumento_cambiado
    )


def _taxonomia(
    case_id: str, base: CaseOutcome | None, cand: CaseOutcome | None, contexto: _Contexto
) -> list[str]:
    """Qué cambió en la salida, cuando hay dos salidas que comparar."""
    if base is None or cand is None:
        return []
    texto_base = contexto.baseline.sample_output.get(case_id)
    texto_cand = contexto.candidate.sample_output.get(case_id)
    if texto_base is None or texto_cand is None:
        return []
    return classify_change(
        texto_base,
        texto_cand,
        baseline_tools=contexto.baseline.sample_tools.get(case_id),
        candidate_tools=contexto.candidate.sample_tools.get(case_id),
    )


# ------------------------------------------------------- agregación e informe


def _agregar_entre_casos(
    metrica: str,
    crudos: list[_Crudo],
    resultados: list[ComparisonOutcome],
    contexto: _Contexto,
    recuento: _Recuento,
) -> None:
    """McNemar y bootstrap por conglomerados. La unidad es el caso."""
    pares = [
        (base, cand)
        for _, base, cand, _ in crudos
        if base is not None and cand is not None and base.is_usable and cand.is_usable
    ]
    if not pares or not contexto.comparable:
        return

    recuento.comparados = max(recuento.comparados, len(pares))
    regla = contexto.suite.case_pass_rule if contexto.suite else "todas"
    base_bin = [binarize(base.case_score or 0.0, regla) for base, _ in pares]
    cand_bin = [binarize(cand.case_score or 0.0, regla) for _, cand in pares]
    mcnemar = mcnemar_exact(base_bin, cand_bin)
    bootstrap = cluster_bootstrap(
        [base.case_score or 0.0 for base, _ in pares],
        [cand.case_score or 0.0 for _, cand in pares],
        baseline_reps=[list(base.values) for base, _ in pares],
        candidate_reps=[list(cand.values) for _, cand in pares],
    )
    # El intervalo agregado se anota en cada caso de la métrica para que el
    # informe pueda enseñar la incertidumbre de la métrica junto al detalle del
    # caso, sin recalcular nada.
    for outcome in resultados:
        outcome.ci_low = bootstrap.ci_low
        outcome.ci_high = bootstrap.ci_high
    if mcnemar.discordant == 0:
        for outcome in resultados:
            outcome.notes.append("ningún caso discordante: no hay información de cambio en esta métrica")


def _informe(
    outcomes: list[ComparisonOutcome],
    por_metrica: dict[str, list[ComparisonOutcome]],
    contexto: _Contexto,
    recuento: _Recuento,
) -> ComparisonReport:
    """Cierra el informe: potencia, veredicto y aviso de comparabilidad."""
    total_casos = len({item.case_id for item in outcomes})
    mde = minimum_detectable_effect(n_cases=recuento.comparados) if recuento.comparados else None

    notas: list[str] = []
    if not contexto.comparable:
        notas.append(
            "Los manifiestos difieren: no se puede atribuir al modelo ninguna diferencia observada. "
            + contexto.comparabilidad.headline()
        )
    notas.extend(contexto.comparabilidad.fingerprint_notes)

    return ComparisonReport(
        baseline_run_id=contexto.baseline.run_id,
        candidate_run_id=contexto.candidate.run_id,
        comparability=contexto.comparabilidad,
        outcomes=outcomes,
        verdict=evaluate_policy(
            por_metrica,
            contexto.policy,
            total_cases=total_casos,
            indeterminate_cases=len(recuento.indeterminados),
        ),
        n_cases=total_casos,
        n_compared=recuento.comparados,
        # Todas las cuentas van en la misma unidad —casos— para que el informe
        # no ponga "4 casos, 16 indeterminados" uno al lado del otro.
        n_indeterminate=len(recuento.indeterminados),
        n_not_applicable=len(recuento.no_aplica),
        n_unstable=len(recuento.inestables),
        minimum_detectable_effect=mde,
        notes=notas,
    )
