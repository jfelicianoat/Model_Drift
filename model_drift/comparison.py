"""Motor de comparación: de dos tiradas a un veredicto honesto.

El orden de las operaciones no es negociable y está tomado del plan §5, §9 y §11:

1. **Comparabilidad primero.** Si el instrumento de evaluación cambió, no hay
   nada que concluir sobre el modelo: los casos salen `indeterminado` y el
   informe lo dice en cabecera.
2. **Nivel 1, cambio.** Se detecta con las repeticiones de cada caso (test
   exacto de Fisher dentro del caso) y se corrige por comparaciones múltiples.
   No juzga calidad.
3. **Nivel 2, dirección.** Solo si el caso aporta evidencia evaluativa. Sin
   ella, `sin_direccion`, y el informe tiene prohibido usar las palabras
   «mejora» y «regresión» para ese caso.
4. **Agregación entre casos.** McNemar sobre casos pareados y bootstrap por
   conglomerados: la unidad es el caso, nunca la repetición.
5. **Relevancia práctica y política.** Significativo no es lo mismo que
   importante.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from model_drift.cases import Case, Suite
from model_drift.domain import (
    EXTERNAL_DEFINED_METRICS,
    METRIC_COST,
    METRIC_LATENCY,
    METRIC_TOKENS,
    CaseOutcome,
    ChangeDirection,
    ChangeStatus,
    Comparability,
    ComparisonOutcome,
    EvidenceKind,
    is_binary_metric,
)
from model_drift.manifest import ComparabilityReport, RunManifest, compare_manifests
from model_drift.policy import RegressionPolicy, RunVerdictReport, evaluate_policy
from model_drift.statistics import (
    benjamini_hochberg,
    binarize,
    classify_practical,
    cluster_bootstrap,
    fisher_exact_2x2,
    mcnemar_exact,
    minimum_attainable_p,
    minimum_detectable_effect,
    permutation_test,
)
from model_drift.taxonomy import classify_change

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


class ComparisonReport(BaseModel):
    """Resultado completo de comparar dos tiradas."""

    baseline_run_id: str
    candidate_run_id: str
    comparability: ComparabilityReport
    outcomes: list[ComparisonOutcome] = Field(default_factory=list)
    verdict: RunVerdictReport | None = None

    n_cases: int = 0
    n_compared: int = 0
    n_indeterminate: int = 0
    n_not_applicable: int = 0
    n_unstable: int = 0
    minimum_detectable_effect: float | None = None
    notes: list[str] = Field(default_factory=list)

    def by_status(self, status: ChangeStatus) -> list[ComparisonOutcome]:
        return [item for item in self.outcomes if item.change_status is status]

    def regressions(self) -> list[ComparisonOutcome]:
        return [item for item in self.outcomes if item.is_regression]

    def actionable(self) -> list[ComparisonOutcome]:
        return [item for item in self.outcomes if item.is_actionable]

    def content_outcomes(self) -> list[ComparisonOutcome]:
        """Solo las métricas sobre el CONTENIDO de la respuesta.

        Las operativas —latencia, coste, tokens— tienen dirección propia y
        pueden decir «mejora» con total legitimidad mientras el contenido sigue
        sin evidencia. Separarlas evita el peor malentendido posible de un
        informe: leer «capturado-01: mejora» y creer que mejoró la respuesta
        cuando lo que bajó fue el coste.
        """
        return [item for item in self.outcomes if item.metric not in OPERATIONAL_METRICS]

    def power_statement(self) -> str:
        """Frase que hace honesto un «sin cambio observable»."""
        if self.minimum_detectable_effect is None:
            return (
                f"Con {self.n_compared} casos comparados no se alcanza potencia suficiente para "
                "detectar ninguna degradación con garantías: la ausencia de señal no es evidencia de ausencia."
            )
        return (
            f"Con {self.n_compared} casos comparados, la menor degradación detectable con potencia 0,8 "
            f"es de {self.minimum_detectable_effect:.1%}. Por debajo de eso, la ausencia de señal "
            "no es evidencia de ausencia."
        )


@dataclass
class RunResults:
    """Lo que una tirada aporta a la comparación."""

    run_id: str
    manifest: RunManifest
    model_key: str
    #: (case_id, metric) -> resultado agregado del caso.
    outcomes: dict[tuple[str, str], CaseOutcome] = field(default_factory=dict)
    #: (case_id) -> texto de una salida representativa, para la taxonomía.
    sample_output: dict[str, str] = field(default_factory=dict)
    #: (case_id) -> herramientas invocadas en esa salida representativa.
    sample_tools: dict[str, list[str]] = field(default_factory=dict)

    def metrics(self) -> set[str]:
        return {metric for _, metric in self.outcomes}

    def case_ids(self, metric: str) -> set[str]:
        return {case_id for case_id, item_metric in self.outcomes if item_metric == metric}


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


def compare_runs(
    baseline: RunResults,
    candidate: RunResults,
    *,
    suite: Suite | None = None,
    policy: RegressionPolicy | None = None,
    judge_calibrated: bool = False,
    judged_cases: set[str] | None = None,
) -> ComparisonReport:
    """Compara dos tiradas y produce el informe completo."""
    policy = policy or RegressionPolicy()
    judged = judged_cases or set()

    comparabilidad = compare_manifests(
        baseline.manifest,
        candidate.manifest,
        baseline_model_key=baseline.model_key,
        candidate_model_key=candidate.model_key,
    )
    comparable = comparabilidad.comparable

    metricas = sorted(baseline.metrics() | candidate.metrics())
    outcomes: list[ComparisonOutcome] = []
    por_metrica: dict[str, list[ComparisonOutcome]] = {}

    casos_indeterminados: set[str] = set()
    casos_no_aplica: set[str] = set()
    casos_inestables: set[str] = set()
    n_comparados = 0

    for metrica in metricas:
        case_ids = sorted(baseline.case_ids(metrica) | candidate.case_ids(metrica))
        crudos: list[tuple[str, CaseOutcome | None, CaseOutcome | None, float]] = []
        p_values: list[float] = []

        for case_id in case_ids:
            base = baseline.outcomes.get((case_id, metrica))
            cand = candidate.outcomes.get((case_id, metrica))
            p = _per_case_p_value(metrica, base, cand) if base and cand else 1.0
            crudos.append((case_id, base, cand, p))
            p_values.append(p)

        # La familia de contrastes se declara antes de mirar: todos los casos de
        # esta métrica. La corrección se aplica sobre esa familia completa.
        ajustados, rechazos = benjamini_hochberg(p_values, fdr=policy.fdr)

        resultados_metrica: list[ComparisonOutcome] = []
        for (case_id, base, cand, p), p_ajustado, rechazo in zip(crudos, ajustados, rechazos, strict=True):
            case = suite.case_by_id(case_id) if suite else None
            evidencia = evidence_for_case(
                case,
                metrica,
                policy=policy,
                judge_calibrated=judge_calibrated,
                has_judge_verdict=case_id in judged,
            )

            sin_potencia = (
                _is_underpowered(metrica, base, cand, policy.fdr)
                if base is not None and cand is not None
                else False
            )
            estado, notas = _status_for_case(
                base, cand, significant=rechazo, comparable=comparable, underpowered=sin_potencia
            )
            if not comparable:
                notas.extend(f"{issue.kind}: {issue.detail}" for issue in comparabilidad.issues)

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

            relevancia = classify_practical(
                efecto or 0.0,
                significant=rechazo,
                minimum_practical_effect=policy.minimum_practical_effect(metrica),
            )

            etiquetas: list[str] = []
            if base is not None and cand is not None:
                texto_base = baseline.sample_output.get(case_id)
                texto_cand = candidate.sample_output.get(case_id)
                if texto_base is not None and texto_cand is not None:
                    etiquetas = classify_change(
                        texto_base,
                        texto_cand,
                        baseline_tools=baseline.sample_tools.get(case_id),
                        candidate_tools=candidate.sample_tools.get(case_id),
                    )

            if not comparable:
                comparabilidad_caso = (
                    Comparability.huella_divergente
                    if comparabilidad.fingerprint_divergent
                    else Comparability.instrumento_cambiado
                )
            else:
                comparabilidad_caso = Comparability.ok

            outcome = ComparisonOutcome(
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
                practical_significance=relevancia,
                comparability=comparabilidad_caso,
                change_taxonomy=etiquetas,
                notes=notas,
            )
            resultados_metrica.append(outcome)

            if estado is ChangeStatus.indeterminado:
                casos_indeterminados.add(case_id)
            if (base and base.is_not_applicable) or (cand and cand.is_not_applicable):
                casos_no_aplica.add(case_id)
            if (base and base.is_unstable) or (cand and cand.is_unstable):
                casos_inestables.add(case_id)

        # --- Agregación entre casos: la unidad es el caso ---
        pares = [
            (base, cand)
            for _, base, cand, _ in crudos
            if base is not None and cand is not None and base.is_usable and cand.is_usable
        ]
        if pares and comparable:
            n_comparados = max(n_comparados, len(pares))
            regla = suite.case_pass_rule if suite else "todas"
            base_bin = [binarize(base.case_score or 0.0, regla) for base, _ in pares]
            cand_bin = [binarize(cand.case_score or 0.0, regla) for _, cand in pares]
            mcnemar = mcnemar_exact(base_bin, cand_bin)
            bootstrap = cluster_bootstrap(
                [base.case_score or 0.0 for base, _ in pares],
                [cand.case_score or 0.0 for _, cand in pares],
                baseline_reps=[list(base.values) for base, _ in pares],
                candidate_reps=[list(cand.values) for _, cand in pares],
            )
            # El intervalo agregado se anota en cada caso de la métrica para que
            # el informe pueda enseñar la incertidumbre de la métrica junto al
            # detalle del caso, sin recalcular nada.
            for outcome in resultados_metrica:
                outcome.ci_low = bootstrap.ci_low
                outcome.ci_high = bootstrap.ci_high
            if mcnemar.discordant == 0:
                for outcome in resultados_metrica:
                    outcome.notes.append("ningún caso discordante: no hay información de cambio en esta métrica")

        outcomes.extend(resultados_metrica)
        por_metrica[metrica] = resultados_metrica

    total_casos = len({item.case_id for item in outcomes})
    mde = minimum_detectable_effect(n_cases=n_comparados) if n_comparados else None

    veredicto = evaluate_policy(
        por_metrica,
        policy,
        total_cases=total_casos,
        indeterminate_cases=len(casos_indeterminados),
    )

    notas_informe: list[str] = []
    if not comparable:
        notas_informe.append(
            "Los manifiestos difieren: no se puede atribuir al modelo ninguna diferencia observada. "
            + comparabilidad.headline()
        )
    notas_informe.extend(comparabilidad.fingerprint_notes)

    return ComparisonReport(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        comparability=comparabilidad,
        outcomes=outcomes,
        verdict=veredicto,
        n_cases=total_casos,
        n_compared=n_comparados,
        # Todas las cuentas van en la misma unidad —casos— para que el informe
        # no ponga "4 casos, 16 indeterminados" uno al lado del otro.
        n_indeterminate=len(casos_indeterminados),
        n_not_applicable=len(casos_no_aplica),
        n_unstable=len(casos_inestables),
        minimum_detectable_effect=mde,
        notes=notas_informe,
    )
