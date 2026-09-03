"""Lo que entra a la comparación (`RunResults`) y lo que sale (`ComparisonReport`)."""
from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from model_drift.comparison.evidencia import OPERATIONAL_METRICS
from model_drift.domain import CaseOutcome, ChangeStatus, ComparisonOutcome
from model_drift.manifest import ComparabilityReport, RunManifest
from model_drift.policy import RunVerdictReport


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
