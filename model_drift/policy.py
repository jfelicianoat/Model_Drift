"""Política de regresión: de los números a una decisión, sin score global.

Un número único que resuma «cómo de bien va el modelo» sería cómodo y sería
falso. Distintas métricas tienen distinta severidad —que la conformidad de
esquema caiga un punto puede ser inaceptable mientras que un 12 % más de
latencia solo merece un aviso— y eso lo decide quien conoce el producto, no un
promedio ponderado inventado por la herramienta.

Por eso aquí no hay puntuación agregada: hay una `RegressionPolicy` declarada
por suite, con métricas **bloqueantes** y de **aviso**, tolerancias y efectos
mínimos de interés. Y cuando unas métricas mejoran y otras empeoran, el
veredicto es `trade_off_detected` con el desglose: fabricar objetividad para
zanjar el dilema sería exactamente el tipo de conclusión más fuerte que la
evidencia que este proyecto existe para evitar.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from model_drift.domain import (
    ChangeDirection,
    Comparability,
    ComparisonOutcome,
    PracticalSignificance,
    RunVerdict,
    default_higher_is_better,
)


class Severity(str, Enum):
    blocking = "blocking"
    warning = "warning"
    #: Se mide y se informa, pero no participa en el veredicto.
    informative = "informative"


class MetricRule(BaseModel):
    """Tolerancia y efecto mínimo de interés de una métrica.

    `higher_is_better` decide qué es empeorar: en exactitud se empeora bajando;
    en latencia y coste, subiendo. Sin este campo la herramienta tendría que
    adivinarlo por el nombre de la métrica, que es como se cuelan los errores
    silenciosos.
    """

    severity: Severity = Severity.warning
    higher_is_better: bool = True
    #: Caída máxima tolerada, en las unidades de la métrica (puntos de tasa
    #: para proporciones: 0.02 = 2 pp).
    max_drop: float | None = None
    #: Subida máxima tolerada, en fracción (0.15 = 15 %). Para latencia y coste.
    max_increase_pct: float | None = None
    #: Efecto mínimo de interés (SESOI). Por debajo, un cambio significativo se
    #: informa como `detectado_irrelevante` y no dispara el veredicto.
    minimum_practical_effect: float = 0.0
    description: str = ""

    def worsens(self, effect: float) -> bool:
        """Si un efecto (candidata - base) va en la dirección mala."""
        return effect < 0 if self.higher_is_better else effect > 0

    def exceeds_tolerance(self, effect: float, baseline: float | None) -> bool:
        """Si el efecto se sale de lo tolerado por la política.

        Solo se evalúa en la dirección mala: una mejora nunca incumple una
        tolerancia.
        """
        if not self.worsens(effect):
            return False
        magnitud = abs(effect)
        if self.max_drop is not None and magnitud > self.max_drop:
            return True
        if self.max_increase_pct is not None:
            if baseline is None or baseline == 0:
                # Sin base no se puede hablar de porcentaje. No se inventa: se
                # cae al criterio absoluto si lo hay, y si no, no incumple.
                return self.max_drop is not None and magnitud > self.max_drop
            if magnitud / abs(baseline) > self.max_increase_pct:
                return True
        return False


class RegressionPolicy(BaseModel):
    """Política declarada de una suite. Sin ella no hay veredicto operativo."""

    metrics: dict[str, MetricRule] = Field(default_factory=dict)
    #: Tasa de falsos descubrimientos para la corrección de comparaciones
    #: múltiples.
    fdr: float = Field(default=0.05, gt=0.0, lt=1.0)
    #: Fracción máxima de casos indeterminados por encima de la cual la tirada
    #: entera no concluye: un veredicto calculado sobre la mitad de la suite no
    #: es un veredicto.
    max_indeterminate_ratio: float = Field(default=0.3, ge=0.0, le=1.0)

    def rule_for(self, metric: str) -> MetricRule:
        """Regla de una métrica. Las no declaradas son informativas.

        No se inventa severidad: una métrica que nadie declaró no puede
        bloquear una entrega. Lo que sí se respeta es su **dirección natural**:
        una latencia que baja es una mejora aunque nadie la haya declarado, y
        heredar el `higher_is_better=True` del defecto la convertiría en
        regresión.
        """
        declarada = self.metrics.get(metric)
        if declarada is not None:
            return declarada
        return MetricRule(
            severity=Severity.informative,
            higher_is_better=default_higher_is_better(metric),
        )

    def minimum_practical_effect(self, metric: str) -> float:
        return self.rule_for(metric).minimum_practical_effect

    @classmethod
    def from_file(cls, path: str | Path) -> RegressionPolicy:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> RegressionPolicy:
        """Acepta la forma corta del plan §13.2 y la forma explícita.

        Forma corta::

            blocking:
              schema_compliance: {max_drop_pp: 0}
            warning:
              latency_p95: {max_increase_pct: 15}
        """
        if "metrics" in raw:
            return cls.model_validate(raw)
        metrics: dict[str, MetricRule] = {}
        for severity in (Severity.blocking, Severity.warning, Severity.informative):
            bloque = raw.get(severity.value) or {}
            for nombre, config in bloque.items():
                config = dict(config or {})
                # `_pp` y `_pct` se escriben en unidades humanas y se guardan en
                # fracción: 2 pp -> 0.02, 15 % -> 0.15.
                if "max_drop_pp" in config:
                    config["max_drop"] = float(config.pop("max_drop_pp")) / 100.0
                if "max_increase_pct" in config:
                    config["max_increase_pct"] = float(config["max_increase_pct"]) / 100.0
                if "minimum_practical_effect_pp" in config:
                    config["minimum_practical_effect"] = float(config.pop("minimum_practical_effect_pp")) / 100.0
                if "minimum_practical_effect_pct" in config:
                    config["minimum_practical_effect"] = float(config.pop("minimum_practical_effect_pct")) / 100.0
                config.setdefault("severity", severity.value)
                metrics[nombre] = MetricRule.model_validate(config)
        extra = {key: raw[key] for key in ("fdr", "max_indeterminate_ratio") if key in raw}
        return cls(metrics=metrics, **extra)


class MetricVerdict(BaseModel):
    """Cómo queda una métrica concreta tras aplicar la política."""

    metric: str
    severity: Severity
    effect: float | None = None
    baseline: float | None = None
    candidate: float | None = None
    worsens: bool = False
    exceeds_tolerance: bool = False
    practical_significance: PracticalSignificance = PracticalSignificance.no_detectado
    n_regressions: int = 0
    n_improvements: int = 0
    detail: str = ""

    @property
    def is_relevant_change(self) -> bool:
        return self.practical_significance is PracticalSignificance.detectado_relevante


class RunVerdictReport(BaseModel):
    """Veredicto de una comparación entre dos tiradas."""

    verdict: RunVerdict
    metrics: list[MetricVerdict] = Field(default_factory=list)
    blocking_exceeded: list[str] = Field(default_factory=list)
    warnings_raised: list[str] = Field(default_factory=list)
    improved: list[str] = Field(default_factory=list)
    worsened: list[str] = Field(default_factory=list)
    indeterminate_cases: int = 0
    total_cases: int = 0
    reasons: list[str] = Field(default_factory=list)

    @property
    def trade_off(self) -> bool:
        return self.verdict is RunVerdict.trade_off_detected

    def summary(self) -> str:
        if self.verdict is RunVerdict.trade_off_detected:
            return (
                "Compensación detectada: mejoran "
                + ", ".join(self.improved)
                + "; empeoran "
                + ", ".join(self.worsened)
                + ". La decisión depende de la política, no de la herramienta."
            )
        if self.verdict is RunVerdict.blocked:
            return "Bloqueado por: " + ", ".join(self.blocking_exceeded) + "."
        if self.verdict is RunVerdict.warning:
            return "Avisos en: " + ", ".join(self.warnings_raised) + "."
        if self.verdict is RunVerdict.indeterminado:
            return "; ".join(self.reasons) or "No hay evidencia suficiente para concluir."
        return "Sin cambios relevantes según la política aplicada."


def build_metric_verdict(
    metric: str,
    outcomes: list[ComparisonOutcome],
    policy: RegressionPolicy,
) -> MetricVerdict:
    """Resume todos los casos de una métrica en un veredicto de métrica."""
    rule = policy.rule_for(metric)
    usables = [item for item in outcomes if item.comparability is Comparability.ok and item.effect_size is not None]
    regresiones = sum(1 for item in usables if item.change_direction is ChangeDirection.regresion)
    mejoras = sum(1 for item in usables if item.change_direction is ChangeDirection.mejora)

    if not usables:
        return MetricVerdict(
            metric=metric,
            severity=rule.severity,
            n_regressions=regresiones,
            n_improvements=mejoras,
            detail="sin casos comparables",
        )

    efectos = [item.effect_size for item in usables if item.effect_size is not None]
    efecto = sum(efectos) / len(efectos)
    bases = [item.baseline_score for item in usables if item.baseline_score is not None]
    candidatos = [item.candidate_score for item in usables if item.candidate_score is not None]
    base = sum(bases) / len(bases) if bases else None
    candidato = sum(candidatos) / len(candidatos) if candidatos else None

    # La relevancia de la métrica la marca el caso más severo que tenga: una
    # regresión relevante en un solo prompt crítico no puede diluirse en la
    # media de cincuenta prompts que no cambiaron.
    relevancia = PracticalSignificance.no_detectado
    orden = {
        PracticalSignificance.no_detectado: 0,
        PracticalSignificance.detectado_irrelevante: 1,
        PracticalSignificance.relevante_no_concluyente: 2,
        PracticalSignificance.detectado_relevante: 3,
    }
    for item in usables:
        if orden[item.practical_significance] > orden[relevancia]:
            relevancia = item.practical_significance

    empeora = rule.worsens(efecto)
    excede = rule.exceeds_tolerance(efecto, base) and relevancia is PracticalSignificance.detectado_relevante

    return MetricVerdict(
        metric=metric,
        severity=rule.severity,
        effect=efecto,
        baseline=base,
        candidate=candidato,
        worsens=empeora,
        exceeds_tolerance=excede,
        practical_significance=relevancia,
        n_regressions=regresiones,
        n_improvements=mejoras,
        detail=f"{len(usables)} casos comparables",
    )


def evaluate_policy(
    outcomes_by_metric: dict[str, list[ComparisonOutcome]],
    policy: RegressionPolicy,
    *,
    total_cases: int = 0,
    indeterminate_cases: int = 0,
) -> RunVerdictReport:
    """Aplica la política y devuelve el veredicto operativo de la comparación."""
    veredictos = [
        build_metric_verdict(metric, outcomes, policy)
        for metric, outcomes in sorted(outcomes_by_metric.items())
    ]

    bloqueantes = [
        item.metric for item in veredictos if item.severity is Severity.blocking and item.exceeds_tolerance
    ]
    avisos = [item.metric for item in veredictos if item.severity is Severity.warning and item.exceeds_tolerance]
    empeoran = [item.metric for item in veredictos if item.exceeds_tolerance]
    mejoran = [
        item.metric
        for item in veredictos
        if item.severity is not Severity.informative
        and item.effect is not None
        and not item.worsens
        and item.is_relevant_change
    ]

    razones: list[str] = []
    if total_cases and indeterminate_cases / total_cases > policy.max_indeterminate_ratio:
        razones.append(
            f"{indeterminate_cases} de {total_cases} casos indeterminados, por encima del "
            f"{policy.max_indeterminate_ratio:.0%} tolerado: la tirada no concluye"
        )
        veredicto = RunVerdict.indeterminado
    elif mejoran and empeoran:
        veredicto = RunVerdict.trade_off_detected
    elif bloqueantes:
        veredicto = RunVerdict.blocked
    elif avisos:
        veredicto = RunVerdict.warning
    else:
        veredicto = RunVerdict.pass_

    return RunVerdictReport(
        verdict=veredicto,
        metrics=veredictos,
        blocking_exceeded=bloqueantes,
        warnings_raised=avisos,
        improved=mejoran,
        worsened=empeoran,
        indeterminate_cases=indeterminate_cases,
        total_cases=total_cases,
        reasons=razones,
    )
