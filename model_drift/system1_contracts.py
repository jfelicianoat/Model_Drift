"""Configuración y auditoría del priorizador semántico; no son evidencia de dirección."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SemanticLabel = Literal["equivalent", "changed_but_not_regression", "regression_candidate", "uncertain"]
SEMANTIC_OPTIONS: tuple[SemanticLabel, ...] = (
    "equivalent", "changed_but_not_regression", "regression_candidate", "uncertain",
)


class System1Config(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    enabled: bool = False
    shadow_mode: bool = True
    use_case: str = Field(default="model_drift_semantic", pattern=r"^[A-Za-z0-9_-]{1,128}$")
    threshold_profile: str | None = Field(default="default", min_length=1, max_length=128)
    timeout_seconds: float = Field(default=75.0, gt=0)
    cloud_allowed: bool = False
    regression_threshold: float = Field(default=0.80, ge=0, le=1)
    high_priority_threshold: float = Field(default=0.95, ge=0, le=1)
    equivalence_threshold: float = Field(default=0.99, ge=0, le=1)
    allow_equivalence_skip: bool = False
    require_calibrated_for_skip: bool = True

    @model_validator(mode="after")
    def ordered_thresholds(self) -> System1Config:
        if self.high_priority_threshold < self.regression_threshold:
            raise ValueError("high_priority_threshold debe ser >= regression_threshold")
        return self


class System1Attempt(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    provider: str
    model: str | None = None
    latency_ms: float = Field(ge=0)
    reason_code: str | None = None
    tokens_input: int | None = Field(default=None, ge=0)
    tokens_output: int | None = Field(default=None, ge=0)


class System1Alternative(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    value: SemanticLabel
    confidence: float = Field(ge=0, le=1, strict=True)


class System1Judgment(BaseModel):
    """Respuesta del contrato 2.11 para nuestro juicio `choice`."""

    model_config = ConfigDict(allow_inf_nan=False)
    use_case: str
    accepted: bool = Field(strict=True)
    decision: SemanticLabel | None
    confidence: float | None = Field(ge=0, le=1, strict=True)
    confidence_is_calibrated: bool = Field(strict=True)
    provider: str | None
    model: str | None = None
    latency_ms: float = Field(ge=0, strict=True)
    fallback_used: bool = Field(strict=True)
    reason_code: str | None = None
    attempts: list[System1Attempt] = Field(default_factory=list)
    alternatives: list[System1Alternative]

    @model_validator(mode="after")
    def accepted_has_decision(self) -> System1Judgment:
        if self.accepted and (self.decision is None or self.confidence is None or not self.provider):
            raise ValueError("juicio aceptado sin decisión, confianza o proveedor")
        if not self.accepted and (self.decision is not None or self.confidence is not None):
            raise ValueError("juicio rechazado con decisión o confianza")
        if self.accepted:
            labels = [item.value for item in self.alternatives]
            if len(labels) != len(set(labels)) or set(labels) != set(SEMANTIC_OPTIONS) - {self.decision}:
                raise ValueError("alternativas incompletas o repetidas")
            if any(item.confidence > (self.confidence or 0.0) for item in self.alternatives):
                raise ValueError("la decisión no tiene la mayor confianza")
        return self


class SemanticAssessment(BaseModel):
    case_id: str
    classification: SemanticLabel = "uncertain"
    source: Literal["deterministic", "system1", "fallback"] = "fallback"
    accepted: bool = False
    confidence: float | None = None
    confidence_is_calibrated: bool = False
    provider: str | None = None
    model: str | None = None
    latency_ms: float = 0.0
    broker_latency_ms: float | None = None
    fallback_used: bool = False
    reason_code: str | None = None
    attempts: list[System1Attempt] = Field(default_factory=list)
    alternatives: list[System1Alternative] = Field(default_factory=list)
    priority: Literal["normal", "review", "high"] = "review"
    # Recomendación del filtro y acción efectiva se guardan separadas en sombra.
    proposed_deep_analysis: bool = True
    deep_analysis_required: bool = True
    deep_analysis_executed: bool = False
    deep_verdict: str | None = None
    deep_judgment: dict[str, object] | None = None
    deep_reason_code: str | None = None
    deterministic_checks: list[dict[str, object]] = Field(default_factory=list)


class SemanticMetrics(BaseModel):
    pairs_evaluated: int = 0
    deterministic_pairs: int = 0
    system1_pairs: int = 0
    regression_candidates: int = 0
    deep_analysis_required: int = 0
    deep_analysis_executed: int = 0
    deep_analysis_skipped: int = 0
    shadow_disagreements: int = 0
    fallbacks: int = 0
    latency_ms: float = 0.0
    providers: dict[str, int] = Field(default_factory=dict)
    # No se confunden métricas no medidas con cero errores.
    false_positives: int | None = None
    false_negatives: int | None = None


class SemanticReview(BaseModel):
    config: System1Config
    assessments: list[SemanticAssessment] = Field(default_factory=list)
    metrics: SemanticMetrics = Field(default_factory=SemanticMetrics)
