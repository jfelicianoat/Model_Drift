"""Plan versionado, referencias y notas en bruto de una calibración."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from model_drift.broker_contracts import ModelReference

Decision = bool | str | int | float


def label_key(value: Decision) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class EvaluationTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["ollama_system1", "laya_mcp"]
    model: str | None = Field(default=None, min_length=1)


class CalibrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    use_case: str = Field(pattern=r"^[A-Za-z0-9_-]{1,128}$")
    decision_type: Literal["binary", "choice", "score"]
    instructions: str = Field(min_length=1, max_length=4000)
    threshold_profile: str = Field(min_length=1, max_length=128)
    options: list[str] = Field(default_factory=list)
    criteria: dict[str, str] = Field(default_factory=dict)
    rubric: list[str] = Field(default_factory=list)
    learner: EvaluationTarget
    teacher_task: ModelReference | None = None
    teacher_judge: EvaluationTarget | None = None
    second_teacher_task: ModelReference | None = None
    teacher_repetitions: int = Field(default=5, ge=3)
    teacher_timeout_retries: int = Field(default=1, ge=0, le=10)
    teacher_warmup: bool = True
    teacher_timeout_seconds: int = Field(default=300, gt=0)
    min_teacher_kappa: float = Field(default=0.8, ge=0, le=1)
    min_human_cases: int = Field(default=50, ge=2)
    target_precision: float = Field(default=0.99, gt=0, lt=1)
    min_calibration_cases: int = Field(default=50, ge=1)
    min_test_cases: int = Field(default=50, ge=1)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    reliability_edges: list[float] = Field(default_factory=lambda: [0, 0.90, 0.95, 0.97, 0.99, 1.0])
    calibration_fraction: float = Field(default=0.6, gt=0, lt=1)
    seed: int = 1234
    probability_sum_tolerance: float = Field(default=0.02, ge=0, lt=0.5)
    max_invalid_score_ratio: float = Field(default=0.05, ge=0, le=1)
    auto_approve_values: list[Any] = Field(default_factory=list)
    cloud_allowed_teacher: bool = False
    cloud_allowed_learner: bool = False
    timeout_seconds: float = Field(default=75.0, gt=0)

    def domain(self) -> list[Decision]:
        if self.decision_type == "binary":
            return [False, True]
        return list(self.options) if self.decision_type == "choice" else list(range(len(self.rubric)))

    def normalize(self, value: Any) -> Decision:
        if self.decision_type == "binary" and type(value) is bool:
            return bool(value)
        if self.decision_type == "choice" and isinstance(value, str) and value in self.options:
            return value
        if (self.decision_type == "score" and type(value) in (int, float)
            and math.isfinite(value) and float(value).is_integer() and 0 <= value < len(self.rubric)):
            return int(value)
        raise ValueError("decisión fuera del tipo o dominio solicitado")

    def request(self, data: dict[str, Any], *, local_only: bool, target: EvaluationTarget | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "use_case": self.use_case, "decision_type": self.decision_type, "instructions": self.instructions,
            "threshold_profile": self.threshold_profile, "input": data,
            "cloud_allowed": self.cloud_allowed_learner and not local_only,
        }
        if self.decision_type == "choice":
            payload.update(options=self.options, criteria=self.criteria)
        if self.decision_type == "score":
            payload["rubric"] = self.rubric
        if target is not None:
            payload["target"] = target.model_dump(exclude_none=True)
        return payload

    @model_validator(mode="after")
    def valid_plan(self) -> CalibrationConfig:
        if (self.teacher_task is None) == (self.teacher_judge is None):
            raise ValueError("elige exactamente un maestro: teacher_task o teacher_judge")
        if self.decision_type == "choice":
            if not 2 <= len(self.options) <= 20 or len(set(self.options)) != len(self.options) or not all(self.options):
                raise ValueError("options debe contener de 2 a 20 etiquetas distintas")
            if self.criteria and set(self.criteria) != set(self.options):
                raise ValueError("criteria debe describir exactamente las opciones")
        elif self.options or self.criteria:
            raise ValueError("options/criteria solo se permiten en choice")
        if self.decision_type == "score":
            if not 2 <= len(self.rubric) <= 20 or not all(self.rubric):
                raise ValueError("rubric debe contener de 2 a 20 niveles")
        elif self.rubric:
            raise ValueError("rubric solo se permite en score")
        edges = self.reliability_edges
        if len(edges) < 2 or edges[0] != 0 or edges[-1] != 1 or any(a >= b for a, b in zip(edges, edges[1:])):
            raise ValueError("tramos de fiabilidad: ordenados, desde 0 hasta 1")
        for value in self.auto_approve_values:
            self.normalize(value)
        return self


class CalibrationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    input: dict[str, Any] = Field(min_length=1)
    provenance: Literal["redactado", "capturado"] = "redactado"
    data_class: Literal["public", "internal", "confidential", "local_only"] = "internal"
    human_label: Any = None
    human_verified: bool = False
    adversarial: bool = False

    @property
    def local_only(self) -> bool:
        return self.data_class in {"confidential", "local_only"}


class CalibrationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    config: CalibrationConfig
    cases: list[CalibrationCase] = Field(min_length=2)

    @model_validator(mode="after")
    def unique_cases(self) -> CalibrationPlan:
        if len({case.id for case in self.cases}) != len(self.cases):
            raise ValueError("casos duplicados")
        inputs = [hashlib.sha256(json.dumps(case.input, sort_keys=True).encode()).hexdigest() for case in self.cases]
        if len(set(inputs)) != len(inputs):
            raise ValueError("entradas duplicadas: provocarían fuga entre calibración y prueba")
        for case in self.cases:
            if case.human_verified:
                self.config.normalize(case.human_label)
        return self

    @classmethod
    def load(cls, path: str | Path) -> CalibrationPlan:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def signature(self) -> str:
        data = self.config.model_dump(mode="json")
        for key in ("teacher_task", "teacher_judge", "second_teacher_task", "learner"):
            data.pop(key)
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


class TeacherLabel(BaseModel):
    source: Literal["distilled_system2"] = "distilled_system2"
    decision: Any = None
    unanimous: bool = False
    decisions: list[Any] = Field(default_factory=list)
    reason_code: str | None = None
    latency_ms: float = 0
    cost_usd: float = 0
    tokens_input: int = 0
    tokens_output: int = 0
    timeouts: int = 0
    telemetry: list[dict[str, Any]] = Field(default_factory=list)


class LearnerNote(BaseModel):
    case_id: str
    partition: Literal["calibration", "test"]
    accepted: bool = False
    decision: Any = None
    confidence: float | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)
    provider: str | None = None
    model: str | None = None
    valid_score: bool = False
    invalid_score: bool = False
    reason_code: str | None = None
    latency_ms: float = 0
    tokens_input: int | None = None
    tokens_output: int | None = None
    reference: Any = None
    reference_eligible: bool = False
    adversarial: bool = False


class CalibrationReport(BaseModel):
    version: Literal[1] = 1
    run_id: str
    created_at: str
    use_case: str
    plan_signature: str
    dataset_hash: str
    config: CalibrationConfig
    status: Literal["calibrado", "no_calibrado", "indeterminado"] = "indeterminado"
    reasons: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    learner_provider: str | None = None
    learner_model: str | None = None
    learner_fingerprint: dict[str, Any] = Field(default_factory=dict)
    teacher_fingerprints: dict[str, dict[str, Any]] = Field(default_factory=dict)
    labels: dict[str, TeacherLabel] = Field(default_factory=dict)
    notes: list[LearnerNote] = Field(default_factory=list)
    teacher_kappa: float | None = None
    teacher_self_consistency: float | None = None
    teacher_qualified: bool = False
    ambiguous_cases: int = 0
    human_cases: int = 0
    warmup: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    recommended_threshold: float | None = None
    test_precision: float | None = None
    test_precision_lower: float | None = None
    test_precision_upper: float | None = None
    test_coverage: float | None = None
    previous_comparison: dict[str, Any] | None = None
