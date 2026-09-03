"""Aserción de similitud semántica, con umbral que exige calibración."""
from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from model_drift.oracles.base import BaseAssertion
from model_drift.oracles.tipos import AssertionResult, EvaluationInput, EvaluationServices
from model_drift.oracles.utilidades import cosine_similarity


class SemanticSimilarityAssertion(BaseAssertion):
    """Similitud semántica contra una respuesta de referencia.

    El umbral **no puede elegirse a ojo**: el modelo de embeddings es un
    instrumento de medida (plan §12) y un umbral sin procedencia convierte la
    medición en una opinión. El validador rechaza la aserción al construirse,
    para que el error salga al cargar la suite y no a mitad de una tirada de
    tres horas.
    """

    type: Literal["semantic_similarity"] = "semantic_similarity"
    reference: str
    threshold: float = Field(ge=0.0, le=1.0)
    #: Referencia al fichero que documenta cómo se obtuvo el umbral.
    calibration: str

    @model_validator(mode="after")
    def calibration_required(self) -> SemanticSimilarityAssertion:
        if not self.calibration.strip():
            raise ValueError(
                "semantic_similarity exige 'calibration': un umbral de similitud "
                "sin procedencia documentada no es un oráculo, es una opinión "
                "con decimales (plan §12)"
            )
        return self

    def _evaluate(self, data: EvaluationInput, services: EvaluationServices) -> AssertionResult:
        if services.embedder is None:
            return self._skip("no hay instrumento de embeddings configurado")
        vectors = services.embedder.embed([data.text, self.reference])
        if len(vectors) != 2:
            return self._skip("el instrumento de embeddings no devolvió dos vectores")
        score = cosine_similarity(vectors[0], vectors[1])
        services.used_thresholds[self.assertion_id()] = self.threshold
        if score >= self.threshold:
            return self._result(True, f"similitud {score:.3f} ≥ {self.threshold}", score=score)
        return self._result(False, f"similitud {score:.3f} < {self.threshold}", score=score)
