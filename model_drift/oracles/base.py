"""Contrato común de las aserciones.

`evaluate` nunca lanza: convierte cualquier fallo en un resultado
`not_applicable`. Una aserción rota deja un hueco medido en la tirada, no
un suspenso inventado ni una tirada de tres horas por el suelo.
"""
from __future__ import annotations

from pydantic import BaseModel

from model_drift.fixtures import FixtureError
from model_drift.oracles.tipos import AssertionResult, EvaluationInput, EvaluationServices


class BaseAssertion(BaseModel):
    """Contrato común. `evaluate` nunca lanza: convierte el fallo en resultado."""

    id: str = ""
    #: Descripción opcional para el informe.
    description: str = ""

    def assertion_id(self) -> str:
        return self.id or f"{self.type}"  # type: ignore[attr-defined]

    def evaluate(self, data: EvaluationInput, services: EvaluationServices) -> AssertionResult:
        kind = getattr(self, "type", "desconocido")
        if not data.produced_output:
            return AssertionResult(
                assertion_id=self.assertion_id(),
                type=kind,
                not_applicable=True,
                reason="la invocación no produjo salida evaluable",
            )
        try:
            return self._evaluate(data, services)
        except FixtureError as error:
            return AssertionResult(
                assertion_id=self.assertion_id(),
                type=kind,
                not_applicable=True,
                reason=f"fixture no disponible: {error}",
            )
        except Exception as error:  # noqa: BLE001 - un oráculo roto no debe tumbar la tirada
            return AssertionResult(
                assertion_id=self.assertion_id(),
                type=kind,
                not_applicable=True,
                reason=f"error evaluando la aserción: {error}",
            )

    def _evaluate(self, data: EvaluationInput, services: EvaluationServices) -> AssertionResult:
        raise NotImplementedError

    def _result(self, passed: bool, reason: str = "", score: float | None = None) -> AssertionResult:
        return AssertionResult(
            assertion_id=self.assertion_id(),
            type=getattr(self, "type", "desconocido"),
            passed=passed,
            reason=reason,
            score=score,
        )

    def _skip(self, reason: str) -> AssertionResult:
        return AssertionResult(
            assertion_id=self.assertion_id(),
            type=getattr(self, "type", "desconocido"),
            not_applicable=True,
            reason=reason,
        )
