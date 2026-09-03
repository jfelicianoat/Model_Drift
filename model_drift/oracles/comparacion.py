"""Aserciones de comparación literal: exacta, contiene y expresión regular."""
from __future__ import annotations

import re
from typing import Literal

from pydantic import Field

from model_drift.oracles.base import BaseAssertion
from model_drift.oracles.tipos import AssertionResult, EvaluationInput, EvaluationServices
from model_drift.oracles.utilidades import normalize_text


class ExactAssertion(BaseAssertion):
    """Coincidencia exacta o normalizada con un valor esperado."""

    type: Literal["exact"] = "exact"
    value: str
    normalize: bool = True

    def _evaluate(self, data: EvaluationInput, services: EvaluationServices) -> AssertionResult:
        actual, expected = data.text, self.value
        if self.normalize:
            actual, expected = normalize_text(actual), normalize_text(expected)
        if actual == expected:
            return self._result(True)
        return self._result(False, f"esperado {expected!r}, obtenido {actual[:160]!r}")


class ContainsAssertion(BaseAssertion):
    """La salida contiene (o no) todos los fragmentos indicados."""

    type: Literal["contains"] = "contains"
    values: list[str] = Field(min_length=1)
    should_contain: bool = True
    case_sensitive: bool = False

    def _evaluate(self, data: EvaluationInput, services: EvaluationServices) -> AssertionResult:
        haystack = data.text if self.case_sensitive else data.text.casefold()
        faltan: list[str] = []
        sobran: list[str] = []
        for value in self.values:
            needle = value if self.case_sensitive else value.casefold()
            present = needle in haystack
            if self.should_contain and not present:
                faltan.append(value)
            elif not self.should_contain and present:
                sobran.append(value)
        if faltan:
            return self._result(False, f"no aparecen: {', '.join(faltan)}")
        if sobran:
            return self._result(False, f"aparecen y no debían: {', '.join(sobran)}")
        return self._result(True)


class RegexAssertion(BaseAssertion):
    """La salida casa (o no) con un patrón."""

    type: Literal["regex"] = "regex"
    pattern: str
    should_match: bool = True
    flags_ignorecase: bool = True

    def _evaluate(self, data: EvaluationInput, services: EvaluationServices) -> AssertionResult:
        flags = re.IGNORECASE | re.DOTALL if self.flags_ignorecase else re.DOTALL
        found = re.search(self.pattern, data.text, flags) is not None
        if found == self.should_match:
            return self._result(True)
        verbo = "no casa con" if self.should_match else "casa indebidamente con"
        return self._result(False, f"la salida {verbo} {self.pattern!r}")
