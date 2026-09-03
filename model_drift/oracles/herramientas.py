"""Aserción sobre las llamadas a herramientas que emite el modelo.

Un modelo sin herramientas no suspende una prueba de herramientas: queda
fuera de la métrica. Es la regla 1 del módulo, aplicada aquí.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from model_drift.oracles.base import BaseAssertion
from model_drift.oracles.tipos import AssertionResult, EvaluationInput, EvaluationServices


class ToolCallAssertion(BaseAssertion):
    """El modelo llamó a la herramienta correcta con los argumentos correctos.

    Requiere que el modelo declare la capacidad `tools`; si no la tiene, el
    caso entero es `not_applicable` y esta aserción no llega a ejecutarse.
    """

    type: Literal["tool_call"] = "tool_call"
    name: str
    #: Subconjunto de argumentos que deben aparecer con ese valor exacto. No se
    #: exige igualdad total: un argumento opcional añadido por el modelo no es
    #: un fallo salvo que la suite lo diga.
    arguments: dict[str, Any] = Field(default_factory=dict)
    should_call: bool = True

    def _evaluate(self, data: EvaluationInput, services: EvaluationServices) -> AssertionResult:
        llamadas = [call for call in data.tool_calls if call.name == self.name]
        if not self.should_call:
            if llamadas:
                return self._result(False, f"llamó a {self.name} y no debía")
            return self._result(True)
        if not llamadas:
            nombres = ", ".join(sorted({c.name for c in data.tool_calls})) or "ninguna"
            return self._result(False, f"no llamó a {self.name} (llamó a: {nombres})")
        for call in llamadas:
            faltan = {
                key: (value, call.arguments.get(key))
                for key, value in self.arguments.items()
                if call.arguments.get(key) != value
            }
            if not faltan:
                return self._result(True)
        detalle = ", ".join(f"{k}: esperado {v[0]!r}, obtenido {v[1]!r}" for k, v in faltan.items())
        return self._result(False, f"argumentos incorrectos en {self.name} ({detalle})")
