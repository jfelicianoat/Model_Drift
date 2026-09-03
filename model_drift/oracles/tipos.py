"""Lo que un oráculo mira y lo que devuelve.

Estos tipos son el contrato con el resto del motor: el `Runner` los
construye, los oráculos los consumen y la comparación los agrega. No
dependen de ninguna aserción concreta, y por eso están en su propio módulo:
cualquiera puede importarlos sin arrastrar el catálogo entero.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, Field

from model_drift.fixtures import FixtureResolver


class ToolCall(BaseModel):
    """Llamada a herramienta emitida por el modelo."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class EvaluationInput(BaseModel):
    """Todo lo que un oráculo puede mirar de una invocación."""

    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    #: Fuentes que el modelo tuvo delante (URLs de contexto, resultados de
    #: herramientas). Base del cotejo de citas.
    sources: list[str] = Field(default_factory=list)
    finish_reason: str | None = None
    #: False cuando la invocación no produjo salida evaluable: error, timeout,
    #: truncamiento. Distinto de "produjo una salida mala".
    produced_output: bool = True


class Embedder(Protocol):
    """Instrumento de similitud. Se inyecta; nunca se instancia aquí."""

    @property
    def identity(self) -> str:
        """Identificación del modelo de embeddings, para el manifiesto."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass
class EvaluationServices:
    """Dependencias que los oráculos necesitan del exterior."""

    resolver: FixtureResolver
    embedder: Embedder | None = None
    #: Ejecutar código generado por un modelo es una decisión del operador, no
    #: un detalle de implementación: por defecto está apagado y las aserciones
    #: que lo necesiten quedan `not_applicable`.
    allow_code_execution: bool = False
    used_thresholds: dict[str, float] = field(default_factory=dict)


class AssertionResult(BaseModel):
    """Resultado de una aserción sobre una única invocación."""

    assertion_id: str
    type: str
    passed: bool = False
    not_applicable: bool = False
    reason: str = ""
    #: Valor continuo cuando la aserción produce uno (similitud, distancia).
    score: float | None = None

    @property
    def counts(self) -> bool:
        """Si esta aserción entra en el cómputo del caso."""
        return not self.not_applicable
