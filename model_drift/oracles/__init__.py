"""Oráculos deterministas: la familia A del plan, y la evidencia más barata.

Un oráculo responde `passed: bool` sin opinar y sin coste de invocación. Es la
primera escalón de la escalera del plan §8.1: si un caso tiene esquema, regex,
tests ejecutables o cotejo de citas, **no se invoca a ningún juez**.

Dos reglas que el módulo impone por construcción:

1. Una aserción que no se puede evaluar devuelve `not_applicable`, nunca
   `passed=False`. Un modelo sin herramientas no suspende una prueba de
   herramientas: queda fuera de esa métrica.
2. Una aserción de similitud semántica **exige umbral con procedencia de
   calibración**. Un umbral elegido a ojo convierte el instrumento de medida en
   una opinión disfrazada de número (plan §12), así que el modelo lo rechaza al
   construirse, no al ejecutarse.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from model_drift.oracles.base import BaseAssertion
from model_drift.oracles.comparacion import ContainsAssertion, ExactAssertion, RegexAssertion
from model_drift.oracles.estructura import (
    FieldEqualsAssertion,
    JsonSchemaAssertion,
    JsonValidAssertion,
    LengthAssertion,
    NumericCloseAssertion,
)
from model_drift.oracles.fundamentacion import AbstentionAssertion, CitationsGroundedAssertion
from model_drift.oracles.herramientas import ToolCallAssertion
from model_drift.oracles.semantica import SemanticSimilarityAssertion
from model_drift.oracles.sql import SqlResultAssertion
from model_drift.oracles.tipos import (
    AssertionResult,
    Embedder,
    EvaluationInput,
    EvaluationServices,
    ToolCall,
)
from model_drift.oracles.utilidades import (
    cosine_similarity,
    extract_json,
    normalize_text,
    resolve_path,
    strip_code_fences,
)

__all__ = [
    "AbstentionAssertion",
    "Assertion",
    "AssertionResult",
    "BaseAssertion",
    "CitationsGroundedAssertion",
    "ContainsAssertion",
    "Embedder",
    "EvaluationInput",
    "EvaluationServices",
    "ExactAssertion",
    "FieldEqualsAssertion",
    "JsonSchemaAssertion",
    "JsonValidAssertion",
    "LengthAssertion",
    "NumericCloseAssertion",
    "RegexAssertion",
    "SemanticSimilarityAssertion",
    "SqlResultAssertion",
    "ToolCall",
    "ToolCallAssertion",
    "aggregate_pass",
    "cosine_similarity",
    "evaluate_assertions",
    "extract_json",
    "normalize_text",
    "resolve_path",
    "strip_code_fences",
]


Assertion = Annotated[
    ExactAssertion
    | ContainsAssertion
    | RegexAssertion
    | JsonValidAssertion
    | JsonSchemaAssertion
    | FieldEqualsAssertion
    | NumericCloseAssertion
    | LengthAssertion
    | ToolCallAssertion
    | CitationsGroundedAssertion
    | AbstentionAssertion
    | SqlResultAssertion
    | SemanticSimilarityAssertion,
    Field(discriminator="type"),
]


def evaluate_assertions(
    assertions: list[Assertion],
    data: EvaluationInput,
    services: EvaluationServices,
) -> list[AssertionResult]:
    return [assertion.evaluate(data, services) for assertion in assertions]


def aggregate_pass(results: list[AssertionResult]) -> tuple[bool | None, str]:
    """Reduce las aserciones de una repetición a un único acierto.

    Devuelve `None` si ninguna aserción era aplicable: eso no es un suspenso,
    es ausencia de medición, y quien llama debe tratarlo como `not_applicable`.
    """
    aplicables = [result for result in results if result.counts]
    if not aplicables:
        return None, "ninguna aserción aplicable"
    fallidas = [result for result in aplicables if not result.passed]
    if fallidas:
        motivo = "; ".join(f"{r.assertion_id}: {r.reason}" for r in fallidas[:3])
        return False, motivo
    return True, ""
