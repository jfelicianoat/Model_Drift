"""Aserciones sobre la estructura de la salida: JSON, esquema, campos y tamaño."""
from __future__ import annotations

from typing import Any, Literal

import jsonschema

from model_drift.oracles.base import BaseAssertion
from model_drift.oracles.tipos import AssertionResult, EvaluationInput, EvaluationServices
from model_drift.oracles.utilidades import _NUMBER, extract_json, resolve_path


class JsonValidAssertion(BaseAssertion):
    """La salida es JSON parseable."""

    type: Literal["json_valid"] = "json_valid"

    def _evaluate(self, data: EvaluationInput, services: EvaluationServices) -> AssertionResult:
        value, reason = extract_json(data.text)
        return self._result(value is not None, reason)


class JsonSchemaAssertion(BaseAssertion):
    """La salida valida contra un esquema JSON.

    La deriva de formato es el vector más frecuente entre versiones y este es
    el oráculo que la caza: claves renombradas, campos nuevos, tipos cambiados.
    """

    type: Literal["json_schema"] = "json_schema"
    #: Esquema en línea o referencia `fixture://`.
    schema_ref: Any

    def _evaluate(self, data: EvaluationInput, services: EvaluationServices) -> AssertionResult:
        schema = services.resolver.resolve_maybe(self.schema_ref)
        value, reason = extract_json(data.text)
        if value is None:
            return self._result(False, reason)
        try:
            jsonschema.validate(instance=value, schema=schema)
        except jsonschema.ValidationError as error:
            ruta = "/".join(str(part) for part in error.absolute_path) or "(raíz)"
            return self._result(False, f"no valida en {ruta}: {error.message}")
        except jsonschema.SchemaError as error:
            return self._skip(f"el esquema es inválido: {error.message}")
        return self._result(True)


class FieldEqualsAssertion(BaseAssertion):
    """Un campo concreto del JSON de salida tiene el valor esperado."""

    type: Literal["field_equals"] = "field_equals"
    path: str
    value: Any
    #: Tolerancia absoluta para comparaciones numéricas.
    tolerance: float | None = None

    def _evaluate(self, data: EvaluationInput, services: EvaluationServices) -> AssertionResult:
        payload, reason = extract_json(data.text)
        if payload is None:
            return self._result(False, reason)
        actual, found = resolve_path(payload, self.path)
        if not found:
            return self._result(False, f"la ruta {self.path} no existe en la salida")
        if self.tolerance is not None and isinstance(actual, (int, float)) and isinstance(self.value, (int, float)):
            delta = abs(float(actual) - float(self.value))
            if delta <= self.tolerance:
                return self._result(True, score=delta)
            return self._result(False, f"{self.path}={actual}, esperado {self.value}±{self.tolerance}", score=delta)
        if actual == self.value:
            return self._result(True)
        return self._result(False, f"{self.path}={actual!r}, esperado {self.value!r}")


class NumericCloseAssertion(BaseAssertion):
    """El primer (o último) número de la salida está cerca del esperado."""

    type: Literal["numeric_close"] = "numeric_close"
    expected: float
    tolerance: float = 0.0
    pick: Literal["first", "last"] = "last"

    def _evaluate(self, data: EvaluationInput, services: EvaluationServices) -> AssertionResult:
        matches = _NUMBER.findall(data.text)
        if not matches:
            return self._result(False, "la salida no contiene ningún número")
        raw = matches[0] if self.pick == "first" else matches[-1]
        actual = float(raw.replace(",", "."))
        delta = abs(actual - self.expected)
        if delta <= self.tolerance:
            return self._result(True, score=delta)
        return self._result(False, f"obtenido {actual}, esperado {self.expected}±{self.tolerance}", score=delta)


class LengthAssertion(BaseAssertion):
    """Restricciones de longitud. Caza la deriva de verbosidad."""

    type: Literal["length"] = "length"
    max_chars: int | None = None
    min_chars: int | None = None
    max_words: int | None = None
    min_words: int | None = None

    def _evaluate(self, data: EvaluationInput, services: EvaluationServices) -> AssertionResult:
        chars = len(data.text.strip())
        words = len(data.text.split())
        problemas: list[str] = []
        if self.max_chars is not None and chars > self.max_chars:
            problemas.append(f"{chars} caracteres > {self.max_chars}")
        if self.min_chars is not None and chars < self.min_chars:
            problemas.append(f"{chars} caracteres < {self.min_chars}")
        if self.max_words is not None and words > self.max_words:
            problemas.append(f"{words} palabras > {self.max_words}")
        if self.min_words is not None and words < self.min_words:
            problemas.append(f"{words} palabras < {self.min_words}")
        if problemas:
            return self._result(False, "; ".join(problemas), score=float(chars))
        return self._result(True, score=float(chars))
