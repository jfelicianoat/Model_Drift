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

import json
import math
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Protocol

import jsonschema
from pydantic import BaseModel, Field, model_validator

from model_drift.fixtures import FixtureError, FixtureResolver

# --------------------------------------------------------------------------
# Entrada de la evaluación
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# Utilidades compartidas
# --------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_URL = re.compile(r"https?://[^\s<>\"'`\]\)}]+", re.IGNORECASE)
_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")
_SQL_FENCE = re.compile(r"```(?:sql|SQL)?\s*(.*?)```", re.DOTALL)


def normalize_text(value: str) -> str:
    """Normalización conservadora: espacios y mayúsculas, nada más.

    No se tocan tildes ni puntuación: en español eso cambiaría el significado
    de lo que se compara y convertiría un oráculo en una heurística.
    """
    return re.sub(r"\s+", " ", value).strip().casefold()


def strip_code_fences(text: str) -> str:
    """Devuelve el contenido de la primera valla de código, o el texto entero.

    Las vallas son la deriva de formato más común entre versiones: el modelo
    empieza a envolver el JSON que antes devolvía desnudo. Se toleran aquí para
    que la aserción de esquema mida el esquema, y se detecta aparte con la
    taxonomía del cambio.
    """
    match = _FENCE.search(text)
    return match.group(1).strip() if match else text.strip()


def extract_json(text: str) -> tuple[Any | None, str]:
    """Intenta obtener un objeto JSON del texto. Devuelve (valor, motivo)."""
    candidate = strip_code_fences(text)
    try:
        return json.loads(candidate), ""
    except json.JSONDecodeError:
        pass
    # Segundo intento: el primer bloque {...} o [...] equilibrado.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        end = candidate.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1]), ""
            except json.JSONDecodeError:
                continue
    return None, "la salida no contiene JSON parseable"


_PATH_TOKEN = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def resolve_path(data: Any, path: str) -> tuple[Any, bool]:
    """Resuelve una ruta tipo `$.factura.lineas[0].total`.

    Subconjunto deliberado de JSONPath: sin comodines ni filtros. Una ruta que
    nadie puede leer de un vistazo no sirve como aserción de un TFM.
    """
    cleaned = path.lstrip("$").lstrip(".")
    current = data
    for token in _PATH_TOKEN.finditer(cleaned):
        key, index = token.group(1), token.group(2)
        if index is not None:
            if not isinstance(current, list) or int(index) >= len(current):
                return None, False
            current = current[int(index)]
            continue
        if not isinstance(current, dict) or key not in current:
            return None, False
        current = current[key]
    return current, True


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)


# --------------------------------------------------------------------------
# Aserciones
# --------------------------------------------------------------------------


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


class CitationsGroundedAssertion(BaseAssertion):
    """Las URLs citadas salieron de lo que el modelo realmente tuvo delante.

    Cotejo determinista, en la misma línea que `app/evidence.py` del Broker:
    comparar URLs no necesita otro modelo opinando y no puede equivocarse a
    favor de nadie. No juzga si la respuesta es correcta, solo si sus enlaces
    tienen procedencia.
    """

    type: Literal["citations_grounded"] = "citations_grounded"
    #: Si es False, basta con que no invente; no se exige que cite.
    require_at_least_one: bool = False

    def _evaluate(self, data: EvaluationInput, services: EvaluationServices) -> AssertionResult:
        citadas = {url.rstrip(".,;:)") for url in _URL.findall(data.text)}
        disponibles = {url.rstrip(".,;:)") for url in data.sources}
        if not citadas:
            if self.require_at_least_one:
                return self._result(False, "no cita ninguna fuente")
            return self._result(True, "no cita fuentes")
        inventadas = sorted(citadas - disponibles)
        if inventadas:
            return self._result(False, f"cita fuentes que no consultó: {', '.join(inventadas[:3])}")
        return self._result(True, f"{len(citadas)} citas con procedencia")


class AbstentionAssertion(BaseAssertion):
    """El modelo se abstiene cuando la respuesta no está en el contexto.

    Métrica central para el Knowledge_Orchestrator: la alternativa a abstenerse
    es inventar.
    """

    type: Literal["abstention"] = "abstention"
    should_abstain: bool = True
    phrases: list[str] = Field(
        default_factory=lambda: [
            "no consta",
            "no aparece",
            "no se menciona",
            "no dispongo",
            "no puedo determinar",
            "no está en el contexto",
            "no hay información",
        ]
    )

    def _evaluate(self, data: EvaluationInput, services: EvaluationServices) -> AssertionResult:
        texto = data.text.casefold()
        abstiene = any(phrase.casefold() in texto for phrase in self.phrases)
        if abstiene == self.should_abstain:
            return self._result(True)
        if self.should_abstain:
            return self._result(False, "no se abstuvo pese a no tener la información")
        return self._result(False, "se abstuvo teniendo la información disponible")


class SqlResultAssertion(BaseAssertion):
    """Ejecuta el SQL generado contra una BD fixture y compara conjuntos.

    Evalúa **semántica, no cadenas**: dos consultas escritas de forma distinta
    que devuelven lo mismo son ambas correctas, y la comparación textual las
    habría suspendido.
    """

    type: Literal["sql_result"] = "sql_result"
    database: str
    expected: Any
    #: Por defecto el orden no importa: solo lo exige un ORDER BY explícito en
    #: el enunciado, y eso lo declara la suite.
    ordered: bool = False

    def _evaluate(self, data: EvaluationInput, services: EvaluationServices) -> AssertionResult:
        path = services.resolver.path_for(self.database)
        if not path.is_file():
            return self._skip(f"BD de prueba no encontrada: {self.database}")
        match = _SQL_FENCE.search(data.text)
        query = (match.group(1) if match else data.text).strip().rstrip(";")
        if not query:
            return self._result(False, "la salida no contiene ninguna consulta")
        esperado = services.resolver.resolve_maybe(self.expected)
        # Solo lectura: una consulta generada por un modelo no toca el fixture.
        uri = f"file:{path.as_posix()}?mode=ro"
        try:
            with sqlite3.connect(uri, uri=True) as connection:
                filas = connection.execute(query).fetchall()
        except sqlite3.Error as error:
            return self._result(False, f"la consulta no se pudo ejecutar: {error}")
        obtenido = [list(fila) for fila in filas]
        referencia = [list(fila) for fila in esperado]
        if not self.ordered:
            obtenido_cmp = sorted(map(repr, obtenido))
            referencia_cmp = sorted(map(repr, referencia))
            igual = obtenido_cmp == referencia_cmp
        else:
            igual = obtenido == referencia
        if igual:
            return self._result(True, f"{len(obtenido)} filas coincidentes")
        return self._result(False, f"devolvió {len(obtenido)} filas; se esperaban {len(referencia)}")


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
