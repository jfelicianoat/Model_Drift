"""Casos y suites: qué se prueba y con qué se compara.

Un caso **no es un prompt** (plan §7.2). Tiene que poder representar
conversación, contexto, herramientas, entorno de ejecución y aserciones, y aun
así seguir siendo trivial de escribir para el caso simple: cuatro líneas —`id`,
un mensaje `user` y una aserción— bastan.

Dos procedencias con reglas distintas:

- **redactado** — escrito a mano, con aserciones. Da evidencia de nivel 2 desde
  el primer día.
- **capturado** — derivado de una invocación real del Broker. Su salida
  histórica es **línea base de cambio, no verdad**: sin aserciones, relación
  metamórfica o etiqueta humana añadidas en la promoción, solo alcanza el nivel
  1. `provides_level_2_evidence` es quien lo decide, y el motor de comparación
  lo respeta.
"""
from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from model_drift.oracles import Assertion


class Provenance(str, Enum):
    redactado = "redactado"
    capturado = "capturado"


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class CaseContext(BaseModel):
    """Documentos y fuentes que el caso pone delante del modelo."""

    documents: list[str] = Field(default_factory=list)
    #: URLs o identificadores que el modelo puede citar legítimamente. Base del
    #: cotejo de citas.
    sources: list[str] = Field(default_factory=list)


class CaseParams(BaseModel):
    """Parámetros **solicitados**. Los efectivos los devuelve el Broker (§10)."""

    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    max_output_tokens: int | None = None


class CaseEnvironment(BaseModel):
    """Lo que hace falta montar para poder ejecutar el caso."""

    tools: list[str] = Field(default_factory=list)
    tool_server: Literal["ninguno", "simulado", "mcp"] = "ninguno"
    programmed_responses: str | None = None
    test_db: str | None = None
    code_tests: str | None = None


class Case(BaseModel):
    """Un caso de prueba completo."""

    id: str = Field(min_length=1)
    provenance: Provenance = Provenance.redactado
    tags: list[str] = Field(default_factory=list)
    description: str = ""

    messages: list[Message] = Field(default_factory=list)
    context: CaseContext = Field(default_factory=CaseContext)
    params: CaseParams = Field(default_factory=CaseParams)

    required_capabilities: list[str] = Field(default_factory=list)
    environment: CaseEnvironment = Field(default_factory=CaseEnvironment)

    assertions: list[Assertion] = Field(default_factory=list)
    metamorphic_relations: list[str] = Field(default_factory=list)

    #: Salida histórica de un caso capturado. Línea base de cambio, NUNCA
    #: verdad de referencia (plan §5). El nombre es deliberadamente feo para
    #: que nadie lo confunda con `expected`.
    baseline_output: str | None = None
    #: Etiqueta humana sobre la salida histórica, si alguien la revisó en la
    #: promoción. Es lo que convierte una línea base en evidencia.
    human_label: Literal["correcta", "incorrecta", "ambigua"] | None = None

    @model_validator(mode="after")
    def require_input(self) -> Case:
        if not self.messages:
            raise ValueError(f"el caso {self.id!r} no tiene ningún mensaje de entrada")
        if not any(message.role == "user" for message in self.messages):
            raise ValueError(f"el caso {self.id!r} no tiene ningún mensaje 'user'")
        return self

    @property
    def user_prompt(self) -> str:
        """Último mensaje de usuario: lo que se envía como prompt al Broker."""
        for message in reversed(self.messages):
            if message.role == "user":
                return message.content
        return ""

    @property
    def system_prompt(self) -> str | None:
        for message in self.messages:
            if message.role == "system":
                return message.content
        return None

    @property
    def is_multi_turn(self) -> bool:
        return sum(1 for message in self.messages if message.role in {"user", "assistant"}) > 1

    def provides_level_2_evidence(self) -> bool:
        """Si este caso puede sostener un veredicto de dirección por sí mismo.

        Un caso capturado sin nada añadido en la promoción **no puede**: su
        salida histórica no es verdad. Devolver False aquí es lo que impide que
        el motor lo etiquete como regresión.
        """
        if self.assertions or self.metamorphic_relations:
            return True
        return self.human_label is not None

    def canonical(self) -> dict[str, Any]:
        """Forma canónica para el hash de la suite."""
        return json.loads(self.model_dump_json(exclude_none=True))


class UnstableCasePolicy(BaseModel):
    """Cuándo un caso deja de ser un detector válido de deriva (plan §9.2)."""

    #: Acuerdo mínimo entre repeticiones para que el caso cuente. Con 1.0 solo
    #: valen los casos perfectamente estables; el defecto deja pasar algo de
    #: ruido sin dejar de excluir lo que es puro azar.
    min_stability: float = Field(default=0.8, ge=0.0, le=1.0)


class Suite(BaseModel):
    """Conjunto versionado de casos con sus reglas de agregación."""

    id: str = Field(min_length=1)
    version: str = "1"
    description: str = ""

    #: Cómo se binariza un caso a partir de sus N repeticiones. `todas` es el
    #: defecto para oráculos deterministas: un caso que falla 3 de 10 veces no
    #: está "aprobado". `mayoria` es más permisivo y debe declararse a
    #: conciencia, porque pierde información.
    case_pass_rule: Literal["todas", "mayoria"] = "todas"
    unstable: UnstableCasePolicy = Field(default_factory=UnstableCasePolicy)

    cases: list[Case] = Field(default_factory=list)

    #: Ruta raíz de la suite, para resolver `fixture://`. No entra en el hash.
    root: Path | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def unique_ids(self) -> Suite:
        vistos: set[str] = set()
        for case in self.cases:
            if case.id in vistos:
                raise ValueError(f"identificador de caso duplicado en la suite {self.id!r}: {case.id!r}")
            vistos.add(case.id)
        return self

    def case_by_id(self, case_id: str) -> Case | None:
        return next((case for case in self.cases if case.id == case_id), None)

    def hash(self) -> str:
        """Hash canónico de la suite. Va al manifiesto de cada tirada."""
        payload = {
            "id": self.id,
            "version": self.version,
            "case_pass_rule": self.case_pass_rule,
            "min_stability": self.unstable.min_stability,
            "cases": [case.canonical() for case in sorted(self.cases, key=lambda c: c.id)],
        }
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def short_hash(self) -> str:
        return self.hash()[:12]


class SuiteLoadError(RuntimeError):
    """La suite no se puede cargar: fichero ausente, YAML roto, caso inválido."""


def _coerce_case(raw: Any, source: Path) -> Case:
    if not isinstance(raw, dict):
        raise SuiteLoadError(f"{source}: cada caso debe ser un mapa, no {type(raw).__name__}")
    # Azúcar sintáctico del caso simple: `prompt: "..."` equivale a un único
    # mensaje de usuario. Sin esto, escribir veinte casos triviales obliga a
    # veinte bloques `messages` de tres líneas.
    if "prompt" in raw and "messages" not in raw:
        prompt = raw.pop("prompt")
        mensajes: list[dict[str, str]] = []
        if raw.get("system"):
            mensajes.append({"role": "system", "content": str(raw.pop("system"))})
        mensajes.append({"role": "user", "content": str(prompt)})
        raw["messages"] = mensajes
    raw.pop("system", None)
    try:
        return Case.model_validate(raw)
    except Exception as error:  # pydantic.ValidationError y derivados
        raise SuiteLoadError(f"{source}: caso inválido: {error}") from error


def load_suite(path: str | Path) -> Suite:
    """Carga una suite desde un fichero YAML o desde un directorio.

    En modo directorio se espera `suite.yaml` con la configuración y los casos
    en `casos/*.yaml` (uno o varios por fichero). Es lo que permite que una
    suite crezca a cientos de casos sin convertirse en un YAML de 6.000 líneas.
    """
    root = Path(path)
    if root.is_dir():
        config_path = root / "suite.yaml"
        cases_dir = root / "casos"
    else:
        config_path = root
        cases_dir = root.parent / "casos"
        root = root.parent

    if not config_path.is_file():
        raise SuiteLoadError(f"no existe el fichero de suite: {config_path}")

    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as error:
        raise SuiteLoadError(f"{config_path}: YAML inválido: {error}") from error
    if not isinstance(config, dict):
        raise SuiteLoadError(f"{config_path}: la raíz debe ser un mapa")

    raw_cases: list[Any] = list(config.pop("cases", []) or [])
    fuentes: list[tuple[Any, Path]] = [(item, config_path) for item in raw_cases]

    if cases_dir.is_dir():
        for case_file in sorted(cases_dir.glob("*.y*ml")):
            try:
                content = yaml.safe_load(case_file.read_text(encoding="utf-8"))
            except yaml.YAMLError as error:
                raise SuiteLoadError(f"{case_file}: YAML inválido: {error}") from error
            if content is None:
                continue
            if isinstance(content, list):
                fuentes.extend((item, case_file) for item in content)
            else:
                fuentes.append((content, case_file))

    config["cases"] = [_coerce_case(dict(item), source) for item, source in fuentes]
    config["root"] = root
    try:
        return Suite.model_validate(config)
    except SuiteLoadError:
        raise
    except Exception as error:
        raise SuiteLoadError(f"{config_path}: suite inválida: {error}") from error
