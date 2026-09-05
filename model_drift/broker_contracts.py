"""Espejo del contrato del AI Broker.

Copia deliberada, no importación: Model_Drift es un cliente HTTP y no comparte
proceso ni paquete con el Broker. La contrapartida es la regla que ya rige en el
Knowledge_Orchestrator — **cada cambio del contrato del Broker exige actualizar
este fichero en el mismo cambio**, porque un estado o una estrategia
desconocidos hacen que el sondeo mate la tarea por validación.

Estado verificado contra el Broker en marcha (contrato v2.10, 05-sep-2026),
**después** de que incorporase la entrega de `Evaluation_Contract.md`:

Lo que aporta el 2.10 y por qué le importa a una herramienta de medición:

- Cada invocación declara **`contractual`**: `true` si ejecuta el trabajo que se
  pidió, `false` si es trabajo propio del Broker. Bajo un mismo `task_id`
  conviven las dos. Model_Drift leía la **última** invocación para sacar los
  parámetros efectivos y la huella; si la última era un `shadow_probe`, medía
  otro modelo y no había forma de saberlo. Se filtra por este booleano, no por
  el nombre del rol: la lista de roles crece y una lista negra se rompe callada.
- **`prompt_compression: {requested, effective}`** por invocación: pedir que no
  se pode el prompt deja de ser una petición sin acuse de recibo. Una medición
  sobre un prompt podado no compara lo que dice comparar, así que se pide `off`
  y se comprueba que se aplicó.
- **`auxiliary_invocations: false`** en la petición: garantiza que solo el
  modelo que responde ve el contenido. Model_Drift ya lo tenía por garantía
  implícita (`target_model` + `fallback_allowed: false`), pero eso solo vale
  contra un Broker 2.10; declararlo es explícito y no depende de la versión.
- **`final: true`** marca el entregable en `/artifacts`: la vía canónica para
  cerrar una unidad de trabajo con su `sha256` sobre los bytes exactos.
- `role` y `status` vienen **enumerados**, así que se validan en vez de
  aceptarse como cadena libre.

- `generation` admite `temperature`, `max_output_tokens`, **`seed` y `top_p`**
  (Cambio 1). El cliente sigue degradando ante un 422 por si apunta a un Broker
  antiguo, pero contra este ya no hace falta.
- `TaskCreateRequest` admite **`exclude_from_model_learning`** (Cambio 4): el
  tráfico de medición ya no contamina el router adaptativo.
- El catálogo trae **`execution_fingerprint`** por modelo (Cambio 2), con
  `components: {nombre: {value, status}}` y `hash`. El vocabulario de `status`
  es exactamente `verificado | declarado | no_disponible`.
- `GET /api/v1/models/availability` devuelve `{checked_at, items[], counts}`.
  La lista va en **`items`**, no en `models`.
- `GET /api/v1/models/context` es **por modelo**: exige `provider`,
  `deployment` y `model` como parámetros de consulta. No devuelve una lista.
- `execution_summary` en el estado de tarea trae `served_by` y `fallback_used`
  **tipados**. `result` sigue teniendo los mismos datos por compatibilidad,
  pero el propio contrato dice que no tiene garantías: no se construye lógica
  sobre él si `execution_summary` está disponible.
- `GET /api/v1/tasks/{id}/invocations` devuelve `{task_id, items[]}` con los
  parámetros **efectivos** de cada llamada, su huella y si alimentó o no las
  métricas del router.
- `generation.seed_status` distingue `not_requested`, `sent` y `unsupported`.
  **El Broker no promete que la semilla se aplique**: solo declara si la envió.
  Una semilla aceptada y descartada por debajo es peor que no tenerla, así que
  la reproducibilidad se declara mirando esto, no lo que se pidió.
- `model_requirements.target_model` + `fallback_allowed` permiten fijar modelo.
- Las capacidades se anuncian en `/api/v1/capabilities`: `contract_version`,
  `exclude_from_model_learning`, `invocation_telemetry`, `execution_fingerprint`
  y, desde 2.10, `invocation_contract`, `prompt_compression_echo`,
  `canonical_artifacts`, `task_artifacts`, `auxiliary_invocations` y
  `auxiliary_invocations_optout`.

El contrato crece de forma aditiva y el propio documento pide no rechazar campos
desconocidos: todos los modelos de este fichero toleran claves que no conocen y
dan por defecto lo que falte.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Estados del Broker. Solo los terminales son contrato estable."""

    queued = "queued"
    waiting_for_memory = "waiting_for_memory"
    waiting_for_dependencies = "waiting_for_dependencies"
    waiting_for_tools = "waiting_for_tools"
    routing = "routing"
    planning = "planning"
    resource_planning = "resource_planning"
    chunking = "chunking"
    generating = "generating"
    proposing = "proposing"
    evaluating = "evaluating"
    debating = "debating"
    synthesizing = "synthesizing"
    verifying = "verifying"
    converting = "converting"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {TaskStatus.completed, TaskStatus.failed, TaskStatus.cancelled}


#: Estados no terminales que el Broker puede devolver y que no conocemos. No se
#: tratan como error: la documentación del Broker pide explícitamente no hacer
#: lógica sobre los estados intermedios concretos.
def is_terminal_status(raw: str) -> bool:
    try:
        return TaskStatus(raw).is_terminal
    except ValueError:
        return False


class InferenceKind(str, Enum):
    chat = "chat"
    embedding = "embedding"


class ModelReference(BaseModel):
    provider: str
    deployment: str
    model: str
    role: str | None = None
    required: bool = False

    @property
    def key(self) -> str:
        return f"{self.provider}/{self.deployment}/{self.model}"


class ModelCatalogEntry(BaseModel):
    """Entrada de `GET /api/v1/models/availability`.

    `features` viene de `/api/v1/models/context` y trae la matriz completa de
    capacidades con la procedencia de la evidencia. Aquí se guarda tal cual: es
    el Broker quien sabe si un `json_mode` está sondeado o inferido, y
    reinterpretarlo desde fuera sería inventar.
    """

    provider: str
    deployment: str
    model: str
    availability: str = "unknown"
    dispatchable: bool = False
    reason: str = ""
    provider_status: str = "unknown"
    model_status: str | None = None
    compatibility: str = "unknown"
    capabilities: list[str] = Field(default_factory=list)
    context_window: int | None = None
    compatibility_error: str | None = None
    features: dict[str, Any] = Field(default_factory=dict)
    #: Huella de ejecución tal y como la publica el Broker (Cambio 2):
    #: `{"components": {nombre: {"value": ..., "status": ...}}, "hash": ...}`.
    #: Se guarda cruda: es el Broker quien sabe qué pudo verificar y qué no, y
    #: reinterpretarlo desde fuera sería inventar.
    execution_fingerprint: dict[str, Any] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.provider}/{self.deployment}/{self.model}"

    @property
    def reference(self) -> ModelReference:
        return ModelReference(provider=self.provider, deployment=self.deployment, model=self.model)

    def supports(self, feature: str) -> bool | None:
        """Busca una capacidad en la matriz. `None` = el Broker no lo sabe.

        Distinguir «no soportado» de «sin comprobar» es la razón de que esto
        devuelva un triestado: un caso que requiere `tools` contra un modelo
        cuyo soporte es desconocido no puede contarse ni como fallo ni como
        éxito, tiene que quedar `not_applicable`.
        """
        for grupo in self.features.values():
            if isinstance(grupo, dict) and feature in grupo:
                valor = grupo[feature]
                if valor == "supported":
                    return True
                if valor == "unsupported":
                    return False
                return None
        # Segunda oportunidad: la lista plana de capacidades del proveedor.
        if feature in {item.lower() for item in self.capabilities}:
            return True
        return None


class SeedStatus(str, Enum):
    """Qué hizo el Broker con la semilla que se le pidió.

    Que un modelo honre la semilla depende de su runtime y no es observable
    desde fuera; el Broker solo promete decir si la envió.
    """

    not_requested = "not_requested"
    sent = "sent"
    unsupported = "unsupported"


class InvocationRole(str, Enum):
    """Vocabulario de `role`, enumerado en el contrato 2.10 (§8.1).

    `unknown` no lo escribe el Broker: es lo que se ve al leer una base escrita
    por una versión con más vocabulario que la que responde. Es «no lo
    reconozco», no un error.
    """

    single = "single"
    agent = "agent"
    proposer = "proposer"
    generalist = "generalist"
    specialist = "specialist"
    skeptic = "skeptic"
    analyst = "analyst"
    reviewer = "reviewer"
    refiner = "refiner"
    arbiter = "arbiter"
    chunk_map = "chunk_map"
    chunk_reduce = "chunk_reduce"
    confidence_judge = "confidence_judge"
    vision = "vision"
    shadow_probe = "shadow_probe"
    unknown = "unknown"


class InvocationStatus(str, Enum):
    """Vocabulario de `status` por invocación (contrato 2.10, §8.1)."""

    started = "started"
    completed = "completed"
    failed = "failed"
    ambiguous = "ambiguous"
    unknown = "unknown"


class PromptCompression(str, Enum):
    """Poda de cortesías y relleno del prompt (§5 y §8.5).

    `broker_default` solo aparece en `requested`: es lo que se declara cuando el
    cliente no se pronunció y mandó la configuración del Broker.
    """

    off = "off"
    light = "light"
    medium = "medium"
    aggressive = "aggressive"
    broker_default = "broker_default"


#: Reserva para Brokers anteriores al 2.10, que no marcan `contractual`. El
#: contrato pide explícitamente NO deducir esto del nombre del rol: la lista se
#: rompe en silencio en cuanto el Broker añade uno. Solo se usa cuando el campo
#: no viene, que es la única situación en la que no hay alternativa.
LEGACY_NON_CONTRACTUAL_ROLES = frozenset({"shadow_probe"})


class InvocationTelemetry(BaseModel):
    """Una entrada de `GET /api/v1/tasks/{id}/invocations`."""

    invocation_id: str = ""
    role: str = ""
    model: dict[str, Any] = Field(default_factory=dict)
    status: str = ""
    error_code: str | None = None
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float = 0.0
    latency_ms: float | None = None
    generation: dict[str, Any] = Field(default_factory=dict)
    execution_fingerprint: dict[str, Any] = Field(default_factory=dict)
    excluded_from_model_learning: bool = False
    #: Contrato 2.10 (§8.1). `None` = este Broker no se pronuncia (2.9 o
    #: anterior), que NO es lo mismo que «no es contractual».
    contractual: bool | None = None
    #: Contrato 2.10 (§8.5): `{"requested": ..., "effective": ...}`. `None` en
    #: filas anteriores al 2.10 y en llamadas que no envían prompt de usuario.
    prompt_compression: dict[str, Any] | None = None
    #: De qué campo de la respuesta salió el texto (§8.1). `reasoning_content`
    #: significa que el modelo dejó el contenido vacío y el Broker rescató su
    #: razonamiento: la respuesta vale, pero al COMPARAR modelos eso importa.
    content_source: str | None = None

    @property
    def seed_status(self) -> SeedStatus:
        crudo = str(self.generation.get("seed_status") or SeedStatus.not_requested.value)
        try:
            return SeedStatus(crudo)
        except ValueError:
            return SeedStatus.not_requested

    @property
    def model_reference(self) -> ModelReference | None:
        try:
            return ModelReference.model_validate(self.model)
        except Exception:
            return None

    @property
    def role_enum(self) -> InvocationRole:
        try:
            return InvocationRole(self.role)
        except ValueError:
            return InvocationRole.unknown

    @property
    def status_enum(self) -> InvocationStatus:
        try:
            return InvocationStatus(self.status)
        except ValueError:
            return InvocationStatus.unknown

    @property
    def is_contractual(self) -> bool:
        """Si esta llamada ejecuta el trabajo que se pidió (§8.1).

        Con `contractual` informado manda el booleano. Sin él —Broker anterior
        al 2.10— no queda más remedio que mirar el rol, y por eso el 2.10 es el
        primer contrato en el que esta pregunta tiene respuesta fiable.
        """
        if self.contractual is not None:
            return self.contractual
        return self.role not in LEGACY_NON_CONTRACTUAL_ROLES

    @property
    def compression_requested(self) -> PromptCompression | None:
        return _compresion(self.prompt_compression, "requested")

    @property
    def compression_effective(self) -> PromptCompression | None:
        """La poda que se aplicó DE VERDAD a esta llamada. `None` = no consta."""
        return _compresion(self.prompt_compression, "effective")


def _compresion(eco: dict[str, Any] | None, clave: str) -> PromptCompression | None:
    if not isinstance(eco, dict):
        return None
    crudo = eco.get(clave)
    if crudo is None:
        return None
    try:
        return PromptCompression(str(crudo))
    except ValueError:
        return None


class TaskArtifact(BaseModel):
    """Un fichero producido por la tarea (`GET /tasks/{id}/artifacts`, §8.3).

    Es la vía **canónica** para recoger el entregable: viene tipado y con su
    `sha256` sobre los bytes exactos que produjo el modelo. `result` no tiene
    esquema en el OpenAPI y no va a tenerlo, así que cerrar una medición sobre
    él es cerrar sobre algo sin contrato.
    """

    artifact_id: str = ""
    artifact_type: str = ""
    filename: str = ""
    media_type: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    created_at: str | None = None
    download_url: str | None = None
    #: `false` = la fila existe y el fichero ya no (poda de retención o
    #: restauración de una copia sin `state/tasks`). La descarga da 410, no 404.
    available: bool = True
    #: `final: true` marca el entregable; una tarea completada tiene
    #: exactamente uno, sea cual sea la estrategia. Se filtra por este booleano
    #: y no por `artifact_type`: la lista de tipos crece, el booleano no.
    #: `None` en Brokers anteriores al 2.10, donde no se puede distinguir.
    final: bool | None = None


class TaskState(BaseModel):
    """Respuesta de `GET /api/v1/tasks/{id}`."""

    task_id: str
    kind: str = "inference"
    status: str
    request_id: str | None = None
    execution_strategy: str | None = None
    progress: dict[str, Any] = Field(default_factory=dict)
    #: Contrato desde 2.9: la vía estable para saber quién respondió. `result`
    #: tiene los mismos datos, pero el propio Broker avisa de que no tiene
    #: contrato.
    execution_summary: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

    @property
    def is_terminal(self) -> bool:
        return is_terminal_status(self.status)

    @property
    def succeeded(self) -> bool:
        return self.status == TaskStatus.completed.value

    @property
    def error_code(self) -> str | None:
        if not self.error:
            return None
        code = self.error.get("code")
        return str(code) if code else None

    # --- Lectura del resultado. Campos verificados en el Broker ---

    @property
    def assistant_content(self) -> str:
        if not self.result:
            return ""
        for clave in ("assistant_content", "result_markdown"):
            valor = self.result.get(clave)
            if isinstance(valor, str):
                return valor
        return ""

    @property
    def embedding(self) -> list[float] | None:
        if not self.result:
            return None
        valor = self.result.get("embedding")
        if isinstance(valor, list) and valor:
            return [float(item) for item in valor]
        return None

    @property
    def model_used(self) -> ModelReference | None:
        """Quién respondió. `execution_summary` manda sobre `result`."""
        for origen, clave in (
            (self.execution_summary, "served_by"),
            (self.result, "model_used"),
        ):
            raw = (origen or {}).get(clave)
            if isinstance(raw, dict):
                try:
                    return ModelReference.model_validate(raw)
                except Exception:
                    continue
        return None

    @property
    def fallback_used(self) -> bool:
        if self.execution_summary is not None and "fallback_used" in self.execution_summary:
            return bool(self.execution_summary["fallback_used"])
        return bool(self.result and self.result.get("fallback_used"))

    @property
    def usage(self) -> dict[str, Any]:
        if not self.result:
            return {}
        raw = self.result.get("usage")
        return raw if isinstance(raw, dict) else {}

    @property
    def pending_tool_calls(self) -> list[dict[str, Any]]:
        if not self.result:
            return []
        raw = self.result.get("pending_tool_calls")
        return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


class BrokerError(RuntimeError):
    """Error del Broker con su código tipado, cuando lo hay."""

    def __init__(self, message: str, *, code: str | None = None, status_code: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code

    @property
    def is_auth(self) -> bool:
        """Un 403 casi nunca es fallo de integración: el Broker se reinició.

        Su token admin cambia en cada arranque salvo que se fije desde fuera,
        así que esto significa «renueva credencial y reintenta», no «tarea
        perdida»: las tareas siguen ahí.
        """
        return self.status_code in {401, 403} or self.code == "ADMIN_AUTH_REQUIRED"
