"""Espejo del contrato del AI Broker.

Copia deliberada, no importación: Model_Drift es un cliente HTTP y no comparte
proceso ni paquete con el Broker. La contrapartida es la regla que ya rige en el
Knowledge_Orchestrator — **cada cambio del contrato del Broker exige actualizar
este fichero en el mismo cambio**, porque un estado o una estrategia
desconocidos hacen que el sondeo mate la tarea por validación.

Estado verificado contra el Broker en marcha (contrato v2.9, 16-ago-2026),
**después** de que incorporase la entrega de `Evaluation_Contract.md`:

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
  `exclude_from_model_learning`, `invocation_telemetry`, `execution_fingerprint`.
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
