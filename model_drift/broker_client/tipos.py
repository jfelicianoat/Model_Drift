"""Configuración del cliente y resultado de una invocación.

Están aparte del cliente porque los mira todo el motor: el `Runner` guarda
`InvocationOutcome` en la base y la GUI construye `BrokerConfig` sin tocar
una sola línea de HTTP.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from model_drift.broker_contracts import (
    ModelReference,
    SeedStatus,
)
from model_drift.fingerprint import (
    ExecutionFingerprint,
)

#: Marca de origen del tráfico de evaluación. El Broker la trata como metadata
#: opaca y permite filtrar por ella en el listado de tareas.
ORIGIN = "model_drift"


@dataclass
class BrokerConfig:
    base_url: str = "http://192.168.1.52:8765"
    admin_token: str | None = None
    timeout_seconds: float = 30.0
    poll_interval_seconds: float = 2.0
    max_wait_seconds: float = 900.0
    #: El Broker acepta `generation.seed`. Se descubre solo al primer 422 y no
    #: hace falta configurarlo a mano.
    supports_seed: bool = True
    #: El Broker admite marcar tráfico como excluido del aprendizaje (Cambio 4).
    #: Se intenta por defecto: contra un Broker que ya lo tiene, la medición
    #: deja de contaminar su router adaptativo sin que nadie configure nada.
    supports_learning_exclusion: bool = True


@dataclass
class InvocationOutcome:
    """Una invocación completa, con su salida cruda y su telemetría."""

    task_id: str
    ok: bool
    text: str = ""
    embedding: list[float] | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    model_served: ModelReference | None = None
    fallback_used: bool = False
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    error_code: str | None = None
    error_message: str = ""
    fingerprint: ExecutionFingerprint = field(default_factory=ExecutionFingerprint)
    applied_params: dict[str, Any] = field(default_factory=dict)
    #: Qué hizo el Broker con la semilla. `not_requested` mientras no haya
    #: telemetría que lo diga: no se supone que se aplicó.
    seed_status: SeedStatus = SeedStatus.not_requested

    @property
    def produced_output(self) -> bool:
        """Si hay salida evaluable. Un fallo de ejecución no es una respuesta mala."""
        return self.ok and bool(self.text or self.embedding)
