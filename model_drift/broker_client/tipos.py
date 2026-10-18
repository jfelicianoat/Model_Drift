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
    PromptCompression,
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
    #: El Broker acepta `prompt_compression` en la petición. Una medición sobre
    #: un prompt podado no compara lo que dice comparar, así que se pide `off`.
    #: Se degrada solo ante un 422, igual que el resto.
    supports_prompt_compression: bool = True
    #: Pedir `auxiliary_invocations: false` (contrato 2.10, §8.4): que solo el
    #: modelo medido vea el contenido. Model_Drift ya lo tenía por garantía
    #: implícita (`target_model` + `fallback_allowed: false`), pero eso solo
    #: vale contra un Broker 2.10 y no se declara en ninguna parte.
    request_exclusive_content: bool = True


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
    #: El Broker rechazó la credencial. Fallarán igual todas las siguientes.
    auth_failed: bool = False
    fingerprint: ExecutionFingerprint = field(default_factory=ExecutionFingerprint)
    applied_params: dict[str, Any] = field(default_factory=dict)
    #: Qué hizo el Broker con la semilla. `not_requested` mientras no haya
    #: telemetría que lo diga: no se supone que se aplicó.
    seed_status: SeedStatus = SeedStatus.not_requested
    #: Contrato 2.10 (§8.5): la poda que se aplicó de verdad a la invocación
    #: contractual que respondió. `None` = el Broker no lo declara, así que no
    #: se puede afirmar que el modelo viese el prompt íntegro.
    prompt_compression_effective: PromptCompression | None = None
    #: Contrato 2.10 (§8.1): de qué campo salió el texto. `reasoning_content`
    #: significa que el modelo dejó el contenido vacío y el Broker rescató su
    #: razonamiento. La respuesta vale, pero al comparar modelos eso importa.
    content_source: str | None = None
    #: Contrato 2.10 (§8.3): `sha256` del artefacto marcado `final: true`, que
    #: cierra sobre los bytes exactos que produjo el modelo. `None` si el Broker
    #: no publica artefactos canónicos.
    deliverable_sha256: str | None = None
    #: Cuántas invocaciones NO contractuales aparecieron bajo este `task_id`
    #: (§8.4): trabajo propio del Broker que vio el mismo contenido. No se
    #: factura a la tarea, pero sí dice quién más lo vio.
    auxiliary_invocations: int = 0
    #: Roles de esas invocaciones auxiliares, para el manifiesto.
    auxiliary_roles: list[str] = field(default_factory=list)

    @property
    def produced_output(self) -> bool:
        """Si hay salida evaluable. Un fallo de ejecución no es una respuesta mala."""
        return self.ok and bool(self.text or self.embedding)
