"""Transporte HTTP: sesión, cabeceras, errores y garantías perdidas.

Todo lo que sale hacia el Broker pasa por `_request`, y ahí se traduce
cualquier respuesta de error a `BrokerError` con su código. Las garantías
que no se pudieron obtener se acumulan en `missing_guarantees`, que va tal
cual al manifiesto: una tirada dice lo que no pudo garantizar.
"""
from __future__ import annotations

from typing import Any, Self

import httpx

from model_drift.broker_client.tipos import BrokerConfig
from model_drift.broker_contracts import (
    BrokerError,
    ModelCatalogEntry,
)


class TransporteBroker:
    """Sesión HTTP contra el Broker y traducción de sus errores."""

    def __init__(self, config: BrokerConfig | None = None, *, client: httpx.Client | None = None) -> None:
        self.config = config or BrokerConfig()
        self._client = client or httpx.Client(
            base_url=self.config.base_url.rstrip("/"),
            timeout=self.config.timeout_seconds,
        )
        self._owns_client = client is None
        #: Garantías que no se pudieron obtener. Va tal cual al manifiesto.
        self.missing_guarantees: list[str] = []
        self._catalog_cache: list[ModelCatalogEntry] | None = None
        self._features_cache: dict[str, dict[str, Any]] = {}
        #: Se rellenan al sondear el contrato (`probe_contract`).
        self.contract_version: str | None = None
        self.supports_invocation_telemetry = False
        #: Contrato 2.10. `invocation_contract`: `role`/`status` enumerados y
        #: `contractual` en cada invocación (§8.1). Sin esto, separar el trabajo
        #: propio del Broker del que se pidió es una heurística sobre el rol.
        self.supports_invocation_contract = False
        #: `prompt_compression_echo`: cada invocación declara la poda que se le
        #: aplicó de verdad (§8.5).
        self.supports_prompt_compression_echo = False
        #: `task_artifacts` y `canonical_artifacts`: listar los ficheros que
        #: produce una tarea, y saber cuál de ellos es el entregable (§8.3).
        self.supports_task_artifacts = False
        self.supports_canonical_artifacts = False
        #: `auxiliary_invocations`: si este Broker sondea otros modelos con el
        #: contenido de la tarea bajo su mismo `task_id` (§8.4). Ausente en 2.9,
        #: donde sí lo hacía: por eso el default optimista es `True`.
        self.performs_auxiliary_invocations = True
        #: `auxiliary_invocations_optout`: si acepta apagarlas en la petición.
        self.supports_auxiliary_optout = False
        self._system1_available: bool | None = None
        self._system1_evaluation_available = False

    # ---------------------------------------------------------------- infra

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _headers(self) -> dict[str, str]:
        if self.config.admin_token:
            return {"X-Admin-Token": self.config.admin_token}
        return {}

    def _declare_missing(self, guarantee: str) -> None:
        if guarantee not in self.missing_guarantees:
            self.missing_guarantees.append(guarantee)

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._client.request(method, url, headers=self._headers(), **kwargs)
        except httpx.TimeoutException as error:
            raise BrokerError("se agotó el plazo de la petición al Broker", code="TIMEOUT") from error
        except httpx.HTTPError as error:
            raise BrokerError(f"no se pudo contactar con el Broker: {error}") from error
        if response.status_code >= 400:
            codigo: str | None = None
            mensaje = response.text[:400]
            try:
                payload = response.json()
                if isinstance(payload, dict):
                    detalle = payload.get("detail")
                    if isinstance(detalle, dict):
                        codigo = str(detalle.get("code") or "") or None
                        mensaje = str(detalle.get("message") or mensaje)
                    elif isinstance(detalle, str):
                        mensaje = detalle
            except ValueError:
                pass
            raise BrokerError(mensaje, code=codigo, status_code=response.status_code)
        return response
