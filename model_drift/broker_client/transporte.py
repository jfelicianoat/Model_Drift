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
