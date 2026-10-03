"""Juicios síncronos System-1 del Broker; nunca invoca proveedores directamente."""
from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from model_drift.broker_client.transporte import TransporteBroker
from model_drift.broker_contracts import BrokerError
from model_drift.system1_contracts import System1Judgment


class System1Mixin(TransporteBroker):
    def system1_raw(
        self, payload: dict[str, Any], *, timeout_seconds: float, require_evaluation: bool = False,
    ) -> dict[str, Any]:
        if self._system1_available is None:
            capabilities = self._request("GET", "/api/v1/capabilities", timeout=timeout_seconds).json()
            self._system1_available = isinstance(capabilities, dict) and capabilities.get("system1_judgments") is True
            self._system1_evaluation_available = (
                isinstance(capabilities, dict) and capabilities.get("system1_evaluation") is True
            )
        if not self._system1_available:
            raise BrokerError("el Broker no anuncia juicios System-1", code="SYSTEM1_UNAVAILABLE")
        if require_evaluation and not self._system1_evaluation_available:
            raise BrokerError("el Broker no anuncia evaluación System-1", code="EVALUATION_UNAVAILABLE")
        response = self._request("POST", "/api/v1/system1/judge", json=payload, timeout=timeout_seconds)
        try:
            raw = response.json()
        except ValueError as error:
            raise BrokerError("respuesta System-1 no es JSON", code="INVALID_OUTPUT") from error
        if not isinstance(raw, dict) or raw.get("use_case") != payload["use_case"]:
            raise BrokerError("respuesta System-1 sin el caso de uso solicitado", code="INVALID_OUTPUT")
        return raw

    def system1_judge(self, payload: dict[str, Any], *, timeout_seconds: float) -> System1Judgment:
        raw = self.system1_raw(payload, timeout_seconds=timeout_seconds)
        try:
            judgment = System1Judgment.model_validate(raw)
        except (ValueError, ValidationError) as error:
            raise BrokerError("respuesta System-1 inválida", code="INVALID_OUTPUT") from error
        if judgment.use_case != payload["use_case"]:
            raise BrokerError("el Broker devolvió otro caso de uso", code="INVALID_OUTPUT")
        return judgment
