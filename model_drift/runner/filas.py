"""Traducción de lo medido a filas de la base de datos.

La salida CRUDA se guarda siempre: es lo que permite volver a juzgar una
tirada sin reinvocar a nadie, que es la diferencia entre poder repetir un
análisis y tener que pagarlo otra vez.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from model_drift.broker_client import InvocationOutcome
from model_drift.domain import CaseOutcome


def _invocation_row(
    run_id: str, case_id: str, model_key: str, repetition: int, outcome: InvocationOutcome
) -> dict[str, Any]:
    return {
        "invocation_id": f"inv-{uuid.uuid4().hex[:16]}",
        "run_id": run_id,
        "case_id": case_id,
        "model_key": model_key,
        "repetition": repetition,
        "variant_id": None,
        "broker_task_id": outcome.task_id or None,
        "ok": outcome.ok,
        # La salida CRUDA, siempre. Es lo que permite re-juzgar sin reinvocar.
        "output_text": outcome.text,
        "tool_calls_json": outcome.tool_calls,
        "tokens_input": outcome.tokens_input,
        "tokens_output": outcome.tokens_output,
        "cost_usd": outcome.cost_usd,
        "latency_ms": outcome.latency_ms,
        "model_served": outcome.model_served.key if outcome.model_served else None,
        "fallback_used": outcome.fallback_used,
        "fingerprint_json": outcome.fingerprint.model_dump(mode="json"),
        "applied_params_json": outcome.applied_params,
        "error_code": outcome.error_code,
        "error_message": outcome.error_message,
        "created_at": datetime.now(UTC).isoformat(),
    }


def _case_row(run_id: str, outcome: CaseOutcome) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "case_id": outcome.case_id,
        "model_key": outcome.model_key,
        "metric": outcome.metric,
        "case_score": outcome.case_score,
        "case_stability": outcome.case_stability,
        "repetitions": outcome.repetitions,
        "failed_repetitions": outcome.failed_repetitions,
        "is_unstable": outcome.is_unstable,
        "is_not_applicable": outcome.is_not_applicable,
        "not_applicable_reason": outcome.not_applicable_reason,
        "values_json": list(outcome.values),
    }
