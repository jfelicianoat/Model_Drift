"""Aplicar la configuración de la app y conservar la auditoría de comparación."""
from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import uuid4

from model_drift.broker_client import BrokerClient
from model_drift.cases import Suite
from model_drift.comparison.comparador import compare_runs
from model_drift.comparison.resultados import ComparisonReport, RunResults
from model_drift.config import AppConfig
from model_drift.db import Database
from model_drift.judge import Judge, JudgeConfig
from model_drift.policy import RegressionPolicy


def compare_configured(
    baseline: RunResults, candidate: RunResults, *, config: AppConfig,
    suite: Suite | None = None, policy: RegressionPolicy | None = None,
    client_factory: Callable[[], BrokerClient] | None = None,
) -> ComparisonReport:
    if not config.system1.enabled:
        return compare_runs(baseline, candidate, suite=suite, policy=policy)
    factory = client_factory or (lambda: BrokerClient(config.broker_config()))
    with factory() as client:
        judge = (
            Judge(client, JudgeConfig(model=config.model_reference(config.judge_model)))
            if config.judge_model else None
        )
        report = compare_runs(baseline, candidate, suite=suite, policy=policy,
                              system1=config.system1, broker=client, judge=judge)
    if report.semantic_review is not None:
        with Database(config.database) as db:
            db.insert("comparison_review", {
                "review_id": f"review-{uuid4().hex}",
                "created_at": datetime.now(UTC).isoformat(),
                "baseline_run_id": baseline.run_id,
                "candidate_run_id": candidate.run_id,
                "baseline_model_key": baseline.model_key,
                "candidate_model_key": candidate.model_key,
                "review_json": report.semantic_review.model_dump_json(),
            })
    return report
