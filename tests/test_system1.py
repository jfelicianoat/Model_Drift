"""Contrato HTTP, integración y decisiones conservadoras del filtro System 1."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from model_drift.broker_client import BrokerClient, BrokerConfig
from model_drift.broker_contracts import ModelReference
from model_drift.cases import Case, Message, Suite
from model_drift.comparison import RunResults, compare_runs
from model_drift.comparison.service import compare_configured
from model_drift.config import AppConfig
from model_drift.db import Database
from model_drift.domain import CaseOutcome, ChangeDirection, ChangeStatus
from model_drift.gui.comparacion import ComparacionMixin
from model_drift.judge import Judge, JudgeConfig
from model_drift.manifest import RunManifest
from model_drift.oracles import Assertion, JsonValidAssertion
from model_drift.policy import MetricRule, RegressionPolicy, Severity
from model_drift.report import render_comparison, render_text_summary
from model_drift.system1_contracts import SEMANTIC_OPTIONS, System1Config


def _pair(*, assertions: list[Assertion] | None = None, candidate_score: float = 1.0,
          baseline_text: str = "Pide el recibo y conserva una copia.",
          candidate_text: str = "Solicita el recibo y guarda una copia.", repetitions: int = 5):
    case = Case(id="c", messages=[Message(role="user", content="Explica qué hacer con el recibo y su copia.")],
                assertions=assertions or [])
    suite = Suite(id="semantic-demo", cases=[case])
    runs = []
    for run_id, text, score in (("base", baseline_text, 1.0), ("candidate", candidate_text, candidate_score)):
        run = RunResults(run_id=run_id, manifest=RunManifest(run_id=run_id, suite_hash=suite.hash()),
                         model_key=f"ollama/local/{run_id}")
        run.sample_output["c"] = text
        run.outcomes[("c", "pass_rate")] = CaseOutcome(
            case_id="c", model_key=run.model_key, case_score=score, case_stability=1.0,
            repetitions=repetitions, values=[score] * repetitions,
        )
        runs.append(run)
    return runs[0], runs[1], suite


def _judgment(label: str = "equivalent", confidence: float = 0.99, **updates: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "use_case": "model_drift_semantic", "decision": label, "confidence": confidence,
        "confidence_is_calibrated": False, "provider": "laya_mcp", "model": "local-model",
        "latency_ms": 2.0, "accepted": True, "fallback_used": False, "reason_code": None,
        "attempts": [{"provider": "laya_mcp", "model": "local-model", "latency_ms": 2.0,
                      "reason_code": None, "tokens_input": 40, "tokens_output": 1}],
        "alternatives": [{"value": option, "confidence": (1.0 - float(confidence or 0.99)) / 3}
                         for option in SEMANTIC_OPTIONS if option != label],
    }
    data.update(updates)
    return data


def _client(judgment: Any = None, *, available: bool = True, failure: str | None = None,
            status: int = 200) -> tuple[BrokerClient, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if failure == "offline":
            raise httpx.ConnectError("offline", request=request)
        if request.url.path == "/api/v1/capabilities":
            return httpx.Response(200, json={"system1_judgments": available})
        assert request.url.path == "/api/v1/system1/judge"
        if failure == "timeout":
            raise httpx.ReadTimeout("timeout", request=request)
        return httpx.Response(status, json=judgment or _judgment())

    http = httpx.Client(base_url="http://broker.test", transport=httpx.MockTransport(handle))
    return BrokerClient(BrokerConfig(base_url="http://broker.test", admin_token="test-token"), client=http), requests


@pytest.mark.parametrize("label,confidence,priority", [
    ("equivalent", 0.99, "normal"), ("changed_but_not_regression", 0.96, "normal"),
    ("regression_candidate", 0.98, "high"), ("regression_candidate", 0.90, "review"),
    ("uncertain", 0.96, "review"),
])
def test_semantic_classifications_do_not_create_evidence(label: str, confidence: float, priority: str) -> None:
    base, cand, suite = _pair()
    client, requests = _client(_judgment(label, confidence))
    previous = compare_runs(base, cand, suite=suite)
    report = compare_runs(base, cand, suite=suite, broker=client,
                          system1=System1Config(enabled=True, shadow_mode=False))
    assert report.semantic_review is not None
    audit = report.semantic_review.assessments[0]
    assert (audit.classification, audit.confidence, audit.priority) == (label, confidence, priority)
    assert audit.deep_analysis_required
    assert report.outcomes == previous.outcomes and report.verdict == previous.verdict
    assert report.outcomes[0].change_direction is ChangeDirection.sin_direccion
    assert len(requests) == 2
    body = json.loads(requests[1].content)
    assert set(body) == {"use_case", "input", "decision_type", "options", "criteria", "instructions", "cloud_allowed"}
    assert body["cloud_allowed"] is False
    assert body["options"] == list(SEMANTIC_OPTIONS)
    assert "model_key" not in json.dumps(body) and "task_id" not in body
    assert body["input"]["deterministic_checks"]
    assert requests[1].headers["X-Admin-Token"] == "test-token"
    assert requests[1].extensions["timeout"]["read"] == 75.0


def test_deterministic_json_failure_never_asks_system1_even_underpowered() -> None:
    base, cand, suite = _pair(assertions=[JsonValidAssertion()], candidate_score=0.0,
                             baseline_text='{"ok":true}', candidate_text="JSON no válido", repetitions=1)
    client, requests = _client()
    report = compare_runs(base, cand, suite=suite, system1=System1Config(enabled=True), broker=client)
    assert report.semantic_review is not None
    assert report.semantic_review.metrics.deterministic_pairs == 1
    assert report.semantic_review.assessments[0].reason_code == "DETERMINISTIC_CHECK_CHANGED_OR_FAILED"
    assert not requests
    assert report.outcomes[0].change_status is ChangeStatus.indeterminado


def test_identical_output_is_resolved_before_network() -> None:
    base, cand, suite = _pair(candidate_text="Pide el recibo y conserva una copia.")
    client, requests = _client()
    report = compare_runs(base, cand, suite=suite, system1=System1Config(enabled=True), broker=client)
    assert report.semantic_review.assessments[0].source == "deterministic"
    assert report.semantic_review.assessments[0].classification == "equivalent"
    assert not requests


@pytest.mark.parametrize("failure,reason", [("offline", "BROKER_UNAVAILABLE"), ("timeout", "TIMEOUT")])
def test_broker_failure_preserves_previous_decisions(failure: str, reason: str) -> None:
    base, cand, suite = _pair()
    client, _ = _client(failure=failure)
    previous = compare_runs(base, cand, suite=suite)
    report = compare_runs(base, cand, suite=suite, system1=System1Config(enabled=True), broker=client)
    audit = report.semantic_review.assessments[0]
    assert audit.fallback_used and audit.reason_code == reason
    assert audit.classification == "uncertain" and audit.deep_analysis_required
    assert report.outcomes == previous.outcomes and report.verdict == previous.verdict


def test_rejected_http_200_is_not_equivalence() -> None:
    base, cand, suite = _pair()
    client, requests = _client(_judgment(accepted=False, decision=None, confidence=None,
                                       alternatives=[], fallback_used=True, reason_code="LOW_CONFIDENCE"))
    report = compare_runs(base, cand, suite=suite, system1=System1Config(enabled=True), broker=client)
    audit = report.semantic_review.assessments[0]
    assert not audit.accepted and audit.classification == "uncertain"
    assert audit.reason_code == "LOW_CONFIDENCE" and audit.deep_analysis_required
    assert len(requests) == 2  # sin reintentos ni sondeo de tareas


def test_provider_fallback_is_broker_owned_and_auditable() -> None:
    base, cand, suite = _pair()
    attempts = [
        {"provider": "laya_mcp", "model": None, "latency_ms": 1.0, "reason_code": "MCP_ERROR"},
        {"provider": "ollama_system1", "model": "fallback-model", "latency_ms": 2.0, "reason_code": None},
    ]
    client, requests = _client(_judgment("regression_candidate", provider="ollama_system1",
                                       fallback_used=True, reason_code="MCP_ERROR", attempts=attempts))
    report = compare_runs(base, cand, suite=suite, system1=System1Config(enabled=True), broker=client)
    audit = report.semantic_review.assessments[0]
    assert audit.accepted and audit.provider == "ollama_system1" and audit.reason_code == "MCP_ERROR"
    assert len(audit.attempts) == 2 and report.semantic_review.metrics.fallbacks == 1
    assert all(request.url.host == "broker.test" for request in requests)


@pytest.mark.parametrize("updates", [
    {"decision": "wrong"}, {"accepted": "true"}, {"confidence": True}, {"confidence": 1.1},
    {"alternatives": []}, {"use_case": "another"}, {"provider": None},
    {"accepted": False, "fallback_used": True}, {"confidence": "0.99"},
])
def test_invalid_outputs_fall_back(updates: dict[str, Any]) -> None:
    base, cand, suite = _pair()
    client, _ = _client(_judgment(**updates))
    report = compare_runs(base, cand, suite=suite, system1=System1Config(enabled=True), broker=client)
    audit = report.semantic_review.assessments[0]
    assert audit.reason_code == "INVALID_OUTPUT" and not audit.accepted
    assert audit.deep_analysis_required and audit.fallback_used


def test_missing_capability_does_not_post() -> None:
    base, cand, suite = _pair()
    client, requests = _client(available=False)
    report = compare_runs(base, cand, suite=suite, system1=System1Config(enabled=True), broker=client)
    assert report.semantic_review.assessments[0].reason_code == "SYSTEM1_UNAVAILABLE"
    assert len(requests) == 1


def test_thresholds_are_configurable_and_not_probabilities() -> None:
    base, cand, suite = _pair()
    client, _ = _client(_judgment("regression_candidate", 0.90))
    report = compare_runs(base, cand, suite=suite, broker=client,
                          system1=System1Config(enabled=True, regression_threshold=0.92, high_priority_threshold=0.99))
    assert report.semantic_review.metrics.regression_candidates == 0
    assert report.semantic_review.assessments[0].deep_analysis_required
    assert not report.semantic_review.assessments[0].confidence_is_calibrated
    client, _ = _client(_judgment("regression_candidate", 0.90))
    report = compare_runs(base, cand, suite=suite, broker=client,
                          system1=System1Config(enabled=True, high_priority_threshold=0.88))
    assert report.semantic_review.assessments[0].priority == "high"


@pytest.mark.parametrize("shadow,calibrated,mandatory,skipped", [
    (False, True, False, True), (True, True, False, False),
    (False, False, False, False), (False, True, True, False),
])
def test_only_qualified_equivalence_can_skip(shadow: bool, calibrated: bool, mandatory: bool, skipped: bool) -> None:
    base, cand, suite = _pair()
    client, _ = _client(_judgment(confidence_is_calibrated=calibrated))
    policy = RegressionPolicy(metrics={"pass_rate": MetricRule(severity=Severity.blocking)}) if mandatory else None
    report = compare_runs(base, cand, suite=suite, broker=client, policy=policy,
                          system1=System1Config(enabled=True, shadow_mode=shadow, allow_equivalence_skip=True))
    audit = report.semantic_review.assessments[0]
    assert audit.deep_analysis_required is not skipped
    assert report.outcomes[0].change_direction is ChangeDirection.sin_direccion
    if shadow:
        assert not audit.proposed_deep_analysis
        assert report.semantic_review.metrics.shadow_disagreements == 1


def test_off_reproduces_previous_report_without_any_network() -> None:
    base, cand, suite = _pair()
    client, requests = _client()
    old = compare_runs(base, cand, suite=suite)
    off = compare_runs(base, cand, suite=suite, system1=System1Config(), broker=client)
    assert off.model_dump() == old.model_dump() and not requests
    assert "System 1" not in render_text_summary(off)


def test_missing_input_and_incomparability_never_claim_no_drift() -> None:
    base, cand, suite = _pair()
    client, requests = _client()
    report = compare_runs(base, cand, system1=System1Config(enabled=True), broker=client)
    assert report.semantic_review.assessments[0].reason_code == "MISSING_TEST_INPUT"
    cand.manifest.suite_hash = "changed"
    report = compare_runs(base, cand, suite=suite, system1=System1Config(enabled=True), broker=client)
    assert report.semantic_review.assessments[0].reason_code == "INCOMPARABLE_RUNS"
    assert report.semantic_review.assessments[0].deep_analysis_required and not requests


def test_configured_comparison_persists_audit_and_reports(tmp_path: Path) -> None:
    base, cand, suite = _pair()
    client, _ = _client()
    config = AppConfig(database=str(tmp_path / "md.sqlite"), system1=System1Config(enabled=True))
    report = compare_configured(base, cand, suite=suite, config=config, client_factory=lambda: client)
    with Database(config.database) as db:
        rows = db.query("SELECT * FROM comparison_review")
    assert len(rows) == 1 and rows[0]["baseline_model_key"] == base.model_key
    data = json.loads(rows[0]["review_json"])
    assert data["config"]["regression_threshold"] == 0.8
    assert data["assessments"][0]["provider"] == "laya_mcp"
    assert "test-token" not in rows[0]["review_json"] and base.sample_output["c"] not in rows[0]["review_json"]
    html = render_comparison(report)
    assert "Modo sombra" in html and "equivalente" in html and "pendiente" in html
    assert "System 1" in render_text_summary(report)
    assert ComparacionMixin._semantic_priority(report, "c") == 2


def test_existing_deep_judge_runs_on_uncertainty_and_failure() -> None:
    from tests.broker_fake import FakeBroker

    base, cand, suite = _pair()
    fake = FakeBroker()
    model = ModelReference(provider="ollama", deployment="local", model="deep")
    fake.add_model(model.key)
    fake.respond_fixed(model.key, '{"veredicto":"empate","motivo":"Mismo sentido"}')
    with fake.client() as judge_client:
        judge = Judge(judge_client, JudgeConfig(model=model))
        failed_client, _ = _client(failure="offline")
        report = compare_runs(base, cand, suite=suite, system1=System1Config(enabled=True),
                              broker=failed_client, judge=judge)
    audit = report.semantic_review.assessments[0]
    assert audit.deep_analysis_executed and audit.deep_verdict == "empate"
    assert audit.deep_judgment["consistent"] is True
    assert report.semantic_review.metrics.deep_analysis_executed == 1
    assert report.outcomes[0].change_direction is ChangeDirection.sin_direccion


def test_config_round_trip_and_invalid_thresholds(tmp_path: Path) -> None:
    config = AppConfig(system1=System1Config(enabled=True, regression_threshold=0.7, timeout_seconds=90))
    config.save(tmp_path / "config.yaml")
    assert AppConfig.load(tmp_path / "config.yaml").system1 == config.system1
    with pytest.raises(ValueError):
        System1Config(regression_threshold=0.95, high_priority_threshold=0.8)
    with pytest.raises(ValueError):
        System1Config(timeout_seconds=float("nan"))
