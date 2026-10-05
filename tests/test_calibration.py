"""Pruebas de calibración con respuestas del contrato 2.11/§15.8.

El contrato se contrasta también con el probe vivo cuando el Broker es accesible;
las fixtures no se presentan como validación de un proveedor real.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
import pytest

from model_drift import cli
from model_drift.calibration.analysis import analyze_calibration, compare_calibrations
from model_drift.calibration.models import (
    CalibrationCase,
    CalibrationConfig,
    CalibrationPlan,
    CalibrationReport,
    EvaluationTarget,
    LearnerNote,
)
from model_drift.calibration.runner import (
    export_distilled,
    fingerprint_changed,
    partition_cases,
    revalidate_changed,
    run_calibration,
    save_artifacts,
)
from model_drift.calibration.validation import validate_note
from model_drift.comparison import compare_runs
from model_drift.config import AppConfig
from model_drift.db import Database
from model_drift.report import render_calibration_summary, write_calibration
from model_drift.statistics.calibration import clopper_pearson
from model_drift.system1_benchmark import ReferenceLabel, score_labels
from model_drift.system1_contracts import System1Config
from tests.broker_fake import FakeBroker
from tests.test_system1 import _client, _pair


def _plan(n: int = 80, **updates: Any) -> CalibrationPlan:
    config = {
        "use_case": "goal_completion",
        "decision_type": "binary",
        "instructions": "La tarea está completada si se demuestra que cumple todos los criterios.",
        "threshold_profile": "goal_completion",
        "learner": {"provider": "ollama_system1", "model": "native"},
        "teacher_task": {"provider": "ollama", "deployment": "local", "model": "qwen3:8b"},
        "teacher_repetitions": 3,
        "min_human_cases": 2,
        "min_calibration_cases": 2,
        "min_test_cases": 2,
        "target_precision": 0.60,
        "auto_approve_values": [True],
    }
    config.update(updates)
    return CalibrationPlan(
        config=CalibrationConfig.model_validate(config),
        cases=[
            CalibrationCase(
                id=f"c{i:03}",
                input={"case_id": i, "truth": i % 2 == 0, "acceptance_criteria": ["A", "B", "C"]},
                human_label=i % 2 == 0,
                human_verified=True,
                adversarial=i % 10 == 0,
            )
            for i in range(n)
        ],
    )


def _raw(
    decision: Any = True,
    confidence: float = 0.99,
    *,
    accepted: bool = True,
    model: str = "native",
    provider: str = "ollama_system1",
    **updates: Any,
) -> dict[str, Any]:
    attempt = {
        "provider": provider,
        "model": model,
        "decision": decision,
        "confidence": confidence,
        "alternatives": [{"value": not decision, "confidence": 1.0 - confidence}],
        "score_source": "native",
        "reason_code": None if accepted else "LOW_CONFIDENCE",
        "latency_ms": 12.0,
        "tokens_input": 30,
        "tokens_output": 1,
    }
    attempt.update(updates)
    return {
        "use_case": "goal_completion",
        "accepted": accepted,
        "decision": decision if accepted else None,
        "confidence": confidence if accepted else None,
        "confidence_is_calibrated": False,
        "provider": provider,
        "model": model,
        "latency_ms": 12.0,
        "fallback_used": not accepted,
        "reason_code": attempt["reason_code"],
        "alternatives": attempt["alternatives"] if accepted else [],
        "attempts": [attempt],
    }


class EvaluationBroker(FakeBroker):
    def __init__(self, *, evaluation: bool = True) -> None:
        super().__init__()
        self.evaluation = evaluation
        self.judgments: list[dict[str, Any]] = []
        self.teacher_counts: Counter[int] = Counter()
        self.teacher_ambiguous = False
        self.teacher_wrong = False
        self.teacher_cold_timeout = False
        self.teacher_judge_timeout = False
        self.invalid_student = False
        self.student_rejected = False
        self.student_false_approvals = False
        self.student_model_change = False
        self.student_digest = "digest-v1"
        self.add_model("ollama/local/qwen3:8b")
        self.add_model("ollama/local/native")
        self.respond("ollama/local/qwen3:8b", self.teacher_response)

    def teacher_response(self, prompt: str, model: str, seen: int) -> str:
        payload = json.loads(prompt[prompt.index("{") :])
        data = payload["input"]
        index = data["case_id"]
        self.teacher_counts[index] += 1
        decision = data["truth"]
        if self.teacher_wrong or (self.teacher_ambiguous and index == 0 and self.teacher_counts[index] % 2 == 0):
            decision = not decision
        return json.dumps({"decision": decision})

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/capabilities":
            data = super().handler(request).json()
            data.update(system1_judgments=True, system1_evaluation=self.evaluation, contract_version="2.11")
            return httpx.Response(200, json=data)
        if request.url.path == "/api/v1/system1/judge":
            payload = json.loads(request.content)
            self.judgments.append(payload)
            teacher = payload.get("target", {}).get("model") == "qwen3:8b"
            if teacher and self.teacher_judge_timeout:
                self.teacher_judge_timeout = False
                raw = _raw(
                    accepted=False,
                    decision=None,
                    confidence=0.7,
                    model="qwen3:8b",
                    reason_code="TIMEOUT",
                    alternatives=[],
                )
                raw["attempts"][0]["confidence"] = None
                return httpx.Response(200, json=raw)
            raw = _raw(
                payload["input"]["truth"] or (not teacher and self.student_false_approvals),
                0.62 if self.student_rejected else 0.99,
                accepted=not self.student_rejected,
                model="qwen3:8b"
                if teacher
                else "changed"
                if self.student_model_change and len(self.judgments) > 1
                else "native",
            )
            if teacher:
                raw["accepted"] = False
                raw["decision"] = raw["confidence"] = None
                raw["reason_code"] = "SELF_REPORTED_SCORE"
                raw["attempts"][0].update(
                    score_source="self_reported", confidence=1.0, alternatives=[], reason_code="SELF_REPORTED_SCORE"
                )
            if self.invalid_student and not teacher:
                raw["attempts"][0]["alternatives"] = []
            return httpx.Response(200, json=raw)
        if request.url.path == "/api/v1/tasks" and request.method == "POST" and self.teacher_cold_timeout:
            self.teacher_cold_timeout = False
            payload = json.loads(request.content)
            self.fail_prompts[payload["content"]["prompt"]] = "TIMEOUT"
            response = super().handler(request)
            self.fail_prompts.clear()
            return response
        return super().handler(request)


def _run(broker: EvaluationBroker, plan: CalibrationPlan, db: Database | None = None) -> CalibrationReport:
    if db is not None:
        with broker.client() as client:
            return run_calibration(client, plan, db)
    with Database(":memory:") as memory, broker.client() as client:
        return run_calibration(client, plan, memory)


def test_unanimous_teacher_and_student_produce_measured_calibration() -> None:
    broker = EvaluationBroker()
    report = _run(broker, _plan())
    assert report.teacher_qualified and report.teacher_kappa == 1.0
    assert report.status == "calibrado", (report.reasons, report.limitations)
    assert report.recommended_threshold == 0.99
    assert report.test_precision_lower >= report.config.target_precision
    assert report.metrics["raw_test"]["accuracy"] == 1.0
    assert report.metrics["test_reliability"]["brier_multiclass"] == pytest.approx(0.0002)
    assert report.metrics["provenance"]["distilled_system2"] == 80
    assert report.metrics["adversarial_test"]["n"] > 0
    assert all(item["target"] == {"provider": "ollama_system1", "model": "native"} for item in broker.judgments)
    for payload in broker.requests:
        assert payload["exclude_from_model_learning"] is True
        assert payload["model_requirements"]["fallback_allowed"] is False
        assert payload["generation"]["temperature"] == 0
        assert payload["generation"]["seed"] == report.config.seed
        assert "student" not in payload["content"]["prompt"]


def test_nonunanimous_teacher_is_ambiguous_and_excluded() -> None:
    broker = EvaluationBroker()
    broker.teacher_ambiguous = True
    report = _run(broker, _plan())
    assert report.ambiguous_cases == 1
    note = next(note for note in report.notes if note.case_id == "c000")
    assert not note.reference_eligible
    assert report.labels["c000"].reason_code == "AMBIGUOUS_TEACHER"


def test_teacher_below_kappa_never_recommends_threshold() -> None:
    broker = EvaluationBroker()
    broker.teacher_wrong = True
    report = _run(broker, _plan())
    assert not report.teacher_qualified and report.status == "no_calibrado"
    assert report.recommended_threshold is None


def test_rejected_student_is_abstention_but_valid_raw_score_calibrates() -> None:
    broker = EvaluationBroker()
    broker.student_rejected = True
    report = _run(broker, _plan())
    assert report.metrics["abstentions"] == 80
    assert report.metrics["accepted_test"]["n"] == 0
    assert report.metrics["raw_test"]["n"] > 0
    assert all(note.valid_score for note in report.notes)
    assert report.recommended_threshold == 0.62


def test_clopper_pearson_is_exact_and_does_not_use_point_estimate() -> None:
    assert clopper_pearson(367, 367, 0.95)[0] < 0.99 <= clopper_pearson(368, 368, 0.95)[0]
    low, high = clopper_pearson(10, 10)
    assert low == pytest.approx(0.025**0.1) and high == 1.0
    assert low < 0.99  # diez aciertos de diez NO demuestran precisión del 99 %
    assert clopper_pearson(5, 10) == pytest.approx((0.187086028447, 0.812913971553))
    assert clopper_pearson(0, 0) == (0.0, 1.0)
    assert clopper_pearson(0, 10)[1] == pytest.approx(1.0 - 0.025**0.1)
    with pytest.raises(ValueError):
        clopper_pearson(11, 10)


def test_small_sample_is_indeterminate() -> None:
    report = _run(EvaluationBroker(), _plan(n=4, target_precision=0.99))
    assert report.status == "indeterminado" and report.recommended_threshold is None


def test_fixed_partition_does_not_depend_on_scores_or_input_order() -> None:
    plan = _plan()
    partition = partition_cases(plan)
    plan.cases.reverse()
    assert partition_cases(plan) == partition
    assert set(partition.values()) == {"calibration", "test"}


def test_threshold_is_never_reselected_on_test_failure() -> None:
    plan = _plan()
    report = _run(EvaluationBroker(), plan)
    for note in report.notes:
        if note.partition == "test":
            note.decision = not note.reference
    report.recommended_threshold = None
    analyze_calibration(report, plan)
    assert report.status == "no_calibrado" and report.recommended_threshold is None
    assert report.metrics["candidate_threshold_from_calibration"] == 0.99


def test_unresolved_evaluation_contract_blocks_recommendation() -> None:
    report = _run(EvaluationBroker(evaluation=False), _plan(n=4))
    assert report.status == "no_calibrado" and report.recommended_threshold is None
    assert any("system1_evaluation" in item for item in report.limitations)


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({"alternatives": []}, "INCOMPLETE_ALTERNATIVES"),
        ({"alternatives": [{"value": False, "confidence": 1.0}]}, "ALTERNATIVE_ABOVE_DECISION"),
        (
            {"alternatives": [{"value": False, "confidence": 0.005}, {"value": False, "confidence": 0.005}]},
            "DUPLICATE_ALTERNATIVE",
        ),
        ({"confidence": float("nan")}, "INVALID_CONFIDENCE"),
        ({"confidence": True}, "INVALID_CONFIDENCE"),
        ({"decision": "true"}, "decisión fuera del tipo o dominio solicitado"),
        ({"score_source": "self_reported"}, "NON_NATIVE_SCORE"),
    ],
)
def test_every_raw_student_note_is_validated(changes: dict[str, Any], expected: str) -> None:
    raw = _raw(**changes)
    note = validate_note(raw, _plan(n=2).config, LearnerNote(case_id="c", partition="test"))
    assert note.invalid_score and not note.valid_score and note.reason_code == expected


def test_score_and_choice_validate_domain_and_complete_alternatives() -> None:
    config = _plan(n=2).config.model_copy()
    config.decision_type = "score"
    config.rubric = ["Mal", "Regular", "Bien"]
    raw = _raw(2.0, alternatives=[{"value": 0, "confidence": 0.005}, {"value": 1, "confidence": 0.005}])
    note = validate_note(raw, config, LearnerNote(case_id="c", partition="test"))
    assert note.valid_score and note.decision == 2
    config.decision_type, config.options = "choice", ["a", "b"]
    raw = _raw("a", alternatives=[{"value": "b", "confidence": 0.01}])
    assert validate_note(raw, config, LearnerNote(case_id="c", partition="test")).valid_score


def test_absent_decision_on_timeout_is_abstention_not_invalid_score() -> None:
    raw = _raw(reason_code="TIMEOUT")
    raw["attempts"][0].update(decision=None, confidence=None, alternatives=[])
    note = validate_note(raw, _plan(n=2).config, LearnerNote(case_id="c", partition="test"))
    assert not note.valid_score and not note.invalid_score and note.reason_code == "TIMEOUT"


def test_model_alias_is_echoed_but_wrong_provider_is_invalid() -> None:
    config = _plan(n=2).config
    config.learner = EvaluationTarget(provider="laya_mcp", model="multilingual")
    raw = _raw(provider="laya_mcp", model="convaiinnovations/laya/multilingual")
    note = validate_note(raw, config, LearnerNote(case_id="c", partition="test"))
    assert note.valid_score and note.model == "convaiinnovations/laya/multilingual"
    raw["attempts"][0]["provider"] = "ollama_system1"
    assert validate_note(raw, config, LearnerNote(case_id="c", partition="test")).invalid_score


def test_invalid_instrument_ratio_prevents_calibration() -> None:
    broker = EvaluationBroker()
    broker.invalid_student = True
    report = _run(broker, _plan(n=6))
    assert report.status == "no_calibrado" and report.metrics["invalid_score_ratio"] == 1.0


def test_model_change_midrun_invalidates_the_run() -> None:
    broker = EvaluationBroker()
    broker.student_model_change = True
    report = _run(broker, _plan(n=6))
    assert report.recommended_threshold is None
    assert any("cambió durante" in reason for reason in report.limitations)


def test_confidential_cases_never_authorize_cloud_teacher_or_student() -> None:
    plan = _plan(n=4, cloud_allowed_teacher=True, cloud_allowed_learner=True)
    for case in plan.cases:
        case.data_class = "local_only"
    broker = EvaluationBroker()
    _run(broker, plan)
    assert all(payload["model_requirements"]["cloud_allowed"] is False for payload in broker.requests)
    assert all(payload["cloud_allowed"] is False for payload in broker.judgments)


def test_teacher_self_reported_confidence_is_ignored() -> None:
    plan = _plan(teacher_task=None, teacher_judge={"provider": "ollama_system1", "model": "qwen3:8b"})
    report = _run(EvaluationBroker(), plan)
    assert report.teacher_qualified and all(label.unanimous for label in report.labels.values())
    assert report.teacher_kappa == 1.0
    assert all("confidence" not in entry for label in report.labels.values() for entry in label.telemetry)


def test_cold_teacher_timeout_is_retried_and_never_becomes_a_label() -> None:
    broker = EvaluationBroker()
    broker.teacher_judge_timeout = True
    report = _run(
        broker, _plan(n=4, teacher_task=None, teacher_judge={"provider": "ollama_system1", "model": "qwen3:8b"})
    )
    assert report.warmup["timeouts"] == 1
    assert all(label.unanimous for label in report.labels.values())
    assert all(len(label.decisions) == 3 for label in report.labels.values())


def test_persistence_exports_provenance_reports_and_negative_control(tmp_path: Path) -> None:
    plan = _plan()
    with Database(tmp_path / "md.sqlite") as db:
        first = _run(EvaluationBroker(), plan, db)
        second = _run(EvaluationBroker(), plan, db)
        assert not second.previous_comparison["calibration_regression"]
        path = export_distilled(db, second, tmp_path / "references.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["references"][0]["label"]["source"] == "distilled_system2"
        assert data["references"][0]["case"]["human_verified"]
        assert len(db.query("SELECT * FROM calibration_case")) == 160
        assert db.query_one("SELECT expired FROM calibration_run WHERE run_id = ?", (first.run_id,))["expired"] == 0
    html = write_calibration(second, tmp_path / "report.html").read_text(encoding="utf-8")
    assert "Fiabilidad" in html and "distilled_system2" in html and "Clopper" not in html
    assert "Calibrado" in render_calibration_summary(second)


def test_teacher_never_sees_student_response_or_confidence() -> None:
    broker = EvaluationBroker()
    _run(broker, _plan(n=4))
    for payload in broker.requests:
        prompt = payload["content"]["prompt"]
        parsed = json.loads(prompt[prompt.index("{") :])
        assert set(parsed["input"]) == {"case_id", "truth", "acceptance_criteria"}
        assert "confidence" not in parsed and "human_label" not in parsed


def test_broker_offline_is_explicit_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = EvaluationBroker()

    def offline(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    monkeypatch.setattr(broker, "handler", offline)
    report = _run(broker, _plan(n=4))
    assert report.status == "no_calibrado" and report.recommended_threshold is None
    assert any("Tirada fallida" in reason for reason in report.reasons)


def test_cli_probar_system1_uses_the_same_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _plan(n=4)
    (tmp_path / "plan.json").write_text(plan.model_dump_json(), encoding="utf-8")
    config = AppConfig(database=str(tmp_path / "md.sqlite"), reports_dir=str(tmp_path / "reports"))
    config.save(tmp_path / "config.yaml")
    monkeypatch.setattr(cli, "_client", lambda config: EvaluationBroker().client())
    result = cli.main(
        ["--config", str(tmp_path / "config.yaml"), "probar-system1", "--plan", str(tmp_path / "plan.json")]
    )
    assert result == 0
    artifacts = list((tmp_path / "reports").glob("*.json"))
    assert len(artifacts) == 1 and json.loads(artifacts[0].read_text())["status"] == "indeterminado"


def test_benchmark_accepts_only_qualified_distilled_references() -> None:
    base, cand, suite = _pair()
    client, _ = _client()
    report = compare_runs(base, cand, suite=suite, system1=System1Config(enabled=True), broker=client)
    labels = {"c": ReferenceLabel(decision="equivalent", source="distilled_system2")}
    scored = score_labels(report.semantic_review, labels)
    assert scored.excluded_references == 1 and scored.classified_pairs == 0
    labels["c"].teacher_qualified = labels["c"].unanimous = True
    labels["c"].teacher_fingerprint = {"digest": "frozen"}
    scored = score_labels(report.semantic_review, labels)
    assert scored.correct_labels == 1 and scored.reference_sources == {"distilled_system2": 1}


def test_fingerprint_change_expires_artifacts_and_automatically_repeats_plan(tmp_path: Path) -> None:
    plan = _plan()
    broker = EvaluationBroker()
    with Database(tmp_path / "md.sqlite") as db, broker.client() as client:
        first = run_calibration(client, plan, db)
        artifact, html = tmp_path / "first.json", tmp_path / "first.html"
        save_artifacts(db, first, artifact, html)
        assert revalidate_changed(client, db, tmp_path) == []
        for entry in broker.catalog:
            if entry["model"] == "native":
                entry["execution_fingerprint"]["components"]["digest"]["value"] = "sha256:updated"
        client.catalog(refresh=True)
        reports = revalidate_changed(client, db, tmp_path)
        assert len(reports) == 1 and reports[0].learner_fingerprint != first.learner_fingerprint
        assert db.query_one("SELECT expired FROM calibration_run WHERE run_id = ?", (first.run_id,))["expired"] == 1
        assert json.loads(artifact.read_text(encoding="utf-8"))["expired"] is True
        assert "Caducado" in html.read_text(encoding="utf-8")
        assert reports[0].previous_comparison["previous_artifact_expired"]
        assert revalidate_changed(client, db, tmp_path) == []


def test_teacher_timeout_exhaustion_fails_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = EvaluationBroker()
    handler = broker.handler

    def timeout(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/tasks" and request.method == "POST":
            payload = json.loads(request.content)
            if payload.get("group") == "model-drift-calibration-teacher":
                broker.fail_prompts[payload["content"]["prompt"]] = "TIMEOUT"
        return handler(request)

    monkeypatch.setattr(broker, "handler", timeout)
    report = _run(broker, _plan(n=4, teacher_warmup=False, teacher_timeout_retries=2))
    assert report.status == "no_calibrado" and report.recommended_threshold is None
    assert any("TIMEOUT" in reason for reason in report.reasons)
    assert report.notes == [] and report.labels["c000"].decisions == []
    assert report.labels["c000"].timeouts == 3


def test_accepted_flag_and_top_level_echo_are_validated() -> None:
    config = _plan(n=2).config
    raw = _raw()
    raw["accepted"] = "true"
    note = validate_note(raw, config, LearnerNote(case_id="c", partition="test"))
    assert note.invalid_score and note.reason_code == "INVALID_ACCEPTED_FLAG"
    raw = _raw()
    raw["decision"] = False
    note = validate_note(raw, config, LearnerNote(case_id="c", partition="test"))
    assert note.invalid_score and note.reason_code == "ACCEPTED_RESPONSE_MISMATCH"
    raw = _raw(reason_code="TIMEOUT")
    note = validate_note(raw, config, LearnerNote(case_id="c", partition="test"))
    assert note.invalid_score and note.reason_code == "SCORE_ON_FAILED_ATTEMPT"


def test_reanalysis_clears_threshold_when_teacher_is_not_qualified() -> None:
    plan = _plan()
    report = _run(EvaluationBroker(), plan)
    assert report.recommended_threshold is not None
    for case in plan.cases:
        case.human_label = True
    analyze_calibration(report, plan)
    assert not report.teacher_qualified and report.recommended_threshold is None


def test_changed_reference_is_excluded_from_paired_drift() -> None:
    first = _run(EvaluationBroker(), _plan())
    second = first.model_copy(deep=True)
    note = next(note for note in second.notes if note.partition == "test")
    note.reference = not note.reference
    comparison = compare_calibrations(first, second)
    assert comparison["reference_disagreements_excluded"] == 1
    assert not comparison["calibration_regression"]


def test_second_teacher_disagreement_requires_human_review() -> None:
    broker = EvaluationBroker()
    broker.add_model("ollama/local/second")

    def disagree(prompt: str, model: str, seen: int) -> str:
        payload = json.loads(prompt[prompt.index("{") :])
        return json.dumps({"decision": not payload["input"]["truth"]})

    broker.respond("ollama/local/second", disagree)
    report = _run(
        broker, _plan(n=4, second_teacher_task={"provider": "ollama", "deployment": "local", "model": "second"})
    )
    assert report.ambiguous_cases == 4 and not report.teacher_qualified
    assert all(label.reason_code == "TEACHERS_DISAGREE_REQUIRES_HUMAN_REVIEW" for label in report.labels.values())
    assert all(not note.reference_eligible for note in report.notes)


def test_cancellation_preserves_partial_report() -> None:
    from threading import Event

    event = Event()
    event.set()
    with Database(":memory:") as db, EvaluationBroker().client() as client:
        report = run_calibration(client, _plan(n=4, teacher_warmup=False), db, cancel=event)
        assert report.status == "no_calibrado" and any("CANCELLED" in reason for reason in report.reasons)
        saved = db.query_one("SELECT status FROM calibration_run WHERE run_id = ?", (report.run_id,))
        assert saved["status"] == "no_calibrado"


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"score_source": "self_reported"}, "NON_NATIVE_SCORE"),
        ({"alternatives": None}, "MISSING_ALTERNATIVES"),
        ({"alternatives": ["false"]}, "INVALID_ALTERNATIVE"),
        ({"alternatives": [{"value": False, "confidence": 0.2}]}, "PROBABILITY_SUM_MISMATCH"),
        ({"model": None}, "MODEL_IDENTITY_MISMATCH"),
    ],
)
def test_unusable_native_instruments_are_excluded(updates: dict[str, Any], reason: str) -> None:
    raw = _raw(**updates)
    note = validate_note(raw, _plan(n=2).config, LearnerNote(case_id="c", partition="test"))
    assert note.invalid_score and not note.valid_score and note.reason_code == reason


def test_absent_or_extra_attempts_cannot_contribute_a_score() -> None:
    raw = _raw(accepted=False)
    raw["attempts"] = []
    raw["reason_code"] = "UNKNOWN_USE_CASE"
    note = validate_note(raw, _plan(n=2).config, LearnerNote(case_id="c", partition="test"))
    assert not note.invalid_score and not note.valid_score and note.reason_code == "UNKNOWN_USE_CASE"
    raw["attempts"] = [{}, {}]
    note = validate_note(raw, _plan(n=2).config, LearnerNote(case_id="c", partition="test"))
    assert note.invalid_score and note.reason_code == "INVALID_ATTEMPT_COUNT"


def test_teacher_judge_requires_evaluation_capability() -> None:
    report = _run(
        EvaluationBroker(evaluation=False),
        _plan(n=4, teacher_task=None, teacher_judge={"provider": "ollama_system1", "model": "qwen3:8b"}),
    )
    assert any("EVALUATION_UNAVAILABLE" in reason for reason in report.reasons)
    assert report.notes == [] and report.recommended_threshold is None


def test_unverified_student_fingerprint_cannot_certify_calibration() -> None:
    broker = EvaluationBroker()
    for entry in broker.catalog:
        if entry["model"] == "native":
            entry["execution_fingerprint"] = {}
    report = _run(broker, _plan())
    assert report.status == "no_calibrado" and report.recommended_threshold is None
    assert any("huella del alumno" in limitation for limitation in report.limitations)


def test_human_class_imbalance_blocks_teacher_qualification() -> None:
    plan = _plan()
    for case in plan.cases[:-1]:
        case.human_label = True
    report = _run(EvaluationBroker(), plan)
    assert not report.teacher_qualified and report.recommended_threshold is None
    assert any("desequilibrado" in reason for reason in report.reasons)


def test_cli_revalidation_and_teacher_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _plan()
    config = AppConfig(database=str(tmp_path / "md.sqlite"), reports_dir=str(tmp_path / "reports"))
    config_path = tmp_path / "config.yaml"
    config.save(config_path)
    broker = EvaluationBroker()
    monkeypatch.setattr(cli, "_client", lambda config: broker.client())
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.model_dump_json(), encoding="utf-8")
    argv = ["--config", str(config_path)]
    assert cli.main(argv + [
        "probar-system1", "--plan", str(plan_path), "--maestro", "ollama/local/qwen3:8b",
        "--exportar-referencias", str(tmp_path / "references.json"),
    ]) == 0
    assert cli.main(argv + ["revalidar-system1"]) == 0
    for entry in broker.catalog:
        if entry["model"] == "native":
            entry["execution_fingerprint"]["components"]["digest"]["value"] = "sha256:changed"
    assert cli.main(argv + ["revalidar-system1"]) == 0
    assert len(list((tmp_path / "reports").glob("*.html"))) == 2


def test_benchmark_cli_reads_audits_and_rejects_invalid_references(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    from model_drift.system1_benchmark import main as benchmark_main

    base, candidate, suite = _pair()
    client, _ = _client()
    comparison = compare_runs(base, candidate, suite=suite, system1=System1Config(enabled=True), broker=client)
    database, labels_path = tmp_path / "md.sqlite", tmp_path / "labels.json"
    with Database(database) as db:
        db.insert("comparison_review", {
            "review_id": "audit", "created_at": "2026-10-03", "baseline_run_id": "old",
            "candidate_run_id": "new", "baseline_model_key": "old", "candidate_model_key": "new",
            "review_json": comparison.semantic_review.model_dump_json(),
        })
    argv = ["--database", str(database), "--review-id", "audit", "--labels", str(labels_path)]
    labels_path.write_text(json.dumps({"c": "equivalent"}), encoding="utf-8")
    assert benchmark_main(argv) == 0
    assert json.loads(capsys.readouterr().out)["correct_labels"] == 1
    invalid_references = (
        [], {"c": "invented"}, {"unknown": "equivalent"},
        {"c": {"decision": "equivalent", "source": "invented"}},
    )
    for invalid in invalid_references:
        labels_path.write_text(json.dumps(invalid), encoding="utf-8")
        with pytest.raises(SystemExit) as error:
            benchmark_main(argv)
        assert error.value.code == 2
    with pytest.raises(SystemExit):
        benchmark_main([
            "--database", str(tmp_path / "absent.sqlite"), "--review-id", "audit", "--labels", str(labels_path),
        ])
    with pytest.raises(SystemExit):
        benchmark_main(["--database", str(database), "--review-id", "missing", "--labels", str(labels_path)])


def test_false_approvals_detect_precision_regression_even_when_correct_approvals_are_unchanged() -> None:
    plan = _plan()
    with Database(":memory:") as db:
        first = _run(EvaluationBroker(), plan, db)
        broker = EvaluationBroker()
        broker.student_false_approvals = True
        second = _run(broker, plan, db)
        comparison = second.previous_comparison
        assert comparison["calibration_regression"]
        assert comparison["current_precision_at_threshold"] < comparison["previous_precision_at_threshold"]
        assert comparison["contrast"]["p_value"] < plan.config.drift_alpha
        assert comparison["previous_artifact_expired"]
        assert db.query_one("SELECT expired FROM calibration_run WHERE run_id = ?", (first.run_id,))["expired"] == 1


def test_changed_model_expires_every_older_certificate_for_that_model() -> None:
    plan = _plan()
    with Database(":memory:") as db:
        first = _run(EvaluationBroker(), plan, db)
        second = _run(EvaluationBroker(), plan, db)
        changed = EvaluationBroker()
        for entry in changed.catalog:
            if entry["model"] == "native":
                entry["execution_fingerprint"]["components"]["digest"]["value"] = "sha256:new"
        _run(changed, plan, db)
        for report in (first, second):
            saved = db.query_one("SELECT expired FROM calibration_run WHERE run_id = ?", (report.run_id,))
            assert saved["expired"] == 1


def test_observed_teacher_change_expires_old_certificate_even_if_new_run_fails() -> None:
    plan = _plan(teacher_warmup=False)
    broker = EvaluationBroker()
    with Database(":memory:") as db:
        first = _run(broker, plan, db)
        with broker.client() as client:
            invoke = client.invoke
            calls = 0

            def changed(**kwargs: Any):  # type: ignore[no-untyped-def]
                nonlocal calls
                outcome = invoke(**kwargs)
                calls += 1
                if calls == 2:
                    outcome.fingerprint.components["digest"].value = "sha256:teacher-changed"
                return outcome

            client.invoke = changed  # type: ignore[method-assign]
            second = run_calibration(client, plan, db)
        assert second.status == "no_calibrado" and second.recommended_threshold is None
        assert any("TEACHER_FINGERPRINT_CHANGED" in reason for reason in second.reasons)
        assert second.notes == []
        assert db.query_one("SELECT expired FROM calibration_run WHERE run_id = ?", (first.run_id,))["expired"] == 1


def test_new_fingerprint_information_is_not_a_model_change() -> None:
    before = {"components": {"digest": {"value": "same", "status": "verificado"}}}
    after = {"components": {
        "digest": {"value": "same", "state": "verificado"},
        "runtime": {"value": "newly-reported", "status": "verificado"},
    }}
    assert not fingerprint_changed(before, after)
    assert not fingerprint_changed(before, {"components": []})
    after["components"]["digest"]["value"] = "different"
    assert fingerprint_changed(before, after)


def test_generative_primary_and_normal_second_teacher_can_qualify_together() -> None:
    broker = EvaluationBroker()
    broker.add_model("ollama/local/second")

    def agree(prompt: str, model: str, seen: int) -> str:
        payload = json.loads(prompt[prompt.index("{") :])
        return json.dumps({"decision": payload["input"]["truth"]})

    broker.respond("ollama/local/second", agree)
    plan = _plan(
        teacher_task=None, teacher_judge={"provider": "ollama_system1", "model": "qwen3:8b"},
        second_teacher_task={"provider": "ollama", "deployment": "local", "model": "second"},
    )
    report = _run(broker, plan)
    assert report.status == "calibrado", (report.reasons, report.limitations)
    assert report.teacher_qualified and len(report.teacher_fingerprints) == 2


def test_cancelled_teacher_stops_repetitions_and_preserves_cases_and_raw_output() -> None:
    from threading import Event

    event, broker = Event(), EvaluationBroker()
    response = broker.teacher_response

    def stop_after_first(prompt: str, model: str, seen: int) -> str:
        event.set()
        return response(prompt, model, seen)

    broker.respond("ollama/local/qwen3:8b", stop_after_first)
    plan = _plan(n=4, teacher_warmup=False)
    with Database(":memory:") as db, broker.client() as client:
        report = run_calibration(client, plan, db, cancel=event)
        assert report.status == "no_calibrado" and any("CANCELLED" in reason for reason in report.reasons)
        assert len(broker.requests) == 1 and broker.judgments == []
        rows = db.query("SELECT case_json, teacher_json FROM calibration_case WHERE run_id = ?", (report.run_id,))
        assert len(rows) == 4
        label = json.loads(rows[0]["teacher_json"])
        assert label["reason_code"] == "CANCELLED"
        assert json.loads(label["telemetry"][0]["output_text"]) == {"decision": True}


def test_precision_ruled_out_by_every_candidate_interval_is_not_calibrated() -> None:
    from model_drift.calibration.models import label_key

    plan = _plan()
    report = _run(EvaluationBroker(), plan)
    for note in report.notes:
        note.decision = not note.reference
        note.probabilities = {label_key(note.decision): 0.99, label_key(note.reference): 0.01}
    analyze_calibration(report, plan)
    assert report.status == "no_calibrado" and report.recommended_threshold is None


def test_second_teacher_timeout_is_preserved_and_cannot_become_disagreement(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = EvaluationBroker()
    broker.add_model("ollama/local/second")
    handler = broker.handler

    def timeout_second(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/tasks" and request.method == "POST":
            payload = json.loads(request.content)
            if payload["model_requirements"]["target_model"]["model"] == "second":
                broker.fail_prompts[payload["content"]["prompt"]] = "TIMEOUT"
        return handler(request)

    monkeypatch.setattr(broker, "handler", timeout_second)
    plan = _plan(
        n=4, teacher_warmup=False,
        second_teacher_task={"provider": "ollama", "deployment": "local", "model": "second"},
    )
    with Database(":memory:") as db:
        report = _run(broker, plan, db)
        assert report.status == "no_calibrado" and any("TIMEOUT" in reason for reason in report.reasons)
        label = json.loads(db.query_one(
            "SELECT teacher_json FROM calibration_case WHERE run_id = ? AND case_id = ?", (report.run_id, "c000"),
        )["teacher_json"])
        assert label["reason_code"] == "TIMEOUT" and label["timeouts"] == 2
        assert not label["unanimous"] and report.notes == []


def test_authored_semantic_corpus_matches_production_contract_and_has_no_human_claims() -> None:
    from model_drift.system1 import CRITERIA, INSTRUCTIONS
    from model_drift.system1_contracts import SEMANTIC_OPTIONS

    directory = Path(__file__).resolve().parents[1] / "suites" / "system1"
    plan = CalibrationPlan.load(directory / "semantic-synthetic.json")
    assert plan.config.use_case == System1Config().use_case
    assert plan.config.threshold_profile == System1Config().threshold_profile
    assert plan.config.instructions == INSTRUCTIONS and plan.config.criteria == CRITERIA
    assert plan.config.options == list(SEMANTIC_OPTIONS)
    assert len(plan.cases) == 80
    assert Counter(case.expected_label for case in plan.cases) == dict.fromkeys(SEMANTIC_OPTIONS, 20)
    assert sum(case.adversarial for case in plan.cases) == 7
    assert Counter(partition_cases(plan).values()) == {"calibration": 48, "test": 32}
    labels = json.loads((directory / "semantic-synthetic.labels.json").read_text(encoding="utf-8"))
    for case in plan.cases:
        assert case.expected_label_source == "synthetic_authored"
        assert case.human_label is None and not case.human_verified and case.local_only
        assert "expected_label" not in case.input and "human_label" not in case.input
        assert labels[case.id] == {"decision": case.expected_label, "source": "synthetic_authored"}


@pytest.mark.parametrize("label,source", [(True, None), (True, "human"), ("equivalent", "synthetic_authored")])
def test_synthetic_reference_requires_explicit_source_and_correct_domain(label: Any, source: str | None) -> None:
    data = _plan(n=4).model_dump(mode="json")
    data["cases"][0].update(expected_label=label, expected_label_source=source)
    with pytest.raises(ValueError):
        CalibrationPlan.model_validate(data)


@pytest.mark.parametrize("rejected,wrong", [(False, False), (True, False), (False, True)])
def test_synthetic_references_produce_diagnostics_without_certifying_teacher(
    rejected: bool, wrong: bool, tmp_path: Path,
) -> None:
    plan = _plan()
    for case in plan.cases:
        case.expected_label = case.human_label
        case.expected_label_source = "synthetic_authored"
        case.human_label, case.human_verified = None, False
    broker = EvaluationBroker()
    broker.student_rejected, broker.student_false_approvals = rejected, wrong
    report = _run(broker, plan)
    assert report.status == "indeterminado" and report.recommended_threshold is None
    assert report.human_cases == 0 and not report.teacher_qualified and report.teacher_kappa is None
    assert all(not note.reference_eligible for note in report.notes)
    assert report.metrics["raw_test"]["n"] == 0
    diagnostic = report.metrics["synthetic_diagnostics"]
    assert diagnostic["source"] == "synthetic_authored" and diagnostic["expected_cases"] == 80
    assert diagnostic["eligible_for_threshold_recommendation"] is False
    assert diagnostic["teacher_agreement_kappa"] == 1.0
    test_notes = [note for note in report.notes if note.partition == "test"]
    correct = sum(note.decision == note.reference for note in test_notes)
    assert diagnostic["raw_test"]["accuracy"] == correct / len(test_notes)
    assert diagnostic["accepted_test"]["n"] == (0 if rejected else len(test_notes))
    assert diagnostic["test_reliability"]["n"] == len(test_notes)
    assert diagnostic["expected_labels"]["c001"] is False
    assert report.metrics["provenance"] == {"distilled_system2": 80, "human_verified": 0, "synthetic_authored": 80}
    assert all("expected_label" not in request["content"]["prompt"] for request in broker.requests)
    assert all("expected_label" not in payload["input"] for payload in broker.judgments)
    html = write_calibration(report, tmp_path / "synthetic.html").read_text(encoding="utf-8")
    assert "Diagnóstico exploratorio con etiquetas sintéticas" in html
    assert "no valida el maestro frente a humanos" in render_calibration_summary(report)


def test_one_class_synthetic_reference_cannot_manufacture_teacher_kappa() -> None:
    plan = _plan(n=4)
    for case in plan.cases:
        case.expected_label, case.expected_label_source = True, "synthetic_authored"
        case.human_verified = False
    report = _run(EvaluationBroker(), plan)
    assert report.metrics["synthetic_diagnostics"]["teacher_agreement_kappa"] is None
    assert not report.teacher_qualified and report.recommended_threshold is None


def test_benchmark_preserves_synthetic_provenance() -> None:
    base, cand, suite = _pair()
    client, _ = _client()
    report = compare_runs(base, cand, suite=suite, system1=System1Config(enabled=True), broker=client)
    labels = {"c": ReferenceLabel(decision="equivalent", source="synthetic_authored")}
    scored = score_labels(report.semantic_review, labels)
    assert scored.correct_labels == 1 and scored.reference_sources == {"synthetic_authored": 1}
    assert scored.excluded_references == 0
