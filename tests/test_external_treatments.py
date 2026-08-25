from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from model_drift.cli import main
from model_drift.external_treatments import ExternalTreatmentError, load_external_treatments

METRICS = {
    "recall_at_k": 1.0,
    "precision_at_k": 0.5,
    "hit_rate": 1.0,
    "reciprocal_rank": 1.0,
    "ndcg_at_k": 0.8,
    "source_coverage": 1.0,
    "redundancy": 0.0,
}


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _artifact(
    root: Path,
    strategy: str,
    *,
    suite: str,
    snapshot: str,
    metric_delta: float = 0.0,
) -> tuple[Path, str]:
    cases = []
    for index in range(1, 4):
        metrics = dict(METRICS)
        metrics["recall_at_k"] = max(0.0, min(1.0, metrics["recall_at_k"] + metric_delta))
        cases.append(
            {
                "case_id": f"case-{index}",
                "kinds": ["fact"],
                "query": f"query {index}",
                "relevant_paths": ["doc.md"],
                "retrieved": [
                    {
                        "rank": 1,
                        "relative_path": f"{strategy.lower()}.md",
                        "chunk_id": f"chunk-{index}",
                        "score": 1.0,
                        "evidence_reference": f"snapshot:sha256:{snapshot}#chunk:chunk-{index}",
                    }
                ],
                "metrics": metrics,
            }
        )
    report = {
        "schema_version": "retrieval-benchmark-report.v1",
        "suite_id": "controlled",
        "suite_fingerprint": suite,
        "human_review_status": "approved",
        "strategy_id": f"{strategy}.test.v1",
        "k": 5,
        "metric_classification": "deterministic",
        "generated_at": "2026-08-24T00:00:00Z",
        "cases": cases,
        "aggregate": {
            name: sum(case["metrics"][name] for case in cases) / len(cases)
            for name in METRICS
        },
    }
    report["content_sha256"] = hashlib.sha256(_canonical(report).encode()).hexdigest()
    report.update(
        {
            "latency_ms": 10.0,
            "snapshot_hash": snapshot,
            "embedding_provider": "test",
            "embedding_model_fingerprint": "e" * 64,
        }
    )
    path = root / f"{strategy}.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("report/retrieval-report.json", _canonical(report) + "\n")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _plan(tmp_path: Path) -> Path:
    suite = "a" * 64
    snapshot = "b" * 64
    r3, r3_hash = _artifact(tmp_path, "R3", suite=suite, snapshot=snapshot, metric_delta=-0.5)
    r4, r4_hash = _artifact(tmp_path, "R4", suite=suite, snapshot=snapshot)
    payload = {
        "schema_version": "local-ai-lab.model-drift-plan.v2",
        "experiment_id": "comparison-1",
        "correlation_id": "correlation-1",
        "suite_fingerprint": suite,
        "snapshot_hash": snapshot,
        "baseline": {
            "experiment_id": "experiment-r3",
            "strategy_id": "R3",
            "artifact_path": str(r3),
            "artifact_sha256": r3_hash,
            "artifact_kind": "retrieval_benchmark_report",
        },
        "candidate": {
            "experiment_id": "experiment-r4",
            "strategy_id": "R4",
            "artifact_path": str(r4),
            "artifact_sha256": r4_hash,
            "artifact_kind": "retrieval_benchmark_report",
        },
        "fallback_allowed": False,
        "exclude_from_model_learning": True,
        "created_at": "2026-08-24T00:00:00Z",
    }
    payload["content_sha256"] = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    path = tmp_path / "plan.json"
    path.write_text(_canonical(payload), encoding="utf-8")
    return path


def _strategy_artifact(root: Path, strategy: str, *, suite: str, snapshot: str) -> tuple[Path, str]:
    report = {
        "schema_version": "strategy-suite-report.v1",
        "strategy_id": strategy,
        "suite_fingerprint": suite,
        "snapshot_hash": snapshot,
        "case_ids": ["case-1", "case-2"],
        "target_model": {"provider": "ollama", "deployment": "local", "model": "exact"},
        "k": 5,
        "quality": {"deterministic_pass_rate": 1.0},
        "retrieval": {},
        "latency_ms": 20.0,
        "privacy": "local_only",
        "formal_status": "unverified",
        "results": [
            {
                "case_id": case_id,
                "response": {"answer": f"{strategy}-{case_id}"},
                "raw_response": None,
                "verification": {"deterministic_pass": strategy == "R4"},
                "latency_ms": 10.0,
                "model_used": {"provider": "ollama", "deployment": "local", "model": "exact"},
                "fallback_used": False,
                "cost": {"amount": "0.001", "currency": "USD", "verification_status": "verified"},
                "context_sha256": "c" * 64,
            }
            for case_id in ("case-1", "case-2")
        ],
    }
    path = root / f"strategy-{strategy}.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("strategy-report.json", _canonical(report) + "\n")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _strategy_plan(tmp_path: Path) -> Path:
    suite = "a" * 64
    snapshot = "b" * 64
    r3, r3_hash = _strategy_artifact(tmp_path, "R3", suite=suite, snapshot=snapshot)
    r4, r4_hash = _strategy_artifact(tmp_path, "R4", suite=suite, snapshot=snapshot)
    payload = {
        "schema_version": "local-ai-lab.model-drift-plan.v2",
        "experiment_id": "comparison-strategy",
        "correlation_id": "correlation-strategy",
        "suite_fingerprint": suite,
        "snapshot_hash": snapshot,
        "baseline": {
            "experiment_id": "strategy-r3",
            "strategy_id": "R3",
            "artifact_path": str(r3),
            "artifact_sha256": r3_hash,
            "artifact_kind": "strategy_suite_report",
        },
        "candidate": {
            "experiment_id": "strategy-r4",
            "strategy_id": "R4",
            "artifact_path": str(r4),
            "artifact_sha256": r4_hash,
            "artifact_kind": "strategy_suite_report",
        },
        "fallback_allowed": False,
        "exclude_from_model_learning": True,
        "created_at": "2026-08-24T00:00:00Z",
    }
    payload["content_sha256"] = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    path = tmp_path / "strategy-plan.json"
    path.write_text(_canonical(payload), encoding="utf-8")
    return path


def test_external_treatments_are_verified_and_imported(tmp_path: Path) -> None:
    _, baseline, candidate = load_external_treatments(_plan(tmp_path))

    assert baseline.model_key == "R3"
    assert candidate.model_key == "R4"
    assert len(baseline.outcomes) == 21
    assert all(item.measurement_deterministic for item in candidate.outcomes.values())
    assert baseline.manifest.instrument_signature() == candidate.manifest.instrument_signature()


def test_external_treatment_cli_generates_html_without_a_database(tmp_path: Path) -> None:
    report = tmp_path / "comparison.html"

    result = main(["evaluar-tratamientos", "--plan", str(_plan(tmp_path)), "--informe", str(report)])

    assert result == 0
    assert report.is_file()
    assert "Mismo instrumento" in report.read_text(encoding="utf-8")


def test_completed_strategy_treatments_are_imported_without_reinvoking_broker(tmp_path: Path) -> None:
    _, baseline, candidate = load_external_treatments(_strategy_plan(tmp_path))

    assert baseline.manifest.model_by_key("R3").model == "exact"  # type: ignore[union-attr]
    assert candidate.outcomes[("case-1", "deterministic_pass")].case_score == 1.0
    assert candidate.outcomes[("case-1", "latency_ms")].measurement_deterministic is False
    assert candidate.outcomes[("case-1", "cost_usd")].case_score == 0.001


def test_external_treatment_rejects_artifact_tampering(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    payload = json.loads(plan.read_text(encoding="utf-8"))
    Path(payload["baseline"]["artifact_path"]).write_bytes(b"tampered")

    with pytest.raises(ExternalTreatmentError, match="artifact SHA-256 mismatch"):
        load_external_treatments(plan)


def test_external_treatment_rejects_plan_tampering(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    payload = json.loads(plan.read_text(encoding="utf-8"))
    payload["snapshot_hash"] = "c" * 64
    plan.write_text(_canonical(payload), encoding="utf-8")

    with pytest.raises(ExternalTreatmentError, match="plan SHA-256 mismatch"):
        load_external_treatments(plan)
