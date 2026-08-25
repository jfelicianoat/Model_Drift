"""Contrato público para comparar tratamientos calculados fuera de Model Drift.

El importador no conoce la base de datos de Local AI Lab. Recibe un plan
sellado y dos artefactos ZIP inmutables, verifica todas sus huellas y los
convierte al mismo modelo de resultados que usan las tiradas propias.
"""
from __future__ import annotations

import hashlib
import json
import math
import zipfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from model_drift.comparison import RunResults
from model_drift.domain import METRIC_COST, METRIC_LATENCY, RETRIEVAL_METRICS, CaseOutcome
from model_drift.manifest import InstrumentRef, ModelUnderTest, RunManifest

PLAN_SCHEMA = "local-ai-lab.model-drift-plan.v2"
REPORT_SCHEMA = "retrieval-benchmark-report.v1"
STRATEGY_REPORT_SCHEMA = "strategy-suite-report.v1"
MAX_ARTIFACT_BYTES = 100 * 1024 * 1024
MAX_REPORT_BYTES = 50 * 1024 * 1024


class ExternalTreatmentError(ValueError):
    """El plan o uno de sus artefactos no es íntegro/comparable."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TreatmentRef(StrictModel):
    experiment_id: str = Field(min_length=1, max_length=256)
    strategy_id: Literal["R3", "R4"]
    artifact_path: str = Field(min_length=1, max_length=4096)
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_kind: Literal["retrieval_benchmark_report", "strategy_suite_report"]


class ExternalTreatmentPlan(StrictModel):
    schema_version: Literal["local-ai-lab.model-drift-plan.v2"]
    experiment_id: str = Field(min_length=1, max_length=256)
    correlation_id: str = Field(min_length=1, max_length=256)
    suite_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline: TreatmentRef
    candidate: TreatmentRef
    fallback_allowed: Literal[False] = False
    exclude_from_model_learning: Literal[True] = True
    created_at: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def treatment_order_is_fixed(self) -> ExternalTreatmentPlan:
        if self.baseline.strategy_id != "R3" or self.candidate.strategy_id != "R4":
            raise ValueError("external comparison requires R3 as baseline and R4 as candidate")
        if self.baseline.experiment_id == self.candidate.experiment_id:
            raise ValueError("baseline and candidate must be different experiments")
        return self


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_external_plan(path: str | Path) -> ExternalTreatmentPlan:
    plan_path = Path(path).resolve(strict=True)
    if not plan_path.is_file() or str(plan_path).startswith(("\\\\", "//")):
        raise ExternalTreatmentError("the external-treatment plan must be a local regular file")
    try:
        raw = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExternalTreatmentError("the external-treatment plan is not valid JSON") from error
    if not isinstance(raw, dict):
        raise ExternalTreatmentError("the external-treatment plan must be a JSON object")
    supplied = raw.get("content_sha256")
    unsigned = {key: value for key, value in raw.items() if key != "content_sha256"}
    expected = hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()
    if supplied != expected:
        raise ExternalTreatmentError("external-treatment plan SHA-256 mismatch")
    try:
        return ExternalTreatmentPlan.model_validate(raw)
    except ValueError as error:
        raise ExternalTreatmentError(f"invalid external-treatment plan: {error}") from error


def _load_report(reference: TreatmentRef) -> dict[str, Any]:
    artifact = Path(reference.artifact_path).resolve(strict=True)
    if not artifact.is_file() or str(artifact).startswith(("\\\\", "//")):
        raise ExternalTreatmentError(f"{reference.strategy_id} artifact must be a local regular file")
    if artifact.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ExternalTreatmentError(f"{reference.strategy_id} artifact exceeds the size limit")
    if _sha256_file(artifact) != reference.artifact_sha256:
        raise ExternalTreatmentError(f"{reference.strategy_id} artifact SHA-256 mismatch")
    try:
        with zipfile.ZipFile(artifact) as archive:
            files = [item for item in archive.infolist() if not item.is_dir()]
            expected_name = (
                "retrieval-report.json"
                if reference.artifact_kind == "retrieval_benchmark_report"
                else "strategy-report.json"
            )
            matches = [item for item in files if Path(item.filename).name == expected_name]
            if len(matches) != 1:
                raise ExternalTreatmentError(
                    f"{reference.strategy_id} artifact must contain exactly one {expected_name}"
                )
            member = matches[0]
            if member.file_size > MAX_REPORT_BYTES or member.compress_size > MAX_REPORT_BYTES:
                raise ExternalTreatmentError(f"{reference.strategy_id} report exceeds the size limit")
            raw = archive.read(member)
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise ExternalTreatmentError(f"{reference.strategy_id} artifact is not a valid ZIP") from error
    try:
        report = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExternalTreatmentError(f"{reference.strategy_id} report is not valid UTF-8 JSON") from error
    expected_schema = (
        REPORT_SCHEMA
        if reference.artifact_kind == "retrieval_benchmark_report"
        else STRATEGY_REPORT_SCHEMA
    )
    if not isinstance(report, dict) or report.get("schema_version") != expected_schema:
        raise ExternalTreatmentError(f"{reference.strategy_id} report schema is not supported")
    if reference.artifact_kind == "retrieval_benchmark_report":
        sealed = {
            key: value
            for key, value in report.items()
            if key
            not in {
                "content_sha256",
                "latency_ms",
                "snapshot_hash",
                "embedding_provider",
                "embedding_model_fingerprint",
            }
        }
        expected_content = hashlib.sha256(_canonical_json(sealed).encode("utf-8")).hexdigest()
        if report.get("content_sha256") != expected_content:
            raise ExternalTreatmentError(f"{reference.strategy_id} report content SHA-256 mismatch")
    return report


def _validated_cases(report: dict[str, Any], reference: TreatmentRef) -> list[dict[str, Any]]:
    cases = report.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ExternalTreatmentError(f"{reference.strategy_id} report has no cases")
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("case_id"), str):
            raise ExternalTreatmentError(f"{reference.strategy_id} contains an invalid case")
        case_id = case["case_id"]
        if not case_id or case_id in seen:
            raise ExternalTreatmentError(f"{reference.strategy_id} contains duplicate/empty case ids")
        metrics = case.get("metrics")
        if not isinstance(metrics, dict) or set(metrics) != set(RETRIEVAL_METRICS):
            raise ExternalTreatmentError(f"{reference.strategy_id}/{case_id} has an invalid metric set")
        for name, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ExternalTreatmentError(f"{reference.strategy_id}/{case_id}/{name} is not finite")
            if not 0.0 <= float(value) <= 1.0:
                raise ExternalTreatmentError(f"{reference.strategy_id}/{case_id}/{name} is outside [0, 1]")
        seen.add(case_id)
        validated.append(case)
    return validated


def _validated_strategy_results(report: dict[str, Any], reference: TreatmentRef) -> list[dict[str, Any]]:
    results = report.get("results")
    if not isinstance(results, list) or not results:
        raise ExternalTreatmentError(f"{reference.strategy_id} strategy report has no results")
    expected_ids = report.get("case_ids")
    if not isinstance(expected_ids, list) or any(not isinstance(item, str) for item in expected_ids):
        raise ExternalTreatmentError(f"{reference.strategy_id} strategy report has invalid case_ids")
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    target_model = report.get("target_model")
    if not isinstance(target_model, dict) or set(target_model) != {"provider", "deployment", "model"}:
        raise ExternalTreatmentError(f"{reference.strategy_id} target model is invalid")
    if report.get("privacy") != "local_only":
        raise ExternalTreatmentError(f"{reference.strategy_id} does not declare local-only privacy")
    for result in results:
        if not isinstance(result, dict) or not isinstance(result.get("case_id"), str):
            raise ExternalTreatmentError(f"{reference.strategy_id} contains an invalid result")
        case_id = result["case_id"]
        if not case_id or case_id in seen:
            raise ExternalTreatmentError(f"{reference.strategy_id} contains duplicate/empty case ids")
        verification = result.get("verification")
        if not isinstance(verification, dict) or not isinstance(
            verification.get("deterministic_pass"), bool
        ):
            raise ExternalTreatmentError(f"{reference.strategy_id}/{case_id} lacks deterministic verification")
        latency = result.get("latency_ms")
        if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(float(latency)):
            raise ExternalTreatmentError(f"{reference.strategy_id}/{case_id} latency is not finite")
        if float(latency) < 0:
            raise ExternalTreatmentError(f"{reference.strategy_id}/{case_id} latency is negative")
        if result.get("fallback_used") is not False:
            raise ExternalTreatmentError(f"{reference.strategy_id}/{case_id} used or did not declare fallback")
        model_used = result.get("model_used")
        if not isinstance(model_used, dict) or any(
            model_used.get(key) != value for key, value in target_model.items()
        ):
            raise ExternalTreatmentError(f"{reference.strategy_id}/{case_id} was served by another model")
        seen.add(case_id)
        validated.append(result)
    if seen != set(expected_ids) or len(expected_ids) != len(seen):
        raise ExternalTreatmentError(f"{reference.strategy_id} case_ids do not match results")
    return validated


def _as_run(
    reference: TreatmentRef,
    report: dict[str, Any],
    *,
    suite_fingerprint: str,
    snapshot_hash: str,
) -> RunResults:
    target = report.get("target_model")
    if isinstance(target, dict) and set(target) == {"provider", "deployment", "model"}:
        provider = str(target["provider"])
        deployment = str(target["deployment"])
        model_name = str(target["model"])
    else:
        provider = "local-ai-lab"
        deployment = "external-treatment"
        model_name = reference.strategy_id
    model = ModelUnderTest(
        key=reference.strategy_id,
        provider=provider,
        deployment=deployment,
        model=model_name,
    )
    schema = str(report["schema_version"])
    instrument_hash = hashlib.sha256(
        "|".join(sorted(RETRIEVAL_METRICS)).encode()
        if reference.artifact_kind == "retrieval_benchmark_report"
        else b"deterministic_pass|latency_ms|cost_usd"
    ).hexdigest()
    manifest = RunManifest(
        run_id=reference.experiment_id,
        suite_id=str(report.get("suite_id") or "local-ai-lab-retrieval"),
        suite_hash=suite_fingerprint,
        models=[model],
        instruments=[
            InstrumentRef(
                id="retrieval-benchmark-metrics",
                kind="evaluator",
                version=schema,
                hash=instrument_hash,
            ),
            InstrumentRef(id="retrieval-depth", kind="parameter", hash=str(report.get("k", 5))),
        ],
        fixtures={"snapshot": snapshot_hash},
        repetitions=1,
        missing_guarantees=(
            [
                "Tratamiento importado: Model Drift no ejecutó ni observó el proceso que produjo el artefacto.",
                "Benchmark determinista sin repeticiones estocásticas; la validez externa depende de la suite.",
            ]
            if reference.artifact_kind == "retrieval_benchmark_report"
            else [
                "Tratamiento importado: Model Drift no ejecutó ni observó el proceso que produjo el artefacto.",
                "Latencia y coste tienen una sola observación por caso; sus diferencias pueden quedar sin potencia.",
            ]
        ),
        notes=[f"Artefacto externo verificado sha256:{reference.artifact_sha256}"],
    )
    run = RunResults(run_id=reference.experiment_id, manifest=manifest, model_key=reference.strategy_id)
    if reference.artifact_kind == "retrieval_benchmark_report":
        for case in _validated_cases(report, reference):
            case_id = case["case_id"]
            for metric, raw_value in case["metrics"].items():
                value = float(raw_value)
                run.outcomes[(case_id, metric)] = CaseOutcome(
                    case_id=case_id,
                    model_key=reference.strategy_id,
                    metric=metric,
                    case_score=value,
                    case_stability=1.0,
                    repetitions=1,
                    values=[value],
                    measurement_deterministic=True,
                )
            retrieved = case.get("retrieved")
            if isinstance(retrieved, list):
                run.sample_output[case_id] = "\n".join(
                    str(item.get("relative_path", ""))
                    for item in retrieved
                    if isinstance(item, dict)
                )
        return run

    for result in _validated_strategy_results(report, reference):
        case_id = result["case_id"]
        passed = 1.0 if result["verification"]["deterministic_pass"] else 0.0
        run.outcomes[(case_id, "deterministic_pass")] = CaseOutcome(
            case_id=case_id,
            model_key=reference.strategy_id,
            metric="deterministic_pass",
            case_score=passed,
            case_stability=1.0,
            repetitions=1,
            values=[passed],
            measurement_deterministic=True,
        )
        latency = float(result["latency_ms"])
        run.outcomes[(case_id, METRIC_LATENCY)] = CaseOutcome(
            case_id=case_id,
            model_key=reference.strategy_id,
            metric=METRIC_LATENCY,
            case_score=latency,
            repetitions=1,
            values=[latency],
        )
        cost = result.get("cost")
        if isinstance(cost, dict) and cost.get("amount") is not None:
            raw_amount = cost["amount"]
            if (
                cost.get("currency") != "USD"
                or cost.get("verification_status") != "verified"
                or isinstance(raw_amount, bool)
                or not isinstance(raw_amount, (int, float, str))
            ):
                raise ExternalTreatmentError(f"{reference.strategy_id}/{case_id} cost is not verified USD")
            try:
                amount = float(raw_amount)
            except ValueError as error:
                raise ExternalTreatmentError(
                    f"{reference.strategy_id}/{case_id} cost is invalid"
                ) from error
            if not math.isfinite(amount) or amount < 0:
                raise ExternalTreatmentError(f"{reference.strategy_id}/{case_id} cost is invalid")
            run.outcomes[(case_id, METRIC_COST)] = CaseOutcome(
                case_id=case_id,
                model_key=reference.strategy_id,
                metric=METRIC_COST,
                case_score=amount,
                repetitions=1,
                values=[amount],
            )
        response = result.get("response")
        run.sample_output[case_id] = (
            _canonical_json(response) if response is not None else str(result.get("raw_response") or "")
        )
    return run


def load_external_treatments(path: str | Path) -> tuple[ExternalTreatmentPlan, RunResults, RunResults]:
    plan = load_external_plan(path)
    baseline_report = _load_report(plan.baseline)
    candidate_report = _load_report(plan.candidate)
    if plan.baseline.artifact_kind != plan.candidate.artifact_kind:
        raise ExternalTreatmentError("R3 and R4 use different external-treatment contracts")
    for reference, report in ((plan.baseline, baseline_report), (plan.candidate, candidate_report)):
        if report.get("suite_fingerprint") != plan.suite_fingerprint:
            raise ExternalTreatmentError(f"{reference.strategy_id} suite fingerprint does not match the plan")
        if report.get("snapshot_hash") != plan.snapshot_hash:
            raise ExternalTreatmentError(f"{reference.strategy_id} snapshot hash does not match the plan")
        if not isinstance(report.get("k"), int) or report["k"] < 1:
            raise ExternalTreatmentError(f"{reference.strategy_id} retrieval depth is invalid")
        strategy = str(report.get("strategy_id", ""))
        if strategy != reference.strategy_id and not strategy.startswith(reference.strategy_id + "."):
            raise ExternalTreatmentError(f"{reference.strategy_id} report strategy does not match the plan")
    if baseline_report["k"] != candidate_report["k"]:
        raise ExternalTreatmentError("R3 and R4 used different retrieval depths")
    if plan.baseline.artifact_kind == "strategy_suite_report":
        if baseline_report.get("target_model") != candidate_report.get("target_model"):
            raise ExternalTreatmentError("R3 and R4 used different target models")
    baseline = _as_run(
        plan.baseline,
        baseline_report,
        suite_fingerprint=plan.suite_fingerprint,
        snapshot_hash=plan.snapshot_hash,
    )
    candidate = _as_run(
        plan.candidate,
        candidate_report,
        suite_fingerprint=plan.suite_fingerprint,
        snapshot_hash=plan.snapshot_hash,
    )
    if {key[0] for key in baseline.outcomes} != {key[0] for key in candidate.outcomes}:
        raise ExternalTreatmentError("R3 and R4 do not contain the same cases")
    return plan, baseline, candidate
