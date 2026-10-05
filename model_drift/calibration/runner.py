"""Orquestación de la prueba: maestro, alumno, análisis y persistencia local."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from typing import Any, Literal
from uuid import uuid4

from model_drift.broker_client import BrokerClient
from model_drift.broker_contracts import BrokerError
from model_drift.calibration.analysis import analyze_calibration, compare_calibrations
from model_drift.calibration.models import CalibrationPlan, CalibrationReport, LearnerNote, label_key
from model_drift.calibration.teacher import distill_case
from model_drift.calibration.validation import validate_note
from model_drift.db import Database


def partition_cases(plan: CalibrationPlan) -> dict[str, Literal["calibration", "test"]]:
    ids = sorted(case.id for case in plan.cases)
    random.Random(plan.config.seed).shuffle(ids)
    cut = max(1, min(len(ids) - 1, int(len(ids) * plan.config.calibration_fraction)))
    return {case_id: "calibration" if index < cut else "test" for index, case_id in enumerate(ids)}


def _catalog_fingerprint(client: BrokerClient, provider: str, model: str, *, refresh: bool = False) -> dict[str, Any]:
    candidates = [
        entry
        for entry in client.catalog(refresh=refresh)
        if entry.model == model
        and entry.provider
        in {
            provider,
            "ollama" if provider == "ollama_system1" else "laya",
        }
    ]
    return candidates[0].execution_fingerprint if len(candidates) == 1 else {}


def verified_model(fingerprint: dict[str, Any]) -> bool:
    components = fingerprint.get("components", {})
    return isinstance(components, dict) and any(
        isinstance(components.get(key), dict)
        and components[key].get("value")
        and (components[key].get("status") or components[key].get("state")) == "verificado"
        for key in ("digest", "system_fingerprint")
    )


def fingerprint_changed(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    """Divergencia observable, sin confundir nueva información con otro modelo."""
    before, after = previous.get("components", {}), current.get("components", {})
    if not isinstance(before, dict) or not isinstance(after, dict):
        return False
    for name in before.keys() & after.keys() - {"requested_params", "applied_params"}:
        old, new = before[name], after[name]
        if not isinstance(old, dict) or not isinstance(new, dict):
            continue
        if (
            old.get("value") is not None
            and new.get("value") is not None
            and (old.get("state") or old.get("status")) == "verificado"
            and (new.get("state") or new.get("status")) == "verificado"
            and old["value"] != new["value"]
        ):
            return True
    return False


def _expire_observed(
    db: Database, report: CalibrationReport, *, learner_fingerprint: dict[str, Any] | None = None,
) -> None:
    """Retira todos los certificados cuyo modelo ya se observó cambiado."""
    rows = db.query("SELECT report_json FROM calibration_run WHERE expired = 0 AND run_id != ?", (report.run_id,))
    learner = report.learner_fingerprint if learner_fingerprint is None else learner_fingerprint
    for row in rows:
        previous = CalibrationReport.model_validate_json(row["report_json"])
        changed = (
            (previous.learner_provider, previous.learner_model) == (report.learner_provider, report.learner_model)
            and fingerprint_changed(previous.learner_fingerprint, learner)
        ) or any(
            fingerprint_changed(previous.teacher_fingerprints[key], report.teacher_fingerprints[key])
            for key in previous.teacher_fingerprints.keys() & report.teacher_fingerprints.keys()
        )
        if changed:
            expire_calibration(db, previous)


def _freeze_direct_teacher(client: BrokerClient, report: CalibrationReport, *, refresh: bool = False) -> None:
    identities = {
        (item.get("provider"), item.get("model"))
        for label in report.labels.values()
        for item in label.telemetry
        if "provider" in item
    }
    if len(identities) != 1:
        raise BrokerError("El maestro generativo cambió o no está identificado", code="TEACHER_IDENTITY_CHANGED")
    provider, model = next(iter(identities))
    if not isinstance(provider, str) or not isinstance(model, str) or not model:
        raise BrokerError("El maestro generativo no está identificado", code="TEACHER_IDENTITY_CHANGED")
    fingerprint = _catalog_fingerprint(client, provider, model, refresh=refresh)
    identity = f"{provider}/{model}"
    old = report.teacher_fingerprints.setdefault(identity, fingerprint)
    if fingerprint_changed(old, fingerprint):
        report.teacher_fingerprints[identity] = fingerprint
        raise BrokerError("Cambió la huella del maestro generativo", code="TEACHER_FINGERPRINT_CHANGED")


def _save(db: Database, report: CalibrationReport) -> None:
    db.insert(
        "calibration_run",
        {
            "run_id": report.run_id,
            "created_at": report.created_at,
            "use_case": report.use_case,
            "status": report.status,
            "expired": report.expired,
            "report_json": report.model_dump_json(),
        },
    )


def run_calibration(
    client: BrokerClient,
    plan: CalibrationPlan,
    db: Database,
    *,
    cancel: Event | None = None,
    progress: Callable[[str], None] | None = None,
) -> CalibrationReport:
    config = plan.config
    report = CalibrationReport(
        run_id=f"cal-{uuid4().hex}",
        created_at=datetime.now(UTC).isoformat(),
        use_case=config.use_case,
        config=config,
        plan_signature=plan.signature(),
        dataset_hash=hashlib.sha256(
            json.dumps(
                [case.model_dump(mode="json") for case in sorted(plan.cases, key=lambda case: case.id)], sort_keys=True
            ).encode()
        ).hexdigest(),
    )
    _save(db, report)
    partitions = partition_cases(plan)
    for case in plan.cases:
        db.insert("calibration_case", {
            "run_id": report.run_id, "case_id": case.id, "partition_name": partitions[case.id],
            "case_json": case.model_dump_json(), "teacher_json": "{}", "learner_json": "{}",
        })
    try:
        if cancel is not None and cancel.is_set():
            raise BrokerError("prueba cancelada", code="CANCELLED")
        capabilities = client.capabilities()
        evaluation = capabilities.get("system1_evaluation") is True
        if not capabilities.get("system1_judgments"):
            raise BrokerError("el Broker no ofrece juicios System 1", code="SYSTEM1_UNAVAILABLE")
        if not evaluation:
            report.limitations.append(
                "Sin system1_evaluation: no se puede fijar alumno ni obtener todas las notas crudas."
            )
        client.probe_contract()
        if (config.teacher_task or config.second_teacher_task) and not client.config.supports_learning_exclusion:
            raise BrokerError(
                "el maestro contaminaría el aprendizaje del Broker", code="LEARNING_EXCLUSION_UNAVAILABLE"
            )
        if config.teacher_judge and not evaluation:
            raise BrokerError("el Broker no puede fijar el maestro generativo", code="EVALUATION_UNAVAILABLE")
        if config.teacher_warmup:
            warm = distill_case(client, config, plan.cases[0], fingerprints={}, repetitions=1, cancel=cancel)
            report.warmup = {
                "latency_ms": warm.latency_ms,
                "cost_usd": warm.cost_usd,
                "timeouts": warm.timeouts,
                "reason_code": warm.reason_code,
                "telemetry": warm.telemetry,
            }
        for index, case in enumerate(sorted(plan.cases, key=lambda item: item.id)):
            if cancel is not None and cancel.is_set():
                raise BrokerError("prueba cancelada", code="CANCELLED")
            if progress:
                progress(f"Caso {index + 1}/{len(plan.cases)}: {case.id}")
            teacher = distill_case(client, config, case, fingerprints=report.teacher_fingerprints, cancel=cancel)
            report.labels[case.id] = teacher
            db.execute(
                "UPDATE calibration_case SET teacher_json = ? WHERE run_id = ? AND case_id = ?",
                (teacher.model_dump_json(), report.run_id, case.id),
            )
            if teacher.reason_code and teacher.reason_code != "AMBIGUOUS_TEACHER":
                raise BrokerError("Maestro sin etiqueta fiable", code=teacher.reason_code)
            if config.second_teacher_task:
                second = distill_case(
                    client, config, case, target=config.second_teacher_task,
                    fingerprints=report.teacher_fingerprints, cancel=cancel,
                )
                if not second.unanimous or label_key(second.decision) != label_key(teacher.decision):
                    teacher.unanimous = False
                    teacher.decision = None
                    teacher.reason_code = "TEACHERS_DISAGREE_REQUIRES_HUMAN_REVIEW"
                teacher.latency_ms += second.latency_ms
                teacher.cost_usd += second.cost_usd
                teacher.tokens_input += second.tokens_input
                teacher.tokens_output += second.tokens_output
                teacher.timeouts += second.timeouts
                teacher.telemetry.extend(second.telemetry)
                if second.reason_code and second.reason_code != "AMBIGUOUS_TEACHER":
                    teacher.reason_code = second.reason_code
                db.execute(
                    "UPDATE calibration_case SET teacher_json = ? WHERE run_id = ? AND case_id = ?",
                    (teacher.model_dump_json(), report.run_id, case.id),
                )
            report.labels[case.id] = teacher
            if teacher.reason_code and teacher.reason_code not in {
                "AMBIGUOUS_TEACHER",
                "TEACHERS_DISAGREE_REQUIRES_HUMAN_REVIEW",
            }:
                raise BrokerError("Maestro sin etiqueta fiable", code=teacher.reason_code)
            if config.teacher_judge:
                _freeze_direct_teacher(client, report)
            if index == 0:
                _expire_observed(db, report)
            payload = config.request(
                case.input, local_only=case.local_only, target=config.learner if evaluation else None
            )
            raw = client.system1_raw(payload, timeout_seconds=config.timeout_seconds)
            note = LearnerNote(
                case_id=case.id, partition=partitions[case.id], adversarial=case.adversarial, raw_response=raw
            )
            if evaluation:
                note = validate_note(raw, config, note)
            else:
                note.accepted = raw.get("accepted") is True
                note.provider = raw.get("provider")
                note.model = raw.get("model")
                note.reason_code = "EVALUATION_UNAVAILABLE"
            if note.reason_code == "MODEL_IDENTITY_MISMATCH":
                report.limitations.append("El proveedor del intento no coincide con target; tirada inválida.")
            if note.provider and note.model:
                if report.learner_model is not None and (report.learner_provider, report.learner_model) != (
                    note.provider,
                    note.model,
                ):
                    report.limitations.append("El proveedor o modelo del alumno cambió durante la tirada.")
                if report.learner_model is None:
                    report.learner_provider, report.learner_model = note.provider, note.model
                    report.learner_fingerprint = _catalog_fingerprint(client, note.provider, note.model)
                    _expire_observed(db, report)
            report.notes.append(note)
            db.insert(
                "calibration_case",
                {
                    "run_id": report.run_id,
                    "case_id": case.id,
                    "partition_name": note.partition,
                    "case_json": case.model_dump_json(),
                    "teacher_json": teacher.model_dump_json(),
                    "learner_json": note.model_dump_json(),
                },
            )
            _save(db, report)
        if report.learner_provider and report.learner_model:
            final_fingerprint = _catalog_fingerprint(
                client, report.learner_provider, report.learner_model, refresh=True
            )
            if fingerprint_changed(report.learner_fingerprint, final_fingerprint):
                report.limitations.append("La huella del alumno cambió durante la tirada.")
                _expire_observed(db, report, learner_fingerprint=final_fingerprint)
        if not verified_model(report.learner_fingerprint):
            report.limitations.append("El catálogo no demuestra la huella del alumno; no se recomienda un umbral.")
        if config.teacher_judge:
            _freeze_direct_teacher(client, report, refresh=True)
        if not report.teacher_fingerprints or any(
            not verified_model(fp) for fp in report.teacher_fingerprints.values()
        ):
            report.limitations.append(
                "El maestro no tiene huella verificada; sus etiquetas no se certifican para reutilizar."
            )
        analyze_calibration(report, plan)
        previous_rows = db.query(
            "SELECT run_id, report_json FROM calibration_run "
            "WHERE use_case = ? AND run_id != ? ORDER BY created_at DESC",
            (report.use_case, report.run_id),
        )
        for row in previous_rows:
            previous = CalibrationReport.model_validate_json(row["report_json"])
            if previous.plan_signature != report.plan_signature:
                continue
            expired = (
                (
                    (previous.learner_provider, previous.learner_model)
                    == (report.learner_provider, report.learner_model)
                    and fingerprint_changed(previous.learner_fingerprint, report.learner_fingerprint)
                )
                or previous.expired
            )
            report.previous_comparison = compare_calibrations(previous, report)
            expired = expired or bool(report.previous_comparison.get("calibration_regression"))
            report.previous_comparison.update(previous_run_id=previous.run_id, previous_artifact_expired=expired)
            if expired:
                expire_calibration(db, previous)
            break
    except (BrokerError, ValueError, ArithmeticError) as error:
        report.status = "no_calibrado"
        report.recommended_threshold = None
        code = error.code if isinstance(error, BrokerError) else "INVALID_INSTRUMENT"
        report.reasons.append(f"Tirada fallida explícitamente: {code or 'BROKER_ERROR'}")
        _expire_observed(db, report)
    _save(db, report)
    return report


def expire_calibration(db: Database, report: CalibrationReport) -> None:
    """Conserva la medición histórica, pero retira su vigencia en datos e informes."""
    report.expired = True
    db.execute(
        "UPDATE calibration_run SET expired = 1, report_json = ? WHERE run_id = ?",
        (report.model_dump_json(), report.run_id),
    )
    # Solo se reescriben informes registrados y que siguen correspondiendo a
    # esta tirada; nunca un archivo que el usuario haya reemplazado.
    if report.artifact_json_path:
        artifact = Path(report.artifact_json_path)
        try:
            existing = json.loads(artifact.read_text(encoding="utf-8"))
            if existing.get("run_id") == report.run_id:
                artifact.write_text(report.model_dump_json(indent=2), encoding="utf-8")
                if report.artifact_html_path:
                    from model_drift.report import write_calibration

                    write_calibration(report, report.artifact_html_path)
        except (OSError, ValueError, AttributeError):
            pass  # La base conserva la caducidad aunque el informe se haya movido.


def save_artifacts(
    db: Database, report: CalibrationReport, artifact: str | Path, html: str | Path,
) -> Path:
    from model_drift.report import write_calibration

    artifact = Path(artifact).resolve()
    report.artifact_json_path = str(artifact)
    report.artifact_html_path = str(Path(html).resolve())
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    result = write_calibration(report, html)
    _save(db, report)
    return result


def revalidate_changed(
    client: BrokerClient, db: Database, reports_dir: str | Path, *,
    cancel: Event | None = None, progress: Callable[[str], None] | None = None,
) -> list[CalibrationReport]:
    """Al refrescar el catálogo, repite planes vigentes cuya huella cambió.

    No es un proceso permanente: las entradas originales y el maestro elegido
    se recuperan de SQLite, y cada cambio se atiende una sola vez.
    """
    rows = db.query("SELECT report_json FROM calibration_run WHERE expired = 0 ORDER BY created_at DESC")
    seen: set[tuple[str, str | None, str | None]] = set()
    results = []
    for row in rows:
        previous = CalibrationReport.model_validate_json(row["report_json"])
        key = previous.plan_signature, previous.learner_provider, previous.learner_model
        if key in seen:
            continue
        seen.add(key)
        if not previous.config.revalidate_on_fingerprint_change or not previous.learner_model:
            continue
        current = _catalog_fingerprint(client, previous.learner_provider or "", previous.learner_model)
        teacher_changed = False
        task_targets = {
            target.key: target
            for target in (previous.config.teacher_task, previous.config.second_teacher_task)
            if target is not None
        }
        for identity, fingerprint in previous.teacher_fingerprints.items():
            task_target = task_targets.get(identity)
            provider, model = (
                (task_target.provider, task_target.model) if task_target else tuple(identity.split("/", 1))
            )
            fresh = _catalog_fingerprint(client, provider, model)
            teacher_changed |= fingerprint_changed(fingerprint, fresh)
        if not teacher_changed and not fingerprint_changed(previous.learner_fingerprint, current):
            continue
        expire_calibration(db, previous)
        cases = db.query("SELECT case_json FROM calibration_case WHERE run_id = ? ORDER BY case_id", (previous.run_id,))
        if len(cases) < 2:
            continue
        plan = CalibrationPlan(config=previous.config, cases=[json.loads(case["case_json"]) for case in cases])
        if progress:
            progress(f"Cambió la huella de {previous.learner_model}; repitiendo {previous.use_case}.")
        report = run_calibration(client, plan, db, cancel=cancel, progress=progress)
        destination = Path(reports_dir) / report.run_id
        export_distilled(db, report, destination.with_suffix(".references.json"))
        save_artifacts(db, report, destination.with_suffix(".json"), destination.with_suffix(".html"))
        results.append(report)
    return results


def export_distilled(db: Database, report: CalibrationReport, destination: str | Path) -> Path:
    rows = db.query(
        "SELECT case_json, teacher_json FROM calibration_case WHERE run_id = ? ORDER BY case_id", (report.run_id,)
    )
    payload = {
        "version": 1,
        "run_id": report.run_id,
        "use_case": report.use_case,
        "teacher_qualified": report.teacher_qualified,
        "teacher_fingerprints": report.teacher_fingerprints,
        "references": [
            {"case": json.loads(row["case_json"]), "label": json.loads(row["teacher_json"])} for row in rows
        ],
    }
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
