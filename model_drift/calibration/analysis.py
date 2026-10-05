"""Calidad del maestro, fiabilidad y selección de umbral sin mirar el test."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from typing import Any

from model_drift.calibration.models import CalibrationPlan, CalibrationReport, LearnerNote, label_key
from model_drift.judge import cohen_kappa
from model_drift.statistics import mcnemar_exact
from model_drift.statistics.calibration import clopper_pearson


def qualify_teacher(report: CalibrationReport, plan: CalibrationPlan) -> None:
    report.teacher_qualified = False
    report.teacher_kappa = None
    labels = report.labels
    report.ambiguous_cases = sum(not label.unanimous for label in labels.values())
    report.teacher_self_consistency = (
        sum(label.unanimous for label in labels.values()) / len(labels) if labels else None
    )
    human, teacher = [], []
    for case in plan.cases:
        label = labels.get(case.id)
        if case.human_verified and label is not None and label.unanimous:
            human.append(label_key(plan.config.normalize(case.human_label)))
            teacher.append(label_key(label.decision))
    report.human_cases = len(human)
    if len(human) < plan.config.min_human_cases:
        report.reasons.append("Muestra humana insuficiente para comprobar el maestro.")
        return
    if len(set(human)) < 2:
        report.reasons.append("El subconjunto humano no contiene al menos dos clases.")
        return
    if max(Counter(human).values()) / len(human) > plan.config.max_human_class_ratio:
        report.reasons.append("El subconjunto humano está desequilibrado entre clases.")
        return
    report.teacher_kappa = cohen_kappa(
        teacher,
        human,
        scale=tuple(label_key(value) for value in plan.config.domain()),
        weighted=plan.config.decision_type == "score",
    )
    report.teacher_qualified = report.teacher_kappa >= plan.config.min_teacher_kappa
    if not report.teacher_qualified:
        report.reasons.append("El maestro no supera el kappa mínimo frente a humanos.")


def _eligible(report: CalibrationReport, partition: str) -> list[LearnerNote]:
    return [
        note for note in report.notes if note.partition == partition and note.valid_score and note.reference_eligible
    ]


def reliability(notes: list[LearnerNote], edges: list[float], confidence_level: float) -> dict[str, Any]:
    bins = []
    ece = 0.0
    for index, (low, high) in enumerate(zip(edges, edges[1:], strict=False)):
        items = [
            note
            for note in notes
            if note.confidence is not None
            and low <= note.confidence
            and (note.confidence < high or index == len(edges) - 2)
        ]
        n = len(items)
        correct = sum(label_key(item.decision) == label_key(item.reference) for item in items)
        accuracy = correct / n if n else None
        average = sum(item.confidence or 0 for item in items) / n if n else None
        interval = clopper_pearson(correct, n, confidence_level)
        bins.append(
            {
                "low": low,
                "high": high,
                "n": n,
                "accuracy": accuracy,
                "mean_confidence": average,
                "ci_low": interval[0],
                "ci_high": interval[1],
            }
        )
        if n and accuracy is not None and average is not None:
            ece += n / len(notes) * abs(accuracy - average)
    brier = (
        sum(
            sum(
                (probability - int(key == label_key(note.reference))) ** 2
                for key, probability in note.probabilities.items()
            )
            for note in notes
        )
        / len(notes)
        if notes
        else None
    )
    return {"n": len(notes), "bins": bins, "brier_multiclass": brier, "ece_top_label": ece if notes else None}


def _class_metrics(notes: list[LearnerNote]) -> dict[str, Any]:
    confusion: dict[str, dict[str, int]] = {}
    for note in notes:
        row = confusion.setdefault(label_key(note.reference), {})
        key = label_key(note.decision)
        row[key] = row.get(key, 0) + 1
    keys = {label_key(note.reference) for note in notes} | {label_key(note.decision) for note in notes}
    classes = {}
    for key in sorted(keys):
        tp = confusion.get(key, {}).get(key, 0)
        predicted = sum(row.get(key, 0) for row in confusion.values())
        actual = sum(confusion.get(key, {}).values())
        classes[key] = {
            "precision": tp / predicted if predicted else None,
            "recall": tp / actual if actual else None,
            "support": actual,
        }
    return {
        "n": len(notes),
        "accuracy": sum(label_key(n.decision) == label_key(n.reference) for n in notes) / len(notes) if notes else None,
        "confusion": confusion,
        "classes": classes,
    }


def _selected(notes: list[LearnerNote], threshold: float, approvals: list[Any]) -> list[LearnerNote]:
    approved = {label_key(value) for value in approvals}
    return [
        note
        for note in notes
        if note.confidence is not None
        and note.confidence >= threshold
        and (not approved or label_key(note.decision) in approved)
    ]


def _precision(notes: list[LearnerNote], confidence: float) -> tuple[float | None, float, float]:
    correct = sum(label_key(note.decision) == label_key(note.reference) for note in notes)
    low, high = clopper_pearson(correct, len(notes), confidence)
    return correct / len(notes) if notes else None, low, high


def synthetic_diagnostics(report: CalibrationReport, plan: CalibrationPlan) -> dict[str, Any] | None:
    """Comparación exploratoria, sin promover etiquetas inventadas a verdad humana."""
    expected = {
        case.id: case.expected_label for case in plan.cases if case.expected_label_source == "synthetic_authored"
    }
    if not expected:
        return None
    notes = [
        note.model_copy(update={"reference": expected[note.case_id]})
        for note in report.notes if note.case_id in expected and note.valid_score and note.partition == "test"
    ]
    references, teacher = [], []
    for cid, value in expected.items():
        label = report.labels.get(cid)
        if label is not None and label.unanimous:
            references.append(label_key(value))
            teacher.append(label_key(label.decision))
    return {
        "source": "synthetic_authored", "expected_cases": len(expected),
        "expected_labels": expected,
        "eligible_for_threshold_recommendation": False,
        "teacher_unanimous_cases": len(teacher),
        "teacher_agreement_kappa": cohen_kappa(
            teacher, references, scale=tuple(label_key(value) for value in plan.config.domain()),
            weighted=plan.config.decision_type == "score",
        ) if len(set(references)) >= 2 else None,
        "raw_test": _class_metrics(notes),
        "accepted_test": _class_metrics([note for note in notes if note.accepted]),
        "adversarial_raw_test": _class_metrics([note for note in notes if note.adversarial]),
        "test_reliability": reliability(notes, plan.config.reliability_edges, plan.config.confidence_level),
    }


def analyze_calibration(report: CalibrationReport, plan: CalibrationPlan) -> None:
    config = plan.config
    report.recommended_threshold = None
    report.test_precision = report.test_precision_lower = report.test_precision_upper = None
    report.test_coverage = None
    qualify_teacher(report, plan)
    for note in report.notes:
        teacher = report.labels.get(note.case_id)
        note.reference_eligible = report.teacher_qualified and teacher is not None and teacher.unanimous
        if teacher is not None and teacher.unanimous:
            note.reference = teacher.decision
    calibration = _eligible(report, "calibration")
    test = _eligible(report, "test")
    accepted = [note for note in test if note.accepted]
    invalid = sum(note.invalid_score for note in report.notes)
    abstentions = sum(not note.accepted for note in report.notes)
    report.metrics = {
        "pairs": len(report.notes),
        "invalid_scores": invalid,
        "invalid_score_ratio": invalid / len(report.notes) if report.notes else None,
        "abstentions": abstentions,
        "abstention_reasons": dict(
            Counter(note.reason_code or "UNSPECIFIED" for note in report.notes if not note.accepted)
        ),
        "accepted_test": _class_metrics(accepted),
        "raw_test": _class_metrics(test),
        "calibration_reliability": reliability(calibration, config.reliability_edges, config.confidence_level),
        "test_reliability": reliability(test, config.reliability_edges, config.confidence_level),
        "adversarial_test": _class_metrics([note for note in accepted if note.adversarial]),
        "learner_latency_ms": sum(note.latency_ms for note in report.notes),
        "learner_tokens_input": sum(note.tokens_input or 0 for note in report.notes),
        "learner_tokens_output": sum(note.tokens_output or 0 for note in report.notes),
        "teacher_latency_ms": sum(label.latency_ms for label in report.labels.values()),
        "teacher_cost_usd": sum(label.cost_usd for label in report.labels.values()),
        "teacher_timeouts": sum(label.timeouts for label in report.labels.values()),
        "provenance": {"distilled_system2": len(report.labels), "human_verified": report.human_cases},
        "partition_sizes": dict(Counter(note.partition for note in report.notes)),
        "served_models": dict(Counter(f"{note.provider}/{note.model}" for note in report.notes)),
    }
    synthetic = synthetic_diagnostics(report, plan)
    if synthetic is not None:
        report.metrics["synthetic_diagnostics"] = synthetic
        report.metrics["provenance"]["synthetic_authored"] = synthetic["expected_cases"]
    if report.notes and invalid / len(report.notes) > config.max_invalid_score_ratio:
        report.status = "no_calibrado"
        report.reasons.append("Exceso de notas inválidas: instrumento defectuoso.")
        return
    if report.limitations or not report.teacher_qualified:
        report.status = "no_calibrado" if report.limitations or report.teacher_kappa is not None else "indeterminado"
        return
    if len(calibration) < config.min_calibration_cases or len(test) < config.min_test_cases:
        report.status = "indeterminado"
        report.reasons.append("Muestra de calibración o prueba insuficiente.")
        return
    # Selección solo sobre calibración. La partición test no participa en el
    # barrido ni se usa para probar otro umbral si falla el primero.
    thresholds = sorted(
        {
            note.confidence
            for note in _selected(calibration, 0, config.auto_approve_values)
            if note.confidence is not None
        }
    )
    chosen = None
    all_ruled_out = bool(thresholds)
    for threshold in thresholds:
        selection = _selected(calibration, threshold, config.auto_approve_values)
        _point, lower, upper = _precision(selection, config.confidence_level)
        all_ruled_out = all_ruled_out and upper < config.target_precision
        if selection and lower >= config.target_precision:
            chosen = threshold
            break
    if chosen is None:
        report.status = "no_calibrado" if all_ruled_out else "indeterminado"
        report.reasons.append("Ningún umbral demuestra la precisión objetivo con su cota exacta.")
        return
    selection = _selected(test, chosen, config.auto_approve_values)
    report.test_precision, report.test_precision_lower, report.test_precision_upper = _precision(
        selection,
        config.confidence_level,
    )
    denominator = sum(note.partition == "test" for note in report.notes)
    report.test_coverage = len(selection) / denominator if denominator else None
    report.metrics["candidate_threshold_from_calibration"] = chosen
    report.metrics["test_selected_cases"] = len(selection)
    if selection and report.test_precision_lower >= config.target_precision:
        report.status = "calibrado"
        report.recommended_threshold = chosen
    else:
        report.status = (
            "no_calibrado"
            if report.test_precision is not None and report.test_precision < config.target_precision
            else "indeterminado"
        )
        report.reasons.append("La prueba separada no demuestra la precisión objetivo con su cota exacta.")


def compare_calibrations(previous: CalibrationReport, current: CalibrationReport) -> dict[str, Any]:
    if (
        previous.plan_signature != current.plan_signature
        or previous.dataset_hash != current.dataset_hash
        or previous.teacher_fingerprints != current.teacher_fingerprints
    ):
        return {"comparable": False, "reason": "cambió el caso de uso, dataset o maestro"}
    threshold = previous.recommended_threshold
    if threshold is None:
        return {"comparable": False, "reason": "la calibración anterior no tiene umbral validado"}
    old = {note.case_id: note for note in _eligible(previous, "test")}
    new = {note.case_id: note for note in _eligible(current, "test")}
    common = sorted(set(old) & set(new))
    ids = [cid for cid in common if label_key(old[cid].reference) == label_key(new[cid].reference)]
    if not ids:
        return {"comparable": False, "reason": "sin casos pareados válidos"}

    def selected(note: LearnerNote) -> bool:
        approved = {label_key(value) for value in previous.config.auto_approve_values}
        return (
            note.confidence is not None
            and note.confidence >= threshold
            and (not approved or label_key(note.decision) in approved)
        )

    def safe(note: LearnerNote) -> bool:
        # Un caso sin aprobación automática no es una aprobación errónea.
        return not selected(note) or label_key(note.decision) == label_key(note.reference)

    result = mcnemar_exact([safe(old[cid]) for cid in ids], [safe(new[cid]) for cid in ids])
    old_selected = _selected([old[cid] for cid in ids], threshold, previous.config.auto_approve_values)
    new_selected = _selected([new[cid] for cid in ids], threshold, previous.config.auto_approve_values)
    old_precision, _, _ = _precision(old_selected, previous.config.confidence_level)
    new_precision, _, _ = _precision(new_selected, current.config.confidence_level)
    return {
        "comparable": True,
        "threshold": threshold,
        "contrast": asdict(result),
        "contrast_metric": "ausencia de aprobación errónea al umbral anterior",
        "reference_disagreements_excluded": len(common) - len(ids),
        "previous_precision_at_threshold": old_precision,
        "current_precision_at_threshold": new_precision,
        "calibration_regression": (
            old_precision is not None
            and new_precision is not None
            and new_precision < old_precision
            and result.effect < 0
            and result.p_value <= current.config.drift_alpha
        ),
    }
