"""Validación local de cada nota cruda. El Broker no la avala como decisión."""

from __future__ import annotations

import math
from typing import Any

from model_drift.calibration.models import CalibrationConfig, LearnerNote, label_key


def validate_note(raw: dict[str, Any], config: CalibrationConfig, note: LearnerNote) -> LearnerNote:
    if type(raw.get("accepted")) is not bool:
        note.invalid_score = True
        note.reason_code = "INVALID_ACCEPTED_FLAG"
        return note
    note.accepted = raw.get("accepted") is True
    attempts = raw.get("attempts")
    if attempts == []:
        note.reason_code = raw.get("reason_code") or "NO_ATTEMPT"
        return note
    if not isinstance(attempts, list) or len(attempts) != 1 or not isinstance(attempts[0], dict):
        note.invalid_score = True
        note.reason_code = "INVALID_ATTEMPT_COUNT"
        return note
    attempt = attempts[0]
    note.provider = attempt.get("provider") if isinstance(attempt.get("provider"), str) else None
    note.model = attempt.get("model") if isinstance(attempt.get("model"), str) else None
    note.reason_code = attempt.get("reason_code") or raw.get("reason_code")
    for key in ("latency_ms", "tokens_input", "tokens_output"):
        value = attempt.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
            setattr(note, key, value)
    if note.provider != config.learner.provider:
        note.invalid_score = True
        note.reason_code = "MODEL_IDENTITY_MISMATCH"
        return note
    if attempt.get("decision") is None or attempt.get("confidence") is None:
        note.reason_code = note.reason_code or "MISSING_RAW_SCORE"
        return note  # abstención, no etiqueta de fallo inventada
    if not note.model:
        note.invalid_score = True
        note.reason_code = "MODEL_IDENTITY_MISMATCH"
        return note
    try:
        if note.reason_code not in (None, "LOW_CONFIDENCE", "INSUFFICIENT_MARGIN"):
            raise ValueError("SCORE_ON_FAILED_ATTEMPT")
        if attempt.get("score_source") != "native":
            raise ValueError("NON_NATIVE_SCORE")
        decision = config.normalize(attempt["decision"])
        confidence = _confidence(attempt["confidence"])
        alternatives = attempt.get("alternatives")
        if not isinstance(alternatives, list):
            raise ValueError("MISSING_ALTERNATIVES")
        probabilities = {label_key(decision): confidence}
        for alternative in alternatives:
            if not isinstance(alternative, dict):
                raise ValueError("INVALID_ALTERNATIVE")
            key = label_key(config.normalize(alternative.get("value")))
            if key in probabilities:
                raise ValueError("DUPLICATE_ALTERNATIVE")
            score = _confidence(alternative.get("confidence"))
            if score > confidence:
                raise ValueError("ALTERNATIVE_ABOVE_DECISION")
            probabilities[key] = score
        if set(probabilities) != {label_key(value) for value in config.domain()}:
            raise ValueError("INCOMPLETE_ALTERNATIVES")
        if abs(sum(probabilities.values()) - 1.0) > config.probability_sum_tolerance:
            raise ValueError("PROBABILITY_SUM_MISMATCH")
        if note.accepted and (
            label_key(config.normalize(raw.get("decision"))) != label_key(decision)
            or _confidence(raw.get("confidence")) != confidence
            or raw.get("provider") != note.provider
            or raw.get("model") != note.model
        ):
            raise ValueError("ACCEPTED_RESPONSE_MISMATCH")
        note.decision = decision
        note.confidence = confidence
        note.probabilities = probabilities
        note.valid_score = True
    except ValueError as error:
        note.invalid_score = True
        note.reason_code = str(error)
    return note


def _confidence(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("INVALID_CONFIDENCE")
    return float(value)
