"""Contrasta una auditoría persistida con referencias de procedencia explícita.

python -m model_drift.system1_benchmark --database state/model_drift.sqlite
    --review-id review-... --labels etiquetas.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import BaseModel, Field

from model_drift.db import Database
from model_drift.system1_contracts import SEMANTIC_OPTIONS, SemanticLabel, SemanticReview


class BenchmarkResult(BaseModel):
    labeled_pairs: int = 0
    classified_pairs: int = 0
    abstentions: int = 0
    correct_labels: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    confusion: dict[str, dict[str, int]] = Field(default_factory=dict)
    reference_sources: dict[str, int] = Field(default_factory=dict)
    excluded_references: int = 0


class ReferenceLabel(BaseModel):
    decision: SemanticLabel
    source: str = "human"
    teacher_qualified: bool = False
    unanimous: bool = False
    teacher_fingerprint: dict[str, object] = Field(default_factory=dict)


def score_labels(review: SemanticReview, labels: dict[str, SemanticLabel | ReferenceLabel]) -> BenchmarkResult:
    """Errores de clasificación, separados de abstenciones y del veredicto final.

    Se distinguen etiquetas humanas, destiladas y sintéticas. Un
    rechazo, fallback o `uncertain` es abstención y nunca acierto fabricado.
    """
    ids = {item.case_id for item in review.assessments}
    if not labels or set(labels) - ids:
        raise ValueError("etiquetas vacías, desconocidas o con casos ajenos a esta auditoría")
    result = BenchmarkResult(labeled_pairs=len(labels))
    for item in review.assessments:
        expected = labels.get(item.case_id)
        if expected is None:
            continue
        source = "human"
        if isinstance(expected, (ReferenceLabel, dict)):
            reference = ReferenceLabel.model_validate(expected)
            source = reference.source
            if source not in {"human", "distilled_system2", "synthetic_authored"}:
                raise ValueError("procedencia desconocida de la etiqueta")
            if source == "distilled_system2" and not (
                reference.teacher_qualified and reference.unanimous and reference.teacher_fingerprint
            ):
                result.excluded_references += 1
                continue
            expected = reference.decision
        if expected not in SEMANTIC_OPTIONS:
            raise ValueError("etiqueta fuera del dominio")
        result.reference_sources[source] = result.reference_sources.get(source, 0) + 1
        if item.source != "system1" or not item.accepted or item.classification == "uncertain":
            result.abstentions += 1
            continue
        result.classified_pairs += 1
        result.correct_labels += int(item.classification == expected)
        row = result.confusion.setdefault(expected, {})
        row[item.classification] = row.get(item.classification, 0) + 1
        result.false_positives += int(
            item.classification == "regression_candidate" and expected != "regression_candidate"
        )
        result.false_negatives += int(
            expected == "regression_candidate" and item.classification != "regression_candidate"
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Contrasta juicios System 1 guardados con referencias explícitas.")
    parser.add_argument("--database", required=True)
    parser.add_argument("--review-id", required=True)
    parser.add_argument("--labels", required=True)
    args = parser.parse_args(argv)
    if not Path(args.database).is_file():
        parser.error("no existe la base de datos")
    with Database(args.database) as db:
        row = db.query_one("SELECT review_json FROM comparison_review WHERE review_id = ?", (args.review_id,))
    if row is None:
        parser.error("no existe la auditoría solicitada")
    try:
        labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
        if not isinstance(labels, dict):
            raise ValueError("el fichero de etiquetas debe ser un objeto case_id: etiqueta")
        result = score_labels(SemanticReview.model_validate_json(row["review_json"]), labels)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
