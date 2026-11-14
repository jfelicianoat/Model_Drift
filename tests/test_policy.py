"""Política de regresión: orden del veredicto, tolerancias y unidades (plan 13-nov, MD-01/02/03/08)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from model_drift import cli
from model_drift.comparison import ComparisonReport
from model_drift.domain import (
    ChangeDirection,
    ChangeStatus,
    ComparisonOutcome,
    EvidenceKind,
    PracticalSignificance,
    RunVerdict,
)
from model_drift.manifest import ComparabilityReport
from model_drift.policy import MetricRule, RegressionPolicy, evaluate_policy


def _caso(case_id: str, metric: str, base: float, cand: float, *, relevante: bool = True) -> ComparisonOutcome:
    efecto = cand - base
    if efecto == 0:
        return ComparisonOutcome(
            case_id=case_id,
            metric=metric,
            change_status=ChangeStatus.sin_cambio_observable,
            baseline_score=base,
            candidate_score=cand,
            effect_size=0.0,
        )
    higher_is_better = metric != "latency_ms"
    mejora = (efecto > 0) == higher_is_better
    return ComparisonOutcome(
        case_id=case_id,
        metric=metric,
        change_status=ChangeStatus.cambio_significativo,
        change_direction=ChangeDirection.mejora if mejora else ChangeDirection.regresion,
        evidence_kind=EvidenceKind.oraculo_determinista,
        evidence_confidence=1.0,
        baseline_score=base,
        candidate_score=cand,
        effect_size=efecto,
        p_value_corrected=0.001,
        practical_significance=(
            PracticalSignificance.detectado_relevante if relevante else PracticalSignificance.detectado_irrelevante
        ),
    )


def _bloqueante_excedida_y_latencia_mejora() -> tuple[dict[str, list[ComparisonOutcome]], RegressionPolicy]:
    outcomes = {
        "pass_rate": [_caso(f"c{i}", "pass_rate", 1.0, 0.0) for i in range(3)],
        "latency_ms": [_caso(f"c{i}", "latency_ms", 900.0, 300.0) for i in range(3)],
    }
    politica = RegressionPolicy.from_mapping(
        {
            "blocking": {"pass_rate": {"max_drop_pp": 0}},
            "warning": {
                "latency_ms": {"max_increase_pct": 15, "higher_is_better": False, "minimum_practical_effect": 50.0}
            },
        }
    )
    return outcomes, politica


# --- MD-01 ---------------------------------------------------------------


def test_bloqueante_excedida_con_otra_metrica_mejorando_da_blocked() -> None:
    outcomes, politica = _bloqueante_excedida_y_latencia_mejora()
    informe = evaluate_policy(outcomes, politica, total_cases=3)
    assert informe.verdict is RunVerdict.blocked
    assert "pass_rate" in informe.blocking_exceeded
    assert "latency_ms" in informe.improved
    assert "latency_ms" in informe.summary(), "el resumen no pierde las mejoras"


def test_cli_devuelve_3_con_bloqueante_excedida_aunque_otra_mejore(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    outcomes, politica = _bloqueante_excedida_y_latencia_mejora()
    veredicto = evaluate_policy(outcomes, politica, total_cases=3)
    informe = ComparisonReport(
        baseline_run_id="a",
        candidate_run_id="b",
        comparability=ComparabilityReport(comparable=True),
        outcomes=[item for lista in outcomes.values() for item in lista],
        verdict=veredicto,
    )

    class _Lado:
        run_id = "x"
        manifest = None

    monkeypatch.setattr(cli, "load_external_treatments", lambda _plan: (None, _Lado(), _Lado()))
    monkeypatch.setattr(cli, "compare_runs", lambda *_a, **_k: informe)
    monkeypatch.setattr(cli, "write_comparison", lambda *_a, **_k: tmp_path / "x.html")
    codigo = cli.main(["evaluar-tratamientos", "--plan", "p.json", "--informe", str(tmp_path / "x.html")])
    assert codigo == cli.EXIT_BLOCKED == 3


# --- MD-02 ---------------------------------------------------------------


def _un_caso_critico() -> list[ComparisonOutcome]:
    return [_caso(f"c{i}", "pass_rate", 1.0, 1.0) for i in range(49)] + [_caso("critico", "pass_rate", 1.0, 0.0)]


def test_la_media_diluye_un_caso_critico_sin_max_regressed_cases() -> None:
    politica = RegressionPolicy(metrics={"pass_rate": MetricRule(severity="blocking", max_drop=0.02)})
    informe = evaluate_policy({"pass_rate": _un_caso_critico()}, politica, total_cases=50)
    assert informe.metrics[0].exceeds_tolerance is False


def test_max_regressed_cases_0_bloquea_con_un_solo_caso_critico() -> None:
    politica = RegressionPolicy(
        metrics={"pass_rate": MetricRule(severity="blocking", max_drop=0.02, max_regressed_cases=0)}
    )
    informe = evaluate_policy({"pass_rate": _un_caso_critico()}, politica, total_cases=50)
    assert informe.metrics[0].exceeds_tolerance is True
    assert informe.verdict is RunVerdict.blocked
    assert "1 con regresión relevante" in informe.metrics[0].detail


def test_max_regressed_cases_1_tolera_un_caso() -> None:
    politica = RegressionPolicy(
        metrics={"pass_rate": MetricRule(severity="blocking", max_drop=0.02, max_regressed_cases=1)}
    )
    informe = evaluate_policy({"pass_rate": _un_caso_critico()}, politica, total_cases=50)
    assert informe.metrics[0].exceeds_tolerance is False


def test_forma_corta_acepta_max_regressed_cases() -> None:
    politica = RegressionPolicy.from_mapping({"blocking": {"pass_rate": {"max_drop_pp": 2, "max_regressed_cases": 0}}})
    assert politica.metrics["pass_rate"].max_regressed_cases == 0


# --- MD-03 ---------------------------------------------------------------


def test_regla_bloqueante_sin_tolerancia_se_rechaza() -> None:
    with pytest.raises(ValidationError):
        MetricRule(severity="blocking")


def test_forma_corta_bloqueante_vacia_se_rechaza() -> None:
    with pytest.raises(ValidationError):
        RegressionPolicy.from_mapping({"blocking": {"x": {}}})


def test_regla_informativa_sin_tolerancia_se_construye() -> None:
    assert MetricRule(severity="informative").max_drop is None


# --- MD-08 ---------------------------------------------------------------


def test_forma_explicita_rechaza_porcentaje_escrito_como_entero() -> None:
    with pytest.raises(ValidationError):
        RegressionPolicy.from_mapping({"metrics": {"lat": {"severity": "warning", "max_increase_pct": 15}}})


def test_forma_corta_convierte_el_porcentaje() -> None:
    politica = RegressionPolicy.from_mapping({"warning": {"lat": {"max_increase_pct": 15}}})
    assert politica.metrics["lat"].max_increase_pct == pytest.approx(0.15)
