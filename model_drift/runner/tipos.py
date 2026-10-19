"""Configuración, avance y salida de una tirada.

Son los tipos que cruzan la frontera del motor: la interfaz construye
`RunConfig`, pinta `RunProgress` y guarda `RunOutput` sin saber nada de
cómo se ejecuta una tirada por dentro.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from model_drift.comparison import RunResults
from model_drift.manifest import RunManifest


@dataclass
class RunConfig:
    """Configuración de una tirada."""

    repetitions: int = 3
    temperature: float | None = 0.0
    top_p: float | None = None
    seed: int | None = 1234
    max_output_tokens: int | None = None
    budget_usd: float | None = None
    label: str = ""
    #: Ejecutar código generado por un modelo es decisión del operador.
    allow_code_execution: bool = False
    timeout_seconds: int | None = None

    def __post_init__(self) -> None:
        # Con 0 o menos el bucle de repeticiones no itera y se guardaría una
        # tirada «completed» que no midió nada.
        if self.repetitions < 1:
            raise ValueError(f"las repeticiones deben ser 1 o más (se pidieron {self.repetitions})")
        if self.budget_usd is not None and self.budget_usd < 0:
            raise ValueError(f"el presupuesto no puede ser negativo ({self.budget_usd})")

    def params(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "max_output_tokens": self.max_output_tokens,
        }


@dataclass
class RunProgress:
    """Un aviso de avance. Lo consume la interfaz para pintar la barra."""

    model_key: str
    case_id: str
    case_index: int
    total_cases: int
    repetition: int
    total_repetitions: int
    invocations: int
    cost_usd: float

    @property
    def fraction(self) -> float:
        if self.total_cases == 0 or self.total_repetitions == 0:
            return 0.0
        hechas = self.case_index * self.total_repetitions + self.repetition + 1
        return min(1.0, hechas / (self.total_cases * self.total_repetitions))

    def describe(self) -> str:
        return (
            f"{self.model_key} · caso {self.case_index + 1}/{self.total_cases} "
            f"({self.case_id}) · repetición {self.repetition + 1}/{self.total_repetitions}"
        )


@dataclass
class RunOutput:
    """Lo que devuelve una tirada."""

    run_id: str
    manifest: RunManifest
    results: dict[str, RunResults] = field(default_factory=dict)
    invocations: int = 0
    cost_usd: float = 0.0
    aborted_reason: str | None = None
    #: Invocaciones sin salida evaluable, y su motivo agrupado como
    #: «código: mensaje» -> veces. Sin esto un token caducado se veía como éxito.
    failed_invocations: int = 0
    errors: dict[str, int] = field(default_factory=dict)
    #: Lo que AI OS Core no llegó a apuntar (recibos, certificaciones). No invalida la tirada.
    core_notices: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        """`failed` si nada produjo salida: una tirada así no midió el modelo."""
        if self.invocations and self.failed_invocations >= self.invocations:
            return "failed"
        return "aborted" if self.aborted_reason else "completed"

    def failure_summary(self) -> str | None:
        """«10/10 invocaciones fallaron (AUTH: …×10)», o `None` si no falló ninguna."""
        if not self.failed_invocations:
            return None
        ordenados = sorted(self.errors.items(), key=lambda kv: -kv[1])
        motivos = ", ".join(f"{motivo} ×{veces}" for motivo, veces in ordenados)
        return f"{self.failed_invocations}/{self.invocations} invocaciones sin salida ({motivos})"

    def for_model(self, model_key: str) -> RunResults | None:
        return self.results.get(model_key)


class BudgetExceeded(RuntimeError):
    """El presupuesto declarado para la tirada se agotó."""
