"""Aserciones de fundamentación: citas cotejadas y abstención.

Miden si la respuesta se apoya en lo que el modelo tenía delante, que es
la deriva que más caro sale en producción y la que menos se ve mirando
solo el texto.
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field

from model_drift.oracles.base import BaseAssertion
from model_drift.oracles.tipos import AssertionResult, EvaluationInput, EvaluationServices
from model_drift.oracles.utilidades import _URL


class CitationsGroundedAssertion(BaseAssertion):
    """Las URLs citadas salieron de lo que el modelo realmente tuvo delante.

    Cotejo determinista, en la misma línea que `app/evidence.py` del Broker:
    comparar URLs no necesita otro modelo opinando y no puede equivocarse a
    favor de nadie. No juzga si la respuesta es correcta, solo si sus enlaces
    tienen procedencia.
    """

    type: Literal["citations_grounded"] = "citations_grounded"
    #: Si es False, basta con que no invente; no se exige que cite.
    require_at_least_one: bool = False

    def _evaluate(self, data: EvaluationInput, services: EvaluationServices) -> AssertionResult:
        citadas = {url.rstrip(".,;:)") for url in _URL.findall(data.text)}
        disponibles = {url.rstrip(".,;:)") for url in data.sources}
        if not citadas:
            if self.require_at_least_one:
                return self._result(False, "no cita ninguna fuente")
            return self._result(True, "no cita fuentes")
        inventadas = sorted(citadas - disponibles)
        if inventadas:
            return self._result(False, f"cita fuentes que no consultó: {', '.join(inventadas[:3])}")
        return self._result(True, f"{len(citadas)} citas con procedencia")


class AbstentionAssertion(BaseAssertion):
    """El modelo se abstiene cuando la respuesta no está en el contexto.

    Métrica central para el Knowledge_Orchestrator: la alternativa a abstenerse
    es inventar.
    """

    type: Literal["abstention"] = "abstention"
    should_abstain: bool = True
    phrases: list[str] = Field(
        default_factory=lambda: [
            "no consta",
            "no aparece",
            "no se menciona",
            "no dispongo",
            "no puedo determinar",
            "no está en el contexto",
            "no hay información",
        ]
    )

    def _evaluate(self, data: EvaluationInput, services: EvaluationServices) -> AssertionResult:
        texto = data.text.casefold()
        abstiene = any(phrase.casefold() in texto for phrase in self.phrases)
        if abstiene == self.should_abstain:
            return self._result(True)
        if self.should_abstain:
            return self._result(False, "no se abstuvo pese a no tener la información")
        return self._result(False, "se abstuvo teniendo la información disponible")
