"""Invocación de un caso y de sus variantes metamórficas."""
from __future__ import annotations

from model_drift.broker_client import InvocationOutcome
from model_drift.broker_contracts import ModelReference
from model_drift.cases import Case
from model_drift.metamorphic import (
    RelationOutcome,
    VariantResult,
    evaluate_relation,
    generate_variants,
)
from model_drift.oracles import EvaluationInput, ToolCall
from model_drift.runner.base import RunnerBase


class InvocacionMixin(RunnerBase):
    """Cómo se pide una respuesta al Broker y qué se le da al oráculo."""

    def _invoke_case(
        self,
        case: Case,
        model: ModelReference,
        group: str,
        *,
        prompt: str | None = None,
    ) -> InvocationOutcome:
        return self.client.invoke(
            prompt=prompt if prompt is not None else case.user_prompt,
            target=model,
            system=case.system_prompt,
            params={**self.config.params(), **case.params.model_dump(exclude_none=True)},
            group=group,
            timeout_seconds=self.config.timeout_seconds,
        )

    @staticmethod
    def _evaluation_input(case: Case, outcome: InvocationOutcome) -> EvaluationInput:
        return EvaluationInput(
            text=outcome.text,
            tool_calls=[
                ToolCall(name=str(item.get("name", "")), arguments=dict(item.get("arguments") or {}))
                for item in outcome.tool_calls
            ],
            sources=list(case.context.sources),
            produced_output=outcome.produced_output,
        )

    def _run_relations(
        self,
        case: Case,
        model: ModelReference,
        group: str,
        reference_output: str,
    ) -> list[RelationOutcome]:
        """Ejecuta las relaciones metamórficas declaradas por el caso."""
        resultados: list[RelationOutcome] = []
        for relation_id in case.metamorphic_relations:
            relacion = self.relations.get(relation_id)
            if relacion is None:
                resultados.append(
                    RelationOutcome(
                        relation_id=relation_id,
                        relation_version="?",
                        passed=False,
                        satisfied=0,
                        total=0,
                        not_applicable=True,
                        reason=f"la relación {relation_id!r} no está cargada",
                    )
                )
                continue
            variantes_prompt = generate_variants(relacion, case.user_prompt, self.resolver)
            variantes: list[VariantResult] = []
            for variant_id, prompt in variantes_prompt:
                salida = self._invoke_case(case, model, group, prompt=prompt)
                variantes.append(
                    VariantResult(variant_id=variant_id, prompt=prompt, output=salida.text if salida.ok else "")
                )
            resultados.append(
                evaluate_relation(relacion, reference_output, variantes, embedder=self.embedder)
            )
        return resultados
