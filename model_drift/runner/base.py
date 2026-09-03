"""Estado del motor y aplicabilidad de un caso a un modelo.

Regla 1 del motor, aplicada aquí: capacidad ausente es `not_applicable`, no
suspenso. Un `None` del catálogo («no comprobado») también excluye — no se
puede contar como fallo lo que nadie ha verificado.
"""
from __future__ import annotations

from pathlib import Path

from model_drift.broker_client import BrokerClient
from model_drift.broker_contracts import ModelReference
from model_drift.cases import Case, Suite
from model_drift.db import Database
from model_drift.fixtures import FixtureResolver
from model_drift.metamorphic import MetamorphicRelation
from model_drift.oracles import Embedder, EvaluationServices
from model_drift.policy import RegressionPolicy
from model_drift.runner.tipos import RunConfig


class RunnerBase:
    """Dependencias del motor y decisiones que no necesitan invocar."""

    def __init__(
        self,
        client: BrokerClient,
        suite: Suite,
        *,
        db: Database | None = None,
        config: RunConfig | None = None,
        policy: RegressionPolicy | None = None,
        relations: dict[str, MetamorphicRelation] | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.client = client
        self.suite = suite
        self.db = db
        self.config = config or RunConfig()
        self.policy = policy or RegressionPolicy()
        self.relations = relations or {}
        self.embedder = embedder
        self.resolver = FixtureResolver(Path(suite.root or "."))
        self.services = EvaluationServices(
            resolver=self.resolver,
            embedder=embedder,
            allow_code_execution=self.config.allow_code_execution,
        )

    # ------------------------------------------------------------ aplicabilidad

    def _not_applicable_reason(self, case: Case, model: ModelReference) -> str | None:
        """Por qué este caso no aplica a este modelo, si es que no aplica.

        Un `None` del catálogo («no comprobado») también excluye: no se puede
        contar como fallo lo que nadie ha verificado que el modelo no sepa hacer.
        """
        if not case.required_capabilities:
            return None
        entrada = self.client.entry_for(model)
        if entrada is None:
            return f"el modelo {model.key} no está en el catálogo del Broker"
        for capacidad in case.required_capabilities:
            soporta = entrada.supports(capacidad)
            if soporta is False:
                return f"el modelo no soporta {capacidad}"
            if soporta is None:
                return f"el Broker no ha comprobado si el modelo soporta {capacidad}"
        return None
