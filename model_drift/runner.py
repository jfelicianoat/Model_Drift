"""Motor de ejecución de una tirada.

Ejecuta una suite contra uno o varios modelos, N repeticiones por caso, y deja
en la base de datos las salidas crudas, la telemetría, el manifiesto y la
agregación de casos.

Tres reglas que el motor impone y que no son negociables:

1. **Capacidad ausente = `not_applicable`.** Un modelo que no declara `tools`
   no suspende las pruebas de herramientas: queda fuera de esa métrica. Contar
   ausencia de capacidad como fallo convertiría el ranking en una mentira.
2. **La unidad es el caso.** Las N repeticiones se agregan aquí mismo, en
   `case_result`. Lo que sale del motor hacia la comparación es una observación
   por caso, no N.
3. **El manifiesto declara lo que faltó.** Si el Broker no admite semilla, si
   no expone identidad técnica, si no permite excluir el tráfico del
   aprendizaje: se escribe. Una ausencia declarada es información; una ausencia
   silenciada es un engaño.
"""
from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from model_drift import __version__
from model_drift.broker_client import BrokerClient, InvocationOutcome
from model_drift.broker_contracts import ModelReference, SeedStatus
from model_drift.cases import Case, Suite
from model_drift.comparison import RunResults
from model_drift.db import Database
from model_drift.domain import (
    METRIC_COST,
    METRIC_LATENCY,
    METRIC_OUTPUT_MATCH,
    METRIC_PASS,
    METRIC_TOKENS,
    CaseOutcome,
)
from model_drift.fixtures import FixtureResolver
from model_drift.manifest import (
    Determinism,
    ExactReproducibility,
    InstrumentRef,
    ModelUnderTest,
    ReproducibilityFlags,
    RunManifest,
)
from model_drift.metamorphic import (
    MetamorphicRelation,
    RelationOutcome,
    VariantResult,
    evaluate_relation,
    generate_variants,
)
from model_drift.oracles import (
    Embedder,
    EvaluationInput,
    EvaluationServices,
    ToolCall,
    aggregate_pass,
    evaluate_assertions,
    normalize_text,
)
from model_drift.policy import RegressionPolicy
from model_drift.statistics import aggregate_binary, aggregate_continuous

#: Reexportadas por comodidad: su definición vive en `domain` para que el
#: motor de comparación sepa cuáles son binarias sin depender del runner.
__all__ = [
    "METRIC_COST",
    "METRIC_LATENCY",
    "METRIC_OUTPUT_MATCH",
    "METRIC_PASS",
    "METRIC_TOKENS",
    "RunConfig",
    "RunOutput",
    "RunProgress",
    "Runner",
]


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

    def for_model(self, model_key: str) -> RunResults | None:
        return self.results.get(model_key)


class BudgetExceeded(RuntimeError):
    """El presupuesto declarado para la tirada se agotó."""


class Runner:
    """Ejecuta una suite contra un conjunto de modelos."""

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

    # --------------------------------------------------------------- ejecución

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

    # ------------------------------------------------------------- manifiesto

    def _instruments(self) -> list[InstrumentRef]:
        instrumentos = [
            InstrumentRef(id="model_drift", kind="app", version=__version__),
            InstrumentRef(id="suite", kind="suite", version=self.suite.version, hash=self.suite.hash()),
        ]
        for relacion in sorted(self.relations.values(), key=lambda item: item.id):
            instrumentos.append(
                InstrumentRef(
                    id=f"mr:{relacion.id}",
                    kind="transformacion",
                    version=relacion.version,
                    detail=relacion.transformation.kind.value,
                )
            )
        if self.embedder is not None:
            instrumentos.append(
                InstrumentRef(id="embedder", kind="embedding", version=str(self.embedder.identity))
            )
        return instrumentos

    def _build_manifest(self, run_id: str, models: list[ModelReference]) -> RunManifest:
        manifiesto = RunManifest(
            run_id=run_id,
            broker_base_url=self.client.config.base_url,
            suite_id=self.suite.id,
            suite_version=self.suite.version,
            suite_hash=self.suite.hash(),
            suite_path=str(self.suite.root) if self.suite.root else None,
            repetitions=self.config.repetitions,
            requested_params={k: v for k, v in self.config.params().items() if v is not None},
            instruments=self._instruments(),
        )
        for modelo in models:
            manifiesto.models.append(
                ModelUnderTest(
                    key=modelo.key,
                    provider=modelo.provider,
                    deployment=modelo.deployment,
                    model=modelo.model,
                    fingerprint=self.client.fingerprint_for(modelo),
                )
            )
        manifiesto.reproducibility = ReproducibilityFlags(
            # Provisional: si el Broker rechaza la semilla, el primer 422 lo
            # descubre y `_seal_reproducibility` corrige la bandera al cerrar la
            # tirada. Declararlo aquí como definitivo afirmaría un control que
            # todavía no se ha ejercido.
            seed_fijada=self.config.seed is not None and self.client.config.supports_seed,
            determinismo_solicitado=self.config.seed is not None and self.config.temperature in (0, 0.0),
            determinismo_observado=Determinism.no_evaluado,
            # Nace en `no_demostrada` y solo una reproducción real la mueve.
            reproducibilidad_exacta=ExactReproducibility.no_demostrada,
        )
        return manifiesto

    def _seal_reproducibility(
        self,
        manifiesto: RunManifest,
        deterministas: list[bool],
        estados_semilla: list[SeedStatus] | None = None,
    ) -> None:
        """Cierra las cuatro banderas con lo que REALMENTE ocurrió.

        Se ejecuta al final a propósito: si la semilla se envió y el Broker la
        rechazó, la tirada no tuvo control de determinismo por mucho que se
        pidiera, y el manifiesto tiene que decir eso y no lo que se intentó.

        Desde el contrato 2.9 la fuente es `seed_status` de cada invocación, no
        lo que se pidió: `sent` significa que el Broker la incluyó en la llamada
        al proveedor; `unsupported`, que ese endpoint no tiene por dónde
        llevarla. Pedirla y darla por buena sería exactamente el tipo de
        garantía inventada que este proyecto existe para no cometer.
        """
        flags = manifiesto.reproducibility
        pedida = self.config.seed is not None and self.client.config.supports_seed
        if estados_semilla:
            flags.seed_fijada = all(estado is SeedStatus.sent for estado in estados_semilla)
            if not flags.seed_fijada:
                manifiesto.declare_missing(
                    "control de determinismo: el Broker no envió la semilla en todas las "
                    "invocaciones (seed_status != sent). La tirada NO es reproducible."
                )
        else:
            flags.seed_fijada = pedida
        if deterministas:
            flags.determinismo_observado = Determinism.si if all(deterministas) else Determinism.no
        if not flags.seed_fijada:
            # Sin control de semilla no hay reproducibilidad exacta que perseguir
            # en este entorno: no es que no se haya demostrado, es que no está
            # disponible.
            flags.reproducibilidad_exacta = ExactReproducibility.no_disponible

    # -------------------------------------------------------------------- run

    def run(
        self,
        models: list[ModelReference],
        *,
        on_progress: Callable[[RunProgress], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> RunOutput:
        """Ejecuta la tirada completa.

        `should_cancel` se consulta antes de cada invocación. Una tirada puede
        durar horas y costar dinero: hay que poder pararla sin matar el proceso
        y perder lo ya medido. Al cancelar se conserva y se persiste todo lo
        hecho hasta ese punto, y el manifiesto lo declara.
        """
        run_id = f"run-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        # Antes de nada, qué admite este Broker: así el manifiesto nace diciendo
        # la verdad en vez de corregirse al final con el primer 422.
        self.client.probe_contract()
        manifiesto = self._build_manifest(run_id, models)
        salida = RunOutput(run_id=run_id, manifest=manifiesto)
        group = f"model-drift-{run_id}"

        filas_invocacion: list[dict[str, Any]] = []
        filas_caso: list[dict[str, Any]] = []
        deterministas: list[bool] = []
        estados_semilla: list[SeedStatus] = []

        total_casos = len(self.suite.cases)
        for modelo in models:
            resultados = RunResults(run_id=run_id, manifest=manifiesto, model_key=modelo.key)
            for indice_caso, case in enumerate(self.suite.cases):
                motivo = self._not_applicable_reason(case, modelo)
                if motivo is not None:
                    outcome = CaseOutcome(
                        case_id=case.id,
                        model_key=modelo.key,
                        metric=METRIC_PASS,
                        is_not_applicable=True,
                        not_applicable_reason=motivo,
                    )
                    resultados.outcomes[(case.id, METRIC_PASS)] = outcome
                    filas_caso.append(_case_row(run_id, outcome))
                    continue

                aciertos: list[bool] = []
                coincidencias: list[bool] = []
                latencias: list[float] = []
                costes: list[float] = []
                tokens: list[float] = []
                textos: list[str] = []
                fallos = 0

                for repeticion in range(self.config.repetitions):
                    if should_cancel is not None and should_cancel():
                        salida.aborted_reason = "cancelada a petición del usuario"
                        break
                    if self.config.budget_usd is not None and salida.cost_usd >= self.config.budget_usd:
                        salida.aborted_reason = (
                            f"presupuesto agotado ({salida.cost_usd:.4f} USD de {self.config.budget_usd:.4f})"
                        )
                        break
                    invocacion = self._invoke_case(case, modelo, group)
                    salida.invocations += 1
                    salida.cost_usd += invocacion.cost_usd
                    filas_invocacion.append(_invocation_row(run_id, case.id, modelo.key, repeticion, invocacion))
                    if on_progress is not None:
                        on_progress(
                            RunProgress(
                                model_key=modelo.key,
                                case_id=case.id,
                                case_index=indice_caso,
                                total_cases=total_casos,
                                repetition=repeticion,
                                total_repetitions=self.config.repetitions,
                                invocations=salida.invocations,
                                cost_usd=salida.cost_usd,
                            )
                        )

                    estados_semilla.append(invocacion.seed_status)
                    latencias.append(invocacion.latency_ms)
                    costes.append(invocacion.cost_usd)
                    tokens.append(float(invocacion.tokens_output))
                    if not invocacion.produced_output:
                        fallos += 1
                        continue
                    textos.append(invocacion.text)

                    # Caso capturado: se coteja contra su salida histórica
                    # congelada. Detecta CAMBIO, no calidad: la línea base no
                    # es verdad de referencia y no puede dar dirección.
                    if case.baseline_output is not None:
                        coincidencias.append(
                            normalize_text(invocacion.text) == normalize_text(case.baseline_output)
                        )

                    if case.assertions:
                        resultados_aserciones = evaluate_assertions(
                            list(case.assertions), self._evaluation_input(case, invocacion), self.services
                        )
                        acierto, _ = aggregate_pass(resultados_aserciones)
                        if acierto is not None:
                            aciertos.append(acierto)

                if salida.aborted_reason:
                    break

                # Determinismo: se MIDE comparando las repeticiones, no se supone.
                if len(textos) > 1:
                    deterministas.append(len(set(textos)) == 1)

                if case.metamorphic_relations and textos:
                    for relacion in self._run_relations(case, modelo, group, textos[0]):
                        if relacion.not_applicable:
                            continue
                        metrica = f"mr:{relacion.relation_id}"
                        agregado = aggregate_binary([relacion.passed])
                        outcome = CaseOutcome(
                            case_id=case.id,
                            model_key=modelo.key,
                            metric=metrica,
                            case_score=agregado.score,
                            case_stability=1.0,
                            repetitions=1,
                            values=list(agregado.values),
                        )
                        resultados.outcomes[(case.id, metrica)] = outcome
                        filas_caso.append(_case_row(run_id, outcome))

                if textos:
                    resultados.sample_output[case.id] = textos[0]

                for metrica, agregado, valores in (
                    (METRIC_PASS, aggregate_binary(aciertos), aciertos),
                    (METRIC_OUTPUT_MATCH, aggregate_binary(coincidencias), coincidencias),
                    (METRIC_LATENCY, aggregate_continuous(latencias), latencias),
                    (METRIC_COST, aggregate_continuous(costes), costes),
                    (METRIC_TOKENS, aggregate_continuous(tokens), tokens),
                ):
                    if not valores:
                        continue
                    inestable = agregado.stability < self.suite.unstable.min_stability
                    outcome = CaseOutcome(
                        case_id=case.id,
                        model_key=modelo.key,
                        metric=metrica,
                        case_score=agregado.score,
                        case_stability=agregado.stability,
                        repetitions=agregado.repetitions,
                        failed_repetitions=fallos,
                        # La estabilidad solo tiene sentido como criterio de
                        # exclusión sobre la métrica de acierto: una latencia
                        # variable es un hecho del modelo, no un caso roto.
                        is_unstable=inestable and metrica in {METRIC_PASS, METRIC_OUTPUT_MATCH},
                        values=list(agregado.values),
                    )
                    resultados.outcomes[(case.id, metrica)] = outcome
                    filas_caso.append(_case_row(run_id, outcome))

            salida.results[modelo.key] = resultados
            if salida.aborted_reason:
                break

        self._seal_reproducibility(manifiesto, deterministas, estados_semilla)
        for garantia in self.client.missing_guarantees:
            manifiesto.declare_missing(garantia)
        manifiesto.fixtures = self.resolver.used_hashes()
        if salida.aborted_reason:
            manifiesto.notes.append(f"tirada interrumpida: {salida.aborted_reason}")

        if self.db is not None:
            self._persist(salida, filas_invocacion, filas_caso)
        return salida

    def _persist(
        self,
        salida: RunOutput,
        filas_invocacion: list[dict[str, Any]],
        filas_caso: list[dict[str, Any]],
    ) -> None:
        assert self.db is not None
        self.db.insert(
            "run",
            {
                "run_id": salida.run_id,
                "created_at": salida.manifest.created_at.isoformat(),
                "suite_id": self.suite.id,
                "suite_hash": self.suite.hash(),
                "label": self.config.label,
                "manifest_json": salida.manifest.to_json(),
                "status": "aborted" if salida.aborted_reason else "completed",
            },
        )
        self.db.insert_many("invocation", filas_invocacion)
        self.db.insert_many("case_result", filas_caso)


def _invocation_row(
    run_id: str, case_id: str, model_key: str, repetition: int, outcome: InvocationOutcome
) -> dict[str, Any]:
    return {
        "invocation_id": f"inv-{uuid.uuid4().hex[:16]}",
        "run_id": run_id,
        "case_id": case_id,
        "model_key": model_key,
        "repetition": repetition,
        "variant_id": None,
        "broker_task_id": outcome.task_id or None,
        "ok": outcome.ok,
        # La salida CRUDA, siempre. Es lo que permite re-juzgar sin reinvocar.
        "output_text": outcome.text,
        "tool_calls_json": outcome.tool_calls,
        "tokens_input": outcome.tokens_input,
        "tokens_output": outcome.tokens_output,
        "cost_usd": outcome.cost_usd,
        "latency_ms": outcome.latency_ms,
        "model_served": outcome.model_served.key if outcome.model_served else None,
        "fallback_used": outcome.fallback_used,
        "fingerprint_json": outcome.fingerprint.model_dump(mode="json"),
        "applied_params_json": outcome.applied_params,
        "error_code": outcome.error_code,
        "error_message": outcome.error_message,
        "created_at": datetime.now(UTC).isoformat(),
    }


def _case_row(run_id: str, outcome: CaseOutcome) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "case_id": outcome.case_id,
        "model_key": outcome.model_key,
        "metric": outcome.metric,
        "case_score": outcome.case_score,
        "case_stability": outcome.case_stability,
        "repetitions": outcome.repetitions,
        "failed_repetitions": outcome.failed_repetitions,
        "is_unstable": outcome.is_unstable,
        "is_not_applicable": outcome.is_not_applicable,
        "not_applicable_reason": outcome.not_applicable_reason,
        "values_json": list(outcome.values),
    }
