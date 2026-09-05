"""La tirada: recorrido de modelos y casos, agregación y persistencia.

`run()` se lee de arriba abajo como el enunciado de lo que hace una tirada:
prepara el manifiesto, recorre los modelos, sella lo que ocurrió y guarda. El
detalle de cada nivel —un modelo, un caso, sus repeticiones— está en su propio
método, porque son tres decisiones distintas y antes estaban en el mismo bucle.

Regla 2 del motor, aplicada aquí: la unidad es el caso. Las N repeticiones se
agregan en este módulo y lo que sale hacia la comparación es **una** observación
por caso, no N.
"""
from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from model_drift.broker_contracts import ModelReference, SeedStatus
from model_drift.cases import Case
from model_drift.comparison import RunResults
from model_drift.domain import (
    METRIC_COST,
    METRIC_LATENCY,
    METRIC_OUTPUT_MATCH,
    METRIC_PASS,
    METRIC_TOKENS,
    CaseOutcome,
)
from model_drift.manifest import BrokerContractFlags, RunManifest
from model_drift.oracles import aggregate_pass, evaluate_assertions, normalize_text
from model_drift.runner.filas import _case_row, _invocation_row
from model_drift.runner.manifiesto import ManifiestoMixin
from model_drift.runner.tipos import RunOutput, RunProgress
from model_drift.statistics import aggregate_binary, aggregate_continuous


@dataclass
class _Acumulado:
    """Lo que la tirada va reuniendo mientras recorre modelos y casos.

    Existe para que los métodos de cada nivel no tengan que devolverse listas
    unos a otros: las filas se escriben una sola vez, al final, y el
    determinismo y la semilla se miden sobre la tirada entera.
    """

    filas_invocacion: list[dict[str, Any]] = field(default_factory=list)
    filas_caso: list[dict[str, Any]] = field(default_factory=list)
    #: Un booleano por caso con más de una repetición: ¿salió siempre lo mismo?
    deterministas: list[bool] = field(default_factory=list)
    estados_semilla: list[SeedStatus] = field(default_factory=list)


@dataclass
class _Medicion:
    """Las N repeticiones de un caso contra un modelo, sin agregar todavía."""

    aciertos: list[bool] = field(default_factory=list)
    coincidencias: list[bool] = field(default_factory=list)
    latencias: list[float] = field(default_factory=list)
    costes: list[float] = field(default_factory=list)
    tokens: list[float] = field(default_factory=list)
    textos: list[str] = field(default_factory=list)
    fallos: int = 0


class Runner(ManifiestoMixin):
    """Ejecuta una suite contra un conjunto de modelos."""

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
        self._sellar_contrato_broker(manifiesto)
        salida = RunOutput(run_id=run_id, manifest=manifiesto)
        acumulado = _Acumulado()
        group = f"model-drift-{run_id}"

        for modelo in models:
            salida.results[modelo.key] = self._ejecutar_modelo(
                modelo,
                run_id=run_id,
                manifiesto=manifiesto,
                salida=salida,
                acumulado=acumulado,
                group=group,
                on_progress=on_progress,
                should_cancel=should_cancel,
            )
            if salida.aborted_reason:
                break

        self._cerrar_manifiesto(manifiesto, salida, acumulado)
        if self.db is not None:
            self._persist(salida, acumulado.filas_invocacion, acumulado.filas_caso)
        return salida

    # ----------------------------------------------------------------- niveles

    def _ejecutar_modelo(
        self,
        modelo: ModelReference,
        *,
        run_id: str,
        manifiesto: RunManifest,
        salida: RunOutput,
        acumulado: _Acumulado,
        group: str,
        on_progress: Callable[[RunProgress], None] | None,
        should_cancel: Callable[[], bool] | None,
    ) -> RunResults:
        """Recorre la suite entera contra un modelo."""
        resultados = RunResults(run_id=run_id, manifest=manifiesto, model_key=modelo.key)
        total_casos = len(self.suite.cases)

        for indice_caso, case in enumerate(self.suite.cases):
            motivo = self._not_applicable_reason(case, modelo)
            if motivo is not None:
                self._registrar_no_aplicable(case, modelo, motivo, run_id, resultados, acumulado)
                continue

            medicion = self._medir_caso(
                case,
                modelo,
                indice_caso=indice_caso,
                total_casos=total_casos,
                run_id=run_id,
                salida=salida,
                acumulado=acumulado,
                group=group,
                on_progress=on_progress,
                should_cancel=should_cancel,
            )
            if salida.aborted_reason:
                break

            # Determinismo: se MIDE comparando las repeticiones, no se supone.
            if len(medicion.textos) > 1:
                acumulado.deterministas.append(len(set(medicion.textos)) == 1)

            if case.metamorphic_relations and medicion.textos:
                self._registrar_relaciones(
                    case, modelo, group, medicion.textos[0], run_id, resultados, acumulado
                )
            if medicion.textos:
                resultados.sample_output[case.id] = medicion.textos[0]

            self._registrar_metricas(case, modelo, medicion, run_id, resultados, acumulado)

        return resultados

    def _medir_caso(
        self,
        case: Case,
        modelo: ModelReference,
        *,
        indice_caso: int,
        total_casos: int,
        run_id: str,
        salida: RunOutput,
        acumulado: _Acumulado,
        group: str,
        on_progress: Callable[[RunProgress], None] | None,
        should_cancel: Callable[[], bool] | None,
    ) -> _Medicion:
        """Las N repeticiones de un caso. Aquí se para si toca parar."""
        medicion = _Medicion()

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
            acumulado.filas_invocacion.append(
                _invocation_row(run_id, case.id, modelo.key, repeticion, invocacion)
            )
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

            acumulado.estados_semilla.append(invocacion.seed_status)
            medicion.latencias.append(invocacion.latency_ms)
            medicion.costes.append(invocacion.cost_usd)
            medicion.tokens.append(float(invocacion.tokens_output))
            if not invocacion.produced_output:
                medicion.fallos += 1
                continue
            medicion.textos.append(invocacion.text)

            # Caso capturado: se coteja contra su salida histórica congelada.
            # Detecta CAMBIO, no calidad: la línea base no es verdad de
            # referencia y no puede dar dirección.
            if case.baseline_output is not None:
                medicion.coincidencias.append(
                    normalize_text(invocacion.text) == normalize_text(case.baseline_output)
                )

            if case.assertions:
                resultados_aserciones = evaluate_assertions(
                    list(case.assertions), self._evaluation_input(case, invocacion), self.services
                )
                acierto, _ = aggregate_pass(resultados_aserciones)
                if acierto is not None:
                    medicion.aciertos.append(acierto)

        return medicion

    # ------------------------------------------------------------- resultados

    def _registrar_no_aplicable(
        self,
        case: Case,
        modelo: ModelReference,
        motivo: str,
        run_id: str,
        resultados: RunResults,
        acumulado: _Acumulado,
    ) -> None:
        outcome = CaseOutcome(
            case_id=case.id,
            model_key=modelo.key,
            metric=METRIC_PASS,
            is_not_applicable=True,
            not_applicable_reason=motivo,
        )
        resultados.outcomes[(case.id, METRIC_PASS)] = outcome
        acumulado.filas_caso.append(_case_row(run_id, outcome))

    def _registrar_relaciones(
        self,
        case: Case,
        modelo: ModelReference,
        group: str,
        texto_referencia: str,
        run_id: str,
        resultados: RunResults,
        acumulado: _Acumulado,
    ) -> None:
        for relacion in self._run_relations(case, modelo, group, texto_referencia):
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
            acumulado.filas_caso.append(_case_row(run_id, outcome))

    def _registrar_metricas(
        self,
        case: Case,
        modelo: ModelReference,
        medicion: _Medicion,
        run_id: str,
        resultados: RunResults,
        acumulado: _Acumulado,
    ) -> None:
        """Agrega las repeticiones a una observación por caso y métrica."""
        for metrica, agregado, valores in (
            (METRIC_PASS, aggregate_binary(medicion.aciertos), medicion.aciertos),
            (METRIC_OUTPUT_MATCH, aggregate_binary(medicion.coincidencias), medicion.coincidencias),
            (METRIC_LATENCY, aggregate_continuous(medicion.latencias), medicion.latencias),
            (METRIC_COST, aggregate_continuous(medicion.costes), medicion.costes),
            (METRIC_TOKENS, aggregate_continuous(medicion.tokens), medicion.tokens),
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
                failed_repetitions=medicion.fallos,
                # La estabilidad solo tiene sentido como criterio de exclusión
                # sobre la métrica de acierto: una latencia variable es un hecho
                # del modelo, no un caso roto.
                is_unstable=inestable and metrica in {METRIC_PASS, METRIC_OUTPUT_MATCH},
                values=list(agregado.values),
            )
            resultados.outcomes[(case.id, metrica)] = outcome
            acumulado.filas_caso.append(_case_row(run_id, outcome))

    def _sellar_contrato_broker(self, manifiesto: RunManifest) -> None:
        """Copia al manifiesto lo que promete el Broker EN MARCHA.

        Se sella al arrancar, no al terminar: una tirada leída dentro de seis
        meses tiene que poder decir qué se pudo demostrar entonces. El mismo
        número contra un Broker 2.9 y contra uno 2.10 no vale lo mismo, porque
        solo en el segundo se sabe que la huella era la de la llamada que
        respondió y que el prompt llegó sin podar.
        """
        manifiesto.broker_contract = BrokerContractFlags(
            version=self.client.contract_version,
            invocation_contract=self.client.supports_invocation_contract,
            prompt_compression_echo=self.client.supports_prompt_compression_echo,
            canonical_artifacts=self.client.supports_canonical_artifacts,
            auxiliary_invocations=self.client.performs_auxiliary_invocations,
            auxiliary_invocations_optout=self.client.supports_auxiliary_optout,
        )

    # ------------------------------------------------------------ cierre

    def _cerrar_manifiesto(
        self, manifiesto: RunManifest, salida: RunOutput, acumulado: _Acumulado
    ) -> None:
        """Sella las banderas y anota lo que faltó, incluida la interrupción."""
        self._seal_reproducibility(manifiesto, acumulado.deterministas, acumulado.estados_semilla)
        for garantia in self.client.missing_guarantees:
            manifiesto.declare_missing(garantia)
        manifiesto.fixtures = self.resolver.used_hashes()
        if salida.aborted_reason:
            manifiesto.notes.append(f"tirada interrumpida: {salida.aborted_reason}")

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
