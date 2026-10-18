"""Invocación: construcción del cuerpo, envío, espera y lectura del resultado.

Dos invariantes que se sostienen aquí y en ningún otro sitio:
`fallback_allowed = False` en toda petición, y el marcado del tráfico como
excluido del aprendizaje. Si el Broker no admite alguno, se declara la
garantía perdida en vez de fingir que se cumplió.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from model_drift.broker_client.huella import HuellaMixin, _huella_publicada
from model_drift.broker_client.tipos import ORIGIN, InvocationOutcome
from model_drift.broker_contracts import (
    BrokerError,
    InferenceKind,
    InvocationTelemetry,
    ModelReference,
    PromptCompression,
    SeedStatus,
    TaskArtifact,
    TaskState,
)


class InvocacionMixin(HuellaMixin):
    """Ciclo completo de una invocación, del cuerpo al `InvocationOutcome`."""

    def _build_request(
        self,
        *,
        prompt: str,
        target: ModelReference,
        system: str | None,
        params: dict[str, Any],
        group: str,
        request_id: str,
        inference_kind: InferenceKind,
        json_schema: dict[str, Any] | None,
        timeout_seconds: int | None,
        cloud_allowed: bool | None = None,
    ) -> dict[str, Any]:
        generation: dict[str, Any] = {}
        if params.get("temperature") is not None:
            generation["temperature"] = params["temperature"]
        if params.get("max_output_tokens") is not None:
            generation["max_output_tokens"] = params["max_output_tokens"]
        if self.config.supports_seed:
            # Se intentan aunque el contrato verificado no los admita: si el
            # Broker ya incorporó el Cambio 1, la tirada gana control sin tocar
            # configuración; si no, el 422 lo descubre y se degrada.
            if params.get("seed") is not None:
                generation["seed"] = params["seed"]
            if params.get("top_p") is not None:
                generation["top_p"] = params["top_p"]

        metadata: dict[str, Any] = {"origin": ORIGIN, "run_group": group}

        cuerpo: dict[str, Any] = {
            "idempotency_key": f"md-{uuid.uuid4().hex}",
            "request_id": request_id,
            "inference_kind": inference_kind.value,
            "content": {"prompt": prompt, "metadata": metadata},
            "generation": generation,
            "model_requirements": {
                "target_model": target.model_dump(mode="json"),
                # Nunca True. Una comparación con sustitución permitida no
                # compara lo que dice comparar.
                "fallback_allowed": False,
            },
            "execution": {"strategy": "single", "preset": "fast"},
            "group": group,
        }
        if system:
            # El Broker no escribe system prompt en `single` (es transparente por
            # contrato), así que el rol se antepone al propio prompt para que
            # llegue al modelo sin que el Broker lo reescriba.
            cuerpo["content"]["prompt"] = f"{system}\n\n{prompt}"
        if cloud_allowed is not None:
            cuerpo["model_requirements"]["cloud_allowed"] = cloud_allowed
        if json_schema is not None:
            cuerpo["output"] = {"format": "json", "json_schema": json_schema}
        if timeout_seconds is not None:
            cuerpo["execution"]["timeout_seconds"] = timeout_seconds
        if self.config.supports_learning_exclusion:
            cuerpo["exclude_from_model_learning"] = True
        if self.config.supports_prompt_compression:
            # Una medición sobre un prompt podado no compara lo que dice
            # comparar: el modelo no vio el caso, vio un resumen del caso. Desde
            # el contrato 2.10 esto además tiene acuse de recibo (§8.5), que es
            # lo que se comprueba en `_verificar_compresion`.
            cuerpo["prompt_compression"] = PromptCompression.off.value
        if self.config.request_exclusive_content and self.supports_auxiliary_optout:
            # Contrato 2.10, §8.4. La garantía implícita (`target_model` +
            # `fallback_allowed: false`) ya cubre esta tarea, pero solo contra
            # un Broker 2.10 y sin dejar rastro de que se pidió. Declararlo lo
            # convierte en algo que se puede leer en el cuerpo de la petición.
            cuerpo["auxiliary_invocations"] = False
        return cuerpo

    def _submit(self, cuerpo: dict[str, Any]) -> str:
        try:
            respuesta = self._request("POST", "/api/v1/tasks", json=cuerpo)
        except BrokerError as error:
            if error.status_code == 422 and self.config.supports_seed and "generation" in str(error):
                # Degradación elegante: el contrato no admite semilla todavía.
                self.config.supports_seed = False
                self._declare_missing(
                    "control de determinismo: el Broker no admite generation.seed/top_p "
                    "(Cambio 1 del contrato de evaluación). La tirada NO es reproducible."
                )
                cuerpo["generation"].pop("seed", None)
                cuerpo["generation"].pop("top_p", None)
                respuesta = self._request("POST", "/api/v1/tasks", json=cuerpo)
            elif error.status_code == 422 and "auxiliary_invocations" in str(error):
                # El Broker anuncia el opt-out y lo rechaza: contradicción suya,
                # pero la tirada no se cae por eso. Se quita y se declara.
                self.supports_auxiliary_optout = False
                self._declare_missing(
                    "exclusividad de contenido: el Broker rechazó auxiliary_invocations "
                    "(contrato 2.10, §8.4) pese a anunciar el opt-out. El prompt puede "
                    "verlo un modelo distinto del medido, bajo el mismo task_id."
                )
                cuerpo.pop("auxiliary_invocations", None)
                respuesta = self._request("POST", "/api/v1/tasks", json=cuerpo)
            elif error.status_code == 422 and self.config.supports_prompt_compression and (
                "prompt_compression" in str(error)
            ):
                self.config.supports_prompt_compression = False
                self._declare_missing(
                    "prompt sin podar: el Broker no admite prompt_compression en la petición. "
                    "El prompt puede llegar al modelo podado y la comparación no es limpia."
                )
                cuerpo.pop("prompt_compression", None)
                respuesta = self._request("POST", "/api/v1/tasks", json=cuerpo)
            elif error.status_code == 422 and self.config.supports_learning_exclusion:
                self.config.supports_learning_exclusion = False
                self._declare_missing(
                    "aislamiento del aprendizaje: el Broker no admite exclude_from_model_learning "
                    "(Cambio 4 del contrato). Esta tirada altera las estadísticas del router."
                )
                cuerpo.pop("exclude_from_model_learning", None)
                respuesta = self._request("POST", "/api/v1/tasks", json=cuerpo)
            else:
                raise
        payload = respuesta.json()
        return str(payload["task_id"])

    def invocations(self, task_id: str) -> list[InvocationTelemetry]:
        """Telemetría por invocación (contrato 2.9, §8.1).

        Es lo que permite saber los parámetros **efectivos**, con qué huella se
        sirvió y si la semilla llegó a enviarse. Antes esto solo estaba bajo el
        panel de administración, que es una superficie administrativa con otra
        autenticación: un cliente no debía depender de ella.
        """
        try:
            payload = self._request("GET", f"/api/v1/tasks/{task_id}/invocations").json()
        except BrokerError:
            self._declare_missing(
                "telemetría por invocación: el Broker no expone "
                "/api/v1/tasks/{id}/invocations (Cambio 3 del contrato de evaluación). "
                "Los parámetros efectivos y la huella de cada llamada no se pueden comprobar."
            )
            return []
        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            return []
        telemetria: list[InvocationTelemetry] = []
        for item in items:
            if isinstance(item, dict):
                try:
                    telemetria.append(InvocationTelemetry.model_validate(item))
                except Exception:
                    continue
        return telemetria

    def artifacts(self, task_id: str) -> list[TaskArtifact]:
        """Ficheros que produjo la tarea (contrato 2.10, §8.3).

        Es la vía **canónica** para recoger el entregable: viene tipado y con su
        `sha256`. `result.assistant_content` sigue siendo la comodidad para
        enseñar la respuesta, pero `result` no tiene esquema en el OpenAPI, así
        que una medición que tiene que rendir cuentas después no puede cerrarse
        solo sobre él.
        """
        if not self.supports_task_artifacts:
            return []
        try:
            payload = self._request("GET", f"/api/v1/tasks/{task_id}/artifacts").json()
        except BrokerError:
            self._declare_missing(
                "entregable con contrato: el Broker no expone "
                "/api/v1/tasks/{id}/artifacts (contrato 2.10, §8.3)."
            )
            return []
        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            return []
        artefactos: list[TaskArtifact] = []
        for item in items:
            if isinstance(item, dict):
                try:
                    artefactos.append(TaskArtifact.model_validate(item))
                except Exception:
                    continue
        return artefactos

    def _sha256_entregable(self, task_id: str) -> str | None:
        """`sha256` del artefacto marcado `final: true` (§8.3).

        Se filtra por el booleano y no por `artifact_type`: la lista de tipos
        crece con cada estrategia nueva y cerrar con «el primero de la lista»
        cerraría con el PNG que acompaña, no con la respuesta.
        """
        if not (self.supports_task_artifacts and self.supports_canonical_artifacts):
            return None
        for artefacto in self.artifacts(task_id):
            if artefacto.final and artefacto.sha256:
                return artefacto.sha256
        return None

    def _verificar_compresion(
        self,
        contractuales: list[InvocationTelemetry],
        target: ModelReference,
    ) -> PromptCompression | None:
        """Comprueba que el prompt llegó al modelo sin podar (§8.5).

        La aserción es sobre las invocaciones **contractuales**: el Broker fuerza
        `off` en las que procesan contenido generado, así que mezclarlas
        confundiría una política con un automatismo. Sin
        `prompt_compression_echo` no hay acuse de recibo y se devuelve `None`:
        no consta, que no es lo mismo que se cumplió.
        """
        if not self.supports_prompt_compression_echo:
            return None
        efectivas = [
            item.compression_effective
            for item in contractuales
            # `null` en llamadas que no envían prompt de usuario: nada que podar.
            if item.compression_effective is not None
        ]
        if not efectivas:
            return None
        podadas = [valor for valor in efectivas if valor is not PromptCompression.off]
        if podadas:
            self._declare_missing(
                f"prompt sin podar en {target.key}: se pidió prompt_compression=off y el "
                f"Broker aplicó {', '.join(sorted({item.value for item in podadas}))} "
                "(§8.5). El modelo no vio el caso íntegro."
            )
            return podadas[0]
        return PromptCompression.off

    def _poll(self, task_id: str) -> TaskState:
        limite = time.monotonic() + self.config.max_wait_seconds
        while True:
            estado = TaskState.model_validate(self._request("GET", f"/api/v1/tasks/{task_id}").json())
            if estado.is_terminal:
                return estado
            if time.monotonic() > limite:
                raise BrokerError(
                    f"la tarea {task_id} no terminó en {self.config.max_wait_seconds:.0f} s",
                    code="MODEL_DRIFT_TIMEOUT",
                )
            time.sleep(self.config.poll_interval_seconds)

    def invoke(
        self,
        *,
        prompt: str,
        target: ModelReference,
        system: str | None = None,
        params: dict[str, Any] | None = None,
        group: str = "model-drift",
        request_id: str | None = None,
        inference_kind: InferenceKind = InferenceKind.chat,
        json_schema: dict[str, Any] | None = None,
        timeout_seconds: int | None = None,
        cloud_allowed: bool | None = None,
    ) -> InvocationOutcome:
        """Ejecuta una invocación y devuelve salida cruda más telemetría."""
        params = params or {}
        cuerpo = self._build_request(
            prompt=prompt,
            target=target,
            system=system,
            params=params,
            group=group,
            request_id=request_id or f"md-{uuid.uuid4().hex[:12]}",
            inference_kind=inference_kind,
            json_schema=json_schema,
            timeout_seconds=timeout_seconds,
            cloud_allowed=cloud_allowed,
        )
        arranque = time.monotonic()
        try:
            task_id = self._submit(cuerpo)
            estado = self._poll(task_id)
        except BrokerError as error:
            return InvocationOutcome(
                task_id="",
                ok=False,
                error_code=error.code or ("AUTH" if error.is_auth else "BROKER_ERROR"),
                error_message=str(error),
                auth_failed=error.is_auth,
                latency_ms=(time.monotonic() - arranque) * 1000.0,
            )
        latencia = (time.monotonic() - arranque) * 1000.0

        if not estado.succeeded:
            return InvocationOutcome(
                task_id=estado.task_id,
                ok=False,
                error_code=estado.error_code or estado.status,
                error_message=str((estado.error or {}).get("message", "")),
                latency_ms=latencia,
            )

        uso = estado.usage
        servido = estado.model_used
        resultado = InvocationOutcome(
            task_id=estado.task_id,
            ok=True,
            text=estado.assistant_content,
            embedding=estado.embedding,
            tool_calls=estado.pending_tool_calls,
            model_served=servido,
            fallback_used=estado.fallback_used,
            tokens_input=int(uso.get("tokens_input") or 0),
            tokens_output=int(uso.get("tokens_output") or 0),
            cost_usd=float(uso.get("cost_usd") or uso.get("cost_actual_usd") or 0.0),
            latency_ms=latencia,
            applied_params=dict(cuerpo["generation"]),
        )
        # Telemetría por invocación: parámetros EFECTIVOS, huella vigente al
        # servir y qué pasó con la semilla. Sustituye a lo que se pidió, que es
        # lo único que se sabía antes de 2.9.
        telemetria = self.invocations(estado.task_id) if self.supports_invocation_telemetry else []
        contractuales = [item for item in telemetria if item.is_contractual]
        auxiliares = [item for item in telemetria if not item.is_contractual]
        resultado.auxiliary_invocations = len(auxiliares)
        resultado.auxiliary_roles = sorted({item.role for item in auxiliares})
        if auxiliares:
            # No es un fallo: el Broker mide su catálogo y lo declara. Pero un
            # contenido que se quería exclusivo lo ha visto otro modelo, y eso
            # tiene que constar en el manifiesto de la tirada.
            self._declare_missing(
                f"exclusividad de contenido en {target.key}: el Broker lanzó "
                f"{len(auxiliares)} invocación(es) no contractual(es) "
                f"({', '.join(resultado.auxiliary_roles)}) bajo el mismo task_id (§8.4)."
            )

        # La ÚLTIMA contractual, no la última a secas: bajo un mismo `task_id`
        # conviven las llamadas que ejecutan la tarea y las que el Broker lanza
        # por su cuenta (§8.1). Leer un `shadow_probe` como si fuera la
        # respuesta atribuye a este modelo la huella y los parámetros de otro.
        principal = contractuales[-1] if contractuales else None
        if principal is not None:
            resultado.seed_status = principal.seed_status
            if principal.generation:
                resultado.applied_params = dict(principal.generation)
            if principal.latency_ms is not None:
                resultado.latency_ms = principal.latency_ms
            resultado.fingerprint = _huella_publicada(principal.execution_fingerprint)
            resultado.content_source = principal.content_source
            if principal.seed_status is SeedStatus.unsupported:
                self._declare_missing(
                    f"determinismo en {target.key}: el endpoint no admite semilla "
                    "(seed_status=unsupported), así que pedirla no cambió nada."
                )
            if principal.content_source == "reasoning_content":
                # La respuesta es válida, pero el modelo no la entrega donde
                # debe. Al comparar modelos eso no es un detalle de formato.
                self._declare_missing(
                    f"entrega limpia en {target.key}: el modelo dejó `content` vacío y el "
                    "Broker rescató su razonamiento (content_source=reasoning_content, §8.1)."
                )
        if not resultado.fingerprint.components:
            resultado.fingerprint = self.fingerprint_for(target, resultado)

        resultado.prompt_compression_effective = self._verificar_compresion(contractuales, target)
        resultado.deliverable_sha256 = self._sha256_entregable(estado.task_id)

        # Fijación dura: si sirvió otro modelo, la invocación no vale para
        # comparar aunque haya devuelto un texto perfecto.
        if servido is not None and servido.key != target.key:
            resultado.ok = False
            resultado.error_code = "MODEL_DRIFT_TARGET_MISMATCH"
            resultado.error_message = f"se pidió {target.key} y respondió {servido.key}"
        elif resultado.fallback_used:
            resultado.ok = False
            resultado.error_code = "MODEL_DRIFT_FALLBACK_USED"
            resultado.error_message = "el Broker declaró fallback pese a fallback_allowed=False"
        return resultado
