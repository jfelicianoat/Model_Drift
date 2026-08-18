"""Cliente HTTP del AI Broker.

Model_Drift no ejecuta inferencia: la pide. Este módulo es la única puerta al
Broker y concentra tres responsabilidades que en otros sitios se mezclan:

1. **Fijar el modelo de verdad.** Toda invocación va con `target_model` y
   `fallback_allowed = False`. Una comparación que permita sustitución no
   compara lo que dice comparar. Si el Broker sirviera otro modelo, se detecta
   y se marca la invocación como no utilizable.
2. **No contaminar la operación del Broker.** El tráfico va marcado en
   `content.metadata.origin` y con `exclude_from_model_learning`, para que las
   invocaciones de medición no muevan el score del router adaptativo ni acerquen
   un modelo a la cuarentena. Si el Broker no admite el marcador, se detecta al
   primer 422 y se **declara la garantía perdida**.
3. **Degradar con honestidad.** Lo mismo con `seed`/`top_p` y con la huella de
   ejecución: contra un Broker que los tiene, se usan; contra uno que no, se
   anota lo que se perdió en vez de fingir reproducibilidad o identidad.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

from model_drift.broker_contracts import (
    BrokerError,
    InferenceKind,
    InvocationTelemetry,
    ModelCatalogEntry,
    ModelReference,
    SeedStatus,
    TaskState,
)
from model_drift.fingerprint import (
    ComponentState,
    ExecutionFingerprint,
    FingerprintBuilder,
    IdentityLayer,
)

#: Marca de origen del tráfico de evaluación. El Broker la trata como metadata
#: opaca y permite filtrar por ella en el listado de tareas.
ORIGIN = "model_drift"


@dataclass
class BrokerConfig:
    base_url: str = "http://192.168.1.52:8765"
    admin_token: str | None = None
    timeout_seconds: float = 30.0
    poll_interval_seconds: float = 2.0
    max_wait_seconds: float = 900.0
    #: El Broker acepta `generation.seed`. Se descubre solo al primer 422 y no
    #: hace falta configurarlo a mano.
    supports_seed: bool = True
    #: El Broker admite marcar tráfico como excluido del aprendizaje (Cambio 4).
    #: Se intenta por defecto: contra un Broker que ya lo tiene, la medición
    #: deja de contaminar su router adaptativo sin que nadie configure nada.
    supports_learning_exclusion: bool = True


@dataclass
class InvocationOutcome:
    """Una invocación completa, con su salida cruda y su telemetría."""

    task_id: str
    ok: bool
    text: str = ""
    embedding: list[float] | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    model_served: ModelReference | None = None
    fallback_used: bool = False
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    error_code: str | None = None
    error_message: str = ""
    fingerprint: ExecutionFingerprint = field(default_factory=ExecutionFingerprint)
    applied_params: dict[str, Any] = field(default_factory=dict)
    #: Qué hizo el Broker con la semilla. `not_requested` mientras no haya
    #: telemetría que lo diga: no se supone que se aplicó.
    seed_status: SeedStatus = SeedStatus.not_requested

    @property
    def produced_output(self) -> bool:
        """Si hay salida evaluable. Un fallo de ejecución no es una respuesta mala."""
        return self.ok and bool(self.text or self.embedding)


class BrokerClient:
    """Cliente síncrono. Una tirada es un lote largo, no una interacción."""

    def __init__(self, config: BrokerConfig | None = None, *, client: httpx.Client | None = None) -> None:
        self.config = config or BrokerConfig()
        self._client = client or httpx.Client(
            base_url=self.config.base_url.rstrip("/"),
            timeout=self.config.timeout_seconds,
        )
        self._owns_client = client is None
        #: Garantías que no se pudieron obtener. Va tal cual al manifiesto.
        self.missing_guarantees: list[str] = []
        self._catalog_cache: list[ModelCatalogEntry] | None = None
        self._features_cache: dict[str, dict[str, Any]] = {}
        #: Se rellenan al sondear el contrato (`probe_contract`).
        self.contract_version: str | None = None
        self.supports_invocation_telemetry = False

    # ---------------------------------------------------------------- infra

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> BrokerClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _headers(self) -> dict[str, str]:
        if self.config.admin_token:
            return {"X-Admin-Token": self.config.admin_token}
        return {}

    def _declare_missing(self, guarantee: str) -> None:
        if guarantee not in self.missing_guarantees:
            self.missing_guarantees.append(guarantee)

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._client.request(method, url, headers=self._headers(), **kwargs)
        except httpx.HTTPError as error:
            raise BrokerError(f"no se pudo contactar con el Broker: {error}") from error
        if response.status_code >= 400:
            codigo: str | None = None
            mensaje = response.text[:400]
            try:
                payload = response.json()
                if isinstance(payload, dict):
                    detalle = payload.get("detail")
                    if isinstance(detalle, dict):
                        codigo = str(detalle.get("code") or "") or None
                        mensaje = str(detalle.get("message") or mensaje)
                    elif isinstance(detalle, str):
                        mensaje = detalle
            except ValueError:
                pass
            raise BrokerError(mensaje, code=codigo, status_code=response.status_code)
        return response

    # ------------------------------------------------------------- catálogo

    def capabilities(self) -> dict[str, Any]:
        payload = self._request("GET", "/api/v1/capabilities").json()
        return payload if isinstance(payload, dict) else {}

    def probe_contract(self) -> dict[str, bool]:
        """Comprueba QUÉ admite este Broker antes de dar nada por hecho.

        Las banderas de `BrokerConfig` nacen optimistas y se corrigen solas al
        primer 422, lo cual sirve para no romper una tirada pero llega tarde
        para avisar: cuando el aviso importa —«esto va a contaminar las
        estadísticas del router»— todavía no se ha enviado nada.

        Se pregunta a `/api/v1/capabilities`, que es donde el contrato dice que
        se anuncian, en vez de deducirlo del esquema de OpenAPI. Si el Broker no
        responde, se deja el optimismo y la red de seguridad del 422 sigue
        puesta, pero se declara que no se pudo comprobar.
        """
        try:
            capacidades = self.capabilities()
        except BrokerError:
            self._declare_missing(
                "no se pudieron leer las capacidades del Broker (/api/v1/capabilities): "
                "su contrato se descubrirá al vuelo, no antes de empezar"
            )
            return {
                "seed": self.config.supports_seed,
                "learning_exclusion": self.config.supports_learning_exclusion,
            }

        self.contract_version = str(capacidades.get("contract_version") or "") or None
        # `seed` no tiene bandera propia: llegó en 2.9 junto a las que sí la
        # tienen, así que su presencia se infiere de la versión anunciada.
        self.config.supports_seed = bool(capacidades.get("execution_fingerprint")) or (
            self.contract_version is not None and self.contract_version >= "2.9"
        )
        self.config.supports_learning_exclusion = bool(capacidades.get("exclude_from_model_learning"))
        self.supports_invocation_telemetry = bool(capacidades.get("invocation_telemetry"))
        if not self.config.supports_seed:
            self._declare_missing(
                "control de determinismo: el Broker no admite generation.seed "
                "(Cambio 1 del contrato de evaluación). La tirada NO es reproducible."
            )
        if not self.config.supports_learning_exclusion:
            self._declare_missing(
                "aislamiento del aprendizaje: el Broker no admite exclude_from_model_learning "
                "(Cambio 4 del contrato). Esta tirada altera las estadísticas del router."
            )
        return {
            "seed": self.config.supports_seed,
            "learning_exclusion": self.config.supports_learning_exclusion,
        }

    def catalog(self, *, refresh: bool = False) -> list[ModelCatalogEntry]:
        """Catálogo de modelos: una sola petición a `/models/availability`.

        La lista viene en **`items`** (el Broker devuelve
        `{checked_at, items, counts}`). Aquí NO se pide la matriz de features:
        `/models/context` es por modelo y con 163 entradas serían 163
        peticiones para datos que casi nunca se van a usar. Las features se
        piden bajo demanda en `features_for`, que es donde de verdad hacen
        falta: al decidir si un caso aplica a un modelo.
        """
        if self._catalog_cache is not None and not refresh:
            return self._catalog_cache
        if refresh:
            self._features_cache.clear()

        payload = self._request("GET", "/api/v1/models/availability").json()
        crudos: Any = payload
        if isinstance(payload, dict):
            # `items` es el contrato actual; `models` se acepta por si se
            # apunta a un Broker anterior a la entrega de evaluación.
            for clave in ("items", "models"):
                if isinstance(payload.get(clave), list):
                    crudos = payload[clave]
                    break
        if not isinstance(crudos, list):
            raise BrokerError(
                "el catálogo del Broker no trae ninguna lista de modelos "
                f"(claves recibidas: {sorted(payload) if isinstance(payload, dict) else type(payload).__name__}). "
                "Puede que el contrato haya cambiado: revisa broker_contracts.py."
            )

        entradas: list[ModelCatalogEntry] = []
        descartadas = 0
        for item in crudos:
            if not isinstance(item, dict):
                descartadas += 1
                continue
            try:
                entradas.append(ModelCatalogEntry.model_validate(item))
            except Exception:
                # Una entrada con forma inesperada se cuenta, no se traga: un
                # catálogo que encoge en silencio es peor que uno que falla.
                descartadas += 1
        if descartadas:
            self._declare_missing(
                f"{descartadas} entradas del catálogo no encajan en el contrato conocido y se han descartado"
            )
        if not entradas:
            raise BrokerError(
                f"el Broker devolvió {len(crudos)} entradas de catálogo y ninguna es utilizable"
            )

        self._catalog_cache = entradas
        return entradas

    def features_for(self, reference: ModelReference) -> dict[str, Any]:
        """Matriz de capacidades de UN modelo, con la procedencia del Broker.

        `/api/v1/models/context` exige proveedor, deployment y modelo como
        parámetros: no es un listado. Se cachea porque una suite pregunta por
        el mismo modelo una vez por caso.
        """
        if reference.key in self._features_cache:
            return self._features_cache[reference.key]
        try:
            payload = self._request(
                "GET",
                "/api/v1/models/context",
                params={
                    "provider": reference.provider,
                    "deployment": reference.deployment,
                    "model": reference.model,
                },
            ).json()
        except BrokerError as error:
            self._declare_missing(
                f"matriz de capacidades de {reference.key}: /api/v1/models/context respondió {error}. "
                "Las capacidades se derivan solo de la lista plana de availability."
            )
            self._features_cache[reference.key] = {}
            return {}
        features = payload.get("features") if isinstance(payload, dict) else None
        resultado = features if isinstance(features, dict) else {}
        self._features_cache[reference.key] = resultado
        return resultado

    def entry_for(self, reference: ModelReference) -> ModelCatalogEntry | None:
        """Entrada del catálogo, enriquecida con sus features la primera vez."""
        entrada = next((item for item in self.catalog() if item.key == reference.key), None)
        if entrada is not None and not entrada.features:
            entrada.features = self.features_for(reference)
        return entrada

    # ------------------------------------------------------------- ejecución

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
        if json_schema is not None:
            cuerpo["output"] = {"format": "json", "json_schema": json_schema}
        if timeout_seconds is not None:
            cuerpo["execution"]["timeout_seconds"] = timeout_seconds
        if self.config.supports_learning_exclusion:
            cuerpo["exclude_from_model_learning"] = True
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
        principal = telemetria[-1] if telemetria else None
        if principal is not None:
            resultado.seed_status = principal.seed_status
            if principal.generation:
                resultado.applied_params = dict(principal.generation)
            if principal.latency_ms is not None:
                resultado.latency_ms = principal.latency_ms
            resultado.fingerprint = _huella_publicada(principal.execution_fingerprint)
            if principal.seed_status is SeedStatus.unsupported:
                self._declare_missing(
                    f"determinismo en {target.key}: el endpoint no admite semilla "
                    "(seed_status=unsupported), así que pedirla no cambió nada."
                )
        if not resultado.fingerprint.components:
            resultado.fingerprint = self.fingerprint_for(target, resultado)

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

    # ------------------------------------------------------------- identidad

    def fingerprint_for(self, target: ModelReference, outcome: InvocationOutcome | None = None) -> ExecutionFingerprint:
        """Huella de ejecución: la que publica el Broker, si la publica.

        Desde el Cambio 2 el catálogo trae `execution_fingerprint` con las tres
        capas ya resueltas y el estado de cada componente. Se adopta tal cual:
        el Broker es quien sabe qué pudo verificar contra el runtime, y
        recalcularlo aquí sería adivinar peor.

        Si el Broker no la publica —versión anterior, o modelo ausente del
        catálogo—, se construye una huella de solo capa A y se **declara la
        garantía perdida**: identidad declarada no es identidad verificada.
        """
        builder = FingerprintBuilder()
        entrada = None
        try:
            entrada = next((item for item in self.catalog() if item.key == target.key), None)
        except BrokerError:
            entrada = None

        publicada = (entrada.execution_fingerprint if entrada else {}) or {}
        componentes = publicada.get("components") if isinstance(publicada, dict) else None

        if isinstance(componentes, dict) and componentes:
            for nombre, crudo in componentes.items():
                if not isinstance(crudo, dict):
                    continue
                valor = crudo.get("value")
                estado = str(crudo.get("status") or "")
                texto = None if valor is None else str(valor)
                if estado == ComponentState.verificado.value:
                    builder.verified(nombre, texto)
                elif estado == ComponentState.declarado.value:
                    builder.declared(nombre, texto)
                else:
                    builder.unavailable(nombre)
        else:
            builder.declared("provider", target.provider)
            builder.declared("deployment", target.deployment)
            builder.declared("model", target.model)
            for nombre in ("digest", "quantization", "parameter_size", "tokenizer", "chat_template", "runtime"):
                builder.unavailable(nombre, layer=IdentityLayer.observable)
            builder.unavailable("system_fingerprint", layer=IdentityLayer.remota)
            self._declare_missing(
                f"identidad técnica de {target.key}: el Broker no publica execution_fingerprint "
                "(Cambio 2 del contrato de evaluación). La identidad es solo declarada: no se puede "
                "afirmar que dos tiradas usaron el mismo binario ni la misma plantilla."
            )

        # Los parámetros pedidos son capa A y los aporta esta invocación, no el
        # catálogo: dos tiradas del mismo modelo con temperaturas distintas son
        # ejecuciones distintas.
        if outcome is not None and outcome.applied_params:
            builder.declared("requested_params", _canonical_params(outcome.applied_params))

        huella = builder.build()
        ausentes = sorted(huella.missing_names() & {"digest", "chat_template", "tokenizer", "runtime"})
        if ausentes:
            self._declare_missing(
                f"identidad técnica de {target.key}: el Broker no pudo verificar "
                + ", ".join(ausentes)
                + ". Para esos componentes la identidad no está comprobada."
            )
        return huella


def _huella_publicada(bloque: dict[str, Any]) -> ExecutionFingerprint:
    """Convierte la huella que publica el Broker a la del dominio.

    Se adopta tal cual, con el estado de cada componente: el Broker es quien
    sabe qué pudo verificar contra el runtime.
    """
    builder = FingerprintBuilder()
    componentes = bloque.get("components") if isinstance(bloque, dict) else None
    if not isinstance(componentes, dict):
        return builder.build()
    for nombre, crudo in componentes.items():
        if not isinstance(crudo, dict):
            continue
        valor = crudo.get("value")
        texto = None if valor is None else str(valor)
        estado = str(crudo.get("status") or "")
        if estado == ComponentState.verificado.value:
            builder.verified(nombre, texto)
        elif estado == ComponentState.declarado.value:
            builder.declared(nombre, texto)
        else:
            builder.unavailable(nombre)
    return builder.build()


def _canonical_params(params: dict[str, Any]) -> str:
    return ";".join(f"{clave}={params[clave]}" for clave in sorted(params))


def _first_str(features: dict[str, Any], name: str) -> str | None:
    for grupo in features.values():
        if isinstance(grupo, dict) and isinstance(grupo.get(name), str):
            return str(grupo[name])
    return None
