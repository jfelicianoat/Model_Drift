"""Broker simulado sobre `httpx.MockTransport`.

Sigue el patrón que ya funcionó en ChatyGPT (`broker/simulated.rs`): un Broker
de verdad al otro lado del transporte, con respuestas programadas y registro de
peticiones. La diferencia con un doble de `BrokerClient` es importante: aquí se
ejercita **el cliente real** —construcción del cuerpo, sondeo, degradación ante
un 422, detección de fallback—, que es justo donde se esconden los fallos de
integración.

**Las formas de las respuestas se copian del Broker real, no se inventan.** Un
simulado que devuelve la lista de modelos bajo la clave `models` cuando el
Broker la devuelve bajo `items` hace pasar todas las pruebas y deja el catálogo
vacío en producción. Pasó exactamente eso el 16-ago-2026. Al tocar aquí, cotejar
antes contra un Broker en marcha.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from model_drift.broker_client import BrokerClient, BrokerConfig

Responder = Callable[[str, str, int], str]
"""(prompt, model_key, nº de veces que se ha visto este prompt) -> texto."""


@dataclass
class FakeBroker:
    """Broker de mentira con catálogo, cola instantánea y contabilidad."""

    #: model_key -> función que produce la respuesta.
    responders: dict[str, Responder] = field(default_factory=dict)
    #: Entradas de catálogo: model_key -> features y capacidades.
    catalog: list[dict[str, Any]] = field(default_factory=list)
    #: El Broker real admite `seed`/`top_p` desde el Cambio 1. Se puede poner
    #: a False para simular uno anterior y probar la degradación.
    supports_seed: bool = True
    #: Ídem con el marcador de exclusión del aprendizaje (Cambio 4).
    supports_learning_exclusion: bool = True
    #: Fuerza que el Broker sirva otro modelo distinto del pedido.
    serve_instead: dict[str, str] = field(default_factory=dict)
    #: Declara `fallback_used` en el resultado.
    declare_fallback: bool = False
    #: Errores a devolver por prompt (prompt -> código de error).
    fail_prompts: dict[str, str] = field(default_factory=dict)
    #: Contrato 2.9: telemetría por invocación. A False simula un Broker previo.
    supports_invocation_telemetry: bool = True
    #: Qué dice el Broker que hizo con la semilla: sent | unsupported.
    seed_status: str = "sent"
    #: Contrato 2.10: `contractual` por invocación y `role`/`status` enumerados
    #: (§8.1). A False simula un Broker 2.9, que no se pronuncia.
    supports_invocation_contract: bool = True
    #: Contrato 2.10: eco `{requested, effective}` por invocación (§8.5).
    supports_prompt_compression_echo: bool = True
    #: Contrato 2.10: /artifacts y el entregable marcado con `final: true`.
    supports_task_artifacts: bool = True
    supports_canonical_artifacts: bool = True
    #: Contrato 2.10: si este Broker sondea otros modelos bajo el mismo
    #: `task_id` (§8.4) y si acepta que se apague desde la petición.
    performs_auxiliary_invocations: bool = True
    supports_auxiliary_optout: bool = True
    #: Poda que el Broker aplica DE VERDAD, pida lo que pida el cliente. Sirve
    #: para simular el caso que el eco existe para destapar: se pidió `off` y se
    #: comprimió igual.
    effective_compression: str | None = None
    #: Añade un `shadow_probe` (no contractual) DESPUÉS de la invocación real,
    #: con otro modelo y otra huella. Es el caso que rompía la lectura por
    #: «última invocación» antes del 2.10.
    shadow_probe_model: str | None = None
    #: El token admin ya no vale (el Broker rota el suyo en cada arranque).
    #: Forma cotejada contra el Broker 2.11 vivo el 18-oct-2026:
    #: `403 {"detail": "ADMIN_AUTH_REQUIRED"}`. El catálogo sigue sin pedirlo.
    reject_admin_token: bool = False

    requests: list[dict[str, Any]] = field(default_factory=list)
    _tasks: dict[str, dict[str, Any]] = field(default_factory=dict)
    _invocation_items: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    _artifact_items: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    _seen: dict[str, int] = field(default_factory=dict)
    _counter: int = 0

    # ------------------------------------------------------------- catálogo

    def add_model(
        self,
        key: str,
        *,
        capabilities: tuple[str, ...] = ("completion",),
        features: dict[str, Any] | None = None,
        context_window: int = 8192,
        fingerprint: dict[str, Any] | None = None,
    ) -> None:
        provider, deployment, model = key.split("/", 2)
        self.catalog.append(
            {
                "provider": provider,
                "deployment": deployment,
                "model": model,
                "availability": "online",
                "dispatchable": True,
                "reason": "modelo simulado",
                "provider_status": "healthy",
                "model_status": "available",
                "compatibility": "compatible",
                "capabilities": list(capabilities),
                "context_window": context_window,
                "compatibility_error": None,
                # `features` NO viaja en availability en el Broker real: se pide
                # aparte, por modelo. Se guarda aquí solo para que el manejador
                # de /models/context sepa qué responder.
                "features": features or {},
                "execution_fingerprint": _fingerprint(model) if fingerprint is None else fingerprint,
            }
        )

    @property
    def contract_version(self) -> str:
        """Versión que anuncia este simulado, coherente con sus interruptores."""
        if self.supports_invocation_contract and self.supports_prompt_compression_echo:
            return "2.10"
        return "2.9" if self.supports_seed else "2.8"

    def respond(self, key: str, responder: Responder) -> None:
        self.responders[key] = responder

    def respond_fixed(self, key: str, text: str) -> None:
        self.responders[key] = lambda prompt, model, seen: text

    # ------------------------------------------------------------ transporte

    def handler(self, request: httpx.Request) -> httpx.Response:
        ruta = request.url.path
        if request.method == "GET" and ruta == "/api/v1/models/availability":
            # Forma real: {checked_at, items, counts}. La lista va en `items`.
            items = [{k: v for k, v in item.items() if k != "features"} for item in self.catalog]
            return httpx.Response(
                200,
                json={
                    "checked_at": "2026-08-16T00:00:00Z",
                    "items": items,
                    "counts": {"online": len(items), "dispatchable": len(items)},
                },
            )
        if request.method == "GET" and ruta == "/api/v1/models/context":
            return self._model_context(request)
        if request.method == "GET" and ruta == "/openapi.json":
            generacion = {"temperature": {}, "max_output_tokens": {}}
            tarea = {"content": {}, "generation": {}, "model_requirements": {}}
            if self.supports_seed:
                generacion.update({"seed": {}, "top_p": {}})
            if self.supports_learning_exclusion:
                tarea["exclude_from_model_learning"] = {}
            return httpx.Response(
                200,
                json={
                    "components": {
                        "schemas": {
                            "GenerationConfig": {"properties": generacion},
                            "TaskCreateRequest": {"properties": tarea},
                        }
                    }
                },
            )
        if request.method == "GET" and ruta == "/api/v1/capabilities":
            return httpx.Response(
                200,
                json={
                    "contract_version": self.contract_version,
                    "strategies": ["single"],
                    "presets": {"single": ["fast"]},
                    "exclude_from_model_learning": self.supports_learning_exclusion,
                    "invocation_telemetry": self.supports_invocation_telemetry,
                    "execution_fingerprint": self.supports_seed,
                    # Contrato 2.10.
                    "invocation_contract": self.supports_invocation_contract,
                    "prompt_compression_echo": self.supports_prompt_compression_echo,
                    "task_artifacts": self.supports_task_artifacts,
                    "canonical_artifacts": self.supports_canonical_artifacts,
                    "auxiliary_invocations": self.performs_auxiliary_invocations,
                    "auxiliary_invocations_optout": self.supports_auxiliary_optout,
                },
            )
        if request.method == "POST" and ruta == "/api/v1/tasks":
            return self._create_task(request)
        match_inv = re.fullmatch(r"/api/v1/tasks/([^/]+)/invocations", ruta)
        if request.method == "GET" and match_inv:
            if not self.supports_invocation_telemetry:
                return httpx.Response(404, json={"detail": {"code": "NOT_FOUND", "message": ruta}})
            return self._invocations(match_inv.group(1))
        match_art = re.fullmatch(r"/api/v1/tasks/([^/]+)/artifacts", ruta)
        if request.method == "GET" and match_art:
            if not self.supports_task_artifacts:
                return httpx.Response(404, json={"detail": {"code": "NOT_FOUND", "message": ruta}})
            return self._artifacts(match_art.group(1))
        match = re.fullmatch(r"/api/v1/tasks/([^/]+)", ruta)
        if request.method == "GET" and match:
            task = self._tasks.get(match.group(1))
            if task is None:
                return httpx.Response(404, json={"detail": {"code": "NOT_FOUND", "message": "no existe"}})
            return httpx.Response(200, json=task)
        return httpx.Response(404, json={"detail": {"code": "NOT_FOUND", "message": ruta}})

    def _invocations(self, task_id: str) -> httpx.Response:
        """`GET /api/v1/tasks/{id}/invocations` -> {task_id, items[]}."""
        tarea = self._tasks.get(task_id)
        if tarea is None:
            return httpx.Response(404, json={"detail": {"code": "TASK_NOT_FOUND", "message": task_id}})
        return httpx.Response(200, json={"task_id": task_id, "items": self._invocation_items.get(task_id, [])})

    def _artifacts(self, task_id: str) -> httpx.Response:
        """`GET /api/v1/tasks/{id}/artifacts` -> {task_id, items[]} (§8.3)."""
        tarea = self._tasks.get(task_id)
        if tarea is None:
            return httpx.Response(404, json={"detail": {"code": "TASK_NOT_FOUND", "message": task_id}})
        return httpx.Response(
            200, json={"task_id": task_id, "items": self._artifact_items.get(task_id, [])}
        )

    def _model_context(self, request: httpx.Request) -> httpx.Response:
        """`/models/context` es POR MODELO y exige los tres parámetros."""
        parametros = request.url.params
        faltan = [nombre for nombre in ("provider", "deployment", "model") if not parametros.get(nombre)]
        if faltan:
            return httpx.Response(
                422,
                json={
                    "code": "CONTRACT_VALIDATION_FAILED",
                    "message": "Request does not satisfy Broker contract v1",
                    "fields": [{"type": "missing", "loc": ["query", nombre]} for nombre in faltan],
                },
            )
        clave = f"{parametros['provider']}/{parametros['deployment']}/{parametros['model']}"
        entrada = next(
            (item for item in self.catalog if f"{item['provider']}/{item['deployment']}/{item['model']}" == clave),
            None,
        )
        if entrada is None:
            return httpx.Response(404, json={"detail": {"code": "MODEL_UNAVAILABLE", "message": clave}})
        return httpx.Response(
            200,
            json={
                "provider": entrada["provider"],
                "deployment": entrada["deployment"],
                "model": entrada["model"],
                "context_window": entrada["context_window"],
                "context_window_known": True,
                "capabilities": entrada["capabilities"],
                "features": entrada["features"],
                "feature_notes": [],
                "compatibility": entrada["compatibility"],
                "compatibility_checked_at": None,
                "compatibility_error": None,
                "execution_fingerprint": entrada["execution_fingerprint"],
            },
        )

    def _create_task(self, request: httpx.Request) -> httpx.Response:
        if self.reject_admin_token:
            return httpx.Response(403, json={"detail": "ADMIN_AUTH_REQUIRED"})
        cuerpo = json.loads(request.content.decode("utf-8"))
        self.requests.append(cuerpo)

        generacion = cuerpo.get("generation") or {}
        if not self.supports_seed and ("seed" in generacion or "top_p" in generacion):
            # El contrato real es estricto: `extra="forbid"` en GenerationConfig.
            return httpx.Response(
                422,
                json={
                    "detail": {
                        "code": "CONTRACT_VALIDATION_FAILED",
                        "message": "generation: extra fields not permitted (seed)",
                    }
                },
            )
        if not self.supports_learning_exclusion and "exclude_from_model_learning" in cuerpo:
            return httpx.Response(
                422,
                json={
                    "detail": {
                        "code": "CONTRACT_VALIDATION_FAILED",
                        "message": "extra fields not permitted (exclude_from_model_learning)",
                    }
                },
            )

        if not self.supports_auxiliary_optout and "auxiliary_invocations" in cuerpo:
            # El contrato real valida con `extra="forbid"`: un campo que no
            # existe hace fallar la petición entera, no se ignora.
            return httpx.Response(
                422,
                json={
                    "detail": {
                        "code": "CONTRACT_VALIDATION_FAILED",
                        "message": "extra fields not permitted (auxiliary_invocations)",
                    }
                },
            )

        objetivo = cuerpo["model_requirements"]["target_model"]
        pedido = f"{objetivo['provider']}/{objetivo['deployment']}/{objetivo['model']}"
        servido = self.serve_instead.get(pedido, pedido)
        prompt = cuerpo["content"]["prompt"]

        self._counter += 1
        task_id = f"task_{self._counter:04d}"

        if prompt in self.fail_prompts:
            self._tasks[task_id] = {
                "task_id": task_id,
                "kind": "inference",
                "status": "failed",
                "progress": {"phase": "failed"},
                "result": None,
                "error": {"code": self.fail_prompts[prompt], "message": "fallo programado", "retryable": False},
            }
            return httpx.Response(202, json={"task_id": task_id, "status": "queued"})

        vistas = self._seen.get(prompt, 0)
        self._seen[prompt] = vistas + 1
        responder = self.responders.get(servido)
        texto = responder(prompt, servido, vistas) if responder else f"[sin responder para {servido}]"

        if cuerpo.get("inference_kind") == "embedding":
            resultado: dict[str, Any] = {
                "inference_kind": "embedding",
                "embedding": _fake_embedding(prompt),
                "usage": {"tokens_input": len(prompt) // 4, "tokens_output": 0, "cost_usd": 0.0},
                "model_used": {
                    "provider": objetivo["provider"],
                    "deployment": objetivo["deployment"],
                    "model": objetivo["model"],
                },
                "fallback_used": False,
            }
        else:
            proveedor, despliegue, modelo = servido.split("/", 2)
            resultado = {
                "inference_kind": "chat",
                "output_format": "markdown",
                "assistant_content": texto,
                "result_markdown": texto,
                "usage": {
                    "tokens_input": max(1, len(prompt) // 4),
                    "tokens_output": max(1, len(texto) // 4),
                    "cost_usd": round(len(texto) * 1e-6, 8),
                },
                "model_used": {"provider": proveedor, "deployment": despliegue, "model": modelo},
                "models_used": [{"provider": proveedor, "deployment": despliegue, "model": modelo}],
                "fallback_used": self.declare_fallback,
            }

        proveedor, despliegue, modelo = servido.split("/", 2)
        referencia = {"provider": proveedor, "deployment": despliegue, "model": modelo}
        pedido_ref = {
            "provider": objetivo["provider"],
            "deployment": objetivo["deployment"],
            "model": objetivo["model"],
        }
        efectiva = dict(generacion)
        efectiva["seed_status"] = self.seed_status if "seed" in generacion else "not_requested"
        pedida = cuerpo.get("prompt_compression") or "broker_default"
        aplicada = self.effective_compression or ("off" if pedida == "broker_default" else pedida)
        principal: dict[str, Any] = {
            "invocation_id": f"inv_{self._counter:04d}",
            "role": "single",
            "model": referencia,
            "status": "completed",
            "error_code": None,
            "tokens_input": resultado["usage"]["tokens_input"],
            "tokens_output": resultado["usage"]["tokens_output"],
            "cost_usd": resultado["usage"]["cost_usd"],
            "latency_ms": 42.0,
            "generation": efectiva,
            "execution_fingerprint": _fingerprint(modelo),
            "excluded_from_model_learning": bool(cuerpo.get("exclude_from_model_learning")),
        }
        if self.supports_invocation_contract:
            principal["contractual"] = True
            principal["content_source"] = "content"
        if self.supports_prompt_compression_echo:
            principal["prompt_compression"] = {"requested": pedida, "effective": aplicada}
        items = [principal]

        # Sondeo en sombra: va DESPUÉS de la real y con otro modelo, que es lo
        # que hacía que leer «la última invocación» midiera al aspirante.
        exclusivo = cuerpo.get("auxiliary_invocations") is False
        if self.shadow_probe_model and self.performs_auxiliary_invocations and not exclusivo:
            sonda_p, sonda_d, sonda_m = self.shadow_probe_model.split("/", 2)
            sonda: dict[str, Any] = {
                "invocation_id": f"inv_{self._counter:04d}_shadow",
                "role": "shadow_probe",
                "model": {"provider": sonda_p, "deployment": sonda_d, "model": sonda_m},
                "status": "completed",
                "error_code": None,
                "tokens_input": 1,
                "tokens_output": 1,
                "cost_usd": 0.0,
                "latency_ms": 999.0,
                "generation": {"seed_status": "not_requested"},
                "execution_fingerprint": _fingerprint(sonda_m),
                "excluded_from_model_learning": True,
            }
            if self.supports_invocation_contract:
                sonda["contractual"] = False
            if self.supports_prompt_compression_echo:
                sonda["prompt_compression"] = {"requested": pedida, "effective": aplicada}
            items.append(sonda)
        self._invocation_items[task_id] = items

        # Artefactos (§8.3): el entregable va marcado con `final: true` y su
        # sha256 cierra sobre los bytes exactos.
        if self.supports_task_artifacts:
            entregable = (resultado.get("assistant_content") or "").encode("utf-8")
            artefacto: dict[str, Any] = {
                "artifact_id": f"art_{self._counter:04d}",
                "artifact_type": "single_output",
                "filename": "respuesta.md",
                "media_type": "text/markdown",
                "size_bytes": len(entregable),
                "sha256": hashlib.sha256(entregable).hexdigest(),
                "created_at": "2026-09-05T00:00:00Z",
                "download_url": f"/api/v1/tasks/{task_id}/artifacts/art_{self._counter:04d}",
                "available": True,
            }
            if self.supports_canonical_artifacts:
                artefacto["final"] = True
            self._artifact_items[task_id] = [artefacto]
        self._tasks[task_id] = {
            "task_id": task_id,
            "kind": "inference",
            "status": "completed",
            "request_id": cuerpo.get("request_id"),
            "progress": {"phase": "completed", "invocations_completed": 1, "invocations_total": 1},
            # Contrato 2.9: la vía tipada para saber quién respondió.
            "execution_summary": {
                "requested_model": pedido_ref,
                "served_by": referencia,
                "models_used": [referencia],
                "fallback_used": self.declare_fallback,
            },
            "result": resultado,
            "error": None,
        }
        return httpx.Response(202, json={"task_id": task_id, "status": "queued"})

    # ---------------------------------------------------------------- cliente

    def client(self, **config: Any) -> BrokerClient:
        transporte = httpx.MockTransport(self.handler)
        http = httpx.Client(transport=transporte, base_url="http://broker.test")
        return BrokerClient(
            BrokerConfig(base_url="http://broker.test", poll_interval_seconds=0.0, **config),
            client=http,
        )


def _fingerprint(model: str) -> dict[str, Any]:
    """Huella con la misma forma y vocabulario que la del Broker real."""
    return {
        "components": {
            "provider": {"value": "ollama", "status": "declarado"},
            "deployment": {"value": "local", "status": "declarado"},
            "model": {"value": model, "status": "declarado"},
            "digest": {"value": f"sha256:{abs(hash(model)) % (10**16):016x}", "status": "verificado"},
            "quantization": {"value": "Q4_K_M", "status": "verificado"},
            "chat_template": {"value": "plantilla-simulada", "status": "verificado"},
            "tokenizer": {"value": "tokenizador-simulado", "status": "verificado"},
            "runtime": {"value": "ollama", "status": "declarado"},
            "runtime_version": {"value": "0.32.13", "status": "verificado"},
            "system_fingerprint": {"value": None, "status": "no_disponible"},
        },
        "hash": f"{abs(hash(model)) % (10**12):012x}",
    }


def _fake_embedding(text: str) -> list[float]:
    """Vector reproducible que depende del texto. No pretende ser semántico."""
    base = [0.0] * 8
    for indice, caracter in enumerate(text):
        base[indice % 8] += (ord(caracter) % 17) / 100.0
    norma = sum(valor * valor for valor in base) ** 0.5 or 1.0
    return [valor / norma for valor in base]
