"""AI OS Core como director: autoriza cada tirada y recibe las certificaciones.

Core es el plano de control del ecosistema del TFM. Model_Drift no lo necesita para
funcionar: sin `core_url` todo sigue igual. Con `core_url`:

- antes de mandar nada al Broker la tirada evalúa la política `model_drift:broker` y
  solo arranca con un ALLOW ejecutable. Si Core no contesta, **no arranca**: quien
  configuró un director no quiere que se le salte cuando se cae;
- la tirada deja recibos en Core (aceptada → terminada/fallida/cancelada);
- al terminar, publica en Core el veredicto de cada modelo sobre la suite: son las
  certificaciones con las que Core decide qué modelos pueden usar las demás apps.
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any

import httpx

from model_drift.db import Database

POLICY_SCOPE = "model_drift:broker"
#: Cómo se presenta esta app ante Core (debe coincidir con su token por app).
PRINCIPAL = "model_drift"
ENV_URL = "MODEL_DRIFT_CORE_URL"
ENV_TOKEN = "MODEL_DRIFT_CORE_TOKEN"

#: Regla del veredicto (la misma del E2E de Core): media de pass_rate sobre casos
#: aplicables, acumulada en todas las tiradas de ese modelo con esa suite.
MIN_CASES_CERTIFIED = 3
CERTIFIED_FROM = 0.8
REJECTED_BELOW = 0.5
#: Un salto para el monitor de Core nunca debe hacer esperar a la tirada.
HOP_TIMEOUT_SECONDS = 3.0


def core_connection(principal: str = PRINCIPAL) -> tuple[str, str] | None:
    """URL y token de esta app en Core, tal como Core los deja en su carpeta de conexión.

    Core genera un token por app y lo escribe en %LOCALAPPDATA%/AIOSCore/conexion: así
    no hay que pegar ninguno. Sin Core instalado (o sin haber arrancado nunca), None.
    """
    local = os.environ.get("LOCALAPPDATA")
    folder = os.environ.get("AI_OS_CONNECT_DIR") or (str(Path(local, "AIOSCore", "conexion")) if local else "")
    if not folder:
        return None
    try:
        data = json.loads(Path(folder, f"{principal}.json").read_text(encoding="utf-8"))
        url, token = str(data["core_url"]), str(data["token"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return (url, token) if url and token else None


class CoreBrokerToken:
    """El token del AI Broker, pedido a Core: el Broker lo rota en cada arranque.

    Se llama en cada petición; `refresh=True` (tras un 401/403 del Broker) lo vuelve a
    pedir. Si Core no lo tiene, se usa `fallback` (el token del entorno, si lo hay).
    """

    def __init__(self, core_url: str, core_token: str, fallback: str | None = None, timeout: float = 5.0) -> None:
        self._url = core_url.rstrip("/") + "/api/v1/credentials/broker"
        self._core_token = core_token
        self._fallback = fallback
        self._timeout = timeout
        self._token: str | None = None

    def __call__(self, refresh: bool = False) -> str | None:
        if refresh or self._token is None:
            self._token = self._fetch() or self._token
        return self._token or self._fallback

    def _fetch(self) -> str | None:
        request = urllib.request.Request(self._url, headers={
            "Authorization": f"Bearer {self._core_token}", "Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                return json.load(response).get("token") or None
        except (OSError, ValueError, AttributeError):
            return None


class CoreUnavailableError(RuntimeError):
    """Core no contestó: por seguridad la tirada no arranca."""


class CoreDeniedError(RuntimeError):
    """Core contestó, pero no autoriza la tirada."""


class CoreClient:
    def __init__(self, base_url: str, token: str | None = None, *, client: httpx.Client | None = None) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("La URL de AI OS Core debe usar http o https.")
        headers = {"Accept": "application/json", "X-Principal-Id": PRINCIPAL}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._http = client or httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=10.0)

    def close(self) -> None:
        self._http.close()

    def authorize_run(self, run_id: str, model_keys: list[str]) -> None:
        """Pide permiso para la tirada; si no hay ALLOW ejecutable, lanza y no se envía nada."""
        decision = self._post("/api/v1/policies/evaluate", {
            "scope": POLICY_SCOPE, "principal_id": PRINCIPAL, "resource_id": run_id,
            # Las suites son prompts de prueba del propio operador
            "context": {"operation_type": "evaluation_run", "data_classification": "internal",
                        "models": model_keys, "flow_id": run_id},
        })
        if decision.get("authorization") != "ALLOW" or decision.get("executable") is not True:
            raise CoreDeniedError(_denied_text(decision))

    def record_receipt(self, run_id: str, status: str, evidence: dict[str, Any]) -> str | None:
        """Devuelve un aviso si Core no lo apuntó; la tirada no se cae por un recibo."""
        try:
            self._post("/api/v1/operations/receipts", {
                "operation_id": run_id, "receiver_service": "ai_broker", "status": status, "evidence": evidence,
            })
        except (CoreUnavailableError, CoreDeniedError) as error:
            return f"AI OS Core no apuntó el recibo «{status}»: {error}"
        return None

    def hop(self, flow_id: str, kind: str, summary: str = "", details: dict[str, Any] | None = None) -> None:
        """Cuenta al monitor de Core lo que la tirada pide al Broker. Enviar y olvidar."""
        try:
            self._http.post("/api/v1/flows/hops", timeout=HOP_TIMEOUT_SECONDS, json={
                "flow_id": flow_id, "source_system": PRINCIPAL, "target_system": "ai_broker",
                "kind": kind, "summary": summary[:200], "details": details or {},
            })
        except Exception:  # noqa: BLE001 — el monitor es secundario: jamás para una tirada
            pass

    def publish_certifications(self, db: Database, suite_id: str, model_keys: list[str]) -> list[str]:
        """Publica el veredicto de cada modelo sobre la suite. Devuelve los avisos."""
        avisos = []
        for model_key in model_keys:
            fila = db.query_one(
                """SELECT COUNT(DISTINCT cr.run_id) AS runs,
                          SUM(CASE WHEN cr.is_not_applicable = 0 AND cr.case_score IS NOT NULL
                                   THEN 1 ELSE 0 END) AS applicable,
                          AVG(CASE WHEN cr.is_not_applicable = 0 THEN cr.case_score END) AS mean_pass
                   FROM case_result cr JOIN run r USING (run_id)
                   WHERE cr.metric = 'pass_rate' AND r.status != 'failed' AND cr.model_key = ? AND r.suite_id = ?""",
                (model_key, suite_id),
            )
            if fila is None or not fila["runs"]:
                continue
            verdict = verdict_for(int(fila["applicable"] or 0), fila["mean_pass"])
            try:
                self._post("/api/v1/certifications", {
                    "model_id": model_key, "capability_id": f"suite:{suite_id}", "version": "1",
                    "source_system": PRINCIPAL, "evaluation_verdict": verdict,
                    "is_applicable": verdict == "CERTIFIED", "revision": int(fila["runs"]),
                })
            except (CoreUnavailableError, CoreDeniedError) as error:
                avisos.append(f"AI OS Core no recibió la certificación de {model_key}: {error}")
        return avisos

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            respuesta = self._http.post(path, json=body)
        except httpx.HTTPError as error:
            raise CoreUnavailableError(
                f"No se pudo consultar a AI OS Core ({error}). Por seguridad no se ha enviado nada al Broker."
            ) from error
        if respuesta.status_code >= 400:
            raise CoreDeniedError(f"AI OS Core respondió {respuesta.status_code}: {respuesta.text[:300]}")
        resultado: dict[str, Any] = respuesta.json()
        return resultado


def verdict_for(applicable: int, mean_pass: float | None) -> str:
    if applicable >= MIN_CASES_CERTIFIED and mean_pass is not None and mean_pass >= CERTIFIED_FROM:
        return "CERTIFIED"
    if applicable >= 1 and mean_pass is not None and mean_pass < REJECTED_BELOW:
        return "REJECTED"
    return "INDETERMINATE"


def _denied_text(decision: dict[str, Any]) -> str:
    autorizacion = decision.get("authorization")
    if autorizacion == "REQUIRE_APPROVAL":
        return f"AI OS Core pide una aprobación humana antes de lanzar la tirada (políticas de «{POLICY_SCOPE}»)."
    if autorizacion == "INDETERMINATE":
        faltan = ", ".join(decision.get("unknown_inputs") or []) or "algún dato"
        return f"AI OS Core no puede decidir (le falta: {faltan}); por seguridad la tirada no arranca."
    return f"AI OS Core no autoriza la tirada ({autorizacion}). Revisa las políticas de «{POLICY_SCOPE}» en Core."
