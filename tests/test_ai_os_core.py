"""AI OS Core dirige las tiradas: sin su ALLOW no sale nada hacia el Broker.

Core se simula con `httpx.MockTransport` (sus formas son las de Core 0.2.1); el
recorrido contra el Core real está en `scripts/verify_ai_os_core.py`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from model_drift.ai_os_core import CoreClient, CoreDeniedError, CoreUnavailableError, verdict_for
from model_drift.broker_contracts import ModelReference
from model_drift.cases import load_suite
from model_drift.db import Database
from model_drift.runner import RunConfig, Runner
from tests.broker_fake import FakeBroker

MODELO = ModelReference(provider="ollama", deployment="local", model="qwen3:6b")
SUITE = {
    "id": "demo-core",
    "version": "1",
    "cases": [
        {"id": f"json-0{i}", "prompt": f"Devuelve el total {i} en JSON.", "assertions": [{"type": "json_valid"}]}
        for i in range(1, 4)
    ],
}


class FakeCore:
    def __init__(self, authorization: str = "ALLOW", *, receipts_fail: bool = False,
                 hops_fail: bool = False) -> None:
        self.authorization = authorization
        self.receipts_fail = receipts_fail
        self.hops_fail = hops_fail
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        self.calls.append((request.url.path, body))
        assert request.headers["X-Principal-Id"] == "model_drift"
        if request.url.path == "/api/v1/policies/evaluate":
            return httpx.Response(200, json={"authorization": self.authorization,
                                             "executable": self.authorization == "ALLOW",
                                             "unknown_inputs": [], "trace_id": "trace_x"})
        if request.url.path == "/api/v1/operations/receipts":
            return httpx.Response(503 if self.receipts_fail else 201, json={})
        if request.url.path == "/api/v1/flows/hops" and self.hops_fail:
            raise httpx.ConnectError("monitor caído")
        return httpx.Response(201, json={})

    def client(self) -> CoreClient:
        http = httpx.Client(base_url="http://core", transport=httpx.MockTransport(self.handler),
                            headers={"X-Principal-Id": "model_drift"})
        return CoreClient("http://core", client=http)

    def paths(self) -> list[str]:
        return [path for path, _ in self.calls]


def _tirada(tmp_path: Path, core: CoreClient) -> tuple[Any, FakeBroker, Database]:
    destino = tmp_path / "suite.yaml"
    destino.write_text(yaml.safe_dump(SUITE, allow_unicode=True), encoding="utf-8")
    broker = FakeBroker()
    broker.add_model(MODELO.key)
    broker.respond_fixed(MODELO.key, '{"total": 100}')
    db = Database(tmp_path / "md.sqlite")
    with broker.client() as cliente:
        runner = Runner(cliente, load_suite(destino), db=db, config=RunConfig(repetitions=2), core=core)
        return runner.run([MODELO]), broker, db


@pytest.mark.parametrize("decision", ["DENY", "REQUIRE_APPROVAL", "INDETERMINATE"])
def test_sin_allow_la_tirada_no_toca_el_broker(tmp_path: Path, decision: str) -> None:
    core = FakeCore(decision)
    with pytest.raises(CoreDeniedError):
        _tirada(tmp_path, core.client())
    assert core.paths() == ["/api/v1/policies/evaluate"]
    contexto = core.calls[0][1]["context"]
    assert contexto["data_classification"] == "internal" and contexto["models"] == [MODELO.key]


def test_core_caido_no_arranca(tmp_path: Path) -> None:
    def caido(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("rechazada", request=request)

    core = CoreClient("http://core", client=httpx.Client(base_url="http://core", transport=httpx.MockTransport(caido)))
    with pytest.raises(CoreUnavailableError):
        _tirada(tmp_path, core)


def test_con_allow_deja_recibos_y_certifica(tmp_path: Path) -> None:
    core = FakeCore()
    salida, broker, db = _tirada(tmp_path, core.client())
    with db:
        assert salida.status == "completed" and salida.invocations == 6
        assert broker._tasks, "con ALLOW la tirada sí llega al Broker"
        recibos = [body for path, body in core.calls if path == "/api/v1/operations/receipts"]
        assert [r["status"] for r in recibos] == ["accepted", "succeeded"]
        assert {r["operation_id"] for r in recibos} == {salida.run_id}
        (cert,) = [body for path, body in core.calls if path == "/api/v1/certifications"]
        assert cert == {"model_id": MODELO.key, "capability_id": "suite:demo-core", "version": "1",
                        "source_system": "model_drift", "evaluation_verdict": "CERTIFIED",
                        "is_applicable": True, "revision": 1}
        assert salida.core_notices == []


def test_la_tirada_se_ve_en_el_monitor_de_core(tmp_path: Path) -> None:
    """Autorización, saltos y recibos comparten flow_id (= id de la tirada): Model_Drift 0.4.1."""
    core = FakeCore()
    salida, _, db = _tirada(tmp_path, core.client())
    db.close()
    assert core.calls[0][1]["context"]["flow_id"] == salida.run_id
    saltos = [body for path, body in core.calls if path == "/api/v1/flows/hops"]
    assert [s["kind"] for s in saltos] == ["request", "response"]
    assert {(s["flow_id"], s["source_system"], s["target_system"]) for s in saltos} == {
        (salida.run_id, "model_drift", "ai_broker")}
    assert saltos[1]["details"]["invocations"] == 6 and saltos[1]["details"]["latency_ms"] >= 0
    # Un salto por tirada, no por caso: el monitor no se inunda
    assert core.paths().count("/api/v1/flows/hops") == 2


def test_el_monitor_caido_no_toca_la_tirada(tmp_path: Path) -> None:
    salida, _, db = _tirada(tmp_path, FakeCore(hops_fail=True).client())
    db.close()
    assert salida.status == "completed" and salida.core_notices == []


def test_un_recibo_perdido_avisa_pero_no_tira_la_tirada(tmp_path: Path) -> None:
    salida, _, db = _tirada(tmp_path, FakeCore(receipts_fail=True).client())
    db.close()
    assert salida.status == "completed"
    assert len(salida.core_notices) == 2 and all("recibo" in aviso for aviso in salida.core_notices)


def test_regla_del_veredicto() -> None:
    assert verdict_for(3, 0.8) == "CERTIFIED"
    assert verdict_for(2, 1.0) == "INDETERMINATE"  # pocos casos para certificar
    assert verdict_for(1, 0.4) == "REJECTED"
    assert verdict_for(5, 0.6) == "INDETERMINATE"
    assert verdict_for(0, None) == "INDETERMINATE"


def test_sin_core_url_no_hay_director(monkeypatch: pytest.MonkeyPatch) -> None:
    from model_drift.config import AppConfig

    monkeypatch.delenv("MODEL_DRIFT_CORE_URL", raising=False)
    assert AppConfig().core_client() is None
    monkeypatch.setenv("MODEL_DRIFT_CORE_URL", "http://127.0.0.1:8000")
    assert AppConfig().core_client() is not None


def test_sin_pegar_tokens_core_da_su_conexion_y_el_token_del_broker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Core deja URL y token de la app; el del Broker se pide a Core y se renueva tras un 403."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from model_drift.broker_client import BrokerClient
    from model_drift.config import AppConfig

    broker_vigente = ["broker-1"]

    class FalsoCore(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            ok = self.headers.get("Authorization") == "Bearer tok-md"
            cuerpo = json.dumps({"token": broker_vigente[0]} if ok else {}).encode()
            self.send_response(200 if ok else 401)
            self.end_headers()
            self.wfile.write(cuerpo)

        def log_message(self, *_: object) -> None:
            pass

    servidor = HTTPServer(("127.0.0.1", 0), FalsoCore)
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{servidor.server_port}"
        monkeypatch.setenv("AI_OS_CONNECT_DIR", str(tmp_path))
        monkeypatch.delenv("MODEL_DRIFT_CORE_URL", raising=False)
        monkeypatch.delenv("MODEL_DRIFT_CORE_TOKEN", raising=False)
        monkeypatch.setenv("MODEL_DRIFT_BROKER_TOKEN", "caducado-pegado")
        (tmp_path / "model_drift.json").write_text(json.dumps({"core_url": url, "token": "tok-md"}))

        config = AppConfig()
        assert config.core_settings() == (url, "tok-md")

        vistos: list[str | None] = []

        def broker(request: httpx.Request) -> httpx.Response:
            vistos.append(request.headers.get("X-Admin-Token"))
            if request.headers.get("X-Admin-Token") != broker_vigente[0]:
                return httpx.Response(403, json={"detail": {"code": "ADMIN_AUTH_REQUIRED", "message": "no"}})
            return httpx.Response(200, json={})

        cliente = BrokerClient(config.broker_config(),
                               client=httpx.Client(base_url="http://broker", transport=httpx.MockTransport(broker)))
        cliente._request("GET", "/api/v1/capabilities")
        broker_vigente[0] = "broker-2"  # el Broker se reinicia y el runner publica el nuevo en Core
        cliente._request("GET", "/api/v1/capabilities")
        assert vistos == ["broker-1", "broker-1", "broker-2"]
    finally:
        servidor.shutdown()
