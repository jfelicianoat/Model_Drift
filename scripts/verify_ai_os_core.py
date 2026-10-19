# ruff: noqa: E501
"""Recorrido real: AI OS Core dirige una tirada de Model_Drift contra el Broker de verdad.

Uso:  python scripts/verify_ai_os_core.py <core_url> <token_model_drift> <token_director> <carpeta_trabajo> [modelo]
      con MODEL_DRIFT_BROKER_TOKEN en el entorno.
Core ha de tener las políticas de `sembrar_ecosistema.py` y el token por app de model_drift;
el token director (común) pone y quita la política DENY de la prueba. La base de datos de la
tirada vive en <carpeta_trabajo>: la de verdad (state/model_drift.sqlite) no se toca.
"""
from __future__ import annotations

import contextlib
import io
import os
import sqlite3
import sys
from pathlib import Path

import httpx
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_drift.cli import main  # noqa: E402

CORE, TOK_MD, TOK_DIR, WORK = sys.argv[1:5]
MODELO = sys.argv[5] if len(sys.argv) > 5 else "ollama/local/lfm2:24b"
work = Path(WORK)
work.mkdir(parents=True, exist_ok=True)
db = work / "md.sqlite"
director = httpx.Client(base_url=CORE, headers={"Authorization": f"Bearer {TOK_DIR}"}, timeout=10)
ok = 0


def step(nombre: str, cond: bool, detalle: str = "") -> None:
    global ok
    print(f"[{'OK ' if cond else 'FALLO'}] {nombre} {detalle}")
    if not cond:
        sys.exit(1)
    ok += 1


def tiradas() -> int:
    if not db.exists():
        return 0
    with sqlite3.connect(db) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM run").fetchone()[0])


def lanzar(core_url: str) -> tuple[int, str]:
    config = work / "model_drift.yaml"
    config.write_text(yaml.safe_dump({"database": str(db), "core_url": core_url, "repetitions": 1}), encoding="utf-8")
    os.environ["MODEL_DRIFT_CORE_TOKEN"] = TOK_MD
    salida = io.StringIO()
    with contextlib.redirect_stdout(salida), contextlib.redirect_stderr(salida):
        codigo = main(["--config", str(config), "ejecutar", str(suite), "--modelo", MODELO, "--etiqueta", "verify-core"])
    return codigo, salida.getvalue()


suite = work / "suite.yaml"
suite.write_text(yaml.safe_dump({"id": "verify-core", "version": "1", "cases": [
    {"id": f"eco-{i}", "prompt": f"Responde solo con la palabra VERDE{i}, sin nada más.",
     "assertions": [{"type": "contains", "values": [f"VERDE{i}"]}]} for i in range(1, 4)]}), encoding="utf-8")

step("core /health", httpx.get(f"{CORE}/health", timeout=5).json()["status"] == "healthy")
step("token de model_drift no puede firmar como otra app", httpx.post(
    f"{CORE}/api/v1/certifications", headers={"Authorization": f"Bearer {TOK_MD}"},
    json={"model_id": MODELO, "capability_id": "suite:x", "version": "1", "source_system": "athena",
          "evaluation_verdict": "CERTIFIED", "is_applicable": True}).status_code == 403)

# 1. DENY: la tirada no existe y el Broker no recibe nada
deny = {"policy_id": "verify_md_deny", "policy_key": "verify_md_deny", "scope": "model_drift:broker",
        "decision_if_matched": "DENY", "conditions": [{"field": "operation_type", "operator": "equals",
                                                        "value": "evaluation_run"}]}
step("director pone DENY", director.post("/api/v1/policies", json=deny).status_code in (200, 201))
antes = tiradas()
codigo, texto = lanzar(CORE)
step("con DENY la CLI falla por Core", codigo != 0 and "no autoriza la tirada (DENY)" in texto, texto.strip()[-160:])
step("con DENY no hay tirada", tiradas() == antes)
step("director quita DENY", director.post("/api/v1/policies", json={**deny, "enabled": False}).status_code in (200, 201))

# 2. Core caído: falla cerrado
codigo, texto = lanzar("http://127.0.0.1:9")
step("con Core caído la CLI falla cerrado", codigo != 0 and "No se pudo consultar a AI OS Core" in texto, texto.strip()[-160:])
step("con Core caído no hay tirada", tiradas() == antes)

# 3. ALLOW: tirada real, recibos y certificación
codigo, texto = lanzar(CORE)
print(texto)
step("con ALLOW la tirada termina", codigo == 0)
with sqlite3.connect(db) as conn:
    run_id, estado = conn.execute("SELECT run_id, status FROM run ORDER BY created_at DESC LIMIT 1").fetchone()
    tareas = conn.execute("SELECT COUNT(*) FROM invocation WHERE run_id = ? AND ok = 1 AND broker_task_id != ''", (run_id,)).fetchone()[0]
step("la tirada llegó al Broker real", estado == "completed" and tareas == 3, f"{run_id} {estado} tareas={tareas}")
recibo = director.post("/api/v1/operations/status", json={"operation_id": run_id, "target_service_id": "ai_broker"}).json()
step("Core tiene el recibo terminal de la tirada", recibo.get("status") == "succeeded", str(recibo)[:200])
cert = director.get(f"/api/v1/certifications/{MODELO}/suite:verify-core/1")
step("Core tiene la certificación del modelo", cert.status_code == 200 and cert.json()["source_system"] == "model_drift",
     cert.text[:200])
print(f"      veredicto {cert.json()['evaluation_verdict']} revisión {cert.json()['revision']}")
print(f"\n{ok} comprobaciones OK")
