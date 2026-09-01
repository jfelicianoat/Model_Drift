# Resumen ejecutivo de actualización (conservador)
- Mantener `requires-python >=3.11` este ciclo: 3.11 sigue en soporte de seguridad hasta 2027-10.
- Añadir Python 3.14 a CI sin elevar todavía el mínimo.
- Mantener versiones runtime del lock cuando ya son actuales.
- Hacer que el lock gobierne CI.
- Corregir comparabilidad de fixtures y transporte inseguro del token antes que cualquier refactor.
- No modificar semántica estadística ni contrato público salvo necesidad.

# Fuentes de versiones (FUENTES_DE_VERSION)
| Componente | Versión detectada | Fichero fuente | Confianza | Comentario |
|---|---|---|---|---|
| Python mínimo | >=3.11 | `pyproject.toml` | Alta | 3.11 security-only |
| Python CI | 3.11, 3.13 | `ci.yml` | Alta | falta 3.14 |
| httpx | 0.28.1 lock / >=0.27 | lock/pyproject | Alta | upstream visible estable |
| Jinja2 | 3.1.6 / >=3.1.4 | lock/pyproject | Alta | 3.1.6 parchea GHSA conocida |
| jsonschema | 4.26.0 / >=4.22 | lock/pyproject | Alta | latest visible |
| Pydantic | 2.13.4 / >=2.7 | lock/pyproject | Alta | latest visible consultado |
| PyYAML | 6.0.3 / >=6.0.1 | lock/pyproject | Alta | soporte 3.14 |
| structlog | 26.1.0 / >=24.1 | lock/pyproject | Alta | sin necesidad de major |
| pytest | 9.1.1 | lock | Alta | dev |
| mypy | 2.3.1 | lock | Alta | dev |
| ruff | 0.16.3 | lock | Alta | dev |
| checkout | v4 | CI | Alta | upstream v7 |
| setup-python | v5 | CI | Alta | upstream v7 |

# Referencias (EOL/soporte/CVEs/breaking changes) con fecha consultada
Fecha consultada: **2026-09-01**.
1. Python active releases: https://www.python.org/getit/ — 3.14 bugfix hasta 2030-10; 3.13 hasta 2029-10; 3.11 security hasta 2027-10.
2. Python 3.14.7: https://www.python.org/downloads/release/python-3147/ — publicada 2026-08-05.
3. setup-python: https://github.com/actions/setup-python — v7 actual; v6 pasó a Node 24 y requiere runner >=2.327.1.
4. Jinja2 GHSA: https://github.com/advisories/GHSA-cpwx-vrp4-4pq7 — afectadas <=3.1.5; parche 3.1.6.
5. HTTPX changelog: https://github.com/encode/httpx/blob/master/CHANGELOG.md — 0.28.1.
6. jsonschema PyPI: https://pypi.org/project/jsonschema/ — 4.26.0 latest visible.
7. Pydantic releases: https://github.com/pydantic/pydantic/releases — 2.13.4 latest visible consultado.
8. PyYAML releases: https://github.com/yaml/pyyaml/releases — 6.0.3 soporta Python 3.14.

# Matriz de obsolescencia (MATRIZ_DE_OBSOLESCENCIA)
| Área | Estado | Evidencia | Consecuencia | Acción |
|---|---|---|---|---|
| Python 3.11 | Riesgo | security-only hasta 2027-10 | ventana decreciente | mantener temporalmente |
| Python 3.13 | OK | bugfix support | target válido | mantener |
| Python 3.14 | Riesgo por ausencia de test | estable actual | compatibilidad no validada | añadir CI |
| Runtime deps lock | OK/Riesgo | versiones actuales | lock no efectivo | usar lock |
| GitHub Actions | Obsoleto relativo | v4/v5 vs v7 | toolchain atrasada | subir controladamente |
| Broker HTTP | Riesgo | token sobre HTTP | exposición de credencial | TLS/opt-in |
| SQLite schema | Riesgo | sin migraciones | upgrades frágiles | guard + migraciones |

# Targets recomendados (mínimo viable) y justificación
- Python: mantener `>=3.11`; CI 3.11 + 3.13 + 3.14.
- Jinja2: mínimo efectivo 3.1.6.
- Runtime deps: conservar lock salvo seguridad/bugfix necesario.
- Actions: v6 como salto conservador; v7 después de validar runner.
- Build: lock/constraints efectivo + `[build-system]`.
- Broker: HTTPS para no-loopback o excepción explícita.

# Plan de cambios por área (runtime/deps/toolchain/config/CI/infra)
## Runtime
Añadir 3.14 a CI; no elevar mínimo aún.

## Dependencias
Mantener versions; auditoría automática y Jinja2 >=3.1.6.

## Toolchain/build
Declarar backend y lock reproducible.

## Config
Rechazar token sobre HTTP no-loopback sin `allow_insecure_broker`.

## CI
Usar lock, añadir 3.14, subir Actions.

## Infra
No hay contenedores/k8s/terraform.

# Touchpoints de cambio (TOUCHPOINTS_DE_CAMBIO)
| Ruta | Símbolo | Cambio esperado | Riesgo | Test |
|---|---|---|---|---|
| `model_drift/manifest.py` | `compare_manifests` | diferencia simétrica fixtures | High/15 | alta/baja fixture |
| `tests/test_embeddings_manifest.py` | tests comparabilidad | casos negativos | Low | pytest |
| `pyproject.toml` | packaging | build-system / floor Jinja | Medium | build/install |
| `requirements.lock` | resolución | regenerar/hacer efectivo | High/16 | lock check |
| `.github/workflows/ci.yml` | job | lock + 3.14 + actions | Medium | CI |
| `model_drift/config.py` | `AppConfig`, `broker_config` | política URL insegura | High/15 | unit |
| `model_drift/broker_client.py` | `BrokerConfig`, init/headers | enforcement transporte | High/15 | HTTP/HTTPS |
| `model_drift/oracles.py` | `SqlResultAssertion._evaluate` | límites SQL | Medium/12 | abort |
| `model_drift/db.py` | `Database._migrate` | guard versión | Medium/12 | DB futura |

## PHASE_0
`manifest.py`, lock/CI y transporte Broker.

## PLAN_POR_FASES
1. Correctness de manifiesto.
2. Build reproducible.
3. Transporte seguro.
4. Hardening SQL/DB.

## NEXT_PHASE_ASK
Siguiente lote recomendado: **`model_drift/manifest.py` + `tests/test_embeddings_manifest.py`**.

# Roadmap (Quick wins / Medio / Largo)
- Quick wins: fixtures, lock, build-system, Jinja floor.
- Medio: TLS policy, 3.14 CI, Actions.
- Largo: migraciones y SQL sandbox.

# TAREAS_UPGRADE_CONSERVADOR
## UGC-001 — Corregir comparabilidad de fixtures
- Archivos: `manifest.py`, tests.
- Pasos: diferencia simétrica + issues + 2 tests.
- Aceptación: añadir/quitar fixture => no comparable.
- Dependencias: ninguna.
- Prioridad: P0.
- Severidad/riesgo: High/15.
- Esfuerzo: S.

## UGC-002 — Hacer efectivo el lock
- Archivos: pyproject/lock/CI/README.
- Pasos: constraints/lock; regenerar; instalar CI desde lock.
- Aceptación: CI no resuelve fuera del lock.
- Prioridad: P0.
- Severidad/riesgo: High/16.
- Esfuerzo: M.

## UGC-003 — Proteger token Broker
- Archivos: config/client/tests/docs.
- Pasos: detectar HTTP no-loopback; opt-in explícito.
- Aceptación: token + HTTP remoto falla por defecto.
- Prioridad: P0.
- Severidad/riesgo: High/15.
- Esfuerzo: M.

## UGC-004 — Python 3.14 y Actions
- Archivo: CI.
- Aceptación: Windows/Linux verde en targets.
- Dependencias: UGC-002.
- Prioridad: P1.
- Severidad/riesgo: Medium/9.
- Esfuerzo: S.

## UGC-005 — Guard de versión SQLite
- Archivos: `db.py`, tests.
- Aceptación: DB futura falla claramente.
- Prioridad: P1.
- Severidad/riesgo: Medium/12.
- Esfuerzo: S/M.

## UGC-006 — Limitar SQL generado
- Archivos: `oracles.py`, tests.
- Aceptación: query normal funciona; patológica se aborta.
- Prioridad: P1.
- Severidad/riesgo: Medium/12.
- Esfuerzo: M.

# Verificación y checklist post-upgrade
- [ ] Instalación reproducible.
- [ ] Ruff, Mypy, Pytest, cobertura >=92 %.
- [ ] CI 3.11/3.13/3.14 Windows/Linux.
- [ ] Fixture añadido/eliminado invalida comparabilidad.
- [ ] HTTP remoto con token bloqueado.
- [ ] Auditoría de dependencias sin findings no aceptados.
- [ ] DB futura rechazada.
- [ ] SQL patológico abortado.

# Supuestos y límites
- Sin cambio de semántica estadística.
- No se presupone TLS nativo en AI Broker; puede terminarse en proxy/túnel.
- No se ejecutó el proyecto.
