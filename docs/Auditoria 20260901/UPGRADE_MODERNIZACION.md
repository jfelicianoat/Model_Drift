# Resumen ejecutivo de modernización
- Target principal Python 3.14; considerar mínimo 3.13.
- Lock moderno reproducible + SBOM + auditoría.
- Modularizar `compare_runs`, `Runner.run` y GUI.
- Contract tests automáticos con AI Broker.
- Migraciones SQLite.
- Observabilidad estructurada/correlacionada.
- Autenticidad verificable para planes externos cuando crucen frontera de confianza.
- Mantener la metodología estadística como núcleo estable.

# Diferencias clave vs plan conservador
| Área | Conservador | Modernización |
|---|---|---|
| Python | >=3.11 + test 3.14 | mínimo 3.13, target 3.14 |
| Build | lock efectivo | lock con hashes/SBOM/bot |
| CI | actions actualizadas | build artifact + security gates |
| Arquitectura | mínimos cambios | pipeline/services/controllers |
| Broker | TLS policy | contract-driven |
| DB | guard | migraciones versionadas |
| Observabilidad | básica | logs+metrics+tracing |
| External plans | checksum claro | firma/autenticidad |

# Targets recomendados (modernización) y justificación
- `requires-python >=3.13`; CI 3.13/3.14.
- `checkout@v7`, `setup-python@v7` con runners compatibles.
- Lock moderno con hashes.
- Wheel/sdist construidos y probados en CI.
- Audit de dependencias + SBOM.
- Migraciones SQLite.
- `structlog` integrado con contexto.

# Plan por áreas (arquitectura/observabilidad/tests/CI/deps/infra)
## Arquitectura
- Descomponer `compare_runs()` en funciones puras.
- Descomponer `Runner.run()` en execute/evaluate/aggregate/persist.
- Extraer controllers de GUI.
- Automatizar contrato Broker.

## Observabilidad
- Contextvars: `run_id`, `case_id`, `model_key`, `correlation_id`.
- JSON logs en CI y human-readable en desktop.
- Métricas de tirada/invocaciones/fallos/cancelaciones/indeterminados.
- Spans opcionales alrededor de Broker/comparison.

## Tests
- Property tests de estadística/manifiesto.
- Contract tests Broker.
- Migration tests.
- Fuzz/negative tests para planes ZIP/JSON y SQL.
- Golden tests mínimos de reportes.

## CI
- 3.13/3.14, Windows/Linux.
- Build wheel/sdist e instalar artefacto.
- Ruff/Mypy/Pytest/coverage.
- Dependency audit + SBOM.
- Contract schema check.
- Dependabot/Renovate.

## Dependencias
- `uv.lock` o equivalente con hashes.
- Patch/minor automáticos; majors manuales.

## Infra
No introducir contenedores salvo necesidad: el producto es local/CLI/desktop.

# Touchpoints

## PHASE_0 — Fundación
- `pyproject.toml`, lock, CI.
- `manifest.py`.
- `config.py`, `broker_client.py`.
- `db.py`.
- `oracles.py`.

## Fase 1 — Upgrade mayor
- Python 3.14.
- Actions v7.
- build artifacts.

## Fase 2 — Refactors
- `comparison.py`.
- `runner.py`.
- `gui.py`.
- `broker_contracts.py`.

## Fase 3 — Hardening
- `external_treatments.py`.
- observabilidad.
- fuzz/property tests.
- performance profiling.

## PLAN_POR_FASES
El inventario completo de símbolos está en `AUDIT_REPORT.md`; ejecutar cambios por módulo para preservar invariantes.

## NEXT_PHASE_ASK
Siguiente módulo recomendado tras Phase 0: **`model_drift/comparison.py`**.

# Roadmap (fases claras)
1. Fundación: correctness, lock, TLS, DB, SQL.
2. Runtime/toolchain: Python 3.14, Actions v7, package artifacts.
3. Refactor: comparison/runner/gui/contract.
4. Hardening: autenticidad, observabilidad, fuzz, performance.

# TAREAS_UPGRADE_MODERNIZACION
## UGM-001 — Baseline Python 3.14
- Archivos: pyproject/CI/docs.
- Aceptación: wheel y tests en 3.13/3.14 Windows/Linux.
- Prioridad: P1.
- Severidad/riesgo: Medium/9.
- Esfuerzo: M.

## UGM-002 — Supply-chain reproducible
- Archivos: packaging/CI.
- Pasos: lock con hashes, SBOM, audit, bot.
- Aceptación: resolución determinista.
- Prioridad: P0.
- Severidad/riesgo: High/16.
- Esfuerzo: M.

## UGM-003 — Pipeline de comparación
- Archivos: `comparison.py`, tests.
- Aceptación: API/resultados equivalentes en fixtures de prueba.
- Prioridad: P2.
- Severidad/riesgo: Medium/12.
- Esfuerzo: L.

## UGM-004 — Runner modular
- Archivos: `runner.py`, tests.
- Aceptación: cancelación/persistencia parcial conservan semántica.
- Prioridad: P2.
- Severidad/riesgo: Medium/12.
- Esfuerzo: L.

## UGM-005 — GUI por controllers
- Archivos: `gui.py` + nuevos módulos.
- Aceptación: workers no tocan widgets; dominio fuera de UI.
- Prioridad: P2.
- Severidad/riesgo: Medium/12.
- Esfuerzo: L.

## UGM-006 — Contract-driven Broker
- Archivos: `broker_contracts.py`, tests/CI.
- Aceptación: CI detecta drift de contrato.
- Prioridad: P1.
- Severidad/riesgo: Medium/12.
- Esfuerzo: M/L.

## UGM-007 — Migraciones SQLite
- Archivos: `db.py`, migrations/tests.
- Aceptación: DB antigua migra transaccionalmente; rollback.
- Prioridad: P1.
- Severidad/riesgo: Medium/12.
- Esfuerzo: M.

## UGM-008 — Autenticidad de planes
- Archivos: `external_treatments.py`, CLI/docs/tests.
- Aceptación: modificar plan + recalcular SHA-256 no basta.
- Prioridad: P1.
- Severidad/riesgo: Medium/8.
- Esfuerzo: M.

## UGM-009 — Observabilidad end-to-end
- Archivos: cross-cutting.
- Aceptación: correlación completa sin token/prompts sensibles por defecto.
- Prioridad: P2.
- Severidad/riesgo: Medium/8.
- Esfuerzo: M.

## UGM-010 — Hardening SQL
- Archivos: `oracles.py`, tests.
- Aceptación: operaciones/coste acotados.
- Prioridad: P1.
- Severidad/riesgo: Medium/12.
- Esfuerzo: M.

# Verificación y checklist post-modernización
- [ ] Wheel/sdist reproducible.
- [ ] Python 3.13/3.14 Windows/Linux.
- [ ] Ruff/Mypy/Pytest/cobertura.
- [ ] Property tests.
- [ ] Contract test Broker.
- [ ] Migration tests.
- [ ] Dependency audit + SBOM.
- [ ] Token sobre canal seguro.
- [ ] Autenticidad de planes si aplica.
- [ ] SQL patológico abortado.
- [ ] Logs correlacionados sin secretos.
- [ ] Benchmark de suite grande.
- [ ] Rollback documentado.

# Supuestos y límites
- No se recomienda reescribir estadística sin evidencia.
- No se recomienda contenedorización por defecto.
- Firma de planes solo si existe frontera de confianza real.
- No se ejecutó el proyecto.
