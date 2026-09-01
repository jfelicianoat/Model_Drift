# Resumen ejecutivo
- Proyecto Python 3 para evaluar deriva de comportamiento de LLMs, con CLI, GUI Tkinter, SQLite, cliente de AI Broker, oráculos, estadística, políticas e importación de tratamientos externos.
- Base de ingeniería sólida: Ruff, Mypy, Pytest, umbral de cobertura del 92 %, CI Windows/Linux y tests amplios.
- **High / Correctness:** `compare_manifests()` no detecta fixtures añadidos o eliminados; solo compara fixtures presentes en ambos manifiestos.
- **High / Reproducibilidad:** existe `requirements.lock`, pero README y CI instalan `.[dev]` desde rangos `>=`; el lock no gobierna la resolución real.
- **High / Seguridad:** el Broker por defecto usa `http://192.168.1.52:8765` y el token admin viaja en `X-Admin-Token`; no hay confidencialidad de transporte.
- **Medium / Integridad:** el “plan sellado” externo usa SHA-256 autocontenido: sirve como checksum, no como autenticación del origen.
- **Medium / Resiliencia:** el oráculo SQL ejecuta SQL generado por el modelo en SQLite de solo lectura, pero sin límite de pasos/tiempo.
- **Medium / Persistencia:** `schema_version` existe, pero no hay migraciones ni rechazo explícito de versiones futuras.
- **Medium / Mantenibilidad:** hotspots: `comparison.compare_runs` (~199 líneas), `runner.Runner.run` (~173), `external_treatments._as_run` (~135) y `gui.py` grande.
- El stack no está EOL: Python 3.11 sigue en soporte de seguridad hasta 2027-10; Python 3.14 es la serie estable actual.
- Análisis estático por lectura del ZIP; no se ejecutó la aplicación ni se asumió que CI/tests estén verdes.

# Mapa del sistema
```text
CLI / GUI
   ├── config ──> BrokerClient ──> AI Broker
   ├── cases / fixtures / metamorphic / oracles / judge / embeddings
   ├── Runner ──> Database(SQLite)
   ├── comparison + statistics + policy
   └── report (HTML / terminal)

plan JSON + ZIPs ──> external_treatments ──> RunResults ──> comparison/report
```

- Código: `model_drift/*.py`
- Config: `pyproject.toml`, `model_drift.yaml`, `politica-ejemplo.yaml`
- CI: `.github/workflows/ci.yml`
- Docs: `README.md`, `docs/*.md`
- Tests: `tests/*.py`
- Datos de ejemplo: `suites/ejemplo/**`, `relaciones/**`

Entrypoints:
- CLI: `model_drift.cli:main`.
- Desktop: `model_drift.gui:main` mediante `arrancar_model_drift.bat`.
- Wrapper Windows: `model_drift.bat`.

# Cómo funciona
1. Carga configuración y suite.
2. `BrokerClient` consulta capacidades/catálogo, fija modelo, prohíbe fallback y solicita exclusión del aprendizaje.
3. `Runner` ejecuta repeticiones, aplica evaluadores, agrega y persiste resultados/manifiesto.
4. `compare_runs()` valida comparabilidad, calcula contrastes y aplica política.
5. `report.py` genera terminal/HTML.
6. `external_treatments.py` importa artefactos externos validados por hash/esquema sin extraer ZIP.

# Hallazgos por fichero

## model_drift/manifest.py
### Rol del fichero
Manifiestos e invariantes de comparabilidad.

### Hallazgos
| Severidad | Tipo | Impacto | Probabilidad | Riesgo | Evidencia | Recomendación | Cambio sugerido |
|---|---|---:|---:|---:|---|---|---|
| High | Correctness | 5 | 3 | 15 | L251-L253 solo recorre intersección de claves de `fixtures` | Detectar diferencia simétrica | Añadir `set(base.fixtures) ^ set(cand.fixtures)` y tests |
| Medium | Testing | 3 | 2 | 6 | Tests visibles cubren fixture cambiado, no alta/baja | Cubrir invariantes negativos | Dos tests específicos |

## pyproject.toml / requirements.lock / .github/workflows/ci.yml
### Rol del fichero
Packaging, dependencias y CI.

### Hallazgos
| Severidad | Tipo | Impacto | Probabilidad | Riesgo | Evidencia | Recomendación | Cambio sugerido |
|---|---|---:|---:|---:|---|---|---|
| High | Dependency/Reliability | 4 | 4 | 16 | CI ejecuta `pip install -e ".[dev]"`; el lock no se usa | Resolver desde lock/constraints | Integrar lock en CI |
| Medium | Dependency | 3 | 4 | 12 | `checkout@v4`, `setup-python@v5`; upstream actual publica v7 | Actualizar por PR aislada | v6 conservador / v7 modernización |
| Medium | Testing | 3 | 3 | 9 | Matriz 3.11/3.13; falta 3.14 | Añadir 3.14 | Mantener 3.11 temporalmente |
| Low | Maintainability | 2 | 3 | 6 | Falta `[build-system]` explícito | Declarar backend | `setuptools.build_meta` |

## model_drift/config.py / model_drift/broker_client.py / model_drift.yaml
### Rol del fichero
Configuración y transporte al Broker.

### Hallazgos
| Severidad | Tipo | Impacto | Probabilidad | Riesgo | Evidencia | Recomendación | Cambio sugerido |
|---|---|---:|---:|---:|---|---|---|
| High | Security | 5 | 3 | 15 | URL por defecto HTTP y token en `X-Admin-Token` | TLS o túnel seguro | Rechazar token sobre HTTP no-loopback salvo opt-in |
| Medium | Config | 3 | 4 | 12 | IP privada hard-coded en código/config/docs | Desacoplar red local | localhost o URL obligatoria |

## model_drift/external_treatments.py
### Rol del fichero
Valida e importa planes/ZIP externos.

### Hallazgos
| Severidad | Tipo | Impacto | Probabilidad | Riesgo | Evidencia | Recomendación | Cambio sugerido |
|---|---|---:|---:|---:|---|---|---|
| Medium | Security | 4 | 2 | 8 | `content_sha256` se recalcula desde el mismo JSON que contiene el hash | No llamarlo sello criptográfico sin autenticidad | Firma detached/Sigstore/HMAC o digest por canal separado |
| Low | Reliability | 2 | 2 | 4 | Hay límites de tamaño y no se extrae ZIP: positivo | Mantener hardening | Límite de miembros/ratio si el contrato crece |

## model_drift/oracles.py
### Rol del fichero
Aserciones deterministas/semánticas.

### Hallazgos
| Severidad | Tipo | Impacto | Probabilidad | Riesgo | Evidencia | Recomendación | Cambio sugerido |
|---|---|---:|---:|---:|---|---|---|
| Medium | Reliability/Security | 4 | 3 | 12 | SQL generado por modelo se ejecuta read-only, sin progress handler/authorizer | Acotar coste y superficie SQL | `set_progress_handler`, `set_authorizer`, límite de filas, `SELECT/WITH` |

## model_drift/db.py
### Rol del fichero
Persistencia SQLite.

### Hallazgos
| Severidad | Tipo | Impacto | Probabilidad | Riesgo | Evidencia | Recomendación | Cambio sugerido |
|---|---|---:|---:|---:|---|---|---|
| Medium | Reliability | 4 | 3 | 12 | `_migrate()` crea schema y solo inserta versión si falta | Migraciones versionadas y rechazo de DB futura | `migrations/<version>.sql` transaccionales |
| Low | Security | 3 | 2 | 6 | `insert()` interpola tabla/columnas; callers actuales son internos | Whitelist | Constantes/Enum de tablas |

## model_drift/comparison.py
### Rol del fichero
Motor de comparación.
| Severidad | Tipo | Impacto | Probabilidad | Riesgo | Evidencia | Recomendación |
|---|---|---:|---:|---:|---|---|
| Medium | Maintainability | 3 | 4 | 12 | `compare_runs()` ~199 líneas | Separar preparación, métrica, agregación y report |

## model_drift/runner.py
### Rol del fichero
Orquestación de tiradas.
| Severidad | Tipo | Impacto | Probabilidad | Riesgo | Evidencia | Recomendación |
|---|---|---:|---:|---:|---|---|
| Medium | Maintainability | 3 | 4 | 12 | `Runner.run()` ~173 líneas | Extraer execute/evaluate/persist |

## model_drift/gui.py
### Rol del fichero
UI Tkinter.
| Severidad | Tipo | Impacto | Probabilidad | Riesgo | Evidencia | Recomendación |
|---|---|---:|---:|---:|---|---|
| Medium | Architecture | 3 | 4 | 12 | UI, estado, hilos y coordinación en un fichero grande | Extraer controllers/presenters |

## model_drift/statistics.py
### Rol del fichero
McNemar, Fisher, permutación, bootstrap, BH, potencia.
| Severidad | Tipo | Impacto | Probabilidad | Riesgo | Evidencia | Recomendación |
|---|---|---:|---:|---:|---|---|
| Low | Testing | 3 | 2 | 6 | Implementaciones propias requieren oráculos de referencia | Tests de equivalencia/property tests |

## model_drift/broker_contracts.py
### Rol del fichero
Espejo tipado del contrato Broker.
| Severidad | Tipo | Impacto | Probabilidad | Riesgo | Evidencia | Recomendación |
|---|---|---:|---:|---:|---|---|
| Medium | Architecture | 4 | 3 | 12 | Espejo manual exige sincronización humana | Contract test/schema diff automático |

## Resto de módulos
`cases.py`, `fixtures.py`, `fingerprint.py`, `metamorphic.py`, `judge.py`, `embeddings.py`, `policy.py`, `report.py`, `taxonomy.py`, `domain.py`, `cli.py`: no se identificó por lectura otro defecto Critical/High respaldado por código. Deben permanecer en regresión ante los cambios propuestos.

# Hallazgos transversales
- Arquitectura: buena separación conceptual; los principales límites están en orquestadores largos y GUI.
- Fiabilidad: comparabilidad de fixtures y migración DB son los riesgos más concretos.
- Seguridad: no se persiste el token; el canal HTTP es la debilidad principal.
- Integridad: hashes externos son checksum, no autenticación.
- Observabilidad: `structlog` está declarado, pero falta una estrategia transversal visible de correlación/métricas.
- Tests: cobertura aparente fuerte; faltan casos negativos específicos y gates de dependencias.
- Dependencias: versiones lock están bastante actuales; el problema es reproducibilidad del proceso.

# Estándares recomendados
- Ruff + Mypy + Pytest como gates obligatorios.
- Lock reproducible y auditoría automática de dependencias.
- ADRs breves para comparabilidad, SQL oracle e integridad externa.
- Migraciones SQLite versionadas.
- Logging con `run_id`, `case_id`, `model_key`, `correlation_id`.
- Contract test con AI Broker.
- TLS obligatorio para credenciales fuera de loopback.

# Roadmap
## Quick wins
- Corregir diferencia simétrica de fixtures.
- Hacer efectivo el lock.
- Declarar `[build-system]`.
- Añadir 3.14 a CI.
- Política de token/HTTP.

## Medio plazo
- Migraciones SQLite.
- Guardrails SQL.
- Actions actuales.
- Contract test Broker.
- Descomponer `compare_runs` y `Runner.run`.

## Largo plazo
- Controllers de GUI.
- Observabilidad end-to-end.
- Firma de planes externos si cruzan frontera no confiable.

# Tareas para ejecución
| ID | Título | Archivos | Prioridad | Severidad/Riesgo | Esfuerzo | Criterio de aceptación |
|---|---|---|---|---|---|---|
| FIX-001 | Fixtures asimétricos | `manifest.py`, tests | P0 | High/15 | S | alta/baja => `comparable=False` |
| CI-001 | Lock efectivo | pyproject/lock/CI | P0 | High/16 | M | CI no resuelve fuera del lock |
| SEC-001 | Token sobre canal seguro | config/client/docs/tests | P0 | High/15 | M | HTTP remoto con token bloqueado |
| DB-001 | Migraciones | `db.py`, tests | P1 | Medium/12 | M | migración transaccional |
| SQL-001 | Limitar SQL generado | `oracles.py`, tests | P1 | Medium/12 | M | query patológica abortada |
| ARCH-001 | Descomponer orquestadores | comparison/runner | P2 | Medium/12 | L | misma API, menor complejidad |
| GUI-001 | Separar controllers | `gui.py` | P2 | Medium/12 | L | workers no tocan widgets |

# Inventario de símbolos del código de producción
Se enumeran clases/funciones/métodos detectados por AST. Es mapa de touchpoints, no prescripción de cambio.

- **model_drift/__init__.py** (12 LOC): _sin clases/funciones_
- **model_drift/broker_client.py** (627 LOC): `BrokerConfig`@L50, `InvocationOutcome`@L66, `InvocationOutcome.produced_output`@L89, `BrokerClient`@L94, `BrokerClient.__init__`@L97, `BrokerClient.close`@L114, `BrokerClient.__enter__`@L118, `BrokerClient.__exit__`@L121, `BrokerClient._headers`@L124, `BrokerClient._declare_missing`@L129, `BrokerClient._request`@L133, `BrokerClient.capabilities`@L157, `BrokerClient.probe_contract`@L161, `BrokerClient.catalog`@L209, `BrokerClient.features_for`@L264, `BrokerClient.entry_for`@L295, `BrokerClient._build_request`@L304, `BrokerClient._submit`@L361, `BrokerClient.invocations`@L388, `BrokerClient._poll`@L417, `BrokerClient.invoke`@L430, `BrokerClient.fingerprint_for`@L529, `_huella_publicada`@L594, `_canonical_params`@L619, `_first_str`@L623
- **model_drift/broker_contracts.py** (307 LOC): `TaskStatus`@L47, `TaskStatus.is_terminal`@L70, `is_terminal_status`@L77, `InferenceKind`@L84, `ModelReference`@L89, `ModelReference.key`@L97, `ModelCatalogEntry`@L101, `ModelCatalogEntry.key`@L130, `ModelCatalogEntry.reference`@L134, `ModelCatalogEntry.supports`@L137, `SeedStatus`@L159, `InvocationTelemetry`@L171, `InvocationTelemetry.seed_status`@L188, `InvocationTelemetry.model_reference`@L196, `TaskState`@L203, `TaskState.is_terminal`@L220, `TaskState.succeeded`@L224, `TaskState.error_code`@L228, `TaskState.assistant_content`@L237, `TaskState.embedding`@L247, `TaskState.model_used`@L256, `TaskState.fallback_used`@L271, `TaskState.usage`@L277, `TaskState.pending_tool_calls`@L284, `BrokerError`@L291, `BrokerError.__init__`@L294, `BrokerError.is_auth`@L300
- **model_drift/cases.py** (268 LOC): `Provenance`@L32, `Message`@L37, `CaseContext`@L42, `CaseParams`@L51, `CaseEnvironment`@L60, `Case`@L70, `Case.require_input`@L97, `Case.user_prompt`@L105, `Case.system_prompt`@L113, `Case.is_multi_turn`@L120, `Case.provides_level_2_evidence`@L123, `Case.canonical`@L134, `UnstableCasePolicy`@L139, `Suite`@L148, `Suite.unique_ids`@L168, `Suite.case_by_id`@L176, `Suite.hash`@L179, `Suite.short_hash`@L191, `SuiteLoadError`@L195, `_coerce_case`@L199, `load_suite`@L219
- **model_drift/cli.py** (387 LOC): `_client`@L50, `cmd_modelos`@L54, `_filtrar`@L85, `_filtrar.pasa`@L92, `_tri`@L111, `cmd_ejecutar`@L115, `_load_run`@L174, `cmd_comparar`@L214, `cmd_evaluar_tratamientos`@L244, `cmd_tiradas`@L267, `cmd_potencia`@L279, `build_parser`@L295, `main`@L362
- **model_drift/comparison.py** (510 LOC): `CaseEvidence`@L63, `evidence_for_case`@L77, `ComparisonReport`@L126, `ComparisonReport.by_status`@L143, `ComparisonReport.regressions`@L146, `ComparisonReport.actionable`@L149, `ComparisonReport.content_outcomes`@L152, `ComparisonReport.power_statement`@L163, `RunResults`@L178, `RunResults.metrics`@L191, `RunResults.case_ids`@L194, `_per_case_p_value`@L198, `_is_underpowered`@L230, `_status_for_case`@L245, `_direction_for_case`@L291, `compare_runs`@L312
- **model_drift/config.py** (99 LOC): `AppConfig`@L25, `AppConfig.broker_token`@L57, `AppConfig.broker_config`@L60, `AppConfig.model_reference`@L67, `AppConfig.load`@L83, `AppConfig.save`@L92
- **model_drift/db.py** (212 LOC): `Database`@L139, `Database.__init__`@L142, `Database._migrate`@L152, `Database.close`@L159, `Database.__enter__`@L162, `Database.__exit__`@L165, `Database.insert`@L170, `Database.insert_many`@L179, `Database.execute`@L190, `Database.query`@L196, `Database.query_one`@L199, `Database.runs`@L202, `_adapt`@L206
- **model_drift/domain.py** (273 LOC): `default_higher_is_better`@L82, `is_binary_metric`@L92, `ChangeStatus`@L105, `ChangeDirection`@L123, `EvidenceKind`@L135, `PracticalSignificance`@L145, `Comparability`@L160, `RunVerdict`@L171, `CaseOutcome`@L183, `CaseOutcome.is_usable`@L211, `ComparisonOutcome`@L216, `ComparisonOutcome.direction_requires_evidence`@L246, `ComparisonOutcome.is_regression`@L263, `ComparisonOutcome.is_actionable`@L267
- **model_drift/embeddings.py** (178 LOC): `BrokerEmbedder`@L21, `BrokerEmbedder.identity`@L36, `BrokerEmbedder.embed`@L39, `BrokerEmbedder.instrument`@L60, `StaticEmbedder`@L72, `StaticEmbedder.identity`@L84, `StaticEmbedder.embed`@L87, `CalibrationPoint`@L92, `ThresholdCalibration`@L101, `ThresholdCalibration.to_json`@L112, `ThresholdCalibration.save`@L115, `calibrate_threshold`@L122
- **model_drift/external_treatments.py** (398 LOC): `ExternalTreatmentError`@L29, `StrictModel`@L33, `TreatmentRef`@L37, `ExternalTreatmentPlan`@L45, `ExternalTreatmentPlan.treatment_order_is_fixed`@L59, `_canonical_json`@L67, `_sha256_file`@L71, `load_external_plan`@L79, `_load_report`@L100, `_validated_cases`@L157, `_validated_strategy_results`@L182, `_as_run`@L226, `load_external_treatments`@L363
- **model_drift/fingerprint.py** (289 LOC): `ComponentState`@L32, `IdentityLayer`@L44, `FingerprintComponent`@L52, `FingerprintComponent.is_present`@L61, `ExecutionFingerprint`@L94, `ExecutionFingerprint.canonical_hash`@L104, `ExecutionFingerprint.short_hash`@L115, `ExecutionFingerprint.get`@L118, `ExecutionFingerprint.present_names`@L122, `ExecutionFingerprint.missing_names`@L125, `ExecutionFingerprint.layer_coverage`@L128, `ExecutionFingerprint.describe`@L136, `FingerprintVerdict`@L144, `FingerprintComparison`@L156, `FingerprintComparison.explain`@L169, `compare_fingerprints`@L182, `FingerprintBuilder`@L220, `FingerprintBuilder.__init__`@L232, `FingerprintBuilder._layer_for`@L235, `FingerprintBuilder._add`@L241, `FingerprintBuilder.declared`@L249, `FingerprintBuilder.verified`@L256, `FingerprintBuilder.unavailable`@L267, `FingerprintBuilder.hashed`@L272, `FingerprintBuilder.has`@L284, `FingerprintBuilder.build`@L288
- **model_drift/fixtures.py** (108 LOC): `FixtureError`@L24, `FixtureResolver`@L28, `FixtureResolver.__init__`@L35, `FixtureResolver.is_reference`@L41, `FixtureResolver.path_for`@L44, `FixtureResolver.read_bytes`@L59, `FixtureResolver.read_text`@L70, `FixtureResolver.read_json`@L73, `FixtureResolver.read_yaml`@L79, `FixtureResolver.read_structured`@L85, `FixtureResolver.hash_of`@L91, `FixtureResolver.resolve_maybe`@L96, `FixtureResolver.used_hashes`@L106
- **model_drift/gui.py** (942 LOC): `Mensaje`@L63, `ModelDriftGUI`@L70, `ModelDriftGUI.__init__`@L80, `ModelDriftGUI._construir`@L116, `ModelDriftGUI._barra_filtros`@L141, `ModelDriftGUI._modelos_filtrados`@L181, `ModelDriftGUI._modelos_filtrados.pasa`@L186, `ModelDriftGUI._aplicar_filtros`@L204, `ModelDriftGUI._al_seleccionar_modelos`@L235, `ModelDriftGUI._actualizar_seleccion_visible`@L244, `ModelDriftGUI._pestana_modelos`@L258, `ModelDriftGUI._pestana_ejecutar`@L301, `ModelDriftGUI._fila_ruta`@L359, `ModelDriftGUI._elegir_ruta`@L369, `ModelDriftGUI._pestana_tiradas`@L383, `ModelDriftGUI._pestana_comparar`@L415, `ModelDriftGUI._escribir`@L468, `ModelDriftGUI.log`@L476, `ModelDriftGUI._credenciales`@L480, `ModelDriftGUI._cliente`@L484, `ModelDriftGUI._en_hilo`@L496, `ModelDriftGUI._en_hilo.envoltorio`@L502, `ModelDriftGUI._bombear_cola`@L511, `ModelDriftGUI._atender`@L521, `ModelDriftGUI.refrescar_catalogo`@L545, `ModelDriftGUI.refrescar_catalogo.trabajo`@L550, `ModelDriftGUI._pintar_catalogo`@L556, `ModelDriftGUI._modelos_elegidos`@L572, `ModelDriftGUI.lanzar_tirada`@L580, `ModelDriftGUI.lanzar_tirada.trabajo`@L633, `ModelDriftGUI.cancelar_tirada`@L663, `ModelDriftGUI._tirada_terminada`@L667, `ModelDriftGUI.refrescar_tiradas`@L685, `ModelDriftGUI._mostrar_detalle_tirada`@L722, `ModelDriftGUI._run_id`@L757, `ModelDriftGUI.comparar`@L760, `ModelDriftGUI.comparar.trabajo`@L776, `ModelDriftGUI._pintar_comparacion`@L795, `ModelDriftGUI.abrir_informe`@L860, `ModelDriftGUI._al_cerrar`@L869, `ModelDriftGUI.cerrar`@L880, `_tri`@L891, `_num`@L895, `main`@L899
- **model_drift/judge.py** (412 LOC): `Verdict`@L39, `Verdict.inverted`@L47, `JudgeConfig`@L82, `PairJudgement`@L97, `PairJudgement.favors_candidate`@L111, `JudgeMetrics`@L121, `JudgeMetrics.is_calibrated`@L133, `JudgeMetrics.interpretation`@L156, `cohen_kappa`@L183, `cohen_kappa.peso`@L202, `raw_agreement`@L230, `Judge`@L241, `Judge.__init__`@L244, `Judge.instrument`@L250, `Judge.rubric_instrument`@L258, `Judge._prompt`@L263, `Judge._ask`@L272, `Judge.judge_pair`@L284, `Judge.measure_position_bias`@L328, `Judge.measure_self_consistency`@L337, `Judge.measure_human_agreement`@L362, `Judge.is_calibrated`@L388, `_parse_verdict`@L393
- **model_drift/manifest.py** (276 LOC): `Determinism`@L39, `ExactReproducibility`@L47, `ReproducibilityFlags`@L60, `ReproducibilityFlags.declare`@L68, `InstrumentRef`@L92, `InstrumentRef.identity`@L106, `ModelUnderTest`@L110, `ModelUnderTest.declared_identity`@L120, `RunManifest`@L124, `RunManifest.instrument_signature`@L159, `RunManifest.model_by_key`@L167, `RunManifest.declare_missing`@L170, `RunManifest.to_json`@L174, `ComparabilityIssue`@L178, `ComparabilityReport`@L185, `ComparabilityReport.instrument_changed`@L193, `ComparabilityReport.fingerprint_divergent`@L197, `ComparabilityReport.headline`@L200, `compare_manifests`@L208
- **model_drift/metamorphic.py** (373 LOC): `RelationKind`@L38, `TransformationKind`@L43, `Transformation`@L59, `Transformation.fixture_required`@L71, `Comparator`@L81, `Threshold`@L87, `Threshold.calibration_required`@L94, `SuccessCriterion`@L103, `SuccessCriterion.parse`@L110, `SuccessCriterion.satisfied`@L122, `MetamorphicRelation`@L130, `MetamorphicRelation.threshold_required_for_fuzzy`@L143, `MetamorphicRelation.identity`@L151, `VariantResult`@L156, `RelationOutcome`@L165, `generate_variants`@L194, `_variants_from_fixture`@L229, `_shuffle_items`@L245, `_shuffle_items._sustituir`@L265, `compare_outputs`@L278, `evaluate_relation`@L311, `load_relations`@L356
- **model_drift/oracles.py** (628 LOC): `ToolCall`@L36, `EvaluationInput`@L43, `Embedder`@L57, `Embedder.identity`@L61, `Embedder.embed`@L65, `EvaluationServices`@L70, `AssertionResult`@L82, `AssertionResult.counts`@L94, `normalize_text`@L109, `strip_code_fences`@L118, `extract_json`@L130, `resolve_path`@L152, `cosine_similarity`@L173, `BaseAssertion`@L189, `BaseAssertion.assertion_id`@L196, `BaseAssertion.evaluate`@L199, `BaseAssertion._evaluate`@L225, `BaseAssertion._result`@L228, `BaseAssertion._skip`@L237, `ExactAssertion`@L246, `ExactAssertion._evaluate`@L253, `ContainsAssertion`@L262, `ContainsAssertion._evaluate`@L270, `RegexAssertion`@L288, `RegexAssertion._evaluate`@L296, `JsonValidAssertion`@L305, `JsonValidAssertion._evaluate`@L310, `JsonSchemaAssertion`@L315, `JsonSchemaAssertion._evaluate`@L326, `FieldEqualsAssertion`@L341, `FieldEqualsAssertion._evaluate`@L350, `NumericCloseAssertion`@L367, `NumericCloseAssertion._evaluate`@L375, `LengthAssertion`@L387, `LengthAssertion._evaluate`@L396, `ToolCallAssertion`@L413, `ToolCallAssertion._evaluate`@L428, `CitationsGroundedAssertion`@L449, `CitationsGroundedAssertion._evaluate`@L462, `AbstentionAssertion`@L475, `AbstentionAssertion._evaluate`@L496, `SqlResultAssertion`@L506, `SqlResultAssertion._evaluate`@L521, `SemanticSimilarityAssertion`@L550, `SemanticSimilarityAssertion.calibration_required`@L567, `SemanticSimilarityAssertion._evaluate`@L576, `evaluate_assertions`@L607, `aggregate_pass`@L615
- **model_drift/policy.py** (326 LOC): `Severity`@L35, `MetricRule`@L42, `MetricRule.worsens`@L63, `MetricRule.exceeds_tolerance`@L67, `RegressionPolicy`@L88, `RegressionPolicy.rule_for`@L100, `RegressionPolicy.minimum_practical_effect`@L117, `RegressionPolicy.from_file`@L121, `RegressionPolicy.from_mapping`@L126, `MetricVerdict`@L159, `MetricVerdict.is_relevant_change`@L175, `RunVerdictReport`@L179, `RunVerdictReport.trade_off`@L193, `RunVerdictReport.summary`@L196, `build_metric_verdict`@L214, `evaluate_policy`@L273
- **model_drift/report.py** (356 LOC): `direction_label`@L68, `_fmt`@L246, `_environment`@L256, `render_comparison`@L262, `write_comparison`@L310, `render_text_summary`@L321
- **model_drift/runner.py** (608 LOC): `RunConfig`@L87, `RunConfig.params`@L101, `RunProgress`@L111, `RunProgress.fraction`@L124, `RunProgress.describe`@L130, `RunOutput`@L138, `RunOutput.for_model`@L148, `BudgetExceeded`@L152, `Runner`@L156, `Runner.__init__`@L159, `Runner._not_applicable_reason`@L186, `Runner._invoke_case`@L207, `Runner._evaluation_input`@L225, `Runner._run_relations`@L236, `Runner._instruments`@L274, `Runner._build_manifest`@L294, `Runner._seal_reproducibility`@L329, `Runner.run`@L368, `Runner._persist`@L542, `_invocation_row`@L565, `_case_row`@L594
- **model_drift/statistics.py** (506 LOC): `CaseAggregate`@L42, `CaseAggregate.is_empty`@L51, `aggregate_binary`@L55, `aggregate_continuous`@L75, `binarize`@L96, `_binomial_tail`@L113, `McNemarResult`@L121, `McNemarResult.discordant`@L134, `mcnemar_exact`@L138, `fisher_exact_2x2`@L162, `fisher_exact_2x2.probabilidad`@L182, `minimum_attainable_p`@L197, `permutation_test`@L219, `BootstrapResult`@L257, `BootstrapResult.excludes_zero`@L266, `cluster_bootstrap`@L270, `benjamini_hochberg`@L329, `classify_practical`@L363, `PowerEstimate`@L397, `estimate_power`@L409, `minimum_detectable_effect`@L452, `power_curve`@L486
- **model_drift/taxonomy.py** (145 LOC): `_cuenta`@L55, `_idioma_aproximado`@L60, `_tiene_rechazo`@L72, `_tiene_abstencion`@L77, `classify_change`@L82

# Supuestos y límites del análisis
- Fuente principal: `Model_Drift-main.zip`.
- GitHub se consultó para corroborar estructura pública, no para diff byte a byte.
- No se ejecutó aplicación, Broker, SQLite real ni CI.
- No se afirma cobertura real/verde; solo se revisa configuración.
- Se verificaron referencias puntuales de soporte/CVEs; no se ejecutó un SBOM scanner completo.
