# Contrato de integración con AI Broker

Revisión: **3 de octubre de 2026** (contrato 2.11 y ampliación §15.8).

Este documento pertenece a Model_Drift y describe únicamente lo que su cliente necesita.
No modifica ni sustituye la documentación de AI Broker.

## Baseline requerido

Model_Drift está diseñado para el contrato **2.10**, y funciona contra un 2.9
declarando lo que no puede demostrar. Antes de ejecutar una suite consulta
`/api/v1/capabilities` y no presupone capacidades por el número de versión.

| Necesidad | Uso en Model_Drift |
| --- | --- |
| Modelo exacto | `model_requirements.target_model` con `fallback_allowed: false` |
| Aislamiento del router | `exclude_from_model_learning: true` |
| Control | `generation.seed` cuando está disponible |
| Identidad | `served_by`, `models_used`, `fallback_used` y huella de ejecución |
| Evidencia temporal | telemetría de invocaciones y resumen de ejecución |
| Durabilidad | creación idempotente, polling y cancelación de tareas |

## Lo que aporta el contrato 2.10

El 2.10 es aditivo, pero para una herramienta de medición no es cosmético: es la
diferencia entre dar un número y poder enseñar cómo se produjo.

| Capacidad | Qué resuelve en Model_Drift |
| --- | --- |
| `invocation_contract` (§8.1) | Cada invocación declara `contractual`. Bajo un mismo `task_id` conviven la llamada que ejecuta la tarea y el **sondeo en sombra** que el Broker lanza por su cuenta, y el sondeo llega *después*. Model_Drift leía la última invocación para sacar los parámetros efectivos y la huella: si esa era un `shadow_probe`, le atribuía al modelo medido la identidad de otro. Se filtra por el booleano, nunca por el nombre del rol |
| `prompt_compression_echo` (§8.5) | Se pide `prompt_compression: "off"` y ahora hay acuse de recibo. Una medición sobre un prompt podado no compara lo que dice comparar; si el eco dice que se comprimió igual, la tirada lo declara en `missing_guarantees` en vez de callarlo |
| `auxiliary_invocations: false` (§8.4) | Exclusividad de contenido explícita. La garantía implícita (`target_model` + `fallback_allowed: false`) ya cubría estas tareas, pero solo contra un 2.10 y sin dejar rastro en la petición |
| `canonical_artifacts` (§8.3) | El entregable se cierra por el artefacto marcado `final: true`, con su `sha256` sobre los bytes exactos. `result` no tiene esquema en el OpenAPI y no va a tenerlo |
| `content_source` (§8.1) | `reasoning_content` significa que el modelo dejó `content` vacío y el Broker rescató su razonamiento. La respuesta vale, pero al **comparar modelos** eso importa y se declara |
| `role`/`status` enumerados | Se validan contra el vocabulario del contrato. `unknown` es «no lo reconozco», no un error |

Cada tirada sella en su manifiesto (`broker_contract`) qué prometía el Broker **en
marcha** cuando se ejecutó, no lo que dice el documento: una misma suite contra un
2.9 y contra un 2.10 produce evidencia distinta aunque los números coincidan.

La evidencia por invocación se persiste en la tabla `invocation`
(`prompt_compression`, `content_source`, `deliverable_sha256`,
`auxiliary_invocations`, `auxiliary_roles_json`). El esquema sube a la versión 2 y
la migración es aditiva: una base creada antes gana las columnas al abrirse.

## Juicios y calibración System 1 (2.11)

Los juicios directos son síncronos: `POST /api/v1/system1/judge` con
`X-Admin-Token`, timeout HTTP configurable y sin identificadores de tarea,
idempotencia ni sondeo. El Broker elige su orden de proveedores. Un rechazo
mantiene el análisis previo; `fallback_used` no sustituye a `accepted`.

Se consulta `system1_judgments`. Para calibrar se exige además la ampliación
aditiva `system1_evaluation`: permite `target: {provider, model?}` sin fallback
y notas nativas crudas en `attempts[]`, incluidas las rechazadas por umbral.
Las capacidades se comprueban en ejecución; el número 2.11 por sí solo no las demuestra.

La etiqueta de un maestro generativo en el mismo endpoint viene del intento,
con `SELF_REPORTED_SCORE` y `accepted: false`; su confianza se ignora. El alumno
debe tener probabilidades nativas completas. La calibración también puede usar
un maestro fijado mediante las tareas normales del Broker.

Un caso de uso propio pide `threshold_profile: default` e instrucciones explícitas
o un perfil acordado con el operador. Las peticiones de calibración reproducen
el perfil de producción y no rebajan sus umbrales para recuperar notas bajas.
Los datos locales/confidenciales fuerzan la prohibición de cloud.

Sin evaluación, puntuaciones completas o huella verificada se declaran limitaciones
y no se emite umbral. La persistencia añade `comparison_review` (versión 3) y
`calibration_run`/`calibration_case` (versión 4), con migración aditiva.
Detalles en [SYSTEM1_CALIBRATION.md](SYSTEM1_CALIBRATION.md).

## Degradación de las tareas normales

- Si no existe exclusión del aprendizaje, la tirada no empieza salvo
  `--acepto-contaminar` o `accept_learning_contamination: true`.
- Si no existe semilla, la ejecución puede continuar pero declara que la reproducibilidad
  exacta no está disponible.
- Si falta identidad técnica o remota, el manifiesto conserva el hueco como
  `no_disponible`; nunca lo rellena por inferencia.
- Si el Broker declara fallback pese a haberlo prohibido, la invocación se considera
  inválida para comparar.
- Si el Broker no declara `contractual` por invocación (2.9 o anterior), el trabajo
  auxiliar solo se puede separar por el nombre del rol y así se declara.
- Si no hay eco de compresión, la tirada **no afirma** que el prompt llegase íntegro:
  «no consta» y «se cumplió» no son lo mismo.
- Si el Broker hace invocaciones auxiliares y no acepta el opt-out, se declara que el
  contenido pudo verlo un modelo distinto del medido.

La copia tipada mantenida por el proyecto está en `model_drift/broker_contracts.py` y las
pruebas del transporte usan `tests/broker_fake.py` con la misma semántica relevante.

