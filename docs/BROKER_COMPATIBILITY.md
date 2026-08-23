# Contrato de integración con AI Broker

Revisión: **23 de agosto de 2026**.

Este documento pertenece a Model_Drift y describe únicamente lo que su cliente necesita.
No modifica ni sustituye la documentación de AI Broker.

## Baseline requerido

Model_Drift está diseñado para el contrato 2.9. Antes de ejecutar una suite consulta
`/api/v1/capabilities` y no presupone capacidades por el número de versión.

| Necesidad | Uso en Model_Drift |
| --- | --- |
| Modelo exacto | `model_requirements.target_model` con `fallback_allowed: false` |
| Aislamiento del router | `exclude_from_model_learning: true` |
| Control | `generation.seed` cuando está disponible |
| Identidad | `served_by`, `models_used`, `fallback_used` y huella de ejecución |
| Evidencia temporal | telemetría de invocaciones y resumen de ejecución |
| Durabilidad | creación idempotente, polling y cancelación de tareas |

## Degradación explícita

- Si no existe exclusión del aprendizaje, la tirada no empieza salvo
  `--acepto-contaminar` o `accept_learning_contamination: true`.
- Si no existe semilla, la ejecución puede continuar pero declara que la reproducibilidad
  exacta no está disponible.
- Si falta identidad técnica o remota, el manifiesto conserva el hueco como
  `no_disponible`; nunca lo rellena por inferencia.
- Si el Broker declara fallback pese a haberlo prohibido, la invocación se considera
  inválida para comparar.

La copia tipada mantenida por el proyecto está en `model_drift/broker_contracts.py` y las
pruebas del transporte usan `tests/broker_fake.py` con la misma semántica relevante.

