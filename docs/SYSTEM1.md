# System 1 en Model_Drift

Revisión: 3 de octubre de 2026. Fuentes: `07_Model_Drift_SYSTEM1.md`,
`09_ADENDA_Model_Drift_Calibracion_System1.md` y `Client_API.md` 2.11, ampliación §15.8.

## Arquitectura y cambio mínimo

El proyecto ya separa ejecución, checks, comparación estadística, política, juez
pareado, SQLite e informes de CLI y Tkinter. La comparación sigue usando casos como
unidad estadística y exige evidencia para declarar dirección. System 1 se añade al
comparador como filtro de revisión de una salida representativa por caso:

1. Los checks deterministas y la comparabilidad se resuelven primero.
2. El cliente consulta `system1_judgments` y envía un juicio síncrono al Broker.
3. Una etiqueta aceptada orienta la prioridad; una abstención conserva la revisión.
4. El juez existente analiza los casos pendientes si está configurado.
5. El informe conserva las medidas originales y añade una auditoría separada.

No se modifican los contrastes, la política ni la evidencia de dirección a partir
de una etiqueta barata. La calibración reutiliza el cliente de tareas, kappa de
`judge`, McNemar de `statistics`, SQLite, informes y escritorio.

## Configuración

`model_drift.yaml` conserva el comportamiento anterior por defecto:

```yaml
system1:
  enabled: false
  shadow_mode: true
  use_case: model_drift_semantic
  threshold_profile: default
  timeout_seconds: 75.0
  cloud_allowed: false
  regression_threshold: 0.80
  high_priority_threshold: 0.95
  equivalence_threshold: 0.99
  allow_equivalence_skip: false
  require_calibrated_for_skip: true
```

Para observar propuestas, activar `enabled` y mantener `shadow_mode`. El nombre
`model_drift_semantic` es propio: el perfil `default` evita `UNKNOWN_USE_CASE`;
un perfil específico debe existir previamente en el Broker. Las instrucciones y
los cuatro criterios están en `model_drift/system1.py`.

`judge_model` admite la referencia completa `proveedor/despliegue/modelo` del juez
profundo. Si falta, se informa «análisis pendiente». El maestro de una calibración
se elige en su plan o en la pestaña de prueba y es independiente de este juez.

El modo activo ordena candidatos de regresión para revisión. Saltar una revisión
por equivalencia requiere una opción explícita, confianza muy alta, calibración
declarada por el Broker y ausencia de checks indeterminados o métricas bloqueantes.
El modo sombra registra la propuesta y mantiene la acción anterior.

## Ejemplos de interpretación

Estos ejemplos explican el flujo; no son mediciones de un proveedor real.

| Caso | Checks deterministas | Propuesta de System 1 | Decisión del flujo |
| --- | --- | --- | --- |
| «Devuelve 42» / «El resultado es 42» | Válidos; redacción distinta | `equivalent` | Mantener revisión en sombra; sólo saltarla bajo la política explícita descrita |
| Añadir una explicación correcta y útil | Válidos; contenido distinto | `changed_but_not_regression` | Revisión ordinaria; no inferir mejora estadística de la etiqueta |
| Respuesta que pierde un matiz obligatorio | Checks estructurales válidos | `regression_candidate` | Priorizar juez/oráculo; la regresión final requiere evidencia suficiente |
| Salida ambigua o Broker con timeout | No hay evidencia semántica suficiente | `uncertain` o fallback | Conservar revisión y declarar el motivo; no afirmar ausencia de deriva |

Un JSON inválido con una aserción `json_valid` se detecta antes del juicio.
Una salida idéntica con herramientas idénticas y sin efecto medido se resuelve
determinísticamente. Una pareja incompleta, inestable o incomparable queda pendiente.

## Auditoría y límites

`comparison_review` guarda configuración, propuestas, intentos, proveedor/modelo,
notas, alternativas, tiempos, razones y acciones del juez. El HTML y la CLI muestran
pares enviados, fallbacks y análisis profundos ejecutados o pendientes.

Los falsos positivos y negativos permanecen sin medir hasta aportar referencias
etiquetadas. `system1_benchmark.py` admite humanas, sintéticas y destiladas, con
su procedencia separada; las destiladas sólo
cuentan con maestro apto, unanimidad y huella. La nota del Broker sigue sin calibrar
salvo evidencia explícita. Se analiza una salida representativa, no cada repetición.

La opción [Probar System 1](SYSTEM1_CALIBRATION.md) mide el instrumento y entrega
evidencia por caso de uso, modelo y huella; no cambia umbrales de otras aplicaciones.
