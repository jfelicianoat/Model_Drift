# Probar y calibrar System 1

Revisión: 3 de octubre de 2026, contrato 2.11 §15.8 y adenda 09.

## Uso desde escritorio y CLI

La pestaña **Probar System 1** permite cargar un plan JSON, elegir el maestro del
catálogo y, opcionalmente, sustituir proveedor/modelo del alumno. Produce informe
HTML, artefacto JSON y exportación de casos con etiquetas y procedencia.

```powershell
model-drift modelos
model-drift probar-system1 --plan suites/system1/calibration-demo.json --maestro "proveedor/despliegue/modelo"
model-drift revalidar-system1
```

El maestro preferido por el usuario es **Qwen3.8**. Su nombre completo y despliegue
deben confirmarse en el catálogo. El plan de demostración contiene un marcador que
hay que sustituir con `--maestro` o desde el escritorio. Sus cuatro casos sirven para
probar el recorrido; carecen de verificación humana y no demuestran calibración.

Para una medición de producción, copiar exactamente `use_case`, `decision_type`,
`instructions`, `threshold_profile` y opciones/criterios/rúbrica de la aplicación.
El plan declara explícitamente la clasificación de datos de cada caso. El maestro
nunca recibe etiquetas humanas ni respuestas o puntuaciones del alumno.

## Plan y referencias

El JSON versionado contiene `config` y `cases`. Cada caso tiene `id`, `input`,
`provenance` (`redactado`/`capturado`), `data_class`, `adversarial` y, si procede,
`human_label` con `human_verified: true`. Sólo una persona puede aportar esa
verificación. Se rechazan identificadores y entradas duplicadas.

El [conjunto sintético](../suites/system1/README.md) contiene 80 parejas del caso
semántico de Model_Drift, equilibradas entre sus cuatro etiquetas. El usuario
autorizó redactar estos ejemplos para la primera exploración. Se guardan como
`expected_label` con `expected_label_source: synthetic_authored`, separados de
`human_label`. Ni el maestro ni el alumno reciben esas referencias. El informe
muestra un diagnóstico exploratorio con aciertos, confusión, kappa del maestro
frente a la referencia sintética y fiabilidad del alumno; ese kappa no sustituye
la comprobación humana ni permite recomendar un umbral. Una hoja CSV acompaña
los casos para su futura revisión.

El maestro habitual es `teacher_task: {provider, deployment, model}`: tarea normal,
modelo exacto, sin fallback, exclusión del aprendizaje, temperatura cero y semilla.
La telemetría conserva `seed_status` y huella; solicitar semilla no demuestra
determinismo. Alternativamente, `teacher_judge: {provider, model}` permite un
maestro generativo fijado por §15.8. Se usa su etiqueta cruda y se ignora la confianza
`self_reported`, incluso cuando vale 1. Puede configurarse `second_teacher_task`;
sus discrepancias requieren revisión humana.

Se realizan cinco repeticiones por defecto (mínimo tres), barajando criterios. Sólo
entran etiquetas unánimes. Antes de usarlas como referencia se exige un subconjunto
humano de 50 casos por defecto, al menos dos clases, proporción máxima por clase 0,8
y kappa mínimo 0,8. Todo es configurable. El acuerdo del maestro se mide por caso de
uso; su unanimidad no convierte por sí sola una etiqueta en verdad.

Hay calentamiento separado de la latencia medida. `TIMEOUT` del maestro permite
reintentos acotados; agotarlos hace fallar la tirada explícitamente. La cancelación
conserva el informe parcial en SQLite. Los datos `confidential`/`local_only` fuerzan
`cloud_allowed: false` tanto para alumno como para maestro.

## Notas y bloqueantes del Broker

Se comprueba `system1_judgments` y la ampliación aditiva `system1_evaluation`, sin
deducirlas del número 2.11. `target` fija un único proveedor del alumno. La identidad
usada en el informe es el eco del intento; se admiten alias resueltos, pero cambios
de proveedor/modelo durante la tirada la invalidan.

Cada nota exige fuente nativa, tipo/dominio correctos, confianza finita en [0,1],
alternativas completas sin duplicados, primera opción dominante y suma cercana a
uno con tolerancia configurable. Las notas inválidas se excluyen y se cuentan;
exceder la proporción configurada impide recomendar umbral.

`accepted: false` es una abstención para la decisión operativa. Una nota nativa
rechazada por baja confianza o margen sigue entrando en la calibración. Un timeout
o salida inválida sin decisión no se cuenta como acierto ni como fallo. Las respuestas
crudas se guardan junto a las notas para poder auditar exclusiones.

Sin capacidad para fijar el modelo, notas crudas o huellas verificadas, el informe
declara la limitación y no recomienda umbral. El flujo antiguo sólo muestra el modelo
servido y abstenciones: no fabrica una calibración a partir de la cola aceptada.

## Medición y artefacto

La partición se fija antes de obtener puntuaciones: 60 % calibración / 40 % prueba,
semilla 1234 por defecto. Se elige el menor umbral observado en calibración cuya
cota inferior exacta de Clopper–Pearson alcanza la precisión objetivo. Se comprueba
ese único umbral en la prueba separada; si falla, no se selecciona otro mirando el test.
`auto_approve_values` limita las etiquetas cuyo acierto se quiere garantizar.

Se publican exactitud, precisión/exhaustividad por clase, matriz de confusión,
abstenciones y motivos, notas inválidas, fiabilidad por tramos, Brier multiclase,
ECE de la primera opción, intervalos exactos, cobertura, costes y tiempos. Los
adversarios se miden aparte. Los tramos son [inicio, fin), salvo el último que incluye 1.

Con confianza del 95 % bilateral y precisión objetivo del 99 %, incluso **368 aciertos
de 368** apenas superan el requisito. El número de casos del test seleccionados por
el umbral es el que importa; una pequeña muestra perfecta resulta indeterminada.

El artefacto incluye fecha, plan, identidad y huellas, procedencia humana/destilada/sintética,
tamaños, umbral validado, cota de precisión y cobertura. Los estados son `calibrado`,
`no_calibrado` e `indeterminado`. Son datos locales de Model_Drift: aplicar un umbral
al Broker o retirar el modo sombra de Agora requiere una decisión del usuario.

## Revalidación y caducidad

Actualizar el catálogo en el escritorio o ejecutar `revalidar-system1` comprueba las
huellas de los planes guardados. Si cambia alumno o maestro, caduca la calibración
anterior y se repite el plan con sus casos originales. No hay proceso permanente.
Se puede desactivar por plan con `revalidate_on_fingerprint_change: false`.

Las repeticiones manuales también comparan con la prueba anterior. Se utiliza McNemar
en los mismos casos de prueba y con el umbral anterior, contrastando la ausencia de
aprobaciones erróneas. Así también se detecta una pérdida de precisión al aprobar
más negativos, aunque el número de aprobaciones correctas siga siendo el mismo.
Se excluyen referencias que discrepan entre tiradas. Se informa regresión cuando
baja la precisión al umbral y aumenta significativamente la tasa de aprobaciones erróneas.
Los informes registrados y SQLite marcan `expired: true`; el umbral histórico se
conserva para auditoría y el resumen deja de recomendarlo. Se caducan todos los
certificados afectados, incluso cuando la nueva tirada falla. Una componente recién
informada, sin divergencia en las observables compartidas, no implica otro modelo.

El maestro debe volver a superar la comprobación humana si cambia su huella. Un
control negativo consiste en repetir el mismo plan/modelo/huella y comprobar que
no aparece regresión; es obligatorio además del ensayo con un modelo cambiado.

## Verificación disponible

Las pruebas locales usan el transporte real del cliente con respuestas simuladas
según §15.8, y cubren también CLI, escritorio, persistencia y caducidad. No certifican
un proveedor real. La prueba en vivo y la primera medición con Qwen3.8 están pendientes
de acceso al Broker configurado y confirmación de su catálogo. Esta sesión deniega
la conexión de red con el código Windows 10013. No se ha medido una
precisión real ni se ha emitido un umbral para producción.
