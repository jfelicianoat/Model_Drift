# Casos de System 1

`semantic-synthetic.json` contiene **80 casos inventados**, 20 por cada etiqueta
de Model_Drift: equivalencia, cambio benigno, candidato de regresión e incertidumbre.
Incluye siete casos con instrucciones incrustadas. Usa las mismas instrucciones,
opciones, criterios y perfil `default` del flujo semántico de la aplicación.

Las referencias llevan `expected_label_source: synthetic_authored`; todos los
casos tienen `human_verified: false`. La etiqueta esperada se conserva fuera de
`input`, para que ni el maestro ni el alumno la reciban. Todos son `local_only`.
El diagnóstico compara sus respuestas con estas referencias y muestra aciertos,
confusión y fiabilidad. Estos resultados son exploratorios y no certifican un
umbral ni la calidad del maestro frente a humanos.

El maestro preferido es Qwen3.8. Hay que elegir su referencia real en el catálogo;
`SELECCIONAR_EN_CATALOGO` es un marcador, no un modelo ejecutable. En la pestaña
**Probar System 1**, cargar el plan y elegir el maestro. La misma prueba en CLI:

```powershell
model-drift probar-system1 --plan suites/system1/semantic-synthetic.json --maestro "proveedor/despliegue/modelo"
```

Con cinco repeticiones por caso y calentamiento separado, la prueba solicita
400 juicios medidos al maestro y 80 al alumno. La partición fija tiene 48 casos
de calibración y 32 de prueba. Su tamaño no demuestra una precisión del 99 %.
Un resultado `indeterminado` es esperable mientras falte la verificación humana.

`semantic-synthetic.review.csv` facilita revisar cada pareja, su referencia y la
justificación inventada. Las columnas `human_label` y `human_verified` quedan sin
verificación; una persona puede utilizarlas para registrar su revisión. El CSV
no se importa automáticamente: sus etiquetas revisadas deben trasladarse al plan
JSON y validar que no se conservan desacuerdos sin resolver.

`semantic-synthetic.labels.json` contiene las referencias sintéticas estructuradas
para `model_drift.system1_benchmark`. Sólo sirve con una auditoría que tenga los
mismos identificadores; el benchmark preserva la procedencia sintética.

`calibration-demo.json` es una demostración de cuatro casos del caso de uso
`goal_completion`. Carece también de verificación humana y sirve para comprobar
el recorrido, no para calibrar el caso semántico de Model_Drift.
