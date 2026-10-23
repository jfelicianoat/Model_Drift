# Model_Drift

Detecta si los prompts de un producto **siguen comportándose igual** cuando cambia el modelo o su configuración de ejecución.

Componente de evaluación del TFM, junto a [AI_Broker](../AI_Broker),
[Knowledge_Orchestrator](../Knowledge_Orchestrator) y [ChatyGPT](../ChatyGPT). El plan
histórico está en [`../docs/Model_Drift_Plan.md`](../docs/Model_Drift_Plan.md); la
arquitectura vigente y el contrato que consume están documentados localmente en
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) y
[`docs/BROKER_COMPATIBILITY.md`](docs/BROKER_COMPATIBILITY.md).

Un benchmark público dice qué modelo es mejor en general. Model_Drift responde otra cosa:

> El prompt nº 17 de tu producto se comportaba así con `qwen3:6b` y así con `qwen3:8b`, el cambio es estadísticamente real, supera el umbral práctico que declaraste, y es de este tipo.

## La idea en una frase

**Cambio no es regresión.** Detectar que una respuesta cambió es un hecho medible. Decir que cambió *a peor* requiere evidencia evaluativa, y si no la hay, la herramienta se calla. Todo lo demás del diseño sale de ahí.

## Arranque rápido en Windows

| Fichero | Qué hace |
|---|---|
| **`arrancar_model_drift.bat`** | **Abre la aplicación de escritorio.** Es la forma normal de usarla |
| `model_drift.bat` | Envoltorio de la línea de comandos, para no recordar la ruta del venv |

La interfaz de escritorio es Tkinter, que viene con Python: **sin dependencias nuevas, sin
servidor, sin puerto y sin navegador**. Cuatro pestañas —Modelos, Ejecutar tirada, Tiradas y
Comparar—, una barra de progreso real durante la tirada y un botón para cancelarla sin perder lo
ya medido.

El informe HTML se genera como fichero y se abre en el navegador con un botón: Tkinter no pinta
HTML y fingirlo con etiquetas sería peor.

Con 162 modelos en el catálogo, las dos listas llevan **filtros por proveedor, por despliegue
(`local`, `cloud`, `api`, o «remotos» agrupando los dos últimos), por texto y por despachables**.
Los filtros son los mismos en las dos pestañas, y **elegir modelos y luego filtrar no pierde la
elección**: se conserva, y la ventana dice cuántos elegidos quedan fuera del filtro en vez de
lanzar la tirada con menos modelos de los que pediste.

El Broker de trabajo es `http://192.168.1.52:8765`, configurado en `model_drift.yaml` y
sobrescribible con `MODEL_DRIFT_BROKER_URL` o desde la propia ventana.

```bat
rem El Broker rota su token en cada arranque: cógelo de su ventana.
set MODEL_DRIFT_BROKER_TOKEN=<token>

model_drift.bat modelos
model_drift.bat modelos --proveedor ollama --despliegue local --solo-despachables
model_drift.bat ejecutar suites\ejemplo --modelo ollama/local/qwen3:6b --relaciones relaciones --acepto-contaminar
model_drift.bat tiradas
model_drift.bat comparar run-A run-B --suite suites\ejemplo --politica politica-ejemplo.yaml --informe informes\x.html
model_drift.bat evaluar-tratamientos --plan plan.json --informe informes\r3-r4.html

arrancar_model_drift.bat
```

`model_drift.bat` no lleva `pause` a propósito: se usa también desde guiones y desde CI, y
propaga el código de salida del comando.

Los dos comprueban que el `.venv` existe, que **arranca** (el venv guarda rutas absolutas y se rompe
al alternar entre los dos PCs) y que Model_Drift está instalado dentro; los tres fallos dan
mensajes distintos porque se arreglan distinto. El de escritorio hace esa comprobación con
`python.exe` y consola visible **antes** de relanzar con `pythonw.exe`, que no tiene consola: un
fallo de importación allí sería un fallo mudo y la ventana simplemente no abriría.

## Arranque rápido (multiplataforma)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"      # Linux/macOS: .venv/bin/python

export MODEL_DRIFT_BROKER_TOKEN=...                  # el Broker rota el suyo en cada arranque
model-drift modelos                                  # ¿qué hay en el catálogo?

# Línea base con el modelo actual
model-drift ejecutar suites/ejemplo \
    --modelo ollama/local/qwen3:6b \
    --relaciones relaciones \
    --etiqueta "antes de actualizar" \
    --acepto-contaminar

# La versión nueva
model-drift ejecutar suites/ejemplo \
    --modelo ollama/local/qwen3:8b \
    --relaciones relaciones \
    --acepto-contaminar

model-drift tiradas
model-drift comparar run-XXXX run-YYYY \
    --suite suites/ejemplo \
    --politica politica-ejemplo.yaml \
    --informe informes/qwen36-vs-qwen38.html

# R3/R4 ya calculados por Local AI Lab; no se accede a su base de datos
model-drift evaluar-tratamientos \
    --plan model-drift/plan.json \
    --informe informes/r3-r4.html
```

`comparar` devuelve código de salida **3** si la política declara la comparación como bloqueada, así que puede cortar una entrega desde CI. (No es 2: ese lo usa argparse para los errores de uso, y un guion no debe confundir un typo con una regresión.)

`evaluar-tratamientos` implementa el contrato público v2 de Local AI Lab para benchmarks de
retrieval y ejecuciones completas RAG. Verifica el SHA-256 del plan y los dos ZIP, además de
suite, snapshot, profundidad `k`, estrategia, modelo servido, ausencia de fallback y casos
antes de medir. Los artefactos se leen como datos, nunca se extraen y no se consulta la base
de datos de la aplicación que los produjo.

### Medir sin degradar lo que se mide

`load_model_stats` del Broker agrega desde `model_invocations`, y `model_quarantine` bebe de la
misma tabla. Sin más, una batería de prompts duros le baja el score de fiabilidad al router
adaptativo y puede llegar a apartar el modelo del catálogo: **medir un modelo lo degradaría en
producción**.

El Broker resolvió esto en el contrato 2.9 con `exclude_from_model_learning` (Cambio 4), y
Model_Drift lo envía en cada invocación. Contra un Broker que aún no lo tenga, `ejecutar`
**se niega a arrancar** sin `--acepto-contaminar`: no se supone la bandera, se comprueba leyendo
el esquema publicado del Broker antes de enviar nada.

## Qué mide

| Familia | Qué caza | Necesita |
|---|---|---|
| **A. Oráculo determinista** | Esquema JSON, regex, campos, tolerancias, llamadas a herramienta, SQL por ejecución, cotejo de citas | Nada más |
| **B. Referencia difusa** | Similitud semántica contra respuesta dorada | Modelo de embeddings **y umbral calibrado** |
| **C. Juez LLM** | Lo que no tiene oráculo | Juez congelado y calibrado |
| **D. Metamórficas** | Paráfrasis, orden, contexto irrelevante, mutación, adulación, anclaje | Relación formal declarada |
| **E. Instrucciones** | Adherencia al system prompt, longitud, rechazos | Aserciones |
| **F. Contexto y fuentes** | Abstención, citas alucinadas | Contexto en el caso |
| **G. Agéntico** | Trayectoria, recuperación de errores | Servidor de herramientas |
| **H. No funcional** | Latencia, coste, **deriva de verbosidad** | Nada: se mide siempre |

## Las cinco decisiones que lo hacen defendible

### 1. Dos niveles, no uno

```
change_status      sin_cambio_observable | cambio_significativo | fallo_nuevo | indeterminado
change_direction   mejora | regresion | cambio_neutro | sin_direccion
```

Son **columnas distintas**, y `ComparisonOutcome` rechaza en el constructor cualquier dirección sin `evidence_kind`. No es una convención que se pueda olvidar: es un error de construcción.

Consecuencia práctica: un caso capturado del tráfico real cuya respuesta histórica era *incorrecta* y ahora se responde bien produce `cambio_significativo` + `sin_direccion`, nunca `regresion`. Una salida histórica sin verdad conocida es línea base de cambio, jamás verdad de referencia.

### 2. La unidad estadística es el caso

50 casos × 10 repeticiones son **50 observaciones**, no 500. Las repeticiones están correlacionadas por construcción y contarlas como muestras infla la potencia y fabrica significación.

- Dentro del caso: las repeticiones producen `case_score` y `case_stability`, y admiten un contraste propio (Fisher exacto para binarias, permutación para continuas).
- Entre casos: McNemar exacto sobre casos pareados y **bootstrap por conglomerados remuestreando casos**, más corrección de Benjamini-Hochberg.

Todo implementado sin scipy: son cuatro procedimientos cortos y el trabajo gana en auditabilidad si se ven.

### 3. Significación no es relevancia

Con suficientes casos, una diferencia de 0,3 pp sale significativa y no le importa a nadie. Cada métrica declara su efecto mínimo de interés, y el cruce produce cuatro resultados con nombre propio:

| | Efecto < SESOI | Efecto ≥ SESOI |
|---|---|---|
| **No significativo** | `no_detectado` | `relevante_no_concluyente` |
| **Significativo** | `detectado_irrelevante` | `detectado_relevante` |

El veredicto operativo sale de una `regression_policy` declarada, con métricas bloqueantes y de aviso. **No hay score global**: cuando unas métricas mejoran y otras empeoran, el resultado es `trade_off_detected` con el desglose, y la decisión es de quien escribió la política.

### 4. El evaluador forma parte del instrumento

Cada tirada guarda un manifiesto con versiones y hashes de todo: suite, fixtures, esquemas, transformaciones metamórficas, rúbricas, modelo de embeddings, juez y configuración efectiva. Antes de comparar dos tiradas se cotejan los manifiestos, y si algo del instrumento cambió, los casos salen `indeterminado` y el informe lo dice **en cabecera**.

Un umbral de similitud sin procedencia de calibración es rechazado al cargar la suite, no a las tres horas de tirada.

**Comparar dos modelos distintos** (p. ej. `qwen3.6:35b` frente a `qwen3.8:27b`): lo que viaja con los pesos —modelo, digest, familia, tamaño, cuantización, tokenizer, plantilla de chat, ventana de contexto y la identidad remota— es la variable del experimento y no cuenta para la comparabilidad. Lo que sí tiene que coincidir es el instrumento: runtime, su versión, parámetros pedidos y aplicados, suite, política e instrumentos de evaluación. Con el **mismo** modelo no se excluye nada: un tokenizer o una plantilla distintos son justo el «mismo `model_id`, distinta ejecución» que se busca. Una tirada de varios modelos se compara modelo a modelo: la ventana ofrece una entrada por modelo y la CLI exige `--modelo-base`/`--modelo-candidato`.

**Tiradas que no midieron nada.** El catálogo del Broker (`/capabilities`, `/models/availability`) responde sin token, así que «Consultar catálogo» funciona aunque el token haya caducado; solo las tareas lo exigen. Si el Broker rechaza la credencial, la tirada se para en la primera invocación y queda `failed` con el motivo (`AUTH: ADMIN_AUTH_REQUIRED`); si ninguna invocación produce salida por cualquier otro motivo, también es `failed`. En los dos casos la CLI sale con código 1, la ventana avisa y esa tirada no se ofrece para comparar.

### 5. Nunca se afirma una garantía que no se tiene

Fijar una semilla es una condición de control, no una garantía de determinismo. Por eso hay cuatro banderas independientes, no una:

```
seed_fijada                sí | no
determinismo_solicitado    sí | no
determinismo_observado     sí | no | no_evaluado        ← se MIDE comparando repeticiones
reproducibilidad_exacta    demostrada | no_demostrada | no_disponible
```

Lo mismo con la identidad del modelo: la huella de ejecución tiene tres capas —declarada, técnica observable, remota— y cada componente lleva su estado `verificado | declarado | no_disponible`. Nunca se rellena un hueco por inferencia.

Y lo mismo con el juez: `temperature = 0` reduce la variabilidad, no demuestra determinismo, así que `judge_self_consistency` se mide repitiendo una submuestra con entrada idéntica.

## Sobre el juez

Recibe `A` y `B` y nada más: sin nombres de modelo, sin saber cuál es la versión nueva, con el orden aleatorizado y **ejecutando siempre los dos órdenes**. Se miden tres cosas distintas que suelen confundirse:

- `position_bias` — veredictos que se invierten al cambiar el orden.
- `judge_self_consistency` — estabilidad ante la misma entrada.
- `judge_human_agreement` — kappa contra 50–100 comparaciones etiquetadas a mano, más `human_human_agreement` sobre una submuestra etiquetada por una segunda persona. Sin ese último dato, una concordancia de 0,55 no se puede interpretar: puede ser un juez mediocre o un conjunto de casos donde dos personas tampoco se ponen de acuerdo.

**Un juez que no supera los mínimos declarados no produce evidencia de nivel 2.** Sus veredictos se registran, pero los casos se quedan en `sin_direccion`.

## Estado actual

Implementadas las fases 0–4 y 6 del plan: motor de ejecución, oráculos, metamórficas,
estadística, política, juez e informes. La fase 5 (evaluación agéntica completa) y la fase 7
(clasificación de deriva no explicada) quedan pendientes.

**Lo que hoy NO se tiene, y el manifiesto lo declara en cada tirada:**

- Sin `generation.seed` en el Broker → la tirada **no es reproducible** y `reproducibilidad_exacta` vale `no_disponible`.
- Sin identidad técnica del modelo → la huella es solo declarada: no se puede afirmar que dos tiradas usaron el mismo binario ni la misma plantilla de chat.
- Sin marcador de exclusión → la tirada altera las estadísticas del router del Broker.

El cliente 2.9 ya consume semilla, telemetría, identidad tipada y exclusión del aprendizaje
cuando el despliegue las anuncia. La ausencia de una garantía sigue registrándose como
ausencia; implementar el campo no demuestra por sí solo determinismo o identidad remota.

## AI OS Core como director (0.4.0)

Opcional: sin `core_url` la app funciona igual que antes. Con `core_url` en `model_drift.yaml`
(o `MODEL_DRIFT_CORE_URL`) y el token por app en `MODEL_DRIFT_CORE_TOKEN` (nunca en disco):

- cada tirada evalúa antes la política `model_drift:broker` y solo arranca con un ALLOW
  ejecutable; con DENY, aprobación pendiente, INDETERMINATE o Core caído **no se envía nada**;
- deja en Core el recibo de la tirada (`accepted` → `succeeded`/`failed_terminal`/`cancelled`);
- publica el veredicto de cada modelo sobre la suite como certificación
  (`model_id` = `proveedor/despliegue/modelo`, `capability_id` = `suite:<id>`, revisión = nº de
  tiradas): media de pass_rate ≥ 0,8 con ≥ 3 casos aplicables → `CERTIFIED`; < 0,5 → `REJECTED`;
  si no, `INDETERMINATE`. Con estas certificaciones Core decide qué modelos usan las demás apps.

Lo que Core no llega a apuntar (un recibo, una certificación) sale como AVISO y no invalida la tirada.
Recorrido real: `scripts/verify_ai_os_core.py <core> <token_model_drift> <token_director> <carpeta>`
(12 comprobaciones, verde el 18-oct-2026 contra el Broker real con `ollama/local/lfm2:24b`).

**0.4.1 — monitor de Core.** Cada tirada aparece en `<core>/monitor`: la autorización y los recibos
llevan `flow_id` = id de la tirada, y la tirada informa de un salto Model_Drift → Broker al empezar
y otro al terminar (respuesta o error, con invocaciones y duración). Uno por tirada, no por caso,
para no inundar el monitor. Si el monitor no contesta, la tirada no se entera.

**0.4.6 — Broker 2.12.**
- **Semilla:** la versión del contrato se comparaba como texto, y `"2.12" >= "2.9"` da falso. Contra un Broker
  que no anunciara `execution_fingerprint`, se apagaba la semilla y la tirada se declaraba no
  reproducible sin motivo. Ahora se compara por números.
- **Embeddings:** la petición de embedding no llevaba `output`, y el contrato exige `format: json`. Desde el 2.12
  ya no hace falta `json_schema`.

## Desarrollo

System 1 puede priorizar revisiones semánticas mediante el Broker, inicialmente
desactivado y en sombra. El modo **Probar System 1**, disponible en el escritorio y
en la CLI, calibra por caso de uso con un maestro elegido por el usuario y una prueba
separada. Consulta [integración](docs/SYSTEM1.md) y [calibración](docs/SYSTEM1_CALIBRATION.md).

```bash
.venv/Scripts/python -m pytest tests/ -q
.venv/Scripts/python -m pytest tests/ --cov=model_drift --cov-report=term-missing
.venv/Scripts/python -m ruff check .
.venv/Scripts/python -m mypy
```

Los tres verdes es el estado normal, igual que en el Broker y en ChatyGPT. Comentarios, documentación y nombres de dominio en español; nombres de campos y términos técnicos estándar en inglés.

Las pruebas no necesitan un Broker vivo: `tests/broker_fake.py` monta un Broker real detrás de `httpx.MockTransport`, con catálogo, respuestas programadas y el mismo 422 ante `generation.seed` que da el contrato actual. Eso ejercita el cliente de verdad —construcción del cuerpo, sondeo, degradación, detección de fallback— en vez de un doble.

## Estructura

```
model_drift/
  domain.py         vocabulario: cambio, deriva, dirección, evidencia
  statistics.py     unidad estadística, McNemar, bootstrap, BH, potencia
  comparison.py     motor de comparación: los dos niveles
  policy.py         SESOI y regression_policy, sin score global
  fingerprint.py    huella de ejecución en tres capas
  manifest.py       manifiesto y regla de comparabilidad
  oracles.py        familia A: aserciones deterministas
  metamorphic.py    familia D: relaciones formales
  judge.py          familia C: juez anonimizado con su error conocido
  embeddings.py     instrumento de similitud y calibración de umbrales
  runner.py         motor de ejecución de una tirada
  report.py         informes HTML y de terminal
  gui.py            la ventana de escritorio (Tkinter)
  cli.py            la línea de comandos
  broker_client.py  cliente del Broker
  broker_contracts.py  espejo del contrato — actualizar con cada cambio del Broker
suites/ejemplo/     suite de demostración, ejecutada por las pruebas
relaciones/         relaciones metamórficas de ejemplo
```

> `broker_contracts.py` espeja el contrato del AI Broker. **Cada cambio del contrato del Broker exige actualizarlo en el mismo cambio**, igual que en el Knowledge_Orchestrator.
