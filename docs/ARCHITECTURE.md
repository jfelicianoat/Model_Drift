# Arquitectura vigente de Model_Drift

Revisión contra el código: **23 de agosto de 2026**.

## Propósito y límite

Model_Drift compara el comportamiento de los prompts de un producto entre dos ejecuciones.
No es un router, un benchmark general ni un sistema de monitorización de producción. Su
unidad estadística es el caso de prueba, y separa obligatoriamente el cambio observable de
la dirección evaluativa del cambio.

## Flujo

```text
suite + fixtures + política
          │
          ▼
runner ──► cliente AI Broker 2.9 ──► modelos seleccionados
  │                    │
  │                    └─ telemetría por invocación y huella observable
  ▼
resultados por repetición
  ▼
agregación por caso ──► comparación pareada ──► política ──► HTML / salida CI
```

La entrada para tratamientos ya calculados comparte el motor, pero no el ejecutor:

```text
plan v2 sellado ──► ZIP R3/R4 inmutables ──► importador externo
                                              │
                                              ▼
                                comparación pareada ──► HTML / salida CI
```

El importador acepta informes deterministas de retrieval y ejecuciones completas RAG de
Local AI Lab, valida sus límites e identidades y los convierte a `RunResults`. No conoce su
SQLite, su almacén CAS ni su implementación interna; las rutas y huellas llegan explícitamente
en el plan.

## Capas

| Capa | Responsabilidad | No debe hacer |
| --- | --- | --- |
| Dominio | estados de cambio, dirección, evidencia y manifiestos | llamar a red o decidir políticas implícitas |
| Ejecución | repeticiones, cancelación, persistencia y reanudación | contar repeticiones como casos independientes |
| Evaluadores | oráculos, embeddings, juez y metamórficas | afirmar regresión sin evidencia calibrada |
| Estadística | McNemar, bootstrap por casos, BH, SESOI y potencia | producir un score global que oculte trade-offs |
| Broker | fijar modelo, enviar tareas y recoger telemetría | permitir fallback silencioso en una comparación |
| Presentación | CLI, Tkinter e informe HTML | recalcular o reinterpretar el veredicto |

## Persistencia y reproducibilidad

Cada tirada conserva suite, configuración efectiva, hashes del instrumento, casos,
repeticiones, resultados, telemetría y huella. Dos tiradas solo son comparables cuando sus
manifiestos lo permiten. Semilla solicitada, determinismo observado, identidad técnica y
reproducibilidad exacta son hechos separados.

## Integración con AI Broker

La compatibilidad está documentada en [BROKER_COMPATIBILITY.md](BROKER_COMPATIBILITY.md).
El cliente exige selección exacta (`fallback_allowed: false`) y marca el tráfico con
`exclude_from_model_learning` cuando el Broker lo soporta. Sin esa capacidad la ejecución
se bloquea salvo consentimiento explícito para contaminar las estadísticas de aprendizaje.

## Estado funcional

Implementado: ejecución, oráculos deterministas, referencias difusas, metamórficas,
estadística, política, juez LLM, informes, CLI y escritorio. Pendiente: evaluación agéntica
completa y clasificación automática de deriva no explicada. Esas ausencias no impiden usar
las familias ya implementadas y aparecen como límites, no como resultados simulados.

## Verificación

```powershell
python -m pytest tests -q
python -m ruff check .
python -m mypy
```
