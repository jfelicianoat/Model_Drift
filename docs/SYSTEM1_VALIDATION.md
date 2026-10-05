# Validación local de System 1 y calibración

Fecha: 3 de octubre de 2026.

## Documentos contrastados

- `docs/Client_API.md`: contrato 2.11, ampliación del 3 de octubre, §15.1–15.8.
- `docs/Prompts_System1_TFM_LAYA_Ollama/07_Model_Drift_SYSTEM1.md`.
- `docs/Prompts_System1_TFM_LAYA_Ollama/09_ADENDA_Model_Drift_Calibracion_System1.md`.

Las rutas anteriores a estos documentos se resuelven desde la carpeta común
`Proyectos TFM`. Los documentos nuevos de integración pertenecen a Model_Drift.

## Resultado medido

| Comprobación | Resultado |
| --- | --- |
| Pytest completo, incluidos escritorio y arranque en subproceso | 312 pruebas aprobadas |
| Cobertura total | 92,57 %, supera el mínimo 92 % |
| Ruff sobre el proyecto | Sin errores |
| Mypy | Sin errores en 80 archivos de código |
| Consistencia del diff | Sin errores de espacios |
| Plan de demostración | JSON válido, cuatro casos, cero verificaciones humanas |
| Conjunto semántico sintético | 80 casos, 20 por etiqueta, siete adversarios, cero verificaciones humanas |

Las pruebas incluyen equivalencia, cambio benigno, candidato de regresión,
incertidumbre, checks antes de IA, desactivación, sombra, Broker caído y rechazo.
La calibración prueba maestro ambiguo/inadecuado, notas bajas, tipos y alternativas
inválidas, alias e identidad cambiante, privacidad, calentamiento y reintentos,
Clopper–Pearson, particiones separadas, exportación, caducidad y revalidación.
El control negativo local repite el mismo plan y no informa regresión.
Las referencias sintéticas también se comprueban: se rechaza una procedencia
ausente o incompatible, se conservan notas nativas rechazadas en el diagnóstico
y nunca se convierten sus etiquetas en verificación humana o umbral certificado.

Los datos de las pruebas son sintéticos. Su kappa, aciertos o umbrales no son
mediciones de Qwen, Nimble, Laya ni de ningún proveedor real.

En esta sesión restringida se ejecutó Pytest con las bibliotecas Tcl/Tk del Python
instalado copiadas a `.test-tmp/python-tcl`, usando variables de entorno temporales.
Eso permitió ejecutar las pruebas del escritorio sin cambiar la instalación de
Python ni la aplicación. Se usó además una carpeta temporal dentro del proyecto.

## Cobertura del cambio

| Componente | Responsabilidad añadida |
| --- | --- |
| `broker_client/semantica.py`, `system1_contracts.py` | Juicios síncronos y auditoría 2.11 |
| `system1.py`, `comparison/service.py`, configuración | Prioridad de revisión después de checks, desactivación y sombra |
| `calibration/models.py`, `teacher.py`, `validation.py` | Plan, maestro y validación de notas nativas |
| `calibration/analysis.py`, `statistics/calibration.py` | Calidad, fiabilidad, umbral y cota exacta |
| `calibration/runner.py`, `db.py` | Ejecución, persistencia, exportación, comparación y caducidad |
| `cli.py`, `gui/calibration.py`, pestaña y catálogo | Prueba integrada y revalidación al refrescar el catálogo |
| `report.py`, `system1_benchmark.py` | Informes y referencias humanas/destiladas/sintéticas diferenciadas |
| `tests/test_system1.py`, `test_calibration.py`, `test_gui.py` | Contrato, comportamiento, medición y escritorio |

El esquema SQLite es la versión 4 y su migración es aditiva. System 1 permanece
desactivado por defecto, en sombra al activarlo. Ninguna prueba cambia umbrales del
Broker ni políticas de Agora.

## Auditoría de aceptación

| Requisito | Evidencia actual | Estado |
| --- | --- | --- |
| Checks antes del filtro, cuatro etiquetas, fallback y modo sombra | `system1.py` y pruebas de contrato/comparación en `test_system1.py` | Comprobado localmente |
| Probar desde escritorio y CLI con el mismo motor | `gui/calibration.py`, `cmd_probar_system1` y sus pruebas | Comprobado localmente |
| Maestro fijado, repeticiones, calidad humana, privacidad y sin fuga de etiquetas | `teacher.py`, `qualify_teacher` y pruebas de maestro/privacidad | Comprobado con respuestas simuladas |
| Validar notas crudas, alternativas, procedencia e identidad del alumno | `validation.py` y pruebas de notas inválidas, rechazo y alias | Comprobado con respuestas simuladas |
| Partición fija, cota exacta, prueba separada y ausencia de recomendación sin evidencia | `analysis.py`, `statistics/calibration.py` y pruebas de selección del umbral | Comprobado localmente |
| Informes, exportación, caducidad y repetición al cambiar la huella | SQLite, informes y pruebas de persistencia/revalidación | Comprobado con respuestas simuladas |
| Corpus inicial y referencias sintéticas explícitas | 80 casos validados y pruebas de procedencia y diagnóstico | Preparado; no verificado por humanos |
| Cotejar el simulado con respuestas reales de §15.8, incluido `score_source` | Sondeo real de capabilities/OpenAPI falla; TCP devuelve Windows 10013 | Pendiente por acceso de red denegado |
| Tirada real con Qwen3.8 y control negativo con el mismo modelo | No hay una ejecución real ni referencia de catálogo confirmada | Pendiente del Broker |

La comprobación más reciente reproduce la denegación de red. No hay un proceso de
calibración en marcha al que esperar ni código local del Broker con el que sustituir
el servicio real. La aceptación completa permanece sin demostrar hasta disponer de
acceso a ese servicio; las pruebas locales no se presentan como prueba de extremo a
extremo contra él.

## Verificación pendiente y observación del contrato

El sondeo de `/api/v1/capabilities` y `/openapi.json` en el Broker configurado
`http://192.168.1.52:8765` devuelve `ConnectError` desde esta sesión. El diagnóstico
TCP precisa `PermissionError`, código Windows 10013: acceso de red denegado,
sin evidencia de que el servicio esté caído. No se ha podido
cotejar el simulado con el servicio vivo ni ejecutar allí el control negativo.
La aceptación completa de la adenda sigue pendiente de esa comprobación.

El texto de §15.8 declara `score_source` en cada intento, pero sus ejemplos JSON
y el de §15.3 no lo incluyen. El validador exige `native` para el alumno y excluye
notas sin esa procedencia. Debe confirmarse su presencia en la respuesta real;
no se deduce que un número sea una probabilidad nativa.

Qwen3.8 es la preferencia del usuario para el maestro. Falta confirmar su referencia
completa en el catálogo. El plan de demostración tiene un marcador y necesita esa
elección antes de ejecutarse. Para certificar producción hacen falta además casos
representativos y etiquetas humanas verificadas; una prueba de recorrido con cuatro
casos sólo puede resultar indeterminada. El conjunto de 80 casos inventados
permite explorar los errores frente a referencias de procedencia explícitamente
sintética; no sustituye esa certificación.
