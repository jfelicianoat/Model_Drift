"""Motor de ejecución de una tirada.

Ejecuta una suite contra uno o varios modelos, N repeticiones por caso, y deja
en la base de datos las salidas crudas, la telemetría, el manifiesto y la
agregación de casos.

Tres reglas que el motor impone y que no son negociables:

1. **Capacidad ausente = `not_applicable`.** Un modelo que no declara `tools`
   no suspende las pruebas de herramientas: queda fuera de esa métrica. Contar
   ausencia de capacidad como fallo convertiría el ranking en una mentira.
   Vive en `base.py`.
2. **La unidad es el caso.** Las N repeticiones se agregan en `ejecucion.py`,
   en `case_result`. Lo que sale del motor hacia la comparación es una
   observación por caso, no N.
3. **El manifiesto declara lo que faltó.** Si el Broker no admite semilla, si
   no expone identidad técnica, si no permite excluir el tráfico del
   aprendizaje: se escribe. Una ausencia declarada es información; una ausencia
   silenciada es un engaño. Vive en `manifiesto.py`.

Este módulo es la fachada: `Runner` y sus tipos entran por aquí, y el reparto
interno (base → invocación → manifiesto → ejecución) puede cambiar sin tocar a
quien lo usa.
"""
from __future__ import annotations

from model_drift.domain import (
    METRIC_COST,
    METRIC_LATENCY,
    METRIC_OUTPUT_MATCH,
    METRIC_PASS,
    METRIC_TOKENS,
)
from model_drift.runner.ejecucion import Runner
from model_drift.runner.tipos import BudgetExceeded, RunConfig, RunOutput, RunProgress

#: Reexportadas por comodidad: su definición vive en `domain` para que el
#: motor de comparación sepa cuáles son binarias sin depender del runner.
__all__ = [
    "METRIC_COST",
    "METRIC_LATENCY",
    "METRIC_OUTPUT_MATCH",
    "METRIC_PASS",
    "METRIC_TOKENS",
    "BudgetExceeded",
    "RunConfig",
    "RunOutput",
    "RunProgress",
    "Runner",
]
