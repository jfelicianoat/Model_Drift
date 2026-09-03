"""Motor de comparación: de dos tiradas a un veredicto honesto.

El orden de las operaciones no es negociable y está tomado del plan §5, §9 y §11:

1. **Comparabilidad primero.** Si el instrumento de evaluación cambió, no hay
   nada que concluir sobre el modelo: los casos salen `indeterminado` y el
   informe lo dice en cabecera.
2. **Nivel 1, cambio.** Se detecta con las repeticiones de cada caso (test
   exacto de Fisher dentro del caso) y se corrige por comparaciones múltiples.
   No juzga calidad.
3. **Nivel 2, dirección.** Solo si el caso aporta evidencia evaluativa. Sin
   ella, `sin_direccion`, y el informe tiene prohibido usar las palabras
   «mejora» y «regresión» para ese caso.
4. **Agregación entre casos.** McNemar sobre casos pareados y bootstrap por
   conglomerados: la unidad es el caso, nunca la repetición.
5. **Relevancia práctica y política.** Significativo no es lo mismo que
   importante.

El paquete sigue ese mismo orden: `evidencia` decide si hay nivel 2,
`contraste` resuelve los niveles 1 y 2 de cada caso, `resultados` define lo que
entra y lo que sale, y `comparador` recorre las métricas. Quien usa el motor
importa de aquí.
"""
from __future__ import annotations

from model_drift.comparison.comparador import compare_runs
from model_drift.comparison.evidencia import OPERATIONAL_METRICS, CaseEvidence, evidence_for_case
from model_drift.comparison.resultados import ComparisonReport, RunResults

__all__ = [
    "OPERATIONAL_METRICS",
    "CaseEvidence",
    "ComparisonReport",
    "RunResults",
    "compare_runs",
    "evidence_for_case",
]
