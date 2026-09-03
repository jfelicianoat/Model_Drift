"""Estadística de la comparación. La unidad de análisis es **el caso**.

Este módulo existe para impedir el error que hunde la mayoría de evaluaciones
caseras de LLMs: tratar las N repeticiones de un caso como N observaciones
independientes. No lo son. Están correlacionadas por construcción —mismo
prompt, mismo modelo, mismo contexto— y contarlas como muestras infla la
potencia artificialmente y produce significación donde no la hay. Con 50 casos
× 10 repeticiones hay **50** observaciones, no 500.

El tratamiento va en dos pisos (plan §9.1), y el paquete está partido igual:

- **Piso 1, dentro del caso** (`agregacion`): las repeticiones estiman la
  *distribución* del caso. De ellas salen `case_score` y `case_stability`. No
  aportan tamaño muestral.
- **Piso 2, entre casos** (`contrastes`): el contraste se hace sobre casos
  pareados, con `case_score` como observación. McNemar para binarias; bootstrap
  **por conglomerados remuestreando casos** para continuas.

Encima van `correccion` (Benjamini-Hochberg), `relevancia` (significativo no es
importante) y `potencia` (qué habría podido detectarse).

Todo está implementado sin scipy ni statsmodels, a propósito: son cuatro
procedimientos cortos, el TFM gana en auditabilidad si se ven, y el paquete
instala en cualquier Python sin ruedas binarias.

Extensión futura consciente: modelos jerárquicos / mixed-effects para modelar
explícitamente la varianza intra e inter caso. **No es requisito del MVP** y no
debe retrasar las fases 1-2.
"""
from __future__ import annotations

from model_drift.statistics.agregacion import CaseAggregate, aggregate_binary, aggregate_continuous, binarize
from model_drift.statistics.contrastes import (
    BootstrapResult,
    McNemarResult,
    cluster_bootstrap,
    fisher_exact_2x2,
    mcnemar_exact,
    minimum_attainable_p,
    permutation_test,
)
from model_drift.statistics.correccion import benjamini_hochberg
from model_drift.statistics.potencia import (
    PowerEstimate,
    estimate_power,
    minimum_detectable_effect,
    power_curve,
)
from model_drift.statistics.relevancia import classify_practical

__all__ = [
    "BootstrapResult",
    "CaseAggregate",
    "McNemarResult",
    "PowerEstimate",
    "aggregate_binary",
    "aggregate_continuous",
    "benjamini_hochberg",
    "binarize",
    "classify_practical",
    "cluster_bootstrap",
    "estimate_power",
    "fisher_exact_2x2",
    "mcnemar_exact",
    "minimum_attainable_p",
    "minimum_detectable_effect",
    "permutation_test",
    "power_curve",
]
