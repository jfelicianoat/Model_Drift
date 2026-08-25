"""Vocabulario del dominio: cambio, deriva y dirección.

Esta es la sección §5 del plan hecha código, y manda sobre el resto del
paquete. Tres conceptos que se confunden constantemente y que aquí son tipos
distintos:

- **Cambio** — la salida observable difiere. Hecho medible, sin juicio.
- **Deriva** — cambio sistemático y sostenido entre configuraciones. Sin juicio.
- **Regresión / mejora** — *dirección* respecto a un criterio de calidad.
  Requiere evidencia evaluativa, que la deriva por sí sola no aporta.

La regla que cierra el caso más peligroso, el de la línea base incorrecta:

    Una salida histórica sin verdad conocida es línea base para detectar
    cambio, jamás verdad de referencia.

Por eso `change_status` y `change_direction` son campos separados y
`change_direction` solo puede alejarse de `sin_direccion` si viene acompañado
de un `evidence_kind` válido. `ComparisonOutcome` lo impone en un validador:
no es una convención que se pueda olvidar, es un error de construcción.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator

#: Métricas que el motor produce siempre. Viven aquí, y no en `runner`, porque
#: el motor de comparación necesita saber cuáles son binarias sin importar el
#: módulo que las generó.
METRIC_PASS = "pass_rate"
METRIC_LATENCY = "latency_ms"
METRIC_COST = "cost_usd"
METRIC_TOKENS = "output_tokens"

#: Coincidencia con la salida histórica congelada de un caso capturado.
#:
#: Es la métrica que hace útil un caso sin aserciones: detecta que la respuesta
#: **cambió**, que es todo lo que una línea base sin verdad conocida puede
#: sostener. Que ese cambio sea a mejor o a peor no lo decide esta métrica —no
#: aporta evidencia evaluativa— y por eso un caso capturado sin nada más se
#: queda en `sin_direccion`.
METRIC_OUTPUT_MATCH = "output_match"

#: Prefijo de las métricas que produce una relación metamórfica.
METRIC_METAMORPHIC_PREFIX = "mr:"


#: Métricas donde MENOS es mejor. Sin esto, una política que no las declare
#: hereda el valor por defecto `higher_is_better=True` y una latencia que baja
#: se informa como regresión. No es un detalle cosmético: invierte el signo de
#: la conclusión sobre latencia, coste y verbosidad.
LOWER_IS_BETTER_METRICS = frozenset(
    {
        METRIC_LATENCY,
        METRIC_COST,
        METRIC_TOKENS,
        "latency_p95",
        "tokens_output",
        "redundancy",
    }
)

# Métricas de retrieval cuyo criterio está definido por el propio benchmark.
# No son opiniones sobre el contenido generado: miden contra un ground truth
# congelado y por eso pueden aportar dirección sin un juez externo.
RETRIEVAL_METRICS = frozenset(
    {
        "recall_at_k",
        "precision_at_k",
        "hit_rate",
        "reciprocal_rank",
        "ndcg_at_k",
        "source_coverage",
        "redundancy",
    }
)

EXTERNAL_DEFINED_METRICS = RETRIEVAL_METRICS | {"deterministic_pass"}


def default_higher_is_better(metric: str) -> bool:
    """Dirección natural de una métrica que la política no declara.

    Una tasa de acierto mejora subiendo; una latencia, bajando. Adivinarlo por
    el nombre sería frágil, así que las conocidas van en una lista explícita y
    el resto asume que más es mejor, que es lo que vale para las proporciones.
    """
    return metric not in LOWER_IS_BETTER_METRICS


def is_binary_metric(metric: str) -> bool:
    """Si la métrica es una proporción de aciertos y no una magnitud.

    Determina qué contraste se aplica dentro del caso: exacto de Fisher sobre
    aciertos y fallos para las binarias, permutación sobre las repeticiones
    para las continuas. Aplicar el primero a una latencia produce cuentas sin
    sentido, así que la distinción no es cosmética.
    """
    return metric in {METRIC_PASS, METRIC_OUTPUT_MATCH, "deterministic_pass"} or metric.startswith(
        METRIC_METAMORPHIC_PREFIX
    )


class ChangeStatus(str, Enum):
    """Nivel 1 — detección de cambio. Siempre computable, no juzga calidad."""

    #: La diferencia no supera el umbral de detección con la potencia disponible.
    sin_cambio_observable = "sin_cambio_observable"
    #: Diferencia sostenida entre configuraciones, corregida por comparaciones
    #: múltiples.
    cambio_significativo = "cambio_significativo"
    #: La ejecución dejó de producir salida evaluable: error, timeout,
    #: truncamiento, salida vacía o degenerada, precondición incumplida.
    #: Es un fallo de EJECUCIÓN, no de calidad: un oráculo que deja de pasar es
    #: `cambio_significativo` + `regresion`, no esto.
    fallo_nuevo = "fallo_nuevo"
    #: No hay potencia suficiente, el caso es inestable, o el instrumento de
    #: evaluación cambió entre las dos tiradas.
    indeterminado = "indeterminado"


class ChangeDirection(str, Enum):
    """Nivel 2 — dirección del cambio. Solo con evidencia evaluativa."""

    mejora = "mejora"
    regresion = "regresion"
    #: Hay cambio y hay evidencia, pero la evidencia no distingue calidad.
    cambio_neutro = "cambio_neutro"
    #: No hay evidencia evaluativa aplicable. Valor por defecto y única salida
    #: honesta para un caso capturado sin oráculo ni juez calibrado.
    sin_direccion = "sin_direccion"


class EvidenceKind(str, Enum):
    """Qué cuenta como evidencia evaluativa para asignar dirección."""

    oraculo_determinista = "oraculo_determinista"
    relacion_metamorfica = "relacion_metamorfica"
    juez_calibrado = "juez_calibrado"
    evaluacion_humana = "evaluacion_humana"
    metrica_definida = "metrica_definida"


class PracticalSignificance(str, Enum):
    """Cruce de significación estadística y relevancia práctica (plan §13.1)."""

    #: Ni significativo ni relevante.
    no_detectado = "no_detectado"
    #: Significativo pero por debajo del efecto mínimo de interés. Resultado
    #: legítimo y frecuente: se informa con este nombre en vez de alarmar.
    detectado_irrelevante = "detectado_irrelevante"
    #: Supera el umbral práctico pero no alcanza significación: haría falta
    #: más potencia.
    relevante_no_concluyente = "relevante_no_concluyente"
    #: Significativo y por encima del umbral práctico.
    detectado_relevante = "detectado_relevante"


class Comparability(str, Enum):
    """Si dos tiradas son comparables. Se evalúa ANTES que ninguna conclusión."""

    ok = "ok"
    #: Algún instrumento de evaluación cambió entre las dos tiradas.
    instrumento_cambiado = "instrumento_cambiado"
    #: Las huellas de ejecución difieren en componentes que no son la variable
    #: declarada del experimento.
    huella_divergente = "huella_divergente"


class RunVerdict(str, Enum):
    """Veredicto operativo de una comparación, derivado solo de la política."""

    pass_ = "pass"
    warning = "warning"
    blocked = "blocked"
    #: Unas métricas mejoran y otras empeoran de forma relevante. No se resuelve
    #: solo: la decisión es de quien escribió la política.
    trade_off_detected = "trade_off_detected"
    indeterminado = "indeterminado"


class CaseOutcome(BaseModel):
    """Resultado agregado de un caso en una tirada: el piso 1 de la §9.1.

    Las N repeticiones **no** son N observaciones: producen `case_score` y
    `case_stability`, y es el caso —no la repetición— la unidad del contraste.
    """

    case_id: str
    model_key: str
    metric: str = "pass_rate"
    #: Tasa de acierto sobre las N repeticiones (binarias) o media/mediana
    #: (continuas).
    case_score: float | None = None
    #: Acuerdo entre repeticiones: 1.0 = todas iguales.
    case_stability: float | None = None
    repetitions: int = 0
    #: Repeticiones que no produjeron salida evaluable.
    failed_repetitions: int = 0
    is_unstable: bool = False
    is_not_applicable: bool = False
    not_applicable_reason: str | None = None
    values: list[float] = Field(default_factory=list)
    #: Una medida determinista no necesita repeticiones para demostrar que el
    #: valor observado cambió. La inferencia a otros casos sigue dependiendo
    #: del tamaño de la suite; esta bandera solo describe el caso medido.
    measurement_deterministic: bool = False

    @property
    def is_usable(self) -> bool:
        """Un caso entra en los contrastes si aplica, es estable y tiene datos."""
        return not self.is_not_applicable and not self.is_unstable and self.case_score is not None


class ComparisonOutcome(BaseModel):
    """Comparación de un caso y una métrica entre dos tiradas.

    Los dos niveles viajan en campos distintos por diseño: una consulta no
    puede tropezarse con una regresión que nadie ha demostrado.
    """

    case_id: str
    metric: str
    # --- Nivel 1: cambio ---
    change_status: ChangeStatus
    # --- Nivel 2: dirección, solo con evidencia ---
    change_direction: ChangeDirection = ChangeDirection.sin_direccion
    evidence_kind: EvidenceKind | None = None
    evidence_confidence: float | None = None
    # --- Magnitud ---
    baseline_score: float | None = None
    candidate_score: float | None = None
    effect_size: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p_value: float | None = None
    p_value_corrected: float | None = None
    practical_significance: PracticalSignificance = PracticalSignificance.no_detectado
    # --- Interpretabilidad ---
    comparability: Comparability = Comparability.ok
    change_taxonomy: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def direction_requires_evidence(self) -> ComparisonOutcome:
        """Sin evidencia no hay dirección. Invariante estructural, no consejo.

        Es el guardián de la regla de la línea base: un caso capturado cuya
        salida histórica era errónea no puede acabar etiquetado como regresión
        solo por diferir de ella.
        """
        tiene_direccion = self.change_direction is not ChangeDirection.sin_direccion
        if tiene_direccion and self.evidence_kind is None:
            raise ValueError(
                f"change_direction={self.change_direction.value} sin evidence_kind: "
                "una diferencia contra la línea base no es una regresión mientras "
                "no haya evidencia evaluativa que le asigne dirección"
            )
        return self

    @property
    def is_regression(self) -> bool:
        return self.change_direction is ChangeDirection.regresion

    @property
    def is_actionable(self) -> bool:
        """Regresión demostrada, relevante en la práctica y comparable."""
        return (
            self.is_regression
            and self.comparability is Comparability.ok
            and self.practical_significance is PracticalSignificance.detectado_relevante
        )
