"""Estadística de la comparación. La unidad de análisis es **el caso**.

Este módulo existe para impedir el error que hunde la mayoría de evaluaciones
caseras de LLMs: tratar las N repeticiones de un caso como N observaciones
independientes. No lo son. Están correlacionadas por construcción —mismo
prompt, mismo modelo, mismo contexto— y contarlas como muestras infla la
potencia artificialmente y produce significación donde no la hay. Con 50 casos
× 10 repeticiones hay **50** observaciones, no 500.

El tratamiento va en dos pisos (plan §9.1):

- **Piso 1, dentro del caso**: las repeticiones estiman la *distribución* del
  caso. De ellas salen `case_score` y `case_stability`. No aportan tamaño
  muestral.
- **Piso 2, entre casos**: el contraste se hace sobre casos pareados, con
  `case_score` como observación. McNemar para binarias; bootstrap **por
  conglomerados remuestreando casos** para continuas.

Todo está implementado sin scipy ni statsmodels, a propósito: son cuatro
procedimientos cortos, el TFM gana en auditabilidad si se ven, y el paquete
instala en cualquier Python sin ruedas binarias.

Extensión futura consciente: modelos jerárquicos / mixed-effects para modelar
explícitamente la varianza intra e inter caso. **No es requisito del MVP** y no
debe retrasar las fases 1-2.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from statistics import mean, median

from model_drift.domain import PracticalSignificance

# --------------------------------------------------------------------------
# Piso 1: agregación dentro del caso
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseAggregate:
    """Resumen de las N repeticiones de un caso. Una sola observación."""

    score: float | None
    stability: float
    repetitions: int
    values: tuple[float, ...] = ()

    @property
    def is_empty(self) -> bool:
        return self.repetitions == 0 or self.score is None


def aggregate_binary(values: list[bool]) -> CaseAggregate:
    """Agrega repeticiones binarias en tasa de acierto y acuerdo.

    `stability` es la proporción del valor mayoritario: 1.0 si todas las
    repeticiones coinciden, 0.5 si están repartidas al 50 %. Un caso que dice
    que sí la mitad de las veces no mide el modelo, mide el azar.
    """
    if not values:
        return CaseAggregate(score=None, stability=0.0, repetitions=0)
    aciertos = sum(1 for value in values if value)
    tasa = aciertos / len(values)
    estabilidad = max(tasa, 1.0 - tasa)
    return CaseAggregate(
        score=tasa,
        stability=estabilidad,
        repetitions=len(values),
        values=tuple(1.0 if value else 0.0 for value in values),
    )


def aggregate_continuous(values: list[float], *, use_median: bool = True) -> CaseAggregate:
    """Agrega repeticiones continuas (latencia, tokens, similitud).

    `stability` = 1 - coeficiente de variación acotado a [0, 1]. Una latencia
    que oscila un 5 % es estable; una que oscila un 200 % no mide nada.
    """
    if not values:
        return CaseAggregate(score=None, stability=0.0, repetitions=0)
    centro = median(values) if use_median else mean(values)
    if len(values) == 1:
        return CaseAggregate(score=centro, stability=1.0, repetitions=1, values=(values[0],))
    promedio = mean(values)
    varianza = sum((value - promedio) ** 2 for value in values) / (len(values) - 1)
    desviacion = math.sqrt(varianza)
    if promedio == 0.0:
        estabilidad = 1.0 if desviacion == 0.0 else 0.0
    else:
        estabilidad = max(0.0, 1.0 - desviacion / abs(promedio))
    return CaseAggregate(score=centro, stability=min(1.0, estabilidad), repetitions=len(values), values=tuple(values))


def binarize(score: float, rule: str) -> bool:
    """Convierte `case_score` en un estado binario según la regla de la suite.

    `todas` es el defecto para oráculos deterministas: un caso que falla 3 de
    10 veces no está aprobado. `mayoria` pierde información a propósito y la
    suite tiene que declararlo.
    """
    if rule == "mayoria":
        return score > 0.5
    return score >= 1.0


# --------------------------------------------------------------------------
# Piso 2: contrastes entre casos pareados
# --------------------------------------------------------------------------


def _binomial_tail(k: int, n: int) -> float:
    """P(X <= k) para X ~ Binomial(n, 0.5). Exacto, sin aproximación normal."""
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(k + 1)) / (2**n)


@dataclass(frozen=True)
class McNemarResult:
    """Contraste de McNemar exacto sobre pares discordantes."""

    #: Casos que pasaban en la base y fallan en la candidata.
    b: int
    #: Casos que fallaban en la base y pasan en la candidata.
    c: int
    p_value: float
    #: Diferencia de tasas: (c - b) / n_casos. Positiva = mejora la candidata.
    effect: float
    n_cases: int

    @property
    def discordant(self) -> int:
        return self.b + self.c


def mcnemar_exact(baseline: list[bool], candidate: list[bool]) -> McNemarResult:
    """McNemar exacto para resultados binarios pareados.

    Solo los pares **discordantes** llevan información: los casos que pasan (o
    fallan) en las dos configuraciones no dicen nada sobre el cambio. Bajo la
    hipótesis nula, un discordante cae a cada lado con probabilidad 1/2, así
    que el contraste es una binomial exacta — sin aproximación de chi-cuadrado,
    que es justo la que se rompe cuando hay pocos discordantes, que es el caso
    normal en una suite pequeña.
    """
    if len(baseline) != len(candidate):
        raise ValueError("McNemar exige pares: las dos listas deben medir lo mismo")
    n = len(baseline)
    b = sum(1 for base, cand in zip(baseline, candidate, strict=True) if base and not cand)
    c = sum(1 for base, cand in zip(baseline, candidate, strict=True) if not base and cand)
    discordantes = b + c
    if discordantes == 0:
        p_value = 1.0
    else:
        p_value = min(1.0, 2.0 * _binomial_tail(min(b, c), discordantes))
    efecto = (c - b) / n if n else 0.0
    return McNemarResult(b=b, c=c, p_value=p_value, effect=efecto, n_cases=n)


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Test exacto de Fisher bilateral sobre una tabla 2x2.

    Se usa **dentro de un caso**, no entre casos, y esa distinción es toda la
    metodología del módulo: las N repeticiones de un mismo caso, con la
    configuración fija, sí son extracciones independientes de la distribución
    de ese caso, así que comparar 8/10 contra 3/10 aciertos es legítimo. Lo que
    no lo es —y lo impide el piso 2— es juntar las 500 repeticiones de 50 casos
    como si fueran 500 unidades independientes para la conclusión agregada.

    Tabla: (a, b) aciertos y fallos de la base; (c, d) de la candidata.
    """
    n = a + b + c + d
    if n == 0:
        return 1.0
    fila1, fila2 = a + b, c + d
    col1, col2 = a + c, b + d
    if fila1 == 0 or fila2 == 0 or col1 == 0 or col2 == 0:
        return 1.0

    def probabilidad(x: int) -> float:
        return math.comb(fila1, x) * math.comb(fila2, col1 - x) / math.comb(n, col1)

    observada = probabilidad(a)
    inferior = max(0, col1 - fila2)
    superior = min(col1, fila1)
    # Bilateral por el criterio de la verosimilitud: se suman todas las tablas
    # al menos tan improbables como la observada. El margen relativo evita que
    # el ruido de coma flotante deje fuera una tabla equiprobable.
    total = sum(
        probabilidad(x) for x in range(inferior, superior + 1) if probabilidad(x) <= observada * (1 + 1e-9)
    )
    return min(1.0, total)


def minimum_attainable_p(n_baseline: int, n_candidate: int, *, binary: bool) -> float:
    """El p-valor más pequeño que puede producir un caso con esas repeticiones.

    Los contrastes exactos tienen suelo: con 3 repeticiones a cada lado, la
    tabla más extrema posible (3 aciertos contra 0) da p = 0,1. Ninguna
    degradación, por total que sea, puede bajar de ahí.

    Esto importa porque sin tenerlo en cuenta el motor cometería un falso
    negativo por construcción: un caso que pasa el 100 % de las veces y luego
    falla el 100 % se informaría como «sin cambio observable» cuando lo honesto
    es decir que con esas repeticiones **no se puede decidir**.
    """
    if n_baseline <= 0 or n_candidate <= 0:
        return 1.0
    if binary:
        return fisher_exact_2x2(n_baseline, 0, 0, n_candidate)
    # Permutación: el mínimo alcanzable es la fracción de particiones tan
    # extremas como la observada, que en el mejor de los casos son dos (la
    # original y su espejo).
    return min(1.0, 2.0 / math.comb(n_baseline + n_candidate, n_baseline))


def permutation_test(
    baseline: list[float],
    candidate: list[float],
    *,
    iterations: int = 2000,
    seed: int = 20260815,
) -> float:
    """Test de permutación sobre las repeticiones de un mismo caso.

    Es el contraste para métricas continuas —latencia, coste, tokens— donde el
    exacto de Fisher no tiene sentido porque no hay aciertos que contar. Bajo la
    hipótesis nula las repeticiones de las dos configuraciones son
    intercambiables, así que se baraja la etiqueta y se mira cuántas veces la
    diferencia observada se alcanza por azar.

    Sin supuestos de normalidad, que con cuatro repeticiones y latencias de cola
    larga serían indefendibles.
    """
    if not baseline or not candidate:
        return 1.0
    observada = abs(mean(candidate) - mean(baseline))
    if observada == 0.0:
        return 1.0
    conjunto = baseline + candidate
    corte = len(baseline)
    rng = random.Random(seed)
    extremas = 0
    for _ in range(iterations):
        rng.shuffle(conjunto)
        diferencia = abs(mean(conjunto[corte:]) - mean(conjunto[:corte]))
        if diferencia >= observada - 1e-12:
            extremas += 1
    # Corrección de continuidad (+1): con permutaciones muestreadas, un p-valor
    # de cero afirmaría una imposibilidad que el muestreo no puede demostrar.
    return (extremas + 1) / (iterations + 1)


@dataclass(frozen=True)
class BootstrapResult:
    """Intervalo de confianza de la diferencia media entre configuraciones."""

    effect: float
    ci_low: float
    ci_high: float
    iterations: int

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0


def cluster_bootstrap(
    baseline: list[float],
    candidate: list[float],
    *,
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260815,
    baseline_reps: list[list[float]] | None = None,
    candidate_reps: list[list[float]] | None = None,
) -> BootstrapResult:
    """Bootstrap remuestreando **casos**, nunca invocaciones.

    Remuestrear invocaciones trataría las repeticiones como independientes y
    devolvería un intervalo demasiado estrecho: exactamente el error que este
    módulo existe para evitar.

    Si se pasan `baseline_reps` / `candidate_reps` se activa el remuestreo en
    dos niveles —primero casos, después repeticiones dentro de cada caso
    elegido— para propagar también la variabilidad intra-caso al intervalo.
    """
    if len(baseline) != len(candidate):
        raise ValueError("el bootstrap pareado exige el mismo número de casos en ambos lados")
    n = len(baseline)
    if n == 0:
        return BootstrapResult(effect=0.0, ci_low=0.0, ci_high=0.0, iterations=0)

    diferencias = [cand - base for base, cand in zip(baseline, candidate, strict=True)]
    efecto = mean(diferencias)

    rng = random.Random(seed)
    dos_niveles = baseline_reps is not None and candidate_reps is not None
    muestras: list[float] = []
    for _ in range(iterations):
        indices = [rng.randrange(n) for _ in range(n)]
        if dos_niveles:
            assert baseline_reps is not None and candidate_reps is not None
            acumulado: list[float] = []
            for index in indices:
                base_reps = baseline_reps[index] or [baseline[index]]
                cand_reps = candidate_reps[index] or [candidate[index]]
                base_muestra = mean(rng.choice(base_reps) for _ in range(len(base_reps)))
                cand_muestra = mean(rng.choice(cand_reps) for _ in range(len(cand_reps)))
                acumulado.append(cand_muestra - base_muestra)
            muestras.append(mean(acumulado))
        else:
            muestras.append(mean(diferencias[index] for index in indices))

    muestras.sort()
    alpha = (1.0 - confidence) / 2.0
    bajo = muestras[max(0, int(math.floor(alpha * iterations)))]
    alto = muestras[min(iterations - 1, int(math.ceil((1.0 - alpha) * iterations)) - 1)]
    return BootstrapResult(effect=efecto, ci_low=bajo, ci_high=alto, iterations=iterations)


# --------------------------------------------------------------------------
# Comparaciones múltiples
# --------------------------------------------------------------------------


def benjamini_hochberg(p_values: list[float], *, fdr: float = 0.05) -> tuple[list[float], list[bool]]:
    """Corrección de Benjamini-Hochberg. Devuelve (p ajustados, rechazos).

    Cincuenta prompts por ocho métricas son cuatrocientos contrastes: sin
    corrección, con α = 0,05, salen veinte "regresiones" de puro azar. Se
    controla la tasa de falsos descubrimientos en vez de la familiar, porque en
    detección de deriva importa más no ahogarse en falsas alarmas que blindar
    contra un único falso positivo.

    La familia de contrastes se declara ANTES de mirar los resultados; este
    módulo solo la corrige.
    """
    m = len(p_values)
    if m == 0:
        return [], []
    indexados = sorted(enumerate(p_values), key=lambda par: par[1])
    ajustados = [0.0] * m
    minimo = 1.0
    # Recorrido descendente para imponer monotonía: un p ajustado nunca puede
    # ser mayor que el de un contraste con p mayor.
    for rango in range(m, 0, -1):
        indice, p = indexados[rango - 1]
        candidato = min(1.0, p * m / rango)
        minimo = min(minimo, candidato)
        ajustados[indice] = minimo
    rechazos = [ajustado <= fdr for ajustado in ajustados]
    return ajustados, rechazos


# --------------------------------------------------------------------------
# Relevancia práctica
# --------------------------------------------------------------------------


def classify_practical(
    effect: float,
    *,
    significant: bool,
    minimum_practical_effect: float,
) -> PracticalSignificance:
    """Cruza significación estadística con relevancia práctica (plan §13.1).

    Con suficientes casos, una diferencia de 0,3 puntos porcentuales sale
    significativa y no le importa a nadie. `detectado_irrelevante` es un
    resultado legítimo y frecuente, y el informe debe decirlo con ese nombre en
    vez de alarmar.
    """
    # El umbral práctico se aplica a la MAGNITUD del cambio, en cualquier
    # dirección: una mejora grande también es relevante para quien decide si
    # adopta la versión nueva. La dirección la aporta `higher_is_better` a
    # quien construye el veredicto (policy.py), no esta clasificación.
    magnitud = abs(effect)
    supera_umbral = magnitud >= minimum_practical_effect
    if significant and supera_umbral:
        return PracticalSignificance.detectado_relevante
    if significant:
        return PracticalSignificance.detectado_irrelevante
    if supera_umbral:
        return PracticalSignificance.relevante_no_concluyente
    return PracticalSignificance.no_detectado


# --------------------------------------------------------------------------
# Potencia: control negativo y control positivo de la Fase 2
# --------------------------------------------------------------------------


@dataclass
class PowerEstimate:
    """Potencia estimada por simulación para una configuración concreta."""

    n_cases: int
    baseline_rate: float
    difference: float
    alpha: float
    power: float
    iterations: int
    detected: int = field(default=0)


def estimate_power(
    *,
    n_cases: int,
    baseline_rate: float,
    difference: float,
    alpha: float = 0.05,
    iterations: int = 400,
    seed: int = 20260815,
) -> PowerEstimate:
    """Simula pares y estima con qué frecuencia se detecta una diferencia dada.

    Sirve para las dos mitades del criterio de aceptación de la Fase 2:

    - con `difference = 0` estima la **tasa de falsas alarmas**, que debe
      quedar en el entorno de `alpha`;
    - con una degradación conocida estima la **potencia de detección**.

    Modelo de simulación: cada caso pasa en la base con probabilidad
    `baseline_rate` y en la candidata con `baseline_rate + difference`, de forma
    independiente entre configuraciones. Es deliberadamente conservador —los
    casos reales están correlacionados entre configuraciones, lo que reduce la
    discordancia y aumenta la potencia—, así que la potencia real será igual o
    mayor que la estimada aquí.
    """
    rng = random.Random(seed)
    rate_candidate = min(1.0, max(0.0, baseline_rate + difference))
    detectados = 0
    for _ in range(iterations):
        base = [rng.random() < baseline_rate for _ in range(n_cases)]
        cand = [rng.random() < rate_candidate for _ in range(n_cases)]
        if mcnemar_exact(base, cand).p_value <= alpha:
            detectados += 1
    return PowerEstimate(
        n_cases=n_cases,
        baseline_rate=baseline_rate,
        difference=difference,
        alpha=alpha,
        power=detectados / iterations if iterations else 0.0,
        iterations=iterations,
        detected=detectados,
    )


def minimum_detectable_effect(
    *,
    n_cases: int,
    baseline_rate: float = 0.9,
    alpha: float = 0.05,
    target_power: float = 0.8,
    iterations: int = 300,
    seed: int = 20260815,
) -> float | None:
    """Menor degradación que esta configuración detecta con la potencia pedida.

    Es el número que hace honesto un «sin cambio observable»: no es lo mismo
    sobre 12 casos que sobre 300, y el informe no debe permitir confundirlos.
    Devuelve `None` si ni siquiera una caída total alcanza la potencia objetivo.
    """
    if n_cases <= 0:
        return None
    paso = 0.01
    diferencia = paso
    while diferencia <= baseline_rate:
        estimate = estimate_power(
            n_cases=n_cases,
            baseline_rate=baseline_rate,
            difference=-diferencia,
            alpha=alpha,
            iterations=iterations,
            seed=seed,
        )
        if estimate.power >= target_power:
            return round(diferencia, 4)
        diferencia += paso
    return None


def power_curve(
    *,
    n_cases: int,
    baseline_rate: float = 0.9,
    differences: tuple[float, ...] = (-0.02, -0.05, -0.10, -0.20, -0.30),
    alpha: float = 0.05,
    iterations: int = 300,
    seed: int = 20260815,
) -> list[PowerEstimate]:
    """Curva de potencia por magnitud. Va al informe de la Fase 2."""
    return [
        estimate_power(
            n_cases=n_cases,
            baseline_rate=baseline_rate,
            difference=difference,
            alpha=alpha,
            iterations=iterations,
            seed=seed,
        )
        for difference in differences
    ]
