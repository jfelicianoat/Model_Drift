"""Contrastes exactos y remuestreo: McNemar, Fisher, permutación y bootstrap.

Implementaciones propias y exactas a propósito: una tirada tiene pocos
casos, y ahí las aproximaciones asintóticas mienten. `minimum_attainable_p`
existe para poder decir «con este N no se puede concluir» en vez de dar un
p-valor que nunca podría bajar del umbral.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import mean


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
