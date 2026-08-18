"""Juez LLM: comparación pareada, anonimizada y con su error conocido.

El juez resuelve lo que no tiene oráculo, y solo eso: la escalera del plan §8.1
lo pone en el segundo escalón, después de lo determinista. Tres decisiones que
lo hacen defendible ante un tribunal:

1. **Anonimización.** El juez recibe `A` y `B` y nada más. No sabe cuál es del
   modelo antiguo, ni el nombre de los modelos, ni la procedencia. Sin eso, el
   sesgo de marca contamina cada veredicto y no hay forma de descontarlo.
2. **Los dos órdenes, siempre.** A–B y B–A. La tasa de veredictos que se
   invierten al cambiar el orden es el `position_bias`, y es una medida del
   instrumento, no un detalle de implementación.
3. **`temperature = 0` reduce la variabilidad; no demuestra determinismo.** Por
   eso `judge_self_consistency` se **mide** repitiendo una submuestra con la
   entrada idéntica, en vez de suponerse.

Y una puerta dura: un juez que no supera los mínimos de calibración **no puede
producir evidencia de nivel 2**. Sus veredictos se registran, pero los casos se
quedan en `sin_direccion`. Un juez sin error conocido es una opinión, y una
opinión no mueve una regresión.
"""
from __future__ import annotations

import random
import uuid
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from model_drift.broker_client import BrokerClient
from model_drift.broker_contracts import ModelReference
from model_drift.manifest import InstrumentRef

#: Escala ordinal del veredicto. El orden importa para el kappa ponderado.
VERDICT_SCALE = ("B_mejor", "empate", "A_mejor")


class Verdict(str, Enum):
    a_mejor = "A_mejor"
    b_mejor = "B_mejor"
    empate = "empate"
    #: El juez no devolvió algo interpretable. No se convierte en empate: eso
    #: contaría un fallo del instrumento como un dato.
    invalido = "invalido"

    def inverted(self) -> Verdict:
        if self is Verdict.a_mejor:
            return Verdict.b_mejor
        if self is Verdict.b_mejor:
            return Verdict.a_mejor
        return self


VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["veredicto", "motivo"],
    "properties": {
        "veredicto": {"type": "string", "enum": ["A", "B", "empate"]},
        "motivo": {"type": "string", "maxLength": 400},
    },
    "additionalProperties": False,
}

RUBRIC_DEFAULT = """Eres un evaluador imparcial. Recibes una petición y dos respuestas, A y B.

Decide cuál responde mejor a la petición. Criterios, por orden:
1. Corrección de la información.
2. Que cumpla lo que se pide (formato, longitud, idioma).
3. Utilidad y claridad.

No sabes quién ha escrito cada respuesta y no debes especular sobre ello.
La posición (A o B) no es información: no la uses.
Si no hay diferencia relevante, responde "empate".

Devuelve SOLO un objeto JSON con las claves "veredicto" ("A", "B" o "empate") y
"motivo" (una frase breve).
"""


@dataclass
class JudgeConfig:
    """Configuración del juez y sus mínimos de calibración."""

    model: ModelReference
    rubric: str = RUBRIC_DEFAULT
    rubric_version: str = "1"
    temperature: float = 0.0
    seed: int = 20260816
    #: Mínimos por debajo de los cuales el juez no da evidencia de nivel 2.
    min_human_agreement: float = 0.6
    min_self_consistency: float = 0.9
    max_position_bias: float = 0.2


@dataclass
class PairJudgement:
    """Un juicio sobre un par, con los dos órdenes ya resueltos."""

    case_id: str
    judgement_id: str
    verdict_ab: Verdict
    verdict_ba: Verdict
    #: Veredicto final tras descontar el orden. `invalido` si se contradicen.
    verdict: Verdict
    consistent: bool
    reason_ab: str = ""
    reason_ba: str = ""

    @property
    def favors_candidate(self) -> bool | None:
        """True si gana la candidata, False si gana la base, None si empate."""
        if self.verdict is Verdict.a_mejor:
            return False
        if self.verdict is Verdict.b_mejor:
            return True
        return None


@dataclass
class JudgeMetrics:
    """Las tres métricas del juez, más la concordancia entre humanos."""

    position_bias: float | None = None
    judge_self_consistency: float | None = None
    judge_human_agreement: float | None = None
    human_human_agreement: float | None = None
    kappa_kind: str = "ponderado"
    n_calibration: int = 0
    n_second_annotator: int = 0
    detail: dict[str, Any] = field(default_factory=dict)

    def is_calibrated(self, config: JudgeConfig) -> tuple[bool, list[str]]:
        """Si el juez puede producir evidencia de nivel 2, y por qué no si no.

        Se exige que las tres métricas estén **medidas**: un juez sin
        calibración no es un juez con calibración perfecta.
        """
        motivos: list[str] = []
        if self.judge_human_agreement is None:
            motivos.append("no se ha medido la concordancia con el criterio humano")
        elif self.judge_human_agreement < config.min_human_agreement:
            motivos.append(
                f"concordancia juez-humano {self.judge_human_agreement:.2f} < {config.min_human_agreement:.2f}"
            )
        if self.judge_self_consistency is None:
            motivos.append("no se ha medido la autoconsistencia del juez")
        elif self.judge_self_consistency < config.min_self_consistency:
            motivos.append(
                f"autoconsistencia {self.judge_self_consistency:.2f} < {config.min_self_consistency:.2f}"
            )
        if self.position_bias is not None and self.position_bias > config.max_position_bias:
            motivos.append(f"sesgo de posición {self.position_bias:.2f} > {config.max_position_bias:.2f}")
        return (not motivos), motivos

    def interpretation(self) -> str:
        """Frase que distingue un juez malo de un problema genuinamente ambiguo."""
        if self.judge_human_agreement is None:
            return "Juez sin calibrar: sus veredictos no pueden asignar dirección a ningún cambio."
        if self.human_human_agreement is None:
            return (
                f"Concordancia juez-humano {self.judge_human_agreement:.2f}. Sin una segunda persona "
                "que etiquete una submuestra no se puede saber si una discrepancia refleja un juez "
                "deficiente o un caso genuinamente ambiguo."
            )
        if self.judge_human_agreement >= self.human_human_agreement:
            return (
                f"El juez concuerda con el humano ({self.judge_human_agreement:.2f}) tanto o más que "
                f"dos personas entre sí ({self.human_human_agreement:.2f}): la discrepancia restante "
                "es del problema, no del instrumento."
            )
        return (
            f"Concordancia juez-humano {self.judge_human_agreement:.2f} por debajo de la concordancia "
            f"entre humanos ({self.human_human_agreement:.2f}): parte del desacuerdo es del juez."
        )


# --------------------------------------------------------------------------
# Concordancia
# --------------------------------------------------------------------------


def cohen_kappa(
    a: list[str],
    b: list[str],
    *,
    scale: tuple[str, ...] = VERDICT_SCALE,
    weighted: bool = True,
) -> float:
    """Kappa de Cohen, simple o con pesos lineales.

    Se usa el ponderado cuando la escala es ordinal —`B_mejor < empate <
    A_mejor`—, porque confundir «empate» con «A mejor» no es el mismo error que
    confundir «A mejor» con «B mejor», y el kappa simple los cobra igual.
    """
    if len(a) != len(b) or not a:
        return 0.0
    indices = {valor: posicion for posicion, valor in enumerate(scale)}
    n = len(a)
    k = len(scale)

    def peso(i: int, j: int) -> float:
        if not weighted or k < 2:
            return 1.0 if i == j else 0.0
        return 1.0 - abs(i - j) / (k - 1)

    conteo_a = Counter(a)
    conteo_b = Counter(b)

    observada = 0.0
    for izquierda, derecha in zip(a, b, strict=True):
        if izquierda not in indices or derecha not in indices:
            continue
        observada += peso(indices[izquierda], indices[derecha])
    observada /= n

    esperada = 0.0
    for etiqueta_a, indice_a in indices.items():
        for etiqueta_b, indice_b in indices.items():
            probabilidad = (conteo_a.get(etiqueta_a, 0) / n) * (conteo_b.get(etiqueta_b, 0) / n)
            esperada += probabilidad * peso(indice_a, indice_b)

    if esperada >= 1.0:
        # Concordancia esperada perfecta: kappa no está definido. Devolver 1.0
        # afirmaría un acuerdo que el estadístico no puede sostener.
        return 0.0
    return (observada - esperada) / (1.0 - esperada)


def raw_agreement(a: list[str], b: list[str]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    return sum(1 for izquierda, derecha in zip(a, b, strict=True) if izquierda == derecha) / len(a)


# --------------------------------------------------------------------------
# El juez
# --------------------------------------------------------------------------


class Judge:
    """Juez pareado que solo ve A y B."""

    def __init__(self, client: BrokerClient, config: JudgeConfig, *, group: str = "model-drift-judge") -> None:
        self.client = client
        self.config = config
        self.group = group
        self.metrics = JudgeMetrics()

    def instrument(self) -> InstrumentRef:
        return InstrumentRef(
            id="judge",
            kind="juez",
            version=f"{self.config.model.key}|rubrica-{self.config.rubric_version}",
            detail=f"temperature={self.config.temperature}",
        )

    def rubric_instrument(self) -> InstrumentRef:
        return InstrumentRef(id="rubrica", kind="rubrica", version=self.config.rubric_version)

    # ------------------------------------------------------------- invocación

    def _prompt(self, question: str, left: str, right: str) -> str:
        """Prompt anonimizado. Ni nombres de modelo ni pistas de procedencia."""
        return (
            f"{self.config.rubric}\n\n"
            f"### Petición\n{question}\n\n"
            f"### Respuesta A\n{left}\n\n"
            f"### Respuesta B\n{right}\n"
        )

    def _ask(self, question: str, left: str, right: str) -> tuple[Verdict, str]:
        resultado = self.client.invoke(
            prompt=self._prompt(question, left, right),
            target=self.config.model,
            params={"temperature": self.config.temperature, "seed": self.config.seed},
            group=self.group,
            json_schema=VERDICT_SCHEMA,
        )
        if not resultado.produced_output:
            return Verdict.invalido, resultado.error_message or "el juez no produjo salida"
        return _parse_verdict(resultado.text)

    def judge_pair(
        self,
        case_id: str,
        question: str,
        baseline_output: str,
        candidate_output: str,
        *,
        rng: random.Random | None = None,
    ) -> PairJudgement:
        """Juzga un par en los dos órdenes y descuenta el sesgo de posición.

        El orden en que se presentan por primera vez se aleatoriza para que un
        sesgo residual no se alinee sistemáticamente con «la base va primero».
        """
        rng = rng or random.Random(self.config.seed)
        base_primero = rng.random() < 0.5

        if base_primero:
            veredicto_ab, motivo_ab = self._ask(question, baseline_output, candidate_output)
            veredicto_ba, motivo_ba = self._ask(question, candidate_output, baseline_output)
            # En el segundo orden A es la candidata: se invierte para hablar
            # siempre en términos de "A = base".
            veredicto_ba = veredicto_ba.inverted()
        else:
            veredicto_ba, motivo_ba = self._ask(question, candidate_output, baseline_output)
            veredicto_ba = veredicto_ba.inverted()
            veredicto_ab, motivo_ab = self._ask(question, baseline_output, candidate_output)

        coherente = veredicto_ab is veredicto_ba and veredicto_ab is not Verdict.invalido
        final = veredicto_ab if coherente else Verdict.invalido

        return PairJudgement(
            case_id=case_id,
            judgement_id=f"jud-{uuid.uuid4().hex[:16]}",
            verdict_ab=veredicto_ab,
            verdict_ba=veredicto_ba,
            verdict=final,
            consistent=coherente,
            reason_ab=motivo_ab,
            reason_ba=motivo_ba,
        )

    # ------------------------------------------------------------ calibración

    def measure_position_bias(self, judgements: list[PairJudgement]) -> float:
        """Proporción de veredictos que se invierten al cambiar el orden."""
        validos = [item for item in judgements if Verdict.invalido not in {item.verdict_ab, item.verdict_ba}]
        if not validos:
            return 0.0
        inconsistentes = sum(1 for item in validos if item.verdict_ab is not item.verdict_ba)
        self.metrics.position_bias = inconsistentes / len(validos)
        return self.metrics.position_bias

    def measure_self_consistency(
        self,
        sample: list[tuple[str, str, str, str]],
        *,
        repetitions: int = 3,
    ) -> float:
        """Repite una submuestra con la entrada IDÉNTICA y mide la estabilidad.

        Entrada idéntica quiere decir el mismo orden y el mismo texto: aquí no
        se está midiendo el sesgo de posición, sino si el juez se contradice
        consigo mismo. Son cosas distintas y el informe las separa.

        `sample` son tuplas `(case_id, pregunta, respuesta_A, respuesta_B)`.
        """
        if not sample or repetitions < 2:
            return 0.0
        estables = 0
        for _, pregunta, izquierda, derecha in sample:
            veredictos = [self._ask(pregunta, izquierda, derecha)[0].value for _ in range(repetitions)]
            if len(set(veredictos)) == 1:
                estables += 1
        self.metrics.judge_self_consistency = estables / len(sample)
        self.metrics.detail["self_consistency_repetitions"] = repetitions
        return self.metrics.judge_self_consistency

    def measure_human_agreement(
        self,
        judge_labels: list[str],
        human_labels: list[str],
        *,
        second_annotator: list[str] | None = None,
        weighted: bool = True,
    ) -> JudgeMetrics:
        """Concordancia juez-humano y, si la hay, humano-humano.

        La segunda anotación no es un lujo: sin ella, una concordancia de 0,55
        no se puede interpretar. Puede ser un juez mediocre o puede ser un
        conjunto de casos donde dos personas tampoco se ponen de acuerdo.
        """
        self.metrics.kappa_kind = "ponderado" if weighted else "simple"
        self.metrics.n_calibration = len(human_labels)
        self.metrics.judge_human_agreement = cohen_kappa(judge_labels, human_labels, weighted=weighted)
        self.metrics.detail["judge_human_raw_agreement"] = raw_agreement(judge_labels, human_labels)
        if second_annotator:
            n = min(len(second_annotator), len(human_labels))
            self.metrics.n_second_annotator = n
            self.metrics.human_human_agreement = cohen_kappa(
                human_labels[:n], second_annotator[:n], weighted=weighted
            )
        return self.metrics

    def is_calibrated(self) -> bool:
        calibrado, _ = self.metrics.is_calibrated(self.config)
        return calibrado


def _parse_verdict(text: str) -> tuple[Verdict, str]:
    """Extrae el veredicto del JSON del juez.

    Una salida que no se puede interpretar es `invalido`, nunca `empate`:
    convertir un fallo del instrumento en un dato sería fabricar evidencia.
    """
    from model_drift.oracles import extract_json

    payload, _ = extract_json(text)
    if not isinstance(payload, dict):
        return Verdict.invalido, "la salida del juez no es un objeto JSON"
    crudo = str(payload.get("veredicto", "")).strip().lower()
    motivo = str(payload.get("motivo", ""))[:400]
    if crudo == "a":
        return Verdict.a_mejor, motivo
    if crudo == "b":
        return Verdict.b_mejor, motivo
    if crudo in {"empate", "tie", "igual"}:
        return Verdict.empate, motivo
    return Verdict.invalido, f"veredicto no reconocido: {crudo!r}"
