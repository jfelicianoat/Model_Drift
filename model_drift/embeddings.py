"""El modelo de embeddings es un instrumento de medida, no una utilidad.

Todo lo que el plan §12 exige vive aquí: modelo congelado e identificado por su
huella, configuración versionada, umbral guardado por decisión y calibrado con
procedencia. Un umbral elegido a ojo convierte la medición en una opinión con
decimales, así que `calibrate_threshold` existe para que haya de dónde sacarlo.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from model_drift.broker_client import BrokerClient
from model_drift.broker_contracts import InferenceKind, ModelReference
from model_drift.manifest import InstrumentRef
from model_drift.oracles import cosine_similarity


@dataclass
class BrokerEmbedder:
    """Instrumento de similitud servido por el Broker.

    Cachea por texto: una relación metamórfica compara la misma salida de
    referencia contra cinco variantes, y pagar cinco veces el mismo embedding
    sería tirar presupuesto.
    """

    client: BrokerClient
    model: ModelReference
    group: str = "model-drift-embeddings"
    _cache: dict[str, list[float]] = field(default_factory=dict)
    failures: int = 0

    @property
    def identity(self) -> str:
        return self.model.key

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectores: list[list[float]] = []
        for texto in texts:
            if texto in self._cache:
                vectores.append(self._cache[texto])
                continue
            resultado = self.client.invoke(
                prompt=texto,
                target=self.model,
                inference_kind=InferenceKind.embedding,
                group=self.group,
            )
            if not resultado.ok or resultado.embedding is None:
                # Sin vector no se inventa una similitud: quien llama lo trata
                # como aserción no aplicable.
                self.failures += 1
                return []
            self._cache[texto] = resultado.embedding
            vectores.append(resultado.embedding)
        return vectores

    def instrument(self) -> InstrumentRef:
        """Referencia para el manifiesto. Cambiarlo rompe la comparabilidad."""
        return InstrumentRef(
            id="embedder",
            kind="embedding",
            version=self.model.model,
            hash=None,
            detail=f"embeddings servidos por el Broker: {self.model.key}",
        )


@dataclass
class StaticEmbedder:
    """Instrumento de pruebas: vectores fijados de antemano.

    No es un modelo de embeddings de mentira metido en producción: sirve para
    que las pruebas del motor no dependan de un Broker vivo, y su identidad lo
    dice sin ambigüedad para que nunca se cuele en un manifiesto real.
    """

    vectors: dict[str, list[float]]
    name: str = "static-test-embedder"

    @property
    def identity(self) -> str:
        return f"pruebas:{self.name}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors.get(texto, [0.0]) for texto in texts]


@dataclass
class CalibrationPoint:
    """Un par etiquetado a mano: ¿son equivalentes estos dos textos?"""

    left: str
    right: str
    equivalent: bool


@dataclass
class ThresholdCalibration:
    """Umbral elegido sobre datos etiquetados, con su punto de operación."""

    threshold: float
    criterion: str
    f1: float
    precision: float
    recall: float
    n_points: int
    embedder_identity: str

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, ensure_ascii=False)

    def save(self, path: str | Path) -> Path:
        destino = Path(path)
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(self.to_json(), encoding="utf-8")
        return destino


def calibrate_threshold(
    points: list[CalibrationPoint],
    embedder: object,
    *,
    criterion: str = "f1",
    steps: int = 100,
) -> ThresholdCalibration:
    """Elige el umbral que optimiza el criterio sobre pares etiquetados.

    El criterio se guarda junto al valor: un umbral de 0,87 no significa nada
    sin saber que se eligió maximizando F1 sobre 120 pares etiquetados con
    tal modelo de embeddings. Ese fichero es el que las aserciones referencian
    en su campo `calibration`.
    """
    if not points:
        raise ValueError("no se puede calibrar un umbral sin pares etiquetados")

    embed = embedder.embed  # type: ignore[attr-defined]
    identidad = str(getattr(embedder, "identity", "desconocido"))

    similitudes: list[tuple[float, bool]] = []
    for punto in points:
        vectores = embed([punto.left, punto.right])
        if len(vectores) != 2:
            continue
        similitudes.append((cosine_similarity(vectores[0], vectores[1]), punto.equivalent))
    if not similitudes:
        raise ValueError("el instrumento de embeddings no devolvió ningún vector utilizable")

    mejor = ThresholdCalibration(
        threshold=0.5,
        criterion=criterion,
        f1=0.0,
        precision=0.0,
        recall=0.0,
        n_points=len(similitudes),
        embedder_identity=identidad,
    )
    for paso in range(steps + 1):
        umbral = paso / steps
        vp = sum(1 for score, etiqueta in similitudes if score >= umbral and etiqueta)
        fp = sum(1 for score, etiqueta in similitudes if score >= umbral and not etiqueta)
        fn = sum(1 for score, etiqueta in similitudes if score < umbral and etiqueta)
        precision = vp / (vp + fp) if vp + fp else 0.0
        recall = vp / (vp + fn) if vp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > mejor.f1:
            mejor = ThresholdCalibration(
                threshold=round(umbral, 4),
                criterion=criterion,
                f1=round(f1, 4),
                precision=round(precision, 4),
                recall=round(recall, 4),
                n_points=len(similitudes),
                embedder_identity=identidad,
            )
    return mejor
