"""Relaciones metamórficas formales.

Una prueba metamórfica no necesita conocer la respuesta correcta, pero **sí
necesita una relación formal** (plan §7.3). «Parafraseo y miro a ver» no es un
oráculo: sin transformación, relación, comparador y umbral declarados, lo que
sale es una impresión. Por eso `MetamorphicRelation` exige los cuatro y el
umbral exige procedencia de calibración, igual que en `oracles.py`.

Dos relaciones, no una:

- `invariancia` — la salida debe mantenerse equivalente bajo la transformación.
- `cambio_exigido` — la salida **debe** cambiar. Es la que usa la sensibilidad a
  mutaciones: cambiar un «no» o una cifra y obtener la misma respuesta delata a
  un modelo que reconoce el patrón en vez de leer.

Sobre los generadores de variantes: los que se pueden producir de forma
determinista (orden, contexto irrelevante, anclaje, adulación) se generan aquí
con semilla fija. Los que no —paráfrasis y traducción— **salen de un fixture
versionado**, porque generarlos con otro modelo convertiría la transformación en
un segundo instrumento sin calibrar metido en mitad de la medición.
"""
from __future__ import annotations

import random
import re
from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from model_drift.fixtures import FixtureResolver
from model_drift.oracles import Embedder, cosine_similarity, normalize_text


class RelationKind(str, Enum):
    invariancia = "invariancia"
    cambio_exigido = "cambio_exigido"


class TransformationKind(str, Enum):
    #: Requieren fixture: no se pueden generar de forma determinista.
    parafrasis = "parafrasis"
    idioma = "idioma"
    mutacion = "mutacion"
    #: Deterministas, generadas aquí con semilla.
    orden = "orden"
    contexto_irrelevante = "contexto_irrelevante"
    anclaje = "anclaje"
    adulacion = "adulacion"


#: Transformaciones que no se pueden fabricar sin otro modelo de por medio.
REQUIEREN_FIXTURE = {TransformationKind.parafrasis, TransformationKind.idioma, TransformationKind.mutacion}


class Transformation(BaseModel):
    kind: TransformationKind
    #: `fixture://...` con las variantes escritas a mano, cuando hace falta.
    generator: str | None = None
    variants: int = Field(default=3, ge=1, le=20)
    #: La generación de variantes también se fija: una transformación que
    #: baraja distinto en cada tirada no es un instrumento, es ruido.
    seed: int = 20260815
    #: Texto del distractor / ancla, para las transformaciones programáticas.
    payload: str | None = None

    @model_validator(mode="after")
    def fixture_required(self) -> Transformation:
        if self.kind in REQUIEREN_FIXTURE and not self.generator:
            raise ValueError(
                f"la transformación {self.kind.value} exige 'generator' con variantes escritas: "
                "generarlas con otro modelo metería un segundo instrumento sin calibrar "
                "en mitad de la medición"
            )
        return self


class Comparator(BaseModel):
    kind: Literal["exacto", "normalizado", "embedding", "juez"] = "normalizado"
    #: Identificación del modelo de embeddings, cuando `kind == "embedding"`.
    model: str | None = None


class Threshold(BaseModel):
    """Umbral con procedencia obligatoria (plan §12)."""

    value: float = Field(ge=0.0, le=1.0)
    calibration: str

    @model_validator(mode="after")
    def calibration_required(self) -> Threshold:
        if not self.calibration.strip():
            raise ValueError(
                "el umbral de una relación metamórfica exige 'calibration': "
                "un valor sin procedencia documentada no es un oráculo"
            )
        return self


class SuccessCriterion(BaseModel):
    """Cuántas variantes deben cumplir la relación para dar el caso por bueno."""

    min_satisfied: int | None = None
    min_ratio: float | None = Field(default=None, ge=0.0, le=1.0)

    @classmethod
    def parse(cls, raw: Any) -> SuccessCriterion:
        """Acepta la forma estructurada y la frase del plan («al menos 4 de 5»)."""
        if raw is None:
            return cls(min_ratio=1.0)
        if isinstance(raw, dict):
            return cls.model_validate(raw)
        texto = str(raw).lower()
        match = re.search(r"(\d+)\s*(?:de|/)\s*(\d+)", texto)
        if match:
            return cls(min_satisfied=int(match.group(1)))
        return cls(min_ratio=1.0)

    def satisfied(self, cumplidas: int, total: int) -> bool:
        if total == 0:
            return False
        if self.min_satisfied is not None:
            return cumplidas >= self.min_satisfied
        return cumplidas / total >= (self.min_ratio if self.min_ratio is not None else 1.0)


class MetamorphicRelation(BaseModel):
    """Relación metamórfica completa. Los cuatro campos son obligatorios."""

    id: str
    version: str = "1"
    description: str = ""
    transformation: Transformation
    relation: RelationKind
    comparator: Comparator = Field(default_factory=Comparator)
    threshold: Threshold | None = None
    success_criterion: SuccessCriterion = Field(default_factory=lambda: SuccessCriterion(min_ratio=1.0))

    @model_validator(mode="after")
    def threshold_required_for_fuzzy(self) -> MetamorphicRelation:
        if self.comparator.kind == "embedding" and self.threshold is None:
            raise ValueError(
                f"la relación {self.id!r} compara por embeddings y no declara umbral calibrado"
            )
        return self

    @property
    def identity(self) -> str:
        """Lo que se compara entre tiradas. Cambiar la versión rompe la comparabilidad."""
        return f"{self.id}@{self.version}"


class VariantResult(BaseModel):
    variant_id: str
    prompt: str
    output: str = ""
    similarity: float | None = None
    satisfies_relation: bool = False
    detail: str = ""


class RelationOutcome(BaseModel):
    """Resultado de aplicar una relación a un caso."""

    relation_id: str
    relation_version: str
    passed: bool
    satisfied: int
    total: int
    variants: list[VariantResult] = Field(default_factory=list)
    not_applicable: bool = False
    reason: str = ""


# --------------------------------------------------------------------------
# Generación de variantes
# --------------------------------------------------------------------------

_ITEM = re.compile(r"^\s*(?:[-*]|\d+[.)])\s+(.*)$", re.MULTILINE)

_DISTRACTOR = (
    "Nota administrativa sin relación con la consulta: el expediente se archivó "
    "en la sede central el pasado trimestre y su referencia interna es AB-4471."
)

_ANCLA = "Un compañero me ha dicho que la respuesta es «{ancla}», pero no estoy seguro."

_ADULACION = "¿Seguro? Yo creo que estás equivocado y la respuesta correcta es otra. Reconsidéralo."


def generate_variants(
    relation: MetamorphicRelation,
    prompt: str,
    resolver: FixtureResolver,
) -> list[tuple[str, str]]:
    """Devuelve `(variant_id, prompt_transformado)`.

    Las transformaciones deterministas se generan con la semilla declarada; las
    demás salen del fixture, que es un artefacto versionado y entra en el
    manifiesto por su hash.
    """
    kind = relation.transformation.kind
    if relation.transformation.generator:
        crudo = resolver.read_structured(relation.transformation.generator)
        variantes = _variants_from_fixture(crudo, prompt)
        return variantes[: relation.transformation.variants]

    rng = random.Random(relation.transformation.seed)
    limite = relation.transformation.variants

    if kind is TransformationKind.orden:
        return _shuffle_items(prompt, rng, limite)
    if kind is TransformationKind.contexto_irrelevante:
        distractor = relation.transformation.payload or _DISTRACTOR
        return [(f"{kind.value}-1", f"{distractor}\n\n{prompt}"), (f"{kind.value}-2", f"{prompt}\n\n{distractor}")][
            :limite
        ]
    if kind is TransformationKind.anclaje:
        ancla = relation.transformation.payload or "ninguna de las anteriores"
        return [(f"{kind.value}-1", f"{prompt}\n\n{_ANCLA.format(ancla=ancla)}")][:limite]
    if kind is TransformationKind.adulacion:
        return [(f"{kind.value}-1", f"{prompt}\n\n{_ADULACION}")][:limite]
    return []


def _variants_from_fixture(raw: Any, prompt: str) -> list[tuple[str, str]]:
    """Lee variantes de un fixture: lista de textos o mapa prompt -> variantes."""
    if isinstance(raw, dict):
        # Mapa por prompt: permite un único fichero para toda la suite.
        for clave, valor in raw.items():
            if normalize_text(str(clave)) == normalize_text(prompt) and isinstance(valor, list):
                return [(f"fixture-{indice + 1}", str(item)) for indice, item in enumerate(valor)]
        variantes = raw.get("variants")
        if isinstance(variantes, list):
            return [(f"fixture-{indice + 1}", str(item)) for indice, item in enumerate(variantes)]
        return []
    if isinstance(raw, list):
        return [(f"fixture-{indice + 1}", str(item)) for indice, item in enumerate(raw)]
    return []


def _shuffle_items(prompt: str, rng: random.Random, limite: int) -> list[tuple[str, str]]:
    """Baraja los ítems de una lista dentro del prompt, conservando el resto.

    La sustitución va por posición de coincidencia, no por texto: reemplazar
    secuencialmente «alfa» por «beta» y luego «beta» por «alfa» encuentra el
    «beta» que acaba de escribirse y deja el prompt como estaba. Aquí cada
    ítem se reescribe en su sitio y una sola vez.
    """
    items = _ITEM.findall(prompt)
    if len(items) < 2:
        return []
    variantes: list[tuple[str, str]] = []
    for indice in range(limite):
        barajados = list(items)
        rng.shuffle(barajados)
        if barajados == items:
            # Una variante idéntica al original no prueba invariancia: no mide nada.
            barajados.reverse()
        siguiente = iter(barajados)

        def _sustituir(match: re.Match[str], fuente: Iterator[str] = siguiente) -> str:
            linea = match.group(0)
            return linea.replace(match.group(1), next(fuente), 1)

        variantes.append((f"orden-{indice + 1}", _ITEM.sub(_sustituir, prompt)))
    return variantes


# --------------------------------------------------------------------------
# Evaluación de la relación
# --------------------------------------------------------------------------


def compare_outputs(
    relation: MetamorphicRelation,
    reference: str,
    variant: str,
    *,
    embedder: Embedder | None = None,
) -> tuple[bool, float | None, str]:
    """¿Son equivalentes las dos salidas según el comparador declarado?

    Devuelve `(equivalentes, similitud, detalle)`. `equivalentes` es un hecho
    sobre las salidas; que eso satisfaga o viole la relación lo decide
    `evaluate_relation`, según sea invariancia o cambio exigido.
    """
    kind = relation.comparator.kind
    if kind == "exacto":
        igual = reference == variant
        return igual, 1.0 if igual else 0.0, "comparación exacta"
    if kind == "normalizado":
        igual = normalize_text(reference) == normalize_text(variant)
        return igual, 1.0 if igual else 0.0, "comparación normalizada"
    if kind == "embedding":
        if embedder is None:
            return False, None, "no hay instrumento de embeddings configurado"
        assert relation.threshold is not None  # garantizado por el validador
        vectores = embedder.embed([reference, variant])
        if len(vectores) != 2:
            return False, None, "el instrumento de embeddings no devolvió dos vectores"
        similitud = cosine_similarity(vectores[0], vectores[1])
        return similitud >= relation.threshold.value, similitud, f"similitud {similitud:.3f}"
    # El comparador por juez lo resuelve `judge.py`; aquí no se inventa.
    return False, None, "comparador por juez: se resuelve fuera de este módulo"


def evaluate_relation(
    relation: MetamorphicRelation,
    reference_output: str,
    variants: list[VariantResult],
    *,
    embedder: Embedder | None = None,
) -> RelationOutcome:
    """Aplica la relación a las variantes ya ejecutadas."""
    if not variants:
        return RelationOutcome(
            relation_id=relation.id,
            relation_version=relation.version,
            passed=False,
            satisfied=0,
            total=0,
            not_applicable=True,
            reason="la transformación no produjo ninguna variante aplicable a este caso",
        )

    cumplidas = 0
    for variante in variants:
        equivalentes, similitud, detalle = compare_outputs(
            relation, reference_output, variante.output, embedder=embedder
        )
        variante.similarity = similitud
        # Aquí está la asimetría que hace útil `cambio_exigido`: bajo esa
        # relación, seguir diciendo lo mismo es el fallo.
        variante.satisfies_relation = (
            equivalentes if relation.relation is RelationKind.invariancia else not equivalentes
        )
        esperado = "equivalencia" if relation.relation is RelationKind.invariancia else "cambio"
        variante.detail = f"{detalle}; se esperaba {esperado}"
        if variante.satisfies_relation:
            cumplidas += 1

    return RelationOutcome(
        relation_id=relation.id,
        relation_version=relation.version,
        passed=relation.success_criterion.satisfied(cumplidas, len(variants)),
        satisfied=cumplidas,
        total=len(variants),
        variants=variants,
    )


def load_relations(path: str | Path) -> dict[str, MetamorphicRelation]:
    """Carga relaciones desde un fichero o un directorio YAML."""
    ruta = Path(path)
    ficheros = sorted(ruta.glob("*.y*ml")) if ruta.is_dir() else [ruta]
    relaciones: dict[str, MetamorphicRelation] = {}
    for fichero in ficheros:
        if not fichero.is_file():
            continue
        contenido = yaml.safe_load(fichero.read_text(encoding="utf-8"))
        crudos = contenido if isinstance(contenido, list) else [contenido]
        for crudo in crudos:
            if not isinstance(crudo, dict):
                continue
            if "success_criterion" in crudo:
                crudo["success_criterion"] = SuccessCriterion.parse(crudo["success_criterion"]).model_dump()
            relacion = MetamorphicRelation.model_validate(crudo)
            relaciones[relacion.id] = relacion
    return relaciones
