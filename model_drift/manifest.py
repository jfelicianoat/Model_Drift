"""Manifiesto de reproducibilidad y regla de comparabilidad.

    El evaluador forma parte del instrumento de medida.

Una modificación en el evaluador tiene que poder distinguirse de una
modificación en el modelo evaluado. Si entre marzo y julio cambió el esquema de
un fixture, el modelo de embeddings o la versión de una transformación
metamórfica, la diferencia observada **no es atribuible al modelo**, y
atribuírsela sería cometer el error que este proyecto existe para detectar.

De ahí las dos piezas de este módulo:

- `RunManifest` — lo que hubo: versiones, hashes, huellas y, sobre todo, qué
  garantías se pudieron verificar y cuáles no.
- `compare_manifests` — el cotejo que se ejecuta **antes** de cualquier
  conclusión. Si el instrumento cambió, los casos afectados salen
  `indeterminado` y el informe lo enseña en cabecera.

Sobre la semilla, la sección §10.2 del plan en una frase: fijar una semilla es
una **condición de control**, no una garantía de determinismo. Hardware,
kernels, paralelismo, orden de reducción en coma flotante o cambios del
proveedor pueden romperlo con la misma semilla. Por eso hay cuatro banderas
independientes y `reproducibilidad_exacta` nace en `no_demostrada`.
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from model_drift import __version__
from model_drift.fingerprint import ExecutionFingerprint, FingerprintVerdict, compare_fingerprints


class Determinism(str, Enum):
    """¿Se observó determinismo? Se mide, no se supone."""

    si = "si"
    no = "no"
    no_evaluado = "no_evaluado"


class ExactReproducibility(str, Enum):
    """¿Se demostró reproducibilidad exacta?

    `demostrada` exige haber reproducido una tirada anterior completa con la
    misma huella y haber obtenido las mismas salidas. Nace en `no_demostrada`
    y solo el hecho la mueve.
    """

    demostrada = "demostrada"
    no_demostrada = "no_demostrada"
    no_disponible = "no_disponible"


class ReproducibilityFlags(BaseModel):
    """Las cuatro banderas que la revisión 1 del plan confundía en una."""

    seed_fijada: bool = False
    determinismo_solicitado: bool = False
    determinismo_observado: Determinism = Determinism.no_evaluado
    reproducibilidad_exacta: ExactReproducibility = ExactReproducibility.no_demostrada

    def declare(self) -> list[str]:
        """Frases para el informe. Nunca afirman más de lo comprobado."""
        lineas = [
            f"Semilla fijada: {'sí' if self.seed_fijada else 'no'}.",
            f"Determinismo solicitado: {'sí' if self.determinismo_solicitado else 'no'}.",
        ]
        if self.determinismo_observado is Determinism.si:
            lineas.append("Determinismo observado: sí (las repeticiones idénticas produjeron la misma salida).")
        elif self.determinismo_observado is Determinism.no:
            lineas.append(
                "Determinismo observado: NO. Misma semilla, salidas distintas: "
                "es lo normal y no invalida la medición."
            )
        else:
            lineas.append("Determinismo observado: no evaluado.")
        if self.reproducibilidad_exacta is ExactReproducibility.demostrada:
            lineas.append("Reproducibilidad exacta: demostrada contra una tirada anterior.")
        elif self.reproducibilidad_exacta is ExactReproducibility.no_disponible:
            lineas.append("Reproducibilidad exacta: no disponible en este entorno.")
        else:
            lineas.append("Reproducibilidad exacta: NO demostrada. No se ha reproducido ninguna tirada anterior.")
        return lineas


class InstrumentRef(BaseModel):
    """Un instrumento de medida, identificado por versión y hash.

    `kind` distingue las clases que el plan §11 enumera; `identity` es lo que
    se compara entre tiradas.
    """

    id: str
    kind: str
    version: str = "1"
    hash: str | None = None
    detail: str = ""

    @property
    def identity(self) -> str:
        return f"{self.kind}:{self.id}@{self.version}:{self.hash or '-'}"


class ModelUnderTest(BaseModel):
    """Un modelo de la tirada, con su identidad declarada y su huella."""

    key: str
    provider: str
    deployment: str
    model: str
    fingerprint: ExecutionFingerprint = Field(default_factory=ExecutionFingerprint)

    @property
    def declared_identity(self) -> str:
        return f"{self.provider}/{self.deployment}/{self.model}"


class BrokerContractFlags(BaseModel):
    """Lo que el Broker declaró poder demostrar (Client_API.md, 12).

    Cada bandera se copia de `/api/v1/capabilities` tal cual, sin deducirla de
    la versión: un operador puede tener apagado el sondeo en sombra en un
    Broker 2.10, y la versión sola no lo diría.
    """

    #: `contract_version` que anunció el Broker. `None` = no se pudo leer.
    version: str | None = None
    #: 8.1: `role`/`status` enumerados y `contractual` por invocación. Sin esto,
    #: separar el trabajo propio del Broker del que se pidió es una heurística.
    invocation_contract: bool = False
    #: 8.5: cada invocación declara la poda que se le aplicó de verdad.
    prompt_compression_echo: bool = False
    #: 8.3: el entregable viene marcado con `final: true` en /artifacts.
    canonical_artifacts: bool = False
    #: 8.4: si este Broker sondea otros modelos con el contenido de la tarea.
    auxiliary_invocations: bool = True
    #: 8.4: si acepta apagarlo con `auxiliary_invocations: false`.
    auxiliary_invocations_optout: bool = False

    @property
    def demonstrable_execution(self) -> bool:
        """Si esta tirada puede probar CÓMO se ejecutó, no solo qué salió.

        Es el conjunto que trae el contrato 2.10: sin él una medición sigue
        dando números, pero no puede enseñar que el prompt llegó íntegro ni que
        la huella que atribuye al modelo es la de la llamada que respondió.
        """
        return self.invocation_contract and self.prompt_compression_echo


class RunManifest(BaseModel):
    """Todo lo que hizo falta para producir una tirada, y qué se pudo verificar."""

    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_drift_version: str = __version__
    broker_version: str | None = None
    broker_base_url: str | None = None

    suite_id: str = ""
    suite_version: str = "1"
    suite_hash: str = ""
    #: Dónde estaba la suite al ejecutar. Se guarda para que una comparación
    #: pueda volver a cargarla sola y saber qué evidencia tiene cada caso: sin
    #: la suite, todo se quedaría en `sin_direccion` por falta de información,
    #: no por falta de evidencia real. Quien compara desde la ventana no tiene
    #: por qué acordarse de dónde estaba.
    suite_path: str | None = None
    policy_hash: str | None = None

    models: list[ModelUnderTest] = Field(default_factory=list)
    instruments: list[InstrumentRef] = Field(default_factory=list)
    fixtures: dict[str, str] = Field(default_factory=dict)

    repetitions: int = 1
    requested_params: dict[str, Any] = Field(default_factory=dict)
    reproducibility: ReproducibilityFlags = Field(default_factory=ReproducibilityFlags)

    #: Qué prometía el Broker EN MARCHA cuando se ejecutó la tirada, no lo que
    #: dice su documentación. Se sella al arrancar, con `probe_contract`, y es
    #: lo que permite leer una tirada vieja sabiendo qué se pudo demostrar
    #: entonces: una misma suite contra un Broker 2.9 y contra uno 2.10 produce
    #: evidencia distinta aunque los números salgan iguales.
    broker_contract: BrokerContractFlags = Field(default_factory=lambda: BrokerContractFlags())

    #: Garantías que NO se pudieron obtener. Se escriben para que el informe
    #: las enseñe: una ausencia declarada es información, una ausencia
    #: silenciada es un engaño.
    missing_guarantees: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    def instrument_signature(self) -> str:
        """Firma del conjunto de instrumentos. Base de la comparabilidad."""
        partes = sorted(item.identity for item in self.instruments)
        partes += [f"fixture:{name}@{digest}" for name, digest in sorted(self.fixtures.items())]
        partes.append(f"suite:{self.suite_hash}")
        payload = "|".join(partes)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def model_by_key(self, key: str) -> ModelUnderTest | None:
        return next((item for item in self.models if item.key == key), None)

    def declare_missing(self, guarantee: str) -> None:
        if guarantee not in self.missing_guarantees:
            self.missing_guarantees.append(guarantee)

    def to_json(self) -> str:
        return json.dumps(json.loads(self.model_dump_json()), indent=2, ensure_ascii=False, sort_keys=True)


class ComparabilityIssue(BaseModel):
    """Un motivo concreto por el que dos tiradas no son comparables."""

    kind: str
    detail: str


class ComparabilityReport(BaseModel):
    """Resultado del cotejo de manifiestos. Se lee antes que nada."""

    comparable: bool
    issues: list[ComparabilityIssue] = Field(default_factory=list)
    fingerprint_notes: list[str] = Field(default_factory=list)

    @property
    def instrument_changed(self) -> bool:
        return any(issue.kind in {"instrumento", "suite", "fixture", "politica"} for issue in self.issues)

    @property
    def fingerprint_divergent(self) -> bool:
        return any(issue.kind == "huella" for issue in self.issues)

    def headline(self) -> str:
        if self.comparable and not self.fingerprint_notes:
            return "Tiradas comparables: mismo instrumento de evaluación."
        if not self.comparable:
            return "NO comparables: " + "; ".join(f"{i.kind}: {i.detail}" for i in self.issues)
        return "Comparables con reservas: " + "; ".join(self.fingerprint_notes)


def compare_manifests(
    baseline: RunManifest,
    candidate: RunManifest,
    *,
    baseline_model_key: str | None = None,
    candidate_model_key: str | None = None,
) -> ComparabilityReport:
    """Coteja dos manifiestos antes de permitir ninguna conclusión.

    Los componentes de huella que **son la variable del experimento** —modelo,
    digest y familia cuando se comparan dos modelos distintos— se excluyen del
    veredicto: van a diferir por diseño. Lo que no puede diferir es todo lo
    demás, y en particular la plantilla de chat o el runtime: ahí es donde se
    esconde el «mismo model_id, distinta ejecución».
    """
    issues: list[ComparabilityIssue] = []

    if baseline.suite_hash != candidate.suite_hash:
        issues.append(
            ComparabilityIssue(
                kind="suite",
                detail=f"la suite cambió ({baseline.suite_hash[:12]} -> {candidate.suite_hash[:12]})",
            )
        )

    if baseline.policy_hash != candidate.policy_hash:
        issues.append(ComparabilityIssue(kind="politica", detail="la política de regresión cambió"))

    base_inst = {item.id: item for item in baseline.instruments}
    cand_inst = {item.id: item for item in candidate.instruments}
    for nombre in sorted(set(base_inst) & set(cand_inst)):
        if base_inst[nombre].identity != cand_inst[nombre].identity:
            issues.append(
                ComparabilityIssue(
                    kind="instrumento",
                    detail=f"{nombre}: {base_inst[nombre].identity} -> {cand_inst[nombre].identity}",
                )
            )
    for nombre in sorted(set(base_inst) ^ set(cand_inst)):
        issues.append(
            ComparabilityIssue(kind="instrumento", detail=f"{nombre} solo está presente en una de las dos tiradas")
        )

    for nombre in sorted(set(baseline.fixtures) & set(candidate.fixtures)):
        if baseline.fixtures[nombre] != candidate.fixtures[nombre]:
            issues.append(ComparabilityIssue(kind="fixture", detail=f"{nombre} cambió de contenido"))

    notas: list[str] = []
    base_model = baseline.model_by_key(baseline_model_key) if baseline_model_key else None
    cand_model = candidate.model_by_key(candidate_model_key) if candidate_model_key else None
    if base_model is not None and cand_model is not None:
        mismo_modelo = base_model.declared_identity == cand_model.declared_identity
        # Comparando dos modelos distintos, su identidad es la variable: se
        # excluye. Comparando el "mismo" modelo consigo mismo, no se excluye
        # nada, que es justo cuando interesa ver si cambió por debajo.
        variable_del_experimento = frozenset({"model", "digest", "family", "parameter_size", "quantization"})
        ignorar = frozenset() if mismo_modelo else variable_del_experimento
        comparacion = compare_fingerprints(base_model.fingerprint, cand_model.fingerprint, ignore=ignorar)
        if comparacion.verdict is FingerprintVerdict.divergente:
            issues.append(
                ComparabilityIssue(
                    kind="huella",
                    detail=f"la ejecución difiere en {', '.join(comparacion.differing)}",
                )
            )
        elif comparacion.verdict is FingerprintVerdict.equivalente_parcial:
            notas.append(comparacion.explain())

    return ComparabilityReport(comparable=not issues, issues=issues, fingerprint_notes=notas)
