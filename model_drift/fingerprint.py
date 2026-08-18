"""Huella de ejecución: qué configuración exacta produjo una salida.

Un nombre de modelo no identifica una ejecución. La misma red con los mismos
pesos responde distinto si cambia la plantilla de chat, el tokenizador o el
runtime que la ejecuta. Aquí se representa la **configuración efectiva
completa** en tres capas de fiabilidad decreciente (plan §10.1):

- **A. Identidad declarada** — lo que se pidió (proveedor, modelo, parámetros).
- **B. Identidad técnica observable** — lo que se puede comprobar del entorno
  (digest, cuantización, tokenizador, plantilla de chat, runtime).
- **C. Identidad remota** — lo que el proveedor dice de sí mismo
  (`system_fingerprint`, snapshot).

Regla que atraviesa el módulo: **nunca se rellena un hueco por inferencia**. Un
componente que no se puede obtener vale `no_disponible`. Una huella incompleta
es un resultado legítimo; una huella inventada, no.

Por qué la comparación NO se hace por hash: si un proveedor empieza a informar
un componente que antes callaba, el hash cambia sin que haya cambiado nada del
modelo. Comparar componente a componente evita esa falsa deriva; ver
`compare_fingerprints`.
"""
from __future__ import annotations

import hashlib
import json
from enum import Enum

from pydantic import BaseModel, Field


class ComponentState(str, Enum):
    """De dónde salió el valor de un componente de la huella."""

    #: Obtenido del entorno de ejecución y contrastado.
    verificado = "verificado"
    #: Dicho por el cliente o la configuración, sin comprobar.
    declarado = "declarado"
    #: No se pudo obtener. No es un error: es información sobre el límite de lo
    #: que se puede afirmar de este proveedor.
    no_disponible = "no_disponible"


class IdentityLayer(str, Enum):
    """Capa de fiabilidad a la que pertenece un componente."""

    declarada = "A"
    observable = "B"
    remota = "C"


class FingerprintComponent(BaseModel):
    """Un componente de la huella, con su procedencia siempre pegada al valor."""

    name: str
    value: str | None = None
    state: ComponentState
    layer: IdentityLayer

    @property
    def is_present(self) -> bool:
        return self.state is not ComponentState.no_disponible and self.value is not None


#: Componentes conocidos y la capa a la que pertenecen. La lista es abierta:
#: un proveedor puede aportar componentes propios y se aceptan como capa C.
KNOWN_COMPONENTS: dict[str, IdentityLayer] = {
    # Capa A — lo que se pidió.
    "provider": IdentityLayer.declarada,
    "deployment": IdentityLayer.declarada,
    "model": IdentityLayer.declarada,
    "requested_params": IdentityLayer.declarada,
    "system_prompt": IdentityLayer.declarada,
    "tool_definitions": IdentityLayer.declarada,
    # Capa B — lo que se puede comprobar del entorno.
    "digest": IdentityLayer.observable,
    "quantization": IdentityLayer.observable,
    "parameter_size": IdentityLayer.observable,
    "family": IdentityLayer.observable,
    "tokenizer": IdentityLayer.observable,
    "chat_template": IdentityLayer.observable,
    "runtime": IdentityLayer.observable,
    "runtime_version": IdentityLayer.observable,
    "context_window": IdentityLayer.observable,
    "applied_params": IdentityLayer.observable,
    # Capa C — lo que dice el proveedor de sí mismo.
    "system_fingerprint": IdentityLayer.remota,
    "model_snapshot": IdentityLayer.remota,
    "remote_model": IdentityLayer.remota,
    "api_version": IdentityLayer.remota,
}


class ExecutionFingerprint(BaseModel):
    """Identidad de una ejecución concreta.

    `canonical_hash` sirve para indexar y mostrar, **no para comparar**: dos
    huellas se comparan componente a componente con `compare_fingerprints`.
    """

    components: dict[str, FingerprintComponent] = Field(default_factory=dict)

    @property
    def canonical_hash(self) -> str:
        """Hash estable sobre los componentes presentes.

        Serialización con orden de claves fijo para que dos arranques con la
        misma configuración produzcan el mismo valor.
        """
        present = {name: comp.value for name, comp in sorted(self.components.items()) if comp.is_present}
        payload = json.dumps(present, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def short_hash(self) -> str:
        return self.canonical_hash[:12]

    def get(self, name: str) -> str | None:
        component = self.components.get(name)
        return component.value if component is not None and component.is_present else None

    def present_names(self) -> set[str]:
        return {name for name, comp in self.components.items() if comp.is_present}

    def missing_names(self) -> set[str]:
        return {name for name, comp in self.components.items() if not comp.is_present}

    def layer_coverage(self) -> dict[IdentityLayer, int]:
        """Cuántos componentes presentes hay por capa. Alimenta el manifiesto."""
        coverage = {layer: 0 for layer in IdentityLayer}
        for component in self.components.values():
            if component.is_present:
                coverage[component.layer] += 1
        return coverage

    def describe(self) -> str:
        """Descripción corta para informes: modelo + hash + capas cubiertas."""
        model = self.get("model") or "?"
        coverage = self.layer_coverage()
        capas = "".join(layer.value for layer in IdentityLayer if coverage[layer] > 0)
        return f"{model}@{self.short_hash} (capas {capas or '-'})"


class FingerprintVerdict(str, Enum):
    """Resultado de comparar dos huellas."""

    #: Todos los componentes comparables coinciden y ninguno falta en un lado.
    identica = "identica"
    #: Los comparables coinciden, pero uno de los lados calla componentes que
    #: el otro sí aporta. No se puede afirmar identidad, solo compatibilidad.
    equivalente_parcial = "equivalente_parcial"
    #: Algún componente comparable difiere: son ejecuciones distintas.
    divergente = "divergente"


class FingerprintComparison(BaseModel):
    """Comparación componente a componente de dos huellas.

    `differing` es lo que importa: si trae `chat_template` y no `digest`, la
    ejecución cambió sin que cambiaran los pesos.
    """

    verdict: FingerprintVerdict
    equal: list[str] = Field(default_factory=list)
    differing: list[str] = Field(default_factory=list)
    only_in_left: list[str] = Field(default_factory=list)
    only_in_right: list[str] = Field(default_factory=list)

    def explain(self) -> str:
        if self.verdict is FingerprintVerdict.identica:
            return "Huellas idénticas en todos los componentes disponibles."
        if self.verdict is FingerprintVerdict.divergente:
            return "Difieren en: " + ", ".join(sorted(self.differing)) + "."
        faltan = sorted(self.only_in_left + self.only_in_right)
        return (
            "Los componentes comparables coinciden, pero no se pudo comparar: "
            + ", ".join(faltan)
            + ". No hay evidencia de identidad completa."
        )


def compare_fingerprints(
    left: ExecutionFingerprint,
    right: ExecutionFingerprint,
    *,
    ignore: frozenset[str] | set[str] | None = None,
) -> FingerprintComparison:
    """Compara dos huellas componente a componente.

    `ignore` excluye los componentes que son **la variable del experimento**:
    al comparar dos modelos distintos, `model` y `digest` van a diferir por
    diseño y no deben contaminar el veredicto.
    """
    skip = set(ignore or ())
    left_present = left.present_names() - skip
    right_present = right.present_names() - skip
    shared = left_present & right_present

    equal = sorted(name for name in shared if left.get(name) == right.get(name))
    differing = sorted(name for name in shared if left.get(name) != right.get(name))
    only_left = sorted(left_present - right_present)
    only_right = sorted(right_present - left_present)

    if differing:
        verdict = FingerprintVerdict.divergente
    elif only_left or only_right:
        verdict = FingerprintVerdict.equivalente_parcial
    else:
        verdict = FingerprintVerdict.identica

    return FingerprintComparison(
        verdict=verdict,
        equal=equal,
        differing=differing,
        only_in_left=only_left,
        only_in_right=only_right,
    )


class FingerprintBuilder:
    """Construye una huella declarando la procedencia de cada componente.

    Se usa así, y la procedencia nunca es opcional::

        builder = FingerprintBuilder()
        builder.declared("model", "qwen3:8b")
        builder.verified("digest", "sha256:abc...")
        builder.unavailable("chat_template", layer=IdentityLayer.observable)
        huella = builder.build()
    """

    def __init__(self) -> None:
        self._components: dict[str, FingerprintComponent] = {}

    def _layer_for(self, name: str, layer: IdentityLayer | None) -> IdentityLayer:
        if layer is not None:
            return layer
        # Un componente desconocido solo puede venir del proveedor: capa C.
        return KNOWN_COMPONENTS.get(name, IdentityLayer.remota)

    def _add(self, name: str, value: str | None, state: ComponentState, layer: IdentityLayer | None) -> None:
        self._components[name] = FingerprintComponent(
            name=name,
            value=value,
            state=state,
            layer=self._layer_for(name, layer),
        )

    def declared(self, name: str, value: str | None, *, layer: IdentityLayer | None = None) -> FingerprintBuilder:
        """Valor dicho por el cliente o la configuración, sin comprobar."""
        if value is None:
            return self.unavailable(name, layer=layer)
        self._add(name, value, ComponentState.declarado, layer)
        return self

    def verified(self, name: str, value: str | None, *, layer: IdentityLayer | None = None) -> FingerprintBuilder:
        """Valor obtenido del entorno de ejecución.

        Un `None` no se convierte en verificado: se registra como no disponible,
        que es la verdad.
        """
        if value is None:
            return self.unavailable(name, layer=layer)
        self._add(name, value, ComponentState.verificado, layer)
        return self

    def unavailable(self, name: str, *, layer: IdentityLayer | None = None) -> FingerprintBuilder:
        """Componente que no se pudo obtener. Se registra, no se omite."""
        self._add(name, None, ComponentState.no_disponible, layer)
        return self

    def hashed(self, name: str, raw: str | None, *, layer: IdentityLayer | None = None) -> FingerprintBuilder:
        """Componente voluminoso (plantilla de chat, esquemas) guardado por hash.

        Guardar la plantilla entera en cada invocación multiplicaría la base de
        datos sin aportar nada: lo que importa es si cambió.
        """
        if raw is None:
            return self.unavailable(name, layer=layer)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        self._add(name, f"sha256:{digest}", ComponentState.verificado, layer)
        return self

    def has(self, name: str) -> bool:
        """Si ese componente ya se declaró, en cualquier estado."""
        return name in self._components

    def build(self) -> ExecutionFingerprint:
        return ExecutionFingerprint(components=dict(self._components))
