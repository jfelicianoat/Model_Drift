"""Huella de ejecución: la que publica el Broker, si la publica.

No se recalcula aquí nada que el Broker ya haya resuelto: él es quien sabe
qué pudo verificar contra el runtime, y adivinarlo desde fuera sería peor.
Lo que no venga se anota como componente ausente, no como identidad
comprobada.
"""
from __future__ import annotations

from typing import Any

from model_drift.broker_client.catalogo import CatalogoMixin
from model_drift.broker_client.tipos import InvocationOutcome
from model_drift.broker_contracts import (
    BrokerError,
    ModelReference,
)
from model_drift.fingerprint import (
    ComponentState,
    ExecutionFingerprint,
    FingerprintBuilder,
    IdentityLayer,
)


class HuellaMixin(CatalogoMixin):
    """Resolución de la huella de ejecución de un modelo."""

    def fingerprint_for(self, target: ModelReference, outcome: InvocationOutcome | None = None) -> ExecutionFingerprint:
        """Huella de ejecución: la que publica el Broker, si la publica.

        Desde el Cambio 2 el catálogo trae `execution_fingerprint` con las tres
        capas ya resueltas y el estado de cada componente. Se adopta tal cual:
        el Broker es quien sabe qué pudo verificar contra el runtime, y
        recalcularlo aquí sería adivinar peor.

        Si el Broker no la publica —versión anterior, o modelo ausente del
        catálogo—, se construye una huella de solo capa A y se **declara la
        garantía perdida**: identidad declarada no es identidad verificada.
        """
        builder = FingerprintBuilder()
        entrada = None
        try:
            entrada = next((item for item in self.catalog() if item.key == target.key), None)
        except BrokerError:
            entrada = None

        publicada = (entrada.execution_fingerprint if entrada else {}) or {}
        componentes = publicada.get("components") if isinstance(publicada, dict) else None

        if isinstance(componentes, dict) and componentes:
            for nombre, crudo in componentes.items():
                if not isinstance(crudo, dict):
                    continue
                valor = crudo.get("value")
                estado = str(crudo.get("status") or "")
                texto = None if valor is None else str(valor)
                if estado == ComponentState.verificado.value:
                    builder.verified(nombre, texto)
                elif estado == ComponentState.declarado.value:
                    builder.declared(nombre, texto)
                else:
                    builder.unavailable(nombre)
        else:
            builder.declared("provider", target.provider)
            builder.declared("deployment", target.deployment)
            builder.declared("model", target.model)
            for nombre in ("digest", "quantization", "parameter_size", "tokenizer", "chat_template", "runtime"):
                builder.unavailable(nombre, layer=IdentityLayer.observable)
            builder.unavailable("system_fingerprint", layer=IdentityLayer.remota)
            self._declare_missing(
                f"identidad técnica de {target.key}: el Broker no publica execution_fingerprint "
                "(Cambio 2 del contrato de evaluación). La identidad es solo declarada: no se puede "
                "afirmar que dos tiradas usaron el mismo binario ni la misma plantilla."
            )

        # Los parámetros pedidos son capa A y los aporta esta invocación, no el
        # catálogo: dos tiradas del mismo modelo con temperaturas distintas son
        # ejecuciones distintas.
        if outcome is not None and outcome.applied_params:
            builder.declared("requested_params", _canonical_params(outcome.applied_params))

        huella = builder.build()
        ausentes = sorted(huella.missing_names() & {"digest", "chat_template", "tokenizer", "runtime"})
        if ausentes:
            self._declare_missing(
                f"identidad técnica de {target.key}: el Broker no pudo verificar "
                + ", ".join(ausentes)
                + ". Para esos componentes la identidad no está comprobada."
            )
        return huella


def _huella_publicada(bloque: dict[str, Any]) -> ExecutionFingerprint:
    """Convierte la huella que publica el Broker a la del dominio.

    Se adopta tal cual, con el estado de cada componente: el Broker es quien
    sabe qué pudo verificar contra el runtime.
    """
    builder = FingerprintBuilder()
    componentes = bloque.get("components") if isinstance(bloque, dict) else None
    if not isinstance(componentes, dict):
        return builder.build()
    for nombre, crudo in componentes.items():
        if not isinstance(crudo, dict):
            continue
        valor = crudo.get("value")
        texto = None if valor is None else str(valor)
        estado = str(crudo.get("status") or "")
        if estado == ComponentState.verificado.value:
            builder.verified(nombre, texto)
        elif estado == ComponentState.declarado.value:
            builder.declared(nombre, texto)
        else:
            builder.unavailable(nombre)
    return builder.build()


def _canonical_params(params: dict[str, Any]) -> str:
    return ";".join(f"{clave}={params[clave]}" for clave in sorted(params))


def _first_str(features: dict[str, Any], name: str) -> str | None:
    for grupo in features.values():
        if isinstance(grupo, dict) and isinstance(grupo.get(name), str):
            return str(grupo[name])
    return None
