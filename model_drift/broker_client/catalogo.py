"""Catálogo de modelos, capacidades y sondeo del contrato.

El catálogo se pide de una vez a `/models/availability`; la matriz de
features es por modelo y solo se busca cuando alguien la necesita, porque
con 163 entradas serían 163 peticiones para datos que casi nunca se usan.
"""
from __future__ import annotations

from typing import Any

from model_drift.broker_client.transporte import TransporteBroker
from model_drift.broker_contracts import (
    BrokerError,
    ModelCatalogEntry,
    ModelReference,
)


class CatalogoMixin(TransporteBroker):
    """Lectura del catálogo y de lo que el Broker dice saber hacer."""

    def capabilities(self) -> dict[str, Any]:
        payload = self._request("GET", "/api/v1/capabilities").json()
        return payload if isinstance(payload, dict) else {}

    def probe_contract(self) -> dict[str, bool]:
        """Comprueba QUÉ admite este Broker antes de dar nada por hecho.

        Las banderas de `BrokerConfig` nacen optimistas y se corrigen solas al
        primer 422, lo cual sirve para no romper una tirada pero llega tarde
        para avisar: cuando el aviso importa —«esto va a contaminar las
        estadísticas del router»— todavía no se ha enviado nada.

        Se pregunta a `/api/v1/capabilities`, que es donde el contrato dice que
        se anuncian, en vez de deducirlo del esquema de OpenAPI. Si el Broker no
        responde, se deja el optimismo y la red de seguridad del 422 sigue
        puesta, pero se declara que no se pudo comprobar.
        """
        try:
            capacidades = self.capabilities()
        except BrokerError:
            self._declare_missing(
                "no se pudieron leer las capacidades del Broker (/api/v1/capabilities): "
                "su contrato se descubrirá al vuelo, no antes de empezar"
            )
            return {
                "seed": self.config.supports_seed,
                "learning_exclusion": self.config.supports_learning_exclusion,
                "invocation_contract": self.supports_invocation_contract,
                "prompt_compression_echo": self.supports_prompt_compression_echo,
                "canonical_artifacts": self.supports_canonical_artifacts,
                "auxiliary_optout": self.supports_auxiliary_optout,
            }

        self.contract_version = str(capacidades.get("contract_version") or "") or None
        # `seed` no tiene bandera propia: llegó en 2.9 junto a las que sí la
        # tienen, así que su presencia se infiere de la versión anunciada.
        self.config.supports_seed = bool(capacidades.get("execution_fingerprint")) or (
            _version(self.contract_version) >= (2, 9)
        )
        self.config.supports_learning_exclusion = bool(capacidades.get("exclude_from_model_learning"))
        self.supports_invocation_telemetry = bool(capacidades.get("invocation_telemetry"))

        # --- Contrato 2.10 --------------------------------------------------
        # Todo aditivo: contra un Broker 2.9 estas banderas se quedan en falso y
        # la medición sigue funcionando, pero declarando qué no pudo demostrar.
        self.supports_invocation_contract = bool(capacidades.get("invocation_contract"))
        self.supports_prompt_compression_echo = bool(capacidades.get("prompt_compression_echo"))
        self.supports_task_artifacts = bool(capacidades.get("task_artifacts"))
        self.supports_canonical_artifacts = bool(capacidades.get("canonical_artifacts"))
        auxiliares = capacidades.get("auxiliary_invocations")
        # Ausente = Broker anterior al 2.10, que las hacía sin anunciarlas.
        self.performs_auxiliary_invocations = True if auxiliares is None else bool(auxiliares)
        self.supports_auxiliary_optout = bool(capacidades.get("auxiliary_invocations_optout"))
        self.config.supports_prompt_compression = (
            self.config.supports_prompt_compression and self._acepta_compresion(capacidades)
        )

        if not self.supports_invocation_contract:
            self._declare_missing(
                "separación del trabajo auxiliar: el Broker no declara `contractual` por "
                "invocación (contrato 2.10, §8.1). Las llamadas que el Broker hace por su "
                "cuenta bajo el mismo task_id solo se pueden distinguir por el nombre del rol."
            )
        if not self.supports_prompt_compression_echo:
            self._declare_missing(
                "prompt sin podar: el Broker no declara `prompt_compression.effective` "
                "(contrato 2.10, §8.5). Pedir `off` no tiene acuse de recibo, así que no se "
                "puede demostrar que el modelo vio el prompt íntegro."
            )
        if self.performs_auxiliary_invocations and not self.supports_auxiliary_optout:
            # No es una garantía perdida de la medición —el sondeo en sombra no
            # se factura ni altera el resultado— pero sí de confidencialidad: el
            # contenido lo ve un modelo que nadie eligió.
            self._declare_missing(
                "exclusividad de contenido: el Broker hace invocaciones auxiliares y no "
                "acepta apagarlas (contrato 2.10, §8.4). El prompt puede verlo un modelo "
                "distinto del medido, bajo el mismo task_id."
            )
        if not self.supports_canonical_artifacts:
            self._declare_missing(
                "entregable con contrato: el Broker no marca `final: true` en /artifacts "
                "(contrato 2.10, §8.3). La salida se lee de `result`, que no tiene esquema."
            )

        if not self.config.supports_seed:
            self._declare_missing(
                "control de determinismo: el Broker no admite generation.seed "
                "(Cambio 1 del contrato de evaluación). La tirada NO es reproducible."
            )
        if not self.config.supports_learning_exclusion:
            self._declare_missing(
                "aislamiento del aprendizaje: el Broker no admite exclude_from_model_learning "
                "(Cambio 4 del contrato). Esta tirada altera las estadísticas del router."
            )
        return {
            "seed": self.config.supports_seed,
            "learning_exclusion": self.config.supports_learning_exclusion,
            "invocation_contract": self.supports_invocation_contract,
            "prompt_compression_echo": self.supports_prompt_compression_echo,
            "canonical_artifacts": self.supports_canonical_artifacts,
            "auxiliary_optout": self.supports_auxiliary_optout,
        }

    @staticmethod
    def _acepta_compresion(capacidades: dict[str, Any]) -> bool:
        """Si tiene sentido pedir `prompt_compression` en la petición.

        `prompt_compression` es campo de primer nivel desde antes del 2.10 y no
        tiene bandera propia en `/capabilities`; lo que el 2.10 añade es el eco
        que permite comprobarlo. Se envía siempre que el Broker declare una
        versión de contrato conocida, y el 422 sigue siendo la red de seguridad
        para uno más antiguo.
        """
        version = str(capacidades.get("contract_version") or "")
        return bool(version)

    def catalog(self, *, refresh: bool = False) -> list[ModelCatalogEntry]:
        """Catálogo de modelos: una sola petición a `/models/availability`.

        La lista viene en **`items`** (el Broker devuelve
        `{checked_at, items, counts}`). Aquí NO se pide la matriz de features:
        `/models/context` es por modelo y con 163 entradas serían 163
        peticiones para datos que casi nunca se van a usar. Las features se
        piden bajo demanda en `features_for`, que es donde de verdad hacen
        falta: al decidir si un caso aplica a un modelo.
        """
        if self._catalog_cache is not None and not refresh:
            return self._catalog_cache
        if refresh:
            self._features_cache.clear()

        payload = self._request("GET", "/api/v1/models/availability").json()
        crudos: Any = payload
        if isinstance(payload, dict):
            # `items` es el contrato actual; `models` se acepta por si se
            # apunta a un Broker anterior a la entrega de evaluación.
            for clave in ("items", "models"):
                if isinstance(payload.get(clave), list):
                    crudos = payload[clave]
                    break
        if not isinstance(crudos, list):
            raise BrokerError(
                "el catálogo del Broker no trae ninguna lista de modelos "
                f"(claves recibidas: {sorted(payload) if isinstance(payload, dict) else type(payload).__name__}). "
                "Puede que el contrato haya cambiado: revisa broker_contracts.py."
            )

        entradas: list[ModelCatalogEntry] = []
        descartadas = 0
        for item in crudos:
            if not isinstance(item, dict):
                descartadas += 1
                continue
            try:
                entradas.append(ModelCatalogEntry.model_validate(item))
            except Exception:
                # Una entrada con forma inesperada se cuenta, no se traga: un
                # catálogo que encoge en silencio es peor que uno que falla.
                descartadas += 1
        if descartadas:
            self._declare_missing(
                f"{descartadas} entradas del catálogo no encajan en el contrato conocido y se han descartado"
            )
        if not entradas:
            raise BrokerError(
                f"el Broker devolvió {len(crudos)} entradas de catálogo y ninguna es utilizable"
            )

        self._catalog_cache = entradas
        return entradas

    def features_for(self, reference: ModelReference) -> dict[str, Any]:
        """Matriz de capacidades de UN modelo, con la procedencia del Broker.

        `/api/v1/models/context` exige proveedor, deployment y modelo como
        parámetros: no es un listado. Se cachea porque una suite pregunta por
        el mismo modelo una vez por caso.
        """
        if reference.key in self._features_cache:
            return self._features_cache[reference.key]
        try:
            payload = self._request(
                "GET",
                "/api/v1/models/context",
                params={
                    "provider": reference.provider,
                    "deployment": reference.deployment,
                    "model": reference.model,
                },
            ).json()
        except BrokerError as error:
            self._declare_missing(
                f"matriz de capacidades de {reference.key}: /api/v1/models/context respondió {error}. "
                "Las capacidades se derivan solo de la lista plana de availability."
            )
            self._features_cache[reference.key] = {}
            return {}
        features = payload.get("features") if isinstance(payload, dict) else None
        resultado = features if isinstance(features, dict) else {}
        self._features_cache[reference.key] = resultado
        return resultado

    def entry_for(self, reference: ModelReference) -> ModelCatalogEntry | None:
        """Entrada del catálogo, enriquecida con sus features la primera vez."""
        entrada = next((item for item in self.catalog() if item.key == reference.key), None)
        if entrada is not None and not entrada.features:
            entrada.features = self.features_for(reference)
        return entrada


def _version(value: str | None) -> tuple[int, ...]:
    """Versión de contrato como números: como texto, «2.12» < «2.9»."""
    try:
        return tuple(int(part) for part in (value or "").split("."))
    except ValueError:
        return ()
