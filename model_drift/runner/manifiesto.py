"""Construcción y sellado del manifiesto de la tirada.

Regla 3 del motor: el manifiesto declara lo que faltó. Las cuatro banderas
se cierran con lo que REALMENTE ocurrió —determinismo medido, semilla
aplicada o no— y no con lo que se pidió al arrancar.
"""
from __future__ import annotations

from model_drift import __version__
from model_drift.broker_contracts import ModelReference, SeedStatus
from model_drift.manifest import (
    Determinism,
    ExactReproducibility,
    InstrumentRef,
    ModelUnderTest,
    ReproducibilityFlags,
    RunManifest,
)
from model_drift.runner.invocacion import InvocacionMixin


class ManifiestoMixin(InvocacionMixin):
    """Lo que la tirada declara de sí misma."""

    def _instruments(self) -> list[InstrumentRef]:
        instrumentos = [
            InstrumentRef(id="model_drift", kind="app", version=__version__),
            InstrumentRef(id="suite", kind="suite", version=self.suite.version, hash=self.suite.hash()),
        ]
        for relacion in sorted(self.relations.values(), key=lambda item: item.id):
            instrumentos.append(
                InstrumentRef(
                    id=f"mr:{relacion.id}",
                    kind="transformacion",
                    version=relacion.version,
                    detail=relacion.transformation.kind.value,
                )
            )
        if self.embedder is not None:
            instrumentos.append(
                InstrumentRef(id="embedder", kind="embedding", version=str(self.embedder.identity))
            )
        return instrumentos

    def _build_manifest(self, run_id: str, models: list[ModelReference]) -> RunManifest:
        manifiesto = RunManifest(
            run_id=run_id,
            broker_base_url=self.client.config.base_url,
            suite_id=self.suite.id,
            suite_version=self.suite.version,
            suite_hash=self.suite.hash(),
            suite_path=str(self.suite.root) if self.suite.root else None,
            repetitions=self.config.repetitions,
            requested_params={k: v for k, v in self.config.params().items() if v is not None},
            instruments=self._instruments(),
        )
        for modelo in models:
            manifiesto.models.append(
                ModelUnderTest(
                    key=modelo.key,
                    provider=modelo.provider,
                    deployment=modelo.deployment,
                    model=modelo.model,
                    fingerprint=self.client.fingerprint_for(modelo),
                )
            )
        manifiesto.reproducibility = ReproducibilityFlags(
            # Provisional: si el Broker rechaza la semilla, el primer 422 lo
            # descubre y `_seal_reproducibility` corrige la bandera al cerrar la
            # tirada. Declararlo aquí como definitivo afirmaría un control que
            # todavía no se ha ejercido.
            seed_fijada=self.config.seed is not None and self.client.config.supports_seed,
            determinismo_solicitado=self.config.seed is not None and self.config.temperature in (0, 0.0),
            determinismo_observado=Determinism.no_evaluado,
            # Nace en `no_demostrada` y solo una reproducción real la mueve.
            reproducibilidad_exacta=ExactReproducibility.no_demostrada,
        )
        return manifiesto

    def _seal_reproducibility(
        self,
        manifiesto: RunManifest,
        deterministas: list[bool],
        estados_semilla: list[SeedStatus] | None = None,
    ) -> None:
        """Cierra las cuatro banderas con lo que REALMENTE ocurrió.

        Se ejecuta al final a propósito: si la semilla se envió y el Broker la
        rechazó, la tirada no tuvo control de determinismo por mucho que se
        pidiera, y el manifiesto tiene que decir eso y no lo que se intentó.

        Desde el contrato 2.9 la fuente es `seed_status` de cada invocación, no
        lo que se pidió: `sent` significa que el Broker la incluyó en la llamada
        al proveedor; `unsupported`, que ese endpoint no tiene por dónde
        llevarla. Pedirla y darla por buena sería exactamente el tipo de
        garantía inventada que este proyecto existe para no cometer.
        """
        flags = manifiesto.reproducibility
        pedida = self.config.seed is not None and self.client.config.supports_seed
        if estados_semilla:
            flags.seed_fijada = all(estado is SeedStatus.sent for estado in estados_semilla)
            if not flags.seed_fijada:
                manifiesto.declare_missing(
                    "control de determinismo: el Broker no envió la semilla en todas las "
                    "invocaciones (seed_status != sent). La tirada NO es reproducible."
                )
        else:
            flags.seed_fijada = pedida
        if deterministas:
            flags.determinismo_observado = Determinism.si if all(deterministas) else Determinism.no
        if not flags.seed_fijada:
            # Sin control de semilla no hay reproducibilidad exacta que perseguir
            # en este entorno: no es que no se haya demostrado, es que no está
            # disponible.
            flags.reproducibilidad_exacta = ExactReproducibility.no_disponible
