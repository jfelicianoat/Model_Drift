"""Cliente HTTP del AI Broker.

Model_Drift no ejecuta inferencia: la pide. Este módulo es la única puerta al
Broker y concentra tres responsabilidades que en otros sitios se mezclan:

1. **Fijar el modelo de verdad.** Toda invocación va con `target_model` y
   `fallback_allowed = False`. Una comparación que permita sustitución no
   compara lo que dice comparar. Si el Broker sirviera otro modelo, se detecta
   y se marca la invocación como no utilizable.
2. **No contaminar la operación del Broker.** El tráfico va marcado en
   `content.metadata.origin` y con `exclude_from_model_learning`, para que las
   invocaciones de medición no muevan el score del router adaptativo ni acerquen
   un modelo a la cuarentena. Si el Broker no admite el marcador, se detecta al
   primer 422 y se **declara la garantía perdida**.
3. **Degradar con honestidad.** Lo mismo con `seed`/`top_p` y con la huella de
   ejecución: contra un Broker que los tiene, se usan; contra uno que no, se
   anota lo que se perdió en vez de fingir reproducibilidad o identidad.

El cliente está partido por lo que hace contra el Broker —transporte,
catálogo, invocación e identidad— y se recompone aquí. Quien lo usa importa
`BrokerClient` de este paquete y no necesita saber en qué módulo vive cada
método.
"""
from __future__ import annotations

from model_drift.broker_client.invocacion import InvocacionMixin
from model_drift.broker_client.semantica import System1Mixin
from model_drift.broker_client.tipos import ORIGIN, BrokerConfig, InvocationOutcome

__all__ = ["ORIGIN", "BrokerClient", "BrokerConfig", "InvocationOutcome"]


class BrokerClient(System1Mixin, InvocacionMixin):
    """Cliente síncrono. Una tirada es un lote largo, no una interacción."""
