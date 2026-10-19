"""Configuración de Model_Drift: fichero YAML más variables de entorno.

El token del Broker **no se escribe en el fichero de configuración**: se lee de
`MODEL_DRIFT_BROKER_TOKEN`. El Broker rota su token admin en cada arranque
salvo que se fije desde fuera, así que un token en disco sería a la vez un
secreto guardado sin necesidad y un valor caduco la mitad del tiempo.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from model_drift.ai_os_core import ENV_TOKEN as CORE_ENV_TOKEN
from model_drift.ai_os_core import ENV_URL as CORE_ENV_URL
from model_drift.ai_os_core import CoreClient
from model_drift.broker_client import BrokerConfig
from model_drift.broker_contracts import ModelReference
from model_drift.system1_contracts import System1Config

ENV_TOKEN = "MODEL_DRIFT_BROKER_TOKEN"
ENV_BASE_URL = "MODEL_DRIFT_BROKER_URL"
DEFAULT_PATH = Path("model_drift.yaml")


class AppConfig(BaseModel):
    """Configuración completa de la aplicación."""

    #: El Broker de trabajo vive en otra máquina de la LAN, no en loopback.
    #: Se cambia aquí, en model_drift.yaml, o con MODEL_DRIFT_BROKER_URL.
    broker_url: str = "http://192.168.1.52:8765"
    #: AI OS Core, el director del ecosistema. Vacío: la app funciona sin él.
    #: También MODEL_DRIFT_CORE_URL; el token solo por MODEL_DRIFT_CORE_TOKEN.
    core_url: str | None = None
    database: str = "state/model_drift.sqlite"
    reports_dir: str = "informes"
    suites_dir: str = "suites"
    relations_dir: str = "relaciones"

    repetitions: int = 3
    temperature: float | None = 0.0
    seed: int | None = 1234
    budget_usd: float | None = None
    timeout_seconds: int | None = 300
    poll_interval_seconds: float = 2.0

    #: Modelo de embeddings, si se quiere similitud semántica.
    embedding_model: str | None = None
    #: Modelo juez. Sin él, la escalera se queda en el escalón determinista.
    judge_model: str | None = None
    system1: System1Config = Field(default_factory=System1Config)

    #: Ejecutar código generado por un modelo exige decisión explícita.
    allow_code_execution: bool = False
    #: Confirmación de que se acepta contaminar las estadísticas del Broker
    #: mientras no exista el marcador de exclusión (Cambio 4 del contrato).
    accept_learning_contamination: bool = False

    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def broker_token(self) -> str | None:
        return os.environ.get(ENV_TOKEN) or None

    def broker_config(self) -> BrokerConfig:
        return BrokerConfig(
            base_url=os.environ.get(ENV_BASE_URL) or self.broker_url,
            admin_token=self.broker_token,
            poll_interval_seconds=self.poll_interval_seconds,
        )

    def core_client(self) -> CoreClient | None:
        url = os.environ.get(CORE_ENV_URL) or self.core_url
        return CoreClient(url, os.environ.get(CORE_ENV_TOKEN) or None) if url else None

    def model_reference(self, raw: str) -> ModelReference:
        """Convierte `proveedor/despliegue/modelo` en una referencia.

        Se exige la identidad completa: `preferred_model` no basta para
        distinguir dos modelos homónimos en despliegues distintos, y una
        comparación que se equivoque de destino no compara nada.
        """
        partes = raw.split("/")
        if len(partes) != 3 or not all(partes):
            raise ValueError(
                f"referencia de modelo inválida: {raw!r}. Formato: proveedor/despliegue/modelo "
                "(por ejemplo ollama/local/qwen3:8b)"
            )
        return ModelReference(provider=partes[0], deployment=partes[1], model=partes[2])

    @classmethod
    def load(cls, path: str | Path | None = None) -> AppConfig:
        ruta = Path(path) if path else DEFAULT_PATH
        if not ruta.is_file():
            return cls()
        crudo = yaml.safe_load(ruta.read_text(encoding="utf-8")) or {}
        if not isinstance(crudo, dict):
            raise ValueError(f"{ruta}: la raíz de la configuración debe ser un mapa")
        return cls.model_validate(crudo)

    def save(self, path: str | Path) -> Path:
        destino = Path(path)
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(
            yaml.safe_dump(self.model_dump(exclude_none=True), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return destino
