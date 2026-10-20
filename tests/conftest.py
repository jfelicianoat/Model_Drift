"""Ajustes comunes de los tests."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _sin_conexion_real_con_core(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    # Core deja la URL y el token de cada app en %LOCALAPPDATA%\AIOSCore\conexion: los tests no la ven
    monkeypatch.setenv("AI_OS_CONNECT_DIR", str(tmp_path_factory.mktemp("conexion_vacia")))
