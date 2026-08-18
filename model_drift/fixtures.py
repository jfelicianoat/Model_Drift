"""Artefactos versionados por hash: `fixture://...`.

Todo lo que un caso necesita del disco —documentos, esquemas, BD de prueba,
definiciones de herramientas, respuestas programadas, tests de código— entra
por aquí y sale con su hash calculado.

El motivo no es la comodidad: **un fixture que cambia es un instrumento que
cambia, no un modelo que cambia** (plan §11). Si un esquema JSON se edita entre
dos tiradas, la caída de conformidad no es deriva del modelo, y la única forma
de saberlo es que el manifiesto lleve el hash del esquema.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

FIXTURE_SCHEME = "fixture://"


class FixtureError(RuntimeError):
    """Un fixture referenciado no existe o no se puede leer."""


class FixtureResolver:
    """Resuelve referencias `fixture://` contra la raíz de una suite.

    Cachea contenido y hash: una tirada de mil casos que comparten el mismo
    esquema lo lee una vez.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._bytes: dict[str, bytes] = {}
        self._hashes: dict[str, str] = {}

    @staticmethod
    def is_reference(value: object) -> bool:
        return isinstance(value, str) and value.startswith(FIXTURE_SCHEME)

    def path_for(self, reference: str) -> Path:
        if not self.is_reference(reference):
            raise FixtureError(f"No es una referencia de fixture: {reference!r}")
        relative = reference[len(FIXTURE_SCHEME) :].lstrip("/")
        if not relative:
            raise FixtureError(f"Referencia de fixture vacía: {reference!r}")
        candidate = (self.root / relative).resolve()
        root = self.root.resolve()
        # Un caso no puede alcanzar ficheros fuera de su suite: la suite es la
        # unidad que se versiona y se comparte, y una ruta con '..' rompería
        # tanto la reproducibilidad como el aislamiento.
        if root != candidate and root not in candidate.parents:
            raise FixtureError(f"La referencia {reference!r} apunta fuera de la suite")
        return candidate

    def read_bytes(self, reference: str) -> bytes:
        if reference in self._bytes:
            return self._bytes[reference]
        path = self.path_for(reference)
        if not path.is_file():
            raise FixtureError(f"Fixture no encontrado: {reference} -> {path}")
        data = path.read_bytes()
        self._bytes[reference] = data
        self._hashes[reference] = hashlib.sha256(data).hexdigest()
        return data

    def read_text(self, reference: str, *, encoding: str = "utf-8") -> str:
        return self.read_bytes(reference).decode(encoding)

    def read_json(self, reference: str) -> Any:
        try:
            return json.loads(self.read_text(reference))
        except json.JSONDecodeError as error:
            raise FixtureError(f"Fixture {reference} no es JSON válido: {error}") from error

    def read_yaml(self, reference: str) -> Any:
        try:
            return yaml.safe_load(self.read_text(reference))
        except yaml.YAMLError as error:
            raise FixtureError(f"Fixture {reference} no es YAML válido: {error}") from error

    def read_structured(self, reference: str) -> Any:
        """Lee JSON o YAML según la extensión. YAML es superconjunto de JSON."""
        if reference.endswith((".yaml", ".yml")):
            return self.read_yaml(reference)
        return self.read_json(reference)

    def hash_of(self, reference: str) -> str:
        if reference not in self._hashes:
            self.read_bytes(reference)
        return self._hashes[reference]

    def resolve_maybe(self, value: Any) -> Any:
        """Devuelve el contenido si es una referencia; el valor tal cual si no.

        Permite que un caso escriba el esquema en línea o lo saque a un fichero
        sin que el evaluador tenga que enterarse.
        """
        if self.is_reference(value):
            return self.read_structured(value)
        return value

    def used_hashes(self) -> dict[str, str]:
        """Hashes de todo lo leído. Va al manifiesto de la tirada."""
        return dict(sorted(self._hashes.items()))
