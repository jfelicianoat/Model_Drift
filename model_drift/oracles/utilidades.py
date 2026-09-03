"""Utilidades compartidas por las aserciones: texto, JSON, rutas y coseno.

Están fuera de las aserciones porque las usan varias familias y también
módulos de fuera (`taxonomy`, `judge`, `metamorphic`, `embeddings`).
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_URL = re.compile(r"https?://[^\s<>\"'`\]\)}]+", re.IGNORECASE)
_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")
_SQL_FENCE = re.compile(r"```(?:sql|SQL)?\s*(.*?)```", re.DOTALL)


def normalize_text(value: str) -> str:
    """Normalización conservadora: espacios y mayúsculas, nada más.

    No se tocan tildes ni puntuación: en español eso cambiaría el significado
    de lo que se compara y convertiría un oráculo en una heurística.
    """
    return re.sub(r"\s+", " ", value).strip().casefold()


def strip_code_fences(text: str) -> str:
    """Devuelve el contenido de la primera valla de código, o el texto entero.

    Las vallas son la deriva de formato más común entre versiones: el modelo
    empieza a envolver el JSON que antes devolvía desnudo. Se toleran aquí para
    que la aserción de esquema mida el esquema, y se detecta aparte con la
    taxonomía del cambio.
    """
    match = _FENCE.search(text)
    return match.group(1).strip() if match else text.strip()


def extract_json(text: str) -> tuple[Any | None, str]:
    """Intenta obtener un objeto JSON del texto. Devuelve (valor, motivo)."""
    candidate = strip_code_fences(text)
    try:
        return json.loads(candidate), ""
    except json.JSONDecodeError:
        pass
    # Segundo intento: el primer bloque {...} o [...] equilibrado.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        end = candidate.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1]), ""
            except json.JSONDecodeError:
                continue
    return None, "la salida no contiene JSON parseable"


_PATH_TOKEN = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


def resolve_path(data: Any, path: str) -> tuple[Any, bool]:
    """Resuelve una ruta tipo `$.factura.lineas[0].total`.

    Subconjunto deliberado de JSONPath: sin comodines ni filtros. Una ruta que
    nadie puede leer de un vistazo no sirve como aserción de un TFM.
    """
    cleaned = path.lstrip("$").lstrip(".")
    current = data
    for token in _PATH_TOKEN.finditer(cleaned):
        key, index = token.group(1), token.group(2)
        if index is not None:
            if not isinstance(current, list) or int(index) >= len(current):
                return None, False
            current = current[int(index)]
            continue
        if not isinstance(current, dict) or key not in current:
            return None, False
        current = current[key]
    return current, True


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)
