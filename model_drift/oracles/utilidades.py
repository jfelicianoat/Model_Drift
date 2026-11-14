"""Utilidades compartidas por las aserciones: texto, JSON, rutas y coseno.

Están fuera de las aserciones porque las usan varias familias y también
módulos de fuera (`taxonomy`, `judge`, `metamorphic`, `embeddings`).
"""
from __future__ import annotations

import json
import math
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_URL = re.compile(r"https?://[^\s<>\"'`\]\)}]+", re.IGNORECASE)
_FECHA_U_HORA = re.compile(
    r"\b\d{4}-\d{2}-\d{2}\b"
    r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    r"|\b\d{1,2}:\d{2}(?::\d{2})?\b"
)
_NUMBER = re.compile(
    r"(?<![\w.,])-?(?:"
    r"\d{1,3}(?:\.\d{3})+,\d+"      # 1.234,56
    r"|\d{1,3}(?:,\d{3})+\.\d+"     # 1,234.56
    r"|\d{1,3}(?:\.\d{3}){2,}"      # 1.234.567
    r"|\d{1,3}(?:,\d{3}){2,}"       # 1,234,567
    r"|\d+(?:[.,]\d+)?"             # 42 / 42,3 / 42.3
    r")(?![\w]|[.,]\d)"
)
_SQL_FENCE = re.compile(r"```(?:sql|SQL)?\s*(.*?)```", re.DOTALL)


def normalize_text(value: str) -> str:
    """Normalización conservadora: espacios y mayúsculas, nada más.

    No se tocan tildes ni puntuación: en español eso cambiaría el significado
    de lo que se compara y convertiría un oráculo en una heurística.
    """
    return re.sub(r"\s+", " ", value).strip().casefold()


def extract_numbers(text: str) -> list[float]:
    """Números del texto, en orden, entendiendo miles y decimales en español e inglés.

    Las fechas y las horas se descartan antes de buscar. Un separador único
    («1.234», «1,234») se lee como decimal: es ambiguo y se mantiene el
    comportamiento anterior.
    """
    limpio = _FECHA_U_HORA.sub(" ", text)
    valores: list[float] = []
    for bruto in _NUMBER.findall(limpio):
        puntos, comas = bruto.count("."), bruto.count(",")
        if puntos and comas:
            if bruto.rfind(",") > bruto.rfind("."):
                bruto = bruto.replace(".", "").replace(",", ".")
            else:
                bruto = bruto.replace(",", "")
        elif puntos > 1:
            bruto = bruto.replace(".", "")
        elif comas > 1:
            bruto = bruto.replace(",", "")
        else:
            bruto = bruto.replace(",", ".")
        valores.append(float(bruto))
    return valores


def normalize_url(url: str) -> str:
    """Forma canónica para cotejar citas: esquema y host en minúsculas, sin
    fragmento y sin barra final. La ruta y la consulta se respetan tal cual."""
    partes = urlsplit(url.strip().rstrip(".,;:)"))
    return urlunsplit((partes.scheme.lower(), partes.netloc.lower(), partes.path.rstrip("/"), partes.query, ""))


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
