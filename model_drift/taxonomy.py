"""Taxonomía del cambio: *en qué* cambió, aunque no se pueda decir si a mejor.

Independiente de la dirección (plan §9.5). Un caso puede quedarse en
`sin_direccion` y aun así tener aquí la información más útil del informe: que
la versión nueva empezó a envolver el JSON en vallas de código, o a rechazar, o
a contestar el doble de largo.

Todos los detectores son deterministas y baratos. Ninguno necesita un juez.
"""
from __future__ import annotations

import re

from model_drift.oracles import extract_json, strip_code_fences

FORMATO = "formato"
VERBOSIDAD = "verbosidad"
RECHAZO = "rechazo"
IDIOMA = "idioma"
HERRAMIENTA = "herramienta"
CONTENIDO = "contenido"
ABSTENCION = "abstencion"

_FENCE_START = re.compile(r"^\s*```", re.MULTILINE)

_RECHAZO_PATTERNS = (
    "no puedo ayudarte",
    "no puedo proporcionar",
    "no puedo generar",
    "lo siento, pero",
    "no está dentro de mis",
    "i can't help",
    "i cannot provide",
    "i'm sorry, but",
    "as an ai",
)

_ABSTENCION_PATTERNS = (
    "no consta",
    "no aparece",
    "no se menciona",
    "no dispongo",
    "no puedo determinar",
    "no hay información",
)

#: Palabras funcionales muy frecuentes y muy distintas entre los dos idiomas
#: que interesan aquí. No es un detector de idioma general: es un detector de
#: *cambio* de idioma entre dos salidas del mismo prompt, que es un problema
#: mucho más fácil y el único que hace falta resolver.
_MARCADORES_ES = (" el ", " la ", " los ", " las ", " de ", " que ", " para ", " con ", " una ", " del ")
_MARCADORES_EN = (" the ", " of ", " and ", " to ", " with ", " for ", " this ", " that ", " is ", " are ")


def _cuenta(texto: str, marcadores: tuple[str, ...]) -> int:
    plano = f" {texto.casefold()} "
    return sum(plano.count(marcador) for marcador in marcadores)


def _idioma_aproximado(texto: str) -> str | None:
    español, ingles = _cuenta(texto, _MARCADORES_ES), _cuenta(texto, _MARCADORES_EN)
    if español == ingles:
        return None
    # Se exige holgura: con dos marcadores de diferencia sobre textos cortos la
    # clasificación sería ruido, y un falso positivo de idioma en el informe
    # cuesta más que un hueco.
    if abs(español - ingles) < 3:
        return None
    return "es" if español > ingles else "en"


def _tiene_rechazo(texto: str) -> bool:
    plano = texto.casefold()
    return any(patron in plano for patron in _RECHAZO_PATTERNS)


def _tiene_abstencion(texto: str) -> bool:
    plano = texto.casefold()
    return any(patron in plano for patron in _ABSTENCION_PATTERNS)


def classify_change(
    baseline: str,
    candidate: str,
    *,
    baseline_tools: list[str] | None = None,
    candidate_tools: list[str] | None = None,
    verbosity_ratio: float = 1.5,
) -> list[str]:
    """Devuelve las etiquetas de cambio entre dos salidas del mismo caso.

    Lista vacía = las dos salidas son equivalentes en todas las dimensiones que
    este módulo sabe mirar. No significa que sean idénticas.
    """
    etiquetas: list[str] = []
    if baseline == candidate:
        return etiquetas

    # --- Formato: vallas de código y JSON ---
    vallas_base = bool(_FENCE_START.search(baseline))
    vallas_cand = bool(_FENCE_START.search(candidate))
    json_base, _ = extract_json(baseline)
    json_cand, _ = extract_json(candidate)
    if vallas_base != vallas_cand:
        etiquetas.append(FORMATO)
    elif (json_base is None) != (json_cand is None):
        etiquetas.append(FORMATO)
    elif isinstance(json_base, dict) and isinstance(json_cand, dict):
        if set(json_base.keys()) != set(json_cand.keys()):
            etiquetas.append(FORMATO)

    # --- Verbosidad ---
    largo_base = len(strip_code_fences(baseline).split())
    largo_cand = len(strip_code_fences(candidate).split())
    if largo_base > 0 and largo_cand > 0:
        ratio = largo_cand / largo_base
        if ratio >= verbosity_ratio or ratio <= 1 / verbosity_ratio:
            etiquetas.append(VERBOSIDAD)
    elif largo_base != largo_cand:
        etiquetas.append(VERBOSIDAD)

    # --- Rechazo y abstención ---
    if _tiene_rechazo(baseline) != _tiene_rechazo(candidate):
        etiquetas.append(RECHAZO)
    if _tiene_abstencion(baseline) != _tiene_abstencion(candidate):
        etiquetas.append(ABSTENCION)

    # --- Idioma ---
    idioma_base, idioma_cand = _idioma_aproximado(baseline), _idioma_aproximado(candidate)
    if idioma_base is not None and idioma_cand is not None and idioma_base != idioma_cand:
        etiquetas.append(IDIOMA)

    # --- Herramientas ---
    if baseline_tools is not None and candidate_tools is not None:
        if sorted(baseline_tools) != sorted(candidate_tools):
            etiquetas.append(HERRAMIENTA)

    # --- Contenido ---
    # Última etiqueta y deliberadamente laxa: si el texto cambió y nada de lo
    # anterior lo explica, el cambio es de contenido. Es la que menos dice y
    # por eso se reserva para cuando no hay nada mejor.
    if not etiquetas:
        etiquetas.append(CONTENIDO)

    return etiquetas
