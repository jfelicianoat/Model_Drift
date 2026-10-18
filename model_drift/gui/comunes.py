"""Piezas compartidas por toda la interfaz: mensajes entre hilos y formato.

Están aquí y no en la ventana porque las usan tanto la construcción de las
pestañas como los hilos de trabajo, y un módulo sin dependencias de widgets se
puede importar desde cualquiera de los dos lados sin ciclos.
"""
from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from typing import Any

#: Agrupa `cloud` y `api`: los dos salen de la máquina, y para la frontera
#: de datos son lo mismo. El Broker distingue tres despliegues, no dos.
REMOTOS = "remotos (cloud + api)"


@dataclass
class Mensaje:
    """Lo que un hilo de trabajo devuelve a la interfaz."""

    tipo: str
    dato: Any = None


def _tri(valor: bool | None) -> str:
    return "sí" if valor is True else "no" if valor is False else "?"


def _num(valor: float | None) -> str:
    return "—" if valor is None else f"{valor:.4g}"


def escribir_texto(widget: tk.Text, texto: str, *, anexar: bool = False) -> None:
    """Escribe en un `Text` de solo lectura sin dejarlo editable después."""
    widget.configure(state="normal")
    if not anexar:
        widget.delete("1.0", "end")
    widget.insert("end", texto)
    # Un registro se lee por el final; un detalle o un resumen, por la cabecera.
    widget.see("end" if anexar else "1.0")
    widget.configure(state="disabled")
