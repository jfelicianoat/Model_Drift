"""Barra de filtros del catálogo.

Se pinta en dos pestañas. Los widgets son distintos pero las variables son las
mismas, así que filtrar en una pestaña filtra en la otra. Con 162 modelos,
tener que ir a otra pantalla para acotar la lista de la que eliges es de las
cosas que hacen abandonar una herramienta.
"""
from __future__ import annotations

from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - solo para el comprobador de tipos
    from model_drift.gui.ventana import ModelDriftGUI


def barra_filtros(app: ModelDriftGUI, padre: ttk.Widget) -> ttk.Frame:
    marco = ttk.Frame(padre)

    ttk.Label(marco, text="Proveedor:").pack(side="left")
    combo_proveedor = ttk.Combobox(
        marco, textvariable=app.filtro_proveedor, width=14, state="readonly", values=["todos"]
    )
    combo_proveedor.pack(side="left", padx=(4, 12))
    app._combos_proveedor.append(combo_proveedor)

    ttk.Label(marco, text="Despliegue:").pack(side="left")
    combo_despliegue = ttk.Combobox(
        marco, textvariable=app.filtro_despliegue, width=18, state="readonly", values=["todos"]
    )
    combo_despliegue.pack(side="left", padx=(4, 12))
    app._combos_despliegue.append(combo_despliegue)

    ttk.Label(marco, text="Buscar:").pack(side="left")
    entrada = ttk.Entry(marco, textvariable=app.filtro_texto, width=22)
    entrada.pack(side="left", padx=(4, 12))

    ttk.Checkbutton(
        marco,
        text="Solo despachables",
        variable=app.solo_despachables,
        command=app._aplicar_filtros,
    ).pack(side="left")

    for combo in (combo_proveedor, combo_despliegue):
        combo.bind("<<ComboboxSelected>>", lambda _: app._aplicar_filtros())
    entrada.bind("<KeyRelease>", lambda _: app._aplicar_filtros())
    return marco
