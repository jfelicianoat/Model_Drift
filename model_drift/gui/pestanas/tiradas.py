"""Pestaña «Tiradas»: listado de lo guardado y detalle del manifiesto."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - solo para el comprobador de tipos
    from model_drift.gui.ventana import ModelDriftGUI

COLUMNAS = (
    ("run", "Tirada", 230),
    ("fecha", "Fecha", 140),
    ("suite", "Suite", 150),
    ("modelos", "Modelos", 230),
    ("etiqueta", "Etiqueta", 150),
    ("estado", "Estado", 90),
    ("faltan", "Garantías ausentes", 130),
)


def construir(app: ModelDriftGUI) -> ttk.Frame:
    marco = ttk.Frame(app)

    arriba = ttk.Frame(marco)
    arriba.pack(fill="x", padx=8, pady=4)
    ttk.Button(arriba, text="Refrescar", command=app.refrescar_tiradas).pack(side="left")

    app.tabla_tiradas = ttk.Treeview(
        marco, columns=[columna for columna, _, _ in COLUMNAS], show="headings", height=12
    )
    for columna, titulo, ancho in COLUMNAS:
        app.tabla_tiradas.heading(columna, text=titulo)
        app.tabla_tiradas.column(columna, width=ancho, anchor="w")
    app.tabla_tiradas.pack(fill="both", expand=True, padx=8, pady=4)
    app.tabla_tiradas.bind("<<TreeviewSelect>>", lambda _: app._mostrar_detalle_tirada())

    ttk.Label(marco, text="Detalle de la tirada seleccionada", style="Titulo.TLabel").pack(
        anchor="w", padx=8, pady=4
    )
    app.detalle_tirada = tk.Text(marco, height=12, wrap="word", state="disabled")
    app.detalle_tirada.pack(fill="both", expand=True, padx=8, pady=4)
    return marco
