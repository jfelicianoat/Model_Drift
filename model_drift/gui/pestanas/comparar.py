"""Pestaña «Comparar»: dos tiradas, veredicto y tabla de casos."""
from __future__ import annotations

import tkinter as tk
from tkinter import StringVar, ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - solo para el comprobador de tipos
    from model_drift.gui.ventana import ModelDriftGUI

COLUMNAS = (
    ("caso", "Caso", 170),
    ("metrica", "Métrica", 130),
    ("cambio", "Cambio", 150),
    ("direccion", "Dirección", 200),
    ("evidencia", "Evidencia", 150),
    ("base", "Base", 70),
    ("cand", "Cand.", 70),
    ("p", "p corregido", 90),
    ("taxonomia", "Qué cambió", 160),
)


def construir(app: ModelDriftGUI) -> ttk.Frame:
    marco = ttk.Frame(app)
    _formulario(app, marco)
    _acciones(app, marco)

    app.resumen = tk.Text(marco, height=9, wrap="word", state="disabled")
    app.resumen.pack(fill="x", padx=8, pady=4)

    _tabla_casos(app, marco)
    return marco


def _formulario(app: ModelDriftGUI, marco: ttk.Frame) -> None:
    formulario = ttk.LabelFrame(marco, text="Qué dos tiradas se comparan")
    formulario.pack(fill="x", padx=8, pady=4)
    app.var_base = StringVar()
    app.var_candidata = StringVar()
    ttk.Label(formulario, text="Línea base:").grid(row=0, column=0, sticky="e", padx=8, pady=4)
    app.combo_base = ttk.Combobox(formulario, textvariable=app.var_base, width=60, state="readonly")
    app.combo_base.grid(row=0, column=1, sticky="we", padx=8, pady=4)
    ttk.Label(formulario, text="Candidata:").grid(row=1, column=0, sticky="e", padx=8, pady=4)
    app.combo_candidata = ttk.Combobox(formulario, textvariable=app.var_candidata, width=60, state="readonly")
    app.combo_candidata.grid(row=1, column=1, sticky="we", padx=8, pady=4)
    app.var_politica_cmp = StringVar()
    app._fila_ruta(formulario, 2, "Política de regresión:", app.var_politica_cmp, carpeta=False)
    formulario.columnconfigure(1, weight=1)


def _acciones(app: ModelDriftGUI, marco: ttk.Frame) -> None:
    acciones = ttk.Frame(marco)
    acciones.pack(fill="x", padx=8, pady=4)
    ttk.Button(acciones, text="Comparar", command=app.comparar).pack(side="left")
    app.boton_informe = ttk.Button(acciones, text="Abrir informe HTML", command=app.abrir_informe, state="disabled")
    app.boton_informe.pack(side="left", padx=6)
    app.var_veredicto = StringVar(value="")
    ttk.Label(acciones, textvariable=app.var_veredicto, style="Titulo.TLabel").pack(side="left", padx=14)


def _tabla_casos(app: ModelDriftGUI, marco: ttk.Frame) -> None:
    app.tabla_casos = ttk.Treeview(marco, columns=[columna for columna, _, _ in COLUMNAS], show="headings")
    for columna, titulo, ancho in COLUMNAS:
        app.tabla_casos.heading(columna, text=titulo)
        app.tabla_casos.column(columna, width=ancho, anchor="w")
    app.tabla_casos.tag_configure("regresion", background="#fee2e2")
    app.tabla_casos.tag_configure("mejora", background="#dcfce7")
    app.tabla_casos.tag_configure("indeterminado", background="#f3f4f6")
    app.tabla_casos.pack(fill="both", expand=True, padx=8, pady=4)
