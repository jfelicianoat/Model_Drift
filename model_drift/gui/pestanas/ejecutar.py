"""Pestaña «Ejecutar tirada»: qué se mide, cómo y contra qué modelos."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import StringVar, ttk
from typing import TYPE_CHECKING

from model_drift.gui.pestanas.filtros import barra_filtros

if TYPE_CHECKING:  # pragma: no cover - solo para el comprobador de tipos
    from model_drift.gui.ventana import ModelDriftGUI

AVISO_REPETICIONES = (
    "Menos de 5 repeticiones no permite concluir nada: el contraste exacto no puede "
    "bajar del nivel exigido y todos los casos saldrían como «indeterminado»."
)


def construir(app: ModelDriftGUI) -> ttk.Frame:
    marco = ttk.Frame(app)
    _formulario(app, marco)
    _parametros(app, marco)
    _seleccion_modelos(app, marco)
    _acciones(app, marco)

    app.registro = tk.Text(marco, height=10, wrap="word", state="disabled")
    app.registro.pack(fill="both", expand=True, padx=8, pady=4)
    return marco


def _formulario(app: ModelDriftGUI, marco: ttk.Frame) -> None:
    formulario = ttk.LabelFrame(marco, text="Qué se va a medir")
    formulario.pack(fill="x", padx=8, pady=4)

    app.var_suite = StringVar(value=str(Path(app.config_app.suites_dir) / "ejemplo"))
    app.var_relaciones = StringVar(value=app.config_app.relations_dir)
    app.var_politica = StringVar()
    app._fila_ruta(formulario, 0, "Suite:", app.var_suite, carpeta=True)
    app._fila_ruta(formulario, 1, "Relaciones metamórficas:", app.var_relaciones, carpeta=True)
    app._fila_ruta(formulario, 2, "Política de regresión:", app.var_politica, carpeta=False)


def _parametros(app: ModelDriftGUI, marco: ttk.Frame) -> None:
    parametros = ttk.LabelFrame(marco, text="Cómo se va a medir")
    parametros.pack(fill="x", padx=8, pady=4)
    app.var_repeticiones = StringVar(value=str(max(5, app.config_app.repetitions)))
    app.var_presupuesto = StringVar()
    app.var_etiqueta = StringVar()
    for columna, (texto, variable, ancho) in enumerate(
        (
            ("Repeticiones por caso:", app.var_repeticiones, 6),
            ("Presupuesto (USD):", app.var_presupuesto, 10),
            ("Etiqueta:", app.var_etiqueta, 28),
        )
    ):
        ttk.Label(parametros, text=texto).grid(row=0, column=columna * 2, sticky="e", padx=8, pady=4)
        ttk.Entry(parametros, textvariable=variable, width=ancho).grid(
            row=0, column=columna * 2 + 1, sticky="w", padx=8, pady=4
        )
    ttk.Label(parametros, text=AVISO_REPETICIONES, wraplength=1000, style="Aviso.TLabel").grid(
        row=1, column=0, columnspan=6, sticky="w", padx=8, pady=4
    )


def _seleccion_modelos(app: ModelDriftGUI, marco: ttk.Frame) -> None:
    seleccion = ttk.LabelFrame(marco, text="Contra qué modelos (selección múltiple con Ctrl)")
    seleccion.pack(fill="both", expand=True, padx=8, pady=4)
    barra_filtros(app, seleccion).pack(fill="x", padx=8, pady=4)
    app.lista_modelos = tk.Listbox(seleccion, selectmode="extended", height=6, exportselection=False)
    app.lista_modelos.pack(fill="both", expand=True, padx=8, pady=4)
    app.lista_modelos.bind("<<ListboxSelect>>", lambda _: app._al_seleccionar_modelos())
    ttk.Label(seleccion, textvariable=app.var_seleccion).pack(anchor="w", padx=8, pady=(0, 4))


def _acciones(app: ModelDriftGUI, marco: ttk.Frame) -> None:
    acciones = ttk.Frame(marco)
    acciones.pack(fill="x", padx=8, pady=4)
    app.boton_lanzar = ttk.Button(acciones, text="Lanzar tirada", command=app.lanzar_tirada)
    app.boton_lanzar.pack(side="left")
    app.boton_cancelar = ttk.Button(acciones, text="Cancelar", command=app.cancelar_tirada, state="disabled")
    app.boton_cancelar.pack(side="left", padx=6)
    app.var_progreso = StringVar(value="Sin tirada en curso.")
    ttk.Label(acciones, textvariable=app.var_progreso).pack(side="left", padx=12)
