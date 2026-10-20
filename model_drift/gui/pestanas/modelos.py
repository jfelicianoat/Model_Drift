"""Pestaña «Modelos»: credenciales del Broker y catálogo consultable."""
from __future__ import annotations

from tkinter import StringVar, ttk
from typing import TYPE_CHECKING

from model_drift.gui.pestanas.filtros import barra_filtros

if TYPE_CHECKING:  # pragma: no cover - solo para el comprobador de tipos
    from model_drift.gui.ventana import ModelDriftGUI

COLUMNAS = (
    ("modelo", "Modelo", 300),
    ("despachable", "Despachable", 90),
    ("contexto", "Contexto", 90),
    ("tools", "Tools", 60),
    ("vision", "Visión", 60),
    ("motivo", "Estado según el Broker", 420),
)

AVISO = (
    "El Broker rota su token en cada arranque: cógelo de su ventana. "
    "Un '?' significa que el Broker no ha comprobado esa capacidad, y un caso "
    "que la exija quedará como «no aplicable» en vez de suspender. La matriz "
    "completa de capacidades se consulta por modelo al lanzar la tirada: aquí "
    "solo se enseña lo que el catálogo trae de una sola vez. Las filas que marques "
    "aquí (Ctrl para varias) son los modelos de la tirada, igual que en «Ejecutar tirada»."
)


def construir(app: ModelDriftGUI) -> ttk.Frame:
    marco = ttk.Frame(app)

    arriba = ttk.Frame(marco)
    arriba.pack(fill="x", padx=8, pady=4)
    ttk.Label(arriba, text="Broker:").pack(side="left")
    app.var_broker = StringVar(value=app.config_app.broker_url)
    ttk.Entry(arriba, textvariable=app.var_broker, width=32).pack(side="left", padx=6)
    app.var_token = StringVar(value=app.config_app.broker_token or "")
    if app.config_app.core_settings():
        # El token del Broker lo da AI OS Core en cada petición: no hay nada que pegar
        ttk.Label(arriba, text="Token: lo da AI OS Core").pack(side="left", padx=6)
    else:
        ttk.Label(arriba, text="Token:").pack(side="left")
        ttk.Entry(arriba, textvariable=app.var_token, width=28, show="*").pack(side="left", padx=6)
    ttk.Button(arriba, text="Consultar catálogo", command=app.refrescar_catalogo).pack(side="left", padx=6)

    barra_filtros(app, marco).pack(fill="x", padx=8, pady=4)

    ttk.Label(marco, text=AVISO, wraplength=1040, style="Aviso.TLabel").pack(fill="x", padx=8, pady=4)

    app.tabla_modelos = ttk.Treeview(
        marco, columns=[columna for columna, _, _ in COLUMNAS], show="headings", selectmode="extended"
    )
    for columna, titulo, ancho in COLUMNAS:
        app.tabla_modelos.heading(columna, text=titulo)
        app.tabla_modelos.column(columna, width=ancho, anchor="w")
    app.tabla_modelos.tag_configure("vacia", foreground="#6b7280")
    app.tabla_modelos.pack(fill="both", expand=True, padx=8, pady=4)
    app.tabla_modelos.bind("<<TreeviewSelect>>", lambda _: app._al_seleccionar_en_tabla())
    ttk.Label(marco, textvariable=app.var_seleccion).pack(anchor="w", padx=8, pady=(0, 4))
    return marco
