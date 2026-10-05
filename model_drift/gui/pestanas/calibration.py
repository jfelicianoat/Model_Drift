"""Formulario de la prueba System 1: plan, alumno y maestro elegidos por la persona."""

from __future__ import annotations

import tkinter as tk
from tkinter import StringVar, ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model_drift.gui.ventana import ModelDriftGUI


def construir(app: ModelDriftGUI) -> ttk.Frame:
    frame = ttk.Frame(app)
    form = ttk.LabelFrame(frame, text="Probar un modelo System 1")
    form.pack(fill="x", padx=8, pady=6)
    app.var_cal_plan, app.var_cal_teacher = StringVar(), StringVar()
    app.var_cal_provider, app.var_cal_model = StringVar(), StringVar()
    ttk.Label(form, text="Plan con casos y etiquetas humanas:").grid(row=0, column=0, sticky="e", padx=6, pady=4)
    ttk.Entry(form, textvariable=app.var_cal_plan, width=65).grid(row=0, column=1, sticky="we", padx=6)
    ttk.Button(form, text="Elegir…", command=app.elegir_plan_calibracion).grid(row=0, column=2, padx=6)
    ttk.Label(form, text="Maestro System 2 (opcional, sustituye al del plan):").grid(row=1, column=0, padx=6, pady=4)
    app.cal_teacher_combo = ttk.Combobox(form, textvariable=app.var_cal_teacher, width=60)
    app.cal_teacher_combo.grid(row=1, column=1, sticky="we", padx=6)
    ttk.Label(form, text="Proveedor del alumno (opcional):").grid(row=2, column=0, sticky="e", padx=6, pady=4)
    ttk.Combobox(
        form, textvariable=app.var_cal_provider, values=["", "ollama_system1", "laya_mcp"], state="readonly"
    ).grid(row=2, column=1, sticky="w", padx=6)
    ttk.Label(form, text="Modelo del alumno:").grid(row=3, column=0, sticky="e", padx=6, pady=4)
    ttk.Entry(form, textvariable=app.var_cal_model).grid(row=3, column=1, sticky="we", padx=6)
    form.columnconfigure(1, weight=1)
    actions = ttk.Frame(frame)
    actions.pack(fill="x", padx=8, pady=4)
    ttk.Button(actions, text="Probar y generar informe", command=app.probar_system1).pack(side="left")
    ttk.Button(actions, text="Cancelar", command=app.cancelar.set).pack(side="left", padx=6)
    ttk.Button(actions, text="Abrir informe", command=app.abrir_informe_calibracion).pack(side="left", padx=6)
    ttk.Label(
        frame,
        text="El maestro debe pasar la comprobación humana. Una muestra pequeña puede ser indeterminada.",
        wraplength=1000,
    ).pack(fill="x", padx=8, pady=4)
    app.cal_summary = tk.Text(frame, height=18, wrap="word", state="disabled")
    app.cal_summary.pack(fill="both", expand=True, padx=8, pady=4)
    return frame
