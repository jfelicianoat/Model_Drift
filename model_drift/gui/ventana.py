"""La ventana: composición de las pestañas y reparto de mensajes.

Aquí solo queda lo que es de la ventana entera y de nadie más: montar el
`Notebook`, la barra de estado, el bombeo de la cola y el cierre limpio. El
comportamiento de cada zona vive en su módulo (`catalogo`, `ejecucion`,
`tiradas`, `comparacion`) y la maquetación en `pestanas`.

Dos reglas de construcción que valen para cualquier interfaz sobre este motor:

1. **El trabajo pesado nunca ocurre en el hilo de Tk.** Una tirada dura horas.
   Va en un hilo aparte y se comunica por cola; los widgets solo se tocan desde
   `_bombear_cola`. Sin eso la ventana se congela y Windows la marca como «no
   responde», que es lo que hace pensar que la aplicación ha muerto.
2. **Lo que el motor no puede garantizar, la interfaz lo enseña.** Las
   garantías ausentes y la potencia declarada tienen su sitio en pantalla, no
   escondidas en un JSON. Una interfaz que las oculte convierte una medición
   honesta en un número bonito.
"""
from __future__ import annotations

import queue
import tkinter as tk
from pathlib import Path
from tkinter import StringVar, messagebox, ttk

from model_drift.config import AppConfig
from model_drift.gui import pestanas
from model_drift.gui.calibration import CalibrationMixin
from model_drift.gui.comparacion import ComparacionMixin
from model_drift.gui.comunes import Mensaje
from model_drift.gui.ejecucion import EjecucionMixin
from model_drift.runner import RunProgress


class ModelDriftGUI(EjecucionMixin, ComparacionMixin, CalibrationMixin):
    """Ventana principal de Model_Drift."""

    def __init__(self, config: AppConfig | None = None, master: tk.Misc | None = None) -> None:
        super().__init__(config, master)
        self._construir()
        # El identificador se guarda para poder cancelarlo al cerrar: un `after`
        # que se reprograma solo y sobrevive al `destroy` acaba disparándose
        # contra un intérprete Tcl ya muerto.
        self._id_bombeo: str | None = self.after(120, self._bombear_cola)
        self.protocol("WM_DELETE_WINDOW", self._al_cerrar)
        self.refrescar_tiradas()

    # ------------------------------------------------------------- construcción

    def _construir(self) -> None:
        self._estilos()

        self.pestanas = ttk.Notebook(self)
        self.pestanas.pack(fill="both", expand=True, padx=6, pady=6)
        self.pestanas.add(pestanas.construir_modelos(self), text="  Modelos  ")
        self.pestanas.add(pestanas.construir_ejecutar(self), text="  Ejecutar tirada  ")
        self.pestanas.add(pestanas.construir_tiradas(self), text="  Tiradas  ")
        self.pestanas.add(pestanas.construir_comparar(self), text="  Comparar  ")
        self.pestanas.add(pestanas.construir_calibration(self), text="  Probar System 1  ")

        barra = ttk.Frame(self)
        barra.pack(fill="x", side="bottom")
        self.estado = StringVar(value=f"Base de datos: {Path(self.config_app.database).resolve()}")
        ttk.Label(barra, textvariable=self.estado, anchor="w").pack(side="left", padx=8, pady=4)
        self.progreso = ttk.Progressbar(barra, mode="determinate", length=220)
        self.progreso.pack(side="right", padx=8, pady=4)

    def _estilos(self) -> None:
        estilo = ttk.Style(self)
        if "vista" in estilo.theme_names():
            estilo.theme_use("vista")
        estilo.configure("Aviso.TLabel", foreground="#8a5300")
        estilo.configure("Grave.TLabel", foreground="#b91c1c")
        estilo.configure("Bien.TLabel", foreground="#15803d")
        estilo.configure("Titulo.TLabel", font=("Segoe UI", 11, "bold"))

    # ---------------------------------------------------------------- mensajes

    def _bombear_cola(self) -> None:
        """Único punto donde se tocan los widgets con datos de otro hilo."""
        try:
            while True:
                mensaje = self.cola.get_nowait()
                self._atender(mensaje)
        except queue.Empty:
            pass
        self._id_bombeo = self.after(120, self._bombear_cola)

    def _atender(self, mensaje: Mensaje) -> None:
        if mensaje.tipo == "catalogo":
            self._pintar_catalogo(mensaje.dato)
        elif mensaje.tipo == "progreso":
            progreso: RunProgress = mensaje.dato
            self.progreso["value"] = progreso.fraction * 100
            self.var_progreso.set(f"{progreso.describe()} · {progreso.cost_usd:.4f} USD")
        elif mensaje.tipo == "log":
            self.log(str(mensaje.dato))
        elif mensaje.tipo == "log_cal":
            self._avance_calibracion(str(mensaje.dato))
        elif mensaje.tipo == "tirada_terminada":
            self._tirada_terminada(mensaje.dato)
        elif mensaje.tipo == "comparacion":
            self._pintar_comparacion(mensaje.dato)
        elif mensaje.tipo == "calibration":
            self._pintar_calibracion(mensaje.dato)
        elif mensaje.tipo == "error":
            self._atender_error(mensaje.dato)

    def _atender_error(self, dato: tuple[str, str]) -> None:
        texto, traza = dato
        self.progreso["value"] = 0
        self.boton_lanzar.configure(state="normal")
        self.boton_cancelar.configure(state="disabled")
        self.log(f"ERROR: {texto}")
        self.log(traza.strip().splitlines()[-1] if traza else "")
        # Sin esto la barra se quedaba en «Comparando…» para siempre.
        self.estado.set(f"La última operación falló: {texto.splitlines()[0] if texto else '?'}")
        messagebox.showerror("Model_Drift", texto, parent=self)

    # ------------------------------------------------------------------ cierre

    def _al_cerrar(self) -> None:
        if self.hilo is not None and self.hilo.is_alive():
            if not messagebox.askyesno(
                "Hay trabajo en curso",
                "Hay una operación ejecutándose. Si cierras ahora se perderá lo que no se haya "
                "guardado todavía.\n\n¿Cerrar de todas formas?",
                parent=self,
            ):
                return
            self.cancelar.set()
        self.cerrar()

    def cerrar(self) -> None:
        """Cierra la ventana sin dejar temporizadores colgando."""
        if self._id_bombeo is not None:
            try:
                self.after_cancel(self._id_bombeo)
            except Exception:  # noqa: BLE001 - si el intérprete ya no está, no hay nada que cancelar
                pass
            self._id_bombeo = None
        self.destroy()
