"""Estado de la ventana y utilidades que no pertenecen a ninguna pestaña.

`VentanaBase` reúne tres cosas y solo tres:

1. **El estado**: configuración, cola de mensajes, catálogo y selección.
2. **Las variables de Tk** que comparten varias pestañas (los filtros se pintan
   dos veces con las mismas variables, y por eso filtrar en una pestaña filtra
   en la otra).
3. **Las utilidades transversales**: log, credenciales, cliente del Broker y el
   lanzamiento de trabajo en segundo plano.

Las anotaciones de widgets que aparecen sin valor son el contrato entre la
clase y los constructores de pestañas: dicen qué widgets tiene que dejar
puestos la construcción para que el resto del código pueda contar con ellos.
Sin ellas, mypy no vería un solo error en el día en que una pestaña deje de
crear su tabla.
"""
from __future__ import annotations

import queue
import threading
import tkinter as tk
import traceback
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from tkinter import BooleanVar, StringVar, filedialog, messagebox, ttk
from typing import Any

from model_drift import __version__
from model_drift.broker_client import BrokerClient
from model_drift.broker_contracts import ModelCatalogEntry
from model_drift.comparison import ComparisonReport
from model_drift.config import AppConfig
from model_drift.gui.comunes import Mensaje, escribir_texto


class VentanaBase(tk.Toplevel):
    """Ventana principal sin comportamiento de pestañas.

    Es un `Toplevel` y no un `Tk` a propósito: quien arranca el proceso es
    dueño de la raíz. Tcl no lleva bien crear y destruir intérpretes repetidos
    —acaba fallando con «invalid command name tcl_findLibrary»—, y una clase
    que no se apropia de la raíz se puede abrir varias veces en un mismo
    proceso sin tocar el intérprete.
    """

    # -- Puestos por la construcción de pestañas (véase `gui.pestanas`). ------
    pestanas: ttk.Notebook
    estado: StringVar
    progreso: ttk.Progressbar
    tabla_modelos: ttk.Treeview
    lista_modelos: tk.Listbox
    var_broker: StringVar
    var_token: StringVar
    var_suite: StringVar
    var_relaciones: StringVar
    var_politica: StringVar
    var_repeticiones: StringVar
    var_presupuesto: StringVar
    var_etiqueta: StringVar
    var_progreso: StringVar
    boton_lanzar: ttk.Button
    boton_cancelar: ttk.Button
    registro: tk.Text
    tabla_tiradas: ttk.Treeview
    detalle_tirada: tk.Text
    var_base: StringVar
    var_candidata: StringVar
    combo_base: ttk.Combobox
    combo_candidata: ttk.Combobox
    var_politica_cmp: StringVar
    var_veredicto: StringVar
    boton_informe: ttk.Button
    resumen: tk.Text
    tabla_casos: ttk.Treeview
    var_cal_plan: StringVar
    var_cal_teacher: StringVar
    var_cal_provider: StringVar
    var_cal_model: StringVar
    cal_teacher_combo: ttk.Combobox
    cal_summary: tk.Text
    cal_report_path: Path | None = None

    def __init__(self, config: AppConfig | None = None, master: tk.Misc | None = None) -> None:
        super().__init__(master)
        self.config_app = config or AppConfig.load()
        self.cola: queue.Queue[Mensaje] = queue.Queue()
        self.cancelar = threading.Event()
        self.hilo: threading.Thread | None = None
        self.catalogo: list[ModelCatalogEntry] = []
        #: Claves de los modelos elegidos. Se guardan aparte de la selección
        #: visible del Listbox: si un filtro esconde un modelo ya elegido, la
        #: elección NO se pierde en silencio.
        self.seleccionados: set[str] = set()
        self._combos_proveedor: list[ttk.Combobox] = []
        self._combos_despliegue: list[ttk.Combobox] = []
        #: Manifiestos de las tiradas listadas, por `run_id`. Los rellena la
        #: pestaña de tiradas y los consulta la de comparación.
        self._manifiestos: dict[str, dict[str, Any]] = {}
        self.ultimo_informe: ComparisonReport | None = None
        self.ruta_informe: Path | None = None

        self.filtro_proveedor = StringVar(value="todos")
        self.filtro_despliegue = StringVar(value="todos")
        self.filtro_texto = StringVar()
        self.solo_despachables = BooleanVar(value=True)
        self.var_seleccion = StringVar(value="0 modelos elegidos")

        self.title(f"Model_Drift {__version__}")
        self.geometry("1120x760")
        self.minsize(900, 620)

    # ------------------------------------------------------------ utilidades

    def _escribir(self, widget: tk.Text, texto: str, *, anexar: bool = False) -> None:
        escribir_texto(widget, texto, anexar=anexar)

    def log(self, texto: str) -> None:
        marca = datetime.now().strftime("%H:%M:%S")
        self._escribir(self.registro, f"[{marca}] {texto}\n", anexar=True)

    def _credenciales(self) -> tuple[str, str]:
        """Lee los widgets del Broker. **Solo desde el hilo principal.**"""
        return self.var_broker.get().strip(), self.var_token.get().strip()

    def _cliente(self, base_url: str = "", token: str = "") -> BrokerClient:
        """Construye el cliente a partir de datos planos, nunca de widgets.

        Leer un `StringVar` desde un hilo de trabajo lanza «main thread is not
        in main loop» —o, peor, cuelga la aplicación sin decir nada—, así que
        los valores se capturan antes de arrancar el hilo y viajan como texto.
        """
        configuracion = self.config_app.broker_config()
        configuracion.base_url = base_url or configuracion.base_url
        configuracion.admin_token = token or configuracion.admin_token
        return BrokerClient(configuracion)

    def _en_hilo(self, trabajo: Callable[[], Mensaje]) -> None:
        """Lanza trabajo en segundo plano. Los widgets NO se tocan desde ahí."""
        if self.hilo is not None and self.hilo.is_alive():
            messagebox.showinfo("Model_Drift", "Ya hay una operación en curso.")
            return

        def envoltorio() -> None:
            try:
                self.cola.put(trabajo())
            except Exception as error:  # noqa: BLE001 - un fallo del hilo no puede matar la ventana
                self.cola.put(Mensaje("error", (str(error), traceback.format_exc())))

        self.hilo = threading.Thread(target=envoltorio, daemon=True)
        self.hilo.start()

    # ----------------------------------------------------------- rutas

    def _fila_ruta(self, padre: ttk.Widget, fila: int, texto: str, variable: StringVar, *, carpeta: bool) -> None:
        ttk.Label(padre, text=texto).grid(row=fila, column=0, sticky="e", padx=8, pady=4)
        ttk.Entry(padre, textvariable=variable, width=70).grid(row=fila, column=1, sticky="we", padx=8, pady=4)
        ttk.Button(
            padre,
            text="Examinar…",
            command=lambda: self._elegir_ruta(variable, carpeta=carpeta),
        ).grid(row=fila, column=2, padx=8, pady=4)
        padre.columnconfigure(1, weight=1)

    def _elegir_ruta(self, variable: StringVar, *, carpeta: bool) -> None:
        elegida = (
            filedialog.askdirectory(title="Elige una carpeta")
            if carpeta
            else filedialog.askopenfilename(
                title="Elige un fichero",
                filetypes=[("YAML", "*.yaml *.yml"), ("Todos", "*.*")],
            )
        )
        if elegida:
            variable.set(elegida)
