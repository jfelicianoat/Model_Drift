"""Interfaz gráfica de escritorio.

Tkinter, que viene con Python. Sin dependencias nuevas, sin servidor, sin
puerto y sin navegador: la aplicación abre, lee su SQLite local y ya está. Es
la única interfaz gráfica del proyecto; hubo una web y se retiró, porque un
servidor y un puerto son tres modos de fallo para algo que se usa en local.

Dos reglas de construcción que valen para cualquier interfaz sobre este motor:

1. **El trabajo pesado nunca ocurre en el hilo de Tk.** Una tirada dura horas.
   Va en un hilo aparte y se comunica por cola; los widgets solo se tocan desde
   `_bombear_cola`. Sin eso la ventana se congela y Windows la marca como «no
   responde», que es lo que hace pensar que la aplicación ha muerto.
2. **Lo que el motor no puede garantizar, la interfaz lo enseña.** Las
   garantías ausentes y la potencia declarada tienen su sitio en pantalla, no
   escondidas en un JSON. Una interfaz que las oculte convierte una medición
   honesta en un número bonito.

El informe HTML sigue siendo un fichero: Tkinter no pinta HTML y fingirlo con
etiquetas sería peor. La ventana genera el informe y lo abre en el navegador,
que para *leer* un documento funciona perfectamente.
"""
from __future__ import annotations

import queue
import threading
import tkinter as tk
import traceback
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import BooleanVar, StringVar, Tk, filedialog, messagebox, ttk
from typing import Any

from model_drift import __version__
from model_drift.broker_client import BrokerClient
from model_drift.broker_contracts import BrokerError, ModelCatalogEntry, ModelReference
from model_drift.cases import Suite, load_suite
from model_drift.cli import _load_run
from model_drift.comparison import ComparisonReport, compare_runs
from model_drift.config import AppConfig
from model_drift.db import Database
from model_drift.domain import ChangeDirection, ChangeStatus
from model_drift.metamorphic import load_relations
from model_drift.policy import RegressionPolicy
from model_drift.report import (
    DIRECTION_LABELS,
    STATUS_LABELS,
    VERDICT_LABELS,
    render_text_summary,
    write_comparison,
)
from model_drift.runner import RunConfig, Runner, RunProgress

#: Agrupa `cloud` y `api`: los dos salen de la máquina, y para la frontera
#: de datos son lo mismo. El Broker distingue tres despliegues, no dos.
REMOTOS = "remotos (cloud + api)"


@dataclass
class Mensaje:
    """Lo que un hilo de trabajo devuelve a la interfaz."""

    tipo: str
    dato: Any = None


class ModelDriftGUI(tk.Toplevel):
    """Ventana principal.

    Es un `Toplevel` y no un `Tk` a propósito: quien arranca el proceso es
    dueño de la raíz. Tcl no lleva bien crear y destruir intérpretes repetidos
    —acaba fallando con «invalid command name tcl_findLibrary»—, y una clase
    que no se apropia de la raíz se puede abrir varias veces en un mismo
    proceso sin tocar el intérprete.
    """

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

        self._construir()
        # El identificador se guarda para poder cancelarlo al cerrar: un `after`
        # que se reprograma solo y sobrevive al `destroy` acaba disparándose
        # contra un intérprete Tcl ya muerto.
        self._id_bombeo: str | None = self.after(120, self._bombear_cola)
        self.protocol("WM_DELETE_WINDOW", self._al_cerrar)
        self.refrescar_tiradas()

    # ------------------------------------------------------------- construcción

    def _construir(self) -> None:
        estilo = ttk.Style(self)
        if "vista" in estilo.theme_names():
            estilo.theme_use("vista")
        estilo.configure("Aviso.TLabel", foreground="#8a5300")
        estilo.configure("Grave.TLabel", foreground="#b91c1c")
        estilo.configure("Bien.TLabel", foreground="#15803d")
        estilo.configure("Titulo.TLabel", font=("Segoe UI", 11, "bold"))

        self.pestanas = ttk.Notebook(self)
        self.pestanas.pack(fill="both", expand=True, padx=6, pady=6)
        self.pestanas.add(self._pestana_modelos(), text="  Modelos  ")
        self.pestanas.add(self._pestana_ejecutar(), text="  Ejecutar tirada  ")
        self.pestanas.add(self._pestana_tiradas(), text="  Tiradas  ")
        self.pestanas.add(self._pestana_comparar(), text="  Comparar  ")

        barra = ttk.Frame(self)
        barra.pack(fill="x", side="bottom")
        self.estado = StringVar(value=f"Base de datos: {Path(self.config_app.database).resolve()}")
        ttk.Label(barra, textvariable=self.estado, anchor="w").pack(side="left", padx=8, pady=4)
        self.progreso = ttk.Progressbar(barra, mode="determinate", length=220)
        self.progreso.pack(side="right", padx=8, pady=4)

    # --------------------------------------------------------------- filtros

    def _barra_filtros(self, padre: ttk.Widget) -> ttk.Frame:
        """Filtros del catálogo. Se pinta en las dos pestañas.

        Los widgets son distintos pero las variables son las mismas, así que
        filtrar en una pestaña filtra en la otra. Con 162 modelos, tener que ir
        a otra pantalla para acotar la lista de la que eliges es de las cosas
        que hacen abandonar una herramienta.
        """
        marco = ttk.Frame(padre)

        ttk.Label(marco, text="Proveedor:").pack(side="left")
        combo_proveedor = ttk.Combobox(
            marco, textvariable=self.filtro_proveedor, width=14, state="readonly", values=["todos"]
        )
        combo_proveedor.pack(side="left", padx=(4, 12))
        self._combos_proveedor.append(combo_proveedor)

        ttk.Label(marco, text="Despliegue:").pack(side="left")
        combo_despliegue = ttk.Combobox(
            marco, textvariable=self.filtro_despliegue, width=18, state="readonly", values=["todos"]
        )
        combo_despliegue.pack(side="left", padx=(4, 12))
        self._combos_despliegue.append(combo_despliegue)

        ttk.Label(marco, text="Buscar:").pack(side="left")
        entrada = ttk.Entry(marco, textvariable=self.filtro_texto, width=22)
        entrada.pack(side="left", padx=(4, 12))

        ttk.Checkbutton(
            marco,
            text="Solo despachables",
            variable=self.solo_despachables,
            command=self._aplicar_filtros,
        ).pack(side="left")

        for combo in (combo_proveedor, combo_despliegue):
            combo.bind("<<ComboboxSelected>>", lambda _: self._aplicar_filtros())
        entrada.bind("<KeyRelease>", lambda _: self._aplicar_filtros())
        return marco

    def _modelos_filtrados(self) -> list[ModelCatalogEntry]:
        proveedor = self.filtro_proveedor.get()
        despliegue = self.filtro_despliegue.get()
        texto = self.filtro_texto.get().strip().casefold()

        def pasa(entrada: ModelCatalogEntry) -> bool:
            if proveedor not in ("", "todos") and entrada.provider != proveedor:
                return False
            if despliegue == REMOTOS:
                # `cloud` y `api` son los dos despliegues que salen de la
                # máquina: para la frontera de datos son lo mismo.
                if entrada.deployment == "local":
                    return False
            elif despliegue not in ("", "todos") and entrada.deployment != despliegue:
                return False
            if self.solo_despachables.get() and not entrada.dispatchable:
                return False
            if texto and texto not in entrada.key.casefold():
                return False
            return True

        return sorted((item for item in self.catalogo if pasa(item)), key=lambda item: item.key)

    def _aplicar_filtros(self) -> None:
        visibles = self._modelos_filtrados()

        self.tabla_modelos.delete(*self.tabla_modelos.get_children())
        for entrada in visibles:
            self.tabla_modelos.insert(
                "",
                "end",
                values=(
                    entrada.key,
                    "sí" if entrada.dispatchable else "no",
                    entrada.context_window or "—",
                    _tri(entrada.supports("tools")),
                    _tri(entrada.supports("vision")),
                    entrada.reason,
                ),
            )

        self.lista_modelos.delete(0, "end")
        for indice, entrada in enumerate(visibles):
            self.lista_modelos.insert("end", entrada.key)
            if entrada.key in self.seleccionados:
                self.lista_modelos.selection_set(indice)

        self._actualizar_seleccion_visible()
        ocultos = len(self.catalogo) - len(visibles)
        self.estado.set(
            f"{len(visibles)} de {len(self.catalogo)} modelos"
            + (f" · {ocultos} ocultos por el filtro" if ocultos else "")
        )

    def _al_seleccionar_modelos(self) -> None:
        """Sincroniza la selección visible con el conjunto persistente."""
        visibles = {item.key for item in self._modelos_filtrados()}
        elegidos = {self.lista_modelos.get(indice) for indice in self.lista_modelos.curselection()}
        # Lo que está oculto por el filtro conserva su estado; de lo visible
        # manda lo que se vea marcado.
        self.seleccionados = (self.seleccionados - visibles) | elegidos
        self._actualizar_seleccion_visible()

    def _actualizar_seleccion_visible(self) -> None:
        total = len(self.seleccionados)
        visibles = {item.key for item in self._modelos_filtrados()}
        ocultos = len(self.seleccionados - visibles)
        plural = "s" if total != 1 else ""
        texto = f"{total} modelo{plural} elegido{plural}"
        if ocultos:
            # Sin esto, filtrar después de elegir haría creer que se ha perdido
            # la elección, o peor, se lanzaría con modelos que ya no se ven.
            texto += f" ({ocultos} fuera del filtro actual)"
        self.var_seleccion.set(texto)

    # --------------------------------------------------------------- modelos

    def _pestana_modelos(self) -> ttk.Frame:
        marco = ttk.Frame(self)

        arriba = ttk.Frame(marco)
        arriba.pack(fill="x", padx=8, pady=4)
        ttk.Label(arriba, text="Broker:").pack(side="left")
        self.var_broker = StringVar(value=self.config_app.broker_url)
        ttk.Entry(arriba, textvariable=self.var_broker, width=32).pack(side="left", padx=6)
        ttk.Label(arriba, text="Token:").pack(side="left")
        self.var_token = StringVar(value=self.config_app.broker_token or "")
        ttk.Entry(arriba, textvariable=self.var_token, width=28, show="*").pack(side="left", padx=6)
        ttk.Button(arriba, text="Consultar catálogo", command=self.refrescar_catalogo).pack(side="left", padx=6)

        self._barra_filtros(marco).pack(fill="x", padx=8, pady=4)

        ttk.Label(
            marco,
            text="El Broker rota su token en cada arranque: cógelo de su ventana. "
            "Un '?' significa que el Broker no ha comprobado esa capacidad, y un caso "
            "que la exija quedará como «no aplicable» en vez de suspender. La matriz "
            "completa de capacidades se consulta por modelo al lanzar la tirada: aquí "
            "solo se enseña lo que el catálogo trae de una sola vez.",
            wraplength=1040,
            style="Aviso.TLabel",
        ).pack(fill="x", padx=8, pady=4)

        columnas = ("modelo", "despachable", "contexto", "tools", "vision", "motivo")
        self.tabla_modelos = ttk.Treeview(marco, columns=columnas, show="headings", selectmode="extended")
        for columna, titulo, ancho in (
            ("modelo", "Modelo", 300),
            ("despachable", "Despachable", 90),
            ("contexto", "Contexto", 90),
            ("tools", "Tools", 60),
            ("vision", "Visión", 60),
            ("motivo", "Estado según el Broker", 420),
        ):
            self.tabla_modelos.heading(columna, text=titulo)
            self.tabla_modelos.column(columna, width=ancho, anchor="w")
        self.tabla_modelos.pack(fill="both", expand=True, padx=8, pady=4)
        return marco

    # -------------------------------------------------------------- ejecutar

    def _pestana_ejecutar(self) -> ttk.Frame:
        marco = ttk.Frame(self)

        formulario = ttk.LabelFrame(marco, text="Qué se va a medir")
        formulario.pack(fill="x", padx=8, pady=4)

        self.var_suite = StringVar(value=str(Path(self.config_app.suites_dir) / "ejemplo"))
        self.var_relaciones = StringVar(value=self.config_app.relations_dir)
        self.var_politica = StringVar()
        self._fila_ruta(formulario, 0, "Suite:", self.var_suite, carpeta=True)
        self._fila_ruta(formulario, 1, "Relaciones metamórficas:", self.var_relaciones, carpeta=True)
        self._fila_ruta(formulario, 2, "Política de regresión:", self.var_politica, carpeta=False)

        parametros = ttk.LabelFrame(marco, text="Cómo se va a medir")
        parametros.pack(fill="x", padx=8, pady=4)
        self.var_repeticiones = StringVar(value=str(max(5, self.config_app.repetitions)))
        self.var_presupuesto = StringVar()
        self.var_etiqueta = StringVar()
        for columna, (texto, variable, ancho) in enumerate(
            (
                ("Repeticiones por caso:", self.var_repeticiones, 6),
                ("Presupuesto (USD):", self.var_presupuesto, 10),
                ("Etiqueta:", self.var_etiqueta, 28),
            )
        ):
            ttk.Label(parametros, text=texto).grid(row=0, column=columna * 2, sticky="e", padx=8, pady=4)
            ttk.Entry(parametros, textvariable=variable, width=ancho).grid(
                row=0, column=columna * 2 + 1, sticky="w", padx=8, pady=4
            )
        ttk.Label(
            parametros,
            text="Menos de 5 repeticiones no permite concluir nada: el contraste exacto no puede "
            "bajar del nivel exigido y todos los casos saldrían como «indeterminado».",
            wraplength=1000,
            style="Aviso.TLabel",
        ).grid(row=1, column=0, columnspan=6, sticky="w", padx=8, pady=4)

        seleccion = ttk.LabelFrame(marco, text="Contra qué modelos (selección múltiple con Ctrl)")
        seleccion.pack(fill="both", expand=True, padx=8, pady=4)
        self._barra_filtros(seleccion).pack(fill="x", padx=8, pady=4)
        self.lista_modelos = tk.Listbox(seleccion, selectmode="extended", height=6, exportselection=False)
        self.lista_modelos.pack(fill="both", expand=True, padx=8, pady=4)
        self.lista_modelos.bind("<<ListboxSelect>>", lambda _: self._al_seleccionar_modelos())
        ttk.Label(seleccion, textvariable=self.var_seleccion).pack(anchor="w", padx=8, pady=(0, 4))

        acciones = ttk.Frame(marco)
        acciones.pack(fill="x", padx=8, pady=4)
        self.boton_lanzar = ttk.Button(acciones, text="Lanzar tirada", command=self.lanzar_tirada)
        self.boton_lanzar.pack(side="left")
        self.boton_cancelar = ttk.Button(acciones, text="Cancelar", command=self.cancelar_tirada, state="disabled")
        self.boton_cancelar.pack(side="left", padx=6)
        self.var_progreso = StringVar(value="Sin tirada en curso.")
        ttk.Label(acciones, textvariable=self.var_progreso).pack(side="left", padx=12)

        self.registro = tk.Text(marco, height=10, wrap="word", state="disabled")
        self.registro.pack(fill="both", expand=True, padx=8, pady=4)
        return marco

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

    # --------------------------------------------------------------- tiradas

    def _pestana_tiradas(self) -> ttk.Frame:
        marco = ttk.Frame(self)

        arriba = ttk.Frame(marco)
        arriba.pack(fill="x", padx=8, pady=4)
        ttk.Button(arriba, text="Refrescar", command=self.refrescar_tiradas).pack(side="left")

        columnas = ("run", "fecha", "suite", "modelos", "etiqueta", "estado", "faltan")
        self.tabla_tiradas = ttk.Treeview(marco, columns=columnas, show="headings", height=12)
        for columna, titulo, ancho in (
            ("run", "Tirada", 230),
            ("fecha", "Fecha", 140),
            ("suite", "Suite", 150),
            ("modelos", "Modelos", 230),
            ("etiqueta", "Etiqueta", 150),
            ("estado", "Estado", 90),
            ("faltan", "Garantías ausentes", 130),
        ):
            self.tabla_tiradas.heading(columna, text=titulo)
            self.tabla_tiradas.column(columna, width=ancho, anchor="w")
        self.tabla_tiradas.pack(fill="both", expand=True, padx=8, pady=4)
        self.tabla_tiradas.bind("<<TreeviewSelect>>", lambda _: self._mostrar_detalle_tirada())

        ttk.Label(marco, text="Detalle de la tirada seleccionada", style="Titulo.TLabel").pack(
            anchor="w", padx=8, pady=4
        )
        self.detalle_tirada = tk.Text(marco, height=12, wrap="word", state="disabled")
        self.detalle_tirada.pack(fill="both", expand=True, padx=8, pady=4)
        return marco

    # -------------------------------------------------------------- comparar

    def _pestana_comparar(self) -> ttk.Frame:
        marco = ttk.Frame(self)

        formulario = ttk.LabelFrame(marco, text="Qué dos tiradas se comparan")
        formulario.pack(fill="x", padx=8, pady=4)
        self.var_base = StringVar()
        self.var_candidata = StringVar()
        ttk.Label(formulario, text="Línea base:").grid(row=0, column=0, sticky="e", padx=8, pady=4)
        self.combo_base = ttk.Combobox(formulario, textvariable=self.var_base, width=60, state="readonly")
        self.combo_base.grid(row=0, column=1, sticky="we", padx=8, pady=4)
        ttk.Label(formulario, text="Candidata:").grid(row=1, column=0, sticky="e", padx=8, pady=4)
        self.combo_candidata = ttk.Combobox(formulario, textvariable=self.var_candidata, width=60, state="readonly")
        self.combo_candidata.grid(row=1, column=1, sticky="we", padx=8, pady=4)
        self.var_politica_cmp = StringVar()
        self._fila_ruta(formulario, 2, "Política de regresión:", self.var_politica_cmp, carpeta=False)
        formulario.columnconfigure(1, weight=1)

        acciones = ttk.Frame(marco)
        acciones.pack(fill="x", padx=8, pady=4)
        ttk.Button(acciones, text="Comparar", command=self.comparar).pack(side="left")
        self.boton_informe = ttk.Button(
            acciones, text="Abrir informe HTML", command=self.abrir_informe, state="disabled"
        )
        self.boton_informe.pack(side="left", padx=6)
        self.var_veredicto = StringVar(value="")
        ttk.Label(acciones, textvariable=self.var_veredicto, style="Titulo.TLabel").pack(side="left", padx=14)

        self.resumen = tk.Text(marco, height=9, wrap="word", state="disabled")
        self.resumen.pack(fill="x", padx=8, pady=4)

        columnas = ("caso", "metrica", "cambio", "direccion", "evidencia", "base", "cand", "p", "taxonomia")
        self.tabla_casos = ttk.Treeview(marco, columns=columnas, show="headings")
        for columna, titulo, ancho in (
            ("caso", "Caso", 170),
            ("metrica", "Métrica", 130),
            ("cambio", "Cambio", 150),
            ("direccion", "Dirección", 200),
            ("evidencia", "Evidencia", 150),
            ("base", "Base", 70),
            ("cand", "Cand.", 70),
            ("p", "p corregido", 90),
            ("taxonomia", "Qué cambió", 160),
        ):
            self.tabla_casos.heading(columna, text=titulo)
            self.tabla_casos.column(columna, width=ancho, anchor="w")
        self.tabla_casos.tag_configure("regresion", background="#fee2e2")
        self.tabla_casos.tag_configure("mejora", background="#dcfce7")
        self.tabla_casos.tag_configure("indeterminado", background="#f3f4f6")
        self.tabla_casos.pack(fill="both", expand=True, padx=8, pady=4)
        return marco

    # ------------------------------------------------------------ utilidades

    def _escribir(self, widget: tk.Text, texto: str, *, anexar: bool = False) -> None:
        widget.configure(state="normal")
        if not anexar:
            widget.delete("1.0", "end")
        widget.insert("end", texto)
        widget.see("end")
        widget.configure(state="disabled")

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
        elif mensaje.tipo == "tirada_terminada":
            self._tirada_terminada(mensaje.dato)
        elif mensaje.tipo == "comparacion":
            self._pintar_comparacion(mensaje.dato)
        elif mensaje.tipo == "error":
            texto, traza = mensaje.dato
            self.progreso["value"] = 0
            self.boton_lanzar.configure(state="normal")
            self.boton_cancelar.configure(state="disabled")
            self.log(f"ERROR: {texto}")
            self.log(traza.strip().splitlines()[-1] if traza else "")
            messagebox.showerror("Model_Drift", texto)

    # -------------------------------------------------------------- catálogo

    def refrescar_catalogo(self) -> None:
        self.estado.set("Consultando el catálogo del Broker…")

        base_url, token = self._credenciales()

        def trabajo() -> Mensaje:
            with self._cliente(base_url, token) as cliente:
                return Mensaje("catalogo", cliente.catalog(refresh=True))

        self._en_hilo(trabajo)

    def _pintar_catalogo(self, entradas: list[ModelCatalogEntry]) -> None:
        self.catalogo = entradas
        # Las opciones salen de lo que hay, no de una lista fija: si mañana
        # aparece un proveedor nuevo, el filtro lo ofrece sin tocar código.
        proveedores = ["todos", *sorted({item.provider for item in entradas})]
        despliegues = ["todos", REMOTOS, *sorted({item.deployment for item in entradas})]
        for combo in self._combos_proveedor:
            combo["values"] = proveedores
        for combo in self._combos_despliegue:
            combo["values"] = despliegues
        # Una elección anterior que ya no exista en el catálogo nuevo se cae.
        self.seleccionados &= {item.key for item in entradas}
        self._aplicar_filtros()

    # -------------------------------------------------------------- ejecutar

    def _modelos_elegidos(self) -> list[ModelReference]:
        """Los elegidos, estén o no visibles con el filtro puesto.

        Leer la selección del Listbox haría que filtrar después de elegir
        descartara modelos sin avisar.
        """
        return [self.config_app.model_reference(clave) for clave in sorted(self.seleccionados)]

    def lanzar_tirada(self) -> None:
        modelos = self._modelos_elegidos()
        if not modelos:
            messagebox.showwarning(
                "Model_Drift",
                "Elige al menos un modelo. Si la lista está vacía, consulta primero el catálogo "
                "en la pestaña «Modelos».",
            )
            return
        try:
            suite = load_suite(self.var_suite.get())
            repeticiones = int(self.var_repeticiones.get())
        except Exception as error:  # noqa: BLE001 - errores de entrada del usuario
            messagebox.showerror("Model_Drift", f"No se puede preparar la tirada:\n\n{error}")
            return

        if repeticiones < 5 and not messagebox.askyesno(
            "Pocas repeticiones",
            f"Con {repeticiones} repeticiones el contraste exacto no puede alcanzar significación:\n\n"
            "todos los casos saldrán como «indeterminado», por mucho que el modelo cambie.\n\n"
            "¿Lanzar de todas formas?",
        ):
            return

        base_url_previa, token_previo = self._credenciales()
        with self._cliente(base_url_previa, token_previo) as sonda:
            contrato = sonda.probe_contract()
        avisar = not contrato["learning_exclusion"] and not self.config_app.accept_learning_contamination
        if avisar and not messagebox.askyesno(
            "Esta tirada altera el Broker",
            "El Broker todavía no admite marcar tráfico como excluido del aprendizaje.\n\n"
            "Las invocaciones de esta tirada entrarán en las estadísticas de su router "
            "adaptativo: las que fallen bajarán el score de fiabilidad de los modelos "
            "medidos y pueden acercarlos a la cuarentena.\n\n"
            "¿Lanzar de todas formas?",
        ):
            return

        self.cancelar.clear()
        self.boton_lanzar.configure(state="disabled")
        self.boton_cancelar.configure(state="normal")
        self.progreso["value"] = 0
        self.log(f"Tirada sobre {suite.id} v{suite.version} ({suite.short_hash()}) con {len(suite.cases)} casos.")
        self.log(f"Modelos: {', '.join(modelo.key for modelo in modelos)}")

        # Fotografía del formulario, en el hilo principal. A partir de aquí el
        # hilo de trabajo no vuelve a mirar ningún widget.
        base_url, token = self._credenciales()
        ruta_relaciones = self.var_relaciones.get().strip()
        ruta_politica = self.var_politica.get().strip()
        texto_presupuesto = self.var_presupuesto.get().strip()
        etiqueta = self.var_etiqueta.get()

        def trabajo() -> Mensaje:
            relaciones = load_relations(ruta_relaciones) if ruta_relaciones else {}
            politica = RegressionPolicy.from_file(ruta_politica) if ruta_politica else RegressionPolicy()
            presupuesto = float(texto_presupuesto) if texto_presupuesto else None
            with self._cliente(base_url, token) as cliente, Database(self.config_app.database) as db:
                runner = Runner(
                    cliente,
                    suite,
                    db=db,
                    config=RunConfig(
                        repetitions=repeticiones,
                        temperature=self.config_app.temperature,
                        seed=self.config_app.seed,
                        budget_usd=presupuesto,
                        label=etiqueta,
                        allow_code_execution=self.config_app.allow_code_execution,
                        timeout_seconds=self.config_app.timeout_seconds,
                    ),
                    policy=politica,
                    relations=relaciones,
                )
                salida = runner.run(
                    modelos,
                    on_progress=lambda progreso: self.cola.put(Mensaje("progreso", progreso)),
                    should_cancel=self.cancelar.is_set,
                )
            return Mensaje("tirada_terminada", salida)

        self._en_hilo(trabajo)

    def cancelar_tirada(self) -> None:
        self.cancelar.set()
        self.log("Cancelación pedida. Se conserva todo lo medido hasta ahora.")

    def _tirada_terminada(self, salida: Any) -> None:
        self.boton_lanzar.configure(state="normal")
        self.boton_cancelar.configure(state="disabled")
        self.progreso["value"] = 100 if not salida.aborted_reason else self.progreso["value"]
        self.var_progreso.set("Sin tirada en curso.")
        self.log(f"Terminada: {salida.run_id} · {salida.invocations} invocaciones · {salida.cost_usd:.4f} USD")
        for linea in salida.manifest.reproducibility.declare():
            self.log(f"  {linea}")
        if salida.manifest.missing_guarantees:
            self.log("  Garantías NO disponibles en esta tirada:")
            for garantia in salida.manifest.missing_guarantees:
                self.log(f"   - {garantia}")
        if salida.aborted_reason:
            self.log(f"  INTERRUMPIDA: {salida.aborted_reason}")
        self.refrescar_tiradas()

    # --------------------------------------------------------------- tiradas

    def refrescar_tiradas(self) -> None:
        with Database(self.config_app.database) as db:
            filas = db.runs(limit=200)
        self.tabla_tiradas.delete(*self.tabla_tiradas.get_children())
        etiquetas: list[str] = []
        self._manifiestos: dict[str, dict[str, Any]] = {}
        for fila in filas:
            import json

            manifiesto = json.loads(fila["manifest_json"])
            self._manifiestos[fila["run_id"]] = manifiesto
            modelos = ", ".join(modelo.get("key", "?") for modelo in manifiesto.get("models", []))
            faltan = len(manifiesto.get("missing_guarantees", []))
            self.tabla_tiradas.insert(
                "",
                "end",
                iid=fila["run_id"],
                values=(
                    fila["run_id"],
                    str(fila["created_at"])[:19].replace("T", " "),
                    fila["suite_id"],
                    modelos,
                    fila["label"] or "—",
                    fila["status"],
                    faltan,
                ),
            )
            etiquetas.append(f"{fila['run_id']} · {modelos}" + (f" · {fila['label']}" if fila["label"] else ""))

        self.combo_base["values"] = etiquetas
        self.combo_candidata["values"] = etiquetas
        if len(etiquetas) >= 2:
            # La más reciente es la candidata natural; la anterior, la base.
            self.var_candidata.set(etiquetas[0])
            self.var_base.set(etiquetas[1])
        self.estado.set(f"{len(filas)} tiradas guardadas · {Path(self.config_app.database).resolve()}")

    def _mostrar_detalle_tirada(self) -> None:
        seleccion = self.tabla_tiradas.selection()
        if not seleccion:
            return
        manifiesto = self._manifiestos.get(seleccion[0], {})
        lineas = [f"Tirada {seleccion[0]}", ""]
        lineas.append(f"Suite      : {manifiesto.get('suite_id')} v{manifiesto.get('suite_version')}")
        lineas.append(f"Hash suite : {manifiesto.get('suite_hash', '')[:16]}")
        lineas.append(f"Repeticiones: {manifiesto.get('repetitions')}")
        lineas.append("")
        lineas.append("Reproducibilidad:")
        for clave, valor in (manifiesto.get("reproducibility") or {}).items():
            lineas.append(f"  {clave}: {valor}")
        lineas.append("")
        lineas.append("Modelos y huella de ejecución:")
        for modelo in manifiesto.get("models", []):
            lineas.append(f"  {modelo.get('key')}")
            componentes = (modelo.get("fingerprint") or {}).get("components", {})
            presentes = [c for c in componentes.values() if c.get("value")]
            ausentes = [c["name"] for c in componentes.values() if not c.get("value")]
            for componente in presentes:
                lineas.append(f"    {componente['name']}: {componente['value']} ({componente['state']})")
            if ausentes:
                lineas.append(f"    no disponibles: {', '.join(ausentes)}")
        faltantes = manifiesto.get("missing_guarantees") or []
        if faltantes:
            lineas.append("")
            lineas.append("GARANTÍAS NO DISPONIBLES EN ESTA TIRADA:")
            for garantia in faltantes:
                lineas.append(f"  - {garantia}")
        self._escribir(self.detalle_tirada, "\n".join(lineas))

    # -------------------------------------------------------------- comparar

    @staticmethod
    def _run_id(etiqueta: str) -> str:
        return etiqueta.split(" · ")[0].strip()

    def comparar(self) -> None:
        if not self.var_base.get() or not self.var_candidata.get():
            messagebox.showwarning("Model_Drift", "Elige las dos tiradas que quieres comparar.")
            return
        base, candidata = self._run_id(self.var_base.get()), self._run_id(self.var_candidata.get())
        if base == candidata:
            messagebox.showwarning(
                "Model_Drift",
                "Has elegido la misma tirada dos veces. Comparar una tirada consigo misma no mide nada.",
            )
            return

        ruta_politica = self.var_politica_cmp.get().strip()
        manifiesto = self._manifiestos.get(candidata, {})
        ruta_suite = manifiesto.get("suite_path")

        def trabajo() -> Mensaje:
            suite: Suite | None = None
            if ruta_suite and Path(str(ruta_suite)).exists():
                suite = load_suite(str(ruta_suite))
            politica = RegressionPolicy.from_file(ruta_politica) if ruta_politica else RegressionPolicy()
            with Database(self.config_app.database) as db:
                _, resultados_base = _load_run(db, base)
                _, resultados_cand = _load_run(db, candidata)
            informe = compare_runs(
                next(iter(resultados_base.values())),
                next(iter(resultados_cand.values())),
                suite=suite,
                policy=politica,
            )
            return Mensaje("comparacion", (informe, base, candidata, suite is None))

        self.estado.set("Comparando…")
        self._en_hilo(trabajo)

    def _pintar_comparacion(self, dato: tuple[ComparisonReport, str, str, bool]) -> None:
        informe, base, candidata, sin_suite = dato
        self.ultimo_informe = informe

        veredicto = VERDICT_LABELS.get(informe.verdict.verdict, "—") if informe.verdict else "—"
        self.var_veredicto.set(veredicto)

        texto = render_text_summary(informe)
        if sin_suite:
            texto += (
                "\n\nAVISO: no se pudo cargar la suite de la tirada candidata, así que ningún caso "
                "tiene evidencia y todo queda sin dirección. Es falta de información, no ausencia "
                "de cambio."
            )
        self._escribir(self.resumen, texto)

        self.tabla_casos.delete(*self.tabla_casos.get_children())
        ordenados = sorted(
            informe.outcomes,
            key=lambda item: (0 if item.is_actionable else 1 if item.is_regression else 2, item.case_id, item.metric),
        )
        for outcome in ordenados:
            etiqueta = ""
            if outcome.change_direction is ChangeDirection.regresion:
                etiqueta = "regresion"
            elif outcome.change_direction is ChangeDirection.mejora:
                etiqueta = "mejora"
            elif outcome.change_status is ChangeStatus.indeterminado:
                etiqueta = "indeterminado"
            self.tabla_casos.insert(
                "",
                "end",
                tags=(etiqueta,) if etiqueta else (),
                values=(
                    outcome.case_id,
                    outcome.metric,
                    STATUS_LABELS[outcome.change_status],
                    DIRECTION_LABELS[outcome.change_direction],
                    outcome.evidence_kind.value if outcome.evidence_kind else "—",
                    _num(outcome.baseline_score),
                    _num(outcome.candidate_score),
                    f"{outcome.p_value_corrected:.4f}" if outcome.p_value_corrected is not None else "—",
                    ", ".join(outcome.change_taxonomy),
                ),
            )

        destino = Path(self.config_app.reports_dir) / f"{base}__vs__{candidata}.html"
        manifiesto = None
        with Database(self.config_app.database) as db:
            fila = db.query_one("SELECT manifest_json FROM run WHERE run_id = ?", (candidata,))
        if fila is not None:
            from model_drift.manifest import RunManifest

            manifiesto = RunManifest.model_validate_json(fila["manifest_json"])
        self.ruta_informe = write_comparison(
            informe,
            destino,
            titulo=f"{base} → {candidata}",
            baseline_label=base,
            candidate_label=candidata,
            manifest=manifiesto,
        )
        self.boton_informe.configure(state="normal")
        self.estado.set(f"Informe generado: {self.ruta_informe}")

    def abrir_informe(self) -> None:
        """Abre el informe HTML en el navegador. Sin servidor: `file://` basta."""
        if self.ruta_informe is None or not self.ruta_informe.is_file():
            messagebox.showwarning("Model_Drift", "Todavía no hay ningún informe generado.")
            return
        webbrowser.open(self.ruta_informe.resolve().as_uri())

    # ---------------------------------------------------------------- cierre

    def _al_cerrar(self) -> None:
        if self.hilo is not None and self.hilo.is_alive():
            if not messagebox.askyesno(
                "Hay trabajo en curso",
                "Hay una operación ejecutándose. Si cierras ahora se perderá lo que no se haya "
                "guardado todavía.\n\n¿Cerrar de todas formas?",
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


def _tri(valor: bool | None) -> str:
    return "sí" if valor is True else "no" if valor is False else "?"


def _num(valor: float | None) -> str:
    return "—" if valor is None else f"{valor:.4g}"


def main(config: AppConfig | None = None) -> int:
    """Arranca la ventana. Cualquier fallo temprano se enseña, no se traga.

    Se lanza con `pythonw.exe`, que no tiene consola: si un error se imprimiera
    en vez de mostrarse, la aplicación «no abriría» sin decir por qué.

    La raíz se crea aquí y **explícitamente**. Dejar que `Toplevel` fabrique la
    suya por omisión abre una segunda ventana vacía llamada «tk» al lado de la
    buena, que es exactamente el tipo de cosa que hace desconfiar de una
    aplicación de escritorio.
    """
    raiz = Tk()
    # Windows identifica el proceso por el título de esta ventana aunque esté
    # oculta: sin esto, la aplicación aparece como «tk» en el administrador.
    raiz.title("Model_Drift")
    raiz.withdraw()
    try:
        ventana = ModelDriftGUI(config, master=raiz)
    except BrokerError as error:
        messagebox.showerror("Model_Drift", f"No se pudo contactar con el Broker:\n\n{error}")
        raiz.destroy()
        return 1
    except Exception as error:  # noqa: BLE001 - sin consola, un traceback perdido es un fallo mudo
        try:
            messagebox.showerror("Model_Drift", f"No se pudo arrancar:\n\n{error}\n\n{traceback.format_exc()}")
        except Exception:
            traceback.print_exc()
        raiz.destroy()
        return 1

    # Crear un Toplevel puede volver a mapear la raíz: se esconde otra vez.
    raiz.withdraw()
    ventana.deiconify()
    ventana.lift()
    ventana.focus_force()
    # `wait_window` procesa eventos hasta que la ventana se cierra, sin que la
    # clase tenga que saber nada de la raíz ni decidir cuándo matarla.
    raiz.wait_window(ventana)
    raiz.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
