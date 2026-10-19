"""Lanzamiento, cancelación y cierre de una tirada.

Todo lo que se lee del formulario se fotografía en el hilo principal antes de
arrancar el hilo de trabajo: a partir de ahí, el hilo no vuelve a mirar ningún
widget. Leer un `StringVar` desde otro hilo cuelga la aplicación.
"""
from __future__ import annotations

from tkinter import messagebox
from typing import Any

from model_drift.cases import load_suite
from model_drift.db import Database
from model_drift.gui.catalogo import CatalogoMixin
from model_drift.gui.comunes import Mensaje
from model_drift.gui.tiradas import TiradasMixin
from model_drift.metamorphic import load_relations
from model_drift.policy import RegressionPolicy
from model_drift.runner import RunConfig, Runner


class EjecucionMixin(CatalogoMixin, TiradasMixin):
    """Ciclo de vida de una tirada disparada desde la ventana."""

    def _leer_parametros(self) -> tuple[int, float | None] | None:
        """Repeticiones y presupuesto validados aquí, con mensajes que digan qué corregir.

        Antes el presupuesto se convertía dentro del hilo y llegaba como
        «could not convert string to float», y unas repeticiones negativas
        guardaban una tirada vacía como «completed».
        """
        texto = self.var_repeticiones.get().strip()
        try:
            repeticiones = int(texto)
        except ValueError:
            repeticiones = 0
        if repeticiones < 1:
            messagebox.showerror(
                "Model_Drift",
                f"Las repeticiones por caso deben ser un número entero de 1 en adelante (has puesto «{texto}»). "
                "Para poder concluir algo hacen falta al menos 5.",
                parent=self,
            )
            return None
        texto = self.var_presupuesto.get().strip().replace(",", ".")
        if not texto:
            return repeticiones, None
        try:
            presupuesto = float(texto)
        except ValueError:
            presupuesto = -1.0
        if presupuesto < 0:
            messagebox.showerror(
                "Model_Drift",
                f"El presupuesto debe ser una cantidad en USD igual o mayor que 0, p. ej. 0.50 "
                f"(has puesto «{self.var_presupuesto.get().strip()}»). Déjalo vacío para no poner tope.",
                parent=self,
            )
            return None
        return repeticiones, presupuesto

    def lanzar_tirada(self) -> None:
        if self._ocupado():
            return
        modelos = self._modelos_elegidos()
        if not modelos:
            messagebox.showwarning(
                "Model_Drift",
                "Elige al menos un modelo. Si la lista está vacía, consulta primero el catálogo "
                "en la pestaña «Modelos».",
                parent=self,
            )
            return
        parametros = self._leer_parametros()
        if parametros is None:
            return
        repeticiones, presupuesto = parametros
        try:
            suite = load_suite(self.var_suite.get())
        except Exception as error:  # noqa: BLE001 - errores de entrada del usuario
            messagebox.showerror("Model_Drift", f"No se puede preparar la tirada:\n\n{error}", parent=self)
            return

        if repeticiones < 5 and not messagebox.askyesno(
            "Pocas repeticiones",
            f"Con {repeticiones} repeticiones el contraste exacto no puede alcanzar significación:\n\n"
            "todos los casos saldrán como «indeterminado», por mucho que el modelo cambie.\n\n"
            "¿Lanzar de todas formas?",
            parent=self,
        ):
            return

        base_url_previa, token_previo = self._credenciales()
        try:
            with self._cliente(base_url_previa, token_previo) as sonda:
                contrato = sonda.probe_contract()
        except Exception as error:  # noqa: BLE001 - sin consola, una excepción en un callback de Tk es muda
            messagebox.showerror("Model_Drift", f"No se pudo consultar el Broker:\n\n{error}", parent=self)
            return
        avisar = not contrato["learning_exclusion"] and not self.config_app.accept_learning_contamination
        if avisar and not messagebox.askyesno(
            "Esta tirada altera el Broker",
            "El Broker todavía no admite marcar tráfico como excluido del aprendizaje.\n\n"
            "Las invocaciones de esta tirada entrarán en las estadísticas de su router "
            "adaptativo: las que fallen bajarán el score de fiabilidad de los modelos "
            "medidos y pueden acercarlos a la cuarentena.\n\n"
            "¿Lanzar de todas formas?",
            parent=self,
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
        etiqueta = self.var_etiqueta.get()

        def trabajo() -> Mensaje:
            relaciones = load_relations(ruta_relaciones) if ruta_relaciones else {}
            politica = RegressionPolicy.from_file(ruta_politica) if ruta_politica else RegressionPolicy()
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
                    core=self.config_app.core_client(),
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
        fallida = salida.status == "failed"
        # La barra llena solo si de verdad se midió algo hasta el final.
        if salida.invocations == 0 or fallida:
            self.progreso["value"] = 0
        elif not salida.aborted_reason:
            self.progreso["value"] = 100
        self.var_progreso.set("Sin tirada en curso.")
        titulo = "FALLIDA" if fallida else "Terminada" if not salida.aborted_reason else "Interrumpida"
        self.log(f"{titulo}: {salida.run_id} · {salida.invocations} invocaciones · {salida.cost_usd:.4f} USD")
        resumen_fallos = salida.failure_summary()
        if resumen_fallos:
            self.log(f"  FALLOS: {resumen_fallos}")
        for aviso in salida.core_notices:
            self.log(f"  AVISO: {aviso}")
        if salida.invocations == 0:
            self.log("  No se hizo ninguna invocación: esta tirada no midió nada.")
        if not fallida:
            for linea in salida.manifest.reproducibility.declare():
                self.log(f"  {linea}")
        if salida.manifest.missing_guarantees:
            self.log("  Garantías NO disponibles en esta tirada:")
            for garantia in salida.manifest.missing_guarantees:
                self.log(f"   - {garantia}")
        if salida.aborted_reason:
            self.log(f"  INTERRUMPIDA: {salida.aborted_reason}")
        self.refrescar_tiradas()
        if fallida or salida.invocations == 0:
            self.estado.set(f"La tirada {salida.run_id} no midió el modelo")
            messagebox.showwarning(
                "La tirada no midió nada",
                f"{salida.run_id}: {resumen_fallos or 'no se hizo ninguna invocación'}."
                + (f"\n\n{salida.aborted_reason}" if salida.aborted_reason else "")
                + "\n\nNo sirve para comparar. Si el motivo es AUTH, copia el token vigente de la "
                "ventana del Broker en la pestaña «Modelos» y relanza.",
                parent=self,
            )
