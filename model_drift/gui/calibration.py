"""La opción «Probar System 1» usa el mismo motor que la CLI."""

from __future__ import annotations

import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox

from model_drift.calibration.models import CalibrationPlan, CalibrationReport, EvaluationTarget
from model_drift.calibration.runner import export_distilled, run_calibration, save_artifacts
from model_drift.db import Database
from model_drift.gui.base import VentanaBase
from model_drift.gui.comunes import Mensaje
from model_drift.report import render_calibration_summary


class CalibrationMixin(VentanaBase):
    def elegir_plan_calibracion(self) -> None:
        path = filedialog.askopenfilename(
            parent=self, title="Plan de calibración", filetypes=[("JSON", "*.json"), ("Todos", "*.*")]
        )
        if path:
            self.var_cal_plan.set(path)

    def probar_system1(self) -> None:
        if self._ocupado():
            return
        path = self.var_cal_plan.get().strip()
        if not path:
            messagebox.showwarning(
                "Model_Drift", "Elige un plan de calibración con los casos y la configuración.", parent=self
            )
            return
        teacher = self.var_cal_teacher.get().strip()
        provider, model = self.var_cal_provider.get().strip(), self.var_cal_model.get().strip()
        broker_url, token = self._credenciales()
        self.cancelar.clear()

        def work() -> Mensaje:
            plan = CalibrationPlan.load(path)
            if teacher:
                plan.config.teacher_task = self.config_app.model_reference(teacher)
                plan.config.teacher_judge = None
            if provider:
                plan.config.learner = EvaluationTarget.model_validate({"provider": provider, "model": model or None})
            with self._cliente(broker_url, token) as client, Database(self.config_app.database) as db:
                # El avance va a esta pestaña: aquí es donde se está mirando.
                report = run_calibration(
                    client, plan, db, cancel=self.cancelar,
                    progress=lambda text: self.cola.put(Mensaje("log_cal", text)),
                )
                destination = Path(self.config_app.reports_dir) / report.run_id
                export_distilled(db, report, destination.with_suffix(".references.json"))
                html = save_artifacts(db, report, destination.with_suffix(".json"), destination.with_suffix(".html"))
            return Mensaje("calibration", (report, html))

        self.estado.set("Probando System 1…")
        self._escribir(self.cal_summary, "Probando System 1…\n")
        self._en_hilo(work)

    def _avance_calibracion(self, texto: str) -> None:
        self._escribir(self.cal_summary, f"{texto}\n", anexar=True)
        self.estado.set(f"Probando System 1… {texto}")
        self.log(texto)

    def _pintar_calibracion(self, data: tuple[CalibrationReport, Path]) -> None:
        report, html = data
        self._escribir(self.cal_summary, render_calibration_summary(report) + f"\n\nInforme: {html}")
        self.estado.set(f"Prueba {report.run_id}: {report.status}")
        self.cal_report_path = html

    def abrir_informe_calibracion(self) -> None:
        if self.cal_report_path is not None:
            webbrowser.open(self.cal_report_path.resolve().as_uri())
