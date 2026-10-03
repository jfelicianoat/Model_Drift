"""Comparación de dos tiradas y generación del informe HTML.

El informe sigue siendo un fichero: Tkinter no pinta HTML y fingirlo con
etiquetas sería peor. La ventana lo genera y lo abre en el navegador, que para
*leer* un documento funciona perfectamente.
"""
from __future__ import annotations

import webbrowser
from pathlib import Path
from tkinter import messagebox

from model_drift.cases import Suite, load_suite
from model_drift.cli import _load_run
from model_drift.comparison import ComparisonReport
from model_drift.comparison.service import compare_configured
from model_drift.db import Database
from model_drift.domain import ChangeDirection, ChangeStatus
from model_drift.gui.base import VentanaBase
from model_drift.gui.comunes import Mensaje, _num
from model_drift.policy import RegressionPolicy
from model_drift.report import (
    DIRECTION_LABELS,
    STATUS_LABELS,
    VERDICT_LABELS,
    render_text_summary,
    write_comparison,
)


class ComparacionMixin(VentanaBase):
    """Elección de las dos tiradas, contraste y volcado del resultado."""

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
        broker_url, broker_token = self._credenciales()

        def trabajo() -> Mensaje:
            suite: Suite | None = None
            if ruta_suite and Path(str(ruta_suite)).exists():
                suite = load_suite(str(ruta_suite))
            politica = RegressionPolicy.from_file(ruta_politica) if ruta_politica else RegressionPolicy()
            with Database(self.config_app.database) as db:
                _, resultados_base = _load_run(db, base)
                _, resultados_cand = _load_run(db, candidata)
            informe = compare_configured(
                next(iter(resultados_base.values())),
                next(iter(resultados_cand.values())),
                suite=suite,
                policy=politica,
                config=self.config_app,
                client_factory=lambda: self._cliente(broker_url, broker_token),
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

        self._pintar_casos(informe)
        self._generar_informe(informe, base, candidata)

    @staticmethod
    def _semantic_priority(informe: ComparisonReport, case_id: str) -> int:
        review = informe.semantic_review
        if review is None or review.config.shadow_mode:
            return 2
        assessment = next((item for item in review.assessments if item.case_id == case_id), None)
        return {"high": 0, "review": 1, "normal": 2}[assessment.priority] if assessment else 2

    def _pintar_casos(self, informe: ComparisonReport) -> None:
        """Vuelca la tabla de casos, con los accionables primero."""
        self.tabla_casos.delete(*self.tabla_casos.get_children())
        ordenados = sorted(
            informe.outcomes,
            key=lambda item: (
                0 if item.is_actionable else 1 if item.is_regression else 2,
                self._semantic_priority(informe, item.case_id), item.case_id, item.metric,
            ),
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

    def _generar_informe(self, informe: ComparisonReport, base: str, candidata: str) -> None:
        """Escribe el HTML en disco y habilita el botón que lo abre."""
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
