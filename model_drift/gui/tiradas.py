"""Listado de tiradas guardadas y detalle del manifiesto.

El detalle enseña la reproducibilidad y, sobre todo, las garantías que la
tirada NO pudo dar. Lo que el motor no puede garantizar la interfaz lo enseña:
esconderlo en un JSON convierte una medición honesta en un número bonito.
"""
from __future__ import annotations

import json
from pathlib import Path

from model_drift.db import Database
from model_drift.gui.base import VentanaBase


class TiradasMixin(VentanaBase):
    """Refresco del listado y pintado del detalle seleccionado."""

    def refrescar_tiradas(self) -> None:
        with Database(self.config_app.database) as db:
            filas = db.runs(limit=200)
        self.tabla_tiradas.delete(*self.tabla_tiradas.get_children())
        etiquetas: list[str] = []
        self._manifiestos = {}
        for fila in filas:
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
