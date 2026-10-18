"""Listado de tiradas guardadas y detalle del manifiesto.

El detalle enseña la reproducibilidad y, sobre todo, las garantías que la
tirada NO pudo dar. Lo que el motor no puede garantizar la interfaz lo enseña:
esconderlo en un JSON convierte una medición honesta en un número bonito.
"""
from __future__ import annotations

import json
from pathlib import Path

from model_drift.cli import fecha_local
from model_drift.db import Database
from model_drift.gui.base import VentanaBase


class TiradasMixin(VentanaBase):
    """Refresco del listado y pintado del detalle seleccionado."""

    def refrescar_tiradas(self) -> None:
        with Database(self.config_app.database) as db:
            filas = db.runs(limit=200)
            medidos: dict[str, list[str]] = {}
            for par in db.query("SELECT DISTINCT run_id, model_key FROM case_result ORDER BY model_key"):
                medidos.setdefault(par["run_id"], []).append(par["model_key"])
        self.tabla_tiradas.delete(*self.tabla_tiradas.get_children())
        self._manifiestos = {}
        #: Etiqueta del combo -> (tirada, modelo). Una entrada por modelo medido:
        #: comparar una tirada de dos modelos ya no elige uno a escondidas, y
        #: una tirada sin resultados (vacía, cancelada pronto o fallida) no se ofrece.
        self._opciones_comparar = {}
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
                    fecha_local(fila["created_at"]),
                    fila["suite_id"],
                    modelos,
                    fila["label"] or "—",
                    fila["status"],
                    faltan,
                ),
            )
            if fila["status"] == "failed":
                continue
            for clave in medidos.get(fila["run_id"], []):
                etiqueta = f"{fila['run_id']} · {clave}" + (f" · {fila['label']}" if fila["label"] else "")
                self._opciones_comparar[etiqueta] = (fila["run_id"], clave)

        etiquetas = list(self._opciones_comparar)
        self.combo_base["values"] = etiquetas
        self.combo_candidata["values"] = etiquetas
        for variable in (self.var_base, self.var_candidata):
            if variable.get() not in self._opciones_comparar:
                variable.set("")
        if len(etiquetas) >= 2 and not (self.var_base.get() and self.var_candidata.get()):
            # La más reciente es la candidata natural; la anterior, la base.
            self.var_candidata.set(etiquetas[0])
            self.var_base.set(etiquetas[1])
        self.estado.set(f"{len(filas)} tiradas guardadas · {Path(self.config_app.database).resolve()}")

    def _mostrar_detalle_tirada(self) -> None:
        seleccion = self.tabla_tiradas.selection()
        if not seleccion:
            return
        run_id = seleccion[0]
        manifiesto = self._manifiestos.get(run_id, {})
        with Database(self.config_app.database) as db:
            total = db.query_one(
                "SELECT COUNT(*) AS n, COALESCE(SUM(ok = 1 AND output_text <> ''), 0) AS con_salida "
                "FROM invocation WHERE run_id = ?",
                (run_id,),
            )
            errores = db.query(
                "SELECT COALESCE(error_code, 'SIN_SALIDA') AS codigo, COALESCE(error_message, '') AS mensaje, "
                "COUNT(*) AS veces FROM invocation WHERE run_id = ? AND NOT (ok = 1 AND output_text <> '') "
                "GROUP BY 1, 2 ORDER BY veces DESC",
                (run_id,),
            )
        invocaciones = int(total["n"]) if total else 0
        con_salida = int(total["con_salida"]) if total else 0
        lineas = [f"Tirada {run_id}", ""]
        if invocaciones == 0:
            lineas.append("ESTA TIRADA NO MIDIÓ NADA: no se hizo ninguna invocación.")
        else:
            lineas.append(
                f"Invocaciones: {invocaciones} · con salida: {con_salida} · sin salida: {invocaciones - con_salida}"
            )
        if con_salida == 0 and invocaciones:
            lineas.append("NINGUNA invocación produjo salida: esta tirada no midió el modelo.")
        for error in errores:
            lineas.append(f"  {error['veces']} × {error['codigo']}: {error['mensaje'] or '(sin mensaje)'}")
        lineas.append("")
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
        notas = manifiesto.get("notes") or []
        if notas:
            lineas.append("")
            lineas.append("Notas:")
            lineas.extend(f"  - {nota}" for nota in notas)
        self._escribir(self.detalle_tirada, "\n".join(lineas))
