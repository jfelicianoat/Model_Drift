"""Catálogo de modelos: consulta al Broker, filtros y selección.

El filtrado y la selección viven juntos porque son el mismo problema: con 162
modelos, lo único que hace usable la lista es acotarla, y lo único que hace
fiable la elección es que un filtro no la borre por detrás.
"""
from __future__ import annotations

from pathlib import Path

from model_drift.broker_contracts import ModelCatalogEntry, ModelReference
from model_drift.calibration.runner import revalidate_changed
from model_drift.db import Database
from model_drift.gui.base import VentanaBase
from model_drift.gui.comunes import REMOTOS, Mensaje, _tri


class CatalogoMixin(VentanaBase):
    """Consulta del catálogo, filtros compartidos y conjunto de elegidos."""

    # --------------------------------------------------------------- filtros

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
            if self.tabla_modelos.exists(entrada.key):
                continue
            self.tabla_modelos.insert(
                "",
                "end",
                iid=entrada.key,
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
        for entrada in visibles:
            self.lista_modelos.insert("end", entrada.key)

        self._sincronizar_selecciones()
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
        self._sincronizar_selecciones()

    def _al_seleccionar_en_tabla(self) -> None:
        """La tabla de «Modelos» elige igual que la lista de «Ejecutar tirada».

        Antes admitía selección sin efecto: marcar filas ahí no lanzaba nada.
        """
        visibles = {item.key for item in self._modelos_filtrados()}
        elegidos = set(self.tabla_modelos.selection()) & visibles
        if (self.seleccionados & visibles) == elegidos:
            return  # eco de nuestra propia `selection_set`
        self.seleccionados = (self.seleccionados - visibles) | elegidos
        self._sincronizar_selecciones()

    def _sincronizar_selecciones(self) -> None:
        """Pinta `seleccionados` en la lista y en la tabla, que muestran lo mismo."""
        self.lista_modelos.selection_clear(0, "end")
        for indice, clave in enumerate(self.lista_modelos.get(0, "end")):
            if clave in self.seleccionados:
                self.lista_modelos.selection_set(indice)
        marcadas = [clave for clave in self.tabla_modelos.get_children() if clave in self.seleccionados]
        if set(self.tabla_modelos.selection()) != set(marcadas):
            self.tabla_modelos.selection_set(marcadas)
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

    # -------------------------------------------------------------- catálogo

    def refrescar_catalogo(self) -> None:
        if self._ocupado():
            return
        self.estado.set("Consultando el catálogo del Broker…")

        base_url, token = self._credenciales()
        self.cancelar.clear()

        def trabajo() -> Mensaje:
            with self._cliente(base_url, token) as cliente:
                entries = cliente.catalog(refresh=True)
                self.cola.put(Mensaje("catalogo", entries))
                with Database(self.config_app.database) as db:
                    reports = revalidate_changed(
                        cliente, db, self.config_app.reports_dir, cancel=self.cancelar,
                        progress=lambda text: self.cola.put(Mensaje("log", text)),
                    )
                if reports:
                    report = reports[-1]
                    return Mensaje("calibration", (report, Path(report.artifact_html_path or "")))
                return Mensaje("log", "Catálogo actualizado.")

        self._en_hilo(trabajo)

    def _pintar_catalogo(self, entradas: list[ModelCatalogEntry]) -> None:
        self.catalogo = entradas
        if hasattr(self, "cal_teacher_combo"):
            self.cal_teacher_combo["values"] = sorted(item.key for item in entradas if item.dispatchable)
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

    # ------------------------------------------------------------- selección

    def _modelos_elegidos(self) -> list[ModelReference]:
        """Los elegidos, estén o no visibles con el filtro puesto.

        Leer la selección del Listbox haría que filtrar después de elegir
        descartara modelos sin avisar.
        """
        return [self.config_app.model_reference(clave) for clave in sorted(self.seleccionados)]
