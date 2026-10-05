"""Construcción de las pestañas de la ventana.

Son funciones y no métodos a propósito: la maquetación es la mitad del código
de una interfaz Tkinter y no tiene nada que ver con el comportamiento. Cada
módulo deja puestos en la ventana los widgets que el resto del código necesita
—el contrato está anotado en `gui.base.VentanaBase`— y devuelve el marco que se
cuelga del `Notebook`.
"""
from __future__ import annotations

from model_drift.gui.pestanas.calibration import construir as construir_calibration
from model_drift.gui.pestanas.comparar import construir as construir_comparar
from model_drift.gui.pestanas.ejecutar import construir as construir_ejecutar
from model_drift.gui.pestanas.filtros import barra_filtros
from model_drift.gui.pestanas.modelos import construir as construir_modelos
from model_drift.gui.pestanas.tiradas import construir as construir_tiradas

__all__ = [
    "barra_filtros",
    "construir_comparar",
    "construir_calibration",
    "construir_ejecutar",
    "construir_modelos",
    "construir_tiradas",
]
