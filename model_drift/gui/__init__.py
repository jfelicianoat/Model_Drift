"""Interfaz gráfica de escritorio.

Tkinter, que viene con Python. Sin dependencias nuevas, sin servidor, sin
puerto y sin navegador: la aplicación abre, lee su SQLite local y ya está. Es
la única interfaz gráfica del proyecto; hubo una web y se retiró, porque un
servidor y un puerto son tres modos de fallo para algo que se usa en local.

El paquete está partido por zonas y no por capas: cada módulo reúne lo que
cambia junto cuando se toca una pestaña.

- `comunes`   — mensajes entre hilos y formateo compartido.
- `base`      — estado de la ventana, hilos, cliente del Broker y rutas.
- `catalogo`  — consulta al Broker, filtros y selección de modelos.
- `ejecucion` — lanzar, cancelar y cerrar una tirada.
- `tiradas`   — listado de tiradas guardadas y detalle del manifiesto.
- `comparacion` — contraste de dos tiradas e informe HTML.
- `pestanas`  — la maquetación de cada pestaña, en funciones.
- `ventana`   — `ModelDriftGUI`: compone lo anterior y reparte los mensajes.

Este módulo es la fachada: lo que importa el resto del proyecto (y las pruebas)
entra por aquí, así que mover código entre los módulos de arriba no rompe a
nadie.
"""
from __future__ import annotations

import traceback
from tkinter import Tk, messagebox

from model_drift.broker_contracts import BrokerError
from model_drift.config import AppConfig
from model_drift.gui.comunes import REMOTOS, Mensaje
from model_drift.gui.ventana import ModelDriftGUI

__all__ = ["REMOTOS", "Mensaje", "ModelDriftGUI", "main", "messagebox"]


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
        # Se busca en este módulo y no en el de la ventana para que las pruebas
        # puedan sustituir la clase sin tocar el arranque.
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
