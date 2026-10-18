"""Pruebas de la interfaz de escritorio.

Una interfaz gráfica no se prueba a golpe de clic, pero sí se puede comprobar
lo que importa: que la ventana se construye entera, que el hilo de trabajo
nunca toca los widgets, y que las reglas de honestidad del motor llegan a la
pantalla en vez de quedarse en el JSON.

Las pruebas se saltan solas si el entorno no tiene servidor gráfico (un runner
de Linux sin X). En Windows, que es donde se usa, corren siempre.
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import tkinter as tk
from pathlib import Path

import pytest
import yaml

from model_drift.broker_contracts import ModelReference
from model_drift.cases import load_suite
from model_drift.config import AppConfig
from model_drift.db import Database
from model_drift.runner import RunConfig, Runner, RunProgress
from tests.broker_fake import FakeBroker

MODELO = ModelReference(provider="ollama", deployment="local", model="qwen3:6b")


@pytest.fixture()
def entorno(tmp_path: Path):  # type: ignore[no-untyped-def]
    """Configuración con dos tiradas reales ya guardadas."""
    ruta_suite = tmp_path / "suite.yaml"
    ruta_suite.write_text(
        yaml.safe_dump(
            {
                "id": "s",
                "cases": [
                    {"id": "c1", "prompt": "uno", "assertions": [{"type": "json_valid"}]},
                    {"id": "c2", "prompt": "dos", "assertions": [{"type": "json_valid"}]},
                ],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    config = AppConfig(
        database=str(tmp_path / "md.sqlite"),
        reports_dir=str(tmp_path / "informes"),
        suites_dir=str(tmp_path),
        relations_dir="",
    )
    suite = load_suite(ruta_suite)

    ids: list[str] = []
    with Database(config.database) as db:
        for texto in ('{"a": 1}', "ya no devuelvo JSON"):
            broker = FakeBroker()
            broker.add_model(MODELO.key)
            broker.respond_fixed(MODELO.key, texto)
            with broker.client() as cliente:
                salida = Runner(cliente, suite, db=db, config=RunConfig(repetitions=6)).run([MODELO])
                ids.append(salida.run_id)
    return config, ruta_suite, ids


@pytest.fixture()
def dialogos(monkeypatch):  # type: ignore[no-untyped-def]
    """Neutraliza los diálogos modales y registra lo que habrían dicho.

    Sin esto, un `messagebox` en una prueba bloquea el proceso para siempre:
    no hay nadie que pulse «Aceptar». Es lo que colgaba la batería entera.
    """
    from model_drift.gui import messagebox

    registrados: dict[str, list[str]] = {"info": [], "aviso": [], "error": [], "pregunta": []}
    monkeypatch.setattr(messagebox, "showinfo", lambda t, m, **_: registrados["info"].append(m))
    monkeypatch.setattr(messagebox, "showwarning", lambda t, m, **_: registrados["aviso"].append(m))
    monkeypatch.setattr(messagebox, "showerror", lambda t, m, **_: registrados["error"].append(m))
    monkeypatch.setattr(messagebox, "askyesno", lambda t, m, **_: registrados["pregunta"].append(m) or True)
    return registrados


@pytest.fixture(scope="session")
def raiz():  # type: ignore[no-untyped-def]
    """Una sola raíz de Tk para toda la sesión.

    Una por sesión y ni una más. Crear y destruir intérpretes Tcl repetidos los
    corrompe —«invalid command name tcl_findLibrary», o un init.tcl que de
    pronto no se puede leer—, y eso incluye el sondeo de «¿hay pantalla?»: si se
    hace creando y destruyendo una raíz de prueba, ya se ha gastado la primera.
    Por eso la comprobación se hace aquí, con la raíz que se va a usar de verdad.

    También es más fiel: la aplicación real crea exactamente una raíz.
    """
    try:
        root = tk.Tk()
    except Exception as error:  # pragma: no cover - runner sin servidor gráfico
        pytest.skip(f"el entorno no tiene servidor gráfico: {error}")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture()
def ventana(entorno, dialogos, raiz):  # type: ignore[no-untyped-def]
    from model_drift.gui import ModelDriftGUI

    config, ruta_suite, ids = entorno
    app = ModelDriftGUI(config, master=raiz)
    app.withdraw()  # no molestar durante la ejecución de las pruebas
    yield app, ruta_suite, ids
    app.cerrar()


# --------------------------------------------------------------------------
# Construcción
# --------------------------------------------------------------------------


def test_la_ventana_se_construye_entera(ventana) -> None:  # type: ignore[no-untyped-def]
    app, _, _ = ventana
    assert app.title().startswith("Model_Drift")
    assert [app.pestanas.tab(i, "text").strip() for i in range(len(app.pestanas.tabs()))] == [
        "Modelos",
        "Ejecutar tirada",
        "Tiradas",
        "Comparar",
        "Probar System 1",
    ]


def test_calibration_requires_a_plan(ventana, dialogos) -> None:  # type: ignore[no-untyped-def]
    app, _, _ = ventana
    app.probar_system1()
    assert "plan de calibración" in dialogos["aviso"][-1]


def test_calibration_worker_captures_widgets_and_paints_report(ventana, monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from tests.test_calibration import EvaluationBroker, _plan

    app, _, _ = ventana
    plan_path = tmp_path / "calibration.json"
    plan_path.write_text(_plan(n=4).model_dump_json(), encoding="utf-8")
    app.var_cal_plan.set(str(plan_path))
    app.var_cal_teacher.set("ollama/local/qwen3:8b")
    app.var_cal_provider.set("ollama_system1")
    app.var_cal_model.set("native")
    jobs = []
    monkeypatch.setattr(app, "_en_hilo", jobs.append)
    monkeypatch.setattr(app, "_cliente", lambda *args: EvaluationBroker().client())
    app.probar_system1()

    def forbidden():  # type: ignore[no-untyped-def]
        raise AssertionError("el hilo leyó un widget")

    for variable in (app.var_cal_plan, app.var_cal_teacher, app.var_cal_provider, app.var_cal_model):
        monkeypatch.setattr(variable, "get", forbidden)
    message = jobs[0]()
    assert message.tipo == "calibration"
    report, html = message.dato
    assert report.config.teacher_task.model == "qwen3:8b"
    assert html.exists() and report.status == "indeterminado"
    app._atender(message)
    assert "Indeterminado" in app.cal_summary.get("1.0", "end")
    opened = []
    monkeypatch.setattr("model_drift.gui.calibration.webbrowser.open", opened.append)
    app.abrir_informe_calibracion()
    assert opened == [html.resolve().as_uri()]


def test_calibration_plan_dialog(ventana, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    app, _, _ = ventana
    monkeypatch.setattr("model_drift.gui.calibration.filedialog.askopenfilename", lambda **kwargs: "plan.json")
    app.elegir_plan_calibracion()
    assert app.var_cal_plan.get() == "plan.json"


def test_la_ventana_arranca_con_las_tiradas_ya_cargadas(ventana) -> None:  # type: ignore[no-untyped-def]
    app, _, ids = ventana
    filas = set(app.tabla_tiradas.get_children())
    assert filas == set(ids)
    # La más reciente se propone como candidata y la anterior como base: es la
    # comparación que quiere hacer alguien el 95 % de las veces.
    assert app._opciones_comparar[app.var_candidata.get()] == (ids[1], MODELO.key)
    assert app._opciones_comparar[app.var_base.get()] == (ids[0], MODELO.key)


def test_el_valor_por_defecto_de_repeticiones_es_utilizable(ventana) -> None:  # type: ignore[no-untyped-def]
    """Menos de 5 no permite concluir nada, así que no puede ser el defecto."""
    app, _, _ = ventana
    assert int(app.var_repeticiones.get()) >= 5


# --------------------------------------------------------------------------
# El detalle enseña lo que el motor NO pudo garantizar
# --------------------------------------------------------------------------


def test_el_detalle_de_la_tirada_enseña_la_huella_y_su_procedencia(ventana) -> None:  # type: ignore[no-untyped-def]
    """Cada componente con su estado: verificado no es lo mismo que declarado."""
    app, _, ids = ventana
    app.tabla_tiradas.selection_set(ids[0])
    app._mostrar_detalle_tirada()
    texto = app.detalle_tirada.get("1.0", "end")
    assert "reproducibilidad_exacta" in texto
    assert "no_demostrada" in texto, "nadie ha reproducido una tirada anterior"
    assert "chat_template" in texto and "verificado" in texto
    assert "declarado" in texto
    # Lo que el Broker no pudo dar sigue saliendo como ausente.
    assert "no disponibles" in texto and "system_fingerprint" in texto


# --------------------------------------------------------------------------
# Comparación
# --------------------------------------------------------------------------


def test_comparar_pinta_la_tabla_y_genera_el_informe(ventana) -> None:  # type: ignore[no-untyped-def]
    app, _, ids = ventana
    app.comparar()
    _drenar(app)

    assert app.ultimo_informe is not None
    filas = app.tabla_casos.get_children()
    assert filas, "la tabla de casos quedó vacía"
    assert app.ruta_informe is not None and app.ruta_informe.is_file()
    assert str(app.boton_informe["state"]) == "normal"

    # La suite se recupera del manifiesto: los casos tienen evidencia.
    evidencias = {app.tabla_casos.item(fila, "values")[4] for fila in filas}
    assert "oraculo_determinista" in evidencias


def test_la_tabla_nunca_dice_regresion_sin_evidencia(ventana) -> None:  # type: ignore[no-untyped-def]
    """La misma regla del informe HTML, en la pantalla."""
    app, _, _ = ventana
    app.comparar()
    _drenar(app)
    for fila in app.tabla_casos.get_children():
        valores = app.tabla_casos.item(fila, "values")
        direccion, evidencia = valores[3], valores[4]
        if evidencia == "—":
            assert "regresión" not in direccion
            assert direccion != "mejora"


def test_comparar_una_tirada_consigo_misma_se_rechaza(ventana, dialogos) -> None:  # type: ignore[no-untyped-def]
    app, _, ids = ventana
    app.var_base.set(app.var_candidata.get())
    app.comparar()
    assert dialogos["aviso"] and "consigo misma" in dialogos["aviso"][0]


# --------------------------------------------------------------------------
# Concurrencia
# --------------------------------------------------------------------------


def test_el_progreso_llega_por_la_cola_y_no_desde_el_hilo(ventana) -> None:  # type: ignore[no-untyped-def]
    """Los widgets solo se tocan en `_bombear_cola`; el hilo solo encola."""
    from model_drift.gui import Mensaje

    app, _, _ = ventana
    progreso = RunProgress(
        model_key="ollama/local/qwen3:6b",
        case_id="c1",
        case_index=1,
        total_cases=4,
        repetition=2,
        total_repetitions=5,
        invocations=8,
        cost_usd=0.0123,
    )
    app.cola.put(Mensaje("progreso", progreso))
    _drenar(app)
    assert app.progreso["value"] > 0
    assert "c1" in app.var_progreso.get()
    assert "0.0123" in app.var_progreso.get()


def test_un_error_del_hilo_no_mata_la_ventana(ventana, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from model_drift.gui import Mensaje

    app, _, _ = ventana
    errores: list[str] = []
    monkeypatch.setattr("model_drift.gui.messagebox.showerror", lambda titulo, texto, **_: errores.append(texto))
    app.cola.put(Mensaje("error", ("el Broker no responde", "Traceback...\nBrokerError: nope")))
    _drenar(app)
    assert errores == ["el Broker no responde"]
    assert app.winfo_exists()
    # Y los botones vuelven a un estado usable en vez de quedarse bloqueados.
    assert str(app.boton_lanzar["state"]) == "normal"
    assert str(app.boton_cancelar["state"]) == "disabled"


def test_cancelar_marca_el_evento_que_consulta_el_motor(ventana) -> None:  # type: ignore[no-untyped-def]
    app, _, _ = ventana
    assert not app.cancelar.is_set()
    app.cancelar_tirada()
    assert app.cancelar.is_set()
    assert "Cancelación pedida" in app.registro.get("1.0", "end")


def test_lanzar_sin_modelos_avisa_en_vez_de_reventar(ventana, dialogos) -> None:  # type: ignore[no-untyped-def]
    app, _, _ = ventana
    app.lanzar_tirada()
    assert dialogos["aviso"] and "al menos un modelo" in dialogos["aviso"][0]


def _drenar(app) -> None:  # type: ignore[no-untyped-def]
    """Ejecuta el trabajo pendiente y vacía la cola, como haría el bucle de Tk."""
    if app.hilo is not None:
        app.hilo.join(timeout=30)
    try:
        while True:
            app._atender(app.cola.get_nowait())
    except queue.Empty:
        pass


def test_ningun_hilo_de_trabajo_toca_un_widget(ventana, dialogos, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """La regla estructural de toda interfaz Tk, aquí como prueba.

    Leer un `StringVar` desde un hilo de trabajo lanza «main thread is not in
    main loop» o, peor, cuelga la ventana sin decir nada. Este guardián revienta
    en el acto si alguna operación en segundo plano vuelve a mirar un widget en
    vez de recibir los datos ya capturados.
    """
    import threading

    original = tk.StringVar.get

    def guardia(self):  # type: ignore[no-untyped-def]
        if threading.current_thread() is not threading.main_thread():
            raise AssertionError(
                "un hilo de trabajo ha leído un widget de Tk: captura el valor en el "
                "hilo principal y pásalo como dato al hilo"
            )
        return original(self)

    monkeypatch.setattr(tk.StringVar, "get", guardia)

    app, _, _ = ventana
    broker = FakeBroker()
    broker.add_model(MODELO.key)
    broker.respond_fixed(MODELO.key, '{"a": 1}')
    app._cliente = lambda base_url="", token="": broker.client()  # type: ignore[assignment]

    app.refrescar_catalogo()
    _drenar(app)
    app.comparar()
    _drenar(app)

    assert dialogos["error"] == [], f"el hilo falló: {dialogos['error']}"
    assert app.ultimo_informe is not None


def test_la_tirada_completa_desde_la_ventana(ventana, dialogos) -> None:  # type: ignore[no-untyped-def]
    """Camino real: catálogo, elegir modelo, lanzar y ver el progreso."""
    app, ruta_suite, _ = ventana
    broker = FakeBroker()
    broker.add_model(MODELO.key)
    broker.respond_fixed(MODELO.key, '{"a": 1}')
    app._cliente = lambda base_url="", token="": broker.client()  # type: ignore[assignment]

    app.refrescar_catalogo()
    _drenar(app)
    assert app.lista_modelos.size() == 1

    app.lista_modelos.selection_set(0)
    app._al_seleccionar_modelos()  # `selection_set` no dispara el evento
    app.var_suite.set(str(ruta_suite))
    app.var_repeticiones.set("5")
    tiradas_antes = len(app.tabla_tiradas.get_children())

    app.lanzar_tirada()
    _drenar(app)

    assert dialogos["error"] == [], f"la tirada falló: {dialogos['error']}"
    assert len(app.tabla_tiradas.get_children()) == tiradas_antes + 1
    registro = app.registro.get("1.0", "end")
    assert "Terminada:" in registro
    # Contra un Broker al día no falta ninguna garantía, y la ventana lo dice
    # sin pasarse de la raya.
    assert "Garantías NO disponibles" not in registro
    assert "Semilla fijada: sí" in registro
    assert "Reproducibilidad exacta: NO demostrada" in registro
    # Y los botones vuelven a su sitio.
    assert str(app.boton_lanzar["state"]) == "normal"
    assert str(app.boton_cancelar["state"]) == "disabled"


# --------------------------------------------------------------------------
# El arranque real
# --------------------------------------------------------------------------


def _cobertura_activa() -> bool:
    """Si esta bateria se esta ejecutando bajo `coverage`."""
    try:
        import coverage
    except ImportError:  # pragma: no cover - sin cobertura instalada
        return False
    return coverage.Coverage.current() is not None


def arrancar_main(tmp_path: Path, guion: str) -> subprocess.CompletedProcess[str]:
    """Ejecuta `main()` en un proceso nuevo, con un solo intérprete de Tcl.

    Estas dos pruebas vivían en el proceso de pytest y fallaban una de cada dos
    veces con «couldn't read file init.tcl». El motivo no era `main()`: era que
    ya había una raíz de sesión viva y `main()` crea la suya, así que el proceso
    acababa con dos intérpretes de Tcl. Tcl no lo lleva bien, y ese escenario no
    existe en producción: `arrancar_model_drift.bat` lanza un proceso limpio con
    exactamente una raíz.

    Probarlo en un subproceso quita el falso rojo **y** es más fiel a lo que se
    ejecuta de verdad. Lo que se afirma sigue siendo lo mismo.
    """
    fichero = tmp_path / "arranque.py"
    fichero.write_text(guion, encoding="utf-8")
    entorno = os.environ.copy()
    # El hijo arranca desde `tmp_path`, asi que no hereda la raiz del proyecto en
    # `sys.path` como si hace pytest. Sin esto la prueba solo pasa cuando
    # `model_drift` esta instalado en el interprete, y falla con un
    # `ModuleNotFoundError` que no dice nada del codigo que se esta probando.
    raiz = str(Path(__file__).resolve().parents[1])
    anterior = entorno.get("PYTHONPATH")
    entorno["PYTHONPATH"] = raiz if not anterior else raiz + os.pathsep + anterior
    # Si la bateria corre con cobertura, el hijo tambien se mide: `coverage`
    # arranca solo cuando ve esta variable, y sin ella `main()` figuraria como
    # no cubierto por el simple hecho de ejecutarse en otro proceso.
    if _cobertura_activa():
        entorno["COVERAGE_PROCESS_START"] = str(
            Path(__file__).resolve().parents[1] / "pyproject.toml"
        )
    return subprocess.run(
        [sys.executable, str(fichero), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=entorno,
    )


def test_main_crea_una_sola_raiz_y_la_esconde(entorno, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """El camino que recorre `arrancar_model_drift.bat`.

    Se prueba porque ya falló dos veces: primero abriendo una segunda ventana
    vacía llamada «tk» (por dejar que Toplevel fabricase su raíz por omisión) y
    después con un NameError que ni ruff ni mypy vieron, porque nada ejecutaba
    esta función.
    """
    resultado = arrancar_main(
        tmp_path,
        """
import sys
import tkinter as tk

from model_drift import gui
from model_drift.config import AppConfig

raiz = sys.argv[1]
creadas = []


class VentanaFalsa(tk.Toplevel):
    def __init__(self, configuracion, master=None):
        super().__init__(master)
        creadas.append(self)
        # Cerrarse enseguida para que `wait_window` devuelva el control.
        self.after(50, self.destroy)


gui.ModelDriftGUI = VentanaFalsa
codigo = gui.main(AppConfig(database=raiz + "/md.sqlite", reports_dir=raiz, suites_dir=raiz))
# Las raíces que quedan vivas: `main()` tiene que haber destruido la suya.
raices = 1 if tk._default_root is not None else 0
print("RESULTADO", codigo, len(creadas), raices)
""",
    )

    assert resultado.returncode == 0, resultado.stdout + resultado.stderr
    assert "RESULTADO 0 1 0" in resultado.stdout, resultado.stdout + resultado.stderr


def test_main_avisa_por_ventana_si_no_puede_arrancar(entorno, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Sin consola, un traceback perdido es un fallo mudo: hay que enseñarlo."""
    resultado = arrancar_main(
        tmp_path,
        """
import sys

from model_drift import gui
from model_drift.config import AppConfig

raiz = sys.argv[1]
avisos = []
gui.messagebox.showerror = lambda titulo, texto, **_: avisos.append(texto)


def explota(configuracion, master=None):
    raise RuntimeError("la base de datos esta corrupta")


gui.ModelDriftGUI = explota
codigo = gui.main(AppConfig(database=raiz + "/md.sqlite", reports_dir=raiz, suites_dir=raiz))
print("RESULTADO", codigo, len(avisos), "corrupta" in (avisos[0] if avisos else ""))
""",
    )

    assert resultado.returncode == 0, resultado.stdout + resultado.stderr
    assert "RESULTADO 1 1 True" in resultado.stdout, resultado.stdout + resultado.stderr


# --------------------------------------------------------------------------
# Filtros del catálogo
# --------------------------------------------------------------------------


def _catalogo_variado(app) -> None:  # type: ignore[no-untyped-def]
    """Un catálogo con los tres despliegues reales del Broker."""
    from model_drift.broker_contracts import ModelCatalogEntry

    def entrada(provider: str, deployment: str, model: str, dispatchable: bool = True):  # type: ignore[no-untyped-def]
        return ModelCatalogEntry(
            provider=provider, deployment=deployment, model=model, dispatchable=dispatchable
        )

    app._pintar_catalogo(
        [
            entrada("ollama", "local", "qwen3:8b"),
            entrada("ollama", "local", "llama4:70b"),
            entrada("ollama", "cloud", "gemma4:31b-cloud"),
            entrada("nvidia", "api", "nemotron-4"),
            entrada("lmstudio", "local", "olmo-3-32b", dispatchable=False),
        ]
    )


def test_el_filtro_por_proveedor(ventana) -> None:  # type: ignore[no-untyped-def]
    app, _, _ = ventana
    _catalogo_variado(app)
    assert len(app._modelos_filtrados()) == 4, "por defecto se ocultan los no despachables"

    app.filtro_proveedor.set("ollama")
    app._aplicar_filtros()
    assert {item.provider for item in app._modelos_filtrados()} == {"ollama"}
    assert len(app.tabla_modelos.get_children()) == 3
    assert app.lista_modelos.size() == 3


def test_el_filtro_por_despliegue_agrupa_cloud_y_api(ventana) -> None:  # type: ignore[no-untyped-def]
    """El Broker tiene tres despliegues, no dos: local, cloud y api."""
    from model_drift.gui import REMOTOS

    app, _, _ = ventana
    _catalogo_variado(app)

    app.filtro_despliegue.set("local")
    app._aplicar_filtros()
    assert {item.deployment for item in app._modelos_filtrados()} == {"local"}

    app.filtro_despliegue.set(REMOTOS)
    app._aplicar_filtros()
    assert {item.deployment for item in app._modelos_filtrados()} == {"cloud", "api"}

    app.filtro_despliegue.set("api")
    app._aplicar_filtros()
    assert [item.key for item in app._modelos_filtrados()] == ["nvidia/api/nemotron-4"]


def test_el_filtro_de_texto_y_el_de_despachables(ventana) -> None:  # type: ignore[no-untyped-def]
    app, _, _ = ventana
    _catalogo_variado(app)

    app.filtro_texto.set("qwen")
    app._aplicar_filtros()
    assert [item.model for item in app._modelos_filtrados()] == ["qwen3:8b"]

    app.filtro_texto.set("")
    app.solo_despachables.set(False)
    app._aplicar_filtros()
    assert len(app._modelos_filtrados()) == 5, "sin el filtro aparece también el no despachable"


def test_las_opciones_del_filtro_salen_del_catalogo(ventana) -> None:  # type: ignore[no-untyped-def]
    """Si mañana aparece un proveedor nuevo, el filtro lo ofrece solo."""
    app, _, _ = ventana
    _catalogo_variado(app)
    proveedores = list(app._combos_proveedor[0]["values"])
    despliegues = list(app._combos_despliegue[0]["values"])
    assert proveedores == ["todos", "lmstudio", "nvidia", "ollama"]
    assert despliegues[:2] == ["todos", "remotos (cloud + api)"]
    assert set(despliegues[2:]) == {"api", "cloud", "local"}


def test_filtrar_despues_de_elegir_no_pierde_la_eleccion(ventana) -> None:  # type: ignore[no-untyped-def]
    """El fallo silencioso que este diseño evita.

    Si los modelos se leyeran de la selección visible del Listbox, acotar el
    filtro después de elegir lanzaría la tirada con menos modelos de los que
    se pidieron, sin decir nada.
    """
    app, _, _ = ventana
    _catalogo_variado(app)

    app.lista_modelos.selection_set(0, 1)
    app._al_seleccionar_modelos()
    elegidos_antes = set(app.seleccionados)
    assert len(elegidos_antes) == 2

    app.filtro_proveedor.set("nvidia")
    app._aplicar_filtros()
    assert app.lista_modelos.size() == 1, "el filtro esconde los elegidos"
    assert app.seleccionados == elegidos_antes, "pero la elección se conserva"
    assert {ref.key for ref in app._modelos_elegidos()} == elegidos_antes
    # Y la ventana lo dice, en vez de dejar creer que se ha perdido.
    assert "fuera del filtro actual" in app.var_seleccion.get()


def test_un_catalogo_nuevo_descarta_elecciones_que_ya_no_existen(ventana) -> None:  # type: ignore[no-untyped-def]
    from model_drift.broker_contracts import ModelCatalogEntry

    app, _, _ = ventana
    _catalogo_variado(app)
    app.lista_modelos.selection_set(0, 1)
    app._al_seleccionar_modelos()
    assert app.seleccionados

    app._pintar_catalogo(
        [ModelCatalogEntry(provider="ollama", deployment="local", model="modelo-nuevo", dispatchable=True)]
    )
    assert app.seleccionados == set()


# --------------------------------------------------------------------------
# Auditoría humana del 15-oct-2026 (INC-02…INC-12)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("texto", ["-1", "0", "abc"])
def test_inc03_inc08_repeticiones_invalidas_no_lanzan_nada(ventana, dialogos, texto) -> None:  # type: ignore[no-untyped-def]
    app, ruta_suite, ids = ventana
    app.seleccionados = {MODELO.key}
    app.var_suite.set(str(ruta_suite))
    app.var_repeticiones.set(texto)
    app.lanzar_tirada()
    assert dialogos["error"] and "Las repeticiones por caso deben ser un número entero" in dialogos["error"][-1]
    assert "invalid literal" not in dialogos["error"][-1]
    assert dialogos["pregunta"] == [], "ni siquiera se ofrece «lanzar de todas formas»"
    assert app.hilo is None
    assert set(app.tabla_tiradas.get_children()) == set(ids)


def test_inc08_presupuesto_invalido_se_explica_y_admite_coma(ventana, dialogos) -> None:  # type: ignore[no-untyped-def]
    app, _, _ = ventana
    app.var_repeticiones.set("5")
    app.var_presupuesto.set("abc")
    assert app._leer_parametros() is None
    assert "El presupuesto debe ser una cantidad en USD" in dialogos["error"][-1]
    app.var_presupuesto.set("0,5")
    assert app._leer_parametros() == (5, 0.5)
    app.var_presupuesto.set("")
    assert app._leer_parametros() == (5, None)


def test_inc02_token_rechazado_se_ve_en_la_ventana(ventana, dialogos) -> None:  # type: ignore[no-untyped-def]
    app, ruta_suite, _ = ventana
    broker = FakeBroker(reject_admin_token=True)
    broker.add_model(MODELO.key)
    app._cliente = lambda base_url="", token="": broker.client()  # type: ignore[assignment]

    app.refrescar_catalogo()  # el catálogo no pide token: sigue funcionando
    _drenar(app)
    app.lista_modelos.selection_set(0)
    app._al_seleccionar_modelos()
    app.var_suite.set(str(ruta_suite))
    app.var_repeticiones.set("5")
    app.lanzar_tirada()
    _drenar(app)

    registro = app.registro.get("1.0", "end")
    assert "FALLIDA:" in registro and "ADMIN_AUTH_REQUIRED" in registro
    assert "Determinismo observado: sí" not in registro
    assert dialogos["aviso"] and "token" in dialogos["aviso"][-1]
    assert app.progreso["value"] == 0, "INC-10: nada de barra llena y verde"

    run_id = app.tabla_tiradas.get_children()[0]
    assert app.tabla_tiradas.item(run_id, "values")[5] == "failed"
    assert all(run != run_id for run, _ in app._opciones_comparar.values()), "no se ofrece para comparar"
    app.tabla_tiradas.selection_set(run_id)
    app._mostrar_detalle_tirada()
    detalle = app.detalle_tirada.get("1.0", "end")
    assert "NINGUNA invocación produjo salida" in detalle
    assert "1 × AUTH: ADMIN_AUTH_REQUIRED" in detalle
    assert "renueva el token" in detalle, "la nota del manifiesto también sale"


def _tirada_en(config: AppConfig, ruta_suite: Path, modelos: list[ModelReference], **run: object) -> str:
    broker = FakeBroker()
    for modelo in modelos:
        broker.add_model(modelo.key)
        broker.respond_fixed(modelo.key, '{"a": 1}')
    with Database(config.database) as db, broker.client() as cliente:
        runner = Runner(cliente, load_suite(ruta_suite), db=db, config=RunConfig(repetitions=5))
        return runner.run(modelos, **run).run_id  # type: ignore[arg-type]


def test_inc04_inc06_los_combos_ofrecen_cada_modelo_y_nunca_tiradas_vacias(ventana, dialogos) -> None:  # type: ignore[no-untyped-def]
    app, ruta_suite, ids = ventana
    otro = ModelReference(provider="ollama", deployment="local", model="qwen3:8b")
    vacia = _tirada_en(app.config_app, ruta_suite, [MODELO], should_cancel=lambda: True)
    multi = _tirada_en(app.config_app, ruta_suite, [MODELO, otro])
    app.refrescar_tiradas()

    opciones = list(app._opciones_comparar.values())
    assert vacia in app.tabla_tiradas.get_children(), "sigue en el histórico…"
    assert all(run != vacia for run, _ in opciones), "…pero no se puede comparar"
    assert (multi, MODELO.key) in opciones and (multi, otro.key) in opciones
    assert len(opciones) == len(set(opciones)), "OB-02: sin entradas repetidas"

    app.var_candidata.set(next(e for e, par in app._opciones_comparar.items() if par == (multi, otro.key)))
    app.var_base.set(next(e for e, par in app._opciones_comparar.items() if par == (ids[0], MODELO.key)))
    app.comparar()
    _drenar(app)
    assert dialogos["error"] == [], dialogos["error"]
    assert f"Candidata: {multi} · {otro.key}" in app.resumen.get("1.0", "end")
    assert app.ruta_informe is not None and "qwen3_8b" in app.ruta_informe.name


def test_inc04_inc09_un_error_sin_texto_se_explica_y_libera_la_barra(ventana, dialogos) -> None:  # type: ignore[no-untyped-def]
    app, _, _ = ventana
    app.estado.set("Comparando…")
    app._en_hilo(lambda: next(iter([])))
    _drenar(app)
    assert dialogos["error"] == ["StopIteration()"], "antes salía un diálogo en blanco"
    assert app.estado.get().startswith("La última operación falló")


def test_inc04_al_comparar_se_borra_el_resultado_anterior(ventana) -> None:  # type: ignore[no-untyped-def]
    app, _, _ = ventana
    app.comparar()
    _drenar(app)
    assert app.tabla_casos.get_children()
    app._en_hilo = lambda trabajo: None  # type: ignore[method-assign]  # la nueva no llega a terminar
    app.comparar()
    assert not app.tabla_casos.get_children() and app.var_veredicto.get() == ""
    assert str(app.boton_informe["state"]) == "disabled"


def test_inc09_con_trabajo_en_curso_no_se_toca_nada(ventana, dialogos) -> None:  # type: ignore[no-untyped-def]
    import threading

    app, _, _ = ventana
    soltar = threading.Event()
    app.hilo = threading.Thread(target=soltar.wait, daemon=True)
    app.hilo.start()
    try:
        app.cancelar.set()
        antes = app.estado.get()
        app.refrescar_catalogo()
        app.probar_system1()
        app.comparar()
        assert app.estado.get() == antes, "no puede quedar escrito «Consultando…»"
        assert app.cancelar.is_set(), "ni limpiarse la cancelación de la tirada en curso"
        assert dialogos["info"] == ["Ya hay una operación en curso."] * 3
    finally:
        soltar.set()
        app.hilo.join()


def test_inc05_los_dialogos_son_hijos_de_la_ventana(ventana, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Con la raíz oculta de dueña, la ventana seguía operable y se apilaban."""
    app, _, _ = ventana
    padres: list[object] = []
    monkeypatch.setattr("model_drift.gui.messagebox.showwarning", lambda t, m, **kw: padres.append(kw.get("parent")))
    app.lanzar_tirada()  # sin modelos
    app.var_base.set("")
    app.comparar()  # sin tiradas elegidas
    assert padres == [app, app]


def test_inc11_la_tabla_de_modelos_elige_de_verdad(ventana) -> None:  # type: ignore[no-untyped-def]
    app, _, _ = ventana
    _catalogo_variado(app)
    app.tabla_modelos.selection_set(["ollama/local/qwen3:8b"])
    app._al_seleccionar_en_tabla()  # sin bucle de Tk el evento no llega solo
    assert app.seleccionados == {"ollama/local/qwen3:8b"}
    assert [app.lista_modelos.get(i) for i in app.lista_modelos.curselection()] == ["ollama/local/qwen3:8b"]

    app.lista_modelos.selection_set(0)
    app._al_seleccionar_modelos()
    assert set(app.tabla_modelos.selection()) == app.seleccionados
    assert len(app.seleccionados) == 2


def test_inc12_el_avance_de_system1_sale_en_su_pestana(ventana) -> None:  # type: ignore[no-untyped-def]
    from model_drift.gui import Mensaje

    app, _, _ = ventana
    app.cola.put(Mensaje("log_cal", "Caso 2/4: maestro respondió"))
    _drenar(app)
    assert "Caso 2/4" in app.cal_summary.get("1.0", "end")
    assert "Caso 2/4" in app.estado.get()


def test_inc07_la_fecha_de_las_tiradas_es_local(ventana) -> None:  # type: ignore[no-untyped-def]
    from model_drift.cli import fecha_local

    app, _, ids = ventana
    with Database(app.config_app.database) as db:
        fila = db.query_one("SELECT created_at FROM run WHERE run_id = ?", (ids[0],))
    assert fila is not None
    assert app.tabla_tiradas.item(ids[0], "values")[1] == fecha_local(fila["created_at"])
