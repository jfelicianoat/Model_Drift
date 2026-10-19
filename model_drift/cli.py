"""Línea de comandos de Model_Drift.

    model-drift modelos                        lista el catálogo del Broker
    model-drift modelos --proveedor ollama --despliegue local
    model-drift ejecutar <suite> --modelo M    lanza una tirada
    model-drift comparar <base> <candidata>    compara dos tiradas
    model-drift tiradas                        lista las tiradas guardadas
    model-drift potencia --casos 50            declara qué se puede detectar

Dos comportamientos deliberados de la CLI:

- `ejecutar` **se niega a lanzar** si el Broker no permite excluir el tráfico
  del aprendizaje y nadie lo ha confirmado. Medir un modelo con prompts duros
  le baja el score al router adaptativo del Broker, y eso es un efecto sobre un
  sistema en uso: exige una decisión humana, no un valor por defecto.
- El código de salida refleja el veredicto (`blocked` -> 3), para que una
  comparación pueda cortar una entrega desde un script de CI. **No es 2**: ese
  lo usa argparse para los errores de uso, y un guion de CI no debe confundir
  «la política ha bloqueado la entrega» con «te has equivocado al escribir el
  comando».
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from model_drift.broker_client import BrokerClient
from model_drift.broker_contracts import ModelCatalogEntry
from model_drift.calibration.models import CalibrationPlan
from model_drift.calibration.runner import export_distilled, revalidate_changed, run_calibration, save_artifacts
from model_drift.cases import load_suite
from model_drift.comparison import RunResults, compare_runs
from model_drift.comparison.service import compare_configured
from model_drift.config import AppConfig
from model_drift.db import Database
from model_drift.domain import CaseOutcome, RunVerdict
from model_drift.external_treatments import load_external_treatments
from model_drift.manifest import RunManifest
from model_drift.metamorphic import load_relations
from model_drift.policy import RegressionPolicy
from model_drift.report import render_calibration_summary, render_text_summary, write_comparison
from model_drift.runner import RunConfig, Runner
from model_drift.statistics import minimum_detectable_effect, power_curve

EXIT_OK = 0
EXIT_ERROR = 1
#: 2 queda reservado a argparse (error de uso), así que el veredicto bloqueante
#: usa 3. Confundir ambos en CI sería tratar un typo como una regresión.
EXIT_BLOCKED = 3


def _client(config: AppConfig) -> BrokerClient:
    return BrokerClient(config.broker_config())


def cmd_modelos(args: argparse.Namespace, config: AppConfig) -> int:
    with _client(config) as cliente:
        catalogo = cliente.catalog()
    if not catalogo:
        print("El Broker no devolvió ningún modelo.")
        return EXIT_ERROR

    entradas = _filtrar(catalogo, args)
    if not entradas:
        print(f"Ningún modelo pasa el filtro (el catálogo tiene {len(catalogo)}).")
        return EXIT_OK
    ancho = max(len(item.key) for item in entradas)
    print(f"{'MODELO'.ljust(ancho)}  DESPACHABLE  CONTEXTO  TOOLS  VISION")
    for entrada in sorted(entradas, key=lambda item: item.key):
        # Se pregunta por `tools` y `vision`, que vienen en la lista plana del
        # catálogo. La matriz completa (`function_calling`, `json_mode`…) es una
        # petición POR MODELO: con 162 modelos serían 162 peticiones para
        # pintar dos columnas, y saldrían todas como "?".
        print(
            f"{entrada.key.ljust(ancho)}  "
            f"{'sí' if entrada.dispatchable else 'no':^11}  "
            f"{str(entrada.context_window or '—'):^8}  "
            f"{_tri(entrada.supports('tools')):^5}  {_tri(entrada.supports('vision')):^6}"
        )
    ocultos = len(catalogo) - len(entradas)
    print(f"\n{len(entradas)} de {len(catalogo)} modelos" + (f" · {ocultos} ocultos por el filtro" if ocultos else ""))
    print("'?' = el Broker no lo ha comprobado. Un caso que lo exija quedará como no aplicable.")
    print("La matriz completa de capacidades se consulta por modelo al lanzar la tirada.")
    return EXIT_OK


def _filtrar(catalogo: list[ModelCatalogEntry], args: argparse.Namespace) -> list[ModelCatalogEntry]:
    """Mismos filtros que la ventana, para que las dos vistas coincidan."""
    proveedor = (getattr(args, "proveedor", None) or "").strip()
    despliegue = (getattr(args, "despliegue", None) or "").strip()
    texto = (getattr(args, "buscar", None) or "").strip().casefold()
    solo_despachables = bool(getattr(args, "solo_despachables", False))

    def pasa(entrada: ModelCatalogEntry) -> bool:
        if proveedor and entrada.provider != proveedor:
            return False
        if despliegue == "remoto":
            # `cloud` y `api` salen los dos de la máquina: para la frontera de
            # datos son lo mismo. El Broker distingue tres despliegues, no dos.
            if entrada.deployment == "local":
                return False
        elif despliegue and entrada.deployment != despliegue:
            return False
        if solo_despachables and not entrada.dispatchable:
            return False
        if texto and texto not in entrada.key.casefold():
            return False
        return True

    return [item for item in catalogo if pasa(item)]


def _tri(value: bool | None) -> str:
    return "sí" if value is True else "no" if value is False else "?"


def cmd_ejecutar(args: argparse.Namespace, config: AppConfig) -> int:
    suite = load_suite(args.suite)
    relaciones = load_relations(args.relaciones) if args.relaciones else {}
    politica = RegressionPolicy.from_file(args.politica) if args.politica else RegressionPolicy()

    with _client(config) as cliente:
        modelos = [config.model_reference(raw) for raw in args.modelo]

        # Se pregunta al Broker en vez de suponerlo: el aviso solo tiene
        # sentido si de verdad va a contaminar.
        cliente.probe_contract()
        if not cliente.config.supports_learning_exclusion and not config.accept_learning_contamination:
            print(
                "Esta tirada alterará las estadísticas del router adaptativo del Broker.\n"
                "El Broker no admite todavía marcar tráfico como excluido del aprendizaje\n"
                "(Cambio 4 del contrato de evaluación), así que las invocaciones fallidas de\n"
                "esta suite bajarán el score de fiabilidad de los modelos medidos y pueden\n"
                "acercarlos a la cuarentena.\n\n"
                "Para lanzarla igualmente: pon accept_learning_contamination: true en la\n"
                "configuración, o usa --acepto-contaminar.",
                file=sys.stderr,
            )
            return EXIT_ERROR

        runner = Runner(
            cliente,
            suite,
            db=Database(config.database),
            config=RunConfig(
                repetitions=args.repeticiones if args.repeticiones is not None else config.repetitions,
                temperature=config.temperature,
                seed=config.seed,
                budget_usd=args.presupuesto if args.presupuesto is not None else config.budget_usd,
                label=args.etiqueta or "",
                allow_code_execution=config.allow_code_execution,
                timeout_seconds=config.timeout_seconds,
            ),
            policy=politica,
            relations=relaciones,
            core=config.core_client(),
        )
        salida = runner.run(modelos)

    print(f"Tirada {salida.run_id}")
    print(f"  suite        {suite.id} v{suite.version} ({suite.short_hash()})")
    print(f"  modelos      {', '.join(modelo.key for modelo in modelos)}")
    print(f"  invocaciones {salida.invocations}")
    print(f"  coste        {salida.cost_usd:.4f} USD")
    print(f"  estado       {salida.status}")
    resumen_fallos = salida.failure_summary()
    if resumen_fallos:
        print(f"  FALLOS       {resumen_fallos}")
    if salida.status != "failed":
        # Sobre una tirada sin ninguna salida, «determinismo observado» no
        # describe al modelo sino la ausencia de respuestas.
        for linea in salida.manifest.reproducibility.declare():
            print(f"  {linea}")
    if salida.manifest.missing_guarantees:
        print("\n  Garantías NO disponibles en esta tirada:")
        for garantia in salida.manifest.missing_guarantees:
            print(f"   - {garantia}")
    for aviso in salida.core_notices:
        print(f"  AVISO        {aviso}")
    if salida.aborted_reason:
        print(f"\n  INTERRUMPIDA: {salida.aborted_reason}")
    if salida.status == "failed":
        print("\n  NINGUNA invocación produjo salida: esta tirada no midió el modelo.", file=sys.stderr)
    return EXIT_ERROR if salida.aborted_reason or salida.status == "failed" else EXIT_OK


def _load_run(db: Database, run_id: str) -> tuple[RunManifest, dict[str, RunResults]]:
    """Reconstruye una tirada guardada desde la base de datos."""
    fila = db.query_one("SELECT * FROM run WHERE run_id = ?", (run_id,))
    if fila is None:
        raise SystemExit(f"no existe la tirada {run_id!r}")
    if fila["status"] == "failed":
        raise ValueError(f"la tirada {run_id} falló entera: ninguna invocación produjo salida, no midió el modelo")
    manifiesto = RunManifest.model_validate_json(fila["manifest_json"])

    resultados: dict[str, RunResults] = {}
    for caso in db.query("SELECT * FROM case_result WHERE run_id = ?", (run_id,)):
        clave = caso["model_key"]
        if clave not in resultados:
            resultados[clave] = RunResults(run_id=run_id, manifest=manifiesto, model_key=clave)
        import json as _json

        resultados[clave].outcomes[(caso["case_id"], caso["metric"])] = CaseOutcome(
            case_id=caso["case_id"],
            model_key=clave,
            metric=caso["metric"],
            case_score=caso["case_score"],
            case_stability=caso["case_stability"],
            repetitions=caso["repetitions"],
            failed_repetitions=caso["failed_repetitions"],
            is_unstable=bool(caso["is_unstable"]),
            is_not_applicable=bool(caso["is_not_applicable"]),
            not_applicable_reason=caso["not_applicable_reason"],
            values=_json.loads(caso["values_json"] or "[]"),
        )

    # Una salida representativa por caso, para la taxonomía del cambio.
    for invocacion in db.query(
        "SELECT case_id, model_key, output_text FROM invocation "
        "WHERE run_id = ? AND ok = 1 ORDER BY repetition",
        (run_id,),
    ):
        clave = invocacion["model_key"]
        if clave in resultados:
            resultados[clave].sample_output.setdefault(invocacion["case_id"], invocacion["output_text"])
    return manifiesto, resultados


def _elegir_modelo(run_id: str, resultados: dict[str, RunResults], clave: str | None) -> RunResults:
    """Los resultados de UN modelo de la tirada. Nunca se elige uno en silencio."""
    if not resultados:
        raise ValueError(f"la tirada {run_id} no tiene resultados comparables: no llegó a medir ningún caso")
    if clave is None:
        if len(resultados) > 1:
            raise ValueError(
                f"la tirada {run_id} midió varios modelos ({', '.join(sorted(resultados))}): "
                "indica cuál comparar (--modelo-base / --modelo-candidato)"
            )
        clave = next(iter(resultados))
    if clave not in resultados:
        raise ValueError(f"la tirada {run_id} no tiene resultados de {clave} (tiene: {', '.join(sorted(resultados))})")
    return resultados[clave]


def fecha_local(iso: str) -> str:
    """`created_at` se guarda en UTC; a la persona se le enseña en su hora."""
    try:
        return datetime.fromisoformat(str(iso)).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return str(iso)[:19].replace("T", " ")


def cmd_comparar(args: argparse.Namespace, config: AppConfig) -> int:
    politica = RegressionPolicy.from_file(args.politica) if args.politica else RegressionPolicy()
    suite = load_suite(args.suite) if args.suite else None

    with Database(config.database) as db:
        _, base = _load_run(db, args.base)
        _, candidata = _load_run(db, args.candidata)

    resultados_base = _elegir_modelo(args.base, base, args.modelo_base)
    resultados_cand = _elegir_modelo(args.candidata, candidata, args.modelo_candidato)
    clave_base, clave_cand = resultados_base.model_key, resultados_cand.model_key
    informe = compare_configured(
        resultados_base, resultados_cand, config=config, suite=suite, policy=politica,
        client_factory=lambda: _client(config),
    )

    print(render_text_summary(informe))

    if args.informe:
        destino = write_comparison(
            informe,
            args.informe,
            titulo=args.titulo or "Informe de deriva",
            baseline_label=f"{args.base} · {clave_base}",
            candidate_label=f"{args.candidata} · {clave_cand}",
            manifest=resultados_cand.manifest,
        )
        print(f"\nInforme HTML: {destino}")

    if informe.verdict and informe.verdict.verdict is RunVerdict.blocked:
        return EXIT_BLOCKED
    return EXIT_OK


def cmd_evaluar_tratamientos(args: argparse.Namespace, config: AppConfig) -> int:
    """Compara dos artefactos externos sin leer bases de datos ajenas."""
    if Path(args.informe).exists():
        raise ValueError("refusing to replace an existing external-treatment report")
    _plan, base, candidata = load_external_treatments(args.plan)
    politica = RegressionPolicy.from_file(args.politica) if args.politica else RegressionPolicy()
    informe = compare_runs(base, candidata, policy=politica)

    print(render_text_summary(informe))
    destino = write_comparison(
        informe,
        args.informe,
        titulo=args.titulo or "Tratamientos externos · R3 frente a R4",
        baseline_label=f"{base.run_id} · R3",
        candidate_label=f"{candidata.run_id} · R4",
        manifest=candidata.manifest,
    )
    print(f"\nInforme HTML: {destino}")
    if informe.verdict and informe.verdict.verdict is RunVerdict.blocked:
        return EXIT_BLOCKED
    return EXIT_OK


def cmd_probar_system1(args: argparse.Namespace, config: AppConfig) -> int:
    plan = CalibrationPlan.load(args.plan)
    if args.maestro:
        plan.config.teacher_task = config.model_reference(args.maestro)
        plan.config.teacher_judge = None
    with _client(config) as client, Database(config.database) as db:
        report = run_calibration(client, plan, db, progress=lambda message: print(message))
        if args.exportar_referencias:
            export_distilled(db, report, args.exportar_referencias)
        artifact = Path(args.artefacto) if args.artefacto else Path(config.reports_dir) / f"{report.run_id}.json"
        html = save_artifacts(db, report, artifact, args.informe or artifact.with_suffix(".html"))
    print(render_calibration_summary(report))
    print(f"Artefacto: {artifact}\nInforme: {html}")
    return EXIT_ERROR if any(reason.startswith("Tirada fallida") for reason in report.reasons) else EXIT_OK


def cmd_revalidar_system1(args: argparse.Namespace, config: AppConfig) -> int:
    with _client(config) as client, Database(config.database) as db:
        client.catalog(refresh=True)
        reports = revalidate_changed(client, db, config.reports_dir, progress=print)
    if not reports:
        print("No se detectaron cambios de huella en los planes guardados.")
    for report in reports:
        print(render_calibration_summary(report))
    failed = any(any(reason.startswith("Tirada fallida") for reason in report.reasons) for report in reports)
    return EXIT_ERROR if failed else EXIT_OK


def cmd_tiradas(args: argparse.Namespace, config: AppConfig) -> int:
    with Database(config.database) as db:
        filas = db.runs(limit=args.limite)
    if not filas:
        print("No hay tiradas guardadas.")
        return EXIT_OK
    for fila in filas:
        etiqueta = f" · {fila['label']}" if fila["label"] else ""
        print(f"{fila['run_id']}  {fecha_local(fila['created_at'])}  {fila['suite_id']}  [{fila['status']}]{etiqueta}")
    return EXIT_OK


def cmd_potencia(args: argparse.Namespace, config: AppConfig) -> int:
    """Declara qué puede y qué no puede detectar una configuración."""
    mde = minimum_detectable_effect(n_cases=args.casos, baseline_rate=args.tasa_base)
    print(f"Casos: {args.casos} · tasa base supuesta: {args.tasa_base:.0%}")
    if mde is None:
        print("Con esta configuración no se alcanza potencia 0,8 ni para una caída total.")
    else:
        print(f"Menor degradación detectable con potencia 0,8: {mde:.1%}")
    print("\nCurva de potencia:")
    for punto in power_curve(n_cases=args.casos, baseline_rate=args.tasa_base):
        print(f"  caída de {abs(punto.difference):>5.0%}  ->  potencia {punto.power:.2f}")
    print("\nUn «sin cambio observable» por debajo del efecto mínimo detectable no es")
    print("evidencia de que no haya cambio: es falta de potencia.")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model-drift",
        description="Detecta si los prompts de un producto siguen comportándose igual "
        "cuando cambia el modelo o su configuración de ejecución.",
    )
    parser.add_argument("--config", help="fichero de configuración (por defecto model_drift.yaml)")
    subparsers = parser.add_subparsers(dest="comando", required=True)

    modelos = subparsers.add_parser("modelos", help="lista el catálogo de modelos del Broker")
    modelos.add_argument("--proveedor", help="ollama, lmstudio, nvidia…")
    modelos.add_argument(
        "--despliegue",
        help="local, cloud, api, o 'remoto' para agrupar cloud y api",
    )
    modelos.add_argument("--buscar", help="subcadena del identificador del modelo")
    modelos.add_argument(
        "--solo-despachables",
        action="store_true",
        dest="solo_despachables",
        help="oculta los que el Broker no puede despachar ahora mismo",
    )

    ejecutar = subparsers.add_parser("ejecutar", help="lanza una tirada de una suite")
    ejecutar.add_argument("suite", help="fichero o directorio de la suite")
    ejecutar.add_argument("--modelo", action="append", required=True, help="proveedor/despliegue/modelo")
    ejecutar.add_argument("--repeticiones", type=int, help="repeticiones por caso")
    ejecutar.add_argument("--presupuesto", type=float, help="tope de coste en USD")
    ejecutar.add_argument("--etiqueta", help="etiqueta para identificar la tirada")
    ejecutar.add_argument("--politica", help="fichero de regression_policy")
    ejecutar.add_argument("--relaciones", help="fichero o directorio de relaciones metamórficas")
    ejecutar.add_argument(
        "--acepto-contaminar",
        action="store_true",
        dest="acepto_contaminar",
        help="acepta que la tirada altere las estadísticas del router del Broker",
    )

    comparar = subparsers.add_parser("comparar", help="compara dos tiradas")
    comparar.add_argument("base")
    comparar.add_argument("candidata")
    comparar.add_argument("--suite", help="suite, para saber qué evidencia tiene cada caso")
    comparar.add_argument("--politica")
    comparar.add_argument("--modelo-base", dest="modelo_base")
    comparar.add_argument("--modelo-candidato", dest="modelo_candidato")
    comparar.add_argument("--informe", help="ruta del informe HTML a generar")
    comparar.add_argument("--titulo")

    externos = subparsers.add_parser(
        "evaluar-tratamientos",
        help="compara artefactos R3/R4 calculados por Local AI Lab",
    )
    externos.add_argument("--plan", required=True, help="plan v2 sellado de tratamientos externos")
    externos.add_argument("--informe", required=True, help="ruta nueva del informe HTML")
    externos.add_argument("--politica", help="fichero de regression_policy")
    externos.add_argument("--titulo")

    tiradas = subparsers.add_parser("tiradas", help="lista las tiradas guardadas")
    tiradas.add_argument("--limite", type=int, default=25)

    potencia = subparsers.add_parser("potencia", help="qué degradación puede detectar una configuración")
    potencia.add_argument("--casos", type=int, required=True)
    potencia.add_argument("--tasa-base", dest="tasa_base", type=float, default=0.9)

    calibration = subparsers.add_parser(
        "probar-system1", help="prueba y calibra un System 1 contra un maestro System 2",
    )
    calibration.add_argument("--plan", required=True, help="plan JSON con casos, producción, alumno y maestro")
    calibration.add_argument("--maestro", help="referencia completa del maestro, para sobrescribir la del plan")
    calibration.add_argument("--artefacto", help="artefacto JSON versionado")
    calibration.add_argument("--informe", help="informe HTML")
    calibration.add_argument(
        "--exportar-referencias", help="exportar casos, etiquetas y procedencia para reutilización",
    )
    subparsers.add_parser("revalidar-system1", help="repite automáticamente los planes cuyas huellas hayan cambiado")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = AppConfig.load(args.config)
    if getattr(args, "acepto_contaminar", False):
        config.accept_learning_contamination = True

    comandos = {
        "modelos": cmd_modelos,
        "ejecutar": cmd_ejecutar,
        "comparar": cmd_comparar,
        "evaluar-tratamientos": cmd_evaluar_tratamientos,
        "tiradas": cmd_tiradas,
        "potencia": cmd_potencia,
        "probar-system1": cmd_probar_system1,
        "revalidar-system1": cmd_revalidar_system1,
    }
    try:
        return comandos[args.comando](args, config)
    except SystemExit:
        raise
    except Exception as error:  # noqa: BLE001 - la CLI no debe volcar una traza al usuario
        print(f"Error: {error}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
