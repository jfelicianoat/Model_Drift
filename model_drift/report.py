"""Informes HTML. La honestidad del sistema se ve aquí o no se ve.

Dos reglas que el generador impone por construcción, no por disciplina de quien
escribe la plantilla:

1. **La cabecera de comparabilidad va antes que cualquier conclusión.** Si los
   manifiestos difieren, lo primero que se lee es que nada de lo que sigue es
   atribuible al modelo.
2. **Nunca se escriben las palabras «mejora» ni «regresión» sobre un caso cuyo
   `change_direction` sea `sin_direccion`.** `direction_label` es la única vía
   por la que una dirección llega a la plantilla, y para ese estado devuelve
   «cambio sin dirección evaluable».

El informe también declara siempre su propia potencia y las garantías que no
tuvo. Un «sin cambios» sobre doce casos y sin control de semilla no significa
lo mismo que sobre trescientos con determinismo observado, y el lector tiene
derecho a distinguirlos de un vistazo.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from jinja2 import Environment, select_autoescape

from model_drift.comparison import ComparisonReport
from model_drift.domain import (
    ChangeDirection,
    ChangeStatus,
    ComparisonOutcome,
    PracticalSignificance,
    RunVerdict,
)
from model_drift.judge import JudgeMetrics
from model_drift.manifest import RunManifest

DIRECTION_LABELS = {
    ChangeDirection.mejora: "mejora",
    ChangeDirection.regresion: "regresión",
    ChangeDirection.cambio_neutro: "cambio neutro",
    # La frase importa: dice que HAY cambio y que NO se le puede poner signo.
    ChangeDirection.sin_direccion: "cambio sin dirección evaluable",
}

STATUS_LABELS = {
    ChangeStatus.sin_cambio_observable: "sin cambio observable",
    ChangeStatus.cambio_significativo: "cambio significativo",
    ChangeStatus.fallo_nuevo: "fallo nuevo de ejecución",
    ChangeStatus.indeterminado: "indeterminado",
}

PRACTICAL_LABELS = {
    PracticalSignificance.no_detectado: "no detectado",
    PracticalSignificance.detectado_irrelevante: "detectado pero irrelevante",
    PracticalSignificance.relevante_no_concluyente: "relevante, sin potencia para concluir",
    PracticalSignificance.detectado_relevante: "detectado y relevante",
}

VERDICT_LABELS = {
    RunVerdict.pass_: "Sin cambios relevantes",
    RunVerdict.warning: "Avisos",
    RunVerdict.blocked: "Bloqueado",
    RunVerdict.trade_off_detected: "Compensación detectada",
    RunVerdict.indeterminado: "Indeterminado",
}


def direction_label(outcome: ComparisonOutcome) -> str:
    """Única vía por la que una dirección llega al informe.

    Sin evidencia no hay palabra: `sin_direccion` nunca se traduce por algo que
    suene a juicio de calidad.
    """
    return DIRECTION_LABELS[outcome.change_direction]


_ESTILO = """<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; line-height: 1.5;
         margin: 0 auto; max-width: 1100px; padding: 2rem 1.25rem; }
  h1 { margin-bottom: .25rem; }
  .sub { color: #6b7280; margin-top: 0; }
  .aviso { border-left: 4px solid #b45309; background: #fffbeb; color: #78350f;
           padding: .9rem 1rem; border-radius: .35rem; margin: 1rem 0; }
  .grave { border-left-color: #b91c1c; background: #fef2f2; color: #7f1d1d; }
  .ok { border-left-color: #15803d; background: #f0fdf4; color: #14532d; }
  .veredicto { display: inline-block; padding: .3rem .75rem; border-radius: .3rem;
               font-weight: 600; }
  .v-pass { background: #dcfce7; color: #14532d; }
  .v-warning { background: #fef3c7; color: #78350f; }
  .v-blocked { background: #fee2e2; color: #7f1d1d; }
  .v-trade_off_detected { background: #e0e7ff; color: #312e81; }
  .v-indeterminado { background: #e5e7eb; color: #374151; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .92rem; }
  th, td { border-bottom: 1px solid #d1d5db; padding: .5rem .6rem; text-align: left;
           vertical-align: top; }
  th { background: #f3f4f6; }
  .tabla-scroll { overflow-x: auto; }
  code { background: #f3f4f6; padding: .1rem .3rem; border-radius: .2rem; }
  .etiqueta { background: #e5e7eb; border-radius: .25rem; padding: .05rem .4rem;
              font-size: .8rem; margin-right: .25rem; }
  .nota { color: #6b7280; font-size: .85rem; }
  details { margin: .5rem 0; }
  @media (prefers-color-scheme: dark) {
    body { background: #0b0f19; color: #e5e7eb; }
    th { background: #1f2937; } th, td { border-color: #374151; }
    code, .etiqueta { background: #1f2937; }
    .aviso { background: #2b2007; color: #fde68a; }
    .grave { background: #300f0f; color: #fecaca; }
    .ok { background: #052e16; color: #bbf7d0; }
  }
</style>
"""

_TEMPLATE = (
    _ESTILO
    + """
<!-- Informe generado por Model_Drift. Autocontenido, sin recursos externos. -->

<h1>{{ titulo }}</h1>
<p class="sub">{{ base }} → {{ candidata }} · generado el {{ fecha }}</p>

<!-- 1. COMPARABILIDAD. Va primero a propósito. -->
{% if not informe.comparability.comparable %}
<div class="aviso grave">
  <strong>Estas dos tiradas no son comparables.</strong><br>
  {{ informe.comparability.headline() }}<br>
  Ninguna diferencia observada puede atribuirse al modelo: pudo cambiarla el instrumento de medida.
</div>
{% elif informe.comparability.fingerprint_notes %}
<div class="aviso">
  <strong>Comparables con reservas.</strong>
  {% for nota in informe.comparability.fingerprint_notes %}{{ nota }} {% endfor %}
</div>
{% else %}
<div class="aviso ok">Mismo instrumento de evaluación en las dos tiradas.</div>
{% endif %}

<!-- 2. VEREDICTO segun la politica declarada -->
{% if informe.verdict %}
<h2>Veredicto</h2>
<p><span class="veredicto v-{{ informe.verdict.verdict.value }}">{{ etiqueta_veredicto }}</span></p>
<p>{{ informe.verdict.summary() }}</p>
{% if informe.verdict.verdict.value == 'trade_off_detected' %}
<p class="nota">La herramienta no resuelve la compensación: la decisión depende de la política
declarada y de quien la escribió.</p>
{% endif %}
{% endif %}

<!-- 3. POTENCIA y GARANTIAS -->
<h2>Qué garantías tuvo esta medición</h2>
<p>{{ informe.power_statement() }}</p>
<ul>
  {% for linea in reproducibilidad %}<li>{{ linea }}</li>{% endfor %}
</ul>
{% if garantias_ausentes %}
<div class="aviso">
  <strong>Garantías no disponibles:</strong>
  <ul>{% for garantia in garantias_ausentes %}<li>{{ garantia }}</li>{% endfor %}</ul>
</div>
{% endif %}
<p class="nota">
  Casos: {{ informe.n_cases }} · comparados: {{ informe.n_compared }} ·
  indeterminados: {{ informe.n_indeterminate }} · no aplicables: {{ informe.n_not_applicable }} ·
  inestables: {{ informe.n_unstable }}
</p>

<!-- 4. METRICAS -->
{% if informe.verdict and informe.verdict.metrics %}
<h2>Métricas</h2>
<div class="tabla-scroll">
<table>
  <tr><th>Métrica</th><th>Severidad</th><th>Base</th><th>Candidata</th><th>Efecto</th>
      <th>Relevancia</th><th>Fuera de tolerancia</th></tr>
  {% for metrica in informe.verdict.metrics %}
  <tr>
    <td><code>{{ metrica.metric }}</code></td>
    <td>{{ metrica.severity.value }}</td>
    <td>{{ '%.4g'|format(metrica.baseline) if metrica.baseline is not none else '—' }}</td>
    <td>{{ '%.4g'|format(metrica.candidate) if metrica.candidate is not none else '—' }}</td>
    <td>{{ '%+.4g'|format(metrica.effect) if metrica.effect is not none else '—' }}</td>
    <td>{{ etiquetas_practicas[metrica.practical_significance] }}</td>
    <td>{{ 'sí' if metrica.exceeds_tolerance else 'no' }}</td>
  </tr>
  {% endfor %}
</table>
</div>
{% endif %}

<!-- 5. CASOS -->
<h2>Casos</h2>
<div class="tabla-scroll">
<table>
  <tr><th>Caso</th><th>Métrica</th><th>Cambio</th><th>Dirección</th><th>Evidencia</th>
      <th>Base</th><th>Cand.</th><th>p corregido</th><th>Qué cambió</th></tr>
  {% for fila in filas %}
  <tr>
    <td><code>{{ fila.outcome.case_id }}</code></td>
    <td>{{ fila.outcome.metric }}</td>
    <td>{{ fila.estado }}</td>
    <td>{{ fila.direccion }}</td>
    <td>{{ fila.evidencia }}</td>
    <td>{{ '%.3g'|format(fila.outcome.baseline_score) if fila.outcome.baseline_score is not none else '—' }}</td>
    <td>{{ '%.3g'|format(fila.outcome.candidate_score) if fila.outcome.candidate_score is not none else '—' }}</td>
    <td>{{ '%.4f'|format(fila.outcome.p_value_corrected) if fila.outcome.p_value_corrected is not none else '—' }}</td>
    <td>{% for etiqueta in fila.outcome.change_taxonomy %}<span class="etiqueta">{{ etiqueta }}</span>{% endfor %}</td>
  </tr>
  {% if fila.outcome.notes %}
  <tr><td colspan="9" class="nota">{{ fila.outcome.notes | join(' · ') }}</td></tr>
  {% endif %}
  {% endfor %}
</table>
</div>

{% if informe.semantic_review %}
{% set review = informe.semantic_review %}
<h2>Revisión semántica orientativa · System 1</h2>
<p>{{ 'Modo sombra: registra propuestas sin cambiar el veredicto ni omitir análisis.'
       if review.config.shadow_mode else 'Modo activo: prioriza análisis y conserva las reglas de evidencia.' }}</p>
<p>La confianza es una puntuación del proveedor. No demuestra una probabilidad de acierto calibrada.</p>
<p>Pares: {{ review.metrics.pairs_evaluated }} · resueltos con checks: {{ review.metrics.deterministic_pairs }} ·
   enviados a System 1: {{ review.metrics.system1_pairs }} ·
   candidatos sobre umbral: {{ review.metrics.regression_candidates }} ·
   jueces ejecutados: {{ review.metrics.deep_analysis_executed }} · fallbacks: {{ review.metrics.fallbacks }}</p>
<div class="tabla-scroll"><table>
  <tr><th>Caso</th><th>Juicio orientativo</th><th>Confianza</th><th>Prioridad</th><th>Análisis profundo</th></tr>
  {% for item in review.assessments %}
  <tr><td>{{ item.case_id }}</td><td>{{ etiquetas_semanticas[item.classification] }}</td>
      <td>{{ '%.3f'|format(item.confidence) if item.confidence is not none else '—' }}</td>
      <td>{{ {'normal': 'normal', 'review': 'revisar', 'high': 'alta'}[item.priority] }}</td>
      <td>{{ 'juez ejecutado' if item.deep_analysis_executed else
             'pendiente' if item.deep_analysis_required else 'no requerido' }}</td></tr>
  {% endfor %}
</table></div>
<details><summary>Auditoría: checks, proveedor, fallback, umbrales y decisiones</summary>
  <pre><code>{{ review.model_dump_json(indent=2) }}</code></pre>
</details>
{% endif %}

<!-- 6. JUEZ -->
{% if juez %}
<h2>El juez como instrumento</h2>
<table>
  <tr><th>Sesgo de posición</th>
      <td>{{ fmt(juez.position_bias) }}</td></tr>
  <tr><th>Autoconsistencia</th>
      <td>{{ fmt(juez.judge_self_consistency) }}</td></tr>
  <tr><th>Concordancia juez-humano ({{ juez.kappa_kind }})</th>
      <td>{{ fmt(juez.judge_human_agreement) }}</td></tr>
  <tr><th>Concordancia humano-humano</th>
      <td>{{ fmt(juez.human_human_agreement) }}</td></tr>
</table>
<p>{{ juez.interpretation() }}</p>
{% endif %}

<!-- 7. MANIFIESTO -->
{% if manifiesto %}
<h2>Manifiesto de la tirada candidata</h2>
<details>
  <summary>Instrumentos, huellas y parámetros ({{ manifiesto.instruments | length }} instrumentos)</summary>
  <pre><code>{{ manifiesto.to_json() }}</code></pre>
</details>
{% endif %}

<p class="nota">Model_Drift · las palabras «mejora» y «regresión» solo aparecen sobre casos con
evidencia evaluativa; el resto se informa como «cambio sin dirección evaluable».</p>
"""
)


def _fmt(value: float | None) -> str:
    """Formatea una métrica del juez. «no medida» nunca se confunde con cero.

    Un cero en autoconsistencia significa que el juez se contradice siempre; no
    haberla medido significa que no se sabe. Colapsarlas en el informe sería
    presentar una ignorancia como un dato.
    """
    return "no medida" if value is None else f"{value:.3f}"


def _environment() -> Environment:
    entorno = Environment(autoescape=select_autoescape(["html"]), trim_blocks=True, lstrip_blocks=True)
    entorno.globals["fmt"] = _fmt
    return entorno


def render_comparison(
    informe: ComparisonReport,
    *,
    titulo: str = "Informe de deriva",
    baseline_label: str = "línea base",
    candidate_label: str = "candidata",
    manifest: RunManifest | None = None,
    judge_metrics: JudgeMetrics | None = None,
) -> str:
    """Genera el HTML del informe comparativo."""
    filas = [
        {
            "outcome": outcome,
            "estado": STATUS_LABELS[outcome.change_status],
            "direccion": direction_label(outcome),
            "evidencia": outcome.evidence_kind.value if outcome.evidence_kind else "—",
        }
        for outcome in sorted(
            informe.outcomes,
            key=lambda item: (
                # Lo accionable primero: una regresión relevante no puede quedar
                # sepultada bajo cuarenta filas sin cambios.
                0 if item.is_actionable else 1 if item.is_regression else 2,
                item.case_id,
                item.metric,
            ),
        )
    ]
    reproducibilidad = manifest.reproducibility.declare() if manifest else []
    garantias = list(manifest.missing_guarantees) if manifest else []

    plantilla = _environment().from_string(_TEMPLATE)
    return plantilla.render(
        titulo=titulo,
        base=baseline_label,
        candidata=candidate_label,
        fecha=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        informe=informe,
        filas=filas,
        reproducibilidad=reproducibilidad,
        garantias_ausentes=garantias,
        manifiesto=manifest,
        juez=judge_metrics,
        etiqueta_veredicto=VERDICT_LABELS.get(informe.verdict.verdict, "—") if informe.verdict else "—",
        etiquetas_practicas=PRACTICAL_LABELS,
        etiquetas_semanticas={
            "equivalent": "equivalente",
            "changed_but_not_regression": "cambio sin regresión propuesta",
            "regression_candidate": "posible regresión: requiere análisis",
            "uncertain": "incierto",
        },
    )


def write_comparison(
    informe: ComparisonReport,
    path: str | Path,
    **kwargs: object,
) -> Path:
    destino = Path(path)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(render_comparison(informe, **kwargs), encoding="utf-8")  # type: ignore[arg-type]
    return destino


def render_text_summary(informe: ComparisonReport) -> str:
    """Resumen para la terminal. Mismas reglas de vocabulario que el HTML."""
    lineas: list[str] = []
    lineas.append(informe.comparability.headline())
    if informe.verdict:
        lineas.append(f"Veredicto: {VERDICT_LABELS.get(informe.verdict.verdict, '—')} — {informe.verdict.summary()}")
    lineas.append(informe.power_statement())
    lineas.append(
        f"Casos {informe.n_cases} · comparados {informe.n_compared} · "
        f"indeterminados {informe.n_indeterminate} · no aplicables {informe.n_not_applicable}"
    )

    accionables = informe.actionable()
    if accionables:
        lineas.append("")
        lineas.append("Regresiones relevantes:")
        for outcome in accionables:
            lineas.append(
                f"  - {outcome.case_id} [{outcome.metric}] "
                f"{outcome.baseline_score:.3g} -> {outcome.candidate_score:.3g} "
                f"({', '.join(outcome.change_taxonomy) or 'sin taxonomía'})"
            )

    sin_direccion = [
        item
        for item in informe.content_outcomes()
        if item.change_status is ChangeStatus.cambio_significativo
        and item.change_direction is ChangeDirection.sin_direccion
    ]
    if sin_direccion:
        lineas.append("")
        lineas.append("Cambios detectados SIN dirección evaluable (hace falta oráculo, relación o juez calibrado):")
        for outcome in sin_direccion:
            lineas.append(f"  - {outcome.case_id} [{outcome.metric}]")
    if informe.semantic_review is not None:
        review = informe.semantic_review
        metrics = review.metrics
        lineas.extend([
            "",
            f"System 1 ({'sombra' if review.config.shadow_mode else 'activo'}): "
            f"{metrics.system1_pairs} pares enviados · {metrics.regression_candidates} candidatos sobre umbral · "
            f"{metrics.deep_analysis_executed} análisis ejecutados · {metrics.fallbacks} fallbacks.",
        ])
        for item in review.assessments:
            pending = " · análisis pendiente" if item.deep_analysis_required and not item.deep_analysis_executed else ""
            lineas.append(f"  - {item.case_id}: {item.classification} · {item.priority}{pending} "
                          f"({item.reason_code or item.provider or item.source})")
    return "\n".join(lineas)

