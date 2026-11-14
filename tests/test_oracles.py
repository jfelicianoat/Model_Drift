"""Pruebas de los oráculos deterministas."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from model_drift.embeddings import StaticEmbedder
from model_drift.fixtures import FixtureResolver
from model_drift.oracles import (
    AbstentionAssertion,
    CitationsGroundedAssertion,
    ContainsAssertion,
    EvaluationInput,
    EvaluationServices,
    ExactAssertion,
    FieldEqualsAssertion,
    JsonSchemaAssertion,
    JsonValidAssertion,
    LengthAssertion,
    NumericCloseAssertion,
    RegexAssertion,
    SemanticSimilarityAssertion,
    SqlResultAssertion,
    ToolCall,
    ToolCallAssertion,
    aggregate_pass,
    cosine_similarity,
    extract_json,
    normalize_text,
    resolve_path,
    strip_code_fences,
)


@pytest.fixture()
def servicios(tmp_path: Path) -> EvaluationServices:
    return EvaluationServices(resolver=FixtureResolver(tmp_path))


def _entrada(texto: str = "", **kwargs: object) -> EvaluationInput:
    return EvaluationInput(text=texto, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------


def test_la_normalizacion_no_toca_tildes() -> None:
    """En español quitar tildes cambia el significado: no es normalizar, es romper."""
    assert normalize_text("  Está   AQUÍ ") == "está aquí"
    assert normalize_text("esta") != normalize_text("está")


def test_extraccion_de_json_con_y_sin_vallas() -> None:
    assert extract_json('{"a": 1}')[0] == {"a": 1}
    assert extract_json('```json\n{"a": 1}\n```')[0] == {"a": 1}
    assert extract_json('Aquí tienes:\n{"a": 1}\nEso es todo.')[0] == {"a": 1}
    valor, motivo = extract_json("no hay json por ningún lado")
    assert valor is None and "parseable" in motivo


def test_strip_code_fences_devuelve_el_texto_si_no_hay_vallas() -> None:
    assert strip_code_fences("texto llano") == "texto llano"


def test_resolucion_de_rutas() -> None:
    datos = {"factura": {"lineas": [{"total": 12.5}]}}
    assert resolve_path(datos, "$.factura.lineas[0].total") == (12.5, True)
    assert resolve_path(datos, "$.factura.nope") == (None, False)
    assert resolve_path(datos, "$.factura.lineas[9].total") == (None, False)


def test_similitud_coseno_bordes() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


# --------------------------------------------------------------------------
# Aserciones básicas
# --------------------------------------------------------------------------


def test_exact_normalizado(servicios: EvaluationServices) -> None:
    aserto = ExactAssertion(value="Hola Mundo")
    assert aserto.evaluate(_entrada("  hola   mundo "), servicios).passed
    assert not aserto.evaluate(_entrada("adiós"), servicios).passed


def test_contains_en_las_dos_direcciones(servicios: EvaluationServices) -> None:
    debe = ContainsAssertion(values=["total", "IVA"])
    assert debe.evaluate(_entrada("El total con IVA es 100"), servicios).passed
    resultado = debe.evaluate(_entrada("El importe es 100"), servicios)
    assert not resultado.passed and "total" in resultado.reason

    no_debe = ContainsAssertion(values=["password"], should_contain=False)
    assert no_debe.evaluate(_entrada("nada sensible"), servicios).passed
    assert not no_debe.evaluate(_entrada("la password es 1234"), servicios).passed


def test_regex(servicios: EvaluationServices) -> None:
    aserto = RegexAssertion(pattern=r"\d{2}/\d{2}/\d{4}")
    assert aserto.evaluate(_entrada("con fecha 01/02/2026"), servicios).passed
    assert not aserto.evaluate(_entrada("sin fecha"), servicios).passed
    negado = RegexAssertion(pattern=r"error", should_match=False)
    assert negado.evaluate(_entrada("todo bien"), servicios).passed


def test_json_valid_y_json_schema(servicios: EvaluationServices) -> None:
    assert JsonValidAssertion().evaluate(_entrada('{"a":1}'), servicios).passed
    assert not JsonValidAssertion().evaluate(_entrada("hola"), servicios).passed

    esquema = JsonSchemaAssertion(
        schema_ref={"type": "object", "required": ["total"], "properties": {"total": {"type": "number"}}}
    )
    assert esquema.evaluate(_entrada('{"total": 10}'), servicios).passed
    fallo = esquema.evaluate(_entrada('{"total": "diez"}'), servicios)
    assert not fallo.passed and "total" in fallo.reason
    assert not esquema.evaluate(_entrada("no es json"), servicios).passed


def test_un_esquema_invalido_no_suspende_al_modelo(servicios: EvaluationServices) -> None:
    """Un instrumento roto no puede contarse como fallo del evaluado."""
    aserto = JsonSchemaAssertion(schema_ref={"type": "no-existe-este-tipo"})
    resultado = aserto.evaluate(_entrada('{"a":1}'), servicios)
    assert resultado.not_applicable is True
    assert resultado.passed is False


def test_field_equals_con_tolerancia(servicios: EvaluationServices) -> None:
    aserto = FieldEqualsAssertion(path="$.total", value=1250.40, tolerance=0.01)
    assert aserto.evaluate(_entrada('{"total": 1250.404}'), servicios).passed
    assert not aserto.evaluate(_entrada('{"total": 1300}'), servicios).passed
    ausente = aserto.evaluate(_entrada('{"otro": 1}'), servicios)
    assert not ausente.passed and "no existe" in ausente.reason


def test_numeric_close(servicios: EvaluationServices) -> None:
    aserto = NumericCloseAssertion(expected=42.0, tolerance=0.5)
    assert aserto.evaluate(_entrada("el resultado es 42,3"), servicios).passed
    assert not aserto.evaluate(_entrada("el resultado es 50"), servicios).passed
    assert not aserto.evaluate(_entrada("sin números"), servicios).passed
    primero = NumericCloseAssertion(expected=1.0, tolerance=0.0, pick="first")
    assert primero.evaluate(_entrada("1 y luego 99"), servicios).passed


def test_length_detecta_verbosidad(servicios: EvaluationServices) -> None:
    aserto = LengthAssertion(max_words=5)
    assert aserto.evaluate(_entrada("una respuesta corta"), servicios).passed
    largo = aserto.evaluate(_entrada(" ".join(["palabra"] * 20)), servicios)
    assert not largo.passed and "palabras" in largo.reason


def test_tool_call_con_argumentos(servicios: EvaluationServices) -> None:
    entrada = _entrada("", tool_calls=[ToolCall(name="consultar_saldo", arguments={"cliente": 42})])
    assert ToolCallAssertion(name="consultar_saldo").evaluate(entrada, servicios).passed
    assert ToolCallAssertion(name="consultar_saldo", arguments={"cliente": 42}).evaluate(entrada, servicios).passed
    malo = ToolCallAssertion(name="consultar_saldo", arguments={"cliente": 7}).evaluate(entrada, servicios)
    assert not malo.passed and "argumentos incorrectos" in malo.reason
    otra = ToolCallAssertion(name="borrar_todo").evaluate(entrada, servicios)
    assert not otra.passed
    assert ToolCallAssertion(name="borrar_todo", should_call=False).evaluate(entrada, servicios).passed


def test_citas_con_procedencia(servicios: EvaluationServices) -> None:
    aserto = CitationsGroundedAssertion()
    con_fuente = _entrada("Según https://ejemplo.org/a, sí.", sources=["https://ejemplo.org/a"])
    assert aserto.evaluate(con_fuente, servicios).passed
    inventada = _entrada("Según https://inventado.org/x, sí.", sources=["https://ejemplo.org/a"])
    resultado = aserto.evaluate(inventada, servicios)
    assert not resultado.passed and "no consultó" in resultado.reason
    # Sin citas: no inventa, así que pasa salvo que se exija citar.
    assert aserto.evaluate(_entrada("Sin enlaces."), servicios).passed
    assert not CitationsGroundedAssertion(require_at_least_one=True).evaluate(
        _entrada("Sin enlaces."), servicios
    ).passed


def test_abstencion(servicios: EvaluationServices) -> None:
    aserto = AbstentionAssertion()
    assert aserto.evaluate(_entrada("Eso no consta en el documento."), servicios).passed
    fallo = aserto.evaluate(_entrada("La respuesta es 42."), servicios)
    assert not fallo.passed and "no se abstuvo" in fallo.reason
    inverso = AbstentionAssertion(should_abstain=False)
    assert inverso.evaluate(_entrada("La respuesta es 42."), servicios).passed


# --------------------------------------------------------------------------
# SQL por ejecución
# --------------------------------------------------------------------------


def test_sql_compara_conjuntos_no_cadenas(tmp_path: Path) -> None:
    """Dos consultas escritas distinto que devuelven lo mismo son ambas correctas."""
    ruta = tmp_path / "facturas.sqlite"
    conexion = sqlite3.connect(ruta)
    conexion.execute("CREATE TABLE facturas (id INTEGER, total REAL)")
    conexion.executemany("INSERT INTO facturas VALUES (?, ?)", [(1, 100.0), (2, 200.0)])
    conexion.commit()
    conexion.close()

    servicios = EvaluationServices(resolver=FixtureResolver(tmp_path))
    aserto = SqlResultAssertion(database="fixture://facturas.sqlite", expected=[[1, 100.0], [2, 200.0]])

    directa = _entrada("SELECT id, total FROM facturas")
    assert aserto.evaluate(directa, servicios).passed
    # Mismo conjunto, otro orden y con valla de código: sigue siendo correcta.
    otra = _entrada("```sql\nSELECT id, total FROM facturas ORDER BY total DESC\n```")
    assert aserto.evaluate(otra, servicios).passed
    # Conjunto distinto: falla.
    filtrada = _entrada("SELECT id, total FROM facturas WHERE id = 1")
    assert not aserto.evaluate(filtrada, servicios).passed
    # SQL inválido: falla con motivo, no revienta.
    rota = aserto.evaluate(_entrada("SELECT * FROM no_existe"), servicios)
    assert not rota.passed and "no se pudo ejecutar" in rota.reason


def test_sql_sin_base_de_datos_no_aplica(servicios: EvaluationServices) -> None:
    aserto = SqlResultAssertion(database="fixture://no-esta.sqlite", expected=[])
    resultado = aserto.evaluate(_entrada("SELECT 1"), servicios)
    assert resultado.not_applicable is True


# --------------------------------------------------------------------------
# Similitud semántica: el umbral exige procedencia
# --------------------------------------------------------------------------


def test_un_umbral_sin_calibracion_se_rechaza_al_construir() -> None:
    """Criterio de aceptación d de la Fase 1: falla al cargar, no a las tres horas."""
    with pytest.raises(ValidationError) as error:
        SemanticSimilarityAssertion(reference="x", threshold=0.87, calibration="   ")
    assert "calibration" in str(error.value)


def test_similitud_con_instrumento_y_umbral_calibrado(tmp_path: Path) -> None:
    embedder = StaticEmbedder(vectors={"igual": [1.0, 0.0], "parecido": [0.99, 0.14]})
    servicios = EvaluationServices(resolver=FixtureResolver(tmp_path), embedder=embedder)
    aserto = SemanticSimilarityAssertion(
        reference="igual", threshold=0.9, calibration="fixture://calib/umbral.json"
    )
    resultado = aserto.evaluate(_entrada("parecido"), servicios)
    assert resultado.passed is True
    assert resultado.score is not None and resultado.score > 0.9
    # El umbral usado queda registrado para el manifiesto.
    assert servicios.used_thresholds


def test_sin_instrumento_de_embeddings_la_similitud_no_aplica(servicios: EvaluationServices) -> None:
    aserto = SemanticSimilarityAssertion(reference="x", threshold=0.9, calibration="fixture://c.json")
    resultado = aserto.evaluate(_entrada("y"), servicios)
    assert resultado.not_applicable is True


# --------------------------------------------------------------------------
# Agregación
# --------------------------------------------------------------------------


def test_una_invocacion_sin_salida_deja_todo_no_aplicable(servicios: EvaluationServices) -> None:
    entrada = EvaluationInput(text="", produced_output=False)
    resultado = JsonValidAssertion().evaluate(entrada, servicios)
    assert resultado.not_applicable is True
    assert resultado.passed is False


def test_sin_aserciones_aplicables_no_hay_suspenso(servicios: EvaluationServices) -> None:
    """Ausencia de medición no es un cero."""
    entrada = EvaluationInput(text="", produced_output=False)
    resultados = [JsonValidAssertion().evaluate(entrada, servicios)]
    acierto, motivo = aggregate_pass(resultados)
    assert acierto is None
    assert "ninguna aserción aplicable" in motivo


def test_agregacion_con_fallo(servicios: EvaluationServices) -> None:
    resultados = [
        JsonValidAssertion().evaluate(_entrada('{"a":1}'), servicios),
        ExactAssertion(value="otra cosa").evaluate(_entrada('{"a":1}'), servicios),
    ]
    acierto, motivo = aggregate_pass(resultados)
    assert acierto is False
    assert "exact" in motivo


def test_agregacion_todo_correcto(servicios: EvaluationServices) -> None:
    resultados = [JsonValidAssertion().evaluate(_entrada('{"a":1}'), servicios)]
    assert aggregate_pass(resultados) == (True, "")


# --------------------------------------------------------------------------
# Plan 13-nov: MD-04 (números), MD-05/06 (SQL), MD-07 (URL), MD-09 (abstención)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("el resultado es 42,3", [42.3]),
        ("sin números", []),
        ("1 y luego 99", [1.0, 99.0]),
        ("Total: 1.234,56 EUR", [1234.56]),
        ("Total: 1,234.56 USD", [1234.56]),
        ("Población 1.234.567 habitantes", [1234567.0]),
        ("valor 1.234", [1.234]),
        ("a fecha 2026-10-05 el saldo es 17", [17.0]),
        ("el 05/10/2026 a las 14:30 subió 3,5", [3.5]),
        ("temperatura -5 grados", [-5.0]),
        ("rango 10-20", [10.0, 20.0]),
        ("versión v2 y 7 casos", [7.0]),
        ("termina en 42.", [42.0]),
        ("12, 13 y 14", [12.0, 13.0, 14.0]),
    ],
)
def test_extract_numbers_entiende_miles_y_descarta_fechas(texto: str, esperado: list[float]) -> None:
    from model_drift.oracles.utilidades import extract_numbers

    assert extract_numbers(texto) == pytest.approx(esperado)


def _bd_dos_filas(tmp_path: Path) -> Path:
    ruta = tmp_path / "t.sqlite"
    conexion = sqlite3.connect(ruta)
    conexion.execute("CREATE TABLE t (id INTEGER, total REAL)")
    conexion.executemany("INSERT INTO t VALUES (?, ?)", [(1, 1.0), (2, 2.0)])
    conexion.commit()
    conexion.close()
    return ruta


def test_sql_deniega_attach_y_no_crea_ficheros(tmp_path: Path) -> None:
    _bd_dos_filas(tmp_path)
    servicios = EvaluationServices(resolver=FixtureResolver(tmp_path))
    otro = tmp_path / "otro.db"
    aserto = SqlResultAssertion(database="fixture://t.sqlite", expected=[])
    resultado = aserto.evaluate(_entrada(f"ATTACH DATABASE '{otro.as_posix()}' AS o"), servicios)
    assert not resultado.passed and "not authorized" in resultado.reason
    assert not otro.exists()


def test_sql_corta_una_cte_infinita(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    from model_drift.oracles import sql as modulo_sql

    monkeypatch.setattr(modulo_sql, "_PLAZO_CONSULTA_S", 0.5)
    _bd_dos_filas(tmp_path)
    servicios = EvaluationServices(resolver=FixtureResolver(tmp_path))
    aserto = SqlResultAssertion(database="fixture://t.sqlite", expected=[[1]])
    inicio = time.monotonic()
    resultado = aserto.evaluate(
        _entrada("WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r) SELECT count(*) FROM r"), servicios
    )
    assert not resultado.passed and "interrupted" in resultado.reason
    assert time.monotonic() - inicio < 10


def test_sql_libera_el_fichero_al_terminar(tmp_path: Path) -> None:
    ruta = _bd_dos_filas(tmp_path)
    servicios = EvaluationServices(resolver=FixtureResolver(tmp_path))
    aserto = SqlResultAssertion(database="fixture://t.sqlite", expected=[[1, 1.0], [2, 2.0]])
    assert aserto.evaluate(_entrada("SELECT id, total FROM t"), servicios).passed
    ruta.unlink()


def test_sql_compara_enteros_y_reales_por_valor(tmp_path: Path) -> None:
    _bd_dos_filas(tmp_path)
    servicios = EvaluationServices(resolver=FixtureResolver(tmp_path))
    consulta = _entrada("SELECT sum(id) FROM t")
    assert SqlResultAssertion(database="fixture://t.sqlite", expected=[[3.0]]).evaluate(consulta, servicios).passed
    assert not SqlResultAssertion(database="fixture://t.sqlite", expected=[[4]]).evaluate(consulta, servicios).passed


@pytest.mark.parametrize(
    ("citada", "pasa"),
    [
        ("https://x.es/a", True),
        ("https://x.es/a#seccion", True),
        ("https://x.es/A", False),
        ("https://otro.es/a", False),
    ],
)
def test_citas_normalizan_host_barra_y_fragmento(servicios: EvaluationServices, citada: str, pasa: bool) -> None:
    aserto = CitationsGroundedAssertion()
    resultado = aserto.evaluate(_entrada(f"según {citada} es así", sources=["https://X.es/a/"]), servicios)
    assert resultado.passed is pasa


@pytest.mark.parametrize(
    ("texto", "abstiene"),
    [
        ("No tengo datos suficientes para responder.", True),
        ("Lo siento, no lo sé.", True),
        ("En el documento no se indica la fecha.", True),
        ("No puedo\nconfirmar   esa cifra.", True),
        ("I don't know the answer.", True),
        ("El importe no figura en las fuentes.", True),
        ("La factura asciende a 1.200 euros.", False),
        ("El contrato se firmó el 3 de marzo.", False),
        ("Sí, el informe recomienda subir el precio.", False),
        ("La capital es Madrid.", False),
        ("Hay tres clientes con deuda.", False),
        ("Según la nota, el plazo es de 30 días.", False),
    ],
)
def test_abstencion_dicha_de_formas_distintas(servicios: EvaluationServices, texto: str, abstiene: bool) -> None:
    resultado = AbstentionAssertion(should_abstain=True).evaluate(_entrada(texto), servicios)
    assert resultado.passed is abstiene
