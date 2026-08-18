"""Pruebas de casos, suites y fixtures."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from model_drift.cases import Case, Provenance, Suite, SuiteLoadError, load_suite
from model_drift.fixtures import FixtureError, FixtureResolver

# --------------------------------------------------------------------------
# Casos
# --------------------------------------------------------------------------


def test_el_caso_simple_son_cuatro_lineas(tmp_path: Path) -> None:
    """Si escribir un caso trivial cuesta un bloque de diez líneas, nadie escribe suites."""
    fichero = tmp_path / "suite.yaml"
    fichero.write_text(
        yaml.safe_dump(
            {
                "id": "s",
                "cases": [
                    {"id": "c1", "prompt": "¿2+2?", "assertions": [{"type": "contains", "values": ["4"]}]}
                ],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    suite = load_suite(fichero)
    caso = suite.cases[0]
    assert caso.user_prompt == "¿2+2?"
    assert caso.system_prompt is None
    assert caso.is_multi_turn is False


def test_el_azucar_prompt_mas_system(tmp_path: Path) -> None:
    fichero = tmp_path / "suite.yaml"
    fichero.write_text(
        yaml.safe_dump({"id": "s", "cases": [{"id": "c1", "system": "Responde en JSON.", "prompt": "dame el total"}]}),
        encoding="utf-8",
    )
    caso = load_suite(fichero).cases[0]
    assert caso.system_prompt == "Responde en JSON."
    assert caso.user_prompt == "dame el total"


def test_un_caso_multi_turno_se_representa_entero() -> None:
    caso = Case(
        id="c",
        messages=[
            {"role": "system", "content": "Sé breve."},
            {"role": "user", "content": "¿Capital de Australia?"},
            {"role": "assistant", "content": "Canberra."},
            {"role": "user", "content": "¿Seguro?"},
        ],
    )
    assert caso.is_multi_turn is True
    assert caso.user_prompt == "¿Seguro?"
    assert caso.system_prompt == "Sé breve."


def test_un_caso_sin_mensaje_de_usuario_se_rechaza() -> None:
    with pytest.raises(ValueError, match="user"):
        Case(id="c", messages=[{"role": "system", "content": "solo system"}])


def test_un_caso_sin_mensajes_se_rechaza() -> None:
    with pytest.raises(ValueError, match="mensaje"):
        Case(id="c")


# --------------------------------------------------------------------------
# La regla de la evidencia de nivel 2
# --------------------------------------------------------------------------


def test_un_caso_capturado_pelado_no_sostiene_direccion() -> None:
    """Su salida histórica es línea base de cambio, no verdad de referencia."""
    caso = Case(
        id="cap",
        provenance=Provenance.capturado,
        messages=[{"role": "user", "content": "x"}],
        baseline_output="lo que respondió en su día",
    )
    assert caso.provides_level_2_evidence() is False


def test_la_promocion_puede_darle_evidencia_de_tres_formas() -> None:
    base = {"id": "cap", "provenance": "capturado", "messages": [{"role": "user", "content": "x"}]}
    con_asercion = Case.model_validate({**base, "assertions": [{"type": "json_valid"}]})
    con_relacion = Case.model_validate({**base, "metamorphic_relations": ["mr-1"]})
    con_humano = Case.model_validate({**base, "human_label": "correcta"})
    assert con_asercion.provides_level_2_evidence()
    assert con_relacion.provides_level_2_evidence()
    assert con_humano.provides_level_2_evidence()


# --------------------------------------------------------------------------
# Suites
# --------------------------------------------------------------------------


def test_el_hash_de_la_suite_cambia_si_cambia_un_caso() -> None:
    """Un hash que no reacciona a un cambio no protege la comparabilidad."""
    primera = Suite(id="s", cases=[Case(id="c", messages=[{"role": "user", "content": "a"}])])
    segunda = Suite(id="s", cases=[Case(id="c", messages=[{"role": "user", "content": "b"}])])
    assert primera.hash() != segunda.hash()
    assert len(primera.short_hash()) == 12


def test_el_hash_no_depende_del_orden_de_los_casos() -> None:
    uno = Case(id="a", messages=[{"role": "user", "content": "x"}])
    dos = Case(id="b", messages=[{"role": "user", "content": "y"}])
    assert Suite(id="s", cases=[uno, dos]).hash() == Suite(id="s", cases=[dos, uno]).hash()


def test_identificadores_duplicados_se_rechazan() -> None:
    caso = Case(id="c", messages=[{"role": "user", "content": "x"}])
    with pytest.raises(ValueError, match="duplicado"):
        Suite(id="s", cases=[caso, caso.model_copy()])


def test_carga_desde_directorio_con_casos_repartidos(tmp_path: Path) -> None:
    (tmp_path / "casos").mkdir()
    (tmp_path / "suite.yaml").write_text(
        yaml.safe_dump({"id": "grande", "version": "2", "case_pass_rule": "mayoria"}), encoding="utf-8"
    )
    (tmp_path / "casos" / "a.yaml").write_text(
        yaml.safe_dump([{"id": "a1", "prompt": "uno"}, {"id": "a2", "prompt": "dos"}]), encoding="utf-8"
    )
    (tmp_path / "casos" / "b.yaml").write_text(yaml.safe_dump({"id": "b1", "prompt": "tres"}), encoding="utf-8")

    suite = load_suite(tmp_path)
    assert suite.case_pass_rule == "mayoria"
    assert {caso.id for caso in suite.cases} == {"a1", "a2", "b1"}
    assert suite.case_by_id("a2") is not None
    assert suite.case_by_id("no-existe") is None


def test_errores_de_carga_dicen_qué_fichero(tmp_path: Path) -> None:
    with pytest.raises(SuiteLoadError, match="no existe"):
        load_suite(tmp_path / "no-esta.yaml")

    roto = tmp_path / "suite.yaml"
    roto.write_text("esto: [no cierra", encoding="utf-8")
    with pytest.raises(SuiteLoadError, match="YAML inválido"):
        load_suite(roto)

    lista = tmp_path / "lista.yaml"
    lista.write_text(yaml.safe_dump([1, 2, 3]), encoding="utf-8")
    with pytest.raises(SuiteLoadError, match="mapa"):
        load_suite(lista)

    caso_malo = tmp_path / "malo.yaml"
    caso_malo.write_text(yaml.safe_dump({"id": "s", "cases": [{"id": "sin-mensajes"}]}), encoding="utf-8")
    with pytest.raises(SuiteLoadError, match="caso inválido"):
        load_suite(caso_malo)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def test_los_fixtures_se_leen_y_se_hashean(tmp_path: Path) -> None:
    (tmp_path / "schemas").mkdir()
    destino = tmp_path / "schemas" / "factura.json"
    destino.write_text(json.dumps({"type": "object"}), encoding="utf-8")

    resolver = FixtureResolver(tmp_path)
    assert resolver.read_json("fixture://schemas/factura.json") == {"type": "object"}
    digest = resolver.hash_of("fixture://schemas/factura.json")
    assert len(digest) == 64
    assert resolver.used_hashes() == {"fixture://schemas/factura.json": digest}


def test_un_fixture_que_cambia_cambia_su_hash(tmp_path: Path) -> None:
    """Es lo que permite distinguir un instrumento que cambió de un modelo que cambió."""
    ruta = tmp_path / "esquema.json"
    ruta.write_text('{"type": "object"}', encoding="utf-8")
    primero = FixtureResolver(tmp_path).hash_of("fixture://esquema.json")
    ruta.write_text('{"type": "array"}', encoding="utf-8")
    segundo = FixtureResolver(tmp_path).hash_of("fixture://esquema.json")
    assert primero != segundo


def test_un_fixture_no_puede_salirse_de_su_suite(tmp_path: Path) -> None:
    """La suite es la unidad que se versiona: una ruta con '..' rompe el aislamiento."""
    resolver = FixtureResolver(tmp_path / "suite")
    with pytest.raises(FixtureError, match="fuera de la suite"):
        resolver.path_for("fixture://../secreto.txt")


def test_errores_de_fixture(tmp_path: Path) -> None:
    resolver = FixtureResolver(tmp_path)
    with pytest.raises(FixtureError, match="[Nn]o es una referencia"):
        resolver.path_for("/ruta/normal.json")
    with pytest.raises(FixtureError, match="vacía"):
        resolver.path_for("fixture://")
    with pytest.raises(FixtureError, match="no encontrado"):
        resolver.read_bytes("fixture://falta.json")

    (tmp_path / "roto.json").write_text("{no es json", encoding="utf-8")
    with pytest.raises(FixtureError, match="no es JSON"):
        resolver.read_json("fixture://roto.json")

    (tmp_path / "roto.yaml").write_text("a: [", encoding="utf-8")
    with pytest.raises(FixtureError, match="no es YAML"):
        resolver.read_yaml("fixture://roto.yaml")


def test_resolve_maybe_deja_pasar_lo_que_no_es_referencia(tmp_path: Path) -> None:
    resolver = FixtureResolver(tmp_path)
    assert resolver.resolve_maybe({"type": "object"}) == {"type": "object"}
    (tmp_path / "x.yaml").write_text(yaml.safe_dump({"a": 1}), encoding="utf-8")
    assert resolver.resolve_maybe("fixture://x.yaml") == {"a": 1}
