"""Pruebas de las relaciones metamórficas.

El caso límite 7 del plan vive aquí: una prueba metamórfica no vale sin
transformación, relación, comparador y umbral declarados.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from model_drift.embeddings import StaticEmbedder
from model_drift.fixtures import FixtureResolver
from model_drift.metamorphic import (
    Comparator,
    MetamorphicRelation,
    RelationKind,
    SuccessCriterion,
    Threshold,
    Transformation,
    TransformationKind,
    VariantResult,
    evaluate_relation,
    generate_variants,
    load_relations,
)


def _relacion(**kwargs: object) -> MetamorphicRelation:
    base: dict[str, object] = {
        "id": "mr-test",
        "transformation": Transformation(kind=TransformationKind.orden),
        "relation": RelationKind.invariancia,
        "comparator": Comparator(kind="normalizado"),
    }
    base.update(kwargs)
    return MetamorphicRelation.model_validate(base)


# --------------------------------------------------------------------------
# CASO 7: los cuatro campos son obligatorios
# --------------------------------------------------------------------------


def test_una_transformacion_de_parafrasis_exige_variantes_escritas() -> None:
    """Generarlas con otro modelo metería un instrumento sin calibrar."""
    with pytest.raises(ValidationError) as error:
        Transformation(kind=TransformationKind.parafrasis)
    assert "generator" in str(error.value)


def test_un_comparador_por_embeddings_exige_umbral() -> None:
    with pytest.raises(ValidationError) as error:
        _relacion(comparator=Comparator(kind="embedding"))
    assert "umbral calibrado" in str(error.value)


def test_un_umbral_exige_procedencia_de_calibracion() -> None:
    with pytest.raises(ValidationError) as error:
        Threshold(value=0.87, calibration="")
    assert "calibration" in str(error.value)


def test_una_relacion_completa_se_construye_y_se_identifica() -> None:
    relacion = _relacion(
        comparator=Comparator(kind="embedding"),
        threshold=Threshold(value=0.87, calibration="fixture://calib/parafrasis.json"),
        version="3",
    )
    assert relacion.identity == "mr-test@3"


# --------------------------------------------------------------------------
# Generación de variantes
# --------------------------------------------------------------------------


def test_la_transformacion_de_orden_baraja_los_items(tmp_path: Path) -> None:
    prompt = "Elige la mejor opción:\n- alfa\n- beta\n- gamma"
    relacion = _relacion(transformation=Transformation(kind=TransformationKind.orden, variants=2))
    variantes = generate_variants(relacion, prompt, FixtureResolver(tmp_path))
    assert len(variantes) == 2
    for _, texto in variantes:
        assert set(texto.split()) == set(prompt.split()), "barajar no puede perder ni añadir ítems"
        assert texto != prompt


def test_la_transformacion_de_orden_es_reproducible(tmp_path: Path) -> None:
    """Una transformación que baraja distinto en cada tirada no es un instrumento."""
    prompt = "Opciones:\n1. uno\n2. dos\n3. tres\n4. cuatro"
    relacion = _relacion(transformation=Transformation(kind=TransformationKind.orden, variants=3, seed=99))
    primera = generate_variants(relacion, prompt, FixtureResolver(tmp_path))
    segunda = generate_variants(relacion, prompt, FixtureResolver(tmp_path))
    assert primera == segunda


def test_sin_lista_no_hay_variantes_de_orden(tmp_path: Path) -> None:
    relacion = _relacion(transformation=Transformation(kind=TransformationKind.orden))
    assert generate_variants(relacion, "Una pregunta sin lista.", FixtureResolver(tmp_path)) == []


def test_contexto_irrelevante_pone_el_distractor_a_los_dos_lados(tmp_path: Path) -> None:
    relacion = _relacion(
        transformation=Transformation(
            kind=TransformationKind.contexto_irrelevante, variants=2, payload="RUIDO."
        )
    )
    variantes = generate_variants(relacion, "¿Cuánto es 2+2?", FixtureResolver(tmp_path))
    assert len(variantes) == 2
    assert variantes[0][1].startswith("RUIDO.")
    assert variantes[1][1].endswith("RUIDO.")


def test_adulacion_y_anclaje(tmp_path: Path) -> None:
    resolver = FixtureResolver(tmp_path)
    adulacion = generate_variants(
        _relacion(transformation=Transformation(kind=TransformationKind.adulacion)), "2+2?", resolver
    )
    assert "Reconsidéralo" in adulacion[0][1]
    anclaje = generate_variants(
        _relacion(transformation=Transformation(kind=TransformationKind.anclaje, payload="cinco")),
        "2+2?",
        resolver,
    )
    assert "cinco" in anclaje[0][1]


def test_las_variantes_pueden_venir_de_un_fixture(tmp_path: Path) -> None:
    (tmp_path / "transforms").mkdir()
    (tmp_path / "transforms" / "parafrasis.yaml").write_text(
        yaml.safe_dump(["¿Cuál es la capital?", "Dime la capital, por favor."], allow_unicode=True),
        encoding="utf-8",
    )
    relacion = _relacion(
        transformation=Transformation(
            kind=TransformationKind.parafrasis, generator="fixture://transforms/parafrasis.yaml", variants=5
        )
    )
    variantes = generate_variants(relacion, "¿Cuál es la capital de Australia?", FixtureResolver(tmp_path))
    assert len(variantes) == 2
    assert variantes[0][1].startswith("¿Cuál es la capital?")


# --------------------------------------------------------------------------
# Evaluación de la relación
# --------------------------------------------------------------------------


def _variantes(*salidas: str) -> list[VariantResult]:
    return [VariantResult(variant_id=f"v{i}", prompt="p", output=texto) for i, texto in enumerate(salidas)]


def test_invariancia_se_cumple_cuando_la_salida_no_cambia() -> None:
    relacion = _relacion(success_criterion=SuccessCriterion(min_ratio=1.0))
    resultado = evaluate_relation(relacion, "Canberra", _variantes("Canberra", "canberra"))
    assert resultado.passed is True
    assert resultado.satisfied == 2


def test_invariancia_se_viola_cuando_la_salida_cambia() -> None:
    relacion = _relacion()
    resultado = evaluate_relation(relacion, "Canberra", _variantes("Canberra", "Sídney"))
    assert resultado.passed is False
    assert resultado.satisfied == 1


def test_cambio_exigido_invierte_el_criterio() -> None:
    """Sensibilidad a mutaciones: seguir diciendo lo mismo ES el fallo."""
    relacion = _relacion(relation=RelationKind.cambio_exigido)
    igual = evaluate_relation(relacion, "Canberra", _variantes("Canberra"))
    assert igual.passed is False, "no cambiar ante una mutación es la violación"
    distinto = evaluate_relation(relacion, "Canberra", _variantes("Sídney"))
    assert distinto.passed is True


def test_el_criterio_al_menos_n_de_m() -> None:
    relacion = _relacion(success_criterion=SuccessCriterion(min_satisfied=2))
    resultado = evaluate_relation(relacion, "A", _variantes("A", "A", "B"))
    assert resultado.satisfied == 2
    assert resultado.passed is True


def test_criterio_desde_la_frase_del_plan() -> None:
    assert SuccessCriterion.parse("al menos 4 de 5 variantes cumplen la relación").min_satisfied == 4
    assert SuccessCriterion.parse(None).min_ratio == 1.0
    assert SuccessCriterion.parse({"min_ratio": 0.5}).min_ratio == 0.5
    assert SuccessCriterion.parse("todas").min_ratio == 1.0


def test_sin_variantes_la_relacion_no_aplica() -> None:
    resultado = evaluate_relation(_relacion(), "A", [])
    assert resultado.not_applicable is True
    assert resultado.passed is False


def test_comparador_por_embeddings_usa_el_umbral() -> None:
    embedder = StaticEmbedder(vectors={"ref": [1.0, 0.0], "cerca": [0.98, 0.2], "lejos": [0.0, 1.0]})
    relacion = _relacion(
        comparator=Comparator(kind="embedding"),
        threshold=Threshold(value=0.9, calibration="fixture://c.json"),
    )
    resultado = evaluate_relation(relacion, "ref", _variantes("cerca", "lejos"), embedder=embedder)
    assert resultado.satisfied == 1
    assert resultado.variants[0].similarity is not None


def test_sin_embedder_el_comparador_por_embeddings_no_satisface() -> None:
    relacion = _relacion(
        comparator=Comparator(kind="embedding"),
        threshold=Threshold(value=0.9, calibration="fixture://c.json"),
    )
    resultado = evaluate_relation(relacion, "ref", _variantes("cerca"))
    assert resultado.passed is False
    assert "embeddings" in resultado.variants[0].detail


def test_comparador_exacto() -> None:
    relacion = _relacion(comparator=Comparator(kind="exacto"))
    assert evaluate_relation(relacion, "A", _variantes("A")).passed is True
    assert evaluate_relation(relacion, "A", _variantes("a")).passed is False


# --------------------------------------------------------------------------
# Carga
# --------------------------------------------------------------------------


def test_carga_de_relaciones_desde_directorio(tmp_path: Path) -> None:
    (tmp_path / "mr-uno.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "mr-uno",
                "version": "2",
                "transformation": {"kind": "orden", "variants": 3},
                "relation": "invariancia",
                "comparator": {"kind": "normalizado"},
                "success_criterion": "al menos 2 de 3",
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    relaciones = load_relations(tmp_path)
    assert set(relaciones) == {"mr-uno"}
    assert relaciones["mr-uno"].success_criterion.min_satisfied == 2
    assert relaciones["mr-uno"].identity == "mr-uno@2"
