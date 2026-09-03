"""Aserción de resultado SQL: se ejecuta la consulta generada en solo lectura.

Ejecutar SQL escrito por un modelo es una decisión del operador, no un
detalle de implementación: sin `allow_code_execution` la aserción queda
`not_applicable`.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Literal

from model_drift.oracles.base import BaseAssertion
from model_drift.oracles.tipos import AssertionResult, EvaluationInput, EvaluationServices
from model_drift.oracles.utilidades import _SQL_FENCE


class SqlResultAssertion(BaseAssertion):
    """Ejecuta el SQL generado contra una BD fixture y compara conjuntos.

    Evalúa **semántica, no cadenas**: dos consultas escritas de forma distinta
    que devuelven lo mismo son ambas correctas, y la comparación textual las
    habría suspendido.
    """

    type: Literal["sql_result"] = "sql_result"
    database: str
    expected: Any
    #: Por defecto el orden no importa: solo lo exige un ORDER BY explícito en
    #: el enunciado, y eso lo declara la suite.
    ordered: bool = False

    def _evaluate(self, data: EvaluationInput, services: EvaluationServices) -> AssertionResult:
        path = services.resolver.path_for(self.database)
        if not path.is_file():
            return self._skip(f"BD de prueba no encontrada: {self.database}")
        match = _SQL_FENCE.search(data.text)
        query = (match.group(1) if match else data.text).strip().rstrip(";")
        if not query:
            return self._result(False, "la salida no contiene ninguna consulta")
        esperado = services.resolver.resolve_maybe(self.expected)
        # Solo lectura: una consulta generada por un modelo no toca el fixture.
        uri = f"file:{path.as_posix()}?mode=ro"
        try:
            with sqlite3.connect(uri, uri=True) as connection:
                filas = connection.execute(query).fetchall()
        except sqlite3.Error as error:
            return self._result(False, f"la consulta no se pudo ejecutar: {error}")
        obtenido = [list(fila) for fila in filas]
        referencia = [list(fila) for fila in esperado]
        if not self.ordered:
            obtenido_cmp = sorted(map(repr, obtenido))
            referencia_cmp = sorted(map(repr, referencia))
            igual = obtenido_cmp == referencia_cmp
        else:
            igual = obtenido == referencia
        if igual:
            return self._result(True, f"{len(obtenido)} filas coincidentes")
        return self._result(False, f"devolvió {len(obtenido)} filas; se esperaban {len(referencia)}")
