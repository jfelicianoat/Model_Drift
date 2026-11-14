"""Aserción de resultado SQL: se ejecuta la consulta generada en solo lectura.

Ejecutar SQL escrito por un modelo es una decisión del operador, no un
detalle de implementación: sin `allow_code_execution` la aserción queda
`not_applicable`.
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from typing import Any, Literal

from model_drift.oracles.base import BaseAssertion
from model_drift.oracles.tipos import AssertionResult, EvaluationInput, EvaluationServices
from model_drift.oracles.utilidades import _SQL_FENCE

#: Lo único que puede hacer una consulta escrita por un modelo: leer.
_ACCIONES_PERMITIDAS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}
#: Segundos que puede tardar la consulta antes de interrumpirla.
_PLAZO_CONSULTA_S = 5.0


def _solo_lectura(accion: int, *_: object) -> int:
    return sqlite3.SQLITE_OK if accion in _ACCIONES_PERMITIDAS else sqlite3.SQLITE_DENY


def _normalizar(valor: Any) -> Any:
    """Enteros y reales se comparan por valor: SUM() devuelve 3 o 3.0 según el motor."""
    return float(valor) if isinstance(valor, (int, float)) and not isinstance(valor, bool) else valor


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
            with closing(sqlite3.connect(uri, uri=True)) as connection:
                connection.set_authorizer(_solo_lectura)
                plazo = time.monotonic() + _PLAZO_CONSULTA_S
                connection.set_progress_handler(lambda: 1 if time.monotonic() > plazo else 0, 10_000)
                filas = connection.execute(query).fetchall()
        except sqlite3.Error as error:
            return self._result(False, f"la consulta no se pudo ejecutar: {error}")
        obtenido = [[_normalizar(valor) for valor in fila] for fila in filas]
        referencia = [[_normalizar(valor) for valor in fila] for fila in esperado]
        if not self.ordered:
            obtenido_cmp = sorted(map(repr, obtenido))
            referencia_cmp = sorted(map(repr, referencia))
            igual = obtenido_cmp == referencia_cmp
        else:
            igual = obtenido == referencia
        if igual:
            return self._result(True, f"{len(obtenido)} filas coincidentes")
        return self._result(False, f"devolvió {len(obtenido)} filas; se esperaban {len(referencia)}")
