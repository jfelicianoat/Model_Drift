"""Pruebas del cliente del Broker: fijación dura, degradación y huella."""
from __future__ import annotations

import httpx

from model_drift.broker_client import BrokerClient, BrokerConfig
from model_drift.broker_contracts import BrokerError, ModelReference
from tests.broker_fake import FakeBroker

VIEJO = ModelReference(provider="ollama", deployment="local", model="qwen3:6b")
NUEVO = ModelReference(provider="ollama", deployment="local", model="qwen3:8b")


def _broker() -> FakeBroker:
    broker = FakeBroker()
    broker.add_model(VIEJO.key)
    broker.add_model(NUEVO.key)
    broker.respond_fixed(VIEJO.key, "respuesta del modelo viejo")
    broker.respond_fixed(NUEVO.key, "respuesta del modelo nuevo")
    return broker


def test_toda_invocacion_prohibe_el_fallback() -> None:
    """Una comparación con sustitución permitida no compara lo que dice."""
    broker = _broker()
    with broker.client() as cliente:
        cliente.invoke(prompt="hola", target=VIEJO)
    assert broker.requests[0]["model_requirements"]["fallback_allowed"] is False
    assert broker.requests[0]["model_requirements"]["target_model"]["model"] == "qwen3:6b"


def test_el_trafico_va_marcado_como_evaluacion() -> None:
    broker = _broker()
    with broker.client() as cliente:
        cliente.invoke(prompt="hola", target=VIEJO, group="run-42")
    metadata = broker.requests[0]["content"]["metadata"]
    assert metadata["origin"] == "model_drift"
    assert metadata["run_group"] == "run-42"
    assert broker.requests[0]["group"] == "run-42"


def test_si_el_broker_sirve_otro_modelo_la_invocacion_no_vale() -> None:
    """CASO límite de la fijación dura: texto perfecto, modelo equivocado."""
    broker = _broker()
    broker.serve_instead[VIEJO.key] = NUEVO.key
    with broker.client() as cliente:
        resultado = cliente.invoke(prompt="hola", target=VIEJO)
    assert resultado.ok is False
    assert resultado.error_code == "MODEL_DRIFT_TARGET_MISMATCH"
    assert "qwen3:8b" in resultado.error_message


def test_un_fallback_declarado_invalida_la_invocacion() -> None:
    broker = _broker()
    broker.declare_fallback = True
    with broker.client() as cliente:
        resultado = cliente.invoke(prompt="hola", target=VIEJO)
    assert resultado.ok is False
    assert resultado.error_code == "MODEL_DRIFT_FALLBACK_USED"


def test_degradacion_elegante_cuando_el_broker_no_admite_semilla() -> None:
    """Contra un Broker anterior al Cambio 1: se degrada y se DECLARA la pérdida."""
    broker = _broker()
    broker.supports_seed = False
    with broker.client() as cliente:
        resultado = cliente.invoke(prompt="hola", target=VIEJO, params={"seed": 7, "temperature": 0.0})
        assert resultado.ok is True
        assert cliente.config.supports_seed is False
        assert any("no es reproducible" in aviso.lower() for aviso in cliente.missing_guarantees)
    # Primer intento con semilla, segundo sin ella.
    assert "seed" in broker.requests[0]["generation"]
    assert "seed" not in broker.requests[1]["generation"]


def test_si_el_broker_admite_semilla_no_se_degrada() -> None:
    broker = _broker()
    broker.supports_seed = True
    with broker.client() as cliente:
        cliente.invoke(prompt="hola", target=VIEJO, params={"seed": 7})
        assert cliente.config.supports_seed is True
        assert cliente.missing_guarantees == [] or all(
            "reproducible" not in aviso for aviso in cliente.missing_guarantees
        )
    assert broker.requests[0]["generation"]["seed"] == 7


def test_una_tarea_fallida_no_produce_salida_evaluable() -> None:
    broker = _broker()
    broker.fail_prompts["revienta"] = "PROVIDER_UNAVAILABLE"
    with broker.client() as cliente:
        resultado = cliente.invoke(prompt="revienta", target=VIEJO)
    assert resultado.ok is False
    assert resultado.produced_output is False
    assert resultado.error_code == "PROVIDER_UNAVAILABLE"


def test_la_huella_adopta_la_que_publica_el_broker() -> None:
    """Desde el Cambio 2, el catálogo trae la huella con las tres capas hechas.

    Se adopta tal cual: el Broker es quien sabe qué pudo verificar contra el
    runtime, y recalcularlo aquí sería adivinar peor.
    """
    broker = _broker()
    with broker.client() as cliente:
        huella = cliente.fingerprint_for(VIEJO)
        assert huella.get("model") == "qwen3:6b"
        assert huella.get("digest") is not None
        assert huella.get("chat_template") is not None
        assert huella.components["digest"].state.value == "verificado"
        assert huella.components["provider"].state.value == "declarado"
        # Lo que el Broker declara ausente sigue ausente, no se rellena.
        assert "system_fingerprint" in huella.missing_names()
        assert cliente.missing_guarantees == []


def test_sin_huella_publicada_la_identidad_es_solo_declarada() -> None:
    """Contra un Broker anterior al Cambio 2 se degrada y se DECLARA."""
    broker = FakeBroker()
    broker.add_model(VIEJO.key, fingerprint={})
    broker.respond_fixed(VIEJO.key, "hola")
    with broker.client() as cliente:
        huella = cliente.fingerprint_for(VIEJO)
        assert huella.get("model") == "qwen3:6b"
        assert huella.get("digest") is None
        assert {"digest", "chat_template", "tokenizer"} <= huella.missing_names()
        assert any("identidad técnica" in aviso for aviso in cliente.missing_guarantees)


def test_el_catalogo_distingue_no_soportado_de_no_comprobado() -> None:
    """Triestado deliberado: sin él, un caso de herramientas suspendería."""
    broker = FakeBroker()
    broker.add_model(
        "ollama/local/con-tools",
        features={"tools": {"function_calling": "supported"}},
    )
    broker.add_model(
        "ollama/local/sin-tools",
        features={"tools": {"function_calling": "unsupported"}},
    )
    broker.add_model("ollama/local/sin-comprobar", features={"tools": {"function_calling": "unknown"}})
    with broker.client() as cliente:
        # Las features se piden por modelo: `/models/context` no es un listado.
        soporta = {
            clave: cliente.entry_for(ModelReference(provider="ollama", deployment="local", model=clave))
            for clave in ("con-tools", "sin-tools", "sin-comprobar")
        }
    assert soporta["con-tools"].supports("function_calling") is True
    assert soporta["sin-tools"].supports("function_calling") is False
    assert soporta["sin-comprobar"].supports("function_calling") is None


def test_la_sonda_de_contrato_averigua_lo_que_admite_el_broker() -> None:
    """El aviso de «esto va a contaminar» tiene que basarse en un hecho.

    Las banderas nacen optimistas y se corrigen al primer 422, que llega tarde:
    cuando hay que avisar todavía no se ha enviado nada. La sonda lo resuelve
    antes de empezar.
    """
    moderno = _broker()
    with moderno.client() as cliente:
        assert cliente.probe_contract() == {"seed": True, "learning_exclusion": True}
        assert cliente.missing_guarantees == []

    antiguo = _broker()
    antiguo.supports_seed = False
    antiguo.supports_learning_exclusion = False
    with antiguo.client() as cliente:
        contrato = cliente.probe_contract()
        assert contrato == {"seed": False, "learning_exclusion": False}
        assert any("NO es reproducible" in aviso for aviso in cliente.missing_guarantees)
        assert any("altera las estadísticas" in aviso for aviso in cliente.missing_guarantees)


# --------------------------------------------------------------------------
# Contrato 2.9: telemetría, execution_summary y seed_status
# --------------------------------------------------------------------------


def test_execution_summary_manda_sobre_result() -> None:
    """`result` no tiene contrato; `execution_summary` sí (§8)."""
    broker = _broker()
    with broker.client() as cliente:
        cliente.probe_contract()
        resultado = cliente.invoke(prompt="hola", target=VIEJO)
    assert resultado.ok is True
    assert resultado.model_served is not None
    assert resultado.model_served.key == VIEJO.key


def test_la_telemetria_trae_los_parametros_EFECTIVOS() -> None:
    """Lo pedido y lo aplicado no tienen por qué coincidir."""
    broker = _broker()
    with broker.client() as cliente:
        cliente.probe_contract()
        resultado = cliente.invoke(prompt="hola", target=VIEJO, params={"seed": 99, "temperature": 0.0})
    assert cliente.supports_invocation_telemetry is True
    assert resultado.applied_params.get("seed") == 99
    assert "seed_status" in resultado.applied_params
    # La huella que se guarda es la de la invocación, no la del catálogo.
    assert resultado.fingerprint.get("digest") is not None


def test_seed_status_sent_es_lo_unico_que_permite_hablar_de_control() -> None:
    from model_drift.broker_contracts import SeedStatus

    broker = _broker()
    with broker.client() as cliente:
        cliente.probe_contract()
        resultado = cliente.invoke(prompt="hola", target=VIEJO, params={"seed": 7})
    assert resultado.seed_status is SeedStatus.sent


def test_un_endpoint_que_no_admite_semilla_se_declara() -> None:
    """`unsupported`: se pidió y no cambió nada. Peor que no pedirla."""
    from model_drift.broker_contracts import SeedStatus

    broker = _broker()
    broker.seed_status = "unsupported"
    with broker.client() as cliente:
        cliente.probe_contract()
        resultado = cliente.invoke(prompt="hola", target=VIEJO, params={"seed": 7})
    assert resultado.seed_status is SeedStatus.unsupported
    assert any("no admite semilla" in aviso for aviso in cliente.missing_guarantees)


def test_sin_telemetria_se_declara_la_perdida() -> None:
    broker = _broker()
    broker.supports_invocation_telemetry = False
    with broker.client() as cliente:
        cliente.probe_contract()
        assert cliente.supports_invocation_telemetry is False
        resultado = cliente.invoke(prompt="hola", target=VIEJO, params={"seed": 7})
    # Se cae a la huella del catálogo, que sigue estando.
    assert resultado.fingerprint.get("model") == "qwen3:6b"


def test_el_trafico_va_marcado_como_excluido_del_aprendizaje() -> None:
    """Medir un modelo no debe bajarle la nota en producción (§5.8)."""
    broker = _broker()
    with broker.client() as cliente:
        cliente.probe_contract()
        cliente.invoke(prompt="hola", target=VIEJO)
    assert broker.requests[0]["exclude_from_model_learning"] is True


def test_la_sonda_degrada_si_el_broker_no_publica_capacidades() -> None:
    """Sin capacidades no se supone nada, pero tampoco se rompe la tirada."""
    broker = FakeBroker()
    broker.add_model(VIEJO.key)

    def sin_capabilities(request):  # type: ignore[no-untyped-def]
        if request.url.path == "/api/v1/capabilities":
            return httpx.Response(500, json={"detail": {"code": "BOOM", "message": "roto"}})
        return broker.handler(request)

    http = httpx.Client(transport=httpx.MockTransport(sin_capabilities), base_url="http://broker.test")
    with BrokerClient(BrokerConfig(base_url="http://broker.test", poll_interval_seconds=0.0), client=http) as c:
        contrato = c.probe_contract()
        assert set(contrato) == {"seed", "learning_exclusion"}
        assert any("no se pudieron leer las capacidades" in aviso for aviso in c.missing_guarantees)


def test_una_telemetria_ilegible_no_tumba_la_invocacion() -> None:
    """El Broker puede añadir campos o devolver una entrada rara: se ignora."""
    broker = FakeBroker()
    broker.add_model(VIEJO.key)
    broker.respond_fixed(VIEJO.key, "hola")

    def telemetria_rara(request):  # type: ignore[no-untyped-def]
        if request.url.path.endswith("/invocations"):
            return httpx.Response(200, json={"task_id": "x", "items": ["esto no es un objeto", {"role": 7}]})
        return broker.handler(request)

    http = httpx.Client(transport=httpx.MockTransport(telemetria_rara), base_url="http://broker.test")
    with BrokerClient(BrokerConfig(base_url="http://broker.test", poll_interval_seconds=0.0), client=http) as c:
        c.probe_contract()
        resultado = c.invoke(prompt="hola", target=VIEJO)
    assert resultado.ok is True
    # Se cae a la huella del catálogo en vez de quedarse sin ninguna.
    assert resultado.fingerprint.get("model") == "qwen3:6b"


def test_un_catalogo_sin_lista_de_modelos_falla_con_un_motivo_util() -> None:
    """Fallar diciendo qué llegó es mejor que devolver una lista vacía."""

    def catalogo_raro(request):  # type: ignore[no-untyped-def]
        if request.url.path == "/api/v1/models/availability":
            return httpx.Response(200, json={"checked_at": "hoy", "counts": {}})
        return httpx.Response(200, json={})

    http = httpx.Client(transport=httpx.MockTransport(catalogo_raro), base_url="http://broker.test")
    with BrokerClient(BrokerConfig(base_url="http://broker.test"), client=http) as c:
        try:
            c.catalog()
        except BrokerError as error:
            assert "ninguna lista de modelos" in str(error)
            assert "broker_contracts.py" in str(error)
        else:  # pragma: no cover
            raise AssertionError("un catálogo sin lista debe fallar, no devolver vacío")


def test_las_entradas_de_catalogo_ilegibles_se_cuentan_no_se_tragan() -> None:
    def catalogo_mixto(request):  # type: ignore[no-untyped-def]
        if request.url.path == "/api/v1/models/availability":
            return httpx.Response(
                200,
                json={
                    "items": [
                        {"provider": "ollama", "deployment": "local", "model": "bueno"},
                        {"esto": "no encaja"},
                        "ni esto",
                    ]
                },
            )
        return httpx.Response(200, json={})

    http = httpx.Client(transport=httpx.MockTransport(catalogo_mixto), base_url="http://broker.test")
    with BrokerClient(BrokerConfig(base_url="http://broker.test"), client=http) as c:
        entradas = c.catalog()
        assert [item.model for item in entradas] == ["bueno"]
        assert any("no encajan en el contrato conocido" in aviso for aviso in c.missing_guarantees)
