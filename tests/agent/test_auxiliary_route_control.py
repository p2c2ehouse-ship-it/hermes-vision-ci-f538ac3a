import asyncio
from types import SimpleNamespace

import hermes_yaml as yaml
import sys
sys.modules.setdefault("hermes_yaml", yaml)
import types

_constants = types.ModuleType("hermes_constants")
_constants.OPENROUTER_BASE_URL = "https://openrouter.invalid/api/v1"
_constants.OPENROUTER_MODELS_URL = "https://openrouter.invalid/api/v1/models"
_constants.openrouter_variant_base = lambda value: value
_constants.get_hermes_dir = lambda *args, **kwargs: __import__("pathlib").Path(".")
_constants.get_hermes_home = lambda: __import__("pathlib").Path(".")
_constants.hermes_home_key = lambda: "test-home"
sys.modules.setdefault("hermes_constants", _constants)
_utils = types.ModuleType("utils")
_utils.base_url_host_matches = lambda *args, **kwargs: False
_utils.base_url_hostname = lambda value: ""
_utils.base_url_origin = lambda value: ""
_utils.env_float = lambda name, default: default
_utils.is_truthy_value = lambda value: bool(value)
_utils.model_forces_max_completion_tokens = lambda model: False
_utils.normalize_proxy_env_vars = lambda: None
_utils.__getattr__ = lambda name: lambda *args, **kwargs: None
sys.modules.setdefault("utils", _utils)
import pytest
_config = types.ModuleType("hermes_cli.config")
_config.get_hermes_home = lambda: __import__("pathlib").Path(".")
_config.__getattr__ = lambda name: lambda *args, **kwargs: None
sys.modules.setdefault("hermes_cli.config", _config)
_config_providers = types.ModuleType("hermes_cli.config_providers")
_config_providers._canonical_api_mode = lambda value: value
sys.modules.setdefault("hermes_cli.config_providers", _config_providers)

def _stub_module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    module.__getattr__ = lambda key: lambda *args, **kwargs: None
    sys.modules.setdefault(name, module)

_stub_module("agent.credential_pool", load_pool=lambda *args, **kwargs: None)
_stub_module("agent.custom_entry_pool")
_stub_module("agent.message_metadata", data_use_requirement=lambda *args, **kwargs: None)
_stub_module("agent.model_metadata", MINIMUM_CONTEXT_LENGTH=0, get_model_context_length=lambda *args: 0)
_stub_module("agent.auxiliary_health")
_stub_module("agent.auxiliary_unavailable", AuxiliaryClientUnavailable=RuntimeError)
_stub_module("hermes_cli.version_info", get_version_info=lambda: SimpleNamespace(base_version="test"))
_stub_module("agent.portal_tags", nous_portal_tags=lambda *args, **kwargs: {})
_stub_module("agent.auxiliary_task_config", _get_auxiliary_task_config=lambda *args, **kwargs: {})
from agent import auxiliary_client as aux

# The auxiliary import needs only two metadata functions. Do not leave its
# lightweight module stub installed for unrelated estimator tests.
import agent as _agent_package
for _module_name in ("model_metadata", "message_metadata"):
    _qualified_name = f"agent.{_module_name}"
    _stub = sys.modules.pop(_qualified_name, None)
    if getattr(_agent_package, _module_name, None) is _stub:
        delattr(_agent_package, _module_name)


def prepared_request():
    return SimpleNamespace(
        client=object(), kwargs={"model": "explicit-model"}, request_provider="openrouter",
        resolved_api_mode="chat_completions", base_info="https://example.invalid",
        resolved_base_url="https://example.invalid",
    )


def test_async_route_control_dispatches_once_without_legacy_ladder(monkeypatch):
    calls = []
    monkeypatch.setattr(aux, "_plan_aux_call", lambda *args, **kwargs: (prepared_request(), {}, {}))
    monkeypatch.setattr(aux, "_provider_requires_stream", lambda *args: False)

    async def relay(*args, **kwargs):
        calls.append(kwargs)
        return "raw-response"

    def forbidden_ladder(*args, **kwargs):
        raise AssertionError("legacy recovery ladder must not start")

    monkeypatch.setattr(aux, "_relay_async_completion", relay)
    monkeypatch.setattr(aux, "_validate_llm_response", lambda response, *args, **kwargs: response)
    monkeypatch.setattr(aux, "_start_recovery_ladder", forbidden_ladder)

    result = asyncio.run(aux._async_call_llm_impl(
        provider="openrouter", model="explicit-model", messages=[], route_control=object(),
    ))

    assert result == "raw-response"
    assert len(calls) == 1


def test_async_route_control_returns_provider_error_to_router(monkeypatch):
    monkeypatch.setattr(aux, "_plan_aux_call", lambda *args, **kwargs: (prepared_request(), {}, {}))
    monkeypatch.setattr(aux, "_provider_requires_stream", lambda *args: False)

    async def relay(*args, **kwargs):
        raise RuntimeError("provider failed")

    def forbidden_ladder(*args, **kwargs):
        raise AssertionError("legacy recovery ladder must not start")

    monkeypatch.setattr(aux, "_relay_async_completion", relay)
    monkeypatch.setattr(aux, "_start_recovery_ladder", forbidden_ladder)

    with pytest.raises(RuntimeError, match="provider failed"):
        asyncio.run(aux._async_call_llm_impl(
            provider="openrouter", model="explicit-model", messages=[], route_control=object(),
        ))


def test_sync_route_control_dispatches_once_without_legacy_ladder(monkeypatch):
    calls = []
    monkeypatch.setattr(aux, "_plan_aux_call", lambda *args, **kwargs: (prepared_request(), {}, {}))
    monkeypatch.setattr(aux, "_provider_requires_stream", lambda *args: False)

    def relay(*args, **kwargs):
        calls.append(kwargs)
        return "raw-response"

    def forbidden_ladder(*args, **kwargs):
        raise AssertionError("legacy recovery ladder must not start")

    monkeypatch.setattr(aux, "_relay_sync_completion", relay)
    monkeypatch.setattr(aux, "_validate_llm_response", lambda response, *args, **kwargs: response)
    monkeypatch.setattr(aux, "_start_recovery_ladder", forbidden_ladder)

    result = aux._call_llm_impl(
        provider="openrouter", model="explicit-model", messages=[], route_control=object(),
    )

    assert result == "raw-response"
    assert len(calls) == 1


def test_explicit_selector_route_cannot_fall_back_to_auto(monkeypatch):
    def forbidden_plan(*args, **kwargs):
        raise AssertionError("auto route must be rejected before resolution")

    monkeypatch.setattr(aux, "_plan_aux_call", forbidden_plan)

    with pytest.raises(ValueError, match="explicit provider and model"):
        asyncio.run(aux._async_call_llm_impl(
            provider="auto", model="explicit-model", messages=[], route_control=object(),
        ))
