import asyncio
import json
from pathlib import Path
import sys
import types
from types import SimpleNamespace

_dir_acl = types.ModuleType("hermes_dir_acl")
_dir_acl.make_user_dir = lambda path, *args, **kwargs: Path(path)
sys.modules.setdefault("hermes_dir_acl", _dir_acl)
_constants = types.ModuleType("hermes_constants")
_constants.OPENROUTER_BASE_URL = "https://openrouter.invalid/api/v1"
_constants.OPENROUTER_MODELS_URL = "https://openrouter.invalid/api/v1/models"
_constants.openrouter_variant_base = lambda value: value
_constants.get_hermes_dir = lambda *args, **kwargs: Path(".")
_constants.get_hermes_home = lambda: Path(".")
_constants.hermes_home_key = lambda: "test-home"
sys.modules.setdefault("hermes_constants", _constants)
_website_policy = types.ModuleType("tools.website_policy")
_website_policy.check_website_access = lambda url: None
_interrupt = types.ModuleType("tools.interrupt")
_interrupt.is_interrupted = lambda: False
sys.modules.setdefault("tools.interrupt", _interrupt)
sys.modules.setdefault("tools.website_policy", _website_policy)
_registry_module = types.ModuleType("tools.registry")
_registry_module.registry = SimpleNamespace(register=lambda **kwargs: None)
_registry_module.tool_error = lambda message, **kwargs: json.dumps({"success": False, "error": message, **kwargs})
sys.modules.setdefault("tools.registry", _registry_module)

from tools import vision_tools as tools
from agent import vision_integration as integration
from agent.vision_routing import SQLiteVisionStore, VisionRouter


def config(mode="active", verified=("image",)):
    return {
        "provider": "auto", "model": "", "timeout": 120,
        "routing": {
            "mode": mode, "output_cap": 512, "hard_output_cap": 1024,
            "allow": {"hosted_free": True, "local": False, "paid": False},
            "candidates": {
                "free": {
                    "provider": "openrouter", "model": "example/free:free",
                    "declared_media": ["image", "video"], "verified_media": list(verified),
                    "cost_class": "hosted_free",
                },
            },
        },
    }


def install_config(monkeypatch, vision_config):
    def cfg(*keys, default=None):
        if keys == ("vision",):
            return vision_config
        if len(keys) == 1 and keys[0] == "video":
            return {}
        if len(keys) == 2 and keys[0] == "vision":
            return vision_config.get(keys[1], default)
        return default

    monkeypatch.setattr(tools, "_cfg_auxiliary", cfg)
    monkeypatch.setattr(tools, "get_hermes_dir", lambda *args, **kwargs: tools._test_state_dir)


def install_prepared(monkeypatch, tmp_path):
    path = tmp_path / "prepared.png"
    path.write_bytes(b"prepared")
    prepared = SimpleNamespace(path=path, mime="image/png", size_bytes=8, crop_offset=None)
    monkeypatch.setattr(tools, "_prepare_image", AsyncMock.return_value(prepared))
    monkeypatch.setattr(tools, "_image_to_base64_data_url", lambda *args, **kwargs: "data:image/png;base64,cHJlcGFyZWQ=")


class AsyncMock:
    def __init__(self, return_value=None, side_effect=None):
        self.return_value = return_value
        self.side_effect = side_effect
        self.calls = []

    @classmethod
    def return_value(cls, value):
        return cls(return_value=value)

    async def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.side_effect:
            raise self.side_effect
        return self.return_value


def test_native_image_fast_path_does_not_call_auxiliary_selector(monkeypatch):
    async def native(*args, **kwargs):
        return "native-result"

    async def forbidden(*args, **kwargs):
        raise AssertionError("auxiliary selector must not run")

    monkeypatch.setattr(tools, "_should_use_native_vision_fast_path", lambda: True)
    monkeypatch.setattr(tools, "_vision_analyze_native", native)
    monkeypatch.setattr(tools, "vision_analyze_tool", forbidden)

    result = asyncio.run(tools._handle_vision_analyze({"image_url": "image", "question": "q"}))

    assert result == "native-result"


def test_active_image_tool_returns_analysis_and_route_metadata(monkeypatch, tmp_path):
    tools._test_state_dir = tmp_path
    install_config(monkeypatch, config())
    install_prepared(monkeypatch, tmp_path)
    calls = []

    async def provider_call(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "active analysis"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(tools, "async_call_llm", provider_call)
    monkeypatch.setattr(tools, "extract_content_or_reasoning", lambda response: "unused")

    raw = asyncio.run(tools._handle_vision_analyze(
        {"image_url": "image", "question": "q"},
        profile="test-profile", session_id="test-session", framework_call_id="call-1",
    ))
    result = json.loads(raw)

    assert result["success"] is True
    assert result["analysis"] == "active analysis"
    assert result["candidate_id"] == "free"
    assert calls[0]["route_control"].attempt_index == 1
    assert calls[0]["provider"] == "openrouter"


def test_active_tool_failure_preserves_structured_choices(monkeypatch, tmp_path):
    tools._test_state_dir = tmp_path
    install_config(monkeypatch, config())
    install_prepared(monkeypatch, tmp_path)

    class Rejected(Exception):
        status_code = 400

    async def provider_call(**kwargs):
        raise Rejected("image input is not supported")

    monkeypatch.setattr(tools, "async_call_llm", provider_call)
    monkeypatch.setattr(tools, "extract_content_or_reasoning", lambda response: "unused")

    result = json.loads(asyncio.run(tools._handle_vision_analyze(
        {"image_url": "image", "question": "q"}, profile="test", session_id="session",
    )))

    assert result["success"] is False
    assert result["state"] == "terminal_failed"
    assert result["automatic_retry_allowed"] is False
    assert [choice["choice"] for choice in result["choices"]] == ["paid", "free", "retry"]


def test_shadow_preserves_legacy_empty_response_retry_sequence(monkeypatch, tmp_path):
    tools._test_state_dir = tmp_path
    install_config(monkeypatch, config(mode="shadow"))
    install_prepared(monkeypatch, tmp_path)
    calls = []

    async def provider_call(**kwargs):
        calls.append(kwargs)
        content = "" if len(calls) == 1 else "legacy recovered"
        return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}

    def extract(response):
        return response["choices"][0]["message"]["content"]

    monkeypatch.setattr(tools, "async_call_llm", provider_call)
    monkeypatch.setattr(tools, "extract_content_or_reasoning", extract)

    result = json.loads(asyncio.run(tools._handle_vision_analyze(
        {"image_url": "image", "question": "q"}, profile="test", session_id="session",
    )))

    assert result == {"success": True, "analysis": "legacy recovered"}
    assert len(calls) == 2
    assert all("route_control" not in call for call in calls)
    events = SQLiteVisionStore(tmp_path / "vision-routing" / "routing.db").events()
    assert [event["event_type"] for event in events] == ["vision_shadow_decision", "vision_legacy_summary"]
    legacy = events[-1]
    assert legacy["success"] is True
    assert legacy["analysis_nonempty"] is True
    assert legacy["candidate_id"] is None
    assert legacy["actual_cost_usd"] is None
    baseline = VisionRouter([], db_path=tmp_path / "vision-routing" / "routing.db").metrics()["legacy"]
    assert baseline["logical_requests"] == 1
    assert baseline["logical_request_success_rate"] == 1
    assert baseline["p50_total_latency_ms"] >= 0


def test_active_video_gate_rejects_image_only_verified_route(monkeypatch):
    install_config(monkeypatch, config(verified=("image",)))

    assert tools.check_video_requirements() is False



def test_active_video_tool_uses_verified_video_route(monkeypatch, tmp_path):
    tools._test_state_dir = tmp_path
    install_config(monkeypatch, config(verified=("image", "video")))
    path = tmp_path / "prepared.mp4"
    path.write_bytes(b"video")
    monkeypatch.setattr(tools, "_materialize_video", AsyncMock.return_value(path))
    monkeypatch.setattr(tools, "_video_to_base64_data_url", lambda *args, **kwargs: "data:video/mp4;base64,dmlkZW8=")
    calls = []

    async def provider_call(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "video analysis"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(tools, "async_call_llm", provider_call)
    monkeypatch.setattr(tools, "extract_content_or_reasoning", lambda response: "unused")

    result = json.loads(asyncio.run(tools._handle_video_analyze(
        {"video_url": "video", "question": "q"}, profile="test", session_id="session",
    )))

    assert result["success"] is True
    assert result["analysis"] == "video analysis"
    assert calls[0]["route_control"].attempt_index == 1



def test_active_image_size_retry_uses_real_resize_preparation(monkeypatch, tmp_path):
    tools._test_state_dir = tmp_path
    install_config(monkeypatch, config())
    install_prepared(monkeypatch, tmp_path)
    monkeypatch.setattr(
        tools, "_resize_prepared",
        AsyncMock.return_value("data:image/jpeg;base64,cmVzaXplZA=="),
    )
    calls = []

    class TooLarge(Exception):
        status_code = 400

    async def provider_call(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise TooLarge("image size too large")
        return {"choices": [{"message": {"content": "resized analysis"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(tools, "async_call_llm", provider_call)
    monkeypatch.setattr(tools, "extract_content_or_reasoning", lambda response: "unused")

    result = json.loads(asyncio.run(tools._handle_vision_analyze(
        {"image_url": "image", "question": "q"}, profile="test", session_id="session",
    )))

    assert result["success"] is True
    assert result["analysis"].endswith("resized analysis")
    assert len(calls) == 2
    assert calls[1]["messages"][0]["content"][1]["image_url"]["url"].endswith("cmVzaXplZA==")



def test_active_image_gate_requires_image_verified_route(monkeypatch):
    install_config(monkeypatch, config(verified=("video",)))
    current = tools._routing_vision_config("image")
    current["routing"]["candidates"]["free"]["declared_media"] = ["video"]
    monkeypatch.setattr(tools, "_routing_vision_config", lambda kind: current)
    monkeypatch.setattr(tools, "_should_use_native_vision_fast_path", lambda: False)

    assert tools.check_vision_requirements() is False



def test_prepared_image_supplies_planning_token_estimate_for_paid_offer(monkeypatch, tmp_path):
    tools._test_state_dir = tmp_path
    paid = config()
    candidate = paid["routing"]["candidates"]["free"]
    candidate["cost_class"] = "paid"
    candidate["pricing"] = {
        "captured_at": integration.current_price_timestamp(), "source": "test-price-snapshot",
        "input_usd_per_token": "0.000001", "output_usd_per_token": "0.000002",
        "max_age_seconds": 99999999999,
    }
    paid["routing"]["allow"]["paid"] = True
    paid["routing"]["budgets"] = {
        "paid_per_request_usd": "2", "paid_daily_usd": "10", "paid_monthly_usd": "20",
    }
    install_config(monkeypatch, paid)
    install_prepared(monkeypatch, tmp_path)

    async def forbidden(**kwargs):
        raise AssertionError("paid route must wait for approval")

    monkeypatch.setattr(tools, "async_call_llm", forbidden)
    monkeypatch.setattr(tools, "extract_content_or_reasoning", lambda response: "unused")
    result = json.loads(asyncio.run(tools._handle_vision_analyze(
        {"image_url": "image", "question": "q"}, profile="test", session_id="session",
    )))

    offer = result["choices"][0]
    assert offer["viable"] is True, (offer, result.get("error_class"))
    assert offer["estimate_method"] == "planning_estimate"
    assert float(offer["estimated_cost_usd"]) > 0
    assert offer["input_token_estimate_source"] == "agent.model_metadata.estimate_messages_tokens_rough"



def test_handler_reuses_framework_tool_call_id_for_idempotency(monkeypatch, tmp_path):
    import contextvars
    tools._test_state_dir = tmp_path
    install_config(monkeypatch, config())
    install_prepared(monkeypatch, tmp_path)
    call_id = contextvars.ContextVar("test_vision_tool_call_id", default="")
    approval_context = types.ModuleType("tools.approval_context")
    approval_context._approval_tool_call_id = call_id
    monkeypatch.setitem(sys.modules, "tools.approval_context", approval_context)
    call_id.set("framework-call-42")
    calls = []

    async def provider_call(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "active analysis"}, "finish_reason": "stop"}]}

    monkeypatch.setattr(tools, "async_call_llm", provider_call)
    monkeypatch.setattr(tools, "extract_content_or_reasoning", lambda response: "unused")
    args = {"image_url": "image", "question": "q"}
    first = json.loads(asyncio.run(tools._handle_vision_analyze(
        args, profile="test", session_id="session",
    )))
    repeated = json.loads(asyncio.run(tools._handle_vision_analyze(
        args, profile="test", session_id="session",
    )))

    assert first["success"] is True
    assert repeated["success"] is True
    assert calls[0]["route_control"].request_id == "framework-call-42"
    assert len(calls) == 1
