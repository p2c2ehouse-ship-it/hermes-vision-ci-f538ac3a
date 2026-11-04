import hashlib
import time
import asyncio
import json
from pathlib import Path

from agent import vision_integration as integration
from agent.vision_routing import MediaRequest


def vision_config(mode="active", media=("image",), verified=None):
    verified = tuple(media if verified is None else verified)
    return {
        "provider": "auto", "model": "",
        "routing": {
            "mode": mode,
            "strategy": "free_first",
            "allow": {"hosted_free": True, "local": False, "paid": False},
            "candidates": {
                "free_image": {
                    "provider": "openrouter", "model": "example/free:free",
                    "base_url": "https://openrouter.invalid/api/v1",
                    "declared_media": list(media), "verified_media": list(verified),
                    "cost_class": "hosted_free", "configured_order": 0,
                },
            },
        },
    }


def media_request(kind="image", **changes):
    values = dict(
        request_id="req-integration", profile="test-profile", session_id="test-session",
        media_kind=kind, media_hash="a" * 64, normalized_prompt_hash="b" * 64,
        mime_type="image/png" if kind == "image" else "video/mp4",
        byte_count=1234, base64_bytes=1646, output_cap=512, hard_output_cap=1024,
        source_kind="local", size_bucket="small",
    )
    values.update(changes)
    return MediaRequest(**values)


def messages(kind="image"):
    part = "image_url" if kind == "image" else "video_url"
    return [{"role": "user", "content": [
        {"type": "text", "text": "secret prompt"},
        {"type": part, part: {"url": "data:image/png;base64,c2VjcmV0LW1lZGlh"}},
    ]}]


def test_active_dispatches_explicit_candidate_with_route_control(tmp_path):
    calls = []

    async def provider_call(**kwargs):
        calls.append(kwargs)
        return {
            "choices": [{"message": {"content": "analysis"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        }

    result = asyncio.run(integration.analyze_prepared(
        media_request(), messages(), {"task": "vision", "temperature": 0.1},
        vision_config(), db_path=tmp_path / "routing.db", retention_dir=tmp_path / "retained",
        provider_call=provider_call,
    ))

    assert result.success is True
    assert result.analysis == "analysis"
    assert len(calls) == 1
    assert calls[0]["provider"] == "openrouter"
    assert calls[0]["model"] == "example/free:free"
    assert calls[0]["base_url"] == "https://openrouter.invalid/api/v1"
    assert calls[0]["max_tokens"] == 512
    assert calls[0]["route_control"].attempt_index == 1
    assert calls[0]["timeout"] <= 60


def test_active_transient_failure_retries_only_through_shared_router(tmp_path):
    calls = []

    async def provider_call(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise TimeoutError("socket timed out")
        return {"choices": [{"message": {"content": "recovered"}, "finish_reason": "stop"}]}

    result = asyncio.run(integration.analyze_prepared(
        media_request(), messages(), {"task": "vision"}, vision_config(),
        db_path=tmp_path / "routing.db", retention_dir=tmp_path / "retained",
        provider_call=provider_call,
    ))

    assert result.success is True
    assert len(calls) == 2
    assert [call["route_control"].attempt_index for call in calls] == [1, 2]
    assert all(call["provider"] == "openrouter" for call in calls)


def test_active_permanent_failure_is_structured_and_not_retried(tmp_path):
    calls = []

    class Rejected(Exception):
        status_code = 400

    async def provider_call(**kwargs):
        calls.append(kwargs)
        raise Rejected("secret upstream body: video is not supported")

    result = asyncio.run(integration.analyze_prepared(
        media_request("video", mime_type="video/mp4"), messages("video"), {"task": "vision"},
        vision_config(media=("image", "video")), db_path=tmp_path / "routing.db",
        retention_dir=tmp_path / "retained", provider_call=provider_call,
    ))

    assert result.success is False
    assert result.state == "terminal_failed"
    assert result.error_class == "capability"
    assert len(calls) == 1
    assert "secret upstream" not in result.analysis
    assert [choice["choice"] for choice in result.choices] == ["paid", "free", "retry"]
    assert result.automatic_retry_allowed is False


def test_shadow_decision_dispatches_nothing(tmp_path):
    selection = integration.shadow_prepared(
        media_request(), vision_config(mode="shadow"), db_path=tmp_path / "routing.db",
    )

    assert selection.selected.candidate_id == "free_image"
    events = integration.SQLiteVisionStore(tmp_path / "routing.db").events()
    assert [event["event_type"] for event in events] == ["vision_shadow_decision"]
    assert events[0]["paid_call_dispatched"] is False


def test_active_video_gate_requires_verified_video_not_image_inheritance():
    assert integration.video_route_available(vision_config(verified=("image",))) is False
    assert integration.video_route_available(vision_config(media=("image", "video"))) is True


def test_active_telemetry_does_not_persist_prompt_media_url_or_path(tmp_path):
    async def provider_call(**kwargs):
        return {"choices": [{"message": {"content": "analysis"}, "finish_reason": "stop"}]}

    db_path = tmp_path / "routing.db"
    result = asyncio.run(integration.analyze_prepared(
        media_request(), messages(), {"task": "vision"}, vision_config(),
        db_path=db_path, retention_dir=tmp_path / "retained", provider_call=provider_call,
    ))

    assert result.success is True
    persisted = db_path.read_bytes()
    assert b"secret prompt" not in persisted
    assert b"c2VjcmV0LW1lZGlh" not in persisted
    assert b"openrouter.invalid" not in persisted
    assert b"retained" not in persisted
    assert list((tmp_path / "retained").glob("*")) == []


def test_paid_only_active_route_returns_approval_choice_without_dispatch(tmp_path):
    config = vision_config()
    candidate = config["routing"]["candidates"]["free_image"]
    candidate["cost_class"] = "paid"
    config["routing"]["allow"]["paid"] = True
    config["routing"]["budgets"] = {
        "paid_per_request_usd": "2", "paid_daily_usd": "10", "paid_monthly_usd": "20",
    }
    candidate["pricing"] = {
        "captured_at": integration.current_price_timestamp(), "source": "test-snapshot",
        "input_usd_per_token": "0", "output_usd_per_token": "0.001", "request_usd": "0.01",
    }
    calls = []

    async def provider_call(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "must not run"}, "finish_reason": "stop"}]}

    req = media_request(estimated_input_tokens=0, input_token_count_exact=True)
    result = asyncio.run(integration.analyze_prepared(
        req, messages(), {"task": "vision"}, config, db_path=tmp_path / "routing.db",
        retention_dir=tmp_path / "retained", provider_call=provider_call,
    ))

    assert result.success is False
    assert result.state == "recovery_required"
    assert result.choices[0]["viable"] is True
    assert result.choices[0]["approval_required"] is True
    assert calls == []



def test_image_size_retry_uses_corrected_media_not_the_original_payload(tmp_path):
    calls = []

    class TooLarge(Exception):
        status_code = 400

    async def provider_call(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise TooLarge("image size too large")
        return {"choices": [{"message": {"content": "downscaled analysis"}, "finish_reason": "stop"}]}

    async def media_correction(current_messages, control):
        assert control.correction == "downscale"
        corrected = json.loads(json.dumps(current_messages))
        corrected[0]["content"][1]["image_url"]["url"] = "data:image/jpeg;base64,ZG93bnNjYWxlZA=="
        return corrected

    result = asyncio.run(integration.analyze_prepared(
        media_request(), messages(), {"task": "vision"}, vision_config(),
        db_path=tmp_path / "routing.db", retention_dir=tmp_path / "retained",
        provider_call=provider_call, media_correction=media_correction,
    ))

    assert result.success is True
    assert len(calls) == 2
    assert calls[0]["messages"][0]["content"][1]["image_url"]["url"].endswith("c2VjcmV0LW1lZGlh")
    assert calls[1]["messages"][0]["content"][1]["image_url"]["url"].endswith("ZG93bnNjYWxlZA==")



def test_free_recovery_reuses_retained_media_and_deletes_it_after_success(tmp_path):
    config = vision_config()
    config["routing"]["candidates"]["free_fallback"] = {
        "provider": "openrouter", "model": "example/fallback:free",
        "declared_media": ["image"], "verified_media": ["image"],
        "cost_class": "hosted_free", "configured_order": 1,
    }
    data_url = messages()[0]["content"][1]["image_url"]["url"]
    req = media_request(media_hash=hashlib.sha256(data_url.encode()).hexdigest())
    db_path = tmp_path / "routing.db"
    retention_dir = tmp_path / "retained"
    calls = []

    async def provider_call(**kwargs):
        calls.append(kwargs)
        if kwargs["model"] == "example/free:free":
            raise integration.ProviderError("rate_limit", retry_after_seconds=0)
        return {"choices": [{"message": {"content": "fallback analysis"}, "finish_reason": "stop"}]}

    offered = asyncio.run(integration.analyze_prepared(
        req, messages(), {"task": "vision"}, config, db_path=db_path,
        retention_dir=retention_dir, provider_call=provider_call,
    ))

    assert offered.state == "recovery_required"
    retained_files = list(retention_dir.glob("*.media"))
    assert len(retained_files) == 1
    assert retained_files[0].stat().st_mode & 0o077 == 0

    recovered = asyncio.run(integration.recover_prepared(
        offered.recovery_id, "free", req, messages(), {"task": "vision"}, config,
        db_path=db_path, retention_dir=retention_dir, provider_call=provider_call,
    ))

    assert recovered.success is True
    assert recovered.analysis == "fallback analysis"
    assert calls[-1]["model"] == "example/fallback:free"
    assert calls[-1]["route_control"].fallback_index == 1
    assert calls[-1]["messages"][0]["content"][1]["image_url"]["url"] == data_url
    assert list(retention_dir.glob("*.media")) == []



def test_expired_recovery_deletes_media_and_grant_and_never_dispatches(tmp_path):
    config = vision_config()
    candidate = config["routing"]["candidates"]["free_image"]
    candidate["cost_class"] = "paid"
    config["routing"]["allow"]["paid"] = True
    config["routing"]["budgets"] = {
        "paid_per_request_usd": "2", "paid_daily_usd": "10", "paid_monthly_usd": "20",
    }
    candidate["pricing"] = {
        "captured_at": integration.current_price_timestamp(), "source": "test-snapshot",
        "input_usd_per_token": "0", "output_usd_per_token": "0.001", "request_usd": "0.01",
    }
    data_url = messages()[0]["content"][1]["image_url"]["url"]
    req = media_request(
        media_hash=hashlib.sha256(data_url.encode()).hexdigest(),
        estimated_input_tokens=0, input_token_count_exact=True,
    )
    db_path = tmp_path / "routing.db"
    retention_dir = tmp_path / "retained"
    calls = []

    async def provider_call(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "must not run"}, "finish_reason": "stop"}]}

    offered = asyncio.run(integration.analyze_prepared(
        req, messages(), {"task": "vision"}, config, db_path=db_path,
        retention_dir=retention_dir, provider_call=provider_call,
    ))
    grant = integration.approve_paid(
        offered.recovery_id, candidate_id="free_image", request=req,
        vision_config=config, db_path=db_path,
    )
    assert len(list(retention_dir.glob("*.media"))) == 1

    integration.cleanup_expired_recoveries(
        db_path=db_path, retention_dir=retention_dir, now=time.time() + 901,
    )

    assert list(retention_dir.glob("*.media")) == []
    store = integration.SQLiteVisionStore(db_path)
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM approval_grants").fetchone()[0] == 0

    replay = asyncio.run(integration.recover_prepared(
        offered.recovery_id, "paid", req, messages(), {"task": "vision"}, config,
        db_path=db_path, retention_dir=retention_dir, provider_call=provider_call,
        approval_token=grant.token,
    ))

    assert replay.success is False
    assert replay.error_class == "recovery_expired"
    assert calls == []



def test_video_size_rejection_does_not_enter_image_downscale_recovery(tmp_path):
    calls = []

    class TooLarge(Exception):
        status_code = 400

    async def provider_call(**kwargs):
        calls.append(kwargs)
        raise TooLarge("payload size too large")

    result = asyncio.run(integration.analyze_prepared(
        media_request("video", mime_type="video/mp4"), messages("video"), {"task": "vision"},
        vision_config(media=("image", "video")), db_path=tmp_path / "routing.db",
        retention_dir=tmp_path / "retained", provider_call=provider_call,
    ))

    assert result.success is False
    assert result.error_class == "image_size"
    assert len(calls) == 1
    assert result.choices[2]["viable"] is False



def test_invalid_candidate_cost_class_is_not_registered():
    config = vision_config()
    config["routing"]["candidates"]["free_image"]["cost_class"] = "paidd"

    assert integration.build_candidates(config) == ()



def test_mismatched_recovery_request_cannot_delete_valid_retained_media(tmp_path):
    config = vision_config()
    config["routing"]["candidates"]["free_fallback"] = {
        "provider": "openrouter", "model": "example/fallback:free",
        "declared_media": ["image"], "verified_media": ["image"],
        "cost_class": "hosted_free", "configured_order": 1,
    }
    data_url = messages()[0]["content"][1]["image_url"]["url"]
    req = media_request(media_hash=hashlib.sha256(data_url.encode()).hexdigest())
    db_path = tmp_path / "routing.db"
    retention_dir = tmp_path / "retained"
    calls = []

    async def provider_call(**kwargs):
        calls.append(kwargs)
        if kwargs["model"] == "example/free:free":
            raise integration.ProviderError("rate_limit", retry_after_seconds=0)
        return {"choices": [{"message": {"content": "fallback analysis"}, "finish_reason": "stop"}]}

    offered = asyncio.run(integration.analyze_prepared(
        req, messages(), {"task": "vision"}, config, db_path=db_path,
        retention_dir=retention_dir, provider_call=provider_call,
    ))
    retained = list(retention_dir.glob("*.media"))
    assert len(retained) == 1
    calls_before_mismatch = len(calls)

    mismatch = asyncio.run(integration.recover_prepared(
        offered.recovery_id, "free", media_request(media_hash="f" * 64), messages(),
        {"task": "vision"}, config, db_path=db_path, retention_dir=retention_dir,
        provider_call=provider_call,
    ))

    assert mismatch.success is False
    assert mismatch.error_class == "recovery_fingerprint_mismatch"
    assert len(calls) == calls_before_mismatch
    assert retained[0].exists()

    recovered = asyncio.run(integration.recover_prepared(
        offered.recovery_id, "free", req, messages(), {"task": "vision"}, config,
        db_path=db_path, retention_dir=retention_dir, provider_call=provider_call,
    ))
    assert recovered.success is True
    assert list(retention_dir.glob("*.media")) == []


def test_invalid_paid_token_does_not_consume_grant_or_retained_media(tmp_path):
    config = vision_config()
    candidate = config["routing"]["candidates"]["free_image"]
    candidate["cost_class"] = "paid"
    config["routing"]["allow"]["paid"] = True
    config["routing"]["budgets"] = {
        "paid_per_request_usd": "2", "paid_daily_usd": "10", "paid_monthly_usd": "20",
    }
    candidate["pricing"] = {
        "captured_at": integration.current_price_timestamp(), "source": "test-snapshot",
        "input_usd_per_token": "0", "output_usd_per_token": "0.001", "request_usd": "0.01",
    }
    data_url = messages()[0]["content"][1]["image_url"]["url"]
    req = media_request(
        media_hash=hashlib.sha256(data_url.encode()).hexdigest(),
        estimated_input_tokens=0, input_token_count_exact=True,
    )
    db_path = tmp_path / "routing.db"
    retention_dir = tmp_path / "retained"
    calls = []

    async def provider_call(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "paid analysis"}, "finish_reason": "stop"}]}

    offered = asyncio.run(integration.analyze_prepared(
        req, messages(), {"task": "vision"}, config, db_path=db_path,
        retention_dir=retention_dir, provider_call=provider_call,
    ))
    grant = integration.approve_paid(
        offered.recovery_id, candidate_id="free_image", request=req,
        vision_config=config, db_path=db_path,
    )
    retained = list(retention_dir.glob("*.media"))
    assert len(retained) == 1

    rejected = asyncio.run(integration.recover_prepared(
        offered.recovery_id, "paid", req, messages(), {"task": "vision"}, config,
        db_path=db_path, retention_dir=retention_dir, provider_call=provider_call,
        approval_token="not-a-valid-grant",
    ))

    assert rejected.success is False
    assert rejected.error_class == "paid_approval_invalid"
    assert calls == []
    assert retained[0].exists()

    approved = asyncio.run(integration.recover_prepared(
        offered.recovery_id, "paid", req, messages(), {"task": "vision"}, config,
        db_path=db_path, retention_dir=retention_dir, provider_call=provider_call,
        approval_token=grant.token,
    ))
    assert approved.success is True
    assert len(calls) == 1
    assert list(retention_dir.glob("*.media")) == []



def test_invalid_provider_reported_cost_is_not_trusted_as_confirmed():
    async def provider_call(**kwargs):
        return {
            "choices": [{"message": {"content": "analysis"}, "finish_reason": "stop"}],
            "provider_confirmed_cost_usd": "-5",
        }

    invoke = integration._provider_invoke(messages(), {}, provider_call)
    candidate = integration.build_candidates(vision_config())[0]
    control = integration.RouteControl("req", 1, 0, 60, 60, output_cap=512)
    response = asyncio.run(invoke(candidate, control))

    assert response.content == "analysis"
    assert response.actual_cost_usd is None
    assert response.cost_source is None



def test_malformed_active_config_never_silently_falls_back_to_legacy():
    config = vision_config()
    config["routing"]["budgets"] = {
        "paid_per_request_usd": "NaN", "paid_daily_usd": "10", "paid_monthly_usd": "20",
    }

    assert integration.routing_mode(config) == "active"


def test_string_false_does_not_enable_paid_or_local_routes():
    config = vision_config()
    config["routing"]["allow"]["paid"] = "false"
    config["routing"]["allow"]["local"] = "false"

    parsed = integration.build_routing_config(config)

    assert parsed.allow_paid is False
    assert parsed.allow_local is False



def test_malformed_allow_mapping_is_a_validation_error_not_a_crash():
    config = vision_config()
    config["routing"]["allow"] = "paid"

    issues = integration.routing_validation_issues(config)

    assert any("allow must be a mapping" in message for _, message, _ in issues)


def test_empty_non_mapping_candidates_is_rejected():
    config = vision_config()
    config["routing"]["candidates"] = []

    issues = integration.routing_validation_issues(config)

    assert any("candidates must be a mapping" in message for _, message, _ in issues)


def test_non_numeric_quality_sample_size_is_a_validation_error_not_a_crash():
    config = vision_config()
    candidate = config["routing"]["candidates"]["free_image"]
    candidate.update(quality_score=0.5, quality_evidence_id="evidence-v1", quality_sample_size="many")

    issues = integration.routing_validation_issues(config)

    assert any("quality_sample_size" in message for _, message, _ in issues)


def test_malformed_active_allow_fails_closed_without_crashing_route_probes():
    config = vision_config()
    config["routing"]["allow"] = ["paid"]

    assert integration.routing_mode(config) == "active"
    assert integration.image_route_available(config) is False


def test_attempt_and_summary_telemetry_excludes_prompt_media_and_provider_secrets(tmp_path):
    config = vision_config()
    payload = messages()
    secret_prompt = "prompt-text-that-must-not-be-stored"
    secret_media = "data:image/png;base64,c2VjcmV0LW1lZGlhLXBheWxvYWQ="
    secret_provider = "provider-secret-that-must-not-be-stored"
    payload[0]["content"][0]["text"] = secret_prompt
    payload[0]["content"][1]["image_url"]["url"] = secret_media
    req = media_request(media_hash=hashlib.sha256(secret_media.encode()).hexdigest())
    db_path = tmp_path / "routing.db"

    async def provider_call(**kwargs):
        raise integration.ProviderError("rate_limit", secret_provider, retry_after_seconds=0)

    result = asyncio.run(integration.analyze_prepared(
        req, payload, {"task": "vision"}, config, db_path=db_path,
        retention_dir=tmp_path / "retained", provider_call=provider_call,
    ))

    assert result.success is False
    events = integration.SQLiteVisionStore(db_path).events()
    assert any(event["event_type"] == "vision_attempt" for event in events)
    assert any(event["event_type"] == "vision_request_summary" for event in events)
    encoded = json.dumps(events, sort_keys=True)
    assert secret_prompt not in encoded
    assert secret_media not in encoded
    assert secret_provider not in encoded
