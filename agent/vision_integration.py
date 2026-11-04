"""Shared auxiliary vision-tool adapter.

Policy lives in agent.vision_routing. This module only translates prepared tool media and
configuration into that policy and normalizes existing auxiliary-client responses.
"""
from __future__ import annotations
import copy

import asyncio
from datetime import datetime, timezone, UTC
from decimal import Decimal, InvalidOperation
import hashlib
import os
from pathlib import Path
import time
import secrets
from typing import Any, Awaitable, Callable, Optional

from agent.vision_routing import (
    BudgetConfig, Candidate, PriceSnapshot, ProviderError, ProviderResponse,
    RecoveryLimits, RouteControl, RoutingConfig, RoutingResult, SQLiteVisionStore,
    VisionRouter,
)


def current_price_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _decimal(value: Any) -> Optional[Decimal]:
    if value is None or value == "":
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def _positive(value: Any, default: float) -> float:
    try:
        parsed = float(value)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def _boolean(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    if value in {0, 1}:
        return bool(value)
    return default


def _price_snapshot(value: Any) -> Optional[PriceSnapshot]:
    if not isinstance(value, dict):
        return None
    try:
        return PriceSnapshot(
            captured_at=str(value["captured_at"]), source=str(value["source"]),
            input_usd_per_token=_decimal(value.get("input_usd_per_token")),
            output_usd_per_token=_decimal(value.get("output_usd_per_token")),
            request_usd=_decimal(value.get("request_usd")) or Decimal(0),
            media_usd=_decimal(value.get("media_usd")) or Decimal(0),
            max_age_seconds=int(value.get("max_age_seconds", 86400)),
            included_fees=tuple(str(item) for item in value.get("included_fees", ())),
            excluded_or_unknown_fees=tuple(str(item) for item in value.get("excluded_or_unknown_fees", ())),
        )
    except (KeyError, TypeError, ValueError):
        return None


def build_candidates(vision_config: dict[str, Any]) -> tuple[Candidate, ...]:
    routing = vision_config.get("routing") or {}
    configured_provider = str(vision_config.get("provider") or "auto")
    configured_model = str(vision_config.get("model") or "")
    candidates = []
    for order, (candidate_id, item) in enumerate((routing.get("candidates") or {}).items()):
        if not isinstance(item, dict):
            continue
        provider = str(item.get("provider") or "")
        model = str(item.get("model") or "")
        if not provider or provider == "auto" or not model:
            continue
        try:
            candidate = Candidate(
            candidate_id=str(candidate_id), provider=provider, model=model,
            base_url=item.get("base_url") or None,
            declared_media=frozenset(str(value) for value in item.get("declared_media", ())),
            verified_media=frozenset(str(value) for value in item.get("verified_media", ())),
            failed_media=frozenset(str(value) for value in item.get("failed_media", ())),
            cost_class=str(item.get("cost_class") or "hosted_free"),
            disclosure_id=str(item.get("disclosure_id") or "hosted_free_shared_quota_v1"),
            privacy_disclosure=item.get("privacy_disclosure"),
            configured_order=int(item.get("configured_order", order)),
            current_configured=(
                provider == configured_provider
                and (not configured_model or model == configured_model)
            ),
            automatic_fallback=bool(item.get("automatic_fallback", False)),
            available=bool(item.get("available", True)),
            unavailable_reason=item.get("unavailable_reason"),
            pricing=_price_snapshot(item.get("pricing")),
            quality_evidence_id=item.get("quality_evidence_id"),
            quality_score=float(item["quality_score"]) if item.get("quality_score") is not None else None,
            quality_sample_size=int(item.get("quality_sample_size", 0)),
            credential_pool_scope=str(item.get("credential_pool_scope") or "default"),
            )
        except (TypeError, ValueError):
            continue
        candidates.append(candidate)
    return tuple(candidates)


def build_routing_config(vision_config: dict[str, Any]) -> RoutingConfig:
    routing = vision_config.get("routing")
    if routing is None:
        routing = {}
    if not isinstance(routing, dict):
        raise ValueError("auxiliary.vision.routing must be a mapping")
    sections = {}
    for name in ("allow", "recovery", "budgets"):
        value = routing.get(name)
        if value is None:
            value = {}
        if not isinstance(value, dict):
            raise ValueError(f"auxiliary.vision.routing.{name} must be a mapping")
        sections[name] = value
    allow = sections["allow"]
    recovery = sections["recovery"]
    disabled_candidates = routing.get("disabled_candidates", ())
    if not isinstance(disabled_candidates, (list, tuple, set)):
        raise ValueError("auxiliary.vision.routing.disabled_candidates must be a list")
    max_same_route = min(2, max(0, int(recovery.get("max_same_route_retries", 2))))
    limits = RecoveryLimits(
        max_provider_attempts=min(3, max(0, int(recovery.get("max_provider_attempts", 3)))),
        max_same_route_retries=max_same_route,
        max_fallback_transitions=min(1, max(0, int(recovery.get("max_fallback_transitions", 1)))),
        automatic_same_route_retries=min(
            max_same_route, 2, max(0, int(recovery.get("automatic_same_route_retries", 1))),
        ),
        offer_ttl_seconds=min(900, max(1, int(recovery.get("offer_ttl_seconds", 900)))),
        image_attempt_timeout_seconds=min(60.0, _positive(recovery.get("image_attempt_timeout_seconds"), 60.0)),
        image_overall_timeout_seconds=min(90.0, _positive(recovery.get("image_overall_timeout_seconds"), 90.0)),
        video_attempt_timeout_seconds=min(180.0, _positive(recovery.get("video_attempt_timeout_seconds"), 180.0)),
        video_overall_timeout_seconds=min(240.0, _positive(recovery.get("video_overall_timeout_seconds"), 240.0)),
    )
    budgets = sections["budgets"]
    pinned = routing.get("pinned_candidate") or None
    explicit_provider = str(vision_config.get("provider") or "auto")
    explicit_model = str(vision_config.get("model") or "")
    if not pinned and (explicit_provider != "auto" or explicit_model):
        matching = [
            candidate.candidate_id for candidate in build_candidates(vision_config)
            if (explicit_provider == "auto" or candidate.provider == explicit_provider)
            and (not explicit_model or candidate.model == explicit_model)
        ]
        pinned = matching[0] if len(matching) == 1 else "unregistered-explicit-route"
    return RoutingConfig(
        mode=str(routing.get("mode") or "legacy"),
        strategy=str(routing.get("strategy") or "free_first"),
        pinned_candidate=pinned,
        disabled_candidates=frozenset(str(value) for value in disabled_candidates),
        allow_hosted_free=_boolean(allow.get("hosted_free"), True),
        allow_local=_boolean(allow.get("local"), False),
        allow_paid=_boolean(allow.get("paid"), False),
        recovery=limits,
        budgets=BudgetConfig(
            scope=str(budgets.get("scope") or "profile"),
            ledger_path=budgets.get("ledger_path") or None,
            paid_per_request_usd=_decimal(budgets.get("paid_per_request_usd")),
            paid_daily_usd=_decimal(budgets.get("paid_daily_usd")),
            paid_monthly_usd=_decimal(budgets.get("paid_monthly_usd")),
        ),
    )


def routing_validation_issues(vision_config: dict[str, Any]) -> list[tuple[str, str, str]]:
    from agent.vision_validation import routing_validation_issues as validate
    return validate(vision_config, build_config=build_routing_config, boolean=_boolean, decimal=_decimal)


def routing_mode(vision_config: Optional[dict[str, Any]]) -> str:
    raw_routing = (vision_config or {}).get("routing")
    declared = str(raw_routing.get("mode") or "legacy") if isinstance(raw_routing, dict) else "legacy"
    try:
        return build_routing_config(vision_config or {}).mode
    except (TypeError, ValueError):
        # An explicit rollout mode must never degrade into unobserved legacy retries or paid calls.
        return declared


def media_route_available(vision_config: dict[str, Any], media_kind: str) -> bool:
    try:
        config = build_routing_config(vision_config)
    except (TypeError, ValueError):
        return False
    if config.mode != "active":
        return True
    for candidate in build_candidates(vision_config):
        if (
            candidate.available
            and media_kind in candidate.declared_media and media_kind in candidate.verified_media
            and media_kind not in candidate.failed_media
            and candidate.candidate_id not in config.disabled_candidates
            and (candidate.cost_class != "hosted_free" or config.allow_hosted_free)
            and (candidate.cost_class != "local_no_api_fee" or config.allow_local)
            and not (media_kind == "video" and candidate.cost_class == "local_no_api_fee")
            and (candidate.cost_class != "paid" or config.allow_paid)
        ):
            if not config.pinned_candidate or config.pinned_candidate == candidate.candidate_id:
                return True
    return False


def video_route_available(vision_config: dict[str, Any]) -> bool:
    return media_route_available(vision_config, "video")


def image_route_available(vision_config: dict[str, Any]) -> bool:
    return media_route_available(vision_config, "image")


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _response_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    choices = _field(response, "choices", ()) or ()
    if not choices:
        return ""
    message = _field(choices[0], "message", {}) or {}
    content = _field(message, "content", "")
    if isinstance(content, list):
        content = "".join(str(_field(part, "text", "")) for part in content)
    if not content:
        content = _field(message, "reasoning_content", "") or _field(message, "reasoning", "")
    return str(content or "")


def _finish_reason(response: Any) -> Optional[str]:
    choices = _field(response, "choices", ()) or ()
    return _field(choices[0], "finish_reason") if choices else None


def _token_usage(response: Any) -> tuple[Optional[int], Optional[int]]:
    usage = _field(response, "usage", {}) or {}
    prompt = _field(usage, "prompt_tokens", _field(usage, "input_tokens"))
    completion = _field(usage, "completion_tokens", _field(usage, "output_tokens"))
    return (int(prompt) if prompt is not None else None, int(completion) if completion is not None else None)


def classify_provider_error(exc: Exception) -> ProviderError:
    status = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    status = status or getattr(response, "status_code", None)
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError):
        status = None
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    retry_after = None
    headers = getattr(response, "headers", {}) or {}
    raw_retry_after = headers.get("retry-after") if hasattr(headers, "get") else None
    try:
        retry_after = max(0.0, float(raw_retry_after)) if raw_retry_after is not None else None
    except (TypeError, ValueError):
        retry_after = None
    if status == 429:
        error_class = "rate_limit"
    elif status == 408:
        error_class = "http_408"
    elif status == 402:
        error_class = "budget_exhausted"
    elif status in {401, 403}:
        error_class = "authentication" if status == 401 else "policy_denied"
    elif status is not None and status >= 500:
        error_class = "provider_5xx"
    elif isinstance(exc, (asyncio.TimeoutError, TimeoutError)) or "timeout" in name:
        error_class = "timeout"
    elif isinstance(exc, ConnectionError) or any(value in name for value in ("connection", "connecterror")):
        error_class = "connection"
    elif status == 400 and any(value in text for value in (
        "does not support", "not support", "multimodal", "image input", "video input", "video_url",
    )):
        error_class = "capability"
    elif status == 400 and any(value in text for value in ("too large", "image size", "payload size")):
        error_class = "image_size"
    elif status == 400:
        error_class = "invalid_request"
    else:
        error_class = "provider_unknown"
    return ProviderError(error_class, retry_after_seconds=retry_after, status_code=status)


def _data_url(messages: list) -> Optional[str]:
    try:
        content = messages[0]["content"]
        for part in content:
            for key in ("image_url", "video_url"):
                if key in part:
                    return str(part[key]["url"])
    except (KeyError, IndexError, TypeError):
        return None
    return None


def _retain(messages: list, directory: Path) -> Optional[str]:
    data_url = _data_url(messages)
    if not data_url or not data_url.startswith("data:"):
        return None
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    reference = secrets.token_urlsafe(24)
    path = directory / f"{reference}.media"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data_url.encode("utf-8"))
    return reference


def _release(reference: Optional[str], directory: Path) -> None:
    if not reference:
        return
    if "/" in reference or "\\" in reference or reference in {".", ".."}:
        return
    try:
        (directory / f"{reference}.media").unlink(missing_ok=True)
    except OSError:
        pass


def _provider_invoke(
    messages: list, call_kwargs: dict[str, Any], provider_call, media_correction=None,
):
    active_messages = messages

    async def invoke(candidate: Candidate, control: RouteControl) -> ProviderResponse:
        nonlocal provider_call, active_messages
        if provider_call is None:
            from agent.auxiliary_client import async_call_llm
            provider_call = async_call_llm
        if control.correction == "downscale":
            if media_correction is None:
                raise ProviderError("image_resize_unavailable")
            try:
                active_messages = await media_correction(active_messages, control)
            except ProviderError:
                raise
            except Exception as exc:
                raise ProviderError("image_resize_unavailable") from exc
        kwargs = dict(call_kwargs)
        kwargs["messages"] = active_messages
        kwargs.update({
            "provider": candidate.provider, "model": candidate.model,
            "timeout": control.max_attempt_seconds, "max_tokens": control.output_cap,
            "route_control": control,
        })
        if candidate.base_url:
            kwargs["base_url"] = candidate.base_url
        if control.correction:
            kwargs["reasoning_config"] = {"effort": "low"}
        try:
            response = await provider_call(**kwargs)
        except ProviderError:
            raise
        except Exception as exc:
            raise classify_provider_error(exc) from exc
        input_tokens, output_tokens = _token_usage(response)
        actual = _decimal(_field(response, "provider_confirmed_cost_usd"))
        if actual is not None and (not actual.is_finite() or actual < 0):
            actual = None
        return ProviderResponse(
            content=_response_text(response), finish_reason=_finish_reason(response),
            input_tokens=input_tokens, output_tokens=output_tokens,
            actual_cost_usd=actual,
            cost_source="provider_confirmed" if actual is not None else None,
            upstream_provider=_field(response, "provider"),
        )

    return invoke


async def analyze_prepared(
    request, messages: list, call_kwargs: dict[str, Any], vision_config: dict[str, Any], *,
    db_path: Path | str, retention_dir: Path | str,
    provider_call: Optional[Callable[..., Awaitable[Any]]] = None,
    media_correction: Optional[Callable[[list, RouteControl], Awaitable[list]]] = None,
) -> RoutingResult:
    config = build_routing_config(vision_config)
    if config.mode != "active":
        raise ValueError("analyze_prepared requires active routing mode")
    router = VisionRouter(build_candidates(vision_config), db_path=db_path, config=config)
    retention_dir = Path(retention_dir)
    cleanup_expired_recoveries(
        db_path=db_path, retention_dir=retention_dir,
        ledger_path=config.budgets.ledger_path if config.budgets.scope == "shared" else None,
    )
    retained = _retain(messages, retention_dir)
    invoke = _provider_invoke(messages, call_kwargs, provider_call, media_correction)
    try:
        result = await router.execute(request, invoke, retained_media_ref=retained)
    except Exception:
        _release(retained, retention_dir)
        raise
    if result.state != "recovery_required":
        _release(retained, retention_dir)
    return result


def _retained_messages(messages: list, reference: str, directory: Path, media_hash: str) -> Optional[list]:
    if not reference or "/" in reference or chr(92) in reference or reference in {".", ".."}:
        return None
    try:
        data_url = (directory / f"{reference}.media").read_text(encoding="utf-8-sig")
    except OSError:
        return None
    if hashlib.sha256(data_url.encode("utf-8")).hexdigest() != media_hash:
        return None
    corrected = copy.deepcopy(messages)
    for part in corrected[0].get("content", []):
        for key in ("image_url", "video_url"):
            if key in part:
                part[key]["url"] = data_url
    return corrected


async def recover_prepared(
    recovery_id: str, choice: str, request, messages: list, call_kwargs: dict[str, Any],
    vision_config: dict[str, Any], *, db_path: Path | str, retention_dir: Path | str,
    provider_call: Optional[Callable[..., Awaitable[Any]]] = None,
    approval_token: Optional[str] = None,
) -> RoutingResult:
    config = build_routing_config(vision_config)
    if config.mode != "active":
        raise ValueError("recover_prepared requires active routing mode")
    router = VisionRouter(build_candidates(vision_config), db_path=db_path, config=config)
    retention_dir = Path(retention_dir)
    cleanup_expired_recoveries(
        db_path=db_path, retention_dir=retention_dir,
        ledger_path=config.budgets.ledger_path if config.budgets.scope == "shared" else None,
    )
    state = router.store.load_recovery(recovery_id)
    reference = state.get("retained_media_ref") if state else None
    fingerprint_matches = bool(
        state and state["request"].get("fingerprint") == request.fingerprint
    )
    recovered_messages = (
        _retained_messages(messages, reference, retention_dir, request.media_hash)
        if state and reference and fingerprint_matches else None
    )
    if (
        state and reference and fingerprint_matches and recovered_messages is None
        and state.get("status") == "offered" and time.time() < float(state.get("expires_at_epoch", 0))
    ):
        router._close_recovery(recovery_id, "secure_media_retention_unavailable")
        _release(reference, retention_dir)
        return RoutingResult(
            False, "terminal_failed", "Secure media retention is unavailable or does not match the request.",
            request.request_id, recovery_id=recovery_id, error_class="secure_media_retention_unavailable",
        )
    invoke = _provider_invoke(recovered_messages or messages, call_kwargs, provider_call)
    result = await router.recover(
        recovery_id, choice, request=request, invoke=invoke, approval_token=approval_token,
    )
    remaining = router.store.load_recovery(recovery_id)
    if remaining is None or remaining.get("status") != "offered":
        _release(reference, retention_dir)
    return result


def approve_paid(
    recovery_id: str, *, candidate_id: str, request, vision_config: dict[str, Any], db_path: Path | str,
):
    router = VisionRouter(
        build_candidates(vision_config), db_path=db_path, config=build_routing_config(vision_config),
    )
    return router.approve_paid(recovery_id, candidate_id=candidate_id, request=request)


def cleanup_expired_recoveries(
    *, db_path: Path | str, retention_dir: Path | str, now: Optional[float] = None,
    ledger_path: Optional[Path | str] = None,
) -> int:
    store = SQLiteVisionStore(db_path)
    retention_dir = Path(retention_dir)
    now = time.time() if now is None else now
    cleaned = 0
    ledger_store = SQLiteVisionStore(ledger_path) if ledger_path else store
    for state in store.recovery_states():
        if state.get("status") == "offered" and now >= float(state.get("expires_at_epoch", 0)):
            _release(state.get("retained_media_ref"), retention_dir)
            for grant in ledger_store.unconsumed_grants(state["recovery_id"]):
                store.emit("vision_paid_approval_expired", state["request"]["request_id"], {
                    "request_id": state["request"]["request_id"], "profile": state["request"]["profile"],
                    "recovery_id": state["recovery_id"], "grant_id": grant["grant_id"],
                    "candidate_id": grant["candidate_id"],
                })
            store.close_recovery(state["recovery_id"], "expired")
            store.emit("vision_recovery_expired", state["request"]["request_id"], {
                "request_id": state["request"]["request_id"], "profile": state["request"]["profile"],
                "recovery_id": state["recovery_id"],
            })
            if ledger_store is not store:
                ledger_store.close_recovery(state["recovery_id"], "expired")
            cleaned += 1
    return cleaned


def record_legacy_outcome(
    request, *, candidate_id: Optional[str], db_path: Path | str, started_monotonic: float,
    analysis: str = "", error: Optional[Exception] = None,
) -> None:
    """Shadow-mode baseline telemetry; records outcomes, never content or provider error text."""
    error_class = classify_provider_error(error).error_class if error is not None else (
        None if analysis else "empty_content"
    )
    SQLiteVisionStore(db_path).emit("vision_legacy_summary", request.request_id, {
        "request_id": request.request_id, "profile": request.profile,
        "framework_call_id": request.framework_call_id, "session_id": request.session_id,
        "media_kind": request.media_kind, "size_bucket": request.size_bucket,
        "media_hash": request.media_hash, "candidate_id": candidate_id,
        "success": error is None and bool(analysis), "analysis_nonempty": bool(analysis),
        "error_class": error_class,
        "total_latency_ms": round(
            (time.monotonic() - started_monotonic) * 1000 + (request.preparation_latency_ms or 0), 3,
        ),
        "estimated_cost_usd": None, "actual_cost_usd": None, "cost_status": "unknown_legacy",
        "provider_attempts": None,
    })


def shadow_prepared(request, vision_config: dict[str, Any], *, db_path: Path | str):
    router = VisionRouter(
        build_candidates(vision_config), db_path=db_path, config=build_routing_config(vision_config),
    )
    return router.shadow(request)


def prompt_fingerprint(prompt: str) -> str:
    normalized = " ".join(prompt.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
