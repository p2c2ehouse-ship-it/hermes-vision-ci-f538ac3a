from __future__ import annotations

import asyncio
from datetime import datetime, timezone, UTC
from decimal import Decimal

import pytest

from agent.vision_routing import Candidate, MediaRequest, ProviderError, ProviderResponse, VisionRouter
from agent import vision_routing as vr


def request(**changes):
    values = dict(
        request_id="req-1", profile="test-profile", session_id="session-1",
        media_kind="image", media_hash="a" * 64, normalized_prompt_hash="b" * 64,
        mime_type="image/png", byte_count=1000, base64_bytes=1400,
        output_cap=512, hard_output_cap=1024, interaction_mode="interactive",
        source_kind="local",
    )
    values.update(changes)
    return MediaRequest(**values)


def candidate(candidate_id="free", **changes):
    values = dict(
        candidate_id=candidate_id, provider="openrouter",
        model="example/free-vision:free", declared_media=frozenset({"image"}),
        verified_media=frozenset({"image"}), cost_class="hosted_free",
        disclosure_id="hosted_free_shared_quota_v1",
    )
    values.update(changes)
    return Candidate(**values)


def test_free_success_dispatches_once_and_records_latency_and_cost(tmp_path):
    router = VisionRouter([candidate()], db_path=tmp_path / "vision.db")
    calls = []

    async def invoke(selected, control):
        calls.append((selected.candidate_id, control))
        return ProviderResponse(content="A diagram of a flowchart.", input_tokens=10, output_tokens=6)

    result = asyncio.run(router.execute(request(), invoke))

    assert result.success is True
    assert result.analysis == "A diagram of a flowchart."
    assert result.state == "succeeded"
    assert result.automatic_retry_allowed is False
    assert len(calls) == 1
    assert calls[0][0] == "free"
    assert calls[0][1].max_attempt_seconds == 60
    assert result.cost_status == "not_applicable_free"
    metrics = router.metrics()
    assert metrics["logical_requests"] == 1
    assert metrics["successful_requests"] == 1
    assert metrics["provider_attempts"] == 1
    assert metrics["estimated_cost_usd"] == Decimal(0)
    assert metrics["p50_total_latency_ms"] >= 0


def test_transient_rate_limit_gets_one_bounded_same_route_retry(tmp_path):
    router = VisionRouter([candidate()], db_path=tmp_path / "vision.db")
    calls = []

    async def invoke(selected, control):
        calls.append(control)
        if len(calls) == 1:
            raise ProviderError("rate_limit", "shared pool 429", retry_after_seconds=0)
        return ProviderResponse(content="recovered")

    result = asyncio.run(router.execute(request(), invoke))

    assert result.success is True
    assert result.analysis == "recovered"
    assert len(calls) == 2
    assert [call.attempt_index for call in calls] == [1, 2]
    assert all(call.fallback_index == 0 for call in calls)
    assert result.cost_status == "not_applicable_free"
    assert router.metrics()["provider_attempts"] == 2



def test_permanent_failure_does_not_retry_and_returns_structured_choices(tmp_path):
    router = VisionRouter([candidate()], db_path=tmp_path / "vision.db")
    calls = []

    async def invoke(selected, control):
        calls.append(control)
        raise ProviderError("capability", "deterministic capability rejection")

    result = asyncio.run(router.execute(request(), invoke, retained_media_ref="retained-1"))

    assert result.success is False
    assert result.state == "terminal_failed"
    assert result.error_class == "capability"
    assert result.automatic_retry_allowed is False
    assert len(calls) == 1
    assert [choice["choice"] for choice in result.choices] == ["paid", "free", "retry"]
    assert all(choice["viable"] is False for choice in result.choices)
    assert result.choices[2]["likelihood"] == "unknown"


def test_no_verified_candidate_fails_closed_without_provider_call(tmp_path):
    image_only = candidate(verified_media=frozenset({"image"}))
    router = VisionRouter([image_only], db_path=tmp_path / "vision.db")
    called = False

    async def invoke(selected, control):
        nonlocal called
        called = True
        return ProviderResponse(content="should not happen")

    result = asyncio.run(router.execute(request(media_kind="video", mime_type="video/mp4"), invoke))

    assert result.success is False
    assert result.state == "terminal_failed"
    assert result.error_class == "no_viable_candidate"
    assert called is False
    assert len(result.choices) == 3


def test_transient_failure_stops_at_shared_retry_bound_and_caches_recovery(tmp_path):
    router = VisionRouter([candidate()], db_path=tmp_path / "vision.db")
    calls = []

    async def invoke(selected, control):
        calls.append(control)
        raise ProviderError("rate_limit", "429", retry_after_seconds=0)

    first = asyncio.run(router.execute(request(), invoke, retained_media_ref="retained-1"))
    second = asyncio.run(router.execute(request(), invoke, retained_media_ref="retained-1"))

    assert first.success is False
    assert first.state == "recovery_required"
    assert first.recovery_id == second.recovery_id
    assert len(calls) == 2
    assert first.choices[2]["viable"] is True
    third = asyncio.run(router.recover(first.recovery_id, "retry", request=request(), invoke=invoke))
    assert third.success is False
    assert len(calls) == 3
    assert third.choices[2]["viable"] is False
    assert third.choices[2]["reason"] == "same_route_retry_budget_exhausted"


def test_free_recovery_uses_fallback_once_and_discloses_hosted_limitations(tmp_path):
    original = candidate("original")
    fallback = candidate(
        "fallback", provider="openrouter", model="example/other:free", configured_order=1,
        privacy_disclosure="Registered provider data-use limitation: review retained-media terms.",
    )
    router = VisionRouter([original, fallback], db_path=tmp_path / "vision.db")
    calls = []

    async def invoke(selected, control):
        calls.append((selected.candidate_id, control))
        if selected.candidate_id == "original":
            raise ProviderError("rate_limit", "429", retry_after_seconds=0)
        return ProviderResponse(content="free fallback analysis")

    first = asyncio.run(router.execute(request(), invoke, retained_media_ref="retained-1"))
    free = next(choice for choice in first.choices if choice["choice"] == "free")

    assert free["viable"] is True
    assert free["cost_class"] == "hosted_free"
    assert free["candidate_id"] == "fallback"
    assert "shared quotas" in " ".join(free["disclosures"])
    assert "leaves the device" in " ".join(free["disclosures"])
    assert "Registered provider data-use limitation" in " ".join(free["disclosures"])
    recovered = asyncio.run(router.recover(first.recovery_id, "free", request=request(), invoke=invoke))
    assert recovered.success is True
    assert recovered.analysis == "free fallback analysis"
    assert calls[-1][0] == "fallback"
    assert calls[-1][1].fallback_index == 1
    assert len(calls) == 3


def test_empty_response_gets_only_one_bounded_correction(tmp_path):
    router = VisionRouter([candidate()], db_path=tmp_path / "vision.db")
    calls = []

    async def invoke(selected, control):
        calls.append(control)
        return ProviderResponse(content="", finish_reason="length")

    result = asyncio.run(router.execute(request(), invoke, retained_media_ref="retained-1"))

    assert result.success is False
    assert len(calls) == 2
    assert calls[0].correction is None
    assert calls[1].correction == "bounded_output_correction"
    assert calls[1].output_cap == 1024
    assert result.choices[2]["viable"] is False


def test_recovery_requires_secure_media_retention(tmp_path):
    router = VisionRouter([candidate()], db_path=tmp_path / "vision.db")

    async def invoke(selected, control):
        raise ProviderError("rate_limit", "429", retry_after_seconds=0)

    result = asyncio.run(router.execute(request(), invoke))

    assert all(choice["viable"] is False for choice in result.choices)
    assert all(choice["reason"] == "secure_media_retention_unavailable" for choice in result.choices)



def paid_candidate(**changes):
    pricing = vr.PriceSnapshot(
        captured_at=datetime.now(UTC).isoformat(), source="test-price-snapshot",
        input_usd_per_token=Decimal("0.001"), output_usd_per_token=Decimal("0.002"),
        request_usd=Decimal("0.01"), max_age_seconds=99999999999,
        included_fees=("input", "output", "request"),
        excluded_or_unknown_fees=("provider taxes or minimums",),
    )
    values = dict(
        candidate_id="paid", provider="paid-provider", model="paid-vision",
        declared_media=frozenset({"image"}), verified_media=frozenset({"image"}),
        cost_class="paid", disclosure_id="hosted_paid_v1", pricing=pricing, configured_order=2,
    )
    values.update(changes)
    return Candidate(**values)


def paid_config(**budget_changes):
    budgets = vr.BudgetConfig(
        paid_per_request_usd=Decimal(2), paid_daily_usd=Decimal(10),
        paid_monthly_usd=Decimal(20), **budget_changes,
    )
    return vr.RoutingConfig(mode="active", allow_paid=True, budgets=budgets)


def paid_request(**changes):
    return request(estimated_input_tokens=100, input_token_count_exact=True, **changes)


def test_paid_offer_has_real_estimate_and_unapproved_recovery_makes_no_call(tmp_path):
    paid_provider = paid_candidate()
    router = VisionRouter([candidate(), paid_provider], db_path=tmp_path / "vision.db", config=paid_config())
    calls = []

    async def invoke(selected, control):
        calls.append(selected.candidate_id)
        if selected.candidate_id == "free":
            raise ProviderError("capability", "permanent")
        return ProviderResponse(content="paid analysis")

    first = asyncio.run(router.execute(paid_request(), invoke, retained_media_ref="retained-1"))
    paid = first.choices[0]

    assert paid["viable"] is True
    assert paid["estimated_cost_usd"] == "1.134"
    assert paid["estimate_method"] == "exact_preflight_cap"
    assert paid["price_timestamp"] == paid_provider.pricing.captured_at
    assert paid["approval_required"] is True
    assert paid["output_cap"] == 512
    assert "actual provider billing can differ" in " ".join(paid["disclosures"])
    denied = asyncio.run(router.recover(first.recovery_id, "paid", request=paid_request(), invoke=invoke))
    assert denied.success is False
    assert denied.error_class == "paid_approval_required"
    assert calls == ["free"]
    assert router.metrics()["paid_approvals_denied"] == 1
    assert router.store.reservations() == []


def test_trusted_single_use_grant_reserves_before_exactly_one_paid_call(tmp_path):
    router = VisionRouter([candidate(), paid_candidate()], db_path=tmp_path / "vision.db", config=paid_config())
    calls = []

    async def invoke(selected, control):
        calls.append(selected.candidate_id)
        if selected.candidate_id == "free":
            raise ProviderError("capability", "permanent")
        assert len(router.store.reservations()) == 1
        return ProviderResponse(
            content="paid analysis", actual_cost_usd=Decimal("1.05"), cost_source="provider_receipt",
        )

    req = paid_request()
    first = asyncio.run(router.execute(req, invoke, retained_media_ref="retained-1"))
    grant = router.approve_paid(first.recovery_id, candidate_id="paid", request=req)
    approved = asyncio.run(router.recover(
        first.recovery_id, "paid", request=req, invoke=invoke, approval_token=grant.token,
    ))

    assert approved.success is True
    assert approved.cost_status == "provider_confirmed"
    assert calls == ["free", "paid"]
    reservation = router.store.reservations()[0]
    assert reservation["estimated_usd"] == "1.134"
    assert reservation["actual_usd"] == "1.05"
    assert reservation["cost_status"] == "provider_confirmed"
    summary = [event for event in router.store.events() if event["event_type"] == "vision_request_summary"][-1]
    assert summary["estimated_cost_usd"] == "1.134"
    assert summary["actual_cost_usd"] == "1.05"
    assert summary["recovery_choice_taken"] == "paid"
    assert router.metrics()["recovery_choices_taken"] == 1
    assert router.metrics()["logical_requests"] == 1
    assert router.metrics()["logical_request_success_rate"] == 1
    with router.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM approval_grants").fetchone()[0] == 0
    replay = asyncio.run(router.recover(
        first.recovery_id, "paid", request=req, invoke=invoke, approval_token=grant.token,
    ))
    assert replay.success is False
    assert calls == ["free", "paid"]


def test_paid_is_disabled_when_any_ceiling_is_unset(tmp_path):
    config = vr.RoutingConfig(mode="active", allow_paid=True)
    router = VisionRouter([paid_candidate()], db_path=tmp_path / "vision.db", config=config)
    called = False

    async def invoke(selected, control):
        nonlocal called
        called = True
        return ProviderResponse(content="not allowed")

    result = asyncio.run(router.execute(paid_request(), invoke, retained_media_ref="retained-1"))

    assert result.success is False
    assert called is False
    assert result.choices[0]["viable"] is False
    assert result.choices[0]["reason"] == "budget_ceilings_unset"


def test_paid_estimate_over_per_request_ceiling_is_disabled_before_dispatch(tmp_path):
    config = paid_config()
    config = vr.RoutingConfig(
        mode="active", allow_paid=True,
        budgets=vr.BudgetConfig(
            paid_per_request_usd=Decimal(1), paid_daily_usd=Decimal(10),
            paid_monthly_usd=Decimal(20),
        ),
    )
    router = VisionRouter([candidate(), paid_candidate()], db_path=tmp_path / "vision.db", config=config)

    async def invoke(selected, control):
        raise ProviderError("capability", "permanent")

    result = asyncio.run(router.execute(paid_request(), invoke, retained_media_ref="retained-1"))

    assert result.choices[0]["viable"] is False
    assert result.choices[0]["reason"] == "estimate_exceeds_per_request_budget"
    assert router.store.reservations() == []


def test_stale_or_missing_price_snapshot_disables_paid_choice(tmp_path):
    stale = paid_candidate()
    old_price = vr.PriceSnapshot(
        captured_at="2020-01-01T00:00:00+00:00", source="expired-test-snapshot",
        input_usd_per_token=Decimal("0.001"), output_usd_per_token=Decimal("0.002"),
    )
    stale = paid_candidate(pricing=old_price)
    router = VisionRouter([candidate(), stale], db_path=tmp_path / "vision.db", config=paid_config())

    async def invoke(selected, control):
        raise ProviderError("capability", "permanent")

    result = asyncio.run(router.execute(paid_request(), invoke, retained_media_ref="retained-1"))

    assert result.choices[0]["viable"] is False
    assert result.choices[0]["reason"] == "current_price_estimate_unavailable"
    assert router.store.reservations() == []



def test_initial_paid_selection_requires_approval_before_first_provider_call(tmp_path):
    free = candidate(quality_score=0.5, quality_sample_size=30, quality_evidence_id="benchmark-v1")
    paid = paid_candidate(
        quality_score=0.9, quality_sample_size=30, quality_evidence_id="benchmark-v1",
        configured_order=1,
    )
    quality_first = paid_config()
    quality_first = vr.RoutingConfig(
        mode="active", strategy="quality_first", allow_paid=True, budgets=quality_first.budgets,
    )
    router = VisionRouter([free, paid], db_path=tmp_path / "vision.db", config=quality_first)
    calls = []

    async def invoke(selected, control):
        calls.append((selected.candidate_id, control))
        return ProviderResponse(content="paid analysis")

    req = paid_request()
    offered = asyncio.run(router.execute(req, invoke, retained_media_ref="retained-1"))

    assert calls == []
    assert offered.state == "recovery_required"
    assert offered.error_class == "paid_approval_required"
    assert offered.candidate_id == "paid"
    assert offered.choices[0]["candidate_id"] == "paid"

    grant = router.approve_paid(offered.recovery_id, candidate_id="paid", request=req)
    approved = asyncio.run(router.recover(
        offered.recovery_id, "paid", request=req, invoke=invoke, approval_token=grant.token,
    ))

    assert approved.success is True
    assert len(calls) == 1
    assert calls[0][0] == "paid"
    assert calls[0][1].attempt_index == 1
    assert calls[0][1].fallback_index == 0



def test_transient_health_circuit_opens_after_three_failures_and_closes_after_half_open_success(tmp_path):
    now = [1_800_000_000.0]
    router = VisionRouter(
        [candidate()], db_path=tmp_path / "vision.db", wall_clock=lambda: now[0],
    )
    calls = []

    async def failing(selected, control):
        calls.append(control.request_id)
        raise ProviderError("rate_limit", "429", retry_after_seconds=0)

    first = asyncio.run(router.execute(request(request_id="r1", media_hash="1" * 64), failing, retained_media_ref="retained-1"))
    second = asyncio.run(router.execute(request(request_id="r2", media_hash="2" * 64), failing, retained_media_ref="retained-2"))

    assert len(calls) == 3
    assert first.state == "recovery_required"
    assert second.choices[2]["viable"] is False
    assert second.choices[2]["reason"] == "circuit_open"
    assert router.metrics()["circuit_openings"] == 1

    third = asyncio.run(router.execute(request(request_id="r3", media_hash="3" * 64), failing))
    assert third.error_class == "no_viable_candidate"
    assert len(calls) == 3

    now[0] += 61

    async def recovered(selected, control):
        calls.append(control.request_id)
        return ProviderResponse(content="half-open recovered")

    fourth = asyncio.run(router.execute(request(request_id="r4", media_hash="4" * 64), recovered))
    assert fourth.success is True
    assert fourth.analysis == "half-open recovered"
    assert len(calls) == 4
    assert router.store.circuit_state("free", "image", "default", now[0]) == "closed"



def test_retry_likelihood_uses_adequate_exact_stratum_first_retry_evidence(tmp_path):
    db_path = tmp_path / "vision.db"
    store = vr.SQLiteVisionStore(db_path)
    route = candidate()
    for index in range(30):
        request_id = f"evidence-{index}"
        media_hash = f"{index:064x}"
        common = {
            "profile": "test-profile", "candidate_id": "free", "provider": route.provider,
            "model": route.model, "media_kind": "image", "size_bucket": "unknown",
            "media_hash": media_hash, "fallback_index": 0,
        }
        store.emit("vision_attempt", request_id, {
            **common, "attempt_index": 1, "error_class": "rate_limit", "outcome": "failed_attempt",
        })
        succeeded = index < 6
        store.emit("vision_attempt", request_id, {
            **common, "attempt_index": 2, "error_class": None if succeeded else "rate_limit",
            "outcome": "nonempty_analysis" if succeeded else "failed_attempt",
        })

    router = VisionRouter([route], db_path=db_path)

    async def failing(selected, control):
        raise ProviderError("rate_limit", "429", retry_after_seconds=0)

    result = asyncio.run(router.execute(
        request(request_id="new-request", media_hash="f" * 64), failing, retained_media_ref="retained-1",
    ))
    retry = result.choices[2]

    assert retry["viable"] is True
    assert retry["numerator"] == 6
    assert retry["denominator"] == 31
    assert retry["likelihood"] == 6 / 31
    assert len(retry["wilson_95_interval"]) == 2
    assert retry["wilson_95_interval"][0] < retry["likelihood"] < retry["wilson_95_interval"][1]
    assert retry["evidence_source"] == "vision_attempt_telemetry"
    assert retry["model"] == route.model
    assert retry["media_kind"] == "image"
    assert retry["error_class"] == "rate_limit"
    assert retry["observation_window_days"] == 30
    assert "not a recognition-accuracy probability" in retry["success_definition"]



def test_framework_call_id_separates_distinct_identical_tool_calls(tmp_path):
    router = VisionRouter([candidate()], db_path=tmp_path / "vision.db")
    calls = []

    async def invoke(selected, control):
        calls.append(control.request_id)
        return ProviderResponse(content="analysis")

    first_request = request(request_id="call-1", framework_call_id="call-1")
    second_request = request(request_id="call-2", framework_call_id="call-2")
    first = asyncio.run(router.execute(first_request, invoke))
    repeated = asyncio.run(router.execute(first_request, invoke))
    distinct = asyncio.run(router.execute(second_request, invoke))

    assert first.success and repeated.success and distinct.success
    assert len(calls) == 2
    assert calls == ["call-1", "call-2"]



def test_shared_budget_ledger_prevents_concurrent_overspend(tmp_path):
    shared_path = tmp_path / "shared-ledger.db"
    config = vr.RoutingConfig(
        mode="active", allow_paid=True,
        budgets=vr.BudgetConfig(
            scope="shared", ledger_path=str(shared_path),
            paid_per_request_usd=Decimal(2), paid_daily_usd=Decimal("1.20"),
            paid_monthly_usd=Decimal(20),
        ),
    )
    routers = [
        VisionRouter([paid_candidate()], db_path=tmp_path / f"profile-{index}.db", config=config)
        for index in range(2)
    ]
    requests = [
        paid_request(
            request_id=f"paid-{index}", session_id=f"session-{index}", media_hash=str(index + 1) * 64,
        )
        for index in range(2)
    ]
    calls = []

    async def invoke(selected, control):
        calls.append(control.request_id)
        await asyncio.sleep(0)
        return ProviderResponse(content="paid analysis")

    offers = [
        asyncio.run(router.execute(req, invoke, retained_media_ref=f"retained-{index}"))
        for index, (router, req) in enumerate(zip(routers, requests))
    ]
    grants = [
        router.approve_paid(offer.recovery_id, candidate_id="paid", request=req)
        for router, offer, req in zip(routers, offers, requests)
    ]

    async def recover_concurrently():
        return await asyncio.gather(*[
            router.recover(
                offer.recovery_id, "paid", request=req, invoke=invoke, approval_token=grant.token,
            )
            for router, offer, req, grant in zip(routers, offers, requests, grants)
        ])

    results = asyncio.run(recover_concurrently())

    assert sum(result.success for result in results) == 1
    assert sum(result.error_class == "daily_budget_exhausted" for result in results) == 1
    assert len(calls) == 1
    assert len(vr.SQLiteVisionStore(shared_path).reservations()) == 1



def test_selection_attempt_and_summary_telemetry_support_baseline_comparison(tmp_path):
    rejected = candidate("unverified", verified_media=frozenset())
    router = VisionRouter([candidate(), rejected], db_path=tmp_path / "vision.db")

    async def invoke(selected, control):
        return ProviderResponse(content="analysis", input_tokens=11, output_tokens=7)

    result = asyncio.run(router.execute(request(size_bucket="small"), invoke))
    assert result.success is True
    events = router.store.events()

    assert [event["event_type"] for event in events] == [
        "vision_route_selected", "vision_attempt", "vision_request_summary",
    ]
    decision, attempt, summary = events
    assert decision["selected_candidate_id"] == "free"
    assert decision["selection_key"] == "free_first:cost>quality>current>order"
    assert {item["candidate_id"] for item in decision["rejected_candidates"]} == {"unverified"}
    for field in (
        "preparation_latency_ms", "download_latency_ms", "provider_latency_ms", "total_latency_ms",
        "timeout_stage", "cancellation_stage", "rejected_candidates", "selection_key",
        "budget_remaining", "framework_call_id",
    ):
        assert field in attempt
    assert attempt["input_tokens"] == 11
    assert attempt["output_tokens"] == 7
    assert summary["estimated_cost_usd"] == "0"
    assert summary["actual_cost_usd"] is None

    metrics = router.metrics()
    route_metrics = metrics["by_route"]["free:image"]
    assert route_metrics["logical_requests"] == 1
    assert route_metrics["successful_requests"] == 1
    assert route_metrics["provider_attempts"] == 1
    assert metrics["by_media_size"]["small"]["logical_requests"] == 1
    assert metrics["labeled_semantic_accuracy"] is None
    assert metrics["recovery_choices_taken"] == 0
    assert metrics["paid_approvals_denied"] == 0
    assert metrics["paid_approvals_expired"] == 0



def test_preparation_time_consumes_the_overall_provider_deadline(tmp_path):
    router = VisionRouter([candidate()], db_path=tmp_path / "vision.db")
    called = False

    async def invoke(selected, control):
        nonlocal called
        called = True
        return ProviderResponse(content="must not run")

    result = asyncio.run(router.execute(request(pre_routing_elapsed_seconds=91), invoke))

    assert result.success is False
    assert result.error_class == "deadline_exhausted"
    assert called is False



def test_local_video_recovery_is_disabled_without_verified_native_video_path(tmp_path):
    original = candidate(
        "original", declared_media=frozenset({"image", "video"}),
        verified_media=frozenset({"image", "video"}),
    )
    local = candidate(
        "local", provider="ollama", model="local-vision", cost_class="local_no_api_fee",
        declared_media=frozenset({"image", "video"}), verified_media=frozenset({"image", "video"}),
        configured_order=1,
    )
    config = vr.RoutingConfig(mode="active", allow_local=True)
    router = VisionRouter([original, local], db_path=tmp_path / "vision.db", config=config)

    async def invoke(selected, control):
        raise ProviderError("capability", "permanent")

    result = asyncio.run(router.execute(
        request(media_kind="video", mime_type="video/mp4"), invoke, retained_media_ref="retained-1",
    ))
    free = result.choices[1]

    assert free["viable"] is False
    assert free["reason"] == "no_verified_native_video_path"
    assert len(router.store.events()) >= 1



def test_free_first_and_quality_first_use_distinct_deterministic_ordering(tmp_path):
    free = candidate(quality_score=0.1, quality_sample_size=30, quality_evidence_id="benchmark-v1")
    paid = paid_candidate(
        quality_score=0.9, quality_sample_size=30, quality_evidence_id="benchmark-v1",
        configured_order=1,
    )
    free_first = VisionRouter([paid, free], db_path=tmp_path / "free.db", config=paid_config())
    quality_first = VisionRouter(
        [paid, free], db_path=tmp_path / "quality.db",
        config=vr.RoutingConfig(mode="active", strategy="quality_first", allow_paid=True, budgets=paid_config().budgets),
    )

    assert free_first.select(request()).candidate_id == "free"
    assert quality_first.select(request()).candidate_id == "paid"


def test_disabled_retry_recovery_choice_cannot_dispatch_provider_call(tmp_path):
    router = VisionRouter(
        [candidate(), candidate("fallback", configured_order=1)], db_path=tmp_path / "vision.db",
    )
    calls = []

    async def invoke(selected, control):
        calls.append(control)
        raise ProviderError("capability", "permanent")

    offered = asyncio.run(router.execute(request(), invoke, retained_media_ref="retained-1"))
    assert offered.choices[2]["viable"] is False
    result = asyncio.run(router.recover(offered.recovery_id, "retry", request=request(), invoke=invoke))

    assert result.success is False
    assert result.error_class == "recovery_choice_unavailable"
    assert len(calls) == 1


def test_unknown_candidate_cost_class_fails_closed():
    with pytest.raises(ValueError, match="cost_class"):
        candidate(cost_class="paidd")


def test_recovery_failure_does_not_extend_original_offer_expiry(tmp_path):
    now = [1_800_000_000.0]
    router = VisionRouter([candidate()], db_path=tmp_path / "vision.db", wall_clock=lambda: now[0])

    async def invoke(selected, control):
        raise ProviderError("rate_limit", retry_after_seconds=0)

    offered = asyncio.run(router.execute(request(), invoke, retained_media_ref="retained-1"))
    original_expiry = router.store.load_recovery(offered.recovery_id)["expires_at_epoch"]
    now[0] += 100
    recovered = asyncio.run(router.recover(offered.recovery_id, "retry", request=request(), invoke=invoke))

    assert recovered.success is False
    assert router.store.load_recovery(offered.recovery_id)["expires_at_epoch"] == original_expiry



def test_negative_or_nonfinite_price_disables_paid_choice(tmp_path):
    invalid_price = vr.PriceSnapshot(
        captured_at=datetime.now(UTC).isoformat(), source="invalid-test-snapshot",
        input_usd_per_token=Decimal("-0.001"), output_usd_per_token=Decimal("0.002"),
    )
    router = VisionRouter(
        [candidate(), paid_candidate(pricing=invalid_price)],
        db_path=tmp_path / "vision.db", config=paid_config(),
    )

    async def invoke(selected, control):
        raise ProviderError("capability", "permanent")

    result = asyncio.run(router.execute(paid_request(), invoke, retained_media_ref="retained-1"))

    assert result.choices[0]["viable"] is False
    assert result.choices[0]["reason"] == "current_price_estimate_unavailable"
    assert router.store.reservations() == []


def test_metrics_keep_reused_request_ids_separate_by_asset_and_use_logical_success(tmp_path):
    router = VisionRouter([candidate()], db_path=tmp_path / "vision.db")
    calls = 0

    async def invoke(selected, control):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ProviderResponse(content="analysis")
        raise ProviderError("capability", "permanent")

    first = asyncio.run(router.execute(request(request_id="reused", media_hash="a" * 64), invoke))
    second = asyncio.run(router.execute(
        request(request_id="reused", media_hash="b" * 64), invoke, retained_media_ref="retained-b",
    ))

    assert first.success is True
    assert second.success is False
    metrics = router.metrics()
    assert metrics["logical_requests"] == 2
    assert metrics["successful_requests"] == 1
    assert metrics["logical_request_success_rate"] == 0.5
    assert metrics["nonempty_analysis_rate"] == 0.5


def test_paid_approval_expiry_is_distinct_from_free_recovery_expiry(tmp_path):
    now = [1_800_000_000.0]
    paid_router = VisionRouter(
        [paid_candidate()], db_path=tmp_path / "paid.db", config=paid_config(),
        wall_clock=lambda: now[0],
    )

    async def invoke(selected, control):
        raise AssertionError("expired recovery must not dispatch")

    req = paid_request()
    offered = asyncio.run(paid_router.execute(req, invoke, retained_media_ref="retained-paid"))
    paid_router.approve_paid(offered.recovery_id, candidate_id="paid", request=req)
    now[0] += 901
    expired = asyncio.run(paid_router.recover(
        offered.recovery_id, "paid", request=req, invoke=invoke, approval_token="invalid-after-expiry",
    ))
    assert expired.error_class == "recovery_expired"
    assert paid_router.metrics()["paid_approvals_expired"] == 1

    free_router = VisionRouter(
        [candidate()], db_path=tmp_path / "free.db", wall_clock=lambda: now[0],
    )

    async def transient(selected, control):
        raise ProviderError("rate_limit", retry_after_seconds=0)

    free_offer = asyncio.run(free_router.execute(request(), transient, retained_media_ref="retained-free"))
    now[0] += 901
    free_expired = asyncio.run(free_router.recover(
        free_offer.recovery_id, "retry", request=request(), invoke=transient,
    ))
    assert free_expired.error_class == "recovery_expired"
    assert free_router.metrics()["paid_approvals_expired"] == 0


def test_failed_paid_attempt_summary_retains_estimated_cost_status(tmp_path):
    router = VisionRouter([paid_candidate()], db_path=tmp_path / "vision.db", config=paid_config())

    async def invoke(selected, control):
        raise ProviderError("provider_5xx", "injected upstream failure", retry_after_seconds=0)

    req = paid_request()
    offered = asyncio.run(router.execute(req, invoke, retained_media_ref="retained-paid"))
    grant = router.approve_paid(offered.recovery_id, candidate_id="paid", request=req)
    failed = asyncio.run(router.recover(
        offered.recovery_id, "paid", request=req, invoke=invoke, approval_token=grant.token,
    ))

    assert failed.success is False
    summary = [event for event in router.store.events() if event["event_type"] == "vision_request_summary"][-1]
    assert summary["cost_status"] == "estimated"
    assert summary["estimated_cost_usd"] == "1.134"
    assert summary["actual_cost_usd"] is None
    reservation = router.store.reservations()[0]
    assert reservation["status"] == "failed"
    assert reservation["cost_status"] == "estimated"
