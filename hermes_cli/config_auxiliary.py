"""Factories for auxiliary task defaults and shared vision routing defaults."""


def _aux(timeout, *, reasoning_effort=True, **extra):
    """Standard auxiliary-task model block (see DEFAULT_CONFIG["auxiliary"]).

    reasoning_effort=False omits that key (MoA blocks configure depth per slot);
    extra keys are appended after the standard ones.
    """
    d = {"provider": "auto", "model": "", "base_url": "", "api_key": "", "timeout": timeout, "extra_body": {}}
    if reasoning_effort:
        d["reasoning_effort"] = ""
    d.update(extra)
    return d


def vision_routing_defaults() -> dict:
    """Return conservative, legacy-only defaults; no provider is probed or selected."""
    return {
        "mode": "legacy",
        "strategy": "free_first",
        "pinned_candidate": None,
        "disabled_candidates": [],
        "allow": {"hosted_free": True, "local": False, "paid": False},
        "candidates": {},
        "output_cap": 512,
        "hard_output_cap": 1024,
        "recovery": {
            "max_provider_attempts": 3,
            "max_same_route_retries": 2,
            "max_fallback_transitions": 1,
            "automatic_same_route_retries": 1,
            "offer_ttl_seconds": 900,
            "image_attempt_timeout_seconds": 60,
            "image_overall_timeout_seconds": 90,
            "video_attempt_timeout_seconds": 180,
            "video_overall_timeout_seconds": 240,
        },
        "budgets": {
            "scope": "profile", "ledger_path": None,
            "paid_per_request_usd": None, "paid_daily_usd": None, "paid_monthly_usd": None,
        },
    }
