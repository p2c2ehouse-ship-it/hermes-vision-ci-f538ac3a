from hermes_cli.config_defaults import DEFAULT_CONFIG


def test_vision_routing_defaults_fail_closed_and_preserve_legacy():
    routing = DEFAULT_CONFIG["auxiliary"]["vision"]["routing"]

    assert routing["mode"] == "legacy"
    assert routing["strategy"] == "free_first"
    assert routing["candidates"] == {}
    assert routing["allow"] == {"hosted_free": True, "local": False, "paid": False}
    assert routing["pinned_candidate"] is None
    assert routing["disabled_candidates"] == []
    assert routing["budgets"] == {
        "scope": "profile", "ledger_path": None,
        "paid_per_request_usd": None, "paid_daily_usd": None, "paid_monthly_usd": None,
    }


def test_vision_routing_default_recovery_limits_do_not_exceed_fleet_maxima():
    recovery = DEFAULT_CONFIG["auxiliary"]["vision"]["routing"]["recovery"]

    assert recovery["max_provider_attempts"] == 3
    assert recovery["max_same_route_retries"] == 2
    assert recovery["max_fallback_transitions"] == 1
    assert recovery["automatic_same_route_retries"] == 1
    assert recovery["offer_ttl_seconds"] == 900


def test_startup_validation_rejects_credentials_auto_approval_and_flags_clamps():
    from hermes_cli.config import validate_config_structure

    config = {
        "auxiliary": {"vision": {"provider": "auto", "model": "", "routing": {
            "mode": "active", "recovery": {"max_provider_attempts": 9},
            "candidates": {"unsafe": {
                "provider": "openrouter", "model": "example/free:free",
                "declared_media": ["image"], "verified_media": ["image"],
                "api_key": "x", "auto_approve_paid": True,
            }},
        }}},
    }

    issues = validate_config_structure(config)
    assert any(issue.severity == "error" and "credential/approval" in issue.message for issue in issues)
    assert any(issue.severity == "error" and "automatic paid approval" in issue.message for issue in issues)
    assert any(issue.severity == "warning" and "fleet maximum" in issue.message for issue in issues)


def test_active_video_gate_rejects_unverified_local_video_claim():
    from agent.vision_integration import video_route_available

    config = {
        "routing": {
            "mode": "active",
            "allow": {"local": True},
            "candidates": {
                "local": {
                    "provider": "ollama", "model": "local-vision",
                    "declared_media": ["image", "video"],
                    "verified_media": ["image", "video"],
                    "cost_class": "local_no_api_fee",
                },
            },
        },
    }

    assert video_route_available(config) is False


def test_malformed_routing_subsections_return_validation_errors():
    from agent.vision_integration import routing_validation_issues

    config = {
        "routing": {
            "mode": "active",
            "allow": [],
            "recovery": [],
            "budgets": [],
            "disabled_candidates": "not-a-list",
            "candidates": {},
        },
    }

    issues = routing_validation_issues(config)
    messages = "\n".join(message for _, message, _ in issues)
    assert "allow must be a mapping" in messages
    assert "recovery must be a mapping" in messages
    assert "budgets must be a mapping" in messages
    assert "disabled_candidates must be a list" in messages
