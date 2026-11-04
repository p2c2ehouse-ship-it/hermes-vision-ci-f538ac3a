#!/usr/bin/env python3
"""Read-only, secret-free inventory of Hermes auxiliary vision rollout state."""
from __future__ import annotations

from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
import sqlite3
from pathlib import Path

import hermes_yaml as yaml

from agent.vision_integration import media_route_available, routing_validation_issues

HERMES_HOME = Path(os.environ.get("HERMES_FLEET_HOME", "/home/jtpktqadzn/.hermes"))
WORKSPACE = Path(__file__).resolve().parent.parent
LIVE_ROOT = Path(os.environ.get("HERMES_LIVE_ROOT", "/opt/hermes"))
MODULES = (
    "agent/vision_routing.py",
    "agent/vision_integration.py",
    "tools/vision_tools.py",
    "agent/auxiliary_client.py",
    "hermes_cli/config.py",
    "hermes_cli/config_defaults.py",
)


def sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def implementation_status() -> dict:
    found = {}
    for relative in MODULES:
        live = LIVE_ROOT / relative
        scratch = WORKSPACE / relative
        found[relative] = {
            "live_path": str(live),
            "live_present": live.is_file(),
            "scratch_sha256": sha256(scratch),
            "live_sha256": sha256(live),
            "matches_scratch": bool(live.is_file() and sha256(live) == sha256(scratch)),
        }
    return found


def _decimal_sum(values: list[object]) -> str:
    total = Decimal(0)
    for value in values:
        if value is None:
            continue
        try:
            total += Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError):
            continue
    return str(total)


def telemetry_summary(path_value: object, home: Path) -> dict:
    path = Path(path_value).expanduser() if isinstance(path_value, str) and path_value.strip() else home / "vision-routing" / "routing.db"
    if not path.is_absolute():
        path = home / path
    result = {"path_scope": "profile_default" if path == home / "vision-routing" / "routing.db" else "configured", "exists": path.is_file()}
    if not path.is_file():
        return result
    try:
        uri = f"file:{path}?mode=ro"
        with sqlite3.connect(uri, uri=True) as db:
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "events" not in tables:
                result["events_table"] = False
                return result
            rows = db.execute("SELECT event_type, payload_json FROM events").fetchall()
        events = []
        for event_type, payload in rows:
            try:
                events.append((event_type, json.loads(payload)))
            except (TypeError, json.JSONDecodeError):
                continue
        attempts = [payload for kind, payload in events if kind == "vision_attempt"]
        summaries = [payload for kind, payload in events if kind == "vision_request_summary"]
        result.update({
            "events_table": True,
            "event_counts": dict(sorted(Counter(kind for kind, _ in events).items())),
            "provider_attempts": len(attempts),
            "request_summaries": len(summaries),
            "successful_request_summaries": sum(bool(item.get("success")) for item in summaries),
            "estimated_cost_usd_sum": _decimal_sum([item.get("estimated_cost_usd") for item in attempts]),
            "provider_confirmed_cost_usd_sum": _decimal_sum([item.get("actual_cost_usd") for item in attempts]),
            "provider_confirmed_cost_events": sum(item.get("actual_cost_usd") is not None for item in attempts),
            "attempt_cost_status_counts": dict(sorted(Counter(str(item.get("cost_status")) for item in attempts).items())),
        })
    except (OSError, sqlite3.Error) as exc:
        result["read_error"] = type(exc).__name__
    return result


def profile_inventory(config_path: Path, label: str, home: Path) -> dict:
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    except (OSError, yaml.YAMLError) as exc:
        return {"profile": label, "config_present": config_path.exists(), "read_error": type(exc).__name__}
    config = raw if isinstance(raw, dict) else {}
    auxiliary = config.get("auxiliary") if isinstance(config.get("auxiliary"), dict) else {}
    vision = auxiliary.get("vision") if isinstance(auxiliary.get("vision"), dict) else {}
    routing = vision.get("routing") if isinstance(vision.get("routing"), dict) else {}
    raw_candidates = routing.get("candidates")
    if isinstance(raw_candidates, dict):
        candidates = [dict(item, id=candidate_id) for candidate_id, item in raw_candidates.items() if isinstance(item, dict)]
    elif isinstance(raw_candidates, list):
        candidates = [item for item in raw_candidates if isinstance(item, dict)]
    else:
        candidates = []
    safe_candidates = [{
        key: item.get(key) for key in (
            "id", "provider", "model", "cost_class", "declared_media",
            "verified_media", "failed_media",
        ) if key in item
    } for item in candidates]
    mode = routing.get("mode", "legacy")
    issues = routing_validation_issues(vision)
    allow = routing.get("allow") if isinstance(routing.get("allow"), dict) else {}
    telemetry_config = routing.get("telemetry") if isinstance(routing.get("telemetry"), dict) else {}
    try:
        active_image_route = media_route_available(vision, "image") if mode == "active" else False
        active_video_route = media_route_available(vision, "video") if mode == "active" else False
    except (TypeError, ValueError):
        active_image_route = False
        active_video_route = False
    return {
        "profile": label,
        "config_present": config_path.exists(),
        "auxiliary_vision_configured": bool(vision),
        "configured_provider": vision.get("provider"),
        "configured_model": vision.get("model"),
        "mode": mode,
        "candidate_count": len(candidates),
        "candidates": safe_candidates,
        "paid_recovery_allowed": bool(allow.get("paid", False)),
        "active_image_route_available": active_image_route,
        "active_video_route_available": active_video_route,
        "validation_issue_count": len(issues),
        "validation_issues": [message for _, message, _ in issues],
        "telemetry": telemetry_summary(telemetry_config.get("db_path"), home),
    }


def main() -> None:
    profiles = [profile_inventory(HERMES_HOME / "config.yaml", "default", HERMES_HOME)]
    for config_path in sorted((HERMES_HOME / "profiles").glob("*/config.yaml")):
        profiles.append(profile_inventory(config_path, config_path.parent.name, config_path.parent))
    print(json.dumps({
        "read_only": True,
        "secrets_included": False,
        "implementation": implementation_status(),
        "profiles": profiles,
        "migrated_profiles": [p["profile"] for p in profiles if p.get("mode") in {"shadow", "active"}],
        "active_profiles": [p["profile"] for p in profiles if p.get("mode") == "active"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
