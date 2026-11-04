"""Startup-safe validation for shared auxiliary vision routing configuration."""
from __future__ import annotations

from typing import Any, Callable


AddIssue = Callable[[str, str, str], None]
_FORBIDDEN = {"auto_approve", "auto_approve_paid", "approval_token", "paid_approval_token"}


def _shape_issues(routing: dict[str, Any], add: AddIssue) -> bool:
    valid = True
    for key in ("allow", "recovery", "budgets"):
        value = routing.get(key)
        if value is not None and not isinstance(value, dict):
            add("error", f"{key} must be a mapping", f"Set auxiliary.vision.routing.{key} to a YAML mapping.")
            valid = False
    disabled = routing.get("disabled_candidates")
    if disabled is not None and not isinstance(disabled, (list, tuple, set)):
        add("error", "disabled_candidates must be a list", "Use a YAML list of registered candidate IDs.")
        valid = False
    return valid


def _selection_issues(routing: dict[str, Any], add: AddIssue) -> None:
    mode = str(routing.get("mode") or "legacy")
    if mode not in {"legacy", "shadow", "active"}:
        add("error", f"unknown mode {mode!r}", "Use legacy, shadow, or active.")
    if str(routing.get("strategy") or "free_first") not in {"free_first", "quality_first"}:
        add("error", "unknown selection strategy", "Use free_first or quality_first.")
    if _FORBIDDEN.intersection(routing):
        add("error", "automatic paid approval is not a supported configuration",
            "Remove approval/auto-approval fields; use the trusted per-call approval adapter.")


def _quality_issues(candidate_id: Any, item: dict[str, Any], add: AddIssue) -> None:
    if item.get("quality_score") is None:
        return
    try:
        sample_size = int(item.get("quality_sample_size", 0) or 0)
    except (TypeError, ValueError):
        sample_size = 0
    if not item.get("quality_evidence_id") or sample_size <= 0:
        add("error",
            f"candidate {candidate_id!r} has unversioned/unsupported quality evidence or invalid quality_sample_size",
            "Provide quality_evidence_id and a positive numeric quality_sample_size, or remove quality_score.")


def _candidate_item_issues(candidate_id: Any, item: dict[str, Any], mode: str, add: AddIssue) -> None:
    for key in item:
        normalized = str(key).lower().replace("-", "_")
        if (
            normalized in {"token", "credential", "credentials", *_FORBIDDEN}
            or normalized.startswith(("api_key", "secret", "password"))
            or normalized.endswith(("_api_key", "_token", "_secret", "_password"))
        ):
            add("error", f"candidate {candidate_id!r} contains forbidden credential/approval field {key!r}",
                "Remove it and use existing Hermes credential resolution plus trusted per-call approval.")
    if _FORBIDDEN.intersection(item):
        add("error", f"candidate {candidate_id!r} requests automatic paid approval",
            "Remove auto-approval and require a fresh trusted grant.")
    if mode == "legacy":
        return
    if not item.get("provider") or item.get("provider") == "auto" or not item.get("model"):
        add("error", f"candidate {candidate_id!r} lacks an explicit provider/model",
            "Pin a registered provider and model; auto is not selectable.")
    declared = item.get("declared_media", ())
    verified = item.get("verified_media", ())
    if not isinstance(declared, (list, tuple, set)) or not isinstance(verified, (list, tuple, set)):
        add("error", f"candidate {candidate_id!r} media capabilities must be lists",
            "Use declared_media and verified_media YAML lists.")
    elif not set(verified).issubset(set(declared)):
        add("error", f"candidate {candidate_id!r} verifies undeclared media",
            "Verified media must be a subset of declared media.")
    if item.get("cost_class", "hosted_free") not in {"hosted_free", "local_no_api_fee", "paid"}:
        add("error", f"candidate {candidate_id!r} has an unknown cost class",
            "Use hosted_free, local_no_api_fee, or paid.")
    _quality_issues(candidate_id, item, add)


def _candidate_issues(candidates: dict[Any, Any], mode: str, add: AddIssue) -> None:
    for candidate_id, item in candidates.items():
        if not isinstance(item, dict):
            add("error", f"candidate {candidate_id!r} must be a mapping",
                "Replace the scalar/list with a candidate mapping.")
            continue
        _candidate_item_issues(candidate_id, item, mode, add)


def _recovery_issues(recovery: dict[str, Any], add: AddIssue) -> None:
    maxima = {
        "max_provider_attempts": 3, "max_same_route_retries": 2,
        "max_fallback_transitions": 1, "automatic_same_route_retries": 2,
        "offer_ttl_seconds": 900, "image_attempt_timeout_seconds": 60,
        "image_overall_timeout_seconds": 90, "video_attempt_timeout_seconds": 180,
        "video_overall_timeout_seconds": 240,
    }
    for key, maximum in maxima.items():
        if key not in recovery:
            continue
        try:
            value = float(recovery[key])
        except (TypeError, ValueError):
            add("error", f"recovery.{key} must be numeric", f"Set {key} to a number no greater than {maximum}.")
            continue
        if value < 0 or (key.endswith("timeout_seconds") and value <= 0):
            add("error", f"recovery.{key} must be positive/bounded",
                f"Set {key} within 0..{maximum} (timeouts must be positive).")
        elif value > maximum:
            add("warning", f"recovery.{key} exceeds fleet maximum {maximum} and will be clamped",
                f"Set {key} <= {maximum}.")


def _paid_issues(routing: dict[str, Any], candidates: dict[Any, Any], add: AddIssue,
                 boolean: Callable[[Any, bool], bool], decimal: Callable[[Any], Any]) -> None:
    paid = [item for item in candidates.values() if isinstance(item, dict) and item.get("cost_class") == "paid"]
    paid_allowed = boolean((routing.get("allow") or {}).get("paid"), False)
    if not (paid and paid_allowed):
        return
    budgets = routing.get("budgets") or {}
    ceilings = ("paid_per_request_usd", "paid_daily_usd", "paid_monthly_usd")
    if any(decimal(budgets.get(key)) is None or decimal(budgets.get(key)) <= 0 for key in ceilings):
        add("error", "paid candidates require all three positive budget ceilings",
            "Set per-request, daily, and monthly USD ceilings before enabling paid routing.")
    for item in paid:
        if not isinstance(item.get("pricing"), dict):
            add("warning", "an enabled paid candidate has no current price snapshot",
                "Add a sourced timestamped price snapshot or leave paid routing disabled.")


def _active_issues(vision_config: dict[str, Any], routing: dict[str, Any], candidates: dict[Any, Any],
                   add: AddIssue, build_config: Callable[[dict[str, Any]], Any],
                   boolean: Callable[[Any, bool], bool], decimal: Callable[[Any], Any]) -> None:
    _recovery_issues(routing.get("recovery") or {}, add)
    pinned = routing.get("pinned_candidate")
    if pinned and pinned not in candidates:
        add("error", f"pinned candidate {pinned!r} is not registered", "Register the candidate or remove the pin.")
    try:
        parsed = build_config(vision_config)
        if parsed.pinned_candidate == "unregistered-explicit-route":
            add("error", "existing auxiliary vision provider/model does not identify one registered candidate",
                "Register exactly one matching candidate or set pinned_candidate explicitly.")
    except (TypeError, ValueError) as exc:
        add("error", f"invalid routing configuration: {exc}", "Correct the reported routing field.")
    _paid_issues(routing, candidates, add, boolean, decimal)


def routing_validation_issues(vision_config: dict[str, Any], *, build_config: Callable[[dict[str, Any]], Any],
                              boolean: Callable[[Any, bool], bool],
                              decimal: Callable[[Any], Any]) -> list[tuple[str, str, str]]:
    issues: list[tuple[str, str, str]] = []

    def add(severity: str, message: str, fix: str) -> None:
        issues.append((severity, f"auxiliary.vision.routing: {message}", fix))

    routing = vision_config.get("routing")
    if routing is None:
        routing = {}
    if not isinstance(routing, dict):
        add("error", "must be a mapping", "Set auxiliary.vision.routing to a YAML mapping.")
        return issues
    subsections_valid = _shape_issues(routing, add)
    _selection_issues(routing, add)
    mode = str(routing.get("mode") or "legacy")
    candidates = routing.get("candidates")
    if candidates is None:
        candidates = {}
    if not isinstance(candidates, dict):
        add("error", "candidates must be a mapping", "Define candidates as id: candidate-mapping entries.")
        return issues
    _candidate_issues(candidates, mode, add)
    if subsections_valid and mode != "legacy":
        _active_issues(vision_config, routing, candidates, add, build_config, boolean, decimal)
    return issues
