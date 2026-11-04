"""Small, startup-safe validators for selected Hermes configuration sections."""
from __future__ import annotations

from typing import Any, Callable


IssueWriter = Callable[[list[Any], str, str, str], None]


def validate_voice(config: dict[str, Any], issues: list[Any], issue: IssueWriter) -> None:
    voice_cfg = config.get("voice")
    if not (isinstance(voice_cfg, dict) and "submit_mode" in voice_cfg):
        return
    submit_mode = voice_cfg.get("submit_mode")
    normalized = submit_mode.strip().lower() if isinstance(submit_mode, str) else None
    if normalized not in {"direct", "draft"}:
        issue(issues, "error", f"voice.submit_mode must be 'direct' or 'draft', got {submit_mode!r}",
              "Set voice.submit_mode to direct (submit immediately) or draft (edit before sending)")


def validate_vision_routing(config: dict[str, Any], issues: list[Any], issue: IssueWriter) -> None:
    auxiliary = config.get("auxiliary")
    vision = auxiliary.get("vision") if isinstance(auxiliary, dict) else None
    if not isinstance(vision, dict) or "routing" not in vision:
        return
    from agent.vision_integration import routing_validation_issues
    for severity, message, fix in routing_validation_issues(vision):
        issue(issues, severity, message, fix)
