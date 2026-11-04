"""Shared policy-driven routing for auxiliary image and video analysis.

The native main-model image path does not use this module. Provider transports stay in
agent.auxiliary_client; this module owns deterministic selection, bounded attempts,
recovery state, and sanitized telemetry.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, UTC
from decimal import Decimal
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import secrets
import sqlite3
import time
from typing import Any, Awaitable, Callable, Iterable, Optional

logger = logging.getLogger(__name__)
_COST_FREE = {"hosted_free", "local_no_api_fee"}
_TRANSIENT_ERRORS = {"rate_limit", "timeout", "connection", "overloaded", "provider_5xx", "http_408"}


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso_now() -> str:
    return _utc_now().isoformat()


def _json_default(value: Any) -> str:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


@dataclass(frozen=True)
class MediaRequest:
    request_id: str
    profile: str
    session_id: str
    media_kind: str
    media_hash: str
    normalized_prompt_hash: str
    mime_type: str
    byte_count: int
    base64_bytes: int
    output_cap: int
    hard_output_cap: int
    interaction_mode: str = "interactive"
    source_kind: str = "unknown"
    duration_seconds: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    estimated_input_tokens: Optional[int] = None
    input_token_count_exact: bool = False
    input_token_estimate_source: str = "unavailable"
    size_bucket: str = "unknown"
    framework_call_id: Optional[str] = None
    preparation_latency_ms: Optional[float] = None
    download_latency_ms: Optional[float] = None
    pre_routing_elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.media_kind not in {"image", "video"}:
            raise ValueError("media_kind must be image or video")
        if not self.request_id or not self.media_hash or not self.normalized_prompt_hash:
            raise ValueError("request and content fingerprints are required")
        if self.byte_count < 0 or self.base64_bytes < 0:
            raise ValueError("media sizes cannot be negative")
        if self.estimated_input_tokens is not None and self.estimated_input_tokens < 0:
            raise ValueError("estimated input tokens cannot be negative")
        if self.output_cap <= 0 or self.hard_output_cap < self.output_cap:
            raise ValueError("a positive, bounded output cap is required")

    @property
    def fingerprint(self) -> str:
        return f"{self.media_kind}:{self.media_hash}:{self.normalized_prompt_hash}"


@dataclass(frozen=True)
class PriceSnapshot:
    captured_at: str
    source: str
    input_usd_per_token: Optional[Decimal] = None
    output_usd_per_token: Optional[Decimal] = None
    request_usd: Decimal = Decimal(0)
    media_usd: Decimal = Decimal(0)
    max_age_seconds: int = 86400
    included_fees: tuple[str, ...] = ()
    excluded_or_unknown_fees: tuple[str, ...] = ()

    def captured_datetime(self) -> datetime:
        parsed = datetime.fromisoformat(self.captured_at)
        return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


@dataclass(frozen=True)
class CostEstimate:
    amount_usd: Decimal
    method: str
    price_timestamp: str
    price_source: str
    included_fees: tuple[str, ...]
    excluded_or_unknown_fees: tuple[str, ...]


@dataclass(frozen=True)
class ApprovalGrant:
    grant_id: str
    token: str
    expires_at_epoch: float


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    provider: str
    model: str
    base_url: Optional[str] = None
    declared_media: frozenset[str] = field(default_factory=frozenset)
    verified_media: frozenset[str] = field(default_factory=frozenset)
    failed_media: frozenset[str] = field(default_factory=frozenset)
    cost_class: str = "hosted_free"
    disclosure_id: str = "hosted_free_shared_quota_v1"
    privacy_disclosure: Optional[str] = None
    configured_order: int = 0
    current_configured: bool = False
    automatic_fallback: bool = False
    available: bool = True
    unavailable_reason: Optional[str] = None
    pricing: Optional[PriceSnapshot] = None
    quality_evidence_id: Optional[str] = None
    quality_score: Optional[float] = None
    quality_sample_size: int = 0
    credential_pool_scope: str = "default"

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.provider or not self.model:
            raise ValueError("candidate id, provider, and model are required")
        if self.cost_class not in {*_COST_FREE, "paid"}:
            raise ValueError("cost_class must be hosted_free, local_no_api_fee, or paid")
        if self.quality_score is not None and (
            not math.isfinite(self.quality_score) or not 0.0 <= self.quality_score <= 1.0
        ):
            raise ValueError("quality_score must be a finite value between zero and one")
        if self.quality_sample_size < 0:
            raise ValueError("quality_sample_size cannot be negative")


@dataclass(frozen=True)
class RecoveryLimits:
    max_provider_attempts: int = 3
    max_same_route_retries: int = 2
    max_fallback_transitions: int = 1
    automatic_same_route_retries: int = 1
    offer_ttl_seconds: int = 900
    image_attempt_timeout_seconds: float = 60.0
    image_overall_timeout_seconds: float = 90.0
    video_attempt_timeout_seconds: float = 180.0
    video_overall_timeout_seconds: float = 240.0

    def __post_init__(self) -> None:
        if self.max_provider_attempts > 3 or self.max_same_route_retries > 2 or self.max_fallback_transitions > 1:
            raise ValueError("recovery limits cannot exceed fleet maxima")
        if min(self.max_provider_attempts, self.max_same_route_retries, self.max_fallback_transitions) < 0:
            raise ValueError("recovery limits cannot be negative")
        if not 0 <= self.automatic_same_route_retries <= self.max_same_route_retries:
            raise ValueError("automatic retries exceed the same-route limit")
        if not 1 <= self.offer_ttl_seconds <= 900:
            raise ValueError("recovery offer TTL must be between 1 and 900 seconds")
        timeouts = (
            self.image_attempt_timeout_seconds, self.image_overall_timeout_seconds,
            self.video_attempt_timeout_seconds, self.video_overall_timeout_seconds,
        )
        if any(not math.isfinite(value) or value <= 0 for value in timeouts):
            raise ValueError("recovery timeouts must be finite and positive")
        if self.image_attempt_timeout_seconds > 60 or self.image_overall_timeout_seconds > 90:
            raise ValueError("image timeouts cannot exceed fleet maxima")
        if self.video_attempt_timeout_seconds > 180 or self.video_overall_timeout_seconds > 240:
            raise ValueError("video timeouts cannot exceed fleet maxima")


@dataclass(frozen=True)
class BudgetConfig:
    scope: str = "profile"
    ledger_path: Optional[str] = None
    paid_per_request_usd: Optional[Decimal] = None
    paid_daily_usd: Optional[Decimal] = None
    paid_monthly_usd: Optional[Decimal] = None

    def __post_init__(self) -> None:
        if self.scope not in {"profile", "shared"}:
            raise ValueError("budget scope must be profile or shared")
        for value in (self.paid_per_request_usd, self.paid_daily_usd, self.paid_monthly_usd):
            if value is not None and (not value.is_finite() or value < 0):
                raise ValueError("paid budget ceilings must be finite and nonnegative")

    def completely_configured(self) -> bool:
        return all(value is not None for value in (
            self.paid_per_request_usd, self.paid_daily_usd, self.paid_monthly_usd,
        ))


@dataclass(frozen=True)
class RoutingConfig:
    mode: str = "legacy"
    strategy: str = "free_first"
    pinned_candidate: Optional[str] = None
    disabled_candidates: frozenset[str] = field(default_factory=frozenset)
    allow_hosted_free: bool = True
    allow_local: bool = False
    allow_paid: bool = False
    recovery: RecoveryLimits = field(default_factory=RecoveryLimits)
    budgets: BudgetConfig = field(default_factory=BudgetConfig)

    def __post_init__(self) -> None:
        if self.mode not in {"legacy", "shadow", "active"}:
            raise ValueError("routing mode must be legacy, shadow, or active")
        if self.strategy not in {"free_first", "quality_first"}:
            raise ValueError("routing strategy must be free_first or quality_first")


@dataclass(frozen=True)
class RouteControl:
    request_id: str
    attempt_index: int
    fallback_index: int
    max_attempt_seconds: float
    deadline_remaining_seconds: float
    correction: Optional[str] = None
    output_cap: Optional[int] = None
    selected_candidate_id: Optional[str] = None


@dataclass(frozen=True)
class ProviderResponse:
    content: str
    finish_reason: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    actual_cost_usd: Optional[Decimal] = None
    cost_source: Optional[str] = None
    upstream_provider: Optional[str] = None


class ProviderError(Exception):
    """Sanitized provider failure visible to routing policy."""

    def __init__(
        self, error_class: str, message: str = "", *, retry_after_seconds: Optional[float] = None,
        status_code: Optional[int] = None,
    ):
        super().__init__(message or error_class)
        self.error_class = error_class
        self.retry_after_seconds = retry_after_seconds
        self.status_code = status_code

    @property
    def retryable(self) -> bool:
        return self.error_class in _TRANSIENT_ERRORS or self.error_class == "image_size"


@dataclass
class AttemptBudget:
    deadline_monotonic: float
    max_provider_calls: int = 3
    same_route_retries_remaining: int = 2
    fallback_transitions_remaining: int = 1
    provider_calls: int = 0
    same_route_retries_used: int = 0
    fallback_transitions_used: int = 0
    corrections_used: int = 0

    def remaining(self, now: Optional[float] = None) -> float:
        return max(0.0, self.deadline_monotonic - (time.monotonic() if now is None else now))

    def can_call(self, now: Optional[float] = None) -> bool:
        return self.provider_calls < self.max_provider_calls and self.remaining(now) > 0

    def can_retry(self, now: Optional[float] = None) -> bool:
        return self.same_route_retries_remaining > 0 and self.can_call(now)

    def consume_call(self, *, same_route_retry: bool = False, fallback: bool = False) -> None:
        self.provider_calls += 1
        if same_route_retry:
            self.same_route_retries_used += 1
            self.same_route_retries_remaining -= 1
        if fallback:
            self.fallback_transitions_used += 1
            self.fallback_transitions_remaining -= 1


@dataclass(frozen=True)
class Selection:
    selected: Optional[Candidate]
    rejected: tuple[dict[str, str], ...]
    selection_key: str


@dataclass(frozen=True)
class RoutingResult:
    success: bool
    state: str
    analysis: str
    request_id: str
    candidate_id: Optional[str] = None
    error_class: Optional[str] = None
    cost_status: str = "not_applicable_free"
    automatic_retry_allowed: bool = False
    recovery_id: Optional[str] = None
    choices: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["choices"] = list(self.choices)
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RoutingResult:
        data = dict(value)
        data["choices"] = tuple(data.get("choices") or ())
        return cls(**data)


ProviderInvoke = Callable[[Candidate, RouteControl], Awaitable[ProviderResponse]]


def estimate_cost(candidate: Candidate, request: MediaRequest, *, now: Optional[float] = None) -> Optional[CostEstimate]:
    price = candidate.pricing
    if candidate.cost_class != "paid" or price is None:
        return None
    try:
        age = (time.time() if now is None else now) - price.captured_datetime().timestamp()
        if age < -300 or age > price.max_age_seconds or price.max_age_seconds <= 0 or not price.source:
            return None
        fees = (
            price.input_usd_per_token, price.output_usd_per_token, price.request_usd, price.media_usd,
        )
        if any(value is None or not value.is_finite() or value < 0 for value in fees):
            return None
        if price.input_usd_per_token is None or price.output_usd_per_token is None:
            return None
        if request.estimated_input_tokens is None and price.input_usd_per_token != 0:
            return None
        input_tokens = Decimal(request.estimated_input_tokens or 0)
        amount = (
            price.request_usd + price.media_usd
            + input_tokens * price.input_usd_per_token
            + Decimal(request.output_cap) * price.output_usd_per_token
        )
        if not amount.is_finite() or amount < 0:
            return None
    except (TypeError, ValueError, ArithmeticError):
        return None
    return CostEstimate(
        amount_usd=amount,
        method="exact_preflight_cap" if request.input_token_count_exact else "planning_estimate",
        price_timestamp=price.captured_at, price_source=price.source,
        included_fees=price.included_fees, excluded_or_unknown_fees=price.excluded_or_unknown_fees,
    )


def _idempotency_fingerprint(request: MediaRequest) -> str:
    if not request.framework_call_id:
        return request.fingerprint
    return hashlib.sha256(f"{request.framework_call_id}:{request.fingerprint}".encode()).hexdigest()


class SQLiteVisionStore:
    """Sanitized routing state. Raw media, prompts, URLs, and grants never enter it."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.connect() as db:
            db.executescript(
                """CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, event_type TEXT NOT NULL,
                request_id TEXT NOT NULL, payload_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS recoveries (
                recovery_id TEXT PRIMARY KEY, request_id TEXT NOT NULL,
                fingerprint TEXT NOT NULL, state_json TEXT NOT NULL,
                expires_at_epoch REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS idempotency (
                scope TEXT NOT NULL, fingerprint TEXT NOT NULL,
                result_json TEXT NOT NULL, created_at_epoch REAL NOT NULL,
                PRIMARY KEY(scope, fingerprint));
                CREATE TABLE IF NOT EXISTS health (
                candidate_id TEXT NOT NULL, media_kind TEXT NOT NULL, pool_scope TEXT NOT NULL,
                consecutive_failures INTEGER NOT NULL, last_failure_epoch REAL NOT NULL,
                cooldown_until_epoch REAL NOT NULL,
                PRIMARY KEY(candidate_id, media_kind, pool_scope));
                CREATE TABLE IF NOT EXISTS approval_grants (
                grant_id TEXT PRIMARY KEY, token_hash TEXT NOT NULL,
                recovery_id TEXT NOT NULL, media_hash TEXT NOT NULL,
                candidate_id TEXT NOT NULL, estimate_usd TEXT NOT NULL,
                output_cap INTEGER NOT NULL, expires_at_epoch REAL NOT NULL,
                used_at_epoch REAL);
                CREATE TABLE IF NOT EXISTS reservations (
                reservation_id TEXT PRIMARY KEY, grant_id TEXT NOT NULL UNIQUE,
                request_id TEXT NOT NULL, candidate_id TEXT NOT NULL,
                estimated_usd TEXT NOT NULL, actual_usd TEXT,
                cost_status TEXT NOT NULL, status TEXT NOT NULL,
                created_at_epoch REAL NOT NULL);"""
            )

    def connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=5.0)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=5000")
        return db

    def emit(self, event_type: str, request_id: str, payload: dict[str, Any]) -> None:
        event = {"schema_version": 1, "event_timestamp": _iso_now(), "request_id": request_id, **payload}
        encoded = json.dumps(event, sort_keys=True, separators=(",", ":"), default=_json_default)
        with self.connect() as db:
            db.execute(
                "INSERT INTO events(event_type, request_id, payload_json) VALUES (?, ?, ?)",
                (event_type, request_id, encoded),
            )
        logger.info("vision_routing_event %s", encoded)

    def events(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("SELECT event_type, payload_json FROM events ORDER BY id").fetchall()
        return [{"event_type": row["event_type"], **json.loads(row["payload_json"])} for row in rows]

    def save_recovery(self, state: dict[str, Any]) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO recoveries(recovery_id, request_id, fingerprint, state_json, expires_at_epoch) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(recovery_id) DO UPDATE SET "
                "state_json=excluded.state_json, expires_at_epoch=excluded.expires_at_epoch",
                (state["recovery_id"], state["request"]["request_id"], state["request"]["fingerprint"],
                 json.dumps(state, default=_json_default), state["expires_at_epoch"]),
            )

    def load_recovery(self, recovery_id: str) -> Optional[dict[str, Any]]:
        with self.connect() as db:
            row = db.execute("SELECT state_json FROM recoveries WHERE recovery_id=?", (recovery_id,)).fetchone()
        return json.loads(row["state_json"]) if row is not None else None

    def recovery_states(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("SELECT state_json FROM recoveries ORDER BY expires_at_epoch").fetchall()
        return [json.loads(row["state_json"]) for row in rows]

    def close_recovery(self, recovery_id: str, status: str) -> None:
        db = self.connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT state_json FROM recoveries WHERE recovery_id=?", (recovery_id,)).fetchone()
            if row is not None:
                state = json.loads(row["state_json"])
                state["status"] = status
                db.execute(
                    "UPDATE recoveries SET state_json=? WHERE recovery_id=?",
                    (json.dumps(state, default=_json_default), recovery_id),
                )
            db.execute("DELETE FROM approval_grants WHERE recovery_id=?", (recovery_id,))
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()


    def save_idempotent(self, scope: str, fingerprint: str, result: RoutingResult, now: float) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO idempotency(scope, fingerprint, result_json, created_at_epoch) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(scope, fingerprint) DO UPDATE SET result_json=excluded.result_json, "
                "created_at_epoch=excluded.created_at_epoch",
                (scope, fingerprint, json.dumps(result.to_dict(), default=_json_default), now),
            )

    def load_idempotent(self, scope: str, fingerprint: str, now: float) -> Optional[RoutingResult]:
        with self.connect() as db:
            row = db.execute(
                "SELECT result_json, created_at_epoch FROM idempotency WHERE scope=? AND fingerprint=?",
                (scope, fingerprint),
            ).fetchone()
        if row is None or now - float(row["created_at_epoch"]) > 300:
            return None
        return RoutingResult.from_dict(json.loads(row["result_json"]))


    def reservations(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("SELECT * FROM reservations ORDER BY created_at_epoch, reservation_id").fetchall()
        return [dict(row) for row in rows]

    def circuit_state(self, candidate_id: str, media_kind: str, pool_scope: str, now: float) -> str:
        with self.connect() as db:
            row = db.execute(
                "SELECT consecutive_failures, cooldown_until_epoch FROM health "
                "WHERE candidate_id=? AND media_kind=? AND pool_scope=?",
                (candidate_id, media_kind, pool_scope),
            ).fetchone()
        if row is None or int(row["consecutive_failures"]) < 3:
            return "closed"
        return "open" if now < float(row["cooldown_until_epoch"]) else "half_open"

    def record_transient_failure(
        self, candidate_id: str, media_kind: str, pool_scope: str, now: float,
        retry_after_seconds: Optional[float] = None,
    ) -> tuple[bool, bool]:
        db = self.connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM health WHERE candidate_id=? AND media_kind=? AND pool_scope=?",
                (candidate_id, media_kind, pool_scope),
            ).fetchone()
            previous = int(row["consecutive_failures"]) if row is not None else 0
            last = float(row["last_failure_epoch"]) if row is not None else 0.0
            old_cooldown = float(row["cooldown_until_epoch"]) if row is not None else 0.0
            was_open = previous >= 3 and now < old_cooldown
            consecutive = previous + 1 if row is not None and now - last <= 300 else 1
            is_open = consecutive >= 3
            cooldown = now + min(300.0, max(60.0, retry_after_seconds or 0.0)) if is_open else 0.0
            db.execute(
                "INSERT INTO health(candidate_id, media_kind, pool_scope, consecutive_failures, "
                "last_failure_epoch, cooldown_until_epoch) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(candidate_id, media_kind, pool_scope) DO UPDATE SET "
                "consecutive_failures=excluded.consecutive_failures, last_failure_epoch=excluded.last_failure_epoch, "
                "cooldown_until_epoch=excluded.cooldown_until_epoch",
                (candidate_id, media_kind, pool_scope, consecutive, now, cooldown),
            )
            db.commit()
            return is_open, is_open and not was_open
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def record_success(self, candidate_id: str, media_kind: str, pool_scope: str) -> None:
        with self.connect() as db:
            db.execute(
                "DELETE FROM health WHERE candidate_id=? AND media_kind=? AND pool_scope=?",
                (candidate_id, media_kind, pool_scope),
            )


    def create_grant(self, grant: dict[str, Any], token: str) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO approval_grants(grant_id, token_hash, recovery_id, media_hash, candidate_id, "
                "estimate_usd, output_cap, expires_at_epoch) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (grant["grant_id"], hashlib.sha256(token.encode("utf-8")).hexdigest(),
                 grant["recovery_id"], grant["media_hash"], grant["candidate_id"],
                 str(grant["estimate_usd"]), grant["output_cap"], grant["expires_at_epoch"]),
            )

    def unconsumed_grants(self, recovery_id: str) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT grant_id, candidate_id, expires_at_epoch FROM approval_grants "
                "WHERE recovery_id=? AND used_at_epoch IS NULL ORDER BY grant_id",
                (recovery_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def reserved_total(self, since_epoch: float) -> Decimal:
        with self.connect() as db:
            rows = db.execute(
                "SELECT estimated_usd, actual_usd FROM reservations WHERE created_at_epoch >= ?",
                (since_epoch,),
            ).fetchall()
        return sum((Decimal(row["actual_usd"] or row["estimated_usd"]) for row in rows), Decimal(0))

    def consume_grant_and_reserve(
        self, *, grant_id: str, token: str, recovery_id: str, request: MediaRequest,
        candidate: Candidate, estimate: CostEstimate, budgets: BudgetConfig, now: float,
    ) -> str:
        reservation_id = f"vrp_{secrets.token_urlsafe(18)}"
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        db = self.connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            grant = db.execute(
                "SELECT * FROM approval_grants WHERE grant_id=? AND token_hash=?", (grant_id, token_hash),
            ).fetchone()
            valid = (
                grant is not None and grant["used_at_epoch"] is None
                and now < float(grant["expires_at_epoch"])
                and grant["recovery_id"] == recovery_id and grant["media_hash"] == request.media_hash
                and grant["candidate_id"] == candidate.candidate_id
                and Decimal(grant["estimate_usd"]) == estimate.amount_usd
                and int(grant["output_cap"]) == request.output_cap
            )
            if not valid:
                raise ProviderError("paid_approval_invalid")
            if not budgets.completely_configured():
                raise ProviderError("budget_ceilings_unset")
            if estimate.amount_usd > budgets.paid_per_request_usd:
                raise ProviderError("estimate_exceeds_per_request_budget")
            current = datetime.fromtimestamp(now, UTC)
            day_start = current.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
            month_start = current.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()
            for ceiling, since, error_class in (
                (budgets.paid_daily_usd, day_start, "daily_budget_exhausted"),
                (budgets.paid_monthly_usd, month_start, "monthly_budget_exhausted"),
            ):
                rows = db.execute(
                    "SELECT estimated_usd, actual_usd FROM reservations WHERE created_at_epoch >= ?", (since,),
                ).fetchall()
                spent = sum((Decimal(row["actual_usd"] or row["estimated_usd"]) for row in rows), Decimal(0))
                if spent + estimate.amount_usd > ceiling:
                    raise ProviderError(error_class)
            db.execute(
                "INSERT INTO reservations(reservation_id, grant_id, request_id, candidate_id, estimated_usd, "
                "cost_status, status, created_at_epoch) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (reservation_id, grant_id, request.request_id, candidate.candidate_id,
                 str(estimate.amount_usd), "estimated", "reserved", now),
            )
            db.execute("UPDATE approval_grants SET used_at_epoch=? WHERE grant_id=?", (now, grant_id))
            db.commit()
            return reservation_id
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def complete_reservation(self, reservation_id: str, response: Optional[ProviderResponse], status: str) -> None:
        actual = response.actual_cost_usd if response else None
        if actual is not None and (not actual.is_finite() or actual < 0):
            actual = None
        cost_status = "provider_confirmed" if actual is not None else "estimated"
        with self.connect() as db:
            db.execute(
                "UPDATE reservations SET actual_usd=?, cost_status=?, status=? WHERE reservation_id=?",
                (str(actual) if actual is not None else None, cost_status, status, reservation_id),
            )


class VisionRouter:
    def __init__(
        self, candidates: Iterable[Candidate], *, db_path: Path | str,
        config: Optional[RoutingConfig] = None,
        monotonic: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        random_source: Optional[random.Random] = None,
    ):
        self.candidates = tuple(candidates)
        self.config = config or RoutingConfig()
        self.store = SQLiteVisionStore(db_path)
        self.budget_store = (
            SQLiteVisionStore(self.config.budgets.ledger_path)
            if self.config.budgets.scope == "shared" and self.config.budgets.ledger_path
            else self.store
        )
        self.monotonic = monotonic
        self.wall_clock = wall_clock
        self.sleep = sleep
        self.random = random_source or random.Random()
        self._selections: dict[str, Selection] = {}
        self._recovery_choices: dict[str, str] = {}

    def _budget(self, request: MediaRequest) -> AttemptBudget:
        limits = self.config.recovery
        overall = limits.video_overall_timeout_seconds if request.media_kind == "video" else limits.image_overall_timeout_seconds
        return AttemptBudget(
            deadline_monotonic=self.monotonic() + max(0.0, overall - request.pre_routing_elapsed_seconds),
            max_provider_calls=limits.max_provider_attempts,
            same_route_retries_remaining=limits.max_same_route_retries,
            fallback_transitions_remaining=limits.max_fallback_transitions,
        )

    def _evaluate(self, request: MediaRequest, *, exclude: Optional[str] = None) -> Selection:
        rejected: list[dict[str, str]] = []
        eligible: list[Candidate] = []
        candidates = [c for c in self.candidates if c.candidate_id != exclude]
        if self.config.pinned_candidate:
            candidates = [c for c in candidates if c.candidate_id == self.config.pinned_candidate]
        for candidate in candidates:
            reason = None
            if candidate.candidate_id in self.config.disabled_candidates:
                reason = "candidate_disabled"
            elif request.media_kind in candidate.failed_media:
                reason = "capability_failed"
            elif request.media_kind not in candidate.declared_media or request.media_kind not in candidate.verified_media:
                reason = "capability_unverified"
            elif self.store.circuit_state(
                candidate.candidate_id, request.media_kind, candidate.credential_pool_scope, self.wall_clock(),
            ) == "open":
                reason = "circuit_open"
            elif not candidate.available:
                reason = candidate.unavailable_reason or "provider_unavailable"
            elif candidate.cost_class == "local_no_api_fee" and request.media_kind == "video":
                reason = "no_verified_native_video_path"
            elif candidate.cost_class == "hosted_free" and not self.config.allow_hosted_free:
                reason = "hosted_free_disabled"
            elif candidate.cost_class == "local_no_api_fee" and not self.config.allow_local:
                reason = "local_disabled"
            elif candidate.cost_class == "paid" and not self.config.allow_paid:
                reason = "paid_routing_disabled"
            elif candidate.cost_class == "paid":
                # Paid routes may win deterministic selection, but execute stops for approval.
                reason = None
            if reason is not None:
                rejected.append({"candidate_id": candidate.candidate_id, "reason": reason})
            else:
                eligible.append(candidate)

        def quality_key(candidate: Candidate) -> tuple[int, float]:
            measured = (
                candidate.quality_score is not None and candidate.quality_sample_size > 0
                and bool(candidate.quality_evidence_id)
            )
            return (0 if measured else 1, -(candidate.quality_score or 0.0))

        if self.config.strategy == "quality_first":
            key = lambda c: (*quality_key(c), c.cost_class not in _COST_FREE, not c.current_configured, c.configured_order, c.candidate_id)
            order = "quality>cost>current>order"
        else:
            key = lambda c: (c.cost_class not in _COST_FREE, *quality_key(c), not c.current_configured, c.configured_order, c.candidate_id)
            order = "cost>quality>current>order"
        selected = min(eligible, key=key) if eligible else None
        selection_key = f"{self.config.strategy}:{order}" if selected else "none"
        return Selection(selected, tuple(rejected), selection_key)

    def select(self, request: MediaRequest) -> Candidate:
        selected = self._evaluate(request).selected
        if selected is None:
            raise LookupError("no_viable_candidate")
        return selected

    def _record_selection(self, request: MediaRequest, selection: Selection) -> None:
        self._selections[request.request_id] = selection
        self.store.emit("vision_route_selected", request.request_id, {
            "request_id": request.request_id, "profile": request.profile,
            "framework_call_id": request.framework_call_id,
            "media_kind": request.media_kind, "media_hash": request.media_hash,
            "selected_candidate_id": selection.selected.candidate_id if selection.selected else None,
            "rejected_candidates": selection.rejected, "selection_key": selection.selection_key,
            "paid_call_dispatched": False,
        })


    def shadow(self, request: MediaRequest) -> Selection:
        selection = self._evaluate(request)
        self.store.emit("vision_shadow_decision", request.request_id, {
            "request_id": request.request_id, "profile": request.profile,
            "framework_call_id": request.framework_call_id,
            "media_kind": request.media_kind, "media_hash": request.media_hash,
            "selected_candidate_id": selection.selected.candidate_id if selection.selected else None,
            "rejected_candidates": selection.rejected, "selection_key": selection.selection_key,
            "paid_call_dispatched": False,
        })
        return selection

    async def execute(
        self, request: MediaRequest, invoke: ProviderInvoke, *, retained_media_ref: Optional[str] = None,
    ) -> RoutingResult:
        started = self.monotonic()

        scope = f"{request.profile}:{request.session_id}"
        cached = self.store.load_idempotent(scope, _idempotency_fingerprint(request), self.wall_clock())
        if cached is not None:
            self.store.emit("vision_loop_prevention", request.request_id, {
                "request_id": request.request_id, "profile": request.profile,
            "framework_call_id": request.framework_call_id,
                "media_hash": request.media_hash, "cache_hit": True,
            })
            return cached
        selection = self._evaluate(request)
        self._record_selection(request, selection)
        budget = self._budget(request)
        if selection.selected is None:
            return self._finish_failure(
                request, budget, None, "no_viable_candidate", retained_media_ref, started, selection.rejected,
            )
        selected = selection.selected
        correction: Optional[str] = None
        if selected.cost_class == "paid":
            return self._finish_failure(
                request, budget, selected.candidate_id, "paid_approval_required", retained_media_ref, started, selection.rejected,
            )
        automatic_retries = 0
        last_error = "unknown"
        rejected = selection.rejected
        while True:
            try:
                response = await self._call(request, selected, budget, invoke, correction=correction, same_route_retry=budget.provider_calls > 0)
                if response.content and response.finish_reason != "length":
                    return self._finish_success(request, selected, response, started)
                last_error = "empty_content" if not response.content else "truncated_content"
                if budget.corrections_used == 0 and budget.can_retry(self.monotonic()):
                    budget.corrections_used += 1
                    correction = "bounded_output_correction"
                    continue
                break
            except ProviderError as exc:
                last_error = exc.error_class
                if getattr(exc, "circuit_open", False):
                    break
                if (
                    exc.retryable
                    and (exc.error_class != "image_size" or request.media_kind == "image")
                    and automatic_retries < self.config.recovery.automatic_same_route_retries
                    and budget.can_retry(self.monotonic())
                ):
                    automatic_retries += 1
                    if exc.error_class == "image_size":
                        correction = "downscale"
                    await self._backoff(exc, budget, automatic_retries)
                    continue
                break
        return self._finish_failure(
            request, budget, selected.candidate_id, last_error, retained_media_ref, started, rejected,
            correction_used=budget.corrections_used > 0,
        )

    async def _backoff(self, exc: ProviderError, budget: AttemptBudget, retry_number: int) -> None:
        remaining = budget.remaining(self.monotonic())
        if remaining <= 0:
            return
        if exc.retry_after_seconds is not None:
            delay = max(0.0, float(exc.retry_after_seconds))
        else:
            delay = self.random.random() * (2.0 if retry_number == 1 else 8.0)
        if delay > 0:
            await self.sleep(min(delay, remaining))

    async def _call(
        self, request: MediaRequest, candidate: Candidate, budget: AttemptBudget, invoke: ProviderInvoke, *,
        correction: Optional[str] = None, same_route_retry: bool = False, fallback: bool = False, reservation_id: Optional[str] = None,
    ) -> ProviderResponse:
        if not budget.can_call(self.monotonic()):
            raise ProviderError("deadline_exhausted" if budget.remaining(self.monotonic()) <= 0 else "attempt_budget_exhausted")
        if same_route_retry and budget.same_route_retries_remaining <= 0:
            raise ProviderError("same_route_retry_budget_exhausted")
        if fallback and budget.fallback_transitions_remaining <= 0:
            raise ProviderError("fallback_budget_exhausted")
        attempt_index = budget.provider_calls + 1
        fallback_index = budget.fallback_transitions_used + (1 if fallback else 0)
        budget.consume_call(same_route_retry=same_route_retry, fallback=fallback)
        max_attempt = (
            self.config.recovery.video_attempt_timeout_seconds if request.media_kind == "video"
            else self.config.recovery.image_attempt_timeout_seconds
        )
        remaining = budget.remaining(self.monotonic())
        output_cap = request.output_cap
        if correction == "bounded_output_correction":
            output_cap = min(request.hard_output_cap, max(request.output_cap + 1, request.output_cap * 2))
        control = RouteControl(
            request.request_id, attempt_index, fallback_index, min(max_attempt, remaining), remaining,
            correction=correction, output_cap=output_cap, selected_candidate_id=candidate.candidate_id,
        )
        reservation_id = locals().get("reservation_id")
        started = self.monotonic()
        try:
            response = await asyncio.wait_for(
                invoke(candidate, control), timeout=min(max_attempt, remaining),
            )
        except TimeoutError as exc:
            error = ProviderError("timeout", "provider attempt timed out")
            health_state = self._record_failure_health(request, candidate, error)
            self._record_attempt(request, candidate, control, None, error, started, reservation_id, health_state)
            raise error from exc
        except ProviderError as exc:
            if reservation_id:
                self.budget_store.complete_reservation(reservation_id, None, "failed")
            health_state = self._record_failure_health(request, candidate, exc)
            self._record_attempt(request, candidate, control, None, exc, started, reservation_id, health_state)
            raise
        self.store.record_success(candidate.candidate_id, request.media_kind, candidate.credential_pool_scope)
        self._record_attempt(request, candidate, control, response, None, started, reservation_id)
        return response

    async def _invoke(self, candidate: Candidate, control: RouteControl) -> ProviderResponse:
        return await self._active_invoke(candidate, control)

    def _record_failure_health(self, request: MediaRequest, candidate: Candidate, error: ProviderError) -> str:
        if error.error_class not in _TRANSIENT_ERRORS:
            return "closed"
        is_open, opened_now = self.store.record_transient_failure(
            candidate.candidate_id, request.media_kind, candidate.credential_pool_scope,
            self.wall_clock(), error.retry_after_seconds,
        )
        error.circuit_open = is_open
        if opened_now:
            self.store.emit("vision_circuit_open", request.request_id, {
                "request_id": request.request_id, "profile": request.profile,
            "framework_call_id": request.framework_call_id,
                "candidate_id": candidate.candidate_id, "media_kind": request.media_kind,
                "credential_pool_scope": candidate.credential_pool_scope,
                "cooldown_seconds": min(300.0, max(60.0, error.retry_after_seconds or 0.0)),
            })
        return "open" if is_open else "closed"


    def _record_attempt(
        self, request: MediaRequest, candidate: Candidate, control: RouteControl,
        response: Optional[ProviderResponse], error: Optional[ProviderError], started: float, reservation_id: Optional[str] = None,
        health_state: str = "closed",
    ) -> None:
        elapsed = round((self.monotonic() - started) * 1000, 3)
        estimate = None if candidate.cost_class in _COST_FREE else estimate_cost(candidate, request, now=self.wall_clock())
        budget_remaining = None
        if estimate is not None and self.config.budgets.completely_configured():
            budget_remaining, _ = self._budget_status(estimate)
        selection = self._selections.get(request.request_id)
        payload = {
            "request_id": request.request_id, "profile": request.profile,
            "framework_call_id": request.framework_call_id,
            "session_id": request.session_id, "media_kind": request.media_kind,
            "size_bucket": request.size_bucket,
            "media_hash": request.media_hash, "mime_type": request.mime_type,
            "source_kind": request.source_kind, "bytes": request.byte_count,
            "base64_bytes": request.base64_bytes, "duration_seconds": request.duration_seconds,
            "width": request.width, "height": request.height, "size_bucket": request.size_bucket,
            "attempt_index": control.attempt_index, "fallback_index": control.fallback_index,
            "configured_candidate_id": self.config.pinned_candidate,
            "candidate_id": candidate.candidate_id, "provider": candidate.provider,
            "model": candidate.model, "upstream_provider": response.upstream_provider if response else None,
            "selection_reason": self.config.strategy,
            "selection_key": selection.selection_key if selection else "recovery",
            "rejected_candidates": list(selection.rejected) if selection else [],
            "capability_state": "declared_and_verified",
            "health_state": health_state, "quality_evidence_id": candidate.quality_evidence_id or "unknown",
            "quality_sample_size": candidate.quality_sample_size,
            "preparation_latency_ms": request.preparation_latency_ms,
            "download_latency_ms": request.download_latency_ms,
            "provider_latency_ms": elapsed,
            "total_latency_ms": elapsed + (request.preparation_latency_ms or 0),
            "timeout_stage": "provider" if error and error.error_class == "timeout" else None,
            "cancellation_stage": "provider" if error and error.error_class == "cancelled" else None,
            "deadline_remaining_seconds": round(control.deadline_remaining_seconds, 6),
            "http_status": error.status_code if error else None,
            "error_class": error.error_class if error else None,
            "retry_after_seconds": error.retry_after_seconds if error else None,
            "finish_reason": response.finish_reason if response else None,
            "empty_content": not bool(response and response.content),
            "retry_disposition": "bounded_by_shared_budget",
            "input_tokens": response.input_tokens if response else None,
            "output_tokens": response.output_tokens if response else None,
            "output_cap": control.output_cap,
            "price_snapshot_timestamp": candidate.pricing.captured_at if candidate.pricing else None,
            "estimated_cost_usd": "0" if candidate.cost_class in _COST_FREE else str(estimate.amount_usd) if estimate else None,
            "actual_cost_usd": response.actual_cost_usd if response else None,
            "cost_status": (
                "not_applicable_free" if candidate.cost_class in _COST_FREE
                else "provider_confirmed" if response and response.actual_cost_usd is not None else "estimated"
            ),
            "cost_source": response.cost_source if response else None,
            "budget_reservation_id": reservation_id, "budget_reservation_status": "reserved" if reservation_id else "not_required",
            "budget_remaining": budget_remaining,
            "outcome": "nonempty_analysis" if response and response.content else "transport_success" if response else "failed_attempt",
        }
        self.store.emit("vision_attempt", request.request_id, payload)

    def _finish_success(
        self, request: MediaRequest, candidate: Candidate, response: ProviderResponse, started: float,
    ) -> RoutingResult:
        cost_status = "not_applicable_free" if candidate.cost_class in _COST_FREE else (
            "provider_confirmed" if response.actual_cost_usd is not None else "estimated"
        )
        result = RoutingResult(
            True, "succeeded", response.content, request.request_id,
            candidate_id=candidate.candidate_id, cost_status=cost_status,
        )
        self._summary(request, result, started, candidate=candidate, response=response)
        self.store.save_idempotent(f"{request.profile}:{request.session_id}", _idempotency_fingerprint(request), result, self.wall_clock())
        return result

    def _disabled(self, choice: str, reason: str) -> dict[str, Any]:
        value = {"choice": choice, "viable": False, "reason": reason, "disclosures": []}
        if choice == "retry":
            value["likelihood"] = "unknown"
            value["likelihood_rationale"] = "No eligible same-stratum first-retry sample is available."
        if choice == "paid":
            value["approval_required"] = True
        return value

    def _budget_status(self, estimate: CostEstimate) -> tuple[dict[str, Optional[str]], Optional[str]]:
        budgets = self.config.budgets
        now = datetime.fromtimestamp(self.wall_clock(), UTC)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()
        daily = budgets.paid_daily_usd - self.budget_store.reserved_total(day_start)
        monthly = budgets.paid_monthly_usd - self.budget_store.reserved_total(month_start)
        remaining = {
            "per_request_usd": str(budgets.paid_per_request_usd),
            "daily_usd": str(max(Decimal(0), daily)),
            "monthly_usd": str(max(Decimal(0), monthly)),
        }
        if estimate.amount_usd > budgets.paid_per_request_usd:
            return remaining, "estimate_exceeds_per_request_budget"
        if estimate.amount_usd > daily:
            return remaining, "daily_budget_exhausted"
        if estimate.amount_usd > monthly:
            return remaining, "monthly_budget_exhausted"
        return remaining, None

    def _paid_choice(self, request: MediaRequest, preferred_candidate_id: Optional[str] = None) -> dict[str, Any]:
        paid_candidates = sorted(
            (c for c in self.candidates if c.cost_class == "paid"),
            key=lambda c: (c.configured_order, c.candidate_id),
        )
        eligible = [
            c for c in paid_candidates
            if request.media_kind in c.declared_media and request.media_kind in c.verified_media
            and request.media_kind not in c.failed_media and c.available
            and c.candidate_id not in self.config.disabled_candidates
            and self.store.circuit_state(
                c.candidate_id, request.media_kind, c.credential_pool_scope, self.wall_clock(),
            ) != "open"
        ]
        if not self.config.allow_paid:
            return self._disabled("paid", "paid_routing_disabled")
        if not paid_candidates or not eligible:
            return self._disabled("paid", "no_paid_candidate")
        if not self.config.budgets.completely_configured():
            return self._disabled("paid", "budget_ceilings_unset")
        if self.config.budgets.scope == "shared" and not self.config.budgets.ledger_path:
            return self._disabled("paid", "shared_ledger_path_required")
        candidate = next((c for c in eligible if c.candidate_id == preferred_candidate_id), eligible[0])
        estimate = estimate_cost(candidate, request, now=self.wall_clock())
        if estimate is None:
            return self._disabled("paid", "current_price_estimate_unavailable")
        remaining, reason = self._budget_status(estimate)
        if reason is not None:
            return self._disabled("paid", reason)
        return {
            "choice": "paid", "viable": True, "reason": "current_estimate_and_budget_available",
            "candidate_id": candidate.candidate_id, "provider": candidate.provider, "model": candidate.model,
            "media_kind": request.media_kind, "bytes": request.byte_count,
            "duration_seconds": request.duration_seconds, "width": request.width, "height": request.height,
            "output_cap": request.output_cap, "estimated_cost_usd": str(estimate.amount_usd),
            "estimate_method": estimate.method,
            "input_token_count_exact": request.input_token_count_exact,
            "input_token_estimate_source": request.input_token_estimate_source,
            "price_timestamp": estimate.price_timestamp,
            "price_source": estimate.price_source, "included_fees": list(estimate.included_fees),
            "excluded_or_unknown_fees": list(estimate.excluded_or_unknown_fees),
            "budget_remaining": remaining, "approval_required": True,
            "disclosures": [
                f"Media leaves the device for {candidate.provider}; review its data-use and retention terms.",
                "Capability and recognition quality can differ from the original provider.",
                "actual provider billing can differ from this estimate; listed exclusions are not included.",
                f"The output cap is {request.output_cap} tokens; it limits generation, not all provider fees.",
                *([candidate.privacy_disclosure] if candidate.privacy_disclosure else []),
            ],
        }

    def _close_recovery(self, recovery_id: str, status: str) -> None:
        self.store.close_recovery(recovery_id, status)
        if self.budget_store is not self.store:
            self.budget_store.close_recovery(recovery_id, status)

    def _state_budget(self, state: dict[str, Any]) -> AttemptBudget:
        budget_data = dict(state["budget"])
        remaining = max(0.0, float(state.get("deadline_wall_epoch", state["expires_at_epoch"])) - self.wall_clock())
        budget_data["deadline_monotonic"] = self.monotonic() + remaining
        return AttemptBudget(**budget_data)

    def _retry_likelihood(
        self, candidate: Candidate, request: MediaRequest, error_class: str,
    ) -> dict[str, Any]:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for event in self.store.events():
            if event["event_type"] == "vision_attempt":
                grouped.setdefault((event["request_id"], str(event.get("media_hash") or "")), []).append(event)
        cutoff = self.wall_clock() - 30 * 86400
        outcomes: list[tuple[bool, float]] = []
        for attempts in grouped.values():
            first = next((event for event in attempts if event.get("attempt_index") == 1), None)
            second = next((event for event in attempts if event.get("attempt_index") == 2), None)
            if first is None or second is None:
                continue
            expected = (
                candidate.candidate_id, candidate.provider, candidate.model,
                request.media_kind, request.size_bucket, error_class,
            )
            observed = (
                first.get("candidate_id"), first.get("provider"), first.get("model"),
                first.get("media_kind"), first.get("size_bucket"), first.get("error_class"),
            )
            if observed != expected or second.get("fallback_index") != 0:
                continue
            if any(second.get(field) != first.get(field) for field in (
                "candidate_id", "provider", "model", "media_kind", "size_bucket",
            )):
                continue
            try:
                timestamp = datetime.fromisoformat(str(second["event_timestamp"])).timestamp()
            except (KeyError, TypeError, ValueError):
                continue
            if timestamp < cutoff or timestamp > self.wall_clock() + 300:
                continue
            outcomes.append((second.get("outcome") == "nonempty_analysis", timestamp))
        denominator = len(outcomes)
        numerator = sum(success for success, _ in outcomes)
        failures = denominator - numerator
        base = {
            "success_definition": "A nonempty analysis, not a recognition-accuracy probability.",
            "evidence_source": "vision_attempt_telemetry",
            "route_candidate_id": candidate.candidate_id, "provider": candidate.provider,
            "model": candidate.model, "media_kind": request.media_kind,
            "error_class": error_class, "size_bucket": request.size_bucket,
            "observation_window_days": 30, "eligible_sample_size": denominator,
        }
        if denominator < 30 or numerator < 5 or failures < 5:
            return {
                **base, "likelihood": "unknown",
                "likelihood_rationale": (
                    f"Eligible same-stratum first-retry outcomes: {denominator}; successes: {numerator}; "
                    f"failures: {failures}. At least 30 outcomes with 5 successes and 5 failures "
                    "inside the 30-day window are required."
                ),
            }
        probability = numerator / denominator
        z = 1.96
        denominator_adjusted = 1.0 + z * z / denominator
        center = (probability + z * z / (2 * denominator)) / denominator_adjusted
        half = z * math.sqrt(
            probability * (1 - probability) / denominator + z * z / (4 * denominator * denominator)
        ) / denominator_adjusted
        timestamps = [timestamp for _, timestamp in outcomes]
        return {
            **base, "likelihood": probability, "numerator": numerator, "denominator": denominator,
            "wilson_95_interval": [max(0.0, center - half), min(1.0, center + half)],
            "likelihood_rationale": f"{numerator}/{denominator} eligible same-stratum first retries returned a nonempty analysis.",
            "observation_started_at": datetime.fromtimestamp(min(timestamps), UTC).isoformat(),
            "observation_ended_at": datetime.fromtimestamp(max(timestamps), UTC).isoformat(),
        }


    def _choices(
        self, request: MediaRequest, budget: AttemptBudget, candidate_id: Optional[str], error_class: str,
        retained_media_ref: Optional[str], correction_used: bool,
    ) -> tuple[dict[str, Any], ...]:
        paid = self._paid_choice(request, candidate_id)
        free = self._disabled("free", "no_free_fallback_candidate")
        retry = self._disabled("retry", "error_not_retryable")
        fallback_selection = self._evaluate(request, exclude=candidate_id)
        fallback = fallback_selection.selected
        if fallback is not None and fallback.cost_class in _COST_FREE:
            if budget.fallback_transitions_remaining <= 0 or not budget.can_call(self.monotonic()):
                free["reason"] = "fallback_or_attempt_budget_exhausted"
            else:
                disclosures = (
                    ["shared quotas and rate limits can recur; availability is not guaranteed.",
                     f"Media leaves the device for {fallback.provider}; capability and quality can differ."]
                    if fallback.cost_class == "hosted_free" else
                    ["No per-call API fee; hardware, storage, electricity, setup, latency, and quality costs remain.",
                     "Local recovery is image-only; no verified native-video path is provided."]
                )
                if fallback.privacy_disclosure:
                    disclosures.append(fallback.privacy_disclosure)
                free = {
                    "choice": "free", "viable": True, "reason": "eligible_fallback",
                    "candidate_id": fallback.candidate_id, "provider": fallback.provider,
                    "model": fallback.model, "cost_class": fallback.cost_class,
                    "disclosures": disclosures, "fallback_transitions_remaining": budget.fallback_transitions_remaining,
                }
        if fallback is None and any(
            item.get("reason") == "no_verified_native_video_path" for item in fallback_selection.rejected
        ):
            free["reason"] = "no_verified_native_video_path"
        retryable = error_class in _TRANSIENT_ERRORS
        original = next((c for c in self.candidates if c.candidate_id == candidate_id), None)
        if original is not None and retryable:
            if budget.same_route_retries_remaining <= 0:
                retry["reason"] = "same_route_retry_budget_exhausted"
            elif self.store.circuit_state(
                original.candidate_id, request.media_kind, original.credential_pool_scope, self.wall_clock(),
            ) == "open":
                retry["reason"] = "circuit_open"
            elif not budget.can_call(self.monotonic()):
                retry["reason"] = "provider_attempt_or_deadline_budget_exhausted"
            else:
                retry.update({
                    "viable": True, "reason": "transient_failure_with_budget_remaining",
                    "candidate_id": original.candidate_id, "provider": original.provider, "model": original.model,
                    "likelihood": "unknown",
                    "likelihood_rationale": "Eligible same-stratum first-retry outcomes: 0; at least 30 with 5 successes and 5 failures are required.",
                    "success_definition": "A nonempty analysis, not recognition accuracy.",
                })
        if original is not None and retryable:
            retry.update(self._retry_likelihood(original, request, error_class))
        if correction_used and error_class in {"empty_content", "truncated_content"}:
            retry["reason"] = "bounded_correction_already_consumed"
        choices = [paid, free, retry]
        if not retained_media_ref:
            for choice in choices:
                choice["viable"] = False
                choice["reason"] = "secure_media_retention_unavailable"
        return tuple(choices)

    def _finish_failure(
        self, request: MediaRequest, budget: AttemptBudget, candidate_id: Optional[str], error_class: str,
        retained_media_ref: Optional[str], started: float,
        rejected: Iterable[dict[str, str]] = (), *, correction_used: bool = False,
        recovery_id: Optional[str] = None, attempt_candidate: Optional[Candidate] = None,
    ) -> RoutingResult:
        choices = self._choices(request, budget, candidate_id, error_class, retained_media_ref, correction_used)
        viable = any(choice["viable"] for choice in choices)
        recovery_id = recovery_id or (f"vr_{secrets.token_urlsafe(18)}" if viable else None)
        attempted_paid = attempt_candidate is not None and attempt_candidate.cost_class == "paid"
        result = RoutingResult(
            False, "recovery_required" if viable else "terminal_failed",
            f"Vision analysis failed ({error_class}). No analysis was substituted.",
            request.request_id, candidate_id=candidate_id, error_class=error_class,
            cost_status="estimated" if attempted_paid else "not_applicable_free",
            recovery_id=recovery_id, choices=choices,
        )
        if recovery_id:
            previous = self.store.load_recovery(recovery_id)
            now = self.wall_clock()
            state = {
                "recovery_id": recovery_id, "request": {**asdict(request), "fingerprint": request.fingerprint},
                "budget": asdict(budget), "candidate_id": candidate_id, "error_class": error_class,
                "retained_media_ref": retained_media_ref, "correction_used": correction_used,
                "rejected": list(rejected),
                "created_at_epoch": (previous or {}).get("created_at_epoch", now),
                "deadline_wall_epoch": now + budget.remaining(self.monotonic()),
                "expires_at_epoch": (previous or {}).get(
                    "expires_at_epoch", now + self.config.recovery.offer_ttl_seconds,
                ),
                "status": "offered",
            }
            self.store.save_recovery(state)
            if not viable:
                self._close_recovery(recovery_id, "terminal_failed")
        self._summary(request, result, started, candidate=attempt_candidate)
        self.store.save_idempotent(f"{request.profile}:{request.session_id}", _idempotency_fingerprint(request), result, self.wall_clock())
        return result

    def approve_paid(self, recovery_id: str, *, candidate_id: str, request: MediaRequest) -> ApprovalGrant:
        state = self.store.load_recovery(recovery_id)
        if state is None or state.get("status") != "offered":
            raise ProviderError("recovery_not_found_or_closed")
        if self.wall_clock() >= float(state["expires_at_epoch"]):
            raise ProviderError("recovery_expired")
        if state["request"]["fingerprint"] != request.fingerprint:
            raise ProviderError("recovery_fingerprint_mismatch")
        budget = self._state_budget(state)
        paid = next(choice for choice in self._choices(
            request, budget, state.get("candidate_id"), state["error_class"],
            state.get("retained_media_ref"), bool(state.get("correction_used")),
        ) if choice["choice"] == "paid")
        if not paid["viable"] or paid.get("candidate_id") != candidate_id:
            raise ProviderError("paid_choice_unavailable", paid["reason"])
        grant_id = f"vra_{secrets.token_urlsafe(18)}"
        secret = secrets.token_urlsafe(32)
        expires = min(self.wall_clock() + 60.0, float(state["expires_at_epoch"]))
        self.budget_store.create_grant({
            "grant_id": grant_id, "recovery_id": recovery_id, "media_hash": request.media_hash,
            "candidate_id": candidate_id, "estimate_usd": paid["estimated_cost_usd"],
            "output_cap": request.output_cap, "expires_at_epoch": expires,
        }, secret)
        return ApprovalGrant(grant_id, f"{grant_id}.{secret}", expires)


    async def recover(
        self, recovery_id: str, choice: str, *, request: MediaRequest, invoke: ProviderInvoke,
        approval_token: Optional[str] = None,
    ) -> RoutingResult:
        state = self.store.load_recovery(recovery_id)
        if state is None:
            return RoutingResult(False, "terminal_failed", "Recovery state was not found.", request.request_id,
                                 error_class="recovery_not_found")
        if state.get("status") != "offered" or self.wall_clock() >= float(state["expires_at_epoch"]):
            self.store.emit("vision_recovery_expired", request.request_id, {
                "request_id": request.request_id, "profile": request.profile, "recovery_id": recovery_id,
            })
            for grant in self.budget_store.unconsumed_grants(recovery_id):
                self.store.emit("vision_paid_approval_expired", request.request_id, {
                    "request_id": request.request_id, "profile": request.profile,
                    "recovery_id": recovery_id, "grant_id": grant["grant_id"],
                    "candidate_id": grant["candidate_id"],
                })
            result = RoutingResult(False, "terminal_failed", "Recovery state expired.", request.request_id,
                                   error_class="recovery_expired")
            self._close_recovery(recovery_id, "expired")
            self.store.save_idempotent(f"{request.profile}:{request.session_id}", _idempotency_fingerprint(request), result, self.wall_clock())
            return result
        if state["request"]["fingerprint"] != request.fingerprint:
            return RoutingResult(False, "terminal_failed", "Recovery media fingerprint mismatch.", request.request_id,
                                 recovery_id=recovery_id, error_class="recovery_fingerprint_mismatch")
        if not state.get("retained_media_ref"):
            return RoutingResult(False, "terminal_failed", "Secure media retention is unavailable.", request.request_id,
                                 recovery_id=recovery_id, error_class="secure_media_retention_unavailable")
        budget = self._state_budget(state)
        candidate_id = state.get("candidate_id")
        offered_choices = {
            item["choice"]: item for item in self._choices(
                request, budget, candidate_id, state["error_class"], state.get("retained_media_ref"),
                bool(state.get("correction_used")),
            )
        }
        if choice in {"paid", "free", "retry"} and not offered_choices[choice]["viable"]:
            if not any(item["viable"] for item in offered_choices.values()):
                self._close_recovery(recovery_id, "terminal_failed")
            return RoutingResult(
                False, "terminal_failed", "Recovery choice is unavailable.", request.request_id,
                recovery_id=recovery_id,
                error_class=(offered_choices[choice]["reason"] if choice == "paid" else "recovery_choice_unavailable"),
            )
        if choice == "retry":
            selected = next((c for c in self.candidates if c.candidate_id == candidate_id), None)
            same_route_retry = True
            fallback = False
        elif choice == "free":
            offered = next((c for c in self._choices(
                request, budget, candidate_id, state["error_class"], state.get("retained_media_ref"),
                bool(state.get("correction_used")),
            ) if c["choice"] == "free"), None)
            selected = next((c for c in self.candidates if offered and c.candidate_id == offered.get("candidate_id")), None)
            same_route_retry = False
            fallback = True
        elif choice == "paid":
            offered = next((c for c in self._choices(
                request, budget, candidate_id, state["error_class"], state.get("retained_media_ref"),
                bool(state.get("correction_used")),
            ) if c["choice"] == "paid"), None)
            selected = next((c for c in self.candidates if offered and c.candidate_id == offered.get("candidate_id")), None)
            same_route_retry = False
            fallback = True
            if not approval_token:
                self.store.emit("vision_paid_approval_denied", request.request_id, {
                    "request_id": request.request_id, "profile": request.profile,
                    "recovery_id": recovery_id, "candidate_id": selected.candidate_id if selected else None,
                })
                self._close_recovery(recovery_id, "denied")
                return RoutingResult(False, "terminal_failed", "A current trusted approval grant is required.",
                                     request.request_id, recovery_id=recovery_id, error_class="paid_approval_required")
            if selected is None or not offered or not offered.get("viable"):
                return RoutingResult(False, "terminal_failed", "Paid recovery is unavailable.", request.request_id,
                                     recovery_id=recovery_id,
                                     error_class=(offered or {}).get("reason", "paid_choice_unavailable"))
            estimate = estimate_cost(selected, request, now=self.wall_clock())
            grant_id, separator, secret = approval_token.partition(".")
            if not separator or estimate is None:
                return RoutingResult(False, "terminal_failed", "Paid approval or estimate is invalid.", request.request_id,
                                     recovery_id=recovery_id, error_class="paid_approval_invalid")
            try:
                reservation_id = self.budget_store.consume_grant_and_reserve(
                    grant_id=grant_id, token=secret, recovery_id=recovery_id, request=request,
                    candidate=selected, estimate=estimate, budgets=self.config.budgets, now=self.wall_clock(),
                )
            except ProviderError as exc:
                return RoutingResult(False, "terminal_failed", "Paid reservation was rejected.", request.request_id,
                                     recovery_id=recovery_id, error_class=exc.error_class)
        else:
            return RoutingResult(False, "terminal_failed", "Recovery choice is unavailable.", request.request_id,
                                 recovery_id=recovery_id, error_class="recovery_choice_unavailable")
        if selected is None or not budget.can_call(self.monotonic()):
            return self._finish_failure(
                request, budget, candidate_id, "recovery_choice_unavailable", state.get("retained_media_ref"),
                self.monotonic(), recovery_id=recovery_id, correction_used=bool(state.get("correction_used")),
            )
        fallback = selected is not None and selected.candidate_id != candidate_id
        reservation_id = locals().get("reservation_id")
        self._recovery_choices[request.request_id] = choice
        self.store.emit("vision_recovery_choice", request.request_id, {
            "request_id": request.request_id, "profile": request.profile,
            "recovery_id": recovery_id, "choice": choice,
            "candidate_id": selected.candidate_id, "media_kind": request.media_kind,
        })
        started = self.monotonic()
        try:
            response = await self._call(
                request, selected, budget, invoke, same_route_retry=same_route_retry, fallback=fallback, reservation_id=reservation_id,
            )
            if not response.content or response.finish_reason == "length":
                raise ProviderError("empty_content" if not response.content else "truncated_content")
        except ProviderError as exc:
            if reservation_id:
                self.budget_store.complete_reservation(reservation_id, None, "failed")
            return self._finish_failure(
                request, budget, candidate_id, exc.error_class, state.get("retained_media_ref"), started,
                recovery_id=recovery_id, correction_used=bool(state.get("correction_used")),
                attempt_candidate=selected,
            )
        if reservation_id:
            self.budget_store.complete_reservation(reservation_id, response, "completed")
        result = self._finish_success(request, selected, response, started)
        self._close_recovery(recovery_id, "succeeded")
        return result

    def _summary(
        self, request: MediaRequest, result: RoutingResult, started: float,
        candidate: Optional[Candidate] = None, response: Optional[ProviderResponse] = None,
    ) -> None:
        estimate = None if candidate is None or candidate.cost_class in _COST_FREE else estimate_cost(
            candidate, request, now=self.wall_clock(),
        )
        self.store.emit("vision_request_summary", request.request_id, {
            "request_id": request.request_id, "profile": request.profile,
            "framework_call_id": request.framework_call_id,
            "session_id": request.session_id, "media_kind": request.media_kind,
            "size_bucket": request.size_bucket,
            "media_hash": request.media_hash, "candidate_id": result.candidate_id,
            "state": result.state, "success": result.success,
            "total_latency_ms": round((self.monotonic() - started) * 1000, 3),
            "estimated_cost_usd": (
                "0" if result.cost_status == "not_applicable_free"
                else str(estimate.amount_usd) if estimate is not None else None
            ),
            "actual_cost_usd": response.actual_cost_usd if response else None,
            "cost_status": result.cost_status, "cost_source": response.cost_source if response else None,
            "recovery_choice_taken": self._recovery_choices.get(request.request_id),
            "recovery_id": result.recovery_id,
            "recovery_choices_shown": [c["choice"] for c in result.choices],
            "automatic_retry_allowed": False,
        })
    def metrics(self) -> dict[str, Any]:
        events = self.store.events()
        attempts = [event for event in events if event["event_type"] == "vision_attempt"]
        summaries = [event for event in events if event["event_type"] == "vision_request_summary"]
        legacy_summaries = [event for event in events if event["event_type"] == "vision_legacy_summary"]

        def percentile(values: list[float], fraction: float) -> Optional[float]:
            if not values:
                return None
            position = max(0, math.ceil(fraction * len(values)) - 1)
            return values[position]

        def logical_groups(summary_group: list[dict]) -> list[list[dict]]:
            grouped: dict[tuple[str, str, str], list[dict]] = {}
            for event in summary_group:
                # Framework IDs can be reused by hosts; separate different assets.
                key = (
                    str(event.get("session_id") or ""), str(event.get("request_id") or ""),
                    str(event.get("media_hash") or ""),
                )
                grouped.setdefault(key, []).append(event)
            return list(grouped.values())

        def aggregate(summary_group: list[dict], attempt_group: list[dict]) -> dict[str, Any]:
            groups = logical_groups(summary_group)
            latest = [group[-1] for group in groups]
            latencies = sorted(
                sum(float(event["total_latency_ms"]) for event in group) for group in groups
            )
            estimated = sum(
                (Decimal(str(event.get("estimated_cost_usd") or "0")) for event in attempt_group), Decimal(0),
            )
            actual_values = [
                Decimal(str(event["actual_cost_usd"])) for event in attempt_group
                if event.get("actual_cost_usd") is not None
            ]
            successes = sum(bool(event.get("success")) for event in latest)
            return {
                "logical_requests": len(groups), "successful_requests": successes,
                "logical_request_success_rate": successes / len(groups) if groups else None,
                "nonempty_analysis_rate": (
                    sum(bool(event.get("analysis_nonempty", event.get("success"))) for event in latest) / len(latest)
                    if latest else None
                ),
                "provider_attempts": len(attempt_group), "estimated_cost_usd": estimated,
                "provider_confirmed_cost_usd": sum(actual_values, Decimal(0)) if actual_values else None,
                "actual_cost_usd": sum(actual_values, Decimal(0)) if actual_values else None,
                "p50_total_latency_ms": percentile(latencies, 0.50),
                "p90_total_latency_ms": percentile(latencies, 0.90),
            }

        def legacy_aggregate(summary_group: list[dict]) -> dict[str, Any]:
            groups = logical_groups(summary_group)
            latest = [group[-1] for group in groups]
            latencies = sorted(
                sum(float(event["total_latency_ms"]) for event in group) for group in groups
            )
            successes = sum(bool(event.get("success")) for event in latest)
            return {
                "logical_requests": len(groups), "successful_requests": successes,
                "logical_request_success_rate": successes / len(groups) if groups else None,
                "nonempty_analysis_rate": (
                    sum(bool(event.get("analysis_nonempty")) for event in latest) / len(latest)
                    if latest else None
                ),
                "provider_attempts": None, "estimated_cost_usd": None, "actual_cost_usd": None,
                "p50_total_latency_ms": percentile(latencies, 0.50),
                "p90_total_latency_ms": percentile(latencies, 0.90),
            }

        result = aggregate(summaries, attempts)
        result["legacy"] = legacy_aggregate(legacy_summaries)
        route_keys = {
            (str(event.get("candidate_id") or "none"), str(event.get("media_kind") or "unknown"))
            for event in attempts + summaries
        }
        result["by_route"] = {
            f"{candidate_id}:{media_kind}": aggregate(
                [event for event in summaries if (str(event.get("candidate_id") or "none"), event.get("media_kind")) == (candidate_id, media_kind)],
                [event for event in attempts if (str(event.get("candidate_id") or "none"), event.get("media_kind")) == (candidate_id, media_kind)],
            )
            for candidate_id, media_kind in sorted(route_keys)
        }
        size_keys = {str(event.get("size_bucket") or "unknown") for event in summaries}
        result["by_media_size"] = {
            size: aggregate([event for event in summaries if str(event.get("size_bucket") or "unknown") == size], [])
            for size in sorted(size_keys)
        }
        result.update({
            "labeled_semantic_accuracy": None,
            "circuit_openings": sum(event["event_type"] == "vision_circuit_open" for event in events),
            "recovery_choices_shown": sum(len(event.get("recovery_choices_shown") or []) for event in summaries),
            "recovery_choices_taken": sum(event["event_type"] == "vision_recovery_choice" for event in events),
            "paid_approvals_denied": sum(event["event_type"] == "vision_paid_approval_denied" for event in events),
            "paid_approvals_expired": sum(event["event_type"] == "vision_paid_approval_expired" for event in events),
            "loop_prevention_hits": sum(event["event_type"] == "vision_loop_prevention" for event in events),
        })
        return result


def fingerprint_text(value: str) -> str:
    """Hash a normalized prompt for persistence without storing prompt text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
