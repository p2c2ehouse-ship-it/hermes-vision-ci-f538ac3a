# Auxiliary vision routing: operations and rollback

This document covers the shared router used by auxiliary image/video analysis. Native image support in the main model is unchanged. The safe default is `legacy`; installing the code does not migrate any agent.

## Safety invariants

- `legacy` preserves the existing route. `shadow` records a routing decision and a legacy-outcome summary but makes no new provider call. `active` uses the shared router.
- Hosted-free routes are allowed by default. Local routes and paid routes are disabled by default.
- A paid attempt requires all three positive budget ceilings, a current sourced price snapshot, and a fresh trusted per-call approval. There is no supported automatic-approval setting.
- Free means no third-party API charge for that attempt; it does not imply unchanged privacy, accuracy, availability, or latency.
- Retry choices report observed first-retry transport success only when the exact telemetry stratum has enough observations. Otherwise likelihood is `unknown`; semantic accuracy is not inferred from transport success.
- Telemetry stores route, timing, outcome, and cost metadata. Prompts, media/base64, provider error bodies, API keys, and raw approval tokens must not be logged.

## Validate configuration

Use registered provider IDs and pinned models. Do not put API keys, approval tokens, or auto-approval fields in routing config. Check each profile before migration:

```bash
hermes --profile PROFILE config check
```

Start with one non-critical profile. Keep its existing provider/model registered as exactly one candidate, or set `pinned_candidate` explicitly. Candidate `verified_media` must be evidence-based and a subset of `declared_media`; `video` is not implied by `image` support.

## Shadow rollout

A minimal hosted-free candidate looks like this (the model is an example, not a current capability or availability claim):

```yaml
auxiliary:
  vision:
    routing:
      mode: shadow
      strategy: free_first
      allow:
        hosted_free: true
        local: false
        paid: false
      candidates:
        current_free:
          provider: openrouter
          model: google/gemma-4-31b-it:free
          declared_media: [image]
          verified_media: [image]
          cost_class: hosted_free
          configured_order: 0
```

In shadow mode, inspect route decisions and legacy outcomes before considering `active`. Shadow is not a quality benchmark and does not establish recognition accuracy.

## Active rollout and failure choices

Change only the selected profile after its shadow evidence is understood:

```bash
hermes --profile PROFILE config set auxiliary.vision.routing.mode active
hermes gateway restart
```

The same media request and recovery offer are shared across consumers. A failed request can offer:

1. **paid** — a provider/target-specific estimated upper bound, price timestamp/source, output cap, and billing-variance disclosure; requires a new user approval and reservation before one attempt;
2. **free** — a distinct fallback target with its data-use disclosure and quality-evidence status; and
3. **retry** — same target only while its bounded retry budget remains, with observed likelihood or `unknown`.

Recovery offers expire. An approval is single-use and expires separately. Repeated automatic retries, duplicate tool calls, disabled recovery choices, and malformed active configuration must not dispatch additional provider calls.

## Paid recovery

Leave `allow.paid: false` unless the operator has approved both the provider and the spend ceilings. Paid routing requires `paid_per_request_usd`, `paid_daily_usd`, and `paid_monthly_usd`; a missing, stale, negative, or non-finite price disables the paid choice. Provider-reported cost is separately labeled from estimated cost. A failed paid attempt remains an estimate, not a confirmed receipt.

To stop new paid attempts immediately without changing other routes:

```bash
hermes --profile PROFILE config set auxiliary.vision.routing.allow.paid false
hermes gateway restart
```

## Telemetry and fleet inventory

The default per-profile database is `$HERMES_HOME/vision-routing/routing.db` (or the profile-equivalent home). Read it read-only. Do not treat estimated totals as provider invoices.

From the implementation workspace, the secret-free inventory checks live/source parity, each profile's mode and validation issues, and aggregate telemetry without media or prompt contents:

```bash
PYTHONPATH="$PWD" .venv/bin/python scripts/vision_rollout_inventory.py > artifacts/vision-rollout-inventory.json
```

The script never resolves credentials or calls a provider. Review actual request counts and denominators before comparing legacy/shadow/active outcomes. Small samples and different media strata are not valid success-rate comparisons.

## Rollback

Behavior rollback is per profile and does not require deleting telemetry:

```bash
hermes --profile PROFILE config set auxiliary.vision.routing.mode legacy
hermes --profile PROFILE config set auxiliary.vision.routing.allow.paid false
hermes gateway restart
```

For the default profile, omit `--profile PROFILE`. Confirm the effective mode and run a tiny local fixture with provider calls mocked or disabled before resuming normal use.

Before changing config files by hand, make a timestamped copy while the gateway is stopped. Default config is `~/.hermes/config.yaml`; named profiles use `~/.hermes/profiles/PROFILE/config.yaml`. Restore only the affected profile's backup, then restart and validate it.

If a code rollback is required, restore the prior Hermes release as a unit rather than mixing partial module versions. Preserve `routing.db` and any shared budget ledger for audit; deleting them is not a rollback and can destroy spend evidence. Existing retained-media cleanup should be allowed to proceed, but no recovery approval should be replayed after rollback.
