# Shared vision routing verification handoff

Generated: 2026-11-04T04:42:31.540683+00:00

## Verdict

The shared routing implementation is verified in this scratch workspace for the mocked acceptance and focused regression scope. It is **not installed or active system-wide**. The live fleet remains in `legacy`, so this is a safe packaged handoff rather than a claim of completed provider migration.

## Measured results

- Focused canonical run: **157 passed, 0 failed** across 10 files (router, integration, tool routing, route control, config defaults, native fast path, vision tools, history budget, region, scale disclosure).
- New configuration/privacy acceptance checks: initial RED run had **3 failures**; after the minimal validation fixes, **22/22 integration tests passed**. The five added checks cover malformed `allow`, empty non-mapping candidates, non-numeric quality sample sizes, fail-closed route probes, and exclusion of prompt/media/provider secrets from attempt and summary telemetry.
- Extended regression run: **304 passed, 1 failed, 4 skipped**, plus **181 setup/collection errors** in the two broad config files. The single failure is `ProviderProfile(hostnames=...)` version drift; the config errors come from missing `hermes_cli.git_commit_graph` in the partial workspace. These are outside the routing patch, but they mean a clean full-tree release gate has not been demonstrated.
- Syntax compilation passed for the implementation, inventory script, and changed focused tests. Ruff was not installed in the workspace environment, so no Ruff result is claimed.
- No real provider calls were made in this verification. No paid approval was granted and no third-party charge was incurred.

## Acceptance coverage

Mocked provider adapters exercised free success, permanent and transient failures, bounded same-route retry, distinct free fallback, missing/stale/invalid pricing, missing budgets, per-request and shared daily budget enforcement, single-use paid grants, paid denial/replay/expiry, concurrent overspend prevention, duplicate-call idempotency, deadline exhaustion, video capability gating, and disabled recovery choices. Results confirmed:

- paid recovery does not dispatch without fresh approval and reservation;
- free recovery identifies a distinct target and preserves data-use/quality disclosures;
- retry stops at its configured bound and cannot be re-entered through a disabled choice;
- insufficient evidence yields `likelihood: unknown`; semantic accuracy remains `null` without labels;
- failed paid attempts retain estimated cost/status and are not mislabeled free or provider-confirmed;
- telemetry contains routing/timing/cost metadata, not the tested prompt, media payload, or provider secret.

These are deterministic mocked results, not provider success-rate estimates.

## Fleet inventory and cost

Inspected **12** configurations: default, beauty, counter, dainas, dump, forge, halp, polish, sexton, shrink, sketch, werk. Migrated profiles (`shadow` or `active`): **none**. Active profiles: **none**. All remain `legacy`; no routing telemetry database was found in the inspected homes.

Live/source parity: missing from `/opt/hermes`: agent/vision_integration.py, agent/vision_routing.py; present but different: agent/auxiliary_client.py, hermes_cli/config.py, hermes_cli/config_defaults.py, tools/vision_tools.py. No live file or profile configuration was changed.

Observed verification cost impact: **USD 0** in real provider calls. Observed fleet routed cost: **no routed requests found**. Estimated costs inside tests are fixtures and are excluded. The upstream handoff's real-canary statement was not independently reproduced and is not counted as live evidence here.

## Baseline comparison and limits

Legacy behavior is covered by native-fast-path, image-routing, and tool regression tests, but there are no live shadow/active requests to compare. No latency, recognition-quality, or reliability delta is claimed. The mocked scenarios establish control behavior only; they do not measure provider availability or semantic accuracy. Any future comparison needs matched media strata, request counts/denominators, and enough labeled observations.

## Remaining exceptions and next evidence

- All 12 profiles remain unmigrated; broader rollout awaits reviewed installation and explicit operator rollout decisions.
- Live provider capability, especially video, current free availability, pricing, and semantic quality remain unverified by this attempt.
- Run the patch in a complete source checkout and resolve the unrelated partial-snapshot config/test drift before using a full-suite release gate.
- Next useful evidence: a reviewed full-tree baseline diff, then one operator-approved non-critical shadow profile with real request counts, followed by a tiny active free-route canary and labeled fixture evaluation. Paid provider or pricing changes require a separate explicit decision.
