# Phase 3 asset episode correction — 2026-11-22

> **Superseded as a status report.** This interim note was written before any live
> execution of the revised contract. Its planned attempt a05 was never executed.
> A later review added a typed `non_object_arguments` policy error at
> `7f1e13d7bacb477c9e5e95a16868a2b0f2484ee6`, the source that actually ran. Results,
> executed identities, test counts and the reconciled budget are in the
> [2026-11-24 report](PHASE3_QUALIFICATION_2026-11-24.md). The text below is kept as
> written.

This continuation uses the original aggregate $300 authorization. The earlier
[a01–a04 report](PHASE3_QUALIFICATION_2026-11-22.md) and evidence remain immutable.
At implementation start, local HEAD and remote PR #4 head were
`4f0d2973b10fbaff4233f5781833e8a6c600cf61`, main was
`5c069f7e9cc23d03050d13360ce43e27e47e54aa`, and the tree was clean.
GitHub's public API confirmed the PR open and draft. The previous executed source
was `07ddf1bad56c10c6b8b4d144293aec4f911df706`.

## Trace-grounded correction

The retained private a04 `final-evidence` contains all 192 original trajectories:
64 baseline and 32 first-collection training attempts per algorithm. Inspection of
the actual messages verified that every generation retained the original system
and user instructions. All 128 baseline trajectories made two valid calls followed
by reflection without tools; every final response was prose. AutoGen 0.7.5's
`_reflect_on_tool_use_flow` retains the full context and sets `tool_choice="none"`.
The old scripted success fixture instead covered one call followed by a final
response through the tool-enabled continuation path.

All five failed sampled training episodes supplied forbidden extra tool arguments.
Four recovered to valid second calls, then produced prose; one repeated the invalid
call and generated JSON with an invalid status enum. Two other completed PPO-MC
responses were JSON tool-call envelopes rather than the required final record.
All original rewards, missing rewards, source identities and classifications stay
unchanged. Raw interactions, exact prompts, observations and token evidence remain
in ignored private local evidence. This report publishes reviewed aggregates only.

The prospective `asset-episode-v2` descriptor is included in the task identity.
Dataset ID, recipe, registry, memberships, dataset bytes and environment hash remain
fixed. The strict whole-response verifier and numerical algorithms are unchanged.

| Observed outcome | New treatment |
|---|---|
| Complete normal final response | Strict independent reward zero or one |
| Explicit invalid action, observed error, recovery and final response | Same strict reward; retain original actions and observations |
| Complete malformed/incorrect final response | Complete task failure, reward zero |
| Two observed invalid actions | Declared terminal task failure, one confirmed zero, no invented answer |
| Infrastructure failure, truncation, timeout, cancellation, missing evidence or unconfirmed reward | Incomplete; no fabricated reward |

The MCP server emits typed policy errors only for explicit input validation failures.
The harness cross-checks the sampled arguments and returned error code. AutoGen's
malformed-argument and unknown-tool branches require exact call/observation evidence.
Unknown errors fail closed. Every model action stays in the observed path; no bad
action is filtered from token accounting or the loss. Error observations remain
observations. A final JSON reminder is appended to the full reflection context by
the same observed client, without constrained decoding, answer repair or hidden labels.
Integration reports separate episode outcomes from successful tool chains.

## Offline verification

Executed the required gates using the existing Python 3.12 CPU environments and
the documented cached-source setup in [PHASE3_VERIFICATION](PHASE3_VERIFICATION.md):

- `bash scripts/test_task_local.sh`: 26 tests passed.
- `bash scripts/test_reader_offline.sh`: 39 PPO-MC, 15 matched-study and 85 reader
  tests passed; historical artifact/accounting checks passed.
- `bash scripts/test_matched_training.sh` with the exact documented cache variables:
  63 tests plus six pinned-configuration checks passed; no required skips.
- A subsequent focused collection-completion guard test passed with the nine other
  contract tests. Its initial sandbox invocation failed to create temporary files;
  the approved rerun passed. That invocation is not live runtime evidence.

Evidence classes remain distinct: real local MCP/process/Git/HTTP/CPU execution;
scripted model responses; synthetic token fixtures passed through pinned adapter
and numerical bodies; mocked provider completion. No local test establishes live
model format compliance. Required branches include two calls plus reflection JSON
or prose, explicit error/recovery, bounded failure, infrastructure failure after tool
success, missing output, truncation, cancellation and failed reward delivery.

## Live development and qualification

The collection-only route declares eight training-smoke questions and four rollouts
per question for each algorithm, with temperature 1 and the existing seed streams.
It uses the production training adapter, validates every rollout/reward/context and
token/fragment/group count, then stops before advantages and optimizer execution.
Its audit retains collection index 1 separately from zero optimizer updates.
Baseline qualification remains temperature 0 on the existing 64-case panel.

Live results and exact newly executed source/plan identities will be recorded after
execution. The implementation alone is not operational qualification.

## Budget and preservation at admission

Fresh read-only Modal reconciliation at 2026-11-22 06:13:43 UTC confirmed all 22
known calls terminal and 25,802,624,741 retained bytes. The completed-hour report
still attributed $5.76165025; this is not a settled invoice. The full inclusive $40
hold remains, leaving $260 unreserved before any new reservation. The original
retention obligation is 2026-11-29 01:54:46 UTC. No historical resource was deleted
and no new paid work occurred during offline implementation.

The publication audit checked working files, index, staged changes and introduced
history. Findings were confined to the previously reviewed synthetic credential
and authenticated-URL rejection fixtures. No private traces, credentials, model
weights or checkpoint tensors belong in this public branch. PR #4 stays draft
and unmerged. The 64-case panel is repeatedly inspected qualification/regression
evidence, not an untouched generalization benchmark.
