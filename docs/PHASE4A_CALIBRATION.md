# Phase 4A development calibration

Calibration asks whether [`recovery-readiness-v1`](PHASE4A_RECOVERY_TASK.md)
produces useful reward variation, before any optimization is paid for. It runs
the base policy only. There are **zero optimizer updates**, no checkpoint and no
final-panel evaluation.

**Status: implemented and tested offline; not run against a model.** No paid
inference, GPU work, image build, remote preparation or preflight has been
executed for Phase 4A. The [proposal](PHASE4A_CALIBRATION_PROPOSAL.md) for the
paid run is unapproved. What was executed locally is in
[offline verification](PHASE4A_VERIFICATION.md).

## What a calibration run contains

One plan covers both algorithm routes, because each route has its own sampling
stream and its own collection path. Per algorithm:

| Stage | Mode | Cases | Rollouts each | Episodes | Gateway route |
|---|---|---:|---:|---:|---|
| `greedy-development` | Greedy | 24 (whole panel) | 1 | 24 | `val`, temperature 0 |
| `sampled-packet-1` | Sampled | 8 | 4 | 32 | `train`, temperature 1 |
| `sampled-packet-2` | Sampled | 8 | 4 | 32 | `train`, temperature 1 |
| `sampled-packet-3` | Sampled | 8 | 4 | 32 | `train`, temperature 1 |

That is 120 episodes per algorithm and **240 in total**, in eight GPU stages on
one lane.

The greedy stage is the existing validation-only pass. Each sampled stage is the
existing production training collection with the one-collection stop that Phase 3
used for its a06 development check: the real adapter builds the batch, checks
rewards, context, token conservation and all eight four-member groups, and then
stops **before advantages, any backward pass, any optimizer step and any
checkpoint save**. No separate sampler imitates the training path, and no
multi-update calibration trainer was written.

Seeds are unchanged: the established `matched-rollout-mean-v2` streams for run
seed 9001. The stages of a job share them, and the packets differ by content. No
arbitrary seed or update count is supported.

## A separately versioned phase

Calibration is the phase `development-calibration` of the existing
`reader-task-v2` plan schema, with a nested declaration `recovery-calibration-v1`.
Existing profiles, plans and archived evidence keep their exact semantics. For a
fixed source identity, the calculator, asset and asset collection-only plans and
all fourteen of their launch configurations are byte-identical to `main`; a test
pins the digests.

`recovery-readiness-v1` supports **only** this phase. A profile for it without
`"purpose": "development-calibration"` is rejected, so no training length,
checkpoint or final evaluation can be planned for the task by accident.

Dispatch is closed. `experiment/reader_modes.py` names every phase, the stages
each phase declares and the tasks allowed in it. An unknown phase, stage or task
raises. Nothing falls through to calculator handling or to weaker historical
validation. Historical matched plans and the archived `reader-reference-v1`
schema keep their original name-based behavior.

### What a plan binds

`plan["calibration"]` and each stage's W&B configuration carry:

- the development panel identifier, the three packet identifiers and every
  ordered case ID;
- the split (`development`), mode, gateway route and temperature of each stage;
- the expected episode counts (24, 32, 32, 32; 120 per algorithm; 240 in total);
- zero optimizer updates and the unchanged seed derivation;
- the final-evaluation panel identifier and ordered-ID hash, marked as forbidden
  for model use;
- the fixed decision screen below.

Source, model, task, environment, template, configuration and context limits are
bound as in every reader plan. Changing any of them makes the plan stale.

## Both dataset arguments are routed

The pinned upstream entry point (`examples/calc_x/train_calc_agent.py`) loads its
training **and** validation files eagerly, whatever the stage does. Calibration
therefore routes both, explicitly:

- The worker verifies the data directory in `development` scope. It opens
  `world.json` and the development files only. The final and training files must
  exist as regular files but are never opened.
- It copies the one development file the stage declares to two run-local paths,
  `calibration-train.parquet` and `calibration-val.parquet`, after checking
  membership, order and split against the plan.
- The resolved configuration is rejected unless both arguments are exactly those
  two paths.

The final file is named `final_eval.parquet`, not `eval_main.parquet`, so no
launcher default reaches it. A test makes the final and training files unreadable
and records every file the interpreter opens while all eight stages are routed,
materialized and validated. None of them is touched. Hash and manifest validation
of the final file happens only in preparation and CPU preflight, which are
dataset checks, not model use.

## What completion requires

A stage is admitted only if all of the following hold. Anything else fails the
stage, and the rest of that algorithm's stages are skipped.

- Exactly the expected episodes: each declared case once (greedy) or four times
  (sampled), every episode complete, every reward confirmed.
- Each episode is re-verified in the worker from its own trace. The reward and
  diagnostics are recomputed from the recorded answer and the recorded joined
  observations and must equal what the harness wrote. Every policy-error code is
  re-derived from its recorded call and observation, and in a completed episode
  every successful call must be joined to the model input that saw its result.
- Trace identities match the stage: task, environment, source, configuration,
  W&B run, route and evidence class.
- Zero `optimizer_step`, `gradient_step`, `matched_advantages` and
  `native_checkpoint_saved` audit events, exactly one weight refresh at step 0,
  and no saved checkpoint.
- For a sampled stage: the adapter's rollout IDs, case IDs and rewards equal the
  traces'; eight groups of four; masked target tokens equal sampled tokens for
  every rollout; the server-reported completion tokens of each episode's calls
  equal the adapter's sampled tokens; no truncated row and no dropped prompt.

Constant groups are kept and counted. A group of four zeros is as valid as a group
of four ones. Nothing is zero-filled, dropped, padded or resampled to make a
collection pass, and a missing reward stays missing.

Optimizer accounting comes from audit events. It is never inferred from W&B
history rows, which appear without any update.

## The report

`experiment/recovery_calibration.py` builds a machine-readable report
(`recovery-calibration-report-v1`) and a readable summary. Per algorithm route it
gives:

- expected, admitted and not-admitted episodes; confirmed and missing rewards;
- JSON syntax, contract format, target and time correctness;
- decision correctness and evidence correctness as separate counts, each over the
  assessable episodes, with the number that is not assessable stated;
- typed action errors by code, recovered episodes and terminal policy failures;
- per-tool coverage, episodes with all required evidence observed, and joined
  observation chains;
- results by case family;
- the reward vector of every question, with each rollout's failure class;
- constant-zero, constant-one and mixed groups, with mixed groups split into
  those that include a semantic or evidence failure and those that vary only by
  format or action;
- sampled tokens, mask tokens, masked context tokens, fragment rows, the largest
  fragment prompt and truncation;
- optimizer, gradient, advantage and checkpoint event counts and refresh steps;
- 95% Wilson intervals for reward rates, stage timings when available, and the
  unverified assumptions.

A malformed answer is never read as a known semantic answer. Its decision and
evidence are recorded as not assessable.

## The decision screen

Fixed in code as `recovery-screen-v1` and bound into every plan, before any live
sampling. Two decisions are reported separately, **per algorithm route**. Routes
are never pooled, so a passing route cannot hide a failing one.

**Operational validity.** All 120 episodes of the route are complete with
confirmed rewards, no optimization event occurred, and the evidence is live. An
incomplete route is not interpreted at all. Evidence class is also decided per
route: one scripted stage makes its own route non-live and leaves the other
route's class alone.

**Scientific suitability.** A starting screen for this panel:

- at least **6 of 24** sampled groups are mixed and include a semantic or evidence
  failure;
- those groups span at least **2** case families;
- format and action failures are at most **half** of the zero rewards.

Six of 24 is about two informative groups per eight-question batch. This is a
project-specific engineering heuristic. It is not a research-established
threshold, it imposes no target accuracy band, and passing it is not evidence
that training would improve anything. The report also states whether both
successes and failures were observed in greedy and in sampled episodes.

A route that does not pass is labelled `saturated` (more than 18 all-correct
groups), `interface_dominated`, `too_difficult` (more than 18 all-zero groups) or
`insufficient_semantic_variation`, checked in that order. **Each of these is a
valid calibration result.** The response is to revise the task, prompt or harness
using development evidence and new identities. An unchanged collection is never
repeated until it looks favorable.

## Commands

### Local and free

```bash
# Plan and validate offline. The supplied profile is a synthetic fixture.
uv run --locked python scripts/reader_training.py plan \
  --profile examples/reader-profile.recovery-calibration.synthetic.json --out outputs/phase4a/plan.json
uv run --locked python scripts/reader_training.py validate \
  --profile examples/reader-profile.recovery-calibration.synthetic.json --plan outputs/phase4a/plan.json

# All 240 episodes through the production harness with scripted inference (about four minutes).
.venv-reader-cpu/bin/python scripts/task_local.py calibration-demo --task recovery-readiness-v1 \
  --out outputs/phase4a/scripted-calibration

# Rebuild the report from stage completions; episode rows are rechecked against the traces.
.venv-reader-cpu/bin/python scripts/task_local.py report --task recovery-readiness-v1 \
  --evidence outputs/phase4a/scripted-calibration --plan outputs/phase4a/scripted-calibration/plan.json \
  --out outputs/phase4a/report
```

`plan` needs a clean committed source and writes only to an ignored path.
`calibration-demo` runs real episodes over real stdio MCP, but the model
responses are a fixed arbitrary script and the adapter and trainer events are
simulated. Its report is labelled `scripted-inference`, says "Not a live policy
measurement", and reports suitability as `not_assessed`. `report` works the same
way on a directory produced by `reader_training.py collect`. `status` and
`collect` report an operationally invalid route only as incomplete evidence; use
`report` to see its counts and stage problems.

### Billable: not run, and not authorized

`setup`, `prepare`, `preflight`, `launch`, `status`, `collect` and `stop-owned`
of `scripts/reader_training.py` reach the provider. `setup` creates a volume and
a registry. `prepare` and `preflight` run paid CPU workers. `launch` runs the
supervisor and eight paid GPU stages. They are implemented for this phase and
tested with the provider boundary mocked. None has been run for Phase 4A. Do not
run them without an approved budget, an owned non-fixture profile with fresh
resource names, and a ledger reconciled against the provider.

The sequence, once approved, is the one in
[reader training](READER_TRAINING.md): `plan` → `setup` → `prepare` → `collect` →
`plan --assets` → `preflight` → `launch` → `collect` → `report`.

For this task the remote preflight also runs the three-tool probe in the training
image and measures the bounded prompts with the tokenizer and template actually
served. It fails if a declared context limit is exceeded.

## An unknown submission: reconcile before any retry

A receipt with status `unknown` or without a call ID is not proof that nothing
ran. The procedure is the one Phase 3 learned the hard way:

1. **Reserve first.** Record the attempt's full maximum exposure in the spending
   ledger before `prepare`, `preflight` or `launch`. The CLI also writes a durable
   registry reservation and a local receipt before it spawns anything.
2. **Keep the client alive** until the command returns. Each of these commands
   resolves the image first, which can take minutes. Run them as background jobs,
   never under a short command timeout.
3. **If the outcome is unknown, do not resubmit.** The CLI refuses a second
   submission for the same plan. Do not clear the reservation to get around that.
4. **Reconcile read-only.** Query the provider for apps and tasks, the registry
   rows for the plan, the volume contents under the plan's root, and billing.
5. **Stop what is owned.** Run `stop-owned` with the same plan. It cancels only
   call IDs recorded for that plan.
6. **Record the finding** with the receipt, and keep the reservation at full
   exposure until every owned call is terminal.
7. **Use a fresh attempt**: new attempt number, new paths and, after any source
   change, a new plan. The earlier evidence is preserved unchanged.

The provider-side stop marker is a registry entry that expires after seven days
without a read. The lasting record is the local receipt and the committed
reconciliation.

## Handing the work over

Assistants and operators do not share conversation history. Before a session
ends, write down, and before a session starts, verify:

- the exact commits: tested source, branch head and PR;
- the checkout and worktree in use, and that one operator owns each mutable
  checkout and each campaign ledger;
- the plans in existence and the commit each is bound to;
- active local processes, especially a client in mid-submission;
- submission receipts and their status;
- resources created, who owns them, and their retention dates;
- completed work, and decisions waiting on the owner.

The incoming operator checks the remote branch head against the brief, compares
the ledger with provider state read-only, and reviews inherited commits against
raw evidence before building on them. The current state for Phase 4A is recorded
at the end of [offline verification](PHASE4A_VERIFICATION.md#handoff-state).

## After calibration

A learning study is conditional on a route passing the screen, and needs its own
proposal. It would first freeze the initialization, task, prompts, tools,
verifier, episode policy, decoding, training schedule, seeds and
checkpoint-selection rule, and then compare the baseline with a fresh-worker
evaluation of the trained checkpoint on the untouched final panel.

Both algorithms stay available. No training length beyond the qualified two
updates is generalized here, and no algorithm comparison is promised before
calibration supports one. Such a study would report group signal for GRPO, and
critic quality and clipping for PPO-MC. Undefined explained variance stays
undefined. A controlled negative learning result is a valid result.
