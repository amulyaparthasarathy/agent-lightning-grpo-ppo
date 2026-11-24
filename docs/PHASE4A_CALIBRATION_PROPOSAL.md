# Phase 4A paid calibration: proposal

**Status: unapproved. Nothing in this document has been executed.** No resource
was created, no image was built, and no model call was made while preparing it.
Publishing it authorizes nothing. It asks for one new decision: a single
aggregate ceiling of **US$200** for the development calibration described in
[the calibration guide](PHASE4A_CALIBRATION.md).

The Phase 3 authorization is closed. Its unused balance (US$226.00) and its
holds (US$74.00) are **not** counted as available funding here.

## What would run

| Item | Value |
|---|---|
| Task | `recovery-readiness-v1`, recipe `recovery-readiness-v1/2026-11-24/329` |
| Environment SHA-256 | `0c384ed90027f7fa0fb0aeb097220e6e83876dfdd85094353f03c8dd035b1ca3` |
| Development panel | `rr1-dev-ac7fd952343b0c7d`, 24 cases, three packets |
| Final panel | `rr1-final-a76397cabf99d31a`; not evaluated |
| Episode policy | `recovery-episode-v1` |
| Context limits | `recovery-context-v1` (trajectory prompt limit 3,072) |
| Calibration schema | `recovery-calibration-v1`, screen `recovery-screen-v1` |
| Model | `meta-llama/Llama-3.2-3B-Instruct`, revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`, parser `llama3_json` |
| Template | vLLM v0.22.0 `tool_chat_template_llama3.1_json.jinja`, unmodified, with the prepared fixed date |
| Agent Lightning | `cbc56e41e0eab643aa71d195e92e5bba652fa683` |
| VERL | source `7aed6b230776f963fa09509c10d9c3a767d1102c`, wheel 0.8.0 |
| Upstream trainer SHA-256 | `a0ddd84022896cd34f1d2e30bf6b35897497df62847f94ade98703c22e02b368` |
| Algorithms | `grpo-matched-v2` and `ppo-mc-matched-v2`, paired seed 9001, unchanged streams |

**Candidate source.** The tested implementation commit is
`664a5a45e6648b298449c7a2a70c205ee683de39`. Documentation commits follow it on the
branch and change no file under `experiment/`. A plan is bound to the commit it
was generated from, and a document cannot name the commit that contains it. The
executable plan must therefore be regenerated from the clean approved head; the
delivery report gives that head and its plan hashes. Identities of the plans
generated at the tested commit are in
[`candidate-identities.json`](../evidence/phase4a-offline-20261124/candidate-identities.json).

**Scope.** 240 episodes and zero optimizer updates:

- per algorithm, 24 greedy development episodes at temperature 0;
- per algorithm, 3 packets × 8 cases × 4 samples = 96 sampled episodes at
  temperature 1 through the production collection path;
- 8 GPU stages in total, one after another on **one GPU lane**
  (`H100!`, 16 CPU cores, 196,608 MiB, one single-use worker per stage);
- no advantage computation, backward pass, optimizer step, checkpoint save,
  warm start, SFT, reward change or model change.

## Rates

From [Modal's pricing page](https://modal.com/pricing) as recorded on 2026-11-20
for Phase 3. They were **not re-checked** for this proposal and must be verified
before any submission.

| Resource | Rate |
|---|---:|
| H100 | US$0.001097 per second |
| CPU | US$0.0000131 per core-second |
| Memory | US$0.00000222 per GiB-second |
| Volume storage | US$0.09 per GiB-month |
| Egress | US$0.04 per GiB |

The worker shape costs about **US$6.24 per hour** at these rates (GPU 3.95, CPU
0.75, memory 1.53). Reservations use the existing conservative **US$10 per
worker-hour**. Credits, cache hits and free allowances are valued at zero.

## What Phase 3 observed

These come from the committed Phase 3 records and its collected stage receipts.

| Observation | Value |
|---|---:|
| a06: two collection-only stages with preparation and preflight | US$1.39 |
| a07: six stages with preparation and preflight | US$5.16 |
| Collection-only stage, wall time | 378 s and 400 s |
| 64-episode greedy stage, wall time | 293–373 s |
| Time spent in episodes within a stage | 49–70 s |

Stages were dominated by startup: about five minutes to load the model and start
the services, against about one minute of episodes.

## Estimate for one clean attempt

Recovery episodes are longer than asset episodes: up to six model calls instead
of three, and prompts of about 1,000–1,900 tokens instead of about 420–480. The
estimate assumes 450–750 seconds per stage. **That range is an assumption.** No
recovery episode has been served by the model.

| Item | Basis | Estimate |
|---|---|---:|
| Eight GPU stages | 3,600–6,000 s at US$6.24 per hour | US$6.20–10.40 |
| Supervisor | 1 CPU, 2 GiB for the same time | US$0.06–0.11 |
| Preparation and CPU preflight | 4 CPU, 16 GiB; at most 3,600 s and 900 s | at most US$0.40 |
| Image build and resolution | cached layers expected; a cold build is at most about US$2.00 at list rates | US$0–2.00 |
| Storage | about 7 GiB of model and data; bounded at 100 GiB for seven days | at most US$2.10 |
| Collection and egress | at most 2 GiB | at most US$0.08 |
| **Expected total** | | **about US$7–15** |

## Proposed ceiling

Reservations are worst-case bounds, not expected spending.

| Category | Bound | Allocation |
|---|---|---:|
| Eight GPU stages plus supervisor | 8 × 1,500 s of worker time (US$33.33) plus 12,600 s of supervisor time accounted at the worker ceiling (US$35.00) | US$70 |
| Image build and resolution | Cold rebuild of every layer, with margin | US$15 |
| Preparation and CPU preflight | Both at their time limits, with margin | US$5 |
| Storage | 100 GiB for seven days (US$2.10) | US$3 |
| Collection, egress and reconciliation | 2 GiB and read-only queries | US$2 |
| **One attempt** | Reserved in full before the first dispatch | **US$95** |
| Recovery allowance | One evidence-driven corrected attempt under fresh identities | US$95 |
| Contingency | Billing lag, rate drift, retention beyond seven days | US$10 |
| **Total proposed ceiling** | | **US$200** |

**Bounded stages.** Each GPU stage has a 1,500-second limit and a 600-second
startup limit, a shortened dispatch deadline, a durable heartbeat and no automatic
retry. Preparation is limited to 3,600 seconds and preflight to 900.

**Failure exposure.** The most one stage can cost is 1,500 seconds: US$4.17 at
the reservation rate, about US$2.60 at list rates. A stage that fails ends that
algorithm's remaining stages; the other algorithm still runs. The most one
attempt can cost is its US$95 reservation. The ledger and these estimates are
operating controls, not provider spending caps. Builds, startup, storage and
telemetry can outlast estimates, and billing can lag.

**Corrections.** Approval of the ceiling would cover ordinary evidence-driven
corrections inside this scope, such as a prompt or harness fix found by the first
attempt, without a new approval for each one. Every correction reserves its full
exposure first, uses new committed source, new plans and fresh identities, and
keeps the earlier evidence. A correction that does not fit the remaining balance
stops. The allowance is not permission to repeat an unchanged collection.

## Resources and retention

Proposed fresh names. None exists yet, and none reuses a Phase 3 object.

| Resource | Proposed name |
|---|---|
| App | `reader-recovery-20261124-c01-app` |
| Registry | `reader-recovery-20261124-c01-registry` |
| Volume | `reader-recovery-20261124-c01-volume` |
| Paths on the volume | `phase4a-assets/…`, `phase4a-staging/c01`, `phase4a-runs` |
| Experiment and attempt | `reader-recovery-20261124-c01`, attempt 1 |
| Secret | The existing `reader-reference-credentials` handle; verified, never created or read |
| W&B | The existing `reader-owned-reference` project; eight new run IDs |

`setup` creates the volume and the registry and refuses to proceed if either
exists. A corrected attempt would use `c02` names.

Retention obligations:

- The volume would hold one prepared model copy, the generated data, traces and
  logs. It would hold **no checkpoint**, because nothing is saved.
- Review at seven days. The owner decides whether to keep or delete; nothing
  deletes itself and no deletion is implied by this proposal.
- Storage accrues at about US$0.63 per month for 7 GiB until that decision.
- Raw traces are private. Only reviewed, redacted summaries are committed.
- No Phase 3 or earlier resource is modified, transferred, repurposed or deleted.

## Reservation and unknown submissions

Before any dispatch the full US$95 is recorded in a new private ledger for this
campaign. The CLI then writes a durable registry reservation and a local receipt
before it spawns. A lost acknowledgement stays **unknown**: it is reconciled
read-only against the provider, stopped with `stop-owned`, recorded, and followed
by a fresh attempt, never by a blind retry. A reservation is reduced only after
every owned call is terminal. The full procedure is in
[the calibration guide](PHASE4A_CALIBRATION.md#an-unknown-submission-reconcile-before-any-retry).

## Scientific stop and revise criteria

Fixed before any live sampling, in `recovery-screen-v1`, and applied to each
algorithm route separately.

- **Stop and do not interpret** a route with any incomplete episode, unconfirmed
  reward, optimization event or failed stage. Incomplete collections are not
  admitted.
- **The screen:** at least 6 of 24 sampled groups mixed with a semantic or
  evidence failure, across at least 2 case families, with format and action
  failures at most half of the zero rewards. This is an engineering heuristic,
  not a research threshold.
- **A negative result is valid.** `saturated`, `too_difficult`,
  `interface_dominated` and `insufficient_semantic_variation` each end the
  attempt with a finding.
- **Revise, never retry.** Any revision uses development evidence only, new
  identities and a new plan. An unchanged collection is not repeated to get a
  better sample.
- **The final panel stays untouched.** If it ever informs a redesign, it becomes
  development evidence and a new holdout is required.

## What this proposal cannot promise

- That the model produces any contract-valid answer. The prompt asks for a
  function call and a final JSON object on the same turn, and Phase 3 showed how
  much that template matters.
- That stage times fall in the assumed range.
- That the local MCP 1.26.0 behavior matches the image's MCP version. The remote
  preflight probe checks the typed error payloads there.
- That a passing screen predicts learning. Calibration measures reward
  variation in the base policy and nothing else.

## If calibration supports a study

A learning study would be proposed separately, and only for a route that passes.
It would freeze the initialization, task, prompts, tools, verifier, episode
policy, decoding, training schedule, seeds and checkpoint-selection rule first,
and then compare the baseline with a fresh-worker evaluation of the trained
checkpoint on the untouched final panel. Both algorithms stay available. No
training length is generalized here and no algorithm comparison is promised. The
study would report group signal for GRPO and critic quality and clipping for
PPO-MC, leave undefined explained variance undefined, and treat a controlled
negative result as valid.
