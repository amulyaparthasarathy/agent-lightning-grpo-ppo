# Asset-status live qualification — 2026-11-22

**Not fully qualified.** Both algorithms passed live baseline integration on all 64 heldout assets, but their first sampled training collections contained incomplete trajectories. Both stopped before any actor or critic optimizer update. No native step-two checkpoint or fresh-worker checkpoint evaluation was produced. This report concerns the newly authorized
US$300 Phase 3 campaign. The completed reference campaign's earlier US$250
allocation remains closed and is excluded. UTC execution began on November 22;
resource slugs retain the request's November 21 date.

## Scope and provenance

Public repository: `amulyaparthasarathy/agent-lightning-grpo-ppo`; branch
`phase3-reader-task-adaptation`; PR [#4](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/pull/4)
remains draft and unmerged. Reviewed base:
`5c069f7e9cc23d03050d13360ce43e27e47e54aa`. The origin, clean committed source,
live PR head/base, draft/open state, owner profile, and credential handles were
verified before paid work. No private history or datasets were imported.

Each corrected source was committed, reviewed for publication, pushed, and used to
generate fresh source-bound plans and explicit resource/attempt/W&B identities.
Preparation-only and asset-bound execution plans are separate. Source, configuration,
task, environment, packaged content, actual image, installed packages and data
identities are recorded separately in the evidence manifest. The final documentation
commit did not execute the experiments described here.

The client used the existing Workbench environment, Python 3.12, uv environments,
and existing CPU dependency caches. CUDA/vLLM training remained in the established
remote x86-64 Agent Lightning/VERL environment. No calculator reference training or
historical 20-update experiment was repeated.

## Executed source and plan identities

Attempt a01:

```text
source commit:    b73f3f7b4497ea553a4ebf1e47e52c81afeae070
preparation plan: 0ee63d537f24df652799618cd3f57fcd09ca097952d46e08a5b22ced34a5b7ab
stage plan:       9444af0daf0536d5fa2ab6f6cea035f07e26036e1e8b43044b3b998f6ec46b69
remote image:     im-YgJB9ZeIqUyuQXvUD8HQrx
```

Attempt a02:

```text
source commit:    de867130ff35afb287d6bf11f83a33f4951014bb
preparation plan: 3619bcd6a537d1283f1e06801a936b0d902a7b162cda381686370b273f9590f6
stage plan:       8a816f2566f1d4d02cb3df808d78029d701da8ca4bdc59385f8fa0ec1b0725c2
remote image:     im-buzZvebIURyGYvJVe3ed0g
```

Attempt a03:

```text
source commit:    e00dc14bb9777dd644b8384be3f18b97c69b972c
preparation plan: 9c21256e339c12f05f01baa007a78ae17dd02b81b3b61ff4297eda4516360498
stage plan:       cad14a7b5355cb02c05ce0bb7fe54b8d6b3cb28d18d2ffa513e7a47148b6fca1
remote image:     im-7SRnhIai3ZHWvdFWHLYw3e
```

Attempt a04:

```text
source commit:    07ddf1bad56c10c6b8b4d144293aec4f911df706
preparation plan: 02f01fb128acdcefdfb0a684685e5b38c74f0ec8103b8664ff8651df1b71e6d9
stage plan:       0947b53fdd507e12a1183a042a3f67f9701233b05052a1e01b6a3a04deae10ce
remote image:     im-a7M1vxet2jtMPeU7RPMawl
```

Plan hashes are the canonical JSON digests used by admission; file manifests separately hash actual file bytes.

The [identity manifest](../evidence/phase3-qualification-20261122/identities.json) also records source-inventory, configuration, task, environment, packaged-content and dataset hashes separately.

## Fixed task, model and runtime

- Task: `asset-status-v1`, deterministic recipe `asset-status-v1/2026-11-20/329`.
- Initial reviewed task SHA-256:
  `6dd4be007519e4e195c3c7dbcc00bad6b0996b6d1528698e124f3adff34b7d1a`.
- Synthetic environment SHA-256:
  `d683b4ca993f4d014de260fe8ba04afa901b1488be036a633c37b4fec7ef4089`.
- Model: `meta-llama/Llama-3.2-3B-Instruct`, revision
  `0cb88a4f764b7a12671c53f0838cd831a0843b95`; existing `llama3_json` parser,
  tokenizer/template and immutable asset checks.
- Agent Lightning: `cbc56e41e0eab643aa71d195e92e5bba652fa683`.
- VERL source: `7aed6b230776f963fa09509c10d9c3a767d1102c`, installed wheel 0.8.0.
- Upstream trainer guard:
  `a0ddd84022896cd34f1d2e30bf6b35897497df62847f94ade98703c22e02b368`.

The prepared model has two weight shards totaling 6,425,529,048 bytes, with hashes
`13cbd6d16e927a0c5bad54102514e6e18b4a47b3a6eb911e39d678d328d19f55`
and `7b770216613ac5c34d7c54bdff1fa616bc4e338a9d0b20af6303e48c295ee23c`.
The effective chat template hash is
`a718674808440b537a553c15e0557b5b45d6d76e49ad9ddbc1a9e92447843e2c`.
Remote preparation verifies actual file bytes. The synthetic data have 256 training,
64 heldout and eight separate smoke assets; the 16 training-smoke cases are a nested
training subset. Remote generated artifact bytes were collected and independently
verified locally. The ordered heldout membership hash is
`493299d2b0fe8db770d5047f65893d67c0fc4c8f6e6c9825305f7b4c6e68464c`;
the registry content hash is
`241b5672d43b7dff7db7d203f2805d91e0a541e7a20e18d4edb608419e670e44`.
The model receives the question and tool observations, not the verifier label.
The isolated MCP child clears the harness environment before importing task code.

Remote preflight captured 302 installed distributions, including MCP 1.30.0,
AutoGen core/agentchat/ext 0.7.5, Torch 2.11.0+cu130, vLLM 0.22.0,
Agent Lightning 1.0.3, flash-attn 2.8.3, Transformers 5.22.2,
PyArrow 25.0.1, Ray 2.60.0 and W&B 0.31.0. Local harness tests used MCP 1.26.0;
that local result was not substituted for remote compatibility evidence.

The actual remote stdio probe checked the exact tool schema, a valid asset lookup,
missing/non-string/extra arguments, a malformed ID, and an unknown ID. Observations
matched the task contract; the context exited, the ownership receipt reported the
process already exited, and cleanup had no anomalies. Package admission retained
six previously reviewed dependency conflicts with no version drift or new conflicts.
This is qualification with reviewed exceptions, not a claim of a consistent package
closure. Image cache hits remain unverified; no independent cold-build claim is made.

## Attempts, failures and corrections

1. **a01 — `b73f3f7b4497ea553a4ebf1e47e52c81afeae070`.** Preparation and
   preflight passed. GRPO failed during Ray worker registration before inference.
   The embedded reader request expanded the serialized runtime environment to
   roughly 135 KB, exceeding Linux's 131,072-byte single-argument limit. A real
   local subprocess reproduced `E2BIG`; the remote raylet's ephemeral error log
   was unavailable, so the precise initial failure cause is an evidence-supported
   diagnosis rather than a retained remote syscall trace. PPO and the supervisor
   were explicitly cancelled as owned calls while diagnosing the shared startup
   problem. Provider terminal states were subsequently confirmed. There were no
   trajectories, hosted W&B runs, optimizer evidence or checkpoints.
2. **a02 — `de867130ff35afb287d6bf11f83a33f4951014bb`.** Modern reader requests
   use an absolute, bounded, hash-bound request file instead of a large environment
   variable; historical request transport remains intact. The corrected runtime
   started live inference. GRPO produced 57 complete rewarded trajectories and
   seven failed trajectories out of 64, with 57 valid live integration chains.
   All 57 completed answers were `TERMINATE` and received reward zero; seven cases
   made invalid tool calls. The captured system prompt showed AutoGen's default
   instruction to reply `TERMINATE`. The complete-evaluation validator rejected
   the baseline, so the later supervisor integration gate was not reached.
   PPO and the supervisor were explicitly cancelled during startup to correct the
   shared harness instruction. No optimizer or checkpoint stage was admitted.
3. **a03 — `e00dc14bb9777dd644b8384be3f18b97c69b972c`.** The harness supplied
   the existing task prompt as its system message, removing the termination
   instruction. Both independent baselines ran all 64 cases. Each failed all 64
   trajectories, made 128 invalid tool calls, delivered zero rewards and produced
   zero valid integration chains. Captured model inputs included the correct tool
   schema, but model calls supplied answer fields as extra tool arguments. Both
   baseline workers cleaned up; the supervisor ended failed and skipped all
   dependents. No owner cancellation was needed. Hosted W&B marked both runs
   finished and recorded 64 traces but zero rewarded rollouts. Its `val/reward=0`
   must not be interpreted as 64 valid zero-reward trajectories.
4. **a04 — `07ddf1bad56c10c6b8b4d144293aec4f911df706`.** The system instruction
   now separates the sole `asset_id` tool argument from status fields returned by
   the tool and the user's final-answer format. It contains no case-specific facts
   or verifier labels. The user task prompt, environment descriptor, data, reward,
   parser, tool validation and incomplete-trajectory policy are unchanged. Both baselines completed 64/64 cases and passed the unchanged supervisor gate. GRPO training completed 29/32 first-collection rollouts; PPO-MC completed 30/32. The remaining trajectories made invalid tool calls and received no reward. Both collections were rejected before optimization; their second collections and checkpoint evaluations were not submitted. The final model download was slow but completed within its CPU deadline; it was never resubmitted.

Every earlier plan, reservation, failed receipt, diagnostic and collected log remains
retained. The corrections did not weaken a baseline gate, tool-error policy,
optimizer check or numerical algorithm. These are adaptive integration corrections
observed on the fixed qualification panel; resulting accuracy is not an independent
benchmark estimate.

## Verification of corrections

Each executed correction passed the relevant full gates: 22 task/source tests;
39 PPO-MC, 15 matched-study and 84 reader/refactor tests; six pinned configuration
checks and 62 matched-runtime tests. No required tests were skipped. Request
transport testing exercised a real Linux argument-size failure and a real Ray
worker with a large file-bound request. Harness tests exercised real stdio MCP,
model-input capture, hidden-label boundaries, reward delivery and process cleanup;
model responses in those tests were explicitly scripted. Remote execution supplied
separate live-model evidence. An initial targeted test invocation used the wrong
CPU environment and lacked Modal; rerunning through the correct repository test
entrypoint passed. This was not relabeled as a production failure or a passed test. A read-only
admission helper initially omitted earlier stage calls because it sorted preparation
snapshots with numeric status snapshots; its assertion stopped admission before any
new spending. Filtering the numeric snapshots restored all 15 prior calls before
a04 was reserved. An early a04 preparation collection captured a still-running CPU
call; a fresh completed collection was required before preflight. A live diagnostic
also encountered a not-yet-written stage receipt; this was a read timing issue, not
a failed submission. No lost acknowledgement was blindly retried.

## Final six-stage results

| Algorithm / stage | Result | Completed / attempted | Valid live-chain episodes | Confirmed rewards / correct | Actor / critic updates |
|---|---|---:|---:|---:|---:|
| GRPO baseline | Passed; baseline gate admitted | 64/64 | 64/64 | 64 / 0 | 0 / 0 |
| GRPO training | First collection rejected | 29/32 | 29/32 | 29 / 0 | 0 / 0 |
| GRPO checkpoint evaluation | Skipped after training failure | Not attempted | No evidence | No evidence | Not attempted |
| PPO-MC baseline | Passed; baseline gate admitted | 64/64 | 64/64 | 64 / 0 | 0 / 0 |
| PPO-MC training | First collection rejected | 30/32 | 30/32 | 30 / 0 | 0 / 0 |
| PPO-MC checkpoint evaluation | Skipped after training failure | Not attempted | No evidence | No evidence | Not attempted |

Each training stage planned two collections of eight questions × four rollouts
(64 rollouts total). Only the first 32 were attempted. GRPO had three failed
trajectories and four tool-error events; PPO-MC had two failed trajectories and two
tool-error events. Neither had an unclassified/ambiguous trajectory or recorded
model-transport error. Missing rewards were not filled with zero. GRPO failures
were on `asset-288`; PPO-MC failures were on `asset-288` and `asset-243`.
Some trajectories recovered to a valid later tool call, but the established
any-tool-error policy still correctly classified them as incomplete.

Both baselines had zero failed/incomplete cases, 128 successful tool invocations,
64 confirmed binary rewards, a single step-zero policy refresh, and no optimizer
activity. Their terminal responses were prose rather than whole-response JSON;
strict terminal correctness was **0/64** for each. This did not prevent baseline
integration admission: zero-reward live chains are valid. Training delivered 29
and 30 observed rewards respectively, all zero; correctness among rewarded
trajectories was 0/29 and 0/30. Missing rewards are excluded from these denominators.

The training workers refreshed the base policy at step zero, then stopped at
`Missing reward/trace or truncated trajectory; collection rejected`. No step-one
or step-two refresh, optimizer event, gradient step, or native checkpoint save
occurred. VLLM also logged handled `KeyError: name` parser diagnostics during GRPO
sampling; the retained traces identify the rejected trajectories' invalid tool
arguments. The parser was not changed.

Collection accounting recorded 32 rollouts in eight groups for each algorithm:
GRPO had 91 fragment rows and 2,678 generated tokens; PPO-MC had 90 rows and 2,731
generated tokens. Fragment token totals matched original generated-token totals.
GRPO had seven complete reward groups, **0/7 informative**, and one incomplete
group. PPO-MC had six complete groups, all constant zero, and two incomplete groups.
These are first-collection diagnostics, not 16 successfully trained groups.

PPO-MC constructed `LlamaForTokenClassification` with critic seed `1864040309`;
only `score.weight` and `score.bias` were newly initialized, with no load errors.
The recorded weight hash was
`6240acef6b84b7cc4a11b242e7d398fd4478666454629c7cd7afdd44069c9367` and bias hash
`df3f619804a92fdb4057192dc43dd748ea778adc52bc498ce80524c014b81119`.
These identify the initial critic head, **not a trained checkpoint**. Critic
prediction-error, whitening, clipping, KL/loss and gradient diagnostics are
unavailable because collection rejection preceded those computations. Offline
regression checks do not establish live numerical-update qualification.

No native step-two checkpoint exists for either algorithm. Therefore required-role
checkpoint hashes, exact-checkpoint reload, distinct fresh evaluation-worker proof,
and step-two policy-refresh evidence are **unavailable**. Volume inspection found
zero checkpoint tensor files. Base-model hashes above are not checkpoint evidence.
See [collection diagnostics](../evidence/phase3-qualification-20261122/a04/collection-diagnostics.json)
and the [stage summaries](../evidence/phase3-qualification-20261122/stage-summary.json).

## Hosted observability and retained evidence

The final independent query enumerated all **24 planned identities** across four
attempts: seven present, configuration-matched, finished runs and 17 confirmed
absent runs. All **35 numeric baseline metric comparisons** matched local records.
For the final candidate:

| Algorithm / stage | Hosted identity / review link | Hosted state and history | Reconciliation |
|---|---|---|---|
| GRPO baseline | [22cff789829df54b5b07fccf](https://wandb.ai/amulyap-stanford-university/reader-owned-reference/runs/22cff789829df54b5b07fccf) | Finished; metric step 0 | Configuration and seven numeric metrics match |
| GRPO training | [4f38e916c29cbb420bb3682e](https://wandb.ai/amulyap-stanford-university/reader-owned-reference/runs/4f38e916c29cbb420bb3682e) | Finished; collection-artifact step 1 | No local optimizer metrics or optimizer updates |
| GRPO checkpoint evaluation | `ca2467ba4cd5db3b77f1a57a` | Absent | Stage never submitted |
| PPO-MC baseline | [6474cbd5e9ba5ddb17974973](https://wandb.ai/amulyap-stanford-university/reader-owned-reference/runs/6474cbd5e9ba5ddb17974973) | Finished; metric step 0 | Configuration and seven numeric metrics match |
| PPO-MC training | [189912b5b07126a34fa6d3a5](https://wandb.ai/amulyap-stanford-university/reader-owned-reference/runs/189912b5b07126a34fa6d3a5) | Finished; collection-artifact step 1 | No local optimizer metrics or optimizer updates |
| PPO-MC checkpoint evaluation | `c82990cd12c1154468e661d3` | Absent | Stage never submitted |

**Hosted step 1 is not an optimizer update.** The locked upstream rollout adapter
logs a trace-fragmentation table and a one-row index to a capped 24-rollout artifact
during collection, before the matched validator rejects missing rewards. The
hosted mismatch-table counts (59 GRPO, 58 PPO-MC) equal fragment rows minus the
32 actual rollouts (91−32 and 90−32). Full local traces cover all 32 attempts;
the hosted compact artifact contains only the upstream limit of 24. The raw hosted
`[1]` and local numeric-metric `[]` step lists remain preserved, with the explanation
verified against the source-locked adapter and hosted artifact metadata. No
optimizer/loss step is inferred from W&B's finished state or history numbering.

Earlier finished but unqualified runs remain reviewable:
[a02 GRPO baseline](https://wandb.ai/amulyap-stanford-university/reader-owned-reference/runs/31d6cfe3315f9dd150962b87),
[a03 GRPO baseline](https://wandb.ai/amulyap-stanford-university/reader-owned-reference/runs/fb269c81691760fec1e19b3a),
and [a03 PPO-MC baseline](https://wandb.ai/amulyap-stanford-university/reader-owned-reference/runs/09016b45b7ae5947249a112c).
Review requires access to the existing W&B project. See the
[hosted reconciliation](../evidence/phase3-qualification-20261122/hosted-reconciliation-summary.json)
and [training-history explanation](../evidence/phase3-qualification-20261122/a04/hosted-training-history-reconciliation.json).
Independent readback used the hosted W&B API and matched the expected configuration,
not merely local `wandb_finished` events. Full project enumeration separately
established absence of unsubmitted runs; a failed individual API lookup alone was
not treated as proof of absence. Local evidence collection and independent hosted
reconciliation remain distinct claims.

The public evidence directory contains reviewed source/data/task inventories,
package inventories, stage summaries, safe causal-chain hashes, native checkpoint
manifests when present, and hashes of retained files. Raw traces, full runtime
configurations, credential-bearing requests, model weights and checkpoint tensors
are not published. Full receipts/configurations/logs remain in ignored local evidence
and explicitly owned remote volumes.

## Budget, resources and retention

All **22 known calls** are terminal and all **12 owned Apps** are stopped with
zero tasks. There are no unknown submissions or active owned calls. Both a04
training results explicitly confirm service/MCP cleanup; their checkpoints lists
are empty. Earlier owner cancellations remain separately recorded, including
stale interrupted-supervisor stage states reconciled against provider terminal
states. No unrelated call or resource was cancelled or deleted.

At 2026-11-22 04:01:17 UTC, Modal's completed-hour report attributed
**$5.76165025** of usage to owned campaign objects. This is not a settled invoice;
current-hour, delayed or unattributed charges remain unresolved. The terminal
App-lifetime calculation bounds compute/build exposure conservatively at **$26.95**:
$10/hour for complete preparation/preflight App lifetimes, and $20/hour for launch
App lifetimes (one $10/hour worker lane plus conservative supervision). Including
$7 aggregate retention and $3 collection/reconciliation, rounded up, leaves an
**inclusive $40 campaign hold and $260 unreserved balance**. The observed $5.76 is
inside that $40, not added again. This reconciled hold replaces earlier live-work
reservations; all reservation transitions are retained. The stop was the required
collection gate, not exhaustion of the spending ceiling.

Four explicitly owned volumes retain **25,802,624,741 bytes (24.0306 GiB)** of
models, generated synthetic data and evidence, with no checkpoint tensors. At the
checked storage rate, a full seven days at this size is approximately **$0.50**;
that estimate is separate from observed charges and the $7 allowance. Four owned
registries and the recorded images are also retained. See the
[cost closeout](../evidence/phase3-qualification-20261122/cost-closeout.json),
[provider reconciliation](../evidence/phase3-qualification-20261122/provider-reconciliation.json)
and [retained resources](../evidence/phase3-qualification-20261122/retained-resources.json).
The allocation was $125 training/supervision, $30 builds, $10 CPU preparation and
preflight, $7 storage, $3 collection/reconciliation, and $125 correction contingency,
for a single $300 ceiling. Credits, cache hits and free allowances were not assumed.
Before each billable action, the current provider rates/options and campaign
reservation were checked. The standard rate was $3.95/H100-hour, $0.0473/core-hour,
$0.008/GiB-hour memory, $0.09/GiB-month volume storage and $0.04/GiB egress.
One H100! lane, 16 cores, 196608 MiB, concurrency one, 3600-second stage limits,
600-second startup limit and the existing supervisor/build/CPU deadlines remained
in force. The supervisor conservatively accounts elapsed time at the worker ceiling.

Reservations were released only after known owned calls were terminal and Apps
stopped, with elapsed exposure recorded. Before a04, the three prior attempts each
retained $40 ($10 builds, $10 CPU, $10 training/supervision, $7 storage, $3 collection),
and a04 reserved a full $175, leaving $5 unreserved. The previous build hold was
reduced using actual completed preparation-App lifetimes of 402, 367 and 305 seconds;
charging their entire lifetimes at $10/hour gives $1.117, $1.019 and $0.847 respectively,
while $10 per attempt remained held. This is a conservative exposure calculation,
not observed provider billing. The reservation history is retained.

Aggregate retained volume capacity remained bounded by 300 GiB. Later profiles
reduced their allowance to account for prior retained volumes. The retention deadline
is **2026-11-29 01:54:46 UTC**. This is a review/cleanup obligation, not a claim that
automatic deletion has been scheduled. Only explicitly owned resources may be
removed; historical and unrelated resources were not touched. Reservation controls
are not a guaranteed provider spending cap.

## Conclusion and publication review

**Baseline integration is qualified for both algorithms; two-update training,
native checkpoint saving and fresh-worker reload are not qualified.** The concrete
stop condition is incomplete sampled trajectories under the unchanged tool-error
and complete-four-rollout-group rules. The model received the correct schema and
explicit argument instructions yet emitted forbidden extra arguments. No remaining
transport, package, ownership or cleanup defect was identified that would justify
another implementation correction. Replaying the unchanged failed collections for
a more favorable sample is not a correction. The campaign therefore stopped at its
required gate, preserving the remaining budget. Filtering/relabeling failures,
changing sampling, adding constrained decoding, replacing the parser/model, or relaxing the task
contract was not substituted to obtain a pass. Two updates are an operational qualification exercise,
not evidence of reliable learning improvement or resumed-training support.

The repository publication audit covered the working tree, index/staged patch,
all branch-introduced blobs including intermediate versions, commit metadata and
prepared PR text. Only the already reviewed synthetic negative-test credential/URL
fixtures were adjudicated. Newly published files and relevant history were manually
reviewed. No secrets, private history, restricted data, raw private traces, weights or
checkpoint tensors were committed. PR #4 remains draft and unmerged.
