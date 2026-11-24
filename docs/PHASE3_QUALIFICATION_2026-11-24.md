# Asset-status correction and six-stage qualification — 2026-11-24

**Operationally qualified for both algorithms.** Under the prospective
`asset-episode-v2` contract, GRPO and PPO-MC each completed baseline evaluation,
two-update training, native step-two checkpoint saving and fresh-worker checkpoint
evaluation on the alternate `asset-status-v1` task. Every one of the 384 qualification
episodes completed with one confirmed binary reward. This is operational evidence.
It does not show that training improved the policy: the corrected baseline already
scores 64/64, so the task has no headroom left to measure learning.

All paid work used the already approved aggregate US$300 Phase 3 ceiling. Nothing
here is a new allocation. Execution happened on 2026-11-24 UTC (the evening of
November 23 in the operator's local time). Resource names keep the `20261122` slug
chosen when the continuation was first planned.

The earlier campaign is preserved unchanged in the
[2026-11-22 report](PHASE3_QUALIFICATION_2026-11-22.md) and its
[evidence](../evidence/phase3-qualification-20261122/README.md). Attempts a01–a04
keep their original rewards, missing rewards, classifications, source identities
and unqualified status. Reviewed evidence for this continuation is in
[`evidence/phase3-qualification-20261124`](../evidence/phase3-qualification-20261124/README.md).

## Starting state

- Repository `amulyaparthasarathy/agent-lightning-grpo-ppo`, branch
  `phase3-reader-task-adaptation`, draft PR
  [#4](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/pull/4),
  base `5c069f7e9cc23d03050d13360ce43e27e47e54aa`.
- The branch head was `484e51d9182c3fa56849661270fcd3e8c58f08ca`, not the last
  reviewed `4f0d2973b10fbaff4233f5781833e8a6c600cf61`. An earlier operator session
  had already committed and pushed a first `asset-episode-v2` implementation,
  described in the [interim continuation note](PHASE3_CONTINUATION_2026-11-22.md).
  That commit was reviewed here against the raw traces rather than assumed correct.
- The same earlier session had reserved US$175 for an attempt a05 and begun a
  `prepare` submission. Its receipt was `unknown` with no call ID, and the ledger
  had not recorded it. See [a05](#a05-the-stranded-submission) below.

## Trace-grounded diagnosis

The retained private a04 evidence holds all 192 trajectories: 64 baseline and 32
first-collection training attempts per algorithm. They were re-read independently,
together with the exact vLLM chat template the model is served with
(`experiment/templates/tool_chat_template_llama3.1_json.jinja`) and the pinned
vLLM 0.22.0 `llama3_json` parser source.

**What the model actually sees.** While tools are advertised, the template rewrites
the first user turn. It prepends "please respond with a JSON for a function call"
and the tool schema ahead of the task's own request for "exactly one JSON object".
Tool results come back in an `ipython` turn as `{"output": "..."}`. AutoGen's final
reflection call sends no tools, so the template drops that preamble entirely. The
original system and user instructions are still present; nothing removed them.

**Greedy baseline, 128/128 trajectories.** Every episode made a valid lookup, made
the same lookup again on the second tool-enabled turn, then answered in prose at
the tool-less reflection: 18 completion tokens, finish reason `stop`, strict reward
zero. The old scripted success fixture exercised one call followed by a final
answer, a path the greedy model never took.

**Sampled training, 64 trajectories.**

| Observed shape | GRPO | PPO-MC | a04 treatment |
|---|---:|---:|---|
| Two valid calls, then prose or other non-conforming text at reflection | 24 | 25 | complete, reward 0 |
| One valid call, prose on the tool-enabled second turn | 5 | 3 | complete, reward 0 |
| Two valid calls, reflection echoes a tool-call envelope | 0 | 2 | complete, reward 0 |
| Extra-argument call, corrected call, reflection prose | 2 | 2 | **incomplete, no reward** |
| Extra-argument call twice, then JSON with an invalid enum | 1 | 0 | **incomplete, no reward** |

All five rejected trajectories supplied the answer fields as extra tool arguments.
Four of them recovered to a valid call. No trajectory had a model-transport error,
a truncated generation or a missing observation. The final failure was always
format: prose, an echoed call envelope, or an out-of-enum value. None was a
wrong-target or wrong-value answer inside a well-formed object.

So there were two separate defects. The harness discarded complete, recoverable
episodes, and the final-generation step never elicited the required object.

## Old and new episode policy

| Outcome | a01–a04 (historical) | `asset-episode-v2` (prospective) |
|---|---|---|
| Complete final response after normal tool use | strict reward 0 or 1 | unchanged |
| Invalid action, error observation, later complete final response | incomplete, no reward, collection rejected | strict reward 0 or 1; action, error and any recovery retained |
| Complete final response, malformed or incorrect | reward 0 | unchanged |
| Two observed invalid actions | incomplete, no reward | declared `policy_error_limit` termination, one confirmed zero, no invented answer |
| Model/API failure, server failure, cancellation, timeout, truncation, missing evidence, unconfirmed reward | incomplete, no reward | unchanged |

The strict whole-response verifier, the deterministic registry, dataset memberships,
dataset bytes and the environment hash
`d683b4ca993f4d014de260fe8ba04afa901b1488be036a633c37b4fec7ef4089` are unchanged.
GRPO and PPO-MC advantage, return, weighting, clipping, KL and optimizer code are
unchanged. Every sampled action stays in the trajectory and the policy loss.

## Implemented changes

**Commit `484e51d` (earlier session, reviewed here).**

- `episode_policy` (`asset-episode-v2`) is part of the task identity, so the task
  hash changes and every plan must be regenerated.
- A policy mistake is recognized only from explicit evidence: a typed error whose
  code the harness re-derives from the retained sampled arguments, or one of two
  exact AutoGen 0.7.5 branches (malformed argument JSON, unknown tool) matched to
  its sampled call. Any other tool error fails closed as incomplete.
- After two observed invalid actions the environment records a termination with
  the full context and posts one zero. No assistant answer is synthesized.
- At AutoGen's reflection call the same observed client appends one fixed user
  reminder asking for the JSON object. It carries no expected values, repairs no
  output and changes no decoding option. The reminder is recorded with the rest of
  the model input.
- Integration reports carry an episode outcome per trace, kept separate from
  successful tool chains. A terminal policy failure is never a valid chain.
- A `collection-only` plan purpose runs one real training collection per algorithm
  on a declared eight-question panel and stops before advantages and optimization.

**Commit `7f1e13d` (this session).** Review found one gap and one avoidable risk.

- *Non-object arguments.* A sampled call such as `"parameters": "asset-047"` is
  valid JSON that MCP `tools/call` cannot carry. The MCP client rejected truthy
  values with an untyped message that varies by library version, and AutoGen
  silently replaced falsy ones with `{}`. The v2 classifier would have treated the
  first as an infrastructure failure and rejected a whole sampled collection for
  one policy mistake. The harness now returns the typed `non_object_arguments`
  error for exactly this case, without contacting the server or rewriting the
  action, and classifies it only when the retained arguments really are a non-object.
- *Typed error transport.* Typed server errors now travel through MCP's
  raised-exception conversion, the path observed live with the remote MCP version
  in a04, rather than a directly returned `CallToolResult` that had only been
  exercised with the local MCP version.
- The remote preflight probe now checks the exact typed payload for every invalid
  call, so a version mismatch would stop the attempt before any GPU stage.
- `episode_policy.policy_error_codes` lists every recognized code.

Diagnostics (`json_syntax`, `format_compliant`, `target_correct`,
`content_correct`, policy-error count, recovery) are written to the private episode
record after the reward is computed. They are never model inputs and add no reward
terms.

## Offline verification

Run at `7f1e13d7bacb477c9e5e95a16868a2b0f2484ee6`, before any paid dispatch, using
the existing Python 3.12 CPU environments and cached pinned sources:

| Command | Result |
|---|---|
| `bash scripts/test_task_local.sh` | 26 tests passed |
| `bash scripts/test_reader_offline.sh` | 39 PPO-MC, 15 matched-study and 86 reader tests passed; accounting demo and historical artifact checks passed |
| `bash scripts/test_matched_training.sh` (documented cache variables) | 63 tests and six pinned-configuration checks passed |

No test was skipped. The same three gates were rerun on the final documentation
commit; see [PHASE3_VERIFICATION](PHASE3_VERIFICATION.md).

Evidence classes stay distinct. The harness tests run the production `Agent`,
`AssistantAgent`, observed client and workbench against the **real local stdio MCP
server** with a real loopback reward receiver; only the model responses are
**scripted**. Numerical tests pass **synthetic token fixtures** through the pinned
adapter and loss code. None of this is live-model evidence.

Branches covered with the production harness: one call then final JSON; two calls,
reflection, final JSON; two calls, reflection, prose; invalid call, typed error,
corrected call, final response (extra argument, malformed JSON, unknown tool, and
string, list and null arguments); two invalid calls and the declared terminal
failure; a successful lookup followed by an untyped backend error; truncation,
cancellation, missing output and transport failure at the second and reflection
steps; and failed reward delivery after a complete episode. The tests assert the
full model-visible message sequence, the reflection options (`tool_choice="none"`,
no tools, no JSON-output mode), hidden-label separation, exactly one reward post for
complete episodes and none for incomplete ones.

## Executed sources and plans

Both live attempts executed the same frozen source. Later commits are documentation
and reviewed evidence only.

```text
execution source: 7f1e13d7bacb477c9e5e95a16868a2b0f2484ee6
task SHA-256:     b64e9a1d40d6f903970ff95e3ca2b92ad0aac0788be792277a9badcfce7c27c2
environment:      d683b4ca993f4d014de260fe8ba04afa901b1488be036a633c37b4fec7ef4089

a06 (collection-only development)
  preparation plan: 6bb254084913ded03abc12e3a7b9f2556aa883536921e03c65f8cfe7ad794e5d
  stage plan:       6c1e259e171e9eebfa32fb438dd80f5d685b04e91e51a9c7f8c419e255c94449
  remote image:     im-UTGiucVTEF7MLNN45fV1Yh

a07 (six-stage qualification)
  preparation plan: 005ef25125787ae1a3ec099ab64c68f166682a2d15a4a5236603993ddf6f6d3a
  stage plan:       2a9b1cc811421c6eca122e344e1cc3a60cd210a3776603c7b8c9a4bd2d44bf21
  remote image:     im-q55YWEYSIoAOgQzw37h4f3
```

Model, parser, template, Agent Lightning and VERL pins are those of the earlier
report. Resources were one `H100!` lane, concurrency one, 16 CPU cores and
196608 MiB per stage worker, 3600-second stage limits and a 600-second startup
limit. Both remote preflights passed, including the real stdio probe with exact
typed error payloads and the unchanged six reviewed dependency exceptions. The
[identity manifest](../evidence/phase3-qualification-20261124/identities.json)
records source, configuration, packaged-content and data hashes separately.

## Live development collection (a06)

Declared before execution: source `7f1e13d`, the plans above, the first eight
training-smoke cases (`asset-011`, `-041`, `-250`, `-271`, `-075`, `-246`, `-111`,
`-039`), four rollouts each, training sampling at temperature 1, run seed 9001 with
the existing seed streams, the standard stage bounds, and zero optimizer updates.
None of these cases is in the 64-case qualification panel.

The run used the production training route: the real rollout adapter, reward
admission, token conservation, fragmentation and four-member grouping. It then
stopped before advantage computation.

| | GRPO | PPO-MC |
|---|---:|---:|
| Rollouts attempted / complete / rewards confirmed | 32 / 32 / 32 | 32 / 32 / 32 |
| Final responses that are contract-format JSON | 32 | 32 |
| Reward 1 | 32 | 31 |
| Policy errors (code) | 0 | 1 (`unknown_tool`) |
| Recovered / terminal policy failures / incomplete | 0 / 0 / 0 | 0 / 0 / 0 |
| Generated tokens / fragment rows | 2,112 / 96 | 2,112 / 96 |
| Groups / informative / constant-one | 8 / 0 / 8 | 8 / 1 / 7 |
| Episodes with a valid live tool chain | 32 | 32 |
| Actor / critic optimizer updates | 0 / 0 | 0 / 0 |

The audit trail has no optimizer, gradient, advantage or checkpoint event, one
step-zero weight refresh and an empty checkpoint directory for each algorithm.
Hosted W&B shows each run finished with one artifact-history row at step 1 holding
only bookkeeping keys. That row is the upstream collection artifact, not an update.

The single policy mistake was a second-turn call to a mis-capitalized tool name.
It was retained, classified from the matching AutoGen observation, and followed by
a well-formed final object with wrong values, which the strict verifier scored
zero. Under the a04 policy that one rollout would have rejected the collection.
Recovery by a later successful call and the two-error terminal limit were **not
sampled live**; they remain verified offline only.

Development was accepted: all rollouts accounted for, no missing reward or
token/context gap, and contract-correct JSON produced by the live sampled policy.
The run is not qualification evidence.

## Six-stage qualification (a07)

| Algorithm / stage | Result | Completed / attempted | Valid live-chain episodes | Confirmed rewards / correct | Actor / critic updates |
|---|---|---:|---:|---:|---:|
| GRPO baseline | Passed; integration gate admitted | 64/64 | 64/64 | 64 / 64 | 0 / 0 |
| GRPO training | Two collections, two updates, checkpoint saved | 64/64 | 64/64 | 64 / 63 | 2 / 0 |
| GRPO checkpoint evaluation | Passed in a fresh worker | 64/64 | 64/64 | 64 / 64 | 0 / 0 |
| PPO-MC baseline | Passed; integration gate admitted | 64/64 | 64/64 | 64 / 64 | 0 / 0 |
| PPO-MC training | Two collections, two updates, checkpoint saved | 64/64 | 64/64 | 64 / 63 | 2 / 2 |
| PPO-MC checkpoint evaluation | Passed in a fresh worker | 64/64 | 64/64 | 64 / 64 | 0 / 0 |

Every stage ran once, in its own container, with trainer exit code 0 and confirmed
service and MCP cleanup. There were no retries, cancellations or skipped stages.
Stage intervals were 319.0, 484.0, 394.8, 375.4, 954.5 and 496.8 seconds
(3,024.5 seconds in total); these are supervisor submission-to-completion times,
not billed GPU seconds.

All 384 episodes have outcome `complete_final`. There were no terminal policy
failures and no incomplete executions. Each training stage ran two complete
collections of eight questions and four rollouts.

| Training collection | Rollouts | Fragment rows | Generated tokens | Groups informative / constant |
|---|---:|---:|---:|---:|
| GRPO 1 | 32 | 96 | 2,116 | 1 / 7 |
| GRPO 2 | 32 | 95 | 2,111 | 0 / 8 |
| PPO-MC 1 | 32 | 96 | 2,116 | 0 / 8 |
| PPO-MC 2 | 32 | 95 | 2,084 | 1 / 7 |

Fragment token totals equal the original generated-token totals for every rollout.
The two zero rewards in training were genuine task failures, each with its tokens
and likelihoods in the batch: a GRPO final object with a misspelled key, and a
PPO-MC rollout that answered in prose on the tool-enabled second turn. PPO-MC also
sampled one unknown-tool call; that episode went on to a correct final object and
earned reward 1 with its policy error retained.

### Optimizer counts and numerical diagnostics

| Update | Role | Optimizer count | Pre-clip gradient norm | Post-clip norm | Loss |
|---|---|---:|---:|---:|---:|
| GRPO 1 | actor | 1 | 0.2708 | 0.2708 | `pg_loss` 1.4e-9 |
| GRPO 2 | actor | 2 | 0.0 | 0.0 | `pg_loss` 0.0 |
| PPO-MC 1 | critic | 1 | 279.0 | 1.0 | `vf_loss` 3.3105 |
| PPO-MC 1 | actor | 1 | 0.7458 | 0.7458 | `pg_loss` 9.5e-5 |
| PPO-MC 2 | critic | 2 | 509.8 | 1.0 | `vf_loss` 3.8388 |
| PPO-MC 2 | actor | 2 | 3.5052 | 1.0 | `pg_loss` 0.0103 |

All values are finite. Policy refreshes were recorded at steps 0, 1 and 2 for both
training stages, at step 0 for both baselines and at step 2 for both checkpoint
evaluations. Evaluation stages recorded no optimizer event. Reported
`actor/ppo_kl`, policy clip fractions and value clip fraction were 0 at both steps,
as expected for a single optimizer call per collection. Learning rates were 1e-6
(actor) and 1e-5 (critic).

Read these numbers for what they are:

- **GRPO update 2 had an exactly zero gradient.** All eight groups had constant
  reward, so every group-relative advantage was zero. The optimizer was called and
  counted, but no task signal was applied. Update 1 used one informative group.
- **The PPO-MC critic was poor.** Its head is freshly initialized (seed
  `1864040309`; only `score.weight` and `score.bias` are new). Pre-update prediction
  error was MSE 6.62 on the first collection, where targets had zero variance and
  explained variance is undefined, and MSE 7.59 with explained variance −131.7 on
  the second. Critic gradients were clipped from norms of 279 and 510.

### Native checkpoints and fresh-worker reload

| Algorithm | Roles and bytes | Checkpoint identity SHA-256 | Training → evaluation container |
|---|---|---|---|
| GRPO | actor, 40,146,400,251 | `2e6cf21b255a0bc46f4629deed908c887c1349987115dae69cd35b065a45bdd7` | `ta-01M85Q6T8WV1FJ5FFBGR7VS44R` → `ta-01M85QNVC7X4XNGSM40K6A2CQR` |
| PPO-MC | actor 40,146,400,251 + critic 38,570,427,714 = 78,716,827,965 | `f8ad498fa245d84e66fd95cfb06fbdf57190a4aa60ff98de5904d3047f42ec3d` | `ta-01M85RDS63WV07KR494CV1CH5R` → `ta-01M85SB6H5D9W0KYMBZ39812QR` |

Each role inventory contains the native `model_`, `optim_` and `extra_state_`
files plus tokenizer and configuration files; per-file bytes and hashes are in the
stage records. Each checkpoint evaluation verified the recorded identity before
starting, restored `global_step_2` from the training output, refreshed the policy
at step 2, ran in a different container and call from training, and reported the
same checkpoint hash. All six stage containers are distinct. No weights or
checkpoint tensors are published.

## Hosted W&B reconciliation

An independent read of the hosted project enumerated all ten continuation
identities: two never-submitted a05 runs (absent), two a06 development runs and six
a07 runs. Presence was established by full project enumeration, not by local
finish events.

| Algorithm / stage | Hosted run | State; history steps | Reconciliation |
|---|---|---|---|
| GRPO baseline | [79f9982122fcd9501dff1ec7](https://wandb.ai/amulyap-stanford-university/reader-owned-reference/runs/79f9982122fcd9501dff1ec7) | finished; `[0]` | configuration and 7 metrics match |
| GRPO training | [db8f80ac687ad9f4cdcc0c5e](https://wandb.ai/amulyap-stanford-university/reader-owned-reference/runs/db8f80ac687ad9f4cdcc0c5e) | finished; `[1, 2]` | configuration and 196 metrics match |
| GRPO checkpoint evaluation | [011b8c28be3bec0dd03c9144](https://wandb.ai/amulyap-stanford-university/reader-owned-reference/runs/011b8c28be3bec0dd03c9144) | finished; `[2]` | configuration and 7 metrics match |
| PPO-MC baseline | [60fe3eaad9748df5561e3425](https://wandb.ai/amulyap-stanford-university/reader-owned-reference/runs/60fe3eaad9748df5561e3425) | finished; `[0]` | configuration and 7 metrics match |
| PPO-MC training | [b45afdfd791a2c68104ad3f0](https://wandb.ai/amulyap-stanford-university/reader-owned-reference/runs/b45afdfd791a2c68104ad3f0) | finished; `[1, 2]` | configuration and 230 metrics match |
| PPO-MC checkpoint evaluation | [080b99cb08f17205cb0cf4b3](https://wandb.ai/amulyap-stanford-university/reader-owned-reference/runs/080b99cb08f17205cb0cf4b3) | finished; `[2]` | configuration and 7 metrics match |

All 454 local numeric metric values equal their hosted values, and hosted step
lists equal local step lists. In the training runs steps 1 and 2 are the two
optimizer updates, confirmed separately by the local audit events. In the a06
development runs
([5a6977e90e34e0b83113f30a](https://wandb.ai/amulyap-stanford-university/reader-owned-reference/runs/5a6977e90e34e0b83113f30a),
[d376b6d80edcc1ae7c41ab5e](https://wandb.ai/amulyap-stanford-university/reader-owned-reference/runs/d376b6d80edcc1ae7c41ab5e))
hosted step 1 is only the collection artifact row; no optimizer step is inferred
from it. Review requires access to the existing W&B project.

## Local re-verification

The collected a07 evidence (1,098 files, 71,979,570 bytes, no collection errors)
was re-validated without remote access. The collection validator reproduced
`operational and harness evidence observed; learning success not established` with
no qualification gaps. Trace joins recomputed from the raw private traces equal the
workers' integration reports for all six stages. Checkpoint identities match their
completion records, and both reloads used the same checkpoint in a distinct
container. The same procedure reproduced the a06 development result.

## Attempts, failures and corrections

1. **a01–a04** are unchanged and remain unqualified. See the earlier report.
2. <a id="a05-the-stranded-submission"></a>**a05 — planned at `484e51d`, never
   executed.** The earlier session's `prepare` client stopped during image
   resolution, before any function was spawned. Read-only reconciliation found an
   app with zero functions and zero tasks, no `entered` or `result` registry row,
   an empty volume 43 hours after the 3600-second preparation limit, and one
   billing row of US$0.00105. A durable stop marker was placed with the public
   `stop-owned` command, run from the exact planned source. The submission was not
   retried, and its two planned W&B identities are absent. The reservation was
   reduced from US$175 to a US$1 hold.
3. **a06 — `7f1e13d`.** Development collection; completed and accepted.
4. **a07 — `7f1e13d`.** Qualification; all six stages completed at the first
   submission.

Operational notes, none of which touched remote work: the first local download of
the a07 evidence was cut off twice by local command time limits. The partial
copies were kept under separate names and the full collection was rerun into a
fresh directory as a detached process. Collection is read-only, so no submission
was repeated. The first full provider-inventory helper hung on a client-library
interaction and was restructured before use.

## Budget, resources and retention

One ledger covers the whole Phase 3 campaign under the single US$300 ceiling.
Before any new paid dispatch, a read-only provider reconciliation confirmed all 22
earlier calls terminal, no running app, 25,802,624,741 retained bytes and the
earlier US$5.76 of observed usage. It also found and resolved a05.

At 2026-11-24 04:02:53 UTC all **36 known calls** were terminal and no app in the
workspace had a running task. Modal's completed-hour usage report attributed:

| Attempt | Observed usage | Conservative exposure bound | Hold |
|---|---:|---:|---:|
| a01–a04 (earlier campaign) | US$5.76165 | US$26.95 | US$40 |
| a05 (never executed) | US$0.00105 | — | US$1 |
| a06 (development) | US$1.39074 | US$5.91 | US$8 |
| a07 (qualification) | US$5.15639 | US$18.14 | US$25 |
| **Total** | **US$12.30982** | | **US$74** |

Every usage row since the campaign began is attributed to an owned object; the
unattributed remainder is zero. Observed usage is a completed-hour report, not a
settled invoice, and it excludes volume storage. The exposure bounds charge whole
App lifetimes at US$10/hour for preparation and preflight and US$20/hour for the
supervised launch. Each hold rounds its bound up with retention, collection and
delayed-billing margin. **The holds include the observed usage; they are not
additional spending. US$226 remains unreserved.** Holds are operating
reservations, not a provider spending cap.

Reservations were made before dispatch and reduced only after every owned call
was terminal: a05 US$175 → US$1, a06 US$70 → US$8, a07 US$175 → US$25. The a06
reservation was sized to its plan, which the supervisor bounds to two stage-hours.
The full history is in the
[ledger](../evidence/phase3-qualification-20261124/spending-ledger.json),
[cost closeout](../evidence/phase3-qualification-20261124/cost-closeout.json) and
[provider reconciliation](../evidence/phase3-qualification-20261124/provider-reconciliation.json).

Retained, explicitly owned resources: seven volumes and seven registries (a01–a07)
holding **157,612,506,976 bytes (146.79 GiB)**, of which 118,863,248,277 bytes are
the two a07 native checkpoints (11 `.pt` files, including trainer state). The a05 volume is empty. None
of the 19 owned apps has a running task; those the provider still lists report
state stopped. At US$0.09/GiB-month, seven more days at this size is
about **US$3.08**, covered by the holds.

Retention obligations:

- a01–a04: review by **2026-11-29 01:54:46 UTC** (unchanged).
- a05–a07: review by **2026-12-01 03:34:30 UTC**, seven days after a07 completed.

These are review and cleanup obligations. Nothing was deleted in this
continuation and no deletion is scheduled. Removing the a07 volume would remove
the only copies of the qualified checkpoints. See
[retained resources](../evidence/phase3-qualification-20261124/retained-resources.json).

## Conclusion and limitations

**Both problems are corrected, and both algorithms are operationally qualified on
the alternate task** for baseline → two-update training → native checkpoint →
fresh-worker evaluation.

Limitations:

- **No learning claim.** Baseline and checkpoint accuracy are both 64/64 for both
  algorithms. Two updates were an operational exercise, and a ceiling this low on
  difficulty cannot show improvement or regression. GRPO saw 1 informative group
  in 16 and one update with zero gradient. PPO-MC made nonzero updates under a
  poorly calibrated critic.
- **The task is now easy for this model.** Sampled training was 126/128 correct.
  The reflection reminder changes the model-visible context, so results under
  `asset-episode-v2` are not comparable with a01–a04 as a measure of policy change.
- **Two v2 branches have no live observation.** Recovery by a later successful
  call, and the two-error terminal limit, occurred in neither a06 nor a07. They
  are verified only offline with scripted inference. Live coverage of policy
  mistakes is two sampled actions, both unknown-tool calls. No extra-argument or
  non-object call was sampled under the revised instructions.
- **Every sampled rollout must still be complete.** A truncated generation, a
  timeout or an untyped tool failure in any one of 32 rollouts rejects that
  collection by design. None occurred in 192 sampled rollouts here; that is not a
  guarantee for other seeds.
- **The 64-case panel is qualification and regression evidence.** It was inspected
  across a01–a04 and again here. It is not an untouched generalization benchmark.
  Prompt and harness changes were made from the a04 traces and tested on training
  cases before the panel was run under v2.
- **Not established:** resumed training, more than two updates, other models or
  parsers, and reproducibility across seeds. vLLM scheduling and GPU kernels are
  not bitwise deterministic; the two paired first collections sampled different
  failures.
- Package admission still relies on the six reviewed dependency exceptions. Image
  layers came from cache, so no independent cold-build claim is made.

## Publication review

The publication audit covered the working tree, index, every blob introduced on
the branch and commit metadata. Its only findings are the two previously reviewed
synthetic negative-test fixtures. Published evidence is limited to identities,
hashes, counts, audit-event summaries, hosted metric comparisons and trajectory
shape aggregates. Raw traces, prompts with observations, runtime requests and
configuration contents, weights and checkpoint tensors remain in ignored local
storage and explicitly owned remote volumes. PR #4 remains draft and unmerged.
