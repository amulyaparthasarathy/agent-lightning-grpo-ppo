# Reader customization and task adaptation

Phase 3 implements two tasks through the same matched training workflow. The new
`asset-status-v1` task is **operationally qualified for both algorithms** under the
`asset-episode-v2` episode contract: baseline, two-update training, native
checkpoint saving and fresh-worker checkpoint evaluation all completed, with 64/64
heldout cases at baseline and after reload. That is operational evidence, not
evidence of learning; the corrected baseline leaves no headroom. See the
[dated report](PHASE3_QUALIFICATION_2026-11-24.md).

The four earlier attempts failed, and how they failed is the main lesson of this
guide. They are preserved unchanged in the
[2026-11-22 report](PHASE3_QUALIFICATION_2026-11-22.md) under their original
episode policy. The [engineering checklist](PHASE3_ENGINEERING_CHECKLIST.md) condenses
what they taught into checks to run before a new attempt. See also
[offline verification](PHASE3_VERIFICATION.md). The local
harness demonstration uses explicitly scripted inference and does not establish
live-model behavior.

The third task, `recovery-readiness-v1`, uses three tools and a verifier that checks both the decision and its supporting evidence. The original task, calibration method and offline checks describe the `recovery-episode-v1` contract. Later, both routes ran live development calibration. GRPO completed the A03 two-update qualification and the P02 16-update pilot, including fresh-worker reloads. [Task](PHASE4A_RECOVERY_TASK.md) · [Calibration method](PHASE4A_CALIBRATION.md) · [Offline checks](PHASE4A_VERIFICATION.md) · [Live calibration](PHASE4B_LIVE_CALIBRATION.md) · [A03](PHASE4C_GRPO_CAMPAIGN_CLOSEOUT.md) · [P02](PHASE4C_FUNCTIONAL_PILOT_CLOSEOUT.md)

On the 24 development cases, P02's terminal greedy policy always answered `not_ready`. Strict successes rose from 9/24 to 12/24 while decision accuracy stayed at 12/24. Recovery PPO-MC live training and reload remain unqualified. The full recovery study is unexecuted, and witness-selection SFT remains deferred. The `collection-only` procedure below is specific to the asset task. [Paired cases and verifier analysis](PHASE4C_PILOT_LEARNING_DIAGNOSIS.md) · [Study protocol](PHASE4C_LEARNING_STUDY_PROPOSAL.md) · [Draft PR #14](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/pull/14)

The completed calculator qualification remains bound to execution source
`5daff64cefd348fb731094cf465b223ddceb5fb2` and plan SHA-256
`a3128740ba4c2133e049bd80e9bdb471aa1ad56df3158479d1594b4e0ab23cc3`.
Both algorithms completed baseline, two-update training, native checkpoint saving,
and fresh-worker evaluation reload. GRPO had **0/16 informative reward groups** and
no observed task-directed policy-gradient signal. PPO-MC made nonzero updates but
had poor critic diagnostics. Operational validity was established; reliable learning
improvement was not. Phase 3 commits did not produce those results. The historical
20-update, three-paired-seed experiment and its learning guide remain separate.

## The two workflows

| Component | Calculator (default) | Asset status |
|---|---|---|
| Task ID | `calculator-v1` | `asset-status-v1` |
| Data | Existing acquired Calc-X selection, unchanged | Versioned deterministic synthetic registry, 256 train / 64 heldout / 8 smoke assets |
| Harness | `observed_calc_agent.Agent` | `asset_agent.Agent` |
| Request | Existing question and `### ANSWER` instructions | Asset request and strict JSON instructions |
| MCP | `mcp_server_calculator`, `calculate(expression)` | Local `asset_server.py`, `lookup_asset(asset_id)` |
| Final parser | Existing answer extractor | Entire response must be one JSON object |
| Verifier | Pinned upstream scalar comparison, tolerance 0.01 | Exact field equality against independently regenerated trusted facts |
| Reward | Existing binary scorer | Exactly one binary terminal-correctness reward; no shaping |
| Local check | Real calculator stdio probe | Real asset stdio probe and scripted harness demonstration |

Both tasks retain the supported Llama 3.2 3B Instruct model/tokenizer locks, GRPO and
this lab's PPO-MC, sampling, group size four, two-update qualification, generated-token
conservation, observation masking, fragmentation accounting, equal-rollout weighting,
PPO-MC terminal Monte Carlo targets and critic/whitening behavior. Ownership,
reservations, supervision, cancellation, deadlines, optimizer/refresh checks, native
checkpoint algorithms and fresh-worker reload are shared.

## Run the local checks

Use the existing Python 3.12 workspace. Install only the locked CPU harness group in
its separate environment; the ordinary client never needs CUDA/vLLM. Installation
may fetch Python packages. The commands below perform no paid remote work.

```bash
UV_PROJECT_ENVIRONMENT=.venv-reader-cpu uv sync --locked --only-group reader-harness
bash scripts/test_task_local.sh
.venv-reader-cpu/bin/python scripts/task_local.py generate --data outputs/asset-v1
.venv-reader-cpu/bin/python scripts/task_local.py verify --data outputs/asset-v1
.venv-reader-cpu/bin/python scripts/task_local.py probe --out outputs/asset-probe.json
.venv-reader-cpu/bin/python scripts/task_local.py probe --task calculator-v1 --out outputs/calculator-probe.json
.venv-reader-cpu/bin/python scripts/task_local.py scripted --out outputs/asset-scripted.json
```

Generation and receipts refuse to overwrite existing output. Choose fresh paths when
repeating them. `--help` is dependency-light; generation/verification require PyArrow,
and probes require the locked MCP/AutoGen group. `scripted` deliberately imports a
**test-only** helper under `tests/`, excluded from the production image. It runs the
production Agent, AssistantAgent, traced client and real MCP server; only model
responses and the reward destination are test boundaries. Its receipt and traces say
`scripted-inference`, and live evidence validation rejects them.

This is distinct from `reader_training.py preflight`, which runs a **billable remote
CPU worker** in the training image and checks model bytes, dependencies, tokenizer,
HF/W&B access and the selected task probe. Do not run it without an approved
spending envelope and a ledger reconciled against the provider.

## Asset data, tool and answer contracts

The fixed recipe is `asset-status-v1/2026-11-20/329`. SHA-256 derives each status from
recipe, asset ID and field name; a separate SHA-256 ordering selects split membership.
IDs `asset-000` through `asset-327` do not encode status. The registry and ordered
cases are regenerated independently during verification. Train smoke is the first
16 training cases; the eight smoke-evaluation assets are separate from both training
and the 64-case heldout panel. Splits are by asset identity, so paraphrasing an asset
cannot create an independent evaluation example.

The real stdio MCP server exposes exactly one tool:

```json
{"name":"lookup_asset","inputSchema":{"type":"object","properties":{"asset_id":{"type":"string","pattern":"^asset-[0-9]{3}$"}},"required":["asset_id"],"additionalProperties":false}}
```

Valid calls return a JSON text observation containing exactly `asset_id`,
`service_state` and `backup_health`. Service states are `operational`, `degraded`,
`offline`; backup health is `healthy`, `stale`, `failed`. Invalid calls produce an MCP
error result whose text is a typed JSON object,
`{"code": ..., "error_type": "asset_policy_action_v2", "instruction": ...}`. Missing,
non-string or extra arguments give code `invalid_arguments`. Malformed IDs give
`invalid_asset_id`; well-formed unknown IDs such as `asset-999` give
`unknown_asset`. There is no random or external inventory lookup. Protocol output
uses stdout; library diagnostics use stderr.

The child starts using an absolute interpreter and `-I`, then clears its environment
before importing task code. Only `PYTHONUNBUFFERED` and `PYTHONDONTWRITEBYTECODE`
remain. No labels, HF/W&B credentials or gateway keys reach it. Linux parent-death
signaling and owned PID/start-time receipts cover separately created MCP process
groups; cleanup never uses a global process-killing command. This implementation
requires the Linux/WSL2 environment used by the lab. Receipts are flushed and atomically
published; interrupted `.pending` files and malformed/ambiguous receipts never authorize
a signal. Cleanup continues through independently valid receipts and reports unresolved
ownership separately from successful group termination. Launcher/probe failure receipts
retain the original execution error and cleanup anomalies; unresolved cleanup is not a
successful result. Reaping belongs to the direct parent. Preserve anomalous receipts
for diagnosis; do not guess a PID or clear ownership records to retry.

Rows intentionally retain the upstream-compatible string fields `question`, `result`,
`case_id`, `split`. `QUESTION` transports the request; `RESULT` transports a canonical
JSON verifier label. This is a common row schema, not a dependency on calculator
scoring. The model sees the explicit request (including target ID), structured-output
instructions, discovered tool schema and returned tool observations. The harness
never stringifies the whole row or appends the privileged `result`. The verifier first
checks that the label agrees with the independently regenerated registry, then scores
the terminal response. A corrupted tool observation cannot redefine the truth.

Only a whole JSON object with exactly the three required string fields is accepted.
Whitespace around it is allowed. Duplicate keys (including escaped equivalents),
extra fields, wrong types or enums, wrong target IDs, missing fields, code fences,
multiple objects, NaN and surrounding prose are rejected. A genuinely completed but
incorrect/malformed response gets zero. No partial credit is given for a tool call.
Empty/missing terminal text, truncation, cancellation, timeout, infrastructure errors,
or failed reward delivery remain failed/incomplete, not valid terminal-zero samples.
The harness accepts a terminal assistant text response only after a `stop` finish
reason and posts once. Failed delivery retains `reward_posted: false`; no retry can
silently fabricate delivery. JSON is requested through the prompt, not an unverified
`response_format` option.

For example (illustrative, not a model transcript):

```text
User: Look up asset asset-047 and report its service state and backup health.
Model: lookup_asset({"asset_id":"asset-047"}), call ID c1
MCP: {"asset_id":"asset-047","backup_health":"stale","service_state":"offline"}
Next model input: the exact observation, matched to c1
Model: {"asset_id":"asset-047","service_state":"offline","backup_health":"stale"}
Verifier: compares against trusted registry record; reward 1 only on exact equality
Gateway: one successful terminal reward delivery
```

`data_manifest.json` binds ordered IDs, actual file hashes, registry bytes, the
versioned environment, tool schema, prompt, output and reward contracts. Task identity
also hashes generator/parser/verifier, server, harness and shared tracing source.
Changed bytes, duplicate/conflicting records, reordered cases, split leakage and
incompatible manifests fail verification even if an attacker updates file hashes.
Synthetic environment data are real task inputs; they do not relax model weight
admission or become simulated model evidence. Asset preparation never downloads the
calculator archive or claims historical calculator-data equivalence.

## Episode outcomes: asset-episode-v2

The task identity includes `episode_policy`. The dataset ID, deterministic recipe,
registry, ordered membership, dataset bytes and environment hash are unchanged.
Source and task hashes changed, so plans from before the revision are stale.
Attempts a01–a04 keep their original policy, missing rewards and status.

| Outcome | Treatment |
|---|---|
| Complete final response after normal tool use | Strict independent verifier; reward 0 or 1 |
| Invalid action, error observation, later complete final response | Same strict reward; the action, error and any recovery stay in the trajectory |
| Complete final response that is malformed or incorrect | Complete task failure; reward 0 |
| Two observed invalid actions | Declared `policy_error_limit` termination; one confirmed zero; no invented answer |
| Model/API or server failure, cancellation, timeout, truncation, missing evidence, unconfirmed reward | Incomplete; no reward; the collection is rejected |

### A policy mistake is not an infrastructure failure

Under the original policy every tool error made the episode incomplete. In a04,
five sampled rollouts out of 64 put answer fields into the tool arguments. Four of
them then made a correct call. All five were discarded, and each one rejected its
whole 32-rollout collection, so neither algorithm reached an optimizer step. The
calculator workflow never had this rule: it keeps tool-error events without
invalidating the episode.

A policy mistake is a sampled action that violates the tool contract. It is the
model's own behavior, it belongs in the trajectory, and its tokens belong in the
loss. An infrastructure failure tells you nothing about the policy. The harness
separates them by evidence, never by a catch-all handler:

- The server raises only for explicit input validation and returns a typed
  `asset_policy_action_v2` error with an allowlisted code. The harness then
  re-derives the same code from the retained sampled arguments. An error string
  alone is not enough.
- AutoGen 0.7.5 answers malformed argument JSON and unknown tool names itself,
  without calling MCP. Those two exact observations are recognized only together
  with the matching sampled call.
- Valid JSON arguments that are not an object, such as `"asset-047"` or
  `["asset-047"]`, cannot be carried by MCP `tools/call`. The MCP client rejects
  truthy values with a message that depends on the library version, and AutoGen
  replaces falsy ones with `{}`. The harness returns the typed
  `non_object_arguments` error for exactly this case, without contacting the
  server or rewriting the action.
- Anything else stays incomplete: an untyped MCP error, a parser exception, a dead
  server, a timeout. `McpWorkbench` converts every exception into `is_error=True`,
  so `is_error` by itself proves nothing.

Extra arguments are rejected, never stripped. `episode_policy.policy_error_codes`
lists every recognized code. The remote preflight probe checks the exact typed
payloads in the training image, because the remote MCP version differs from the
local one.

Counting note: AutoGen never calls the workbench for an unknown tool, so that
mistake appears in `policy_errors` and the `action_outcome` record but not in
`tool_calls` or `tool_errors`.

### Recovery and explicit terminal failure

After an invalid call the model sees the error observation and may try again. A
complete final answer is scored by the same strict verifier whether or not a
mistake came first. `recovered` is recorded when a successful call follows an
error. The reward does not depend on it.

Two observed invalid actions end the episode. The environment writes a
`termination` record with the full context and both error observations, posts one
zero, and generates nothing further. That zero is legitimate because it follows a
declared limit and an observed trajectory with all its sampled tokens. A timeout
is not a horizon, and a missing answer is never filled in.

### Inspect the full effective prompt

Reading the instruction in source does not tell you what the model saw. For this
task the served chat template rewrites the first user turn while tools are
advertised. It prepends "please respond with a JSON for a function call" and the
tool schema, directly ahead of the task's own request for "exactly one JSON
object". Tool results arrive as `{"output": "..."}` in an `ipython` turn. In a03
the model merged the two JSON instructions and passed answer fields as tool
arguments. Render the template for a real message list, and read the recorded
`messages`, `tools`, `generation_options` and `client_generation_defaults` of each
model call in the private trace, before changing a prompt.

Read the exact final text too. "Reward zero" hides whether the failure was prose,
a code fence, an extra key, a wrong target or a wrong value. Each episode record
carries `json_syntax`, `format_compliant`, `target_correct` and `content_correct`.
They are written after the reward is computed. They are never model inputs and add
no reward terms.

### Reflection and final output

With `reflect_on_tool_use=True`, AutoGen makes a last call with
`tool_choice="none"` and no tools once tool iterations are used up. It keeps the
whole context: the pinned implementation does not drop the system or user message.
What changes is the template. Without tools the function-call preamble disappears,
and after `ipython` turns the greedy model simply described the result in prose,
128 times out of 128.

The correction keeps the agent in charge of the answer. The system message states
the whole-response JSON contract, and at the reflection call the same observed
client appends one fixed user reminder after the complete tool context. The
reminder contains no expected values. Nothing repairs or rewrites the output, no
second model is called, and decoding is unconstrained, so sampled tokens and their
likelihoods are exactly what the numerical runtime already accounts for. The
reminder is recorded as part of that call's input.

On the second tool-enabled turn the greedy model still repeats its lookup. That is
its own choice under the template, it costs one redundant call, and it is left in
the trajectory.

### Greedy evaluation does not test sampled training

Baseline evaluation runs at temperature 0. Training samples four rollouts per
question at temperature 1. a04 passed its baseline gate with 64/64 complete
episodes and then lost both first training collections to actions the greedy
policy never takes. Validate the sampled boundary before paying for optimization.

Set `"purpose": "collection-only"` in a fresh asset profile. The plan then declares
the first eight training-smoke cases, the seed streams, source and task hashes and
bounds, and schedules one training-stage collection per algorithm. The production
adapter checks rewards, context, token conservation, fragmentation and all eight
four-member groups, then stops before advantages or any optimizer call. Completion
requires no optimizer, gradient or checkpoint event and only the step-zero refresh.

```bash
uv run --locked python scripts/reader_training.py plan --profile outputs/phase3/dev-profile.json --out outputs/phase3/dev-preparation-plan.json
uv run --locked python scripts/reader_training.py setup --profile outputs/phase3/dev-profile.json --plan outputs/phase3/dev-preparation-plan.json
uv run --locked python scripts/reader_training.py prepare --profile outputs/phase3/dev-profile.json --plan outputs/phase3/dev-preparation-plan.json --out outputs/phase3/dev-preparation-submission.json
# Poll, then collect the completed preparation into a fresh directory.
uv run --locked python scripts/reader_training.py status --profile outputs/phase3/dev-profile.json --plan outputs/phase3/dev-preparation-plan.json
uv run --locked python scripts/reader_training.py collect --profile outputs/phase3/dev-profile.json --plan outputs/phase3/dev-preparation-plan.json --out outputs/phase3/dev-prepared
uv run --locked python scripts/reader_training.py plan --profile outputs/phase3/dev-profile.json --assets outputs/phase3/dev-prepared/prepare/assets.json --out outputs/phase3/dev-plan.json
uv run --locked python scripts/reader_training.py preflight --profile outputs/phase3/dev-profile.json --plan outputs/phase3/dev-plan.json --out outputs/phase3/dev-preflight-submission.json
# Wait for the same-plan preflight receipt to pass before launch.
uv run --locked python scripts/reader_training.py launch --profile outputs/phase3/dev-profile.json --plan outputs/phase3/dev-plan.json --out outputs/phase3/dev-submission.json
uv run --locked python scripts/reader_training.py collect --profile outputs/phase3/dev-profile.json --plan outputs/phase3/dev-plan.json --out outputs/phase3/dev-results
```

These online commands are billable. They need the configured `MODAL_PROFILE`,
owned resource bindings and a reserved budget. `status` and `collect` exit 2 until
the evidence is complete, so the preparation-only collection exits 2 by design;
read the preparation receipt instead of treating that as a failure. Keep the
client running until `prepare`, `preflight` and `launch` return. Each resolves the
image before it spawns, which can take minutes, and a client killed in that window
leaves a receipt with status `unknown`. Never repeat such a submission. Reconcile
it against the provider and stop it with `stop-owned`. Collecting a full six-stage
attempt downloads about a thousand small files and can take ten minutes.

Accept development only after reviewing every attempted rollout: complete and
incomplete counts, format compliance, rewards, policy errors, recovery, terminal
failures and group informativeness. Universal format failure or all-zero reward
calls for a substantive correction before optimization. Do not replay an unchanged
collection to get a better sample, and do not tune prompts against the 64-case
qualification panel. For the full six-stage run remove `purpose` and use fresh
identities and plans. Because the task semantics changed, a v2 training stage
cannot be combined with a pre-v2 baseline.

A development run cannot promise that later batches succeed. It exercises the
boundary greedy evaluation misses.

### A valid zero is not a missing reward

A zero reward says the policy finished the episode and was wrong. It is data, and
the optimizer needs it. A missing reward says the execution cannot be trusted. It
is never filled with zero, never dropped from its group and never resampled until
the group passes. One incomplete rollout rejects its collection.

Read hosted dashboards with the same care. In a03 the hosted `val/reward=0` sat on
top of 64 traces with no reward at all. Count confirmed reward posts, not means.

Constant groups are valid as well. A group of four ones or four zeros is retained
and reported. It simply carries no group-relative signal for GRPO.

### Operational completion is not informative learning

A terminal task failure is a complete episode but not a successful tool chain.
Integration evidence reports the two separately, and each algorithm and stage
still needs at least one real model → tool → observation → continuation → reward
chain of its own.

In the same way, completing two optimizer updates says the machinery works. It
says nothing about improvement. In the qualifying run the corrected baseline was
already 64/64, GRPO saw one informative group in sixteen and its second update had
an exactly zero gradient, and the fresh PPO-MC critic predicted returns badly.
Report optimizer counts, gradient norms, group informativeness and critic error
next to accuracy, and claim only what they support. The 64-case panel has been
inspected repeatedly; it is qualification and regression evidence, not an
untouched benchmark.

## A reader-owned fork and a source-bound plan

Commit your changes in a checkout whose `origin` identifies your GitHub repository.
Supported forms are credential-free `https://github.com/OWNER/REPO[.git]`,
`git@github.com:OWNER/REPO[.git]` and `ssh://git@github.com/OWNER/REPO[.git]`.
Plans store a lowercase `owner/repo`, never a transport URL. Userinfo, query strings,
fragments, unsafe paths, command-like values and other hosts/schemes are rejected
without echoing the supplied origin. HTTPS tokens must be managed by Git credentials,
not embedded in origin. A derivative checkout must identify its actual repository.

The distinct identities are:

1. `source.repository` and the full `source.commit`: your execution source.
2. `source.baseline`: canonical public repository and pinned baseline
   `5c069f7e9cc23d03050d13360ce43e27e47e54aa`.
3. `upstream`: separately pinned Agent Lightning, VERL source/wheel and trainer hash.
4. `source.files` and `source.runtime_files`: admitted source and exact image inputs.
5. Profile/configuration hash, task/environment manifests, assets and resolved config.
   Remote preflight separately records actual installed runtime distributions.

Local **raw** Git objects must prove baseline ancestry. Every admission Git command
uses a controlled environment: replacement refs and both local/inherited graft files
are disabled, as is cached commit-graph ancestry. Working bytes are compared with the
recorded HEAD's raw tree, even when index flags hide modifications. Reader Git metadata
is never deleted or rewritten, and linked worktrees remain supported. Parsing and ancestry do **not**
authenticate GitHub ownership or establish that HEAD has been pushed. All planning
checks work offline. If the baseline object is missing, fetch canonical public history
explicitly; for a shallow clone fetch enough history or unshallow before retrying:

```bash
git remote add lab-upstream https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo.git
git fetch lab-upstream main
# If this checkout is shallow, obtain its missing history before planning:
git fetch --unshallow origin
```

Do not substitute fork HEAD for the baseline. An unrelated root must be rebased onto
canonical history. Clean admission rejects dirty/staged/untracked source. Ignored
outputs are allowed, but ignored/untracked code is never copied into a reader image.
The explicit sorted `experiment/reader_runtime.json` inventory must list tracked,
regular, path-safe files, including new task/config inputs. At worker entry, the
complete packaged lab tree and hashes must match; undeclared inputs and symlink
escapes fail. Bytecode caches are forbidden in that tree, and image/worker/child
settings disable their generation. Installed third-party packages remain a separate
inventory guarded by existing source/dependency checks. The historical builder path
is retained; the reader path no longer recursively uploads arbitrary files.

The supplied profiles are deliberately fixtures. These offline commands create and
validate a non-executable preparation plan for either task, after source is committed:

```bash
uv run --locked python scripts/reader_training.py plan \
  --profile examples/reader-profile.asset.synthetic.json --out outputs/asset-plan.json
uv run --locked python scripts/reader_training.py validate \
  --profile examples/reader-profile.asset.synthetic.json --plan outputs/asset-plan.json
# Default calculator profile: examples/reader-profile.synthetic.json
```

For later authorized execution, create a credential-free owned profile in an ignored
output directory, select `asset-status-v1`, use fresh owned resource/path/attempt
identities, and set `fixture` false. Keep model/resources/numerical protocol fixed.
Generate the preparation plan; **only after approval**, the existing preparation flow
acquires the locked model and generates this task's data. Bind the returned assets
with a second `plan --assets PATH` invocation, then perform remote preflight and
supervised launch as described in [reader training](READER_TRAINING.md). Plans under
the repository must be written to ignored paths. Never commit a plan purporting to
identify the commit containing itself. Any source/config change requires new plans,
W&B identities and a reconciled fresh attempt.

## Execution and evidence

The CLI validates source/profile, then reserves and prepares owned assets. Preflight
verifies the exact package, upstream and assets and executes the selected probe.
The supervisor schedules baseline → training → checkpoint-eval for each algorithm.
For `reader-task-v2` qualification, `baseline-live-integration-v2` is an automatic
dependency gate: a durably completed baseline must declare a valid live integration
chain before its training can be submitted. The supervisor records the gate decision.
An unqualified baseline remains completed with its diagnostic evidence; that algorithm's
training and reload receive skipped receipts with `baseline_integration_unqualified`.
The other algorithm is independent. Zero terminal reward can pass; accuracy, informative
GRPO groups, nonzero task gradients and improvement are not admission criteria.
Historical/v1 scheduling is unchanged.
The stage worker validates identities and invokes `matched_launch`; its generated
worker calls the unmodified hash-guarded `examples/calc_x/train_calc_agent.py`.
That trainer loads the chosen Parquet files and calls Agent Lightning `run_ppo`.
The local reconciler maps the row to the explicit environment and imports the chosen
agent. The agent runs MCP and delivers reward; the unchanged rollout/optimization
overlay enforces token/fragment/rollout accounting. Native checkpoints are inventoried
and evaluated in a fresh worker. Collection joins durable receipts and W&B identities.
There is no upstream task-name switch. The Calc-X filename remains because its verified
trainer is reusable. Tests compose its actual `OmegaConf.merge` configuration and
check the **entire** environment mapping; retained QUESTION/RESULT entries are intentional.

Reader v2 traces bind task, environment, source and configuration in addition to the
existing stage/model/rollout/attempt/W&B identities. Integration requires a unique
model tool call, matching ID/arguments, actual successful tool execution, exact
observation in a later model input, continuation, terminal answer and delivered reward.
Duplicates, inconsistent identities/timing, wrong observations or missing continuation
fail the join. A chain can have terminal reward zero: integration and correctness are
separate outcomes. `reader-integration-v2` requires at least one valid live chain in
**each** of baseline, training and fresh reload, separately for each algorithm.
Coverage counts remain visible; one successful baseline cannot certify other routes.
Local harness tests pin MCP **1.26.0**; the completed reference recorded remote MCP
**1.30.0**. These are different environments. Local passes do not establish exact remote
compatibility: the later campaign requires a fresh remote package inventory and real
task probe before GPU stages. New asset plans use `asset-episode-v2`; archived
attempts retain their original error policy. Numerical reward verification is unchanged.

`reader-task-v2` replaces executable v1 plans. Old plans fail with a regeneration
message rather than accidentally entering historical routing. Archived v1 reports
retain their `any-stage-v1` interpretation through the read-only evidence validator;
dated reports are not rewritten or promoted to v2 qualification. Calculator prompts,
scoring and data remain intact; the new v2 evidence rule applies to new plans for
both tasks.

## Common failures and a third task

| Failure | Next step |
|---|---|
| Dirty or stale source | Commit intended source, preserve unrelated work, regenerate plans into ignored outputs |
| Baseline missing / unrelated | Fetch sufficient canonical history; check ancestry; never invent a baseline |
| Baseline integration unqualified | Preserve completed baseline/trace diagnostics; reconcile the attempt before any corrected plan; do not bypass the gate |
| Cleanup unresolved | Inspect receipt anomalies and owned provider/process state; never infer signal targets from partial JSON |
| Runtime inventory/hash mismatch | Add intended tracked inputs to the inventory and rebuild from the committed source |
| Large reader request prevents Ray startup | Use the bounded, hash-bound request-file transport; do not put the full request into a runtime environment variable |
| Model supplies answer fields as tool arguments | Render the effective prompt: the template's function-call preamble sits beside the task's JSON request. Under v2 the call is a retained policy mistake with a typed error, not an incomplete episode |
| Every final answer is prose | Read the reflection call's recorded input and options; see “Reflection and final output” |
| Tool error with no recognized code | The episode is incomplete by design; inspect the observation before widening any classification |
| Training collection has incomplete groups | Retain every attempt and stop before optimization; do not fill missing rewards, filter failures or weaken admission. Classify the cause first: policy mistake or untrusted execution |
| Submission receipt says `unknown` | The client stopped before acknowledgement; reconcile provider state, place `stop-owned`, use a fresh attempt |
| Missing MCP/Arrow dependency | Sync the isolated locked CPU group; essential tests fail rather than silently skip |
| Dataset mismatch | Preserve the bad output; generate into a new directory from the intended version |
| Tool success but reward zero | Inspect the whole terminal JSON and independent expected record |
| Timeout/truncation/reward failure | Keep failed evidence; do not fill missing rewards with zero |
| Missing stage integration | Report the stage unqualified, even when other stages completed |
| Existing/unknown reservation | Reconcile owned calls and prior exposure; never clear reservations or blindly resubmit |

To add a third small task, start with its independent deterministic data and verifier,
then a real MCP tool and a harness using `observed_runtime`. Define an explicit prompt
that selects visible fields. Add the versioned descriptor in `reader_contract`, its
preparation/inspection and local-probe dispatch, and all source inputs to the runtime
inventory. Prefer the common row schema when it fits; if replacing it, explicitly
replace inherited environment mappings and test full upstream composition. Add real
transport and harness checks, strict parser/failure tests and cross-task evidence
rejections. Keep numerical mathematics and lifecycle safeguards shared. Qualify the
new route separately before making live execution or learning claims.

Arbitrary model compatibility, resumed training, more than two updates and accuracy
improvement are not established by this implementation.
