# Adapt a calculator agent to report asset status

For the calculator, the model chooses an expression and uses the result to answer a numerical question. For asset status, it chooses an asset ID, reads a tool observation and reports two facts as JSON. In both jobs the trainer receives the model's sampled actions and one verified outcome per complete attempt. Walk through the contract below, then try giving the right tool result and the wrong final answer.

## The two workflows

| Component | Calculator (default) | Asset status |
|---|---|---|
| Task ID | `calculator-v1` | `asset-status-v1` |
| Data | Existing acquired Calc-X selection, unchanged | Versioned deterministic synthetic registry, 256 train / 64 heldout / 8 smoke assets |
| Harness | [Calculator agent](../experiment/observed_calc_agent.py) builds the calculator request and observes calls | [Asset agent](../experiment/asset_agent.py) advertises the lookup, passes the observation to the next model call and records termination |
| Request | Existing question and `### ANSWER` instructions | Asset request and strict JSON instructions |
| Tool (Model Context Protocol, MCP) | `mcp_server_calculator`, `calculate(expression)` | Local `asset_server.py`, `lookup_asset(asset_id)` |
| Final parser | Text inside the first answer marker, otherwise the stripped final response | Entire response must be one JSON object |
| Verifier | Exact string match or numerical comparison with relative tolerance 0.01 | Exact field equality against independently regenerated trusted facts |
| Reward | Existing binary scorer | Exactly one binary terminal-correctness reward; no shaping |
| Local check | Real calculator stdio probe | Real asset stdio probe and scripted harness demonstration |

Both tasks retain the supported Llama 3.2 3B Instruct model/tokenizer locks, GRPO and
this lab's PPO-MC, sampling, group size four, two-update qualification, generated-token
conservation, observation masking, fragmentation accounting, equal-rollout weighting,
PPO-MC terminal Monte Carlo targets and critic/whitening behavior. Ownership,
reservations, supervision, cancellation, deadlines, optimizer/refresh checks, native
checkpoint algorithms and fresh-worker reload are shared.

## Run the local checks

Use one Bash terminal at the [checkout root](../README.md#credential-free-offline-quick-start) on Linux or WSL2 with Git, Python 3.12 and uv. These steps use the existing CPU harness group. Installing it can download Python packages. The exercises themselves run locally without accounts, model weights or GPU access. The tool communicates over standard input and output (stdio). The scripted harness posts its reward to a receiver on the same computer, so local sockets and child processes must work.

### Build a task the verifier can check

**Predict.** Two requests about the same asset would share the same truth. How should we split the dataset to keep an evaluation asset out of training?

```bash
UV_PROJECT_ENVIRONMENT=.venv-reader-cpu uv sync --locked --inexact --only-group reader-harness --python 3.12
mkdir -p .local
asset_dir=$(mktemp -d .local/asset-lesson-XXXXXX)
echo "$asset_dir"
task_local() { .venv-reader-cpu/bin/python scripts/task_local.py "$@"; }
task_local generate --data "$asset_dir/data" > "$asset_dir/generated.json"
task_local verify --data "$asset_dir/data" > "$asset_dir/verified.json"
python3 - "$asset_dir/verified.json" <<'PY'
import json, sys

with open(sys.argv[1]) as f:
    manifest = json.load(f)
print(manifest["task"])
for name in ("train_main.parquet", "train_smoke.parquet", "eval_main.parquet", "eval_smoke.parquet"):
    print(name, manifest["files"][name]["rows"])
PY
```

Expect `asset-status-v1` and counts **256, 16, 64, 8** in that order. The 16 are a subset of the 256 training assets. The 64 heldout assets and eight evaluation-smoke assets are separate identities. Verification regenerates the expected records, compares their order and checks their hashes. If verification reports changed bytes or cases, retain the output and generate a fresh directory from the intended source. If PyArrow is missing, check that the command used the CPU interpreter above.

### Compare the two tools, then follow an actual harness call

What arguments would you send to a calculator? Which fields should the asset tool accept? A *probe* asks a real local tool to describe its interface and process known requests. A *scripted episode* goes one step further: the production harness runs, but a fixture supplies the model's actions and answers.

```bash
task_local probe --task calculator-v1 --out "$asset_dir/calculator-probe.json"
task_local probe --out "$asset_dir/asset-probe.json"
task_local scripted --out "$asset_dir/scripted.json"
```

| Receipt | Expected result | What to inspect |
|---|---|---|
| Calculator probe | `status: passed` | `calculate` takes one expression. Its two checks obtain `20` for `(2 + 3) * 4` and a tool error for an unknown symbol. |
| Asset probe | `status: passed`, six valid checks | `lookup_asset` accepts just an asset ID. One valid lookup and five deliberately invalid inputs check both facts and typed error codes. |
| Scripted episode | `status: passed`, one chain, `live_qualification: false` | In `records` in the file, follow the model request, tool invocation, observation in the next model input, final response and delivered reward. Inference is scripted, the stdio tool and local reward transport run. |

A tool may pass its probe even when the harness fails to deliver the observation to the next model call. If the scripted check fails, inspect its receipt for the chain and owned-process cleanup, then check the locked package versions. Receipts use fresh filenames on each run.

### A successful lookup and two different outcomes

Use the CPU environment from [the setup step](#build-a-task-the-verifier-can-check). The fixture's first heldout question asks for `asset-281`. Its tool reports `operational` and `failed`. Predict what happens if the model reports a *healthy* backup, then predict what happens if its second call returns empty text. Both attempts already have a successful lookup.

```bash
PYTHONPATH="$PWD/tests:$PWD" .venv-reader-cpu/bin/python - <<'PY'
from tempfile import TemporaryDirectory

from asset_harness_support import run_case
from experiment import asset_task as task

answer = {"asset_id": "asset-281", "service_state": "operational", "backup_health": "healthy"}
for mode in ("normal", "missing-second"):
    with TemporaryDirectory() as directory:
        result = run_case(directory, final=task.dumps(answer), mode=mode)
    tool = next(r for r in result["records"] if r["kind"] == "tool")
    episode = next(r for r in result["records"] if r["kind"] == "episode")
    rewards = [post["data"]["value"] for post in result["posts"]]
    print(mode, tool["execution_status"], episode["execution_status"], rewards)
PY
```

| Scripted continuation | Tool status | Episode status | Rewards received locally |
|---|---|---|---|
| `normal`, completed JSON with wrong backup health | `ok` | `ok` | `[0.0]` |
| `missing-second`, empty continuation | `ok` | `exception` | `[]` |

The first JSON parses, names the right asset and still disagrees with the verifier's trusted record. Its confirmed zero is an outcome the trainer can use. For the second attempt the harness cannot establish a terminal answer. Filling in zero would invent an outcome and admitting the other three siblings would change the training group. One incomplete attempt rejects its collection. The receipt above comes from the [scripted fixture](../tests/asset_harness_support.py). The [parser and verifier](../experiment/asset_task.py) and [harness](../experiment/asset_agent.py) are the production implementations.

**Optional alteration.** In the Python block change only `"backup_health": "healthy"` to `"backup_health": "failed"`. The first reward list becomes `[1.0]`, and the second remains empty. A correct answer string supplied to the fixture cannot repair a missing continuation.

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

Use the [reader procedure](READER_TRAINING.md#configure-a-reader-owned-allocation) with the asset template and this purpose in your own funded allocation. It supplies the same plan, setup, prepare, asset binding, preflight, launch, status, collect and stop commands. For this variant, expect **two training-stage collections of 32 attempts, one per algorithm** in place of the six-stage qualification table. Keep the client running through image resolution and submission. An unknown submission needs reconciliation. A complete collection-only result qualifies the sampled episode boundary and supplies no learning or checkpoint evidence.

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

A confirmed zero records a complete task failure under the declared episode policy. It is data, and
the optimizer needs it. A missing reward says the execution cannot be trusted. It
is never filled with zero, never dropped from its group and never resampled until
the group passes. One incomplete rollout rejects its collection.

Read hosted dashboards with the same care. In a03 the hosted `val/reward=0` sat on
top of 64 traces with no reward at all. Count confirmed reward posts, not means.

Constant groups are valid as well. A group of four ones or four zeros is retained
and reported. It simply carries no group-relative signal for GRPO.

### Operational completion is not informative learning

A completed wrong final answer can still demonstrate a successful model → tool → observation → continuation → reward chain. Two invalid calls followed by policy-limit termination supply a confirmed zero without such a chain. Each algorithm and stage needs live evidence of the chain to qualify integration.

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

Try the [practice plan and asset-template alteration](READER_TRAINING.md#try-a-plan-without-an-account). The same procedure binds a real source and returned manifest for reader-owned execution. A source or configuration change requires fresh plans and attempt identities.

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

## Common failures and a new task

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

To add another small task, start with its independent deterministic data and verifier,
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

## Recorded qualifications

Phase 3 implements two tasks through the same matched training workflow. The new
`asset-status-v1` task is **operationally qualified for both algorithms** under the
`asset-episode-v2` episode contract: baseline, two-update training, native
checkpoint saving and fresh-worker checkpoint evaluation all completed, with 64/64
heldout cases at baseline and after reload. That is operational evidence, not
evidence of learning; the corrected baseline leaves no headroom. See the
[dated report](PHASE3_QUALIFICATION_2026-11-24.md).

The four earlier attempts failed. They are preserved unchanged in the
[2026-11-22 report](PHASE3_QUALIFICATION_2026-11-22.md) under their original
episode policy. The [engineering checklist](PHASE3_ENGINEERING_CHECKLIST.md) condenses
what they taught into checks to run before a new attempt. See also
[offline verification](PHASE3_VERIFICATION.md). The local
harness demonstration uses explicitly scripted inference and does not establish
live-model behavior.

The third task, `recovery-readiness-v1`, uses three tools and a verifier that checks both the decision and its supporting evidence. The original task, calibration method and offline checks describe the `recovery-episode-v1` contract. Later, both routes ran live development calibration. GRPO completed the A03 two-update qualification and the P02 16-update pilot, including fresh-worker reloads. [Task](PHASE4A_RECOVERY_TASK.md) · [Calibration method](PHASE4A_CALIBRATION.md) · [Offline checks](PHASE4A_VERIFICATION.md) · [Live calibration](PHASE4B_LIVE_CALIBRATION.md) · [A03](PHASE4C_GRPO_CAMPAIGN_CLOSEOUT.md) · [P02](PHASE4C_FUNCTIONAL_PILOT_CLOSEOUT.md)

On the 24 development cases, P02's terminal greedy policy always answered `not_ready`. Strict successes rose from 9/24 to 12/24 while decision accuracy stayed at 12/24. Recovery PPO-MC live training and reload remain unqualified. The full recovery study is unexecuted, and witness-selection SFT remains deferred. The asset `collection-only` procedure above is specific to the asset task. [Paired cases and verifier analysis](PHASE4C_PILOT_LEARNING_DIAGNOSIS.md) · [Study protocol](PHASE4C_LEARNING_STUDY_PROPOSAL.md) · [Draft PR #14](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/pull/14)

The completed calculator qualification remains bound to execution source
`5daff64cefd348fb731094cf465b223ddceb5fb2` and plan SHA-256
`a3128740ba4c2133e049bd80e9bdb471aa1ad56df3158479d1594b4e0ab23cc3`.
Both algorithms completed baseline, two-update training, native checkpoint saving,
and fresh-worker evaluation reload. GRPO had **0/16 informative reward groups** and
no observed task-directed policy-gradient signal. PPO-MC made nonzero updates but
had poor critic diagnostics. Operational validity was established; reliable learning
improvement was not. Phase 3 commits did not produce those results. The historical
20-update, three-paired-seed experiment and its learning guide remain separate.
