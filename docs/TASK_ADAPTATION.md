# Reader customization and task adaptation

Phase 3 implements two tasks through the same matched training workflow. The new
`asset-status-v1` task has **offline integration evidence only**. Local MCP execution
is real; the harness demonstration uses scripted inference and a loopback reward
receiver. Neither demonstrates live-model behavior, GPU training, checkpoint reload,
or learning improvement. See [verification](PHASE3_VERIFICATION.md) and the
[proposed live qualification](PHASE3_QUALIFICATION_PROPOSAL.md).

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
HF/W&B access and the selected task probe. Do not run that command without a new
approved Phase 3 spending envelope.

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
`offline`; backup health is `healthy`, `stale`, `failed`. Missing, non-string or extra
arguments produce an MCP error result (`invalid_arguments`). Malformed IDs produce
`invalid_asset_id`; well-formed unknown IDs such as `asset-999` produce
`unknown_asset`. There is no random or external inventory lookup. Protocol output
uses stdout; library diagnostics use stderr.

The child starts using an absolute interpreter and `-I`, then clears its environment
before importing task code. Only `PYTHONUNBUFFERED` and `PYTHONDONTWRITEBYTECODE`
remain. No labels, HF/W&B credentials or gateway keys reach it. Linux parent-death
signaling and owned PID/start-time receipts cover separately created MCP process
groups; cleanup never uses a global process-killing command. This implementation
requires the Linux/WSL2 environment used by the lab.

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
Empty/missing terminal text, truncation, cancellation, timeout, transport/tool errors,
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

Local Git objects must prove baseline ancestry. Parsing and ancestry do **not**
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
| Runtime inventory/hash mismatch | Add intended tracked inputs to the inventory and rebuild from the committed source |
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

Arbitrary model compatibility, resumed training, GPU asset-task qualification and
accuracy improvement are not established by this implementation.
