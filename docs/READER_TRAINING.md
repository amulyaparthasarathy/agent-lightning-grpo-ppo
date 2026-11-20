# Reader-owned reference training

**Phase 3 update:** New plans use `reader-task-v2`, support reader-owned GitHub forks and the synthetic `asset-status-v1` task. See [Task adaptation](TASK_ADAPTATION.md) for the offline workflow, migration and all-stage integration rule. The dated qualification report retains its original v1 semantics and exact execution identities. No asset-task live qualification has run.

**The reference Llama/calculator workflow has completed real baseline evaluation, two-update training, native checkpoint saving and fresh-worker reload for both algorithms.** CPU preflight and model/tool integration passed. GRPO observed zero informative reward groups; useful learning and reliable accuracy improvement are not established. See the [dated qualification report](READER_QUALIFICATION_2026-11-19.md) for exact executed source, evidence, earlier failures and costs. This guide is the canonical reader procedure. It does not reopen the completed educational release or attribute its results to new code.

The current execution schema is `reader-task-v2`; archived `reader-reference-v1` plans require regeneration for new execution. Its numerical protocol remains `matched-rollout-mean-v2`: terminal binary rewards; eight question presentations and four distinct complete rollouts per presentation; generated-token conservation, observation masking and equal-rollout loss weighting. Missing rewards, invalid context, incomplete groups and truncation remain errors. PPO-MC retains the learned critic, gamma/lambda one, true-terminal zero bootstrap, terminal Monte Carlo returns, tokenwise residual advantages and valid-token whitening. The calculator prompt, extractor and verifier are unchanged.

| Required configuration | GRPO | PPO-MC |
|---|---|---|
| Reference Llama + calculator | Live operational, integration and reload checks passed; 0/16 informative groups | Live operational, integration and reload checks passed; learning benefit not established |
| Alternate compatible model + calculator | Subsequent implementation and qualification | Subsequent implementation and qualification |
| Reference Llama + asset-status-v1 | Offline implementation and real local MCP/scripted harness checks passed; live qualification pending | Offline implementation and real local MCP/scripted harness checks passed; live qualification pending |

Every cell needs actual baseline, training, checkpoint and fresh-reload evidence before the milestone is complete. Qualifying the alternate rows does not qualify their cross-product. Short qualification does not require accuracy improvement and does not establish resumed-training support.

## Client and source

Run every command below from this public repository's root in the existing Unix Workbench Dev Container, or a Linux/WSL filesystem. Python 3.12, Git and uv are prerequisites. No local CUDA, Torch, PyArrow or Docker socket is required for ordinary client commands. Do not copy a virtual environment between architectures; the remote image targets Linux x86-64/H100. The offline verification for this PR uses the available aarch64 Python 3.12 client.

```bash
uv sync --locked --python 3.12
uv run --locked python scripts/reader_training.py --help
```

For the local synthetic Parquet tests only, install the optional locked test group. This does not install a GPU stack or acquire model/dataset assets:

```bash
uv sync --locked --group reader-test --python 3.12
bash scripts/test_reader_offline.sh
```

The script clears credentials, points Modal configuration at `/dev/null`, disables Hub access and denies network transports through a test-only `sitecustomize`. It runs the two accounting/results commands, 39 PPO-MC tests, 15 matched-study tests, and the reader/refactor tests. Model files in tests are explicitly synthetic non-model bytes; remote submissions and health responses are explicitly mocked. Parquet selection, actual child process environments, cleanup and Modal SDK function discovery are exercised locally. These checks do not qualify inference or training. Do not add the test guard to real execution environments.

A real calculator-only stdio probe can also run locally without model/data acquisition or a GPU stack. This uses an isolated CPU environment with the same pinned calculator and AutoGen versions. Dependency installation accesses the package index; the probe itself needs no credentials or network. Choose a fresh receipt filename:

```bash
UV_PROJECT_ENVIRONMENT=.venv-reader-cpu uv sync --locked --only-group reader-harness
mkdir -p .local/reader-cpu
env -i PATH="$PATH" HOME=/tmp MODAL_CONFIG_PATH=/dev/null \
  PYTHONPATH="$PWD/tests/offline_guard:$PWD" \
  .venv-reader-cpu/bin/python experiment/reader_calculator.py \
  --out .local/reader-cpu/calculator-preflight.json
```

The probe checks the advertised `calculate(expression: string)` schema, evaluates `(2 + 3) * 4` to `20`, and checks that `unknown_reader_symbol + 1` reports the expected tool error. Its receipt records the actual schema, requests, responses, package versions and validation outcome. RPC reads have a 15-second timeout; an isolated process bounds startup/calls/teardown to 90 seconds with up to 10 seconds for cleanup. Normal teardown exits the workbench context; timeout cleanup kills the owned client and separately registered MCP server process groups. Failure receipts are retained and never overwritten. A passing local calculator receipt alone establishes calculator CPU behavior; it cannot qualify remote packages, model/tool integration, training or reload. The [completed live reference run](READER_QUALIFICATION_2026-11-19.md) supplies separate evidence for its recorded configuration. The remote preflight runs the same probe again in its actual image and writes `preflight/calculator-preflight.json`.

Plans require a clean committed checkout with a credential-free GitHub `origin` identifying the actual reader repository, plus local ancestry from the pinned canonical public baseline; see [fork admission](TASK_ADAPTATION.md#a-reader-owned-fork-and-a-source-bound-plan). The commit, packaged source hashes, lockfiles, relevant configuration inputs, resolved profile, exact asset manifests and task/verifier identities are bound into the plan. Changing source, configuration or assets requires a new plan. Keep generated plans outside Git: committing a plan that names its own source commit would change that commit. The CLI rejects dirty or uncommitted source before creating a plan; keeping the generated plan itself outside Git is required. `validate` checks current source/profile and manifest consistency, not remote bytes or account ownership.

## Non-secret profile and offline planning

The complete [synthetic example](../examples/reader-profile.synthetic.json) is deliberately non-executable (`fixture: true`). It uses example names, has no credentials and supplies no asset hashes. `plan` and `validate` import no cloud apps, authenticate nowhere, create no remote resources, build nothing and perform no acquisition or preparation.

```bash
uv run --locked python scripts/reader_training.py plan \
  --profile examples/reader-profile.synthetic.json --out .local/reader-demo/plan.json
uv run --locked python scripts/reader_training.py validate \
  --profile examples/reader-profile.synthetic.json --plan .local/reader-demo/plan.json
```

Expected output: `offline: true`, explicit unverified ownership/availability, and `training_admissible: false`. `plan` exclusively creates its output; choose a fresh filename if it exists. `validate` writes no files. A preparation-only plan without real manifests cannot launch training. Commit intended source before planning; no dirty-source CLI preview is admitted.

For a future authorized run, copy the example into ignored local storage:

```bash
mkdir -p .local/reader
cp examples/reader-profile.synthetic.json .local/reader/profile.json
```

Edit that non-secret JSON before creating plans. Set `fixture` to `false` only for your actual configuration. Supply your explicit Modal local profile **and intended authenticated workspace**, existing environment, distinct app/registry/Volume/Secret names, W&B entity/project, fresh Volume-relative model/data/staging/output paths, experiment ID and attempt number. All fields are required. There is no maintainer-resource fallback. Historical names and path roots, traversal, absolute paths, symlink escapes, overlapping paths, unsupported models/tasks/algorithms, nonfinite bounds and insufficient budgets are rejected.

Keep the exact model selection: `meta-llama/Llama-3.2-3B-Instruct`, revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`, profile `llama3b`, parser `llama3_json`. Both algorithms are required. The first proposal retains one H100, 16 CPUs and 196608 MiB host memory with concurrency one. This continuity choice is not a minimum-resource claim.

The example permits six stage calls of at most 3600 seconds, startup heartbeat within 600 seconds, a 22200-second supervisor, CPU preparation up to 3600 seconds and CPU preflight up to 900 seconds. It budgets $10/build, $5/preparation and preflight, $125/training and supervision, $7/storage, and $0.08/collection within $148 total. The supervisor deliberately charges its own time against the conservative $10/hour worker ceiling; the $0.10/hour supervisor estimate is informational. Reader dispatch deadlines reserve 30 seconds for cross-host clock uncertainty: a 900-second CPU limit dispatches a deadline 870 seconds after the client clock sample, and a 3600-second stage dispatches 3570 seconds. This shortens exposure; the strict worker deadline admission and provider timeouts remain intact. CPU receipts retain the admitted deadline and entry time. Each rate is a planning ceiling, not a provider-enforced spending cap. Recheck current rates and account allowance before spending. There are no automatic retries, budget extensions or later campaign stages.

## Explicit online setup and preparation

**Everything from here requires live authorization.** The dated qualification report records the completed reference allocation; it does not authorize another reader allocation or establish compatibility of a future image/build. Obtain gated model access under the applicable model terms and configure Modal/HF/W&B through their normal private stores. Create a W&B project and an existing named Modal Secret containing `HF_TOKEN` and `WANDB_API_KEY` through your normal account setup. Never place their values in the profile, commands, Git, logs or chat. This workflow does not create or modify Secrets or environments.

Set `MODAL_PROFILE` to exactly the profile value you chose. The CLI checks `Workspace.from_context().hydrate().name` against the intended workspace before resource operations; an environment variable alone is not ownership evidence. Remote entrypoints check the workspace again. The profile's environment scopes every resource lookup. API token environment overrides that authenticate to a different workspace are rejected.

```bash
uv run --locked python scripts/reader_training.py plan \
  --profile .local/reader/profile.json --out .local/reader/preparation-plan.json
uv run --locked python scripts/reader_training.py validate \
  --profile .local/reader/profile.json --plan .local/reader/preparation-plan.json
uv run --locked python scripts/reader_training.py setup \
  --profile .local/reader/profile.json --plan .local/reader/preparation-plan.json
uv run --locked python scripts/reader_training.py prepare \
  --profile .local/reader/profile.json --plan .local/reader/preparation-plan.json \
  --out .local/reader/preparation-submission.json
```

`setup` verifies the existing Secret handle and explicitly creates the named Volume and registry, refusing existing names. If creation partially succeeds, inspect the resources; do not rerun blindly, delete them or assume they are unused. No creation happens during planning or import. The environment and Secret must already exist.

`prepare` records local and remote submission intent before image resolution. It can build the pinned CUDA/training image on CPU, then submits one detached CPU preparation worker, with four CPUs/16 GiB and the declared deadline. The separate `/opt/agent-lightning/.venv/bin/python` already provides PyArrow, HF and gdown. The ordinary client never needs them for this operation. The shared heavy builder retains 8 CPUs/64 GiB/7200 seconds, the dependency finalizer 4 CPUs/16 GiB/1800 seconds, and the reader Transformers overlay 4 CPUs/16 GiB/900 seconds. Registry/base-package layers use provider defaults; their actual charges and build logs must be reviewed. Cache availability is unverified. No forced rebuild is requested.

Calculator preparation reuses `prepare_model.py` and `prepare_data.py`: acquire the immutable Llama snapshot, install the fixed-date Llama JSON template, acquire the released calculator archive (`gdown` source ID `1FQMyKLLd6hP9dw9rfZn1EZOWNvKaDsqw`), extract only `train.parquet`/`test.parquet`, then select with seed 329. Inputs and partial files stay under the owned staging path. All model weights, their complete shard inventory, sizes, content hashes, config, tokenizer and template must match the existing reference lock. The calculator source dataset is not content-addressed by its download URL; its actual acquired hashes are recorded. Asset-status preparation instead generates its versioned synthetic data locally inside the preparation worker and never acquires the calculator archive.

All four prepared panels and evaluation JSONL files are checked for hashes, schemas, ordered IDs, duplicate/conflicting questions, split isolation and JSONL/Parquet agreement. The final panel contains 64 cases, the training pool 256, with 16 nested training-smoke and eight separate evaluation-smoke cases. Worked solutions are discarded. Only `question` enters the prompt; `result` remains the existing verifier's privileged label. Preflight and stages rehash raw inputs and reproduce deterministic selection, in addition to checking prepared outputs. Do not remove staging before qualification.

No existing asset or staging directory is reused or overwritten. A failure can leave useful partial outputs; inspect the receipt/logs and select a new allocation after resolving the cause. The historical working-Volume `prepare` refusal remains intact: the new reader route is a separate action.

```bash
uv run --locked python scripts/reader_training.py status \
  --profile .local/reader/profile.json --plan .local/reader/preparation-plan.json
uv run --locked python scripts/reader_training.py collect \
  --profile .local/reader/profile.json --plan .local/reader/preparation-plan.json \
  --out .local/reader/preparation-evidence
```

`status` is read-only and reports operation reservations, terminal CPU receipts and the supervisor index when available. A submission or missing receipt remains unresolved. Wait for `operations.prepare.result.status: completed` before collecting preparation outputs. `collect` downloads bounded JSON/JSONL/log/text evidence into a new local directory; weights stay in the Volume. Both commands exit 2 while full six-stage training evidence is incomplete, including successful preparation-only inspection. The preparation manifest is `preparation-evidence/prepare/assets.json`. Build/package evidence and logs accompany the receipts. A hard timeout may leave only started evidence; that is a failed/unknown attempt, never qualification.

## Bind actual inputs, preflight, then launch

```bash
uv run --locked python scripts/reader_training.py plan \
  --profile .local/reader/profile.json \
  --assets .local/reader/preparation-evidence/prepare/assets.json \
  --out .local/reader/training-plan.json
uv run --locked python scripts/reader_training.py validate \
  --profile .local/reader/profile.json --plan .local/reader/training-plan.json
uv run --locked python scripts/reader_training.py preflight \
  --profile .local/reader/profile.json --plan .local/reader/training-plan.json \
  --out .local/reader/preflight-submission.json
uv run --locked python scripts/reader_training.py status \
  --profile .local/reader/profile.json --plan .local/reader/training-plan.json
```

Preflight is an explicit, potentially billable detached CPU call. It revalidates exact model/data/source identities; source guards and installed package versions; the six narrowly reviewed dependency exceptions; actual tokenizer loading; gated model revision access; W&B project read access; and the selected task stdio MCP schema, valid-request and error-request checks (plus existing calculator verifier sanity on that route). It loads no model weights. It does not demonstrate W&B write completion, GPU compatibility or a successful agent trajectory. Those require the subsequent stage evidence. Changed dependency resolution, unavailable licensed model/data, missing project/Secret or changed input bytes stop the run.

Wait for the durable same-plan preflight result to pass. Then, only within the separately authorized spending decision:

```bash
uv run --locked python scripts/reader_training.py launch \
  --profile .local/reader/profile.json --plan .local/reader/training-plan.json \
  --out .local/reader/training-submission.json
uv run --locked python scripts/reader_training.py status \
  --profile .local/reader/profile.json --plan .local/reader/training-plan.json
uv run --locked python scripts/reader_training.py collect \
  --profile .local/reader/profile.json --plan .local/reader/training-plan.json \
  --out .local/reader/training-evidence
```

The first qualification requires preflight, not a prior qualification of itself. The detached supervisor executes each algorithm's **baseline → two-update training → checkpoint evaluation**, sequentially in one GPU lane. Each stage uses a single-use worker and a new launcher/trainer process. Per-stage GPU/FlashAttention preflight runs before the actual model workflow. Inputs are offline during training. Logs, traces, audit events, resolved configuration, build evidence and native checkpoints are committed durably. The actual image/App/call IDs are recorded after resolution; a cache hit is never inferred merely from successful execution.

Acceptance requires all six stages and their exact receipt/plan/source/config/model/task/data identities; complete 64-case baseline and reload evaluations; exactly two actor optimizer calls per algorithm and two critic calls for PPO-MC; training refreshes at 0, 1, 2; no optimizer calls during either evaluation; complete final native role inventories with sizes/hashes; exact checkpoint verification before and after reload; W&B finish evidence; and distinct training/reload worker and process identities. Reader receipts and completions must agree on the provider-assigned `MODAL_TASK_ID`; training and reload must have different container IDs and call IDs. Hostnames remain diagnostic only: distinct Modal containers can both report `modal`. Process identity is the container ID paired with its recorded launcher/trainer PID, so PID reuse across separate containers is not mistaken for process reuse. See [Modal runtime identity variables](https://modal.com/docs/guide/environment_variables). GRPO checkpoints contain actor state; PPO-MC checkpoints contain actor and critic state. Each role includes native model, optimizer and extra-state files. W&B hosted history is not independently queried by `collect`; local remote-init/finish receipts and metrics remain explicit evidence.

Additional checks require positive valid generated-token counts, two complete collections, finite loss/gradient diagnostics and checkpoint-save events. Require `actor/pg_loss` at both global updates 1 and 2; PPO-MC additionally requires `critic/vf_loss` at both updates. Finite zero is valid. Records are joined by update/metric, permitting split role records and identical duplicates while rejecting missing, invalid or conflicting evidence. The pinned [VERL loss emitter](https://github.com/volcengine/verl/blob/7aed6b230776f963fa09509c10d9c3a767d1102c/verl/workers/utils/losses.py) and [Agent Lightning trainer](https://github.com/microsoft/agent-lightning/blob/cbc56e41e0eab643aa71d195e92e5bba652fa683/agentlightning/verl/trainer.py) reduce minibatch metrics and log at global training steps; this check does not assume one JSONL record per update. The lab's existing matched runtime emits the corresponding critic metric. Retain reward-group variation. If all GRPO groups have equal reward, report successful operational execution separately from the absence of informative task-directed policy-gradient signal. Nonzero optimizer calls alone do not establish learning. Reload here qualifies evaluation of the saved native checkpoint; resumed training and arbitrary weight exports are separate claims.

`status` and `collect` report four distinct dimensions per algorithm: operational stage completion; model/tool integration coverage by baseline/training/checkpoint-eval stage; informative GRPO reward-group count (not applicable to PPO-MC); and remaining qualification gaps. `evidence_complete` and exit code zero mean the operational receipt set is complete, **not** that integration or learning is established. `live_qualification` retains “gaps remain” when needed.

Each completion includes `model_tool_integration`, with trace file hashes and observed chain identifiers. The reader joins existing schema-1 traces by experiment/W&B/stage, trace, case, rollout and attempt, episode parent span, tool-call ID and time order. A valid chain requires a successful recorded model request returning the selected task tool (`calculate` or `lookup_asset`), the matching successful tool invocation with the same arguments, a subsequent successful model request containing that exact tool observation, and a successful episode with a delivered binary terminal reward. It uses AutoGen's existing `FunctionCall`, `ToolResult.to_text()` and `FunctionExecutionResultMessage` representations. Synthetic, inconsistent or duplicate spans do not qualify. Under `reader-integration-v2`, at least one valid live chain in **each algorithm's baseline, training and fresh-reload stage** is required to qualify harness integration. The dated v1 report retains its older any-stage interpretation. A reward of zero can still demonstrate integration; correct answers without an observed tool-continuation chain cannot.

Operational completion may succeed with zero informative GRPO groups. The report then explicitly states that informative group-relative task-advantage evidence was not observed. Neither nonzero loss, improved accuracy nor universal tool use is required. Missing evidence or unfavorable outcomes must be retained as observed; do not automatically retry, tune rewards, change inputs or select favorable trajectories. Hosted W&B history, meaningful learning and resumed training remain separate claims.

## Stop and recover within ownership

```bash
uv run --locked python scripts/reader_training.py stop-owned \
  --profile .local/reader/profile.json --plan .local/reader/training-plan.json
```

Use `preparation-plan.json` instead to cancel that exact preparation plan. `stop-owned` writes a durable stop marker and requests cancellation only for exact recorded calls and stage-entry reservations belonging to the supplied plan. It never stops an app by display name or cancels another attempt. Missing call IDs remain unknown; cancellation cannot prove that an unacknowledged spawn did not happen. Follow with status/collection. A cancelled or timed-out trajectory is not converted to a terminal-zero example.

Reservations survive ambiguous submissions and failures. The registry allows only one qualification allocation; do not clear it to retry. After investigation and a new spending decision, use a new registry/app allocation and fresh attempt/output paths, preserving previous evidence. Startup without heartbeat, a stale supervisor lease, stage/supervisor deadlines, source/input mismatch, storage ceiling or owner cancellation stops owned workers. The 300-GiB ceiling is checked using owned file sizes, not Volume filesystem free-space placeholders; a write can overshoot between checks. The seven-day retention interval is a review point, not automatic deletion. Keep native checkpoints, staging and evidence until retention is explicitly decided. Retention beyond the proposal incurs further storage costs.

## Historical matching, build limits and next seams

The calculator asset manifest compares stable prepared-file hashes and ordered case IDs with `matched_data_lock.json`. Preparation timestamps and absolute local paths do not themselves imply different data. Historical acquisition hashes are absent from that lock and remain unverified. Exact prepared byte matches are reported per file. Different serialization/content or unavailable original bytes receive a new exact input identity and are a **methodology rerun with identified input differences**, not a recreation of historical results. Old locks are never rewritten to make new inputs pass.

The client lock is not a full remote transitive dependency lock. The shared pinned build scripts, guarded backend sources and reviewed package-conflict policy are retained. Cached historical success does not prove a cold build, provider capacity, H100 availability or compatibility with future artifact resolution. A failed new build is evidence, not permission to relax the guards. No author checkpoints, ignored helper, private Git object or author-prepared assets are dependencies of this procedure.

The immutable historical distinctions remain: private export `eef9c78707b33725847fb15ca19592733a84c6c3`; r03 execution `5fb6adfd48ee245509f7dff3ab1d849fcf707d86`; attributed public representation `89fe32f92f29209803c13264baed951713741d42`; study `agl-explore-20-v1-r03`. The [publication provenance](../PUBLICATION_PROVENANCE.json), [licenses](../LICENSING.md), historical evidence and learning guide retain their scope. Reader code and runs have their own commit and execution identities.

Task customization is implemented at `reader_contract.task_identity`, task preparation/inspection/probe dispatch and the shared launcher. [Task adaptation](TASK_ADAPTATION.md) explains the working asset-status task, exact visible/privileged data boundary and third-task extension points. Alternate models still need a new immutable actor/tokenizer/template/parser/critic compatibility profile and live qualification; a legacy selector is insufficient.
