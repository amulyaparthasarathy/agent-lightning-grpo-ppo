# Reader-owned reference training

Use this procedure to take one model through **baseline evaluation → two training updates → checkpoint evaluation on a fresh worker** for both GRPO and PPO-MC. Start with a practice plan you can make locally. To learn without running a model, the [two offline checks](learning/08-reproduce-and-operate.md#two-credential-free-checks) reproduce synthetic arithmetic and published results. The [asset walkthrough](TASK_ADAPTATION.md#run-the-local-checks) exercises the harness with scripted responses.

## Client and source

Run every shell block **in the same Bash terminal at the root of your checkout**, on Linux or WSL2. An existing Unix Workbench environment works. You need Git, Python 3.12 and uv. The [quick start](../README.md#credential-free-offline-quick-start) shows how to obtain the public checkout. If you change the task code, use a clean, committed checkout of your own fork with its actual GitHub origin and the canonical baseline in its history. [Fork and source admission](TASK_ADAPTATION.md#a-reader-owned-fork-and-a-source-bound-plan)

| Where | Responsibility | Dependencies and access |
|---|---|---|
| Your terminal | Write and validate plans, submit owned work, read status, collect evidence or request a stop | Locked Python client. Package installation may download from the package index. Only online commands need Modal authentication. |
| Remote CPU | Build or resolve the training image, prepare the exact model and data on the Volume, check the tokenizer and run the tool probe | Remote Linux x86-64 image, licensed model access, task inputs and the reader's Secret |
| Remote GPU | Run the harness and inference server, collect rewards, optimize, save the checkpoint, then evaluate it in a new worker | The remote image supplies CUDA, Torch, VERL and vLLM. One H100 stage runs at a time. |

Can your client describe an allocation before you log in? Install the locked client packages in the existing project environment and inspect the command list. Use a project-local virtual environment for this checkout. The CPU-only harness environment used in the adaptation lesson is separate.

```bash
uv sync --locked --inexact --python 3.12
.venv/bin/python scripts/reader_training.py --help
```

Expect the command list from `plan` to `stop-owned`. A missing script usually means you are outside the root. A missing client package calls for checking the Python version and locked sync. Nothing in this step acquires weights or requires a local GPU.

## Try a plan without an account

**Predict.** If a plan names a GPU and two algorithms but only has example account names and no asset manifest, should it be able to start training?

Choose a template and create a fresh ignored directory for this session. The shell function below is a short way to run the same client with the selected profile throughout the procedure. Keep the printed directory if you need to return to the receipts later.

```bash
mkdir -p .local
reader_dir=$(mktemp -d .local/reader-XXXXXX)
echo "$reader_dir"
reader_template=examples/reader-profile.synthetic.json
reader_profile="$reader_template"
reader() { .venv/bin/python scripts/reader_training.py "$@" --profile "$reader_profile"; }
reader plan --out "$reader_dir/practice-plan.json"
reader validate --plan "$reader_dir/practice-plan.json"
```

Both commands print the same `plan_sha256` (a fingerprint of the plan), with `valid: true`, `offline: true`, `training_admissible: false` and `UNVERIFIED` ownership and availability. Open `practice-plan.json` and find `source.commit`, `task_identity.name`, `assets` and the stages in `jobs`. A null asset inventory is enough for planning preparation. A fixture can never launch. The client checks the clean committed source and profile locally. If it reports stale source, changed configuration or a used output path, preserve the old plan and generate a fresh one against the intended committed inputs.

**Optional alteration.** Repeat the block with `reader_template=examples/reader-profile.asset.synthetic.json`. The task becomes `asset-status-v1`, and its contract names `lookup_asset`. The reference model, two algorithms and offline admission result stay the same. You can finish the lesson here.

## Configure a reader-owned allocation

For a new run, use **your own separately funded resources**. The commands for `setup`, `prepare`, `preflight`, `launch`, `status`, `collect` and `stop-owned` contact your account. Preparation, image builds, preflight, GPU stages, transfers and storage can incur charges. Review current rates, capacity, outstanding exposure and retention against your own ceiling before setup. You can still run `plan` and `validate` offline with your actual profile.

Start with the calculator template for the reference workflow, or the asset template for the adaptation. The example already has the required JSON structure. Copy the selected template once and edit the copy in your editor.

```bash
reader_profile="$reader_dir/profile.json"
cp "$reader_template" "$reader_profile"
```

| Profile input | Supply or check before planning |
|---|---|
| `fixture`, `task`, `algorithms` | Set `fixture` to `false` for the real allocation. Keep the selected task and both algorithms. For the standard six stages, omit `purpose`. The asset-only [development variant](TASK_ADAPTATION.md#greedy-evaluation-does-not-test-sampled-training) has a different stage contract. |
| `modal` | Use your authenticated local profile, its intended workspace and existing environment. Name an existing Secret in that environment containing `HF_TOKEN` and `WANDB_API_KEY`. Choose fresh owned app, registry and Volume names. `setup` creates the latter two. |
| `paths`, `experiment`, `attempt` | Supply distinct Volume-relative model, data, staging and output paths, an experiment ID and attempt number. Preserve earlier allocations. |
| `wandb` and model access | Supply a real W&B entity and existing project. Your HF account must have gated access to the model. Keep credentials in their private stores. The JSON contains names only. |
| `model`, `resources`, `bounds` | Keep the supported Llama 3.2 3B Instruct revision, tokenizer and parser from the template. The reference allocation uses one H100, 16 CPUs and 196608 MiB of host memory. Review the ceilings below. |

The example permits six stage calls of at most 3600 seconds, startup heartbeat within 600 seconds, a 22200-second supervisor, CPU preparation up to 3600 seconds and CPU preflight up to 900 seconds. Its example budget allows $10 for builds, $5 for preparation/preflight, $125 for training/supervision, $7 for storage and $0.08 for collection within $148 total. The supervisor reserves against the conservative worker rate. These values are planning inputs. Provider billing and later retention charges can differ. The client rejects unsupported or underfunded profiles.

**What distinguishes the two plans you will make?** The first describes *where to prepare* assets. Select your Modal profile from the edited JSON and validate this first plan locally. Expect matching hashes and `training_admissible: false` because there are no prepared assets yet.

```bash
export MODAL_PROFILE="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["modal"]["profile"])' "$reader_profile")"
prep_plan="$reader_dir/preparation-plan.json"
reader plan --out "$prep_plan"
reader validate --plan "$prep_plan"
```

## Create storage and prepare the model and data

**First online operation.** Where will the next worker find the inputs when the preparation worker exits? Run this command only within the reader-owned allocation. It checks the authenticated workspace and Secret, then creates the scoped Volume (persistent files) and registry (operation records).

```bash
reader setup --plan "$prep_plan"
```

Expect `status: resources_created`. A different authenticated workspace or an existing resource name stops setup. If creation partially succeeds, inspect your actual account before making any further change.

```bash
reader prepare --plan "$prep_plan" --out "$reader_dir/preparation-submission.json"
```

Expect a submission receipt with `status: submitted` and a call ID after image resolution. Keep the client running until it returns. The remote CPU worker acquires the pinned model, installs the fixed-date tool template and verifies the weight shards, configuration and tokenizer. For the calculator it downloads the released Calc-X archive and selects the data with seed 329. For asset status it generates the versioned synthetic registry and panels. Input hashes and ordered case IDs go into a manifest. If a build fails, access is denied or a submission is `unknown`, retain the receipts. An ambiguous submission can still be running. Use status and the [owned stop procedure](#stop-and-recover-within-ownership) before considering another allocation.

Poll with the following command. Expect exit code **2** while full workflow evidence is incomplete. Read the JSON: a completed preparation is `operations.prepare.result.status` equal to `completed`. A missing or failed result cannot supply the next plan.

```bash
reader status --plan "$prep_plan"
```

Once preparation completes, collect its evidence into a fresh directory. This inspection also returns 2 because the six GPU stages have yet to run.

```bash
reader collect --plan "$prep_plan" --out "$reader_dir/prepared"
```

Expect `prepared/prepare/assets.json` under your session directory and an empty `collection_errors` list in `prepared/collection.json`. If files or hashes are missing, inspect the preparation receipt and collection errors. The model weights stay in the Volume. Keep staging for the independent preflight checks.

## Bind actual inputs, preflight, then launch

**Can a different set of model or dataset bytes satisfy the preparation plan alone?** Make the second plan locally, this time binding the returned manifest. Both outputs should agree on the new plan hash and say `training_admissible: true` for the real qualification profile. This checks manifest consistency locally. The worker must still verify the actual files.

```bash
train_plan="$reader_dir/training-plan.json"
reader plan --assets "$reader_dir/prepared/prepare/assets.json" --out "$train_plan"
reader validate --plan "$train_plan"
```

Preflight submits a **billable remote CPU** check in the training image. It rehashes inputs, checks source/package identities and the reviewed dependency exceptions, loads the tokenizer, checks HF/W&B read access and exercises valid and invalid tool requests over stdio. It loads no model weights.

```bash
reader preflight --plan "$train_plan" --out "$reader_dir/preflight-submission.json"
reader status --plan "$train_plan"
```

Expect `submitted` first. Repeat the status command until `operations.preflight.result.status` is `completed` and its nested `result.status` is `passed`. Exit 2 still applies. If a hash, package, tool probe or account check fails, inspect the preflight evidence. Launch requires a passing durable receipt for this exact plan.

**Which stages should change the weights?** Within the funded allocation, launch once. The detached supervisor schedules single-use workers in one GPU lane. Expect a submission receipt. Poll status to follow the stages.

```bash
reader launch --plan "$train_plan" --out "$reader_dir/training-submission.json"
reader status --plan "$train_plan"
```

| Stage for each algorithm | Look for this evidence | If it is missing |
|---|---|---|
| Baseline | 64 scored episodes from the base actor, zero optimizer calls. A live chain links a model action to the tool, the observation in a subsequent model call and a delivered reward. | A completed baseline lacking a chain causes training and reload for that algorithm to be skipped with `baseline_integration_unqualified`. |
| Training | Two collections of 8 questions × 4 complete attempts. Two actor optimizer calls, also two critic calls for PPO-MC. Serving refresh at steps 0, 1 and 2. Native step-two checkpoint with hashes for each role. | A missing reward or missing generated tokens invalidates the collection. Keep the failure. Four confirmed zero rewards in one group are valid data. |
| Checkpoint evaluation | A different worker verifies and loads the checkpoint, refreshes at step 2 and evaluates all 64 cases with zero optimizer calls. | Check the matched checkpoint digest, distinct call/container identities and complete evaluation receipt. A directory name alone cannot establish reload. |

Once the supervisor reaches a terminal outcome, collect the bounded evidence, including failures, into a new directory. `status` can return 2 both while running and after a failed stage. Inspect the supervisor index and durable receipts to distinguish them.

```bash
reader collect --plan "$train_plan" --out "$reader_dir/training-evidence"
```

Open `training-evidence/collection.json`. A successful full qualification has `evidence_complete: true`, zero exit status, and integration coverage in baseline, training and fresh reload for **each** algorithm. Read operational completion, integration and informative GRPO groups separately. Operational checks can pass with integration gaps or with zero informative groups. A chain ending in a confirmed reward of zero can establish tool integration. Only the final answer determines task correctness. For a measured comparison, read the [calculator qualification](READER_QUALIFICATION_2026-11-19.md#live-gpu-outcomes) and the [asset qualification](PHASE3_QUALIFICATION_2026-11-24.md).

## Stop and recover within ownership

If the allocation must stop, use the plan for the operation you intend to cancel. Use `"$prep_plan"` in place of `"$train_plan"` below for preparation. This is an online cancellation command.

```bash
reader stop-owned --plan "$train_plan"
```

Expect `status: cancellation_requested`, the known owned calls and any cancellation errors. The command writes a plan-specific stop marker. A missing call ID leaves the outcome unknown. Check status again, inspect owned provider state and collect to a fresh directory. Preserve ambiguous reservations and partial evidence. An interrupted episode has no confirmed training reward. Reconcile exposure before any fresh attempt. The 300-GiB storage ceiling can be overshot between checks. The seven-day retention interval is a review point. Stopping calls leaves assets and checkpoints in storage.

## What the reference workflow qualifies


**Phase 3 update:** New plans use `reader-task-v2`, support reader-owned GitHub forks and the synthetic `asset-status-v1` task. See [Task adaptation](TASK_ADAPTATION.md) for the workflow, migration and all-stage integration rule. The first [asset live report](PHASE3_QUALIFICATION_2026-11-22.md) records 64/64 baseline integration for both algorithms, followed by incomplete training collections and zero optimizer updates. After the `asset-episode-v2` correction, the [2026-11-24 report](PHASE3_QUALIFICATION_2026-11-24.md) records all six asset stages completed, with native checkpoints and fresh-worker reload for both algorithms. The [closeout](PHASE3_CLOSEOUT_2026-11-24.md) holds the preservation inventory and accounting, and the [engineering checklist](PHASE3_ENGINEERING_CHECKLIST.md) lists what to check before a new attempt. The earlier calculator report retains its original v1 semantics and exact execution identities.

**Recovery status.** Both collection routes completed development calibration (120 episodes each, zero updates). GRPO then passed its recorded functional gates in A03 (two actor updates) and P02 (16), each including fresh-worker checkpoint reload. P02's terminal greedy policy answered `not_ready` on all 24 development cases. Strict success rose from 9/24 to 12/24 while decision accuracy stayed at 12/24. [Calibration](PHASE4B_LIVE_CALIBRATION.md) · [A03](PHASE4C_GRPO_CAMPAIGN_CLOSEOUT.md) · [P02](PHASE4C_FUNCTIONAL_PILOT_CLOSEOUT.md) · [Paired decisions](PHASE4C_PILOT_LEARNING_DIAGNOSIS.md)

Recovery-specific PPO-MC live training and reload remain unqualified. The full three-seed recovery study remains unexecuted, including its final-panel evaluations. The witness-selection supervised fine-tuning (SFT) proposal remains deferred. [Offline qualification checks](PHASE4C_VERIFICATION.md) · [Study protocol](PHASE4C_LEARNING_STUDY_PROPOSAL.md) · [Draft PR #14](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/pull/14)

**The reference Llama/calculator workflow has completed real baseline evaluation, two-update training, native checkpoint saving and fresh-worker reload for both algorithms.** CPU preflight and model/tool integration passed. GRPO observed zero informative reward groups; useful learning and reliable accuracy improvement are not established. See the [dated qualification report](READER_QUALIFICATION_2026-11-19.md) for exact executed source, evidence, earlier failures and costs. This guide is the canonical reader procedure. It does not reopen the completed educational release or attribute its results to new code.

The current execution schema is `reader-task-v2`; archived `reader-reference-v1` plans require regeneration for new execution. Its numerical protocol remains `matched-rollout-mean-v2`: terminal binary rewards; eight question presentations and four distinct complete rollouts per presentation; generated-token conservation, observation masking and equal-rollout loss weighting. Missing rewards, invalid context, incomplete groups and truncation remain errors. PPO-MC retains the learned critic, gamma/lambda one, true-terminal zero bootstrap, terminal Monte Carlo returns, tokenwise residual advantages and valid-token whitening. The calculator prompt, extractor and verifier are unchanged.

| Required configuration | GRPO | PPO-MC |
|---|---|---|
| Reference Llama + calculator | Live operational, integration and reload checks passed; 0/16 informative groups | Live operational, integration and reload checks passed; learning benefit not established |
| Alternate compatible model + calculator | Subsequent implementation and qualification | Subsequent implementation and qualification |
| Reference Llama + recovery-readiness-v1 (`recovery-episode-v1`) | A03 two-update qualification and P02 16-update development pilot passed their functional gates and reloaded in fresh workers | Development collection ran live. Training and reload have offline checks only |
| Reference Llama + asset-status-v1 (`asset-episode-v2`) | Live baseline, two updates, native checkpoint and fresh-worker reload completed; 64/64 before and after; 1/16 informative groups; learning benefit not established. Earlier attempts a01–a04 remain unqualified under their original policy | Live baseline, two actor and two critic updates, native checkpoint and fresh-worker reload completed; 64/64 before and after; poor critic diagnostics; learning benefit not established. Earlier attempts a01–a04 remain unqualified under their original policy |

Each cell needs its own execution evidence. Evaluation reload qualifies scoring with saved weights. Resumed optimizer-state training requires a different test.

### Prepared inputs and worker identities

The example uses the pinned model revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`, the `llama3b` profile and the `llama3_json` tool parser. CPU preparation uses four CPUs and 16 GiB. Image resolution may invoke the shared heavy builder (8 CPUs, 64 GiB, 7200 seconds), dependency finalizer (4 CPUs, 16 GiB, 1800 seconds) and reader overlay (4 CPUs, 16 GiB, 900 seconds). Provider-default layers and cache availability also affect exposure. The client shortens its dispatch deadlines by 30 seconds for clock uncertainty. See the [profile](../examples/reader-profile.synthetic.json) and [build and runtime evidence](READER_QUALIFICATION_2026-11-19.md).

All four prepared panels and evaluation JSONL files are checked for hashes, schemas, ordered IDs, duplicate/conflicting questions, split isolation and JSONL/Parquet agreement. The final panel contains 64 cases, the training pool 256, with 16 nested training-smoke and eight separate evaluation-smoke cases. Worked solutions are discarded. Only `question` enters the prompt; `result` remains the verifier's private answer key. Preflight and stages rehash raw inputs and reproduce deterministic selection, in addition to checking prepared outputs. Do not remove staging before qualification.

Each worker checks source, plan, configuration and model/data identities. GPU stages run CUDA and FlashAttention checks before the model workflow. Each role's native checkpoint inventory includes model, optimizer and extra-state files with sizes and hashes. The reload must match the checkpoint before and after evaluation. Receipts bind each worker to its provider container ID and call ID. Training and reload can share a hostname or reuse a process number and still be distinct workers. Read the container and call identities together with the recorded launcher/trainer process IDs. [Stage evidence validator](../experiment/reader_evidence.py)

For deeper local checks, the repository includes the [reader offline suite](../scripts/test_reader_offline.sh) and [task contract suite](../scripts/test_task_local.sh). The walkthrough above uses focused exercises. Full suite inputs include synthetic bytes and mocked service boundaries.

### Reading the collected metrics

Additional checks require positive valid generated-token counts, two complete collections, finite loss/gradient diagnostics and checkpoint-save events. Require `actor/pg_loss` at both global updates 1 and 2; PPO-MC additionally requires `critic/vf_loss` at both updates. Finite zero is valid. Records are joined by update/metric, permitting split role records and identical duplicates while rejecting missing, invalid or conflicting evidence. The pinned [VERL loss emitter](https://github.com/volcengine/verl/blob/7aed6b230776f963fa09509c10d9c3a767d1102c/verl/workers/utils/losses.py) and [Agent Lightning trainer](https://github.com/microsoft/agent-lightning/blob/cbc56e41e0eab643aa71d195e92e5bba652fa683/agentlightning/verl/trainer.py) reduce minibatch metrics and log at global training steps; this check does not assume one JSONL record per update. The lab's existing matched runtime emits the corresponding critic metric. Retain reward-group variation. If all GRPO groups have equal reward, report successful operational execution separately from the absence of informative task-directed policy-gradient signal. Nonzero optimizer calls alone do not establish learning. Reload here qualifies evaluation of the saved native checkpoint; resumed training and arbitrary weight exports are separate claims.

`status` and `collect` report four distinct dimensions per algorithm: operational stage completion; model/tool integration coverage by baseline/training/checkpoint-eval stage; informative GRPO reward-group count (not applicable to PPO-MC); and remaining qualification gaps. `evidence_complete` and exit code zero mean the operational receipt set is complete, **not** that integration or learning is established. `live_qualification` retains “gaps remain” when needed.

Each completion includes `model_tool_integration`, with trace file hashes and observed chain identifiers. The reader joins existing schema-1 traces by experiment/W&B/stage, trace, case, rollout and attempt, episode parent span, tool-call ID and time order. A valid chain requires a successful recorded model request returning the selected task tool (`calculate` or `lookup_asset`), the matching successful tool invocation with the same arguments, a subsequent successful model request containing that exact tool observation, and a successful episode with a delivered binary terminal reward. It uses AutoGen's existing `FunctionCall`, `ToolResult.to_text()` and `FunctionExecutionResultMessage` representations. Synthetic, inconsistent or duplicate spans do not qualify. Under `reader-integration-v2`, at least one valid live chain in **each algorithm's baseline, training and fresh-reload stage** is required to qualify harness integration. The dated v1 report retains its older any-stage interpretation. A reward of zero can still demonstrate integration; correct answers without an observed tool-continuation chain cannot.

Operational completion may succeed with zero informative GRPO groups. The report then explicitly states that informative group-relative task-advantage evidence was not observed. Neither nonzero loss, improved accuracy nor universal tool use is required. Missing evidence or unfavorable outcomes must be retained as observed; do not automatically retry, tune rewards, change inputs or select favorable trajectories. Hosted W&B history, meaningful learning and resumed training remain separate claims.

## Historical matching, build limits and next seams

The calculator asset manifest compares stable prepared-file hashes and ordered case IDs with `matched_data_lock.json`. Preparation timestamps and absolute local paths do not themselves imply different data. Historical acquisition hashes are absent from that lock and remain unverified. Exact prepared byte matches are reported per file. Different serialization/content or unavailable original bytes receive a new exact input identity and are a **methodology rerun with identified input differences**, not a recreation of historical results. Old locks are never rewritten to make new inputs pass.

The client lock is not a full remote transitive dependency lock. The shared pinned build scripts, guarded backend sources and reviewed package-conflict policy are retained. Cached historical success does not prove a cold build, provider capacity, H100 availability or compatibility with future artifact resolution. A failed new build is evidence, not permission to relax the guards. No author checkpoints, ignored helper, private Git object or author-prepared assets are dependencies of this procedure.

The immutable historical distinctions remain: private export `eef9c78707b33725847fb15ca19592733a84c6c3`; r03 execution `5fb6adfd48ee245509f7dff3ab1d849fcf707d86`; attributed public representation `89fe32f92f29209803c13264baed951713741d42`; study `agl-explore-20-v1-r03`. The [publication provenance](../PUBLICATION_PROVENANCE.json), [licenses](../LICENSING.md), historical evidence and learning guide retain their scope. Reader code and runs have their own commit and execution identities.

Task customization is implemented at `reader_contract.task_identity`, task preparation/inspection/probe dispatch and the shared launcher. [Task adaptation](TASK_ADAPTATION.md) explains the working asset-status task, exact visible/privileged data boundary and third-task extension points. Alternate models still need a new immutable actor/tokenizer/template/parser/critic compatibility profile and live qualification; a legacy selector is insufficient.
