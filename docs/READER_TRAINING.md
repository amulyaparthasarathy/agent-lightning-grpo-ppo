# Reader-owned reference training

Use your own Modal workspace, Hugging Face account and Weights & Biases (W&B) project to run one fixed experiment. You can practice planning on your computer before opening any accounts. Remote work requires a separate allocation that you fund.

| What you choose | What the reference run does |
|---|---|
| [Calculator](../examples/reader-profile.synthetic.json) or [synthetic asset status](../examples/reader-profile.asset.synthetic.json) | Uses `meta-llama/Llama-3.2-3B-Instruct` at revision `0cb88a4f764b7a12671c53f0838cd831a0843b95` with its matched tokenizer and tool parser. |
| Your accounts, project, resource names and storage paths | Runs **both** GRPO and this lab’s PPO-MC. Each gets a baseline, two actor updates and a fresh-worker checkpoint evaluation. PPO-MC also takes two critic updates. |
| Time and budget bounds within the [reader contract](../experiment/reader_contract.py) | Runs six stages in one GPU lane. Each requests one `H100!`, 16 CPUs and 196608 MiB (192 GiB) of **host** memory. |

Changing the W&B account is supported. A different model needs a compatible actor, tokenizer, parser and critic configuration and its own qualification. The standard JSON has no learning-rate or update-count switch. This short run answers whether the workflow operates on your allocation. The [three-seed calculator study](PHASE2_CLOSEOUT_2026-11-15.md) measured a different, longer experiment.

## Client and source

Use Bash on Linux or WSL2, including an existing Unix Workbench. You need Git, uv, an editor, and network access for the initial clone and package downloads. The local client uses Python 3.12 and pins Modal to 1.6.3. uv can acquire Python 3.12 during sync. CUDA, Torch, VERL and vLLM run on Modal. The optional [local CPU harness](TASK_ADAPTATION.md#run-the-local-checks) uses a separate environment. [Install uv](https://docs.astral.sh/uv/getting-started/installation/) · [Python with uv](https://docs.astral.sh/uv/guides/install-python/) · [Client dependencies](../pyproject.toml)

Start in a directory you own. Clone once, then run the rest of the procedure **in the same Bash terminal at the repository root**. With an existing clean full clone of the intended revision, start at the history check.

```bash
git clone https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo.git
cd agent-lightning-grpo-ppo
git merge-base --is-ancestor 5c069f7e9cc23d03050d13360ce43e27e47e54aa HEAD
git rev-parse HEAD
uv sync --locked --inexact --python 3.12
.venv/bin/python --version
.venv/bin/python scripts/reader_training.py --help
git status --short
```

Expect the ancestry check to exit 0 silently, a commit ID, Python 3.12, the command list from `plan` to `stop-owned`, and empty Git status. Keep that revision for the allocation. A ZIP lacks Git history and origin. A shallow clone can omit the required baseline above. Use a full clone if the baseline is missing. If Git status lists files, preserve your changes and use a clean checkout for the allocation.

A plan binds the GitHub origin, commit, tracked source bytes (including Markdown), profile and any asset manifest. Edits and nonignored untracked files can block planning. A pull that changes HEAD invalidates an existing plan even when the checkout is clean. Keep private profiles, plans and receipts under ignored `.local/`. Keep the source and profile stable through collection. For deliberate code changes, use a clean committed fork descended from the baseline. [Source admission](../experiment/reader_source.py)

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

**Optional alteration.** Repeat the block with `reader_template=examples/reader-profile.asset.synthetic.json`. The task becomes `asset-status-v1`, and its contract names `lookup_asset`. The reference model, two algorithms and offline admission result stay the same. You can finish the lesson here. The [two local exercises](learning/08-reproduce-and-operate.md#two-credential-free-checks) also work without accounts.

## Configure a reader-owned allocation

Three different things are called a profile or credential in this workflow. Put each value where its consumer can read it.

| Where | What belongs there | Who uses it |
|---|---|---|
| Local Modal authentication profile | Modal credentials stored by its CLI under a local name such as `reader-lab` | Your terminal selects this profile to contact a workspace. |
| Reader JSON in `.local/` | Names, paths, model/task settings and budget bounds | The planner fingerprints these values. The remote application receives the configuration. Keep tokens out of this file and its copies in receipts. |
| Named Modal Secret | Two private values under the exact keys `HF_TOKEN` and `WANDB_API_KEY` | Modal injects them into the remote preparation, preflight and GPU workers. A local W&B login or a local `.env` file does not populate the Secret. [Worker bindings](../reader_app.py) |

### 1. Select a Modal workspace and environment

Sign in to Modal and select the workspace that will own the resources and charges. A workspace contains environments, each with its own Secrets, Volumes and Dicts. Use an existing environment such as the default `main`. Check your permission to create resources there, your ability to fund this allocation and GPU access. Modal requires a valid payment method for GPU use. H100 capacity in your workspace is only known at execution. [Workspaces](https://modal.com/docs/guide/workspaces) · [Environments](https://modal.com/docs/guide/environments) · [GPU requirements](https://modal.com/docs/guide/gpu)

**Check name compatibility before creating anything.** Every `modal.*` name, the W&B entity and project, and `experiment` must match `[a-z0-9][a-z0-9-]{0,47}`. Start with a lowercase ASCII letter or digit, use only lowercase letters, digits and hyphens, and stay within 48 characters. The validator also rejects the prefixes `cs329z`, `agl-explore`, `agl-matched` and the exact historical name `agent-lightning-credentials`. App, registry, Volume and Secret need four distinct names. A real provider identifier can be valid there and still fail here (for example `my_team` or `ResearchLab`). Copy existing identifiers exactly. If one is incompatible, use a compatible namespace you actually own or seek a reviewed validator change. Changing spelling only in the JSON points at a different identity. [Name and path checks](../experiment/reader_contract.py)

Choose a local profile label. For a **new** label, run the following and complete the browser flow for the intended workspace. To reuse an existing profile, export its actual name and skip `token new`.

```bash
export MODAL_PROFILE=reader-lab
.venv/bin/modal token new --profile "$MODAL_PROFILE"
```

Check the effective context in that terminal. The last two commands contact Modal and read metadata.

```bash
.venv/bin/modal profile current
.venv/bin/modal token info
.venv/bin/modal environment list
```

Copy the local label into `modal.profile`, the **Workspace name** from `token info` into `modal.workspace`, and a listed environment name into `modal.environment`. Token info also displays IDs, so keep its output private. Environment credentials such as `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` take precedence over stored profile values. Selecting a profile alone cannot switch an overriding credential pair. If the workspace differs, use a shell without unintended credential overrides (including Modal OAuth variables if configured), select the profile again and recheck the workspace before proceeding. You can diagnose this without printing credential values. [Modal configuration precedence](https://modal.com/docs/sdk/py/latest/config)

### 2. Obtain Hugging Face access to the pinned model

Sign in as the person whose token the worker will use. Open the model page, review its terms, request access and wait for approval if needed. Gated access belongs to the individual HF account. In **Settings → Access Tokens**, create a read token or a fine-grained token permitted to download this gated repository. Use it as `HF_TOKEN` in step 4. Downloading weights for Modal requires repository access. Inference runs on Modal. [Llama model and access form](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) · [Gated access](https://huggingface.co/docs/hub/en/models-gated) · [Token settings](https://huggingface.co/settings/tokens) · [Token scopes](https://huggingface.co/docs/hub/en/security-tokens)

### 3. Select a W&B project and create a W&B API key

Sign in to multi-tenant W&B through CoreWeave Forge. Open the project your team will use. To create one, open the global navigation, select **Projects → View all → Create new project**, choose the owning team, a compatible name and visibility, then create it. From its URL, `https://forge.coreweave.com/wandb/ENTITY/PROJECT`, copy the two path segments into `wandb.entity` and `wandb.project`. The project must already exist when CPU preflight lists readable projects. [Forge navigation and URLs](https://docs.coreweave.com/forge-whats-moved) · [Projects](https://docs.coreweave.com/products/wandb/track/project-page)

Use an identity that can read the project and create, log to and finish runs there. In the Forge account selector, open the intended organization's settings and its **API Keys** tab to create a key. Personal keys, where permitted, are under **User settings → API keys**. Copy the full key, including its prefix, when it is shown. Use this W&B-compatible key as `WANDB_API_KEY` in step 4. The existing `wandb` SDK and API endpoint still work with Forge. Leave the SDK and endpoint settings as supplied. Preflight tests readability. Each execution stage needs to write and finish its run. [API key management](https://docs.coreweave.com/products/wandb/platform/hosting/iam/api-keys) · [Preflight implementation](../experiment/reader_preflight.py)

### 4. Create the Secret in the exact Modal environment

In the Modal dashboard, switch to the workspace and environment from step 1. Open **Secrets**, create a custom Secret with a fresh permitted name (for example `reader-lab-credentials`), and add **both** `HF_TOKEN` and `WANDB_API_KEY` with their respective private values. Save it and put only its name in `modal.secret`. The earlier operator instructions refer to the protected `agent-lightning-credentials` Secret. Leave that historical resource alone. [Modal Secrets](https://modal.com/docs/guide/secrets)

List Secret metadata in the selected environment (replace `main` if needed). Look for your exact name. This confirms visibility there. The remote workers will exercise the values.

```bash
.venv/bin/modal secret list --env main
```

### 5. Fill in the reader JSON

Copy the maintained template for your chosen task into the session directory and edit that copy. Keep the original template for offline practice.

```bash
reader_profile="$reader_dir/profile.json"
cp "$reader_template" "$reader_profile"
```

Here is a complete **fictitious calculator allocation**. Replace its provider identities with the exact values from steps 1–4. For asset status, start with the asset template so the task and data destination already agree.

```json
{
  "schema": "reader-task-v2",
  "fixture": false,
  "modal": {
    "profile": "reader-lab",
    "workspace": "my-modal-workspace",
    "environment": "main",
    "app": "reader-calc-001-app",
    "registry": "reader-calc-001-registry",
    "volume": "reader-calc-001-volume",
    "secret": "reader-lab-credentials"
  },
  "paths": {
    "model": "reader-assets/llama3b-v1",
    "data": "reader-assets/calculator-v1",
    "staging": "reader-staging/preparation-a01",
    "output": "reader-runs"
  },
  "experiment": "reader-calc-001",
  "attempt": 1,
  "wandb": {"entity": "my-wandb-team", "project": "agent-rl-lab"},
  "model": {
    "id": "meta-llama/Llama-3.2-3B-Instruct",
    "revision": "0cb88a4f764b7a12671c53f0838cd831a0843b95",
    "profile": "llama3b",
    "parser": "llama3_json"
  },
  "task": "calculator-v1",
  "algorithms": ["grpo-matched-v2", "ppo-mc-matched-v2"],
  "resources": {"gpu": "H100!", "cpu": 16, "memory_mib": 196608},
  "bounds": {
    "stage_seconds": 3600,
    "startup_seconds": 600,
    "preparation_seconds": 3600,
    "preflight_seconds": 900,
    "concurrency": 1,
    "worker_hourly_usd": 10,
    "supervisor_hourly_usd": 0.1,
    "preparation_budget_usd": 5,
    "training_budget_usd": 125,
    "build_budget_usd": 10,
    "storage_gib": 300,
    "retention_days": 7,
    "storage_budget_usd": 7,
    "collection_gib": 2,
    "total_budget_usd": 148
  }
}
```

| Reader JSON field | Action | Where to get the value or what it controls |
|---|---|---|
| `schema` | Keep fixed | `reader-task-v2` for new plans. |
| `fixture` | Must replace | Change the practice template’s `true` to `false` for the real allocation. |
| `modal.profile` | Must replace | Local label from `modal profile current`. |
| `modal.workspace` | Must replace | Workspace name reported by `modal token info`. |
| `modal.environment` | Must replace | Existing namespace selected from `modal environment list`. |
| `modal.secret` | Must replace | Exact name of the two-key Secret you just created in that environment. |
| `modal.app` | Choose/review | Fresh name for the app under which workers run. |
| `modal.volume` | Choose/review | Fresh name for persistent files. Reader setup creates this Volume. |
| `modal.registry` | Choose/review | Fresh name for the Modal Dict (key/value store). Setup creates it for reservations, operation status and call identities. |
| `wandb.entity`, `wandb.project` | Must replace | Exact owning namespace and existing project from the W&B project URL. |
| `experiment`, `attempt` | Choose/review | Label this allocation and choose an integer 1–99 (start with 1). Together they name the run directory, e.g. `reader-runs/reader-calc-001/a01`. |
| `paths.model` | Choose/review | Destination for the pinned weights and tokenizer inside the Volume. Preparation populates it. Model selection is fixed below. |
| `paths.data` | Choose/review | Prepared task data in the same Volume. Use the selected task’s template default for a first run. |
| `paths.staging` | Choose/review | Acquisition and preparation files retained for independent verification. |
| `paths.output` | Choose/review | Parent directory for this allocation’s stage evidence and checkpoints. |
| `model.id`, `model.revision`, `model.profile`, `model.parser` | Keep fixed | Exact Hub identifier, commit, lab model/tokenizer configuration and tool-call parser shown above. |
| `task` | Choose/review | Select `calculator-v1` or `asset-status-v1` via its template. Omit `purpose` for this six-stage procedure. The [asset development variant](TASK_ADAPTATION.md#greedy-evaluation-does-not-test-sampled-training) has different stages. |
| `algorithms` | Keep fixed | Both algorithm IDs in the order shown. |
| `resources.gpu`, `resources.cpu`, `resources.memory_mib` | Keep fixed | `H100!`, 16 CPUs and 196608 MiB host memory per GPU worker. |

All four paths are relative to the root of your Volume, mounted at `/data` on workers. For example, `reader-assets/llama3b-v1` becomes `/data/reader-assets/llama3b-v1` remotely. Use 1–180 characters from ASCII letters, digits, underscore, dot, slash and hyphen. Empty components, `.`, `..`, a leading slash and the protected first components `models`, `runs`, `exports`, `restored`, `phase2`, `data-small` are rejected. The four paths must differ and none may contain another.

**Fixed training settings.** The plan uses seed 9001 with derived random streams. Calculator data selection uses seed 329. Each algorithm makes two collections of eight questions × four attempts. Training samples at temperature 1.0 and evaluation uses 0.0. The actor learning rate is `1e-6`. PPO-MC adds a critic at `1e-5` with discount and GAE parameters both 1. The actor KL penalty and entropy bonus are disabled. The terminal checkpoint is step 2. Each stage writes its effective settings in `resolved_config.json`. [Matched configuration](../experiment/matched_launch.py) · [Actor and critic settings](../experiment/ppo_contract.py) · [Serving temperatures](../experiment/launch.py)

### 6. Review the allocation before the first submission

All `bounds` values must be positive and finite. Times, sizes, days and concurrency below require integers. The dollar figures are **example planning allowances**. Review current prices, existing exposure, capacity and retention for your separately funded account. Your actual provider charges can exceed $148. This JSON neither enforces a provider spending cap nor grants spending authority. [Modal pricing](https://modal.com/pricing) · [Budget validation](../experiment/reader_contract.py)

| Field under `bounds` | Action | Example | Meaning and accepted limits |
|---|---|---:|---|
| `stage_seconds` | Choose/review | 3600 | Deadline per GPU stage, 300–7200 seconds. |
| `startup_seconds` | Choose/review | 600 | Startup heartbeat deadline, 60–1800 seconds and strictly less than the stage deadline. |
| `preparation_seconds` | Choose/review | 3600 | CPU asset preparation, 300–3600 seconds. |
| `preflight_seconds` | Choose/review | 900 | CPU verification, 60–900 seconds. |
| `concurrency` | Keep fixed | 1 | Exactly one GPU stage at a time. |
| `worker_hourly_usd` | Choose/review | 10 | Combined worker rate used in reservations, minimum $7/hour. |
| `supervisor_hourly_usd` | Choose/review | 0.10 | Declared supervisor rate, minimum $0.07/hour. The six-stage reservation charges supervisor time at the higher example worker rate. |
| `preparation_budget_usd` | Choose/review | 5 | Preparation and preflight allowance, minimum $3. |
| `training_budget_usd` | Choose/review | 125 | Must cover all six stage deadlines plus the supervisor deadline at the worker rate. |
| `build_budget_usd` | Choose/review | 10 | Image-build allowance, minimum $5. Cache availability, build time and cold-build compatibility can vary. |
| `storage_gib` | Choose/review | 300 | Monitored file-size threshold for the four paths, 100–300 GiB. Usage can overshoot between checks. |
| `retention_days` | Choose/review | 7 | Review retained files in 1–7 days. No automatic expiry or deletion takes place. |
| `storage_budget_usd` | Choose/review | 7 | At least `storage_gib × 0.09 × retention_days / 30` dollars. Ongoing retention can incur additional charges. |
| `collection_gib` | Choose/review | 2 | Bounded evidence download, 1–5 GiB per collection. Reserve $0.04/GiB in the total. |
| `total_budget_usd` | Choose/review | 148 | At least the sum of training, preparation, build and storage allowances plus the collection reservation. |

For the example, six stage deadlines total 21,600 seconds. The supervisor gets those 21,600 seconds **plus 600**, so the training allowance must cover `(21,600 + 22,200) × $10 / 3,600`, about $121.67. The example reserves $125 for that category. All categories together require `$125 + $5 + $10 + $7 + $0.08 = $147.08`. A previously successful cached image provides no measurement of a fresh cold build.

### Make the real preparation plan locally

The preparation plan describes where the worker will put the model and data. Select your Modal profile from the edited JSON and validate this first plan locally. Expect matching hashes and `training_admissible: false` because there are no prepared assets yet.

```bash
MODAL_PROFILE="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["modal"]["profile"])' "$reader_profile")"
export MODAL_PROFILE
prep_plan="$reader_dir/preparation-plan.json"
reader plan --out "$prep_plan"
reader validate --plan "$prep_plan"
```

### If a check or submission fails

| Symptom | Next action |
|---|---|
| Missing baseline, dirty checkout or stale plan | Check history, Git status, revision and profile. Keep prior receipts. Make a fresh plan only against the intended clean source. |
| Invalid name, wrong workspace or missing Secret | Compare exact provider names with the naming rule. Check credential overrides, token workspace, environment and Secret keys before any resource creation. |
| Access denied during preparation or preflight | Confirm the HF token belongs to the approved account and can download the pinned repository. For W&B check the URL segments, project existence and the key’s membership. If logging fails after preflight, check run-write permission. |
| Setup created only the Volume | Inspect Volume and Dict in the exact environment. Setup creates them in order and refuses existing names. Repeating it cannot repair a partial setup. |
| Unknown submission or existing reservation | A missing acknowledgment or call ID can mean work is still running. Inspect status, receipts and owned calls before considering another action. Use the [plan-specific stop](#stop-and-recover-within-ownership) if needed. |
| Registry reports an allocation already reserved | Preparation reserves `asset-allocation` and launch reserves `qualification-allocation` in the Dict, even before a worker is confirmed. Changing only `attempt` or `experiment` leaves these reservations in place. Reconcile old work and exposure before planning any separately funded fresh allocation with new resources. |
| `status` or `collect` exits 2 | If JSON was returned, inspect the operations, supervisor index and incomplete evidence. A run may be pending, skipped or failed. If the CLI prints an error instead, diagnose that error. Never infer a retry from the exit code. |
| Receipt or collection destination already exists | Read the existing receipt. Collect again only to a new directory so the partial evidence remains available. |
| Image build or dependency guard fails | Preserve the failure and review the build evidence and exposure. A historical cached build does not establish a successful cold build today. |

## Create storage and prepare the model and data

**First resource creation by the reader CLI.** Account, project and Secret setup already used the providers. The next command verifies the authenticated workspace and existing Secret with both required keys. It then creates the Volume followed by the Dict, with existing names disallowed. Use it only when you can fund the allocation.

```bash
reader setup --plan "$prep_plan"
```

Continue only after setup returns `status: resources_created`. The next command can start image resolution and a billable CPU worker.

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

**Which stages should change the weights?** Within the funded allocation, launch once. **One launch schedules the entire six-stage plan**, including both training and checkpoint evaluation. The detached supervisor schedules single-use workers in one GPU lane. Expect a submission receipt. Poll status to follow the stages.

```bash
reader launch --plan "$train_plan" --out "$reader_dir/training-submission.json"
reader status --plan "$train_plan"
```

| Stage for each algorithm | Look for this evidence | If it is missing |
|---|---|---|
| Baseline | 64 scored episodes from the base actor, zero optimizer calls. A live chain links a model action to the tool, the observation in a subsequent model call and a delivered reward. | A completed baseline lacking a chain causes training and reload for that algorithm to be skipped with `baseline_integration_unqualified`. |
| Training | Two collections of 8 questions × 4 complete attempts. Two actor optimizer calls, also two critic calls for PPO-MC. Serving refresh at steps 0, 1 and 2. Native step-two checkpoint with hashes for each role. | A missing reward or missing generated tokens invalidates the collection. Keep the failure. Four confirmed zero rewards in one group are valid data. |
| Checkpoint evaluation | A different worker verifies and loads the checkpoint, refreshes at step 2 and evaluates all 64 cases with zero optimizer calls. | Check the matched checkpoint digest, distinct call/container identities and complete evaluation receipt. A directory name alone cannot establish reload. |

A complete run therefore has 384 episodes across both algorithms. Once the supervisor reaches a terminal outcome, collect the bounded evidence, including failures, into a new directory. `status` can return 2 both while running and after a failed stage. Inspect the supervisor index and durable receipts to distinguish them.

```bash
reader collect --plan "$train_plan" --out "$reader_dir/training-evidence"
```

### Find your outputs and interpret the result

| Where to look | What you can inspect |
|---|---|
| Your printed local session directory | `profile.json`, the two plans, and submission receipts with plan hashes and call IDs. The practice plan is separate. |
| Local `training-evidence/collection.json` and its per-algorithm stage directories | The supervisor index, operation results and downloaded text evidence such as `receipt.json`, `completion.json`, `resolved_config.json`, training `metrics.jsonl` and logs. Check `collection_errors` for partial downloads. |
| W&B project in Forge | Hosted metrics. Find each stage by the ID or run name in the training plan’s `jobs[].stages[].wandb`. Collection checks local evidence of logging and explicitly leaves hosted history independently unverified. |
| Your Modal Volume | Prepared weights and data at `paths.model` and `paths.data`. Each stage writes to its plan’s `jobs[].stages[].output`. The native tensors are under the training stage’s `checkpoints/global_step_2/`, including the actor and, for PPO-MC, the critic. |

**Collect does not download model tensors.** It copies selected `.json`, `.jsonl`, `.log` and `.txt` files under the plan’s run root, skips temporary directories and applies the collection limit. A checkpoint inventory in the download describes the bytes that stay in the Volume. [Collection code](../scripts/reader_training.py)

Read `collection.json` in this order:

1. **Did the operations finish?** Look for `evidence_complete: true`, `operational_completion: completed`, and empty collection errors. Exit 0 reports complete operational evidence.
2. **Did the harness work?** Inspect each algorithm’s `harness_integration_qualified` and `model_tool_integration`. Each of its three stages needs a live tool call, its observation in a later model request, and a delivered reward. Even a confirmed zero reward can establish integration.
3. **Was there a training signal?** Read `informative_grpo_reward_groups`. A group with four equal rewards supplies no relative reward contrast. Read the actor and PPO-MC critic diagnostics alongside the update counts.
4. **Did answers improve?** Compare baseline and reload answers and verifier scores case by case. Optimizer calls and successful tool use alone cannot establish improvement on this panel or generalization to new cases.
5. **Which checkpoint served evaluation?** Match the checkpoint hashes, distinct worker call/container identities and zero evaluation updates. This qualifies inference reload. Resuming optimizer-state training requires a separate test.

See the [evidence validator](../experiment/reader_evidence.py), [measured calculator run](READER_QUALIFICATION_2026-11-19.md#live-gpu-outcomes) and [measured asset run](PHASE3_QUALIFICATION_2026-11-24.md).

## Stop and recover within ownership

If the allocation must stop, use the plan for the operation you intend to cancel. Use `"$prep_plan"` in place of `"$train_plan"` below for preparation. This is an online cancellation command.

```bash
reader stop-owned --plan "$train_plan"
```

Expect `status: cancellation_requested`, the known owned calls and any cancellation errors. The command writes a plan-specific stop marker. A missing call ID leaves the outcome unknown. Check status again, inspect owned provider state and collect to a fresh directory. Preserve ambiguous reservations and partial evidence. Stopping calls leaves the assets and checkpoints on the Volume. Reconcile outstanding work, charges and retention before any new allocation.

### Resume from another terminal

Return to the **same checkout and revision**. Run this block at its root and paste the session directory printed during practice. Use the unchanged real profile and the plan already saved for the operation of interest.

```bash
read -r -p 'Existing reader session directory: ' reader_dir
reader_profile="$reader_dir/profile.json"
prep_plan="$reader_dir/preparation-plan.json"
train_plan="$reader_dir/training-plan.json"
reader() { .venv/bin/python scripts/reader_training.py "$@" --profile "$reader_profile"; }
MODAL_PROFILE="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["modal"]["profile"])' "$reader_profile")"
export MODAL_PROFILE
```

If reading the profile fails, check the directory before continuing. Recheck the Modal context as in step 1. Use `reader status --plan "$prep_plan"` for preparation or `reader status --plan "$train_plan"` once the asset-bound plan exists. Collect partial or terminal evidence to a **new** directory each time. To cancel preparation, use `reader stop-owned --plan "$prep_plan"`. A stop request may leave an unknown call to reconcile.

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
