# Agent Lightning: bounded course experiment

Publication note: this is a retained historical training runbook, not a qualified reader-owned execution recipe. The [public source map](../docs/learning/source-map.md#public-source-representations) supplies attributed code equivalents; private course/account/artifact locators are archival only. Start with the root README's two standard-library commands.

**Current result, November 15, 2026:** the [r03 closeout](../docs/PHASE2_CLOSEOUT_2026-11-15.md) supersedes pre-execution status below. Six matched study runs completed at executed source `5fb6adfd48ee245509f7dff3ab1d849fcf707d86`. This runbook preserves the historical Phase 1 procedure; it is not permission for a new cloud run.

**Historical Phase 2 handoff (before r03):** PPO-MC two-step smoke and native checkpoint reload completed in the personal workspace. The Stanford workspace bootstrap baseline completed 64/64 (26/64 correct); see [observed evidence](../docs/PHASE2_STANFORD_BASELINE_2026-11-14.md). The [PPO-MC smoke contract](../docs/PHASE2_PPO_PROTOCOL.md) remains a historical implementation decision record. See the separate [matched GRPO/PPO study contract](../docs/PHASE2_MATCHED_STUDY_CONTRACT.md). Current `ppo-mc-v1` still supports **only smoke/reload**; the new offline plan generator does not authorize or execute 20/60-step PPO or parallel study jobs. `grpo` remains the legacy default. Historical procedures and evidence below remain unchanged.

This is the Modal runbook for the existing `agent-lightning-lab` repository. Begin with the [deployment README](../README.md). Run all commands from the repository root in the selected Linux client terminal described there. Modal builds and runs the separate training environment in the cloud.

Prepared November 11, 2026; current procedure reconciled November 13, 2026. The original course schedule below is historical context.

**Recommended stack:** one Llama 3.2 3B Instruct checkpoint; the released Calc-X AutoGen + MCP calculator harness; Agent Lightning v1.0.3; VERL 0.8.0; vLLM 0.22.0; one cloud H100 80GB; W&B Models plus optional Weave agent traces. Use the client terminal for editing and submission; model computation runs separately on Modal.

**Historical execution boundary, November 13, 2026:** the image, real Llama/vLLM tool loop, two-step smoke, 20-step main run and native reload have completed on H100 under the frozen protocol. Read the [smoke](../evidence/grpo-smoke-01.md) and [final evidence](../evidence/grpo-main-01.md). Scores are success/accuracy under the frozen answer extractor and reward, not independent semantic grading: 27/64 baseline, 34/64 train-final, 35/64 reload. Six dependency exceptions, sparse signal and scorer/replay limits remain. Hosted training-run W&B/Weave readback, independent tensor identity and a new cold rebuild are unverified. [VERIFICATION.json](VERIFICATION.json) remains the historical preparation snapshot.

The commands below are a procedure for a **new, separately authorized reproduction**, not instructions to rerun the completed experiment for a preferred score. Cloud commands allocate billable resources, including CPU builds/preparation/collection. They were not executed in the documentation closeout. Use fresh `repro-a-` names throughout; replace the prefix everywhere if already occupied. All measured runs used `weave=false`, so the primary commands omit `--weave`.

## 1. What this experiment establishes

The primary engineering question is whether a small conventional model can run, train and be evaluated through the existing harness and proxy architecture. The empirical question is whether a short GRPO run changes held-out success under the frozen answer extractor/reward or observable tool behavior. Improvement is not assumed.

The model controls whether and how to call a calculator through MCP. AutoGen executes the requests and returns observations. Agent Lightning captures the model interactions and routes reward to VERL, which updates model weights. This is online reinforcement learning with a programmatic outcome reward. It is not inverse reinforcement learning, a prompt optimizer, or an LLM-judge experiment.

The task is intentionally small, but the model endpoint, real MCP process, agent loop, reward delivery, training update, checkpoint and trace recording are real components. The connection to Cohesity is the reusable architecture for training tool-using agents and examining evidence. This exercise does not measure vulnerability discovery, backup recovery or production incident-response quality.

### Harness decision

- **AutoGen + MCP for this paper experiment:** matches the released small Agent Lightning example. AutoGen is now archived, so this is a pinned educational/reproduction dependency, not a recommendation to start a new production platform on AutoGen.
- **OpenHands later for repository work:** useful when the agent must edit files, run terminals and execute repository tests. That brings a broader environment than this small experiment needs.
- **LangGraph or Microsoft Agent Framework for a new production-oriented general harness:** reasonable subsequent choices, after the model-training boundary is understood. They are not additional installations in this runbook.
- **NVIDIA Nemotron is a model family, not the harness.** NVIDIA's NeMo Agent Toolkit documentation now marks it deprecated; a new NVIDIA orchestration integration is outside this deadline.

### Model decision

`meta-llama/Llama-3.2-3B-Instruct` is the primary profile because it is a small US-vendor, conventional dense model with a known vLLM JSON tool parser. It requires access approval under the Llama 3.2 license. The prepared model revision is `0cb88a4f764b7a12671c53f0838cd831a0843b95`.

The 1B Llama is smaller, but tool capability can become the bottleneck. Nemotron Mini 4B has different architecture/template details, and the current larger/hybrid Nemotron models add backend and memory questions. A smaller parameter count alone does not guarantee an easier RL integration.

An explicit `reference-qwen` profile exists for a public academic reference run if you choose to use a non-US model for coursework. It is never selected automatically. It is closer to the released example, but our bounds and instrumentation are still adaptations. It should not be treated as a model choice for a workplace with a US-only requirement.

## 2. Fixed scope and workload

| Setting | Value |
|---|---|
| Agent Lightning | v1.0.3, commit `cbc56e41e0eab643aa71d195e92e5bba652fa683` |
| Trainer / inference | VERL 0.8.0 / vLLM 0.22.0 |
| GPU wheel stack | CUDA 13.0, FlashAttention 2.8.3, Python 3.12 |
| Harness | AutoGen 0.7.5, one agent, MCP calculator |
| Agent bounds | Two tool iterations, sequential tool calls, up to 1,024 generated tokens per model request |
| Training | Full-weight FSDP with parameter/optimizer CPU offload; no extra LoRA path |
| Main training pool | 256 distinct examples from released training data |
| Final evaluation | 64 fixed examples, separate from training and smoke evaluation |
| Smoke evaluation | Eight separate examples; not part of the final 64 |
| Training batch | Eight questions, four sampled rollouts each = 32 rollouts per step |
| Actor microbatch | One per GPU |
| Context limit | 4,096 total tokens; additional prompt/response aggregation limits in the saved configuration |
| Smoke | Two training steps, checkpoint each step |
| Main | 20 training steps, checkpoint every five steps, retain two |
| Main rollout count | 640 sampled training attempts before retries/failures, plus evaluation |
| Evaluation sampling | Gateway temperature zero, identical before and after |
| Wall-clock defaults | 45 minutes for smoke/evaluation, 180 minutes for main training |

The historical smoke and main runs demonstrate memory fit on one H100 80GB with the recorded settings and 128 GiB host RAM. This is not a capacity guarantee for another model, context, stack or environment. CPU offload reduces GPU memory demand but can slow updates.

The default reward is the upstream Calc-X numerical/string answer comparison with tolerance 0.01. It does **not** require use of the calculator. Therefore correctness and tool-use frequency must be reported separately; a correct answer without a tool call is not automatically a failure under this reward.

### Source corrections included

The pinned `scripts/setup_verl.sh` installs vLLM **0.22.0**, although the rendered installation guide still says 0.20.2. Use the pinned script, then preserve the installed package manifest.

The stock launcher hardcodes the Qwen gateway name. This launcher supplies the same local model path to both the gateway and trainer. It also explicitly sets the gateway's validation temperature to zero, because that gateway setting overrides incoming sampling settings.

For Llama, the vLLM parser is `llama3_json`; the template is vendored from vLLM 0.22.0. Model preparation installs the same template in the tokenizer and server and fixes its date string so it does not change across days. The separate VERL native `multi_turn.format` setting is left at the upstream default because Agent Lightning owns the rollout loop here.

## 3. Set up the local client and credentials

Follow the [root README local setup](../README.md#2-local-client-setup) and [current validation path](../VALIDATION.md#current-clean-start-path): clone with authorized private-repository access, use Python 3.12 and `uv sync --locked`, run the offline suite, and select your own Modal workspace/environment and W&B entity. The lab supplies no Dev Container configuration. Existing host/container paths and authentication mounts are environment-specific. Keep any container-wide environment separate from the repository `.venv`; never copy `.venv` across architectures.

The remote Secret `agent-lightning-credentials` supplies approved HF read access and W&B write access. Configure it explicitly in the selected environment without overwriting an existing Secret. Local `.env` staging is optional and is not automatically loaded. See [credential handling](../VALIDATION.md#2-credentials-optional-local-env-named-remote-secret). CUDA, Torch, VERL and model weights belong to the separate remote runtime.

Hosted Weave tracing is an explicit opt-in: append `--weave` to a training or evaluation command to send public Calc-X prompts, visible model output, tool calls and observations to your W&B project. Local JSONL recording remains available when `--weave` is omitted. A Weave initialization/export failure also falls back to local recording; W&B training authentication is still required for this prepared cloud path.

### Action order and acceptance evidence

Use the current validation guide for **services → image → GPU checks**, then execute the actions below individually in order. Every cloud action can build/resolve its image and is billable; `modal volume get` only retrieves existing files and submits no worker, although storage/network charges may apply. A process exit or submitted App is insufficient without its acceptance artifacts.

Paths are relative to `cs329z-agent-lightning`. Every experiment action first claims and commits `submissions/NAME/receipt.json`, copies source hashes/build evidence, then commits terminal status on ordinary completion/failure. Run actions serially. A hard stop can leave a `started` receipt or partial files; inspect before advancing. The guard refuses reused names even after failure and is not a distributed lock.

| Action / where | Inputs and artifact paths | Acceptance / first failure evidence |
|---|---|---|
| `prepare` / Modal CPU, 4 cores, 16 GiB, 1 h cap | Pinned model, released Calc-X archive; `models/llama3b/model_manifest.json`, `data-small/data_manifest.json`, `submissions/NAME/prepare-*.log` | Completed receipt; model revision/template and files recorded; 256/64/16/8 disjoint/nested split design and output hashes. On failure inspect staging and prepare/download logs; do not replace historical inputs. |
| `smoke` / Modal H100 | Prepared model; `train_smoke.parquet`, `eval_smoke.parquet`; `runs/NAME/{result,resolved_config}.json`, logs/traces, `checkpoints/global_step_2` | Two iterations, real tool-result continuation, received rewards, finite metrics, some informative groups, refresh/save evidence. Inspect trainer/controller/gateway and preflight logs. |
| `smoke-reload-eval` / Modal H100 | Exact smoke `global_step_2`, 8 smoke-eval cases; new run receipt/logs/traces | Actual restoration/refresh path, 8 completed rewards, no training or new checkpoint; requested-path metadata alone is insufficient. |
| `baseline-eval` / Modal H100 | Original prepared model, `eval_main.parquet`; new run metadata/traces | Exactly 64 held-out IDs once, completed posted/received rewards; validation-only, no actor updates/checkpoints. |
| `main` / Modal H100 | Original model, `train_main.parquet`, same held-out file; step metrics, traces, native saves | 20 completed iterations or explicitly report shorter budget; complete finite metrics/rewards, useful group signal, row/drop accounting; separate initial/final 64-case evaluations. Never label every iteration a nonzero update. |
| `reload-eval` / Modal H100 | Completed main checkpoint, same held-out file; separate run | Exact provenance, restoration/refresh and `val_only`; 64 completed rewards, no actor updates/new checkpoints. Tensor identity is a separate limit. |
| `summarize` / Modal CPU, 4 cores, 16 GiB, 1 h cap | Baseline/post traces plus `data-small/eval_main.jsonl`; `evidence/NAME/summary.json` | Require `comparison.available=true`, 64 pairs, reconcile rewards and counts. On false preserve `reasons`; inspect `submissions/NAME/summarize.log`. |
| `export` / Modal CPU, 4 cores, 16 GiB, 1 h cap | Existing evidence trees; `exports/NAME.tgz`, receipt with size/SHA256 | Completed receipt, later verified downloaded archive/inventory; excludes checkpoint/model tensors and is not model backup. |

`modal_app.py` now requires **`modal run --detach`** and submits one phase with `.spawn()`. It prints `SUBMITTED` with App/FunctionCall IDs and saves a local receipt under `.local/experiment-submissions/`; the worker commits `submissions/NAME/receipt.json` with its call ID and terminal status. A submitted job can continue after client disconnect. Neither submission nor a local receipt proves completion; wait for the durable terminal receipt before advancing. No later phase is queued by this command. The [implementation contract](../docs/INTEGRATION.md) explains resuming inspection from another machine, isolated working-Volume binding and explicit export-evaluation filters. Historical `.remote()` execution records remain unchanged.

## 4. Build once and prepare persistent assets on CPU

The checked-in [Modal adapter](../modal_app.py) defines the image, resources, Secret and Volume. Its CPU image build runs [the pinned training installer](../scripts/install_modal_training.sh) against `nvidia/cuda:13.0.0-devel-ubuntu24.04`, adds Python 3.12 and installs Agent Lightning at `/opt/agent-lightning`. The build sets `TORCH_CUDA_ARCH_LIST=9.0` and FlashAttention's own `FLASH_ATTN_CUDA_ARCHS=90` for the H100 target. A successful image is cached for subsequent runs. The user-observed `image-06` CPU validation and `gpu-01` H100 BF16/FlashAttention preflight both passed on November 12, 2026, with six reviewed metadata exceptions. Neither check loaded the Llama model or verified an Agent Lightning rollout, gradient step, or checkpoint reload.

### Image recipe and rebuild limits

The [heavy installer](../scripts/install_modal_training.sh) requires Linux x86_64, verifies Agent Lightning commit `cbc56e41e0eab643aa71d195e92e5bba652fa683`, and selects `/usr/bin/gcc`/`g++` for `CC`, `CXX` and `CUDAHOSTCXX` before the CUDA/FlashAttention build. This corrects the original inherited clang host-compiler failure. SM90 settings permit CPU compilation without a visible GPU. [modal_app.py](../modal_app.py) caps this layer at 8 CPUs/64 GiB/2 hours; a successful unchanged heavy layer is cacheable.

The [light finalizer](../scripts/finalize_modal_training.sh) installs agent requirements, then PyArrow **25.0.1** with `--no-deps --no-build`, preserving NumPy **1.26.4** and Datasets **5.2.0** compatibility. Its cap is 4 CPUs/16 GiB/30 minutes. It records the scoped package freeze and `uv pip check` output/status. An image build tolerating a recognizable conflict report is not qualification: [the validation policy](../validation/compatibility.py) requires exact Python/architecture/source/package identity and the six reviewed conflicts. New, removed or changed conflicts, version drift, ambiguous metadata, import or data-round-trip failures reject the diagnostic; do not widen exceptions to make a new build pass. See [historical diagnostics](../VALIDATION.md#historical-diagnostics-do-not-replay).

The scoped training-venv Protobuf inventory is 5.29.6; ambient Modal metadata can expose 6.31.1. The validation worker scopes inventory and checks the import, but the experiment's unscoped package dictionaries do not establish what Protobuf it actually imported. That remains unverified. Full dependency closure is not clean, and the remote transitive installation is not wholly locked by local `uv.lock`. Cached-image execution and manifests are historical evidence, not an independently repeated cold rebuild; upstream artifacts/resolution can drift or disappear. ARM64 GPU training is outside this qualified recipe.

The prepared training Python is `/opt/agent-lightning/.venv/bin/python`. Runtime commands use it directly. Do not run `uv sync` inside that environment after adding the pinned GPU extras: an upstream project sync can remove packages installed after its base lockfile. `experiment/bootstrap_gpu.sh` is retained for optional self-managed GPU hosts and is not invoked by Modal.

The launcher preserves the Python executable's virtual-environment symlink. Resolving that symlink to the base Python would bypass the installed training packages. An offline regression test executes a real temporary venv to check this boundary.

Start the CPU preparation action:

```bash
uv run --locked modal run --detach modal_app.py \
  --action prepare --name repro-a-prepare --model-profile llama3b
```

Open the App URL printed by Modal and wait for successful completion before submitting another action. `--detach` lets the submitted remote work continue after your terminal disconnects. It does not submit the later commands in this guide. **Run one action at a time**, and use a new `--name` for each attempted action. `--model-profile` selects the model; Modal reserves `--profile` for its own authentication profile. Existing names are refused to preserve evidence; a failed preparation can be retried as `repro-b-prepare` after fixing its cause.

The named Volume **`cs329z-agent-lightning`** is created as needed and mounted at `/data`:

| Path in the remote container | Purpose |
|---|---|
| `/data/models/llama3b` | Completed pinned BF16/Safetensors checkpoint and prepared tokenizer/template, not GGUF |
| `/data/cache/huggingface` | Hugging Face cache, reused between containers |
| `/data/data-source` | Released Calc-X archive |
| `/data/data-small` | Frozen training, smoke and final-evaluation splits and manifest |
| `/data/runs/NAME` | Logs, traces, configuration, results and native checkpoints |
| `/data/evidence/NAME` | Summaries and verification evidence |
| `/data/exports/NAME.tgz` | Small evidence archive for download |
| `/data/submissions/NAME` | Submission logs and receipts |
| `/data/staging/NAME` | Partial model downloads and extracted source Parquet files, excluded from evidence export |

The action records the model revision and template hashes, downloads the official dataset and freezes the deterministic seed-329 splits. Worked solutions are excluded from agent inputs. Case IDs and normalized questions cannot overlap between training and either evaluation set. The eight smoke-evaluation cases are separate from the final 64; sixteen smoke-training cases come from the 256 training cases. Do not change the final split or reward after examining final results.

Completed model assets are reused after their manifest checks. An incomplete model download is preserved under `/data/staging/NAME`; retry preparation with a new name. A persistent HF cache does not guarantee that every interrupted transfer resumes efficiently.

If Google Drive blocks the dataset download, use the [official dataset link](https://drive.google.com/file/d/1FQMyKLLd6hP9dw9rfZn1EZOWNvKaDsqw/view) in your browser. Save it locally as `.local/calc-x-data.zip`. For a new authorized reproduction, upload it only if `data-source/calc-x-data.zip` is absent and the failed action has ended:

```bash
uv run --locked modal volume put cs329z-agent-lightning \
  .local/calc-x-data.zip data-source/calc-x-data.zip
```

Retry `prepare` with a new name. It reuses a valid archive at that path and extracts the unique `train.parquet` and `test.parquet` files into the new staging directory. If a pre-existing file blocks the upload or a source identity differs from the intended manifest, stop and preserve it. Diagnose the mismatch and plan a separately authorized repair or isolated reproduction; do not overwrite historical source data.

### Frozen input identity

[prepare_model.py](prepare_model.py) pins Llama revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`, records the original/prepared tokenizer config hashes and fixes the template date to `11 Nov 2026`. The vendored template's upstream source is in [PROVENANCE.json](templates/PROVENANCE.json). Historical effective template SHA256 is `a718674808440b537a553c15e0557b5b45d6d76e49ad9ddbc1a9e92447843e2c`; prepared `tokenizer_config.json` is `7fe399d0739ac28d562a04f0d294093d5d38ad3f00e6ea0e624f932ccf397116`. Reuse verifies these metadata files and weight-file presence; it is not tensor hashing.

[prepare_data.py](prepare_data.py) records source `train.parquet`/`test.parquet` hashes, selected case IDs, output hashes, seed and exclusion rules in `data_manifest.json`. Seed 329 alone is insufficient if upstream contents change. Compare the source and output identities with the retained historical manifest before claiming the same dataset. [The result report](../evidence/grpo-main-01.md#sources-and-frozen-inputs) records the main/eval output hashes and counts. Worked solutions are excluded, and held-out IDs/normalized questions cannot overlap training.

The released Drive ZIP is not an immutable availability guarantee; source Parquets and full frozen data are not checked into Git. Without the matching archive or retained assets, the historical data cannot be reconstructed from the published CSV alone. A newly downloadable source is not automatically the old source. Preserve mismatches and report this gap instead of silently replacing/regenerating historical data. The existing preparation action reuses completed model/data assets after checks; a fresh action name does not create a fresh dataset.

The [Stanford cold-build incident and fix](../docs/STANFORD_COLD_BUILD.md) records
failed image `im-fc2RdjhSAGjfhNX3zuDS5W`, the isolated build entrypoints, cache
implications, offline regression evidence and the independent restored-input
staging blocker. Cold-build qualification remains separately approved work.

### GPU resources, preflight and cost

Legacy GRPO training/evaluation actions request **one `H100!`**, **16 physical CPU cores** and **128 GiB RAM**. The bounded `ppo-mc-v1` smoke and smoke-reload-eval actions override only host memory to **192 GiB** (`memory=196608` MiB). This is a reservation, not a hard memory limit or demonstrated PPO capacity. The installed Modal client accepts this resource shape; server admission and actual PPO memory fit remain unverified. The exact GPU selector avoids silently accepting a different GPU tier.

The historical `actor/perf/cpu_memory_used_gb` metric comes from `psutil.virtual_memory().used / 2**30`. It measures host-visible used memory, not actor RSS or offloaded-state bytes; do not add an estimated critic allocation to it. An authorized PPO smoke must retain cgroup current/peak/OOM counters and membership scope, process-local CUDA allocated/reserved peaks, optimizer-state bytes, checkpoint sizes and stage timing. Verify whole-container coverage and obtain separate evidence for whole-device/process-tree peaks; the existing CUDA counters exclude other processes and driver allocations. Keep measured values separate from estimates, including during initialization and checkpoint serialization. No smoke or reload is authorized by this resource patch.

Each GPU action saves `nvidia-smi` output and runs preflight before the experiment. Preflight requires one visible CUDA GPU, executes a BF16 matrix operation and FlashAttention kernel, checks the pinned VERL/vLLM versions, and records GPU memory, architecture and package versions. It does not establish sufficient host/shared memory or model-training compatibility. An optional standalone check is:

```bash
uv run --locked modal run --detach modal_app.py \
  --action preflight --name repro-a-preflight
```

No GPU is requested by `prepare`, `summarize` or `export`; dependency compilation is also CPU-only. A clean dependency resolver or image build does not establish GPU compatibility. Preserve preflight errors before changing any version.

At [Modal's published prices](https://modal.com/pricing), checked November 11, 2026, one H100 is about $3.95/hour. The requested 16 CPU cores and 128 GiB add about $1.78/hour, for **approximately $5.73/hour** of active runtime resources. Image-build CPU time, storage, other usage and applicable credits are separate. This is resource-price arithmetic, not a training-duration estimate. Check the live dashboard and pricing for your account. If setting a workspace budget, remember that it also affects your other Modal projects.

## 5. Pass the two-step and reload checks

Run the first GPU phase under supervision. Keep `--weave` omitted to match the measured protocol; hosted agent views are a separate opt-in.

```bash
uv run --locked modal run --detach modal_app.py \
  --action smoke --name repro-a-smoke --model-profile llama3b \
  --wandb-entity YOUR_WANDB_TEAM
```

Use the App's logs while it runs, or follow them from another client terminal with the actual App ID printed on submission:

```bash
uv run --locked modal app logs YOUR_APP_ID --follow
```

The adapter streams trainer/controller/gateway logs to Modal and retains the original files on the Volume. W&B Models supplies training metrics, and Weave supplies hosted traces when enabled. A live GPU process alone is not a successful smoke test. Confirm:

1. The model loads and produces correctly parsed tool requests.
2. At least one real MCP calculator call completes and its result reaches a later model request.
3. Reward reaches Agent Lightning; gateway/controller errors are not mistaken for zero-score tasks.
4. Two training steps finish with finite metrics and the actor-to-rollout refresh succeeds.
5. Native checkpoints are written. Some gradient/update signal and mixed group outcomes are needed before expecting learning; all-equal GRPO groups provide no within-group discrimination.
6. Local trace records and, when enabled, the Weave view agree about the observed execution.

After the action ends, inspect its committed result without downloading weights:

```bash
uv run --locked modal volume get cs329z-agent-lightning \
  runs/repro-a-smoke/result.json -
```

Then verify checkpoint reload using only the eight smoke-evaluation cases:

```bash
uv run --locked modal run --detach modal_app.py \
  --action smoke-reload-eval --name repro-a-smoke-reload --model-profile llama3b \
  --checkpoint runs/repro-a-smoke/checkpoints/global_step_2 \
  --wandb-entity YOUR_WANDB_TEAM
```

This restores the two-step checkpoint, refreshes the inference backend and evaluates without another actor update. **Both phases must pass before the main experiment.** A new environment must establish these gates for itself; preserve errors and use a new name for a revised attempt.

The inner launcher caps smoke/evaluation at 45 minutes and main training at 180 minutes. Each GPU function has a four-hour outer timeout. Application-exception retries are disabled. Modal may still replace a container after infrastructure failures; persisted submission records and run-directory guards are intended to prevent silently repeating an already claimed experiment. They are not a distributed lock across simultaneous Apps, which is another reason to run serially.

## 6. Run the controlled before/after experiment

Execute these three commands **separately**, waiting for each action to finish successfully before continuing.

### Baseline: the original model on the final 64 cases

```bash
uv run --locked modal run --detach modal_app.py \
  --action baseline-eval --name repro-a-base --model-profile llama3b \
  --wandb-entity YOUR_WANDB_TEAM
```

This uses the original checkpoint in validation-only mode with no actor updates. The separate phase gives traces an unambiguous base-policy label.

### Main: 20 training steps from the original model

```bash
uv run --locked modal run --detach modal_app.py \
  --action main --name repro-a-train --model-profile llama3b \
  --wandb-entity YOUR_WANDB_TEAM
```

The main run starts fresh from the base checkpoint, not the smoke-trained checkpoint. It holds the batch, group size, learning rate and harness fixed; saves at steps 5, 10, 15 and 20; and retains the two newest checkpoints. There are 640 nominal sampled training attempts plus the main run's initial/final validation. Retries and failures can change actual executed work.

Read the paper while this runs after the first updates and traces look healthy. W&B Models gives the aggregate training view; use a few traces to diagnose behavior. After warm-up, use the median of three steady step durations to estimate:

`remaining time ≈ remaining steps × median step duration + final evaluation + checkpoint/export`

For example, 20 steps at two minutes each is 40 minutes of training steps; at six minutes each it is two hours. These are arithmetic examples, not measured performance. The three-hour main limit can stop a slow run before step 20. Report its actual completed budget.

### Reload: the saved checkpoint on the same final 64 cases

For a completed 20-step run:

```bash
uv run --locked modal run --detach modal_app.py \
  --action reload-eval --name repro-a-post --model-profile llama3b \
  --checkpoint runs/repro-a-train/checkpoints/global_step_20 \
  --wandb-entity YOUR_WANDB_TEAM
```

If main training stopped earlier, retrieve `runs/repro-a-train/result.json` using the same `modal volume get` pattern as above. Its `latest_checkpoint` is based on `runs/repro-a-train/checkpoints/latest_checkpointed_iteration.txt`, not just the existence of a directory. Use that actual checkpoint, mapping it to the logical Volume-relative path when passing `--checkpoint` (runtime absolute paths can also use Modal's internal mount prefix). The Modal argument accepts a path relative to the Volume, such as `runs/repro-a-train/checkpoints/global_step_15`.

Reload checks the source-run metadata, model revision, template and data hashes. It loads the native actor checkpoint, refreshes the inference backend and evaluates with no further updates in a new run directory. A successful training step alone does not prove that checkpoint reload works. If reload fails, preserve and explain that limitation.

## 7. Compare metrics and inspect trajectories

After baseline and post-training evaluation complete:

```bash
uv run --locked modal run --detach modal_app.py \
  --action summarize --name repro-a-paired \
  --baseline-name repro-a-base --post-name repro-a-post

uv run --locked modal volume get cs329z-agent-lightning \
  evidence/repro-a-paired/summary.json -
```

The summary separates completed, failed and incomplete attempts. Missing rewards are not converted to zero. It computes paired wins/losses only when every expected held-out case appears exactly once with a completed, posted binary reward in both runs. Otherwise it explains why comparison is unavailable.

Report correctness counts, completion rate, tool calls/errors, reported tokens and measured durations. With 64 tasks and one short run, modest changes are descriptive and may not establish a general improvement. If correctness stays flat while tool use changes, inspect traces before assigning a cause.

### Local analysis alternative

Cloud pairing was chosen because the traces already lived on the Volume, not because summarization requires GPU libraries. [summarize.py](summarize.py) uses only Python's standard library and can run on a compatible Python (3.12 is the documented client) on ARM64 or x86-64, without CUDA or the training environment. With authorized existing files available locally, use:

```bash
python3 experiment/summarize.py \
  --run-dir .local/repro-a/runs/repro-a-post/traces \
  --phase reload-eval --route-mode val \
  --baseline-dir .local/repro-a/runs/repro-a-base/traces \
  --baseline-phase baseline-eval \
  --heldout-file .local/repro-a/data-small/eval_main.jsonl \
  --output-dir .local/repro-a-paired-local
```

Required inputs are the original `trajectory-*.jsonl` files for each run and the matching frozen `eval_main.jsonl` (with case IDs/questions); the sanitized CSV is not a substitute for those schemas. Preserve each trace file's basename and avoid duplicates. The output directory must be new. Omitting `--output-dir` prints JSON; other flags above are the existing CLI, not new options. Require `comparison.available=true` and inspect invalid/missing records and reasons. No training-step selector exists: do not combine train-initial and train-final traces into a single 128-case evaluation. Their historical timing partition is already documented in the report. This closeout tested the published CSV command only, without rerunning raw-trace analysis.

For enabled Weave traces, open the project's Agents view and find agent `cs329z-calc-x`. Compare the same case ID under `baseline-eval` and `reload-eval`: the requested tool and arguments, MCP response/error, whether the observation entered the next model request, the final answer and whether the reward POST succeeded.

These are visible messages and actions, recorded without adding model calls. The wrapper omits provider `thought` and token-logprob fields from human-readable traces and does not reconstruct hidden reasoning. Agent Lightning remains responsible for token/logprob data needed for training. Returned text is an output to inspect, not a guaranteed causal account of the model's decision.

The exact internal per-rollout model version in main-training traces is explicitly unknown. Use the separate baseline and reloaded-checkpoint phases for clear before/after video examples. Local JSONL remains usable if hosted export fails. A returned SDK `finish()` is not proof of hosted ingestion; check the actual Weave browser view and disclose missing traces.

## 8. Export evidence and finish cloud work

Create an archive on CPU, then download it into the ignored local output directory:

```bash
uv run --locked modal run --detach modal_app.py \
  --action export --name repro-a-export

mkdir -p .local/downloads
uv run --locked modal volume get cs329z-agent-lightning \
  exports/repro-a-export.tgz .local/downloads/repro-a-export.tgz
```

The exporter includes the existing `runs`, `evidence`, `data-small` and prior `submissions` trees plus model manifests; it is not restricted to one run or to sanitized publication data. It excludes checkpoint directories, model tensors, caches, staging downloads and symlinks. Review logs/traces and full frozen data before sharing. The export action's own in-progress submission is excluded; retain its terminal receipt separately.

This is **not a model backup**. A later explicitly authorized export/download/verification should compare the downloaded archive's size/SHA256 to the receipt, inspect its inventory and retain a durable independent copy with the source identity. No such verified archive or tensor backup is established by this closeout, and export was not invoked. Native resume checkpoints need a separate archive scope; deployment requires validated conversion, tokenizer/template checks and separately authorized publication. Historical `train-01` retains actor checkpoints at steps 15/20; step-5/10 directories contain only `data.pt`. Never treat directory existence as proof of a full checkpoint.

A Modal function exits when its action completes; there is no rented Pod to stop. Confirm in the dashboard that the App has finished. To cancel an active action, use the exact App ID printed by its launch:

```bash
uv run --locked modal app stop YOUR_APP_ID
```

The adapter commits the Volume on ordinary completion/failure, and Modal also provides background commits. Hard termination can bypass cleanup, so the last log lines or checkpoint writes may not persist. Check committed evidence after cancellation; a partially written checkpoint is not a valid completed save. Volume storage persists after compute ends and may remain billable. Retain it until the evidence and any desired checkpoints are safely copied.

## 9. Historical course deadline and stop decisions

The original November 2026 planning record below is retained for context, not a new reproduction deadline. The [public course handout](https://web.stanford.edu/class/cs329z/assignments/paper_video.pdf) lists **Friday, November 13, 2026, 11:59 p.m. Pacific**, and a **10–15 minute** video. Treat Saturday as additional time only if your permitted extension covers it.

| Work block | Target outcome | Time boundary |
|---|---|---|
| Wednesday | Read the v1.0 report, confirm model access and existing Modal/W&B accounts, run the CPU accounting exercise | No GPU needed for these first checks |
| Thursday, first block | Image build, two-step smoke, reload and useful traces | Roughly 2–3 focused hours; cap one compatibility-debugging cycle |
| Thursday, next block | Baseline, bounded main run, checkpoint reload and fixed evaluation | Use measured step time; main process capped at three hours |
| Friday morning | Freeze evidence and write your own analysis | No new harness/model port |
| Friday afternoon/evening | Your own slides/script, practice, record, upload and verify playback | Reserve 4–6 hours |
| Saturday, if allowed | Recording/upload buffer or a narrowly scoped rerun | Do not plan the first complete run for Saturday |

These are planning allocations, not performance guarantees. If the GPU port still fails after the first focused debugging block, preserve the precise failure stage and use the CPU exercise with a clear account of the attempted reproduction. The course accepts small, honest extensions and explained failed reproductions.

## 10. CPU accounting extension, available immediately

From the repository root, with the same local uv environment:

```bash
uv run --locked python experiment/cpu_accounting_demo.py \
  --output-dir .local/cpu-accounting-01
```

This keeps the same predefined token-loss values for two synthetic rollouts, then changes how one rollout is split into rows. Uniform row averaging changes from 5 to 3.5; equal-rollout token averaging stays at 5. It illustrates Section 2.3, Equations 15 and 16 of the [v1.0 report](https://arxiv.org/html/2608.17528v1#S2.SS3).

The output is synthetic. It is an arithmetic illustration, not execution of the released loss implementation, a trained model or empirical evidence of improved stability. Inspect the code, explain in your own words why the weights change, and connect that limitation to the real run or failed run. Do not present the supplied numbers as measured RL outcomes.

## 11. Troubleshooting boundaries

| Symptom | First check | Decision |
|---|---|---|
| uv warns about `/opt/venv` | A container-wide `VIRTUAL_ENV` | Use repository `.venv` through `uv run`; do not add `--active` |
| Modal authentication/Secret error | Selected workspace/environment, existing profile, exact Secret keys | Fix account configuration before resubmitting |
| HF 401/403 | Token account and gated model access | Resolve during CPU preparation |
| Google Drive download failure | Official archive link and Volume path | Upload the archive; retry preparation with a new name |
| CPU image-build failure | First compiler/CUDA error in Modal build logs | Preserve the error; one focused repair attempt |
| CUDA/kernel error | GPU preflight, actual driver and package versions | Fix the identified mismatch; do not upgrade every package |
| Model text appears but tools never execute | `llama3_json`, prepared template and recorded tool schema/request | Inference alone is not a successful harness integration |
| OOM | Context, KV budget, concurrency and CPU offload | Preserve configuration; use a separately named revised attempt |
| Every GRPO group has equal rewards | Task difficulty, parser, initial policy and verifier | More steps or larger learning rate do not create reward discrimination |
| Weave empty | `--weave`, SDK/account/project, export warning and Agents view | Use local JSONL and verify hosted delivery manually |
| Summary refuses comparison | Missing/duplicate cases, reward posts and split identity | Report partial outcomes without inventing scores |
| Main timeout or interrupted App | Completed-checkpoint receipt and committed result/logs | Evaluate only a completed checkpoint; report actual steps |
| Name already exists | Prior submission/run records | Inspect its outcome and choose a fresh name; do not delete evidence to retry |

## 12. Contents and provenance

- [`../modal_app.py`](../modal_app.py): Modal image, resources, serial action entrypoint, persistent Volume and evidence export.
- [`../scripts/install_modal_training.sh`](../scripts/install_modal_training.sh): pinned CPU image-build installation; `check_gpu.py` performs runtime checks.
- `bootstrap_gpu.sh`: optional legacy self-managed GPU bootstrap, outside the Modal path.
- `prepare_model.py`, `prepare_data.py`: pinned model/template and frozen dataset preparation.
- `launch.py`: bounded processes, baseline/train/reload phases, manifests and logs.
- `observed_calc_agent.py`: real AutoGen/MCP operation recording and optional Weave spans.
- `summarize.py`: outcome/trace accounting with explicit missingness.
- `cpu_accounting_demo.py`, `example_outputs/cpu-accounting/`: synthetic arithmetic exercise and its recorded output.
- `templates/`, `LICENSE-AgentLightning.txt`: versioned upstream template, licenses and provenance.
- `VERIFICATION.json`: preparation checks and their limits.

The code is an implementation aid. The assignment permits AI help with understanding/debugging and requires your own slides, script, analysis and narration. Keep the code revision, chosen configuration, dataset manifest and measured outputs with your course submission. Keep credentials and large weights outside Git.

A useful proposed course-playbook decision is: verify the complete inference → tool → reward → update → checkpoint → reload path before scaling model size or harness complexity. Keep the selected Linux terminal as the editing/client environment, and describe the independent Modal runtime with code in this repository.

### Primary sources

- [Course paper-video requirements](https://web.stanford.edu/class/cs329z/assignments/paper_video.pdf)
- [Selected v1.0 report, August 2026](https://arxiv.org/abs/2608.17528)
- [Official Calc-X guide](https://microsoft.github.io/agent-lightning/stable/50-example-calc-x/)
- [Pinned training installer](https://github.com/microsoft/agent-lightning/blob/v1.0.3/scripts/setup_verl.sh)
- [Pinned Calc-X trainer](https://github.com/microsoft/agent-lightning/blob/v1.0.3/examples/calc_x/train_calc_agent.py)
- [Pinned gateway configuration](https://github.com/microsoft/agent-lightning/blob/v1.0.3/agentlightning/config/server.yaml)
- [Llama model card](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [vLLM 0.22 tool calling](https://docs.vllm.ai/en/v0.22.0/features/tool_calling/)
- [VERL backend compatibility](https://verl.readthedocs.io/en/latest/advance/fsdp_extension.html)
- [AutoGen status](https://github.com/microsoft/autogen)
- [OpenHands SDK](https://github.com/OpenHands/software-agent-sdk)
- [NVIDIA toolkit status](https://docs.nvidia.com/nemo/agent-toolkit/latest/index.html)
- [Modal images](https://modal.com/docs/guide/images), [GPU selection](https://modal.com/docs/guide/gpu), [CPU/memory resources](https://modal.com/docs/guide/resources) and [CUDA](https://modal.com/docs/guide/cuda)
- [Modal Secrets](https://modal.com/docs/guide/secrets), [Volumes](https://modal.com/docs/guide/volumes) and [Volume CLI](https://modal.com/docs/cli/latest/volume)
- [Modal profile CLI](https://modal.com/docs/cli/latest/profile), [run CLI](https://modal.com/docs/cli/latest/run) and [App lifecycle](https://modal.com/docs/guide/apps)
- [Modal retries](https://modal.com/docs/guide/retries), [preemption](https://modal.com/docs/guide/preemption), [budgets](https://modal.com/docs/guide/budgets) and [pricing](https://modal.com/pricing)
- [uv project synchronization](https://docs.astral.sh/uv/concepts/projects/sync/) and [project environments](https://docs.astral.sh/uv/concepts/projects/config/)
- [Weave agent tracing](https://docs.coreweave.com/products/wandb/weave/guides/tracking/trace-agents)
