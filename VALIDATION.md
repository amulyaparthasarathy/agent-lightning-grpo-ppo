# Validate Modal, dependencies, W&B and laptop disconnection

This is the setup gate for the existing Agent Lightning experiment. It does not change the model, reward, training algorithm, or frozen evaluation set. Use public/synthetic data only.

**Historical experiment status, November 13, 2026:** CPU image/data validation and H100 kernel preflight passed in the recorded November 12 milestones, followed by [real Llama smoke/reload](evidence/grpo-smoke-01.md) and [20-step/final paired evaluation](evidence/grpo-main-01.md). These are narrowly qualified historical executions, with six dependency exceptions. [validation/VERIFICATION.json](validation/VERIFICATION.json) and [VERIFICATION.json](VERIFICATION.json) are preserved chronological snapshots, not a current checklist. This documentation closeout launches no cloud work.

## Current clean-start path

Start at the [root README](README.md#2-local-client-setup) for clone/access, Python 3.12/uv and your own accounts. Use the current reviewed source; **do not pull `feat/detached-validation` or replay image-01 through image-06**. Those commands are retained only inside [historical diagnostics](#historical-diagnostics-do-not-replay).

For a new environment, the order is local offline suite → account/Secret selection → `services` → `image` → `gpu` → [experiment preparation and smoke](experiment/README.md#action-order-and-acceptance-evidence). The commands here use fresh `repro-a-` names and `YOUR_WANDB_TEAM`; replace the prefix consistently if occupied and substitute your authorized entity. Each cloud command is a separate billable action, not a read-only check. Review each terminal receipt before the next action; never run this page as a batch script. Existing qualified artifacts can be inspected instead of rerunning the author's jobs.

## 1. Local client and separate remote packages

Use a suitable Linux/WSL terminal or an existing compatible Linux container. Run commands from your reader-owned checkout root. The lab has no `.devcontainer/` definition and needs no container rebuild; see the [client setup](README.md#2-local-client-setup).

| Location | Required software | How it gets installed |
|---|---|---|
| Linux client terminal | Python 3.12, uv, Git and Unix shell | Your existing Linux/WSL environment or compatible container |
| Lab's local `.venv` | Locked Modal client and Hugging Face client | `uv sync --locked` |
| Small remote services-test image | Python 3.12, W&B 0.24.2 and huggingface-hub 0.36.2 | Defined in `validation_services.py`; built by Modal |
| Remote training image | Existing Agent Lightning/VERL/vLLM, CUDA, Torch, FlashAttention, AutoGen/MCP, W&B and Weave | Existing `modal_app.py` and `scripts/install_modal_training.sh` |

No local `pip install wandb weave torch`, CUDA installation, local inference service or Docker socket is needed. Imports of remote-only packages happen inside the remote validation program. The local `.venv` and the remote training `.venv` are different environments.

In a clean, updated checkout:

```bash
uv sync --locked --python 3.12
uv run --locked modal --version
uv run --locked modal profile current
uv run --locked python -m unittest discover -s tests -v
```

`uv sync --locked` refuses to silently update an inconsistent lockfile. `uv run --locked` selects the repository `.venv`; do not use `--active` to target a container-wide environment or copy `.venv` between architectures. Reuse your existing Modal profile. Only when authentication is missing, run `uv run --locked modal token new`. `profile current` reports a local selection, not proof of valid remote credentials.

Keep the same Modal workspace and environment for Secrets, Apps, logs and Volumes. For a non-default environment, set `MODAL_ENVIRONMENT` to its actual name before all commands below; otherwise reuse your selected default.

## 2. Credentials: optional local .env, named remote Secret

A local `.env` is optional staging convenience, not a requirement of training. Creating the Secret directly in the Modal dashboard is equally valid and avoids a local copy. Neither the tests nor training automatically load a local `.env`.

To use the template, create `.env` inside your own checkout. Run the following from the repository root; no fixed host or container path is required.

```bash
# Do not overwrite an existing .env.
if [ ! -e .env ]; then
  (umask 077; cp .env.example .env)
fi
chmod 600 .env
```

Open `.env` in your editor and replace the two placeholders. Do not put keys in shell commands, screenshots, chat, or Git. Do not run `source .env`; this file is data, not shell code. Use a Hugging Face **read** token whose account has accepted the Llama 3.2 license and has access to `meta-llama/Llama-3.2-3B-Instruct`. The W&B API key must be able to write to your chosen entity/team.

```bash
uv run --locked python scripts/check_validation_env.py .env
```

This local-only check refuses placeholders, duplicate/unexpected keys, tracked/unignored files, symlinks and group/world-readable permissions. It prints no values and does not validate the tokens with either service.

If `agent-lightning-credentials` does not already exist in your selected Modal environment:

```bash
uv run --locked --with python-dotenv modal secret create agent-lightning-credentials --from-dotenv .env
```

Modal's `--from-dotenv` CLI option needs the optional local `python-dotenv` library. `uv run --with python-dotenv` makes it available for this one command without changing `pyproject.toml`, `uv.lock`, or your client environment. The regular Modal SDK is already in the locked project environment.

This sends the two values to Modal's Secret service, not into an image. The runtime uses `Secret.from_name(...)`. If the Secret already exists, reuse it or inspect/edit it deliberately in the dashboard. Do not blindly use `--force`, which replaces the existing Secret. Uploading the Secret is an explicit account change; none of these instructions has been run on your behalf.

No `wandb login`, Hugging Face write token, OpenAI key or GitHub key is needed in the training container. After the Secret is configured, later job submissions do not need `.env`; keep any retained local copy private and protected. `.gitignore` already ignores `.env` and permits only the placeholder `.env.example`.

The checker and `.env.example` accept exactly `HF_TOKEN` and `WANDB_API_KEY`. Keep namespace/repository configuration separate. Local `.env` is not automatically loaded by HF clients or the launcher. The [current handoff](docs/PORTABLE_HANDOFF.md) distinguishes training read access from separately scoped publishing access; do not add publisher keys to this two-key file or replace the training Secret.

## 3. Three separate checks

Run **one check at a time**, with a new name on every attempt. Names are shared across the validation Volume. Existing names are refused, even after failure. The guard is not a distributed lock across concurrent Apps.

| Check | What it does | Resource request / limit | What it does not prove |
|---|---|---|---|
| `services` | W&B online run, HF authentication and pinned gated `config.json` download, synthetic heartbeats, W&B server readback | 1 CPU, 2 GiB; 30-minute function limit; default 180-second heartbeat period | Training-image dependencies, GPU, model weights or RL |
| `image` | Reuses/builds the real training image; audits interpreter, pinned versions, `uv pip check`, CPU-safe imports; repeats HF/W&B checks | 4 CPU, 16 GiB; 30-minute function limit, plus separate image build | CUDA kernels, vLLM GPU initialization, actual training |
| `gpu` | Repeats the image checks, imports vLLM/FlashAttention, runs the existing CUDA BF16 matrix multiplication and FlashAttention preflight; repeats HF/W&B checks | 1 H100, 4 CPU, 16 GiB; 20-minute function limit | Loading the Llama weights, an optimizer step, tool loop, checkpoint reload or Weave delivery |

All are **synthetic infrastructure checks**, not learning experiments. They never train or upload model weights. HF downloads only the pinned model configuration. The small services image intentionally uses familiar CS224R client versions and is not evidence of training-stack compatibility.

`image` and `gpu` import the existing training image definition rather than maintaining another CUDA recipe. Both add only validation code. Modal caches unchanged image layers; edits to the runtime recipe can cause rebuilds. A first build is unavoidable when no matching cached image exists. Do not force-rebuild or upgrade versions just because a test fails. Retain the error and fix its specific cause.

CPU work and image builds can be billable; `gpu` additionally allocates an H100. The validation Volume persists and may incur storage charges. Limits are caps, not estimates of completion time. No cloud resource is launched by this documentation closeout.

## 4. Test W&B and a real laptop disconnect first

The isolated W&B project is **`agent-lightning-validation`**, separate from training project `cs329z-agent-lightning`. The default entity is `amulyap-stanford-university`; override `--wandb-entity` if that is not your actual writable slug.

The first `wandb.init()` creates the test project when your permissions allow it. To control privacy explicitly, create that named project in W&B with **Private** visibility before running the test, or verify its visibility immediately afterward. The code does not claim to enforce workspace privacy settings.

```bash
bash scripts/validate_modal.sh services repro-a-services \
  --wandb-entity YOUR_WANDB_TEAM --hold-seconds 600
```

The wrapper always uses `uv run --locked modal run --detach`. The local entrypoint submits with `.spawn()` and then exits. A `SUBMITTED` record contains the App ID, FunctionCall ID and expected result path. It is also saved under `.local/validation-submissions/`.

**Do not shut down during image creation/upload or before `SUBMITTED` appears.** For the disconnect experiment, also confirm the remote log shows `REMOTE_STARTED` and at least one `HEARTBEAT`. Record the wall-clock time, close the terminal or shut down the laptop, and reconnect after the heartbeat period. No later part of this test depends on your laptop: heartbeat generation, HF access, W&B writes, readback and final Volume persistence all happen remotely.

In W&B, the run name is `repro-a-services`, tagged `synthetic` and `validation`. Plot `validation/heartbeat` or `validation/elapsed_seconds`. No synthetic metric is labelled accuracy, loss or reward.

After reconnecting:

```bash
uv run --locked modal volume get agent-lightning-validation \
  runs/repro-a-services/result.json -
uv run --locked modal volume get agent-lightning-validation \
  runs/repro-a-services/launcher.json -
uv run --locked modal volume get agent-lightning-validation \
  runs/repro-a-services/heartbeat.jsonl -
```

Accept the services gate only when `result.json` says `status: passed`, the HF checks are true, the W&B readback checks are true, and `launcher.json` says `status: completed`. For the **disconnect** check, additionally compare heartbeat timestamps with your recorded laptop-off interval and confirm W&B contains later heartbeats.

`disconnect_observed` deliberately stays `false`: remote code cannot independently observe your laptop being switched off. Retain a small manual note identifying the run and observed disconnect/reconnect times. A passing test while the laptop stayed on is not disconnect evidence.

W&B verification reads the finished run and its final heartbeat/summary back from the server after `finish()`. It does not verify every historical point or Weave traces. If SDK logging succeeds but server readback fails, the validation reports failure rather than inventing success. The W&B run may say finished while `result.json` reports a readback failure; the latter is the gate result.

## 5. Watching logs is optional; cancellation is different

Use the App ID from the submission receipt:

```bash
uv run --locked modal app logs YOUR_APP_ID --follow
```

Closing this viewer, pressing Ctrl-C in this separate log command, closing the browser or disconnecting the laptop does not intentionally stop an already-submitted detached job. The Modal dashboard also shows its logs. Build logs are separate; a failed image build can occur before a FunctionCall or result file exists.

To **cancel compute**, use the actual App ID deliberately:

```bash
uv run --locked modal app stop YOUR_APP_ID
```

An ordinary attached `modal run` can stop when its client disconnects. Detached execution is not unlimited fault tolerance: timeouts, explicit stops, resource failures and quota limits can still end a job. The validation helpers disable application retries. Hard termination can leave only partial artifacts; a file with `started` is not a passing result.

## 6. Validate the actual training environment, then the GPU

After the services gate, submit **only the CPU image check** and wait for its terminal evidence:

```bash
bash scripts/validate_modal.sh image repro-a-image \
  --wandb-entity YOUR_WANDB_TEAM
```

Retrieve existing evidence (these commands do not submit workers):

```bash
uv run --locked modal volume get agent-lightning-validation runs/repro-a-image/result.json -
uv run --locked modal volume get agent-lightning-validation runs/repro-a-image/launcher.json -
uv run --locked modal volume get agent-lightning-validation runs/repro-a-image/dependency-conflicts.json -
uv run --locked modal volume get agent-lightning-validation runs/repro-a-image/data-smoke.json -
```

Require completed launcher/exit 0, successful HF/W&B readback, CPU imports and three-row data interoperability. The known recipe reports `diagnostic_passed_with_reviewed_exceptions`, `dependency_closure_consistent=false` and exactly six accepted conflicts. That is a diagnostic pass, not clean dependency closure. `packages.json`, `package-inventory-details.json`, `protobuf-import.json` and `imports.json` supply scoped identity/import evidence; any changed version, requirement, architecture, source, ambiguous inventory or failed import rejects the gate. Inspect `dependency-progress.json`, `import-failure.json`, `data-smoke-failure.json` and `runtime.log` when present. Do not weaken the policy to advance.

Only after reviewing that pass, submit the separate H100 kernel check:

```bash
bash scripts/validate_modal.sh gpu repro-a-gpu \
  --wandb-entity YOUR_WANDB_TEAM
```

```bash
uv run --locked modal volume get agent-lightning-validation runs/repro-a-gpu/result.json -
uv run --locked modal volume get agent-lightning-validation runs/repro-a-gpu/launcher.json -
uv run --locked modal volume get agent-lightning-validation runs/repro-a-gpu/gpu_preflight.json -
```

Require the same CPU/service evidence plus finite BF16/FlashAttention kernel results and the expected H100. A kernel pass does not itself prove Llama memory fit, tool calls, gradients or reload. Those require the subsequent [experiment gates](experiment/README.md#action-order-and-acceptance-evidence). A new environment must establish them even though the historical experiment passed.

The [two-stage build explanation](experiment/README.md#image-recipe-and-rebuild-limits) links the authoritative installers and exact-conflict policy. It explains GCC/H100 settings, PyArrow 25.0.1 with NumPy 1.26.4/Datasets 5.2.0, interpreter isolation and cached-layer limits. No new cold build is performed by this closeout.

### Historical diagnostics: do not replay

The following November 11–12 record preserves failed attempts, old branch commands and then-current next steps. Its imperative commands and “not yet verified” statements are historical. Use the current path above; never delete evidence or reuse these names.

<details>
<summary>Historical image-01 through image-06 diagnostics and November 12 milestones</summary>

### Original diagnostic sequence (historical)

The first two image attempts have already failed and remain preserved. For the new
**pilot-only, exact-conflict diagnostic image**, pull the new draft PR branch and run the
offline suite before any more billable work:

```bash
git status --short
git pull --ff-only origin feat/detached-validation
uv run --locked python -m unittest discover -s tests -v

# CPU image-build and diagnostic validation. Billable build; no H100 reserved.
bash scripts/validate_modal.sh image image-03 \
  --wandb-entity amulyap-stanford-university
```

This changes the original expensive build script, so the first new two-layer
image will compile FlashAttention again. Later changes to
`scripts/finalize_modal_training.sh` or the runtime auditor should reuse the
cached CUDA compilation layer as long as its inputs remain unchanged.

**Do not interpret a completed image build as a qualified remote worker.** After
the diagnostic job is submitted and finishes, retrieve:

```bash
uv run --locked modal volume get agent-lightning-validation runs/image-03/result.json -
uv run --locked modal volume get agent-lightning-validation runs/image-03/launcher.json -
uv run --locked modal volume get agent-lightning-validation runs/image-03/dependency-conflicts.json -
uv run --locked modal volume get agent-lightning-validation runs/image-03/dependency-check.txt -
uv run --locked modal volume get agent-lightning-validation runs/image-03/imports.json -
```

Expected **pilot diagnostic** status is
`diagnostic_passed_with_reviewed_exceptions`, accompanied by
`dependency_closure_consistent: false`, six exact reviewed exceptions and
an independently recorded W&B run. An alternative genuinely consistent environment
could report `status: passed`. In either case, the remote launcher must
say `status: completed` and the requested CPU imports must succeed.
Anything else is a failed gate. The image does not become a production-qualified
dependency environment merely because its diagnostic status passed.

**Only after reviewing the image-03 evidence**, the separate, billable H100
preflight can be submitted with a fresh name:

```bash
bash scripts/validate_modal.sh gpu gpu-01 \
  --wandb-entity amulyap-stanford-university
```

After GPU completion:

```bash
uv run --locked modal volume get agent-lightning-validation runs/gpu-01/result.json -
uv run --locked modal volume get agent-lightning-validation runs/gpu-01/launcher.json -
uv run --locked modal volume get agent-lightning-validation runs/gpu-01/gpu_preflight.json -
```

`packages.json` records installed distributions; `imports.json`, `dependency-progress.json`, and `dependency-check.txt` localize dependency failures. The runtime verifies that the training checks run in `/opt/agent-lightning/.venv` on Python 3.12, not in the lightweight Modal worker interpreter. `result.json` records the actual Modal image ID when supplied by Modal. Submission/launcher records retain source commit and selected file hashes, never environment dumps or secret values.

The GPU report checks Torch/FlashAttention ABI and finite kernel outputs. It is not an independent numerical accuracy test. The training stack's transitive dependencies are recorded, not claimed to be fully locked by the small local `uv.lock`.

### Observed `image-01` failure and compiler correction (November 11, 2026)

The user-reported first training image build (Modal App `ap-chK2LvajVifYbfjLsFWwZn`) failed **before the remote validation function was submitted**. Its final build stage installed Agent Lightning 1.0.3, PyTorch 2.11.0+cu130, vLLM 0.22.0 and VERL 0.8.0, then failed while building `flash-attn==2.8.3` from source. PyTorch saw `clang++` version `0.0.0` as the host compiler and rejected it for CUDA 13.0. This is not a missing NVIDIA driver in a CPU image build; the earlier driver warning is expected.

The branch installer now exports `CC=/usr/bin/gcc`, `CXX=/usr/bin/g++` and `CUDAHOSTCXX=/usr/bin/g++` before calling pinned upstream `setup_verl.sh`, checks that these compilers exist, and records the selected compiler/version in build evidence. No training package pins or evaluation logic were changed. The added `tests/test_cuda_build_compiler.py` checks shell syntax and compiler-selection ordering **offline**. Passing these tests does not prove FlashAttention compiles or that the complete dependency stack is compatible.

The first compiler correction was tested in the `image-02` build; see the outcome below. Both failed image-build attempts are retained as historical evidence.

### Confirmed `image-02` failure: incompatible published metadata (November 11, 2026)

`image-02` successfully built `flash-attn==2.8.3` from source with GCC 13.3.0 in **38 minutes 46 seconds**. The build then installed the experiment agent libraries and failed at the intentionally strict `uv pip check` with **six incompatibilities**:

- `ml-dtypes==0.6.0` and `opencv-python-headless==5.0.0.93` require NumPy 2+, but VERL 0.8.0 selected NumPy 1.26.4.
- `xgrammar==0.2.5.post1` requires Transformers <5, but the VERL install selected Transformers 5.22.2.
- `google-api-core==2.44.0`, `tensorboard==2.21.0`, and `proto-plus==1.29.0` require Protobuf 6+, but the agent install selected Protobuf 5.29.6.

**Additional constraint confirmed from published PyPI metadata:** `vllm==0.22.0` requires `opencv-python-headless>=4.13.0`; OpenCV 4.13+ requires NumPy >=2 on Python 3.12, while `verl==0.8.0` requires `numpy<2`. Therefore **no single fully metadata-consistent Python environment exists for this exact version combination**. Downgrading OpenCV to 4.11 does not solve it; vLLM would then be inconsistent instead. A pin-only overlay was drafted and reverted before recommendation after this additional incompatibility was identified.

**The current draft PR implements option 3 for a bounded academic diagnostic only.** It retains the exact pinned upstream stack and records the nonzero `uv pip check` output and exit code. The remote auditor accepts only the six exact `image-02` conflicts with original package versions, required ranges, installed versions, Python 3.12, x86-64, CUDA-tagged PyTorch, and pinned upstream commit. A changed or unparseable report is rejected. This is **not** a clean dependency closure, and the original pinned source, model and evaluation code remain unchanged. Do not run GPU or model training merely because an image builds.

### Observed `image-03` outcome: build succeeded, remote audit failed (November 11, 2026)

The user supplied live Modal build logs showing: the first CPU build worker was **preempted** after approximately 29 minutes; Modal automatically rescheduled the same image build; FlashAttention 2.8.3 compiled successfully in the replacement worker after 51 minutes 10 seconds; the second `finalize_training_stack` layer completed and recorded the same six known `uv pip check` conflicts. **The final image then started an `image-03` CPU validation worker, which exited with `VALIDATION_FAILED stage=training_dependencies type=RuntimeError`.** This is not an image build error, GPU driver error or evidence of a GPU allocation. No W&B run was initiated by that failed worker.

The top-level Modal logs intentionally do not include the reason for the worker's internal `RuntimeError`. Do **not** infer that the conflict allowlist is wrong or that particular imports failed until the evidence below has been examined.

Retrieve read-only results from the already committed `agent-lightning-validation` Volume (no image build, GPU or training):

```bash
uv run --locked modal volume ls agent-lightning-validation runs/image-03

uv run --locked modal volume get agent-lightning-validation runs/image-03/dependency-progress.json -
uv run --locked modal volume get agent-lightning-validation runs/image-03/dependency-conflicts.json -
uv run --locked modal volume get agent-lightning-validation runs/image-03/dependency-check.txt -
uv run --locked modal volume get agent-lightning-validation runs/image-03/runtime.log -
uv run --locked modal volume get agent-lightning-validation runs/image-03/launcher.json -
```

If a file is missing, record that fact rather than manufacturing it. `dependency-progress.json` distinguishes pinned-version checks from module-import progress; `dependency-conflicts.json` identifies unexpected metadata, installed-version or source-identity drift; `runtime.log` records the worker's known failure class and stage. Inspect these files and identify the precise gate **before** modifying the allowlist or initiating `image-04`. Successful FlashAttention and agent layers should remain cacheable while only validation code changes.

### Image-03 Volume diagnosis and validation-only correction (November 12, 2026)

The user retrieved the persisted `image-03` Volume artifacts. The remote `dependency-check.txt` contains **six unchanged** requirement conflicts but begins with `Using Python 3.12.1 environment at: /opt/agent-lightning/.venv`. The original exact parser wrongly required `Checked ...` to be the first line, so `dependency-conflicts.json` reported `report_parsed: false`.

The original `packages.json` inventory also selected **protobuf 6.31.1**, while the same worker's `uv pip check --python /opt/agent-lightning/.venv/bin/python` saw **protobuf 5.29.6**. The former inventory used unscoped `importlib.metadata.distributions()`, which can observe distributions outside the selected training venv. **Whether that accounts for the exact discrepancy has not yet been verified.** Do not silently change the approved Protobuf version or exception list.

The bounded correction touches **validation code only**, preserving cached CUDA/FlashAttention and agent dependency layers:

- `validation/compatibility.py` now accepts either no banner or exactly the pinned training-venv Python 3.12 banner; unexpected output, new conflicts, range/version drift, or incorrect counts still fail closed.
- `validation/runtime.py` enumerates package metadata explicitly under `/opt/agent-lightning/.venv/lib/python3.12/site-packages`, rejects ambiguous installed dist-info, and records `package-inventory-details.json` with the scoped and ambient Protobuf observations. It imports `google.protobuf` and rejects a runtime version different from the scoped distribution.
- The same runtime previously returned exit code 1 for its own accepted diagnostic status. The corrected exit handler permits `diagnostic_passed_with_reviewed_exceptions` **only** when the exact six conflicts and completed W&B/Hugging Face/CPU-import evidence accompany it. Any failed gate remains nonzero.
- Added `tests/test_image03_diagnostics.py` and updated the mocked CPU-import test. The entire changed suite must run locally before another Modal invocation.

**The Modal logs show `function:image_check`, which uses four CPUs and no GPU**. The separately declared `gpu_check` function requests an H100; its existence in the same app does not prove that an H100 was allocated.

Before any new billable attempt:

```bash
git status --short
git pull --ff-only origin feat/detached-validation
uv run --locked python -m unittest discover -s tests -v
```

If offline tests pass and the next CPU-only diagnostic run is explicitly approved, use a fresh name, `image-04`. Existing successful build layers should be cached, but this is **not a guarantee** against Modal cache invalidation or infrastructure preemption. Retain `image-03` unchanged. After `image-04`, inspect `dependency-conflicts.json`, `package-inventory-details.json`, `protobuf-import.json`, `imports.json`, `result.json` and `launcher.json`. If the scoped venv still shows Protobuf 6.31.1 or duplicates, stop and diagnose rather than weakening the exception policy. Do not start `gpu-01` solely on image-build completion.

### Image-04 result: dependency policy accepted; `datasets` import failed (November 12, 2026)

The user executed `uv run --locked python -m unittest discover -s tests -v` in the Workbench: **63/63 tests passed** before submitting `image-04`. The remote CPU worker reused the prior training image (no H100 invocation shown). Its persistent Volume evidence reports `accepted: true`, `report_parsed: true`, `identity_matches: true`, and exactly six reviewed dependency conflicts.

The Python distribution ambiguity from `image-03` is understood: `protobuf==5.29.6` is installed in the Agent Lightning virtualenv; an independent `protobuf==6.31.1` distribution is also visible at `/__modal/deps/protobuf-6.31.1.dist-info`. The scoped venv audit correctly selects 5.29.6. `protobuf-import.json` is absent because the CPU import loop stopped earlier.

The next failure was **`ImportError` within the `training_dependencies` stage**. `imports.json` is 56 bytes, consistent with `torch` and `transformers` marked imported. `dependency-progress.json` is 37 bytes, consistent with the last attempted `datasets` module; the CLI display elided its body. This strongly localizes the failure to a transitive import when loading `datasets==5.2.0`, but the exact missing name or symbol has not yet been captured. Do not change versions based on speculation.

The bounded code correction adds `record_import_failure()` in `validation/runtime.py`. When `importlib.import_module` raises `ImportError` (including `ModuleNotFoundError`), the remote worker now persists `import-failure.json` containing only the attempted module, exception type/name, optional path and bounded message/cause; current Hugging Face and W&B secret values are redacted. It **re-raises the exception**, so the failing import is never skipped. `tests/test_image04_import_failure.py` adds offline regression cases; they still require the user's full local test run.

After pulling this validation-only commit and passing the local tests, the next bounded diagnostic (billable CPU, but should reuse the completed CUDA and agent layers) is:

```bash
bash scripts/validate_modal.sh image image-05 \
  --wandb-entity amulyap-stanford-university

uv run --locked modal volume get agent-lightning-validation runs/image-05/import-failure.json -
uv run --locked modal volume get agent-lightning-validation runs/image-05/dependency-progress.json -
```

A missing `import-failure.json` means the worker did not fail with an `ImportError` inside the guarded import loop; inspect `result.json` and `launcher.json` instead. **No GPU job is justified yet**. The draft PR is not production qualified.

### Image-05 runtime failure and PyArrow compatibility correction (November 12, 2026)

The user's `image-05` CPU validation run reached `import datasets` and recorded the exact failure in `runs/image-05/import-failure.json`:

```json
{"attempted_module": "datasets", "exception_type": "ImportError",
 "message": "pyarrow requires NumPy 2.0 or newer, found 1.26.4"}
```

This is a **real binary import incompatibility**, not a parser issue. VERL 0.8.0 declares `numpy<2`; the latest selected PyArrow 26 requires NumPy >=2 when NumPy is installed, and Datasets 5.2.0 permits PyArrow >=24 without an upper cap. Apache Arrow 25.0.1 supports NumPy 1.26.4, and a published Python 3.12 Linux x86-64 wheel is available. References: [Arrow 25 install guide](https://arrow.apache.org/docs/25.0/python/install.html), [Arrow issue 53954](https://github.com/apache/arrow/issues/53954), [Datasets 5.2.0 setup](https://github.com/huggingface/datasets/blob/5.2.0/setup.py), and [VERL 0.8.0 setup](https://github.com/verl-project/verl/blob/v0.8.0/setup.py).

**Bounded academic compatibility change:** the fast `scripts/finalize_modal_training.sh` stage explicitly installs `pyarrow==25.0.1` with `--no-deps --no-build` **after** all upstream and agent dependencies. It does not change NumPy, VERL, Datasets, PyTorch, vLLM, FlashAttention, the expensive image-build script, or experiment logic. The exact remote compatibility policy now requires `pyarrow==25.0.1` and `datasets==5.2.0` in addition to prior versions. The six separately reviewed dependency conflicts remain visible and unchanged. Do not expand the six-conflict allowlist.

The remote CPU import audit now tests `numpy` and `pyarrow` before `datasets` and runs a **synthetic three-row Datasets → Parquet → PyArrow/NumPy → Datasets round-trip**. Successful evidence is written to `data-smoke.json` with version identifiers. On failure it writes `data-smoke-failure.json` with a sanitized, bounded error and rejects the run. No model weights or external dataset are required. New offline tests use stub modules, so passing the Workbench suite alone is **not** proof of the actual PyArrow C-extension ABI.

After pulling the corrected draft branch, run the complete offline test suite first:

```bash
cd /workspace/projects/repos/agent-lightning-lab
git status --short
git pull --ff-only origin feat/detached-validation
uv run --locked python -m unittest discover -s tests -v
```

If tests pass, **only one explicitly authorized, billable CPU validation job** should be submitted, using the fresh name `image-06`:

```bash
bash scripts/validate_modal.sh image image-06 \
  --wandb-entity amulyap-stanford-university

uv run --locked modal volume get agent-lightning-validation runs/image-06/data-smoke.json -
uv run --locked modal volume get agent-lightning-validation runs/image-06/dependency-conflicts.json -
uv run --locked modal volume get agent-lightning-validation runs/image-06/result.json -
uv run --locked modal volume get agent-lightning-validation runs/image-06/launcher.json -
```

If `image-06` fails, inspect `dependency-progress.json`, `import-failure.json`, and `data-smoke-failure.json` **when they exist**. A completed image build alone is not a passed CPU gate. **No H100 GPU job or RL training has been authorized or executed by this fix.** Review the completed CPU diagnostic (including independent W&B readback and required imports) before considering `gpu-01`.

### CPU and H100 kernel-validation milestones passed (November 12, 2026)

User-supplied **`image-06`** evidence confirms completed remote CPU validation on exact source commit `33753ecf9fbf4406f021b2cc68c9f13ae7ce2e3d`. The three-row Datasets/Parquet/NumPy round-trip passed using NumPy 1.26.4, PyArrow 25.0.1 and Datasets 5.2.0. All required CPU imports, gated Hugging Face `config.json` access, W&B finished-state/summary/heartbeat readback and the Modal launcher passed. Status: `diagnostic_passed_with_reviewed_exceptions`; six exact metadata conflicts are still declared.

User-supplied **`gpu-01`** Volume evidence confirms the real `gpu_check` function completed on `NVIDIA H100 80GB HBM3` (`85009367040` bytes, CUDA 13.0) with `gpu_kernel_preflight_passed`. The GPU preflight imported vLLM 0.22.0 and FlashAttention 2.8.3 and ran finite BF16 matrix multiplication and FlashAttention kernels. The worker also repeated CPU/data validation and W&B readback; the launcher completed with exit code 0.

**What these passes do not establish:** Llama 3.2 3B weight download, actual vLLM model loading or structured tool calls, Agent Lightning rollouts, reward delivery, gradient updates, checkpoint persistence/reload or Weave trace delivery. Six conflicting published package metadata requirements remain formally unresolved; the diagnostic exception is not a production dependency qualification.

**Next bounded milestone: CPU-only asset preparation**, in the separate training Volume `cs329z-agent-lightning`:

```bash
uv run --locked modal run --detach modal_app.py \
  --action prepare --name prepare-01 --model-profile llama3b
```

This requests no GPU but downloads a gated, pinned Llama checkpoint and the official Calc-X dataset, which incurs CPU/storage charges. Only submit after reviewing the cost/storage and token's weight access. The remote action preserves staging data on failure and commits a terminal `submissions/prepare-01/receipt.json`; it does not validate model inference.

Read the resulting evidence without downloading model weights:

```bash
uv run --locked modal volume get cs329z-agent-lightning submissions/prepare-01/receipt.json -
uv run --locked modal volume get cs329z-agent-lightning models/llama3b/model_manifest.json -
uv run --locked modal volume get cs329z-agent-lightning data-small/data_manifest.json -
```

Accept only if `receipt.json` is `completed`, model `revision` is `0cb88a4f764b7a12671c53f0838cd831a0843b95` with `LlamaForCausalLM`, the frozen tokenizer/template hashes are recorded, the model files exist, and the manifest has 256 training / 64 final held-out / 16 training smoke / 8 separate smoke-evaluation cases with no train/eval ID overlap. Do not run the first two-step H100 GRPO smoke until the preparation receipts are reviewed.

### Conditions before proceeding beyond `image-03`

1. The CPU diagnostic is explicitly tagged in the remote result and W&B config/summary as `diagnostic_with_reviewed_dependency_exceptions`, with `dependency_closure_consistent: false`.
2. All affected packages must import, especially `cv2`, `ml_dtypes`, `xgrammar`, TensorBoard, and `google.api_core`. The validation code does not treat import failures as harmless.
3. A passing H100 preflight establishes only CUDA/FlashAttention kernels. A separate vLLM Llama tool-call and GRPO/reload smoke is required before any learning claim, particularly for xgrammar + Transformers 5.x.
4. No change to a reviewed package, its requirement string, the installed dependency version, CUDA-tagged PyTorch, Python architecture, or upstream source commit can silently reuse the exception. Such changes fail and require review.
5. A passing diagnostic does **not** authorize enterprise use, establish performance, or justify merging the draft PR without reviewing evidence.

Relevant primary metadata: [VERL 0.8.0](https://pypi.org/pypi/verl/0.8.0/json), [vLLM 0.22.0](https://pypi.org/pypi/vllm/0.22.0/json), [OpenCV 4.13.0.90](https://pypi.org/pypi/opencv-python-headless/4.13.0.90/json).

</details>

## 7. Continue the real experiment only after these gates

Use [README section 5](README.md#5-prepare-on-modal-then-pass-the-smoke-checks) and the [experiment runbook](experiment/README.md#action-order-and-acceptance-evidence): `prepare`, two-step `smoke`, and `smoke-reload-eval` before baseline/main/final evaluation. Preserve `--detach` on every long-running Modal invocation. The existing training commands call a remote function that does the entire phase; their local log connection is not required once that remote phase is submitted.

Each invocation submits **one** action. A shell script waiting on your laptop to submit a later action will not continue after laptop shutdown. Detachment does not queue the rest of a local command sequence. These tests do not add a full remote multi-phase scheduler.

Keep the first training smoke supervised for correctness and cost, even though the job is detached. Confirm real model generation, tool-result continuation, reward delivery, finite updates, weight refresh, native checkpoint save and reload. With `--weave`, separately inspect real hosted traces; none of the three infrastructure checks claims Weave delivery.

## 8. Failure diagnostics and retained evidence

| Symptom | Inspect / next action |
|---|---|
| Local `.env` check fails | Permissions, ignored/untracked status, two non-placeholder keys; do not print the file |
| Secret missing | Selected workspace/environment and exact name `agent-lightning-credentials` |
| No `SUBMITTED` / no result file | Local CLI error, authentication, source upload or image-build log; no worker success can be inferred |
| `wandb_init` failure | W&B key, actual entity slug, project write permission; no local `wandb login` is needed |
| `hugging_face` failure | Token, Llama license/access, network; downloading public metadata alone would not prove gated access |
| `training_dependencies` failure | `dependency-progress.json`, package inventory, import log and dependency-check output |
| GPU preflight failure | CUDA/GPU availability, native library ABI, `runtime.log`; do not start training |
| `wandb_readback` failure | W&B run link and service state; keep evidence, use a new name after diagnosing |
| Process ends, result still `started` | `launcher.json`, Modal timeout/stop/infrastructure logs; treat as incomplete |

The Volume `agent-lightning-validation` is separate from the training Volume. Artifacts are under `runs/NAME/`. Download them explicitly; `.local` is not a backup:

```bash
mkdir -p .local/validation-downloads
uv run --locked modal volume get agent-lightning-validation \
  runs/repro-a-services .local/validation-downloads/repro-a-services
```

Do not delete a shared Volume or Secret as routine cleanup. Confirm jobs ended and retain selected evidence before considering storage removal. Review logs for sensitive metadata before sharing them or committing them to Git.

## Primary references

- [Modal App lifetime and detach](https://modal.com/docs/guide/apps)
- [Modal background batch submission](https://modal.com/docs/guide/batch-processing)
- [Modal image dependencies and caching](https://modal.com/docs/guide/images)
- [Modal Secret CLI and dotenv import](https://modal.com/docs/cli/latest/secret)
- [Modal runtime environment identifiers](https://modal.com/docs/guide/environment_variables)
- [uv lock and synchronization](https://docs.astral.sh/uv/concepts/projects/sync/)
- [W&B Python SDK quickstart](https://github.com/wandb/wandb/blob/main/README.md)
