# Reviewed environment and deployment decision

Reviewed November 11, 2026. These are repository and supplied-runbook observations, not inspection of a live laptop. The initial deployment recommendation below is retained as history; the dated Modal amendment supersedes it. Workbench facts describe the reviewed snapshot, not a required client layout.

## Current status (November 13, 2026)

Use the [root README](../README.md) for current setup and reader paths. The later [smoke](../evidence/grpo-smoke-01.md) and [20-step/reload evidence](../evidence/grpo-main-01.md) establish actual image execution, Llama inference/tool continuation, H100 memory fit for the recorded settings, and operational native checkpoint reload. They supersede the earlier runtime-unknown status below without changing what was checked on November 11. Six dependency exceptions, unverified tensor identity/hosted readback/experiment Protobuf import, and scorer/reproducibility limits remain. A new cold rebuild and ARM64 GPU training are not qualified. [The documentation closeout](CLOSEOUT_2026-11-13.md) records new local checks separately.

## Sources inspected

- Unix AI Workbench `main` at `3e3951396bc23e05c4d95084e76ae6f5a60052c3`: `README.md`, `AGENTS.md`, `.devcontainer/devcontainer.json`, `.devcontainer/Dockerfile`, `.devcontainer/requirements.txt`, `.gitignore`, `workbench`, `scripts/prepare-host.sh`, `docs/repository-preparation.md`.
- CS329Z repository `main` at `d0fb0ed3bcebe968126f06920e574e8681954334`: `README.md`, `PROJECT_CONTEXT.md`, `playbook/decisions.md`.
- User-supplied `Surface Laptop Ultra local model runbook 111126-0902.md`.
- The preceding Agent Lightning weekend starter and its explicit verification record.

## Current Workbench facts

The Workbench uses `mcr.microsoft.com/devcontainers/base:ubuntu-24.04`. It includes Python, Git, GitHub CLI, rsync and uv 0.12.13, alongside its existing document tools. The image has no model server or GPU setup. It exposes the repository at `/workspace`, host state at `/state`, and persistent user tools at `/home/vscode/.local`. The shared Python environment is `/opt/venv`.

The parent `.gitignore` excludes `/projects/repos/`, so independent child repositories are already the intended pattern. The Workbench definition has no Docker socket mount. Its current runtime configuration does not join `llm` or set `LOCAL_LLM_BASE_URL`. The existing helper's offline test uses `--network none`.

Repository facts do not prove that a particular currently running laptop container has been rebuilt from that commit. Local uncommitted changes can also differ from GitHub.

## Surface Ultra record

The attachment records ARM64 WSL Docker Engine, NVIDIA Container Toolkit, GPU visibility, working llama.cpp GGUF inference and Compose ownership of the local inference service. It records `~/src/local-ai-stack`, `~/ai/models`, external network `llm`, container `llama`, loopback host port 9931, and client URL `http://llama:9931/v1` on that network.

Workbench wiring, OpenHands/Hermes integration, and the separate vLLM qualification remain pending in that record. No local Agent Lightning/VERL GPU training was demonstrated. Preserve the working local Compose files and their actual image identity. The attachment does not provide a digest that this starter can safely pin on the user's behalf.

One storage sentence in the attachment still names the native Windows Hugging Face cache. Its implemented container path is WSL `~/ai/models`; this starter uses that implemented boundary and does not move model files.

## Initial deployment decision (superseded below)

Reuse the existing Workbench. Keep this bounded experiment in one small private `agent-lightning-lab` repository under `projects/repos/`, consistent with the existing independent-project pattern. Keep the course repository as the study and paper-video navigation location. This is a source-ownership choice, not a new application platform. Do not fork the Workbench, duplicate its large image, add GPU packages to shared `/opt/venv`, or turn the course extension into a second ReliOps project.

Use the existing managed x86 cloud GPU container for the deadline. The optional ordinary Docker runtime in this package is for a future self-managed x86 Linux GPU host. It is not a second container to run inside a RunPod Pod. Its regular runtime could later be attached to VS Code without making a second general-purpose Workbench.

Local Ultra inference remains owned by `local-ai-stack`. Connecting a CPU client to it is useful for connectivity and eventual tool-loop checks. It does not make the GGUF service the training backend. This experiment's trainer requires its own trainable model and controlled weight synchronization.

## Proposed course-playbook note

Keep reusable editing tools, experiment source, and hardware-specific runtime dependencies in separate ownership boundaries. Git stores the recipe and selected evidence. Rebuild architecture-specific environments on the target host, preserve actual run records, and verify inference, tools, reward, update, checkpoint and reload before scaling.

At the time of the initial starter review, no upstream or user GitHub repository had been edited.

## Primary references

- [Workbench README at reviewed commit](https://github.com/amulyaparthasarathy/unix-ai-workbench/blob/3e3951396bc23e05c4d95084e76ae6f5a60052c3/README.md)
- [Workbench Dev Container configuration](https://github.com/amulyaparthasarathy/unix-ai-workbench/blob/3e3951396bc23e05c4d95084e76ae6f5a60052c3/.devcontainer/devcontainer.json)
- [Workbench Dockerfile](https://github.com/amulyaparthasarathy/unix-ai-workbench/blob/3e3951396bc23e05c4d95084e76ae6f5a60052c3/.devcontainer/Dockerfile)
- [Workbench Git exclusions](https://github.com/amulyaparthasarathy/unix-ai-workbench/blob/3e3951396bc23e05c4d95084e76ae6f5a60052c3/.gitignore)
- [CS329Z ownership and learning context](https://github.com/amulyaparthasarathy/cs329z-fall-2026/blob/d0fb0ed3bcebe968126f06920e574e8681954334/PROJECT_CONTEXT.md)
- [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/)
- [NVIDIA Container Toolkit installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)
- [Compose networking](https://docs.docker.com/compose/how-tos/networking/)
- [Compose GPU support](https://docs.docker.com/compose/how-tos/gpu-support/)
- [Docker bind mounts](https://docs.docker.com/engine/storage/bind-mounts/)

## Amendment: Modal is the primary cloud path (November 11, 2026)

The user has created the private [`agent-lightning-lab`](https://github.com/amulyaparthasarathy/agent-lightning-lab) repository and explicitly selected Modal, using their existing experience from CS224R and CS224N. The reviewed lab baseline is `1e849089fe503a40570f46a140e509a1799344e0`. The current [deployment README](../README.md), [experiment runbook](../experiment/README.md) and [Modal adapter](../modal_app.py) replace the initial RunPod deployment procedure.

Reuse the current Workbench, the independent lab repository under `projects/repos/`, and the existing Modal profile through `MODAL_CONFIG_PATH=/state/modal.toml`. Use `uv sync --locked` for the repository's local client environment and `uv run --locked` for commands. uv creates and manages `.venv`; it does not eliminate virtual environments. The Workbench's shared `/opt/venv` remains a separate environment, so do not use `uv sync --active` for this project.

The deployment boundaries are now:

| Boundary | Owner and contents |
|---|---|
| Existing Workbench | Editing, Git, uv and the Modal/Hugging Face clients |
| Existing lab repository | Experiment code, pinned dependencies, Modal image/resource definition and selected evidence |
| Modal image | Python 3.12, the pinned Agent Lightning/VERL/vLLM training environment and H100-targeted dependencies |
| Modal Secret | `agent-lightning-credentials`, containing `HF_TOKEN` and `WANDB_API_KEY` |
| Modal Volume | `cs329z-agent-lightning` at `/data`, retaining model assets, frozen splits, run records and checkpoints |
| Existing local inference service | The user's separately owned Ultra/WSL model-serving setup |

The [image installer](../scripts/install_modal_training.sh) builds the training environment on CPU. CPU preparation downloads model/data assets before the first GPU phase. GPU actions request one H100, 16 physical CPU cores and 128 GiB RAM, then run the existing launcher after runtime preflight. Run actions serially with unique names, retain the two-step and checkpoint-reload gates, and export small evidence archives through the Volume CLI. Hosted Weave traces are explicitly enabled with `--weave`.

No new Dev Container, local NVIDIA driver installation, Docker socket mount or Workbench network change is required for this cloud path. The current local inference service remains useful for separate work; it is not a prerequisite for the Modal experiment. The optional native Compose runtime and `bootstrap_gpu.sh` are retained for future self-managed GPU qualification and are not steps in the current runbook.

At this November 11 amendment, only the deployment recipe had been source-reviewed: no Modal build, GPU training, Llama load or hosted Weave delivery had been performed during that preparation. CUDA compatibility, memory fit and reload were then pending; see the current-status amendment above for later measured evidence. Cloud cost and hosted Weave delivery are not established by that later evidence. Git preserves the recipes and selected results; large assets persist in the Volume rather than being reconstructed from Git alone.

Updated proposed playbook decision: keep one reusable Workbench and one focused experiment repository; describe the cloud training runtime in code, manage local client dependencies with uv, and preserve the complete inference → tool → reward → update → checkpoint → reload evidence before scaling.

Amendment references: [Modal images](https://modal.com/docs/guide/images), [resources](https://modal.com/docs/guide/resources), [Volumes](https://modal.com/docs/guide/volumes), [Secrets](https://modal.com/docs/guide/secrets), [uv project synchronization](https://docs.astral.sh/uv/concepts/projects/sync/) and [uv project environments](https://docs.astral.sh/uv/concepts/projects/config/).
