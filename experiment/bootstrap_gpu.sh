#!/usr/bin/env bash
# Run once on the fresh x86 cloud GPU environment. Does not provision cloud GPUs.
set -euo pipefail

LAB_PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGL_LAB_ROOT="${1:-/workspace/agl-course}"
AGL_PIN="cbc56e41e0eab643aa71d195e92e5bba652fa683"
if [[ "$(uname -m)" != "x86_64" ]]; then
    echo "This pinned GPU recipe targets Linux x86_64. Run it on the cloud GPU, not ARM WSL." >&2
    exit 1
fi
for executable in git uv nvidia-smi gcc g++ make; do
    if ! command -v "$executable" >/dev/null 2>&1; then
        echo "Missing prerequisite: $executable. Follow README system setup first." >&2
        exit 1
    fi
done
DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | cut -d. -f1)"
if [[ ! "$DRIVER_MAJOR" =~ ^[0-9]+$ ]] || (( DRIVER_MAJOR < 580 )); then
    echo "CUDA 13 requires a compatible host driver (R580+ baseline). Choose a compatible GPU host." >&2
    exit 1
fi
mkdir -p "$AGL_LAB_ROOT"
AGL_LAB_ROOT="$(cd "$AGL_LAB_ROOT" && pwd)"
export UV_CACHE_DIR="$AGL_LAB_ROOT/cache/uv"
export HF_HOME="$AGL_LAB_ROOT/cache/huggingface"
export PIP_CACHE_DIR="$AGL_LAB_ROOT/cache/pip"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$PIP_CACHE_DIR" "$AGL_LAB_ROOT/evidence/setup"
AGL_REPO="$AGL_LAB_ROOT/agent-lightning"
if [[ ! -d "$AGL_REPO" ]]; then
    git clone --branch v1.0.3 --depth 1 https://github.com/microsoft/agent-lightning.git "$AGL_REPO"
fi
if [[ "$(git -C "$AGL_REPO" rev-parse HEAD)" != "$AGL_PIN" ]]; then
    echo "Existing checkout is not the pinned revision. Use a fresh lab directory." >&2
    exit 1
fi
git -C "$AGL_REPO" diff --quiet
cd "$AGL_REPO"
uv sync --frozen --python 3.12
export TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d ' ')"
export MAX_JOBS="${MAX_JOBS:-4}"
bash scripts/setup_verl.sh 0.8.0 cu130
uv pip install --python .venv/bin/python -r "$LAB_PACKAGE_DIR/requirements-agent.txt"
uv pip install --python .venv/bin/python "weave==0.54.1" "gdown==5.2.0"
uv pip check --python .venv/bin/python
uv pip freeze --python .venv/bin/python > "$AGL_LAB_ROOT/evidence/setup/installed-packages.txt"
uv --version > "$AGL_LAB_ROOT/evidence/setup/uv-version.txt"
nvidia-smi > "$AGL_LAB_ROOT/evidence/setup/nvidia-smi.txt"
git rev-parse HEAD > "$AGL_LAB_ROOT/evidence/setup/agent-lightning-commit.txt"
{
    printf 'export AGL_LAB_ROOT=%q\n' "$AGL_LAB_ROOT"
    printf 'export AGL_REPO=%q\n' "$AGL_REPO"
    printf 'export AGL_PACKAGE=%q\n' "$LAB_PACKAGE_DIR"
    printf 'export HF_HOME=%q\n' "$HF_HOME"
    printf 'export UV_CACHE_DIR=%q\n' "$UV_CACHE_DIR"
    printf 'source %q\n' "$AGL_REPO/.venv/bin/activate"
} > "$AGL_LAB_ROOT/env.sh"
"$AGL_REPO/.venv/bin/python" "$LAB_PACKAGE_DIR/check_gpu.py" \
    --output "$AGL_LAB_ROOT/evidence/setup/gpu_preflight.json"
printf 'Installed pinned stack. In this shell run: source %q\n' "$AGL_LAB_ROOT/env.sh"
echo "Do not run uv sync again after adding the GPU extras; use the venv Python directly."
