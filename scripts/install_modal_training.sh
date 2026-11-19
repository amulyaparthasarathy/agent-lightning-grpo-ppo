#!/usr/bin/env bash
# CPU-only Modal image build. No credentials, model download, or GPU reservation.
set -euo pipefail

AGL_REPO=/opt/agent-lightning
AGL_PIN=cbc56e41e0eab643aa71d195e92e5bba652fa683
AGL_BUILD=/opt/agl-build
export UV_CACHE_DIR=/tmp/agl-build-uv-cache
export UV_LINK_MODE=copy
export TORCH_CUDA_ARCH_LIST=9.0
export FLASH_ATTN_CUDA_ARCHS=90
export MAX_JOBS=4

if [[ "$(uname -m)" != x86_64 ]]; then
    echo 'This image recipe requires Linux x86_64.' >&2
    exit 1
fi
# Modal's Python image can select clang++ via the inherited build environment.
# PyTorch's C++ extensions and nvcc must use the GCC host toolchain installed
# by build-essential, matching the GCC-built PyTorch CUDA wheel.
# Export before upstream setup_verl.sh runs its FlashAttention source build.
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
for compiler in "$CC" "$CXX" "$CUDAHOSTCXX"; do
    if [[ ! -x "$compiler" ]]; then
        echo "Required CUDA host compiler missing: $compiler" >&2
        exit 1
    fi
done
printf 'CUDA extension compiler: %s (version %s)\n' "$CXX" "$("$CXX" -dumpfullversion -dumpversion)"

git clone --branch v1.0.3 --depth 1 https://github.com/microsoft/agent-lightning.git "$AGL_REPO"
test "$(git -C "$AGL_REPO" rev-parse HEAD)" = "$AGL_PIN"
cd "$AGL_REPO"
uv sync --frozen --python 3.12
# The upstream script installs the CUDA 13 compiler packages and builds FlashAttention.
# Explicit SM90 architecture makes compilation independent of a visible GPU.
bash scripts/setup_verl.sh 0.8.0 cu130
# This is the expensive, reusable CUDA compilation layer. Agent packages and
# the exact-conflict diagnostic gate are added in a later, lighter image layer.
git diff --exit-code
mkdir -p "$AGL_BUILD/evidence"
uv pip freeze --python .venv/bin/python > "$AGL_BUILD/evidence/base-installed-packages.txt"
uv --version > "$AGL_BUILD/evidence/base-uv-version.txt"
git rev-parse HEAD > "$AGL_BUILD/evidence/base-agent-lightning-commit.txt"
sha256sum "$AGL_BUILD/install_modal_training.sh" \
    > "$AGL_BUILD/evidence/base-build-inputs.sha256"
printf 'TORCH_CUDA_ARCH_LIST=%s\nFLASH_ATTN_CUDA_ARCHS=%s\nCC=%s\nCXX=%s\nCUDAHOSTCXX=%s\nCXX_VERSION=%s\nGPU allocated during build=false\n' \
    "$TORCH_CUDA_ARCH_LIST" "$FLASH_ATTN_CUDA_ARCHS" "$CC" "$CXX" "$CUDAHOSTCXX" \
    "$("$CXX" -dumpfullversion -dumpversion)" > "$AGL_BUILD/evidence/build-settings.txt"
uv cache clean
