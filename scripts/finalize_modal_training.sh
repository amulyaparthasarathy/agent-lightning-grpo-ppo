#!/usr/bin/env bash
# Fast agent dependency overlay after the expensive cached CUDA/FlashAttention layer.
# This stage records uv conflicts; the remote validation worker alone qualifies
# exact, version-bound exceptions. No package version is silently repinned.
set -euo pipefail

AGL_REPO=/opt/agent-lightning
AGL_BUILD=/opt/agl-build
export UV_CACHE_DIR=/tmp/agl-finalize-uv-cache
export UV_LINK_MODE=copy
cd "$AGL_REPO"
test "$(git rev-parse HEAD)" = cbc56e41e0eab643aa71d195e92e5bba652fa683

uv pip install --python .venv/bin/python \
    -r "$AGL_BUILD/requirements-agent.txt" 'gdown==5.2.0'

# VERL 0.8.0 requires NumPy <2, while PyArrow >=26 rejects NumPy 1.x at
# import time. Datasets 5.2.0 needs PyArrow >=24; 25.0.1 satisfies both.
# Keep this after upstream VERL and agent installation so neither upgrades it.
# --no-deps prevents unrelated changes to the pinned CUDA/training environment.
# --no-build requires a published CPython 3.12 Linux x86-64 wheel.
uv pip install --python .venv/bin/python --no-deps --no-build \
    'pyarrow==25.0.1'

mkdir -p "$AGL_BUILD/evidence"
uv pip freeze --python .venv/bin/python > "$AGL_BUILD/evidence/installed-packages.txt"

# Capture the actual status, including an unexpected command failure. Exit 1
# can mean incompatibilities, but the *exact* conflict set is checked remotely.
if uv pip check --python .venv/bin/python > "$AGL_BUILD/evidence/pip-check.txt" 2>&1; then
    check_rc=0
else
    check_rc=$?
fi
printf '%s\n' "$check_rc" > "$AGL_BUILD/evidence/pip-check-exit-code.txt"
cat "$AGL_BUILD/evidence/pip-check.txt"
if [[ "$check_rc" -ne 0 ]]; then
    if [[ "$check_rc" -ne 1 ]] || ! grep -Eq '^Found [1-9][0-9]* incompatibilities$' "$AGL_BUILD/evidence/pip-check.txt"; then
        echo 'uv pip check failed for an unexpected reason; refusing to build the final image' >&2
        exit 1
    fi
    echo 'WARNING: metadata conflicts recorded; image is NOT qualified until strict remote audit' >&2
fi

git diff --exit-code
uv --version > "$AGL_BUILD/evidence/uv-version.txt"
git rev-parse HEAD > "$AGL_BUILD/evidence/agent-lightning-commit.txt"
sha256sum "$AGL_BUILD/install_modal_training.sh" \
    "$AGL_BUILD/finalize_modal_training.sh" \
    "$AGL_BUILD/requirements-agent.txt" \
    > "$AGL_BUILD/evidence/build-inputs.sha256"
uv cache clean
