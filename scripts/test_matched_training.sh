#!/usr/bin/env bash
# Local CPU only. The default is verification, with no downloads or API calls.
set -euo pipefail
cd "$(dirname "$0")/.."
MATCHED_TEST_PYTHON="${MATCHED_TEST_PYTHON:-.local/matched-test-venv/bin/python}"
if [[ "${1:-}" == "--prepare" ]]; then
    uv venv .local/matched-test-venv
    uv pip install --python "$MATCHED_TEST_PYTHON" 'torch==2.11.0' --index-url https://download.pytorch.org/whl/cpu
    uv pip install --python "$MATCHED_TEST_PYTHON" 'numpy==1.26.4' 'tensordict==0.10.0' 'omegaconf==2.3.0' 'hydra-core==1.3.2' 'packaging==25.0' 'modal==1.6.3' 'ray==2.60.0' 'pyzmq==27.3.0' 'psutil==7.2.2' 'torchdata==0.11.0' 'datasets==5.2.0' 'pyarrow==25.0.1' 'einops==0.8.2'
    # Download code only, never models. Reuse existing reviewed checkouts when available.
    "$MATCHED_TEST_PYTHON" scripts/prepare_matched_test_sources.py
elif [[ $# -ne 0 ]]; then
    echo "usage: $0 [--prepare]" >&2
    exit 2
fi
export MODAL_CONFIG_PATH=/dev/null
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="tests${PYTHONPATH:+:$PYTHONPATH}"
"$MATCHED_TEST_PYTHON" - <<'PY'
import torch, numpy, tensordict, omegaconf, hydra, ray, zmq
from pinned_training_support import ROOTS, LOCK, source
assert torch.__version__.split('+')[0] == '2.11.0'
assert numpy.__version__ == '1.26.4'
assert tensordict.__version__ == '0.10.0'
assert omegaconf.__version__ == '2.3.0'
assert hydra.__version__ == '1.3.2'
assert ray.__version__ == '2.60.0'
assert zmq.__version__ == '27.3.0'
for package in ROOTS:
    for relative in LOCK[package]:
        source(package, relative)
print('Pinned code and CPU numerical dependencies verified; no skipped training tests allowed.')
PY
"$MATCHED_TEST_PYTHON" -m unittest discover -s tests -p 'test_matched*.py' -v
