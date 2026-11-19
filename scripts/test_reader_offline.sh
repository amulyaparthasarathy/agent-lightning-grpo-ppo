#!/usr/bin/env bash
# Credential-free focused checks. Install locked client/test dependencies first.
set -euo pipefail
cd "$(dirname "$0")/.."
repo_root="$PWD"
test_cache="$(uv cache dir)"
test_home="$(mktemp -d /tmp/reader-offline-XXXXXX)"
trap 'rm -rf "$test_home"' EXIT
offline() {
  env -i PATH="$PATH" HOME="$test_home" UV_CACHE_DIR="$test_cache" UV_OFFLINE=1 \
    MODAL_CONFIG_PATH=/dev/null HF_HUB_OFFLINE=1 \
    PYTHONPATH="$repo_root/tests/offline_guard:$repo_root:$repo_root/tests" "$@"
}
offline python3 experiment/cpu_accounting_demo.py
offline python3 scripts/check_phase2_results.py
offline uv run --locked python -m unittest discover -s tests -p 'test_ppo_mc.py' -v
offline uv run --locked python -m unittest discover -s tests -p 'test_matched_study.py' -v
offline uv run --locked --group reader-test python -m unittest \
  test_reader_training test_modal_build_import test_modal_import test_modal_runner \
  test_cuda_build_compiler test_training_interpreter test_dependency_exceptions -v
