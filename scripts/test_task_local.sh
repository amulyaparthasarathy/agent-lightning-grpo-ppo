#!/usr/bin/env bash
# Essential real MCP/harness tests fail with an actionable setup error, never skip.
set -euo pipefail
cd "$(dirname "$0")/.."
task_python="${TASK_TEST_PYTHON:-.venv-reader-cpu/bin/python}"
if [[ ! -x "$task_python" ]]; then
  echo 'Run UV_PROJECT_ENVIRONMENT=.venv-reader-cpu uv sync --locked --only-group reader-harness' >&2
  exit 2
fi
"$task_python" - <<'PY'
from importlib.metadata import version
for name, expected in {'autogen-core':'0.7.5', 'autogen-agentchat':'0.7.5', 'autogen-ext':'0.7.5',
                       'mcp':'1.26.0', 'openai':'2.54.0', 'pyarrow':'25.0.1'}.items():
    assert version(name) == expected, 'Use the locked reader-harness environment: '+name
PY
task_home="$(mktemp -d /tmp/task-local-XXXXXX)"
trap 'rm -rf "$task_home"' EXIT
# Loopback HTTP rewards are required; no cloud/model/API boundary is invoked.
env -i PATH="$PATH" HOME="$task_home" MODAL_CONFIG_PATH=/dev/null HF_HUB_OFFLINE=1 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD:$PWD/tests" \
  "$task_python" -m unittest test_asset_task test_phase3_source -v
