#!/usr/bin/env bash
# Always detached. Does not load .env or pass secret values on command lines.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ $# -lt 2 ]]; then
  echo 'Usage: bash scripts/validate_modal.sh services|image|gpu NEW_NAME [extra Modal entrypoint arguments]' >&2
  exit 2
fi
kind="$1"; name="$2"; shift 2
case "$kind" in
  services) exec uv run --locked modal run --detach validation_services.py --name "$name" "$@" ;;
  image|gpu) exec uv run --locked modal run --detach validation_training.py --kind "$kind" --name "$name" "$@" ;;
  *) echo 'Unknown test. Choose services, image, or gpu.' >&2; exit 2 ;;
esac
