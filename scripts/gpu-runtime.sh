#!/usr/bin/env bash
# Optional sibling runtime on a self-managed native x86_64 GPU host.
# Run on that host, outside the Workbench and outside a provider Pod.
set -Eeuo pipefail

AGL_SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export AGL_SOURCE_DIR
AGL_COMPOSE_FILE="$AGL_SOURCE_DIR/containers/compose.gpu.yaml"

fail() { printf '%s\n' "$*" >&2; exit 1; }
usage() {
    cat <<'USAGE'
Optional self-managed Linux x86_64 GPU runtime:
  export AGL_RUNTIME_DIR="$HOME/ai/runs/agent-lightning-lab"
  export AGL_GPU_DEVICE=0          # Optional: one index or GPU UUID
  bash scripts/gpu-runtime.sh check
  bash scripts/gpu-runtime.sh config
  bash scripts/gpu-runtime.sh up
  bash scripts/gpu-runtime.sh shell
  bash scripts/gpu-runtime.sh down

check is read-only. config renders configuration without starting a container.
up creates only the dedicated runtime directory, builds the image, and starts
an idle container. Install/train explicitly after entering shell.
down removes this Compose runtime, not its persistent bind-mounted files.
It does not stop a rented host or its provider billing.

The course path uses Modal: follow README.md and modal_app.py instead.
This helper intentionally does not run inside the Workbench or any container.
No .env file or host credentials are passed to the container.
USAGE
}

validate_runtime_path() {
    : "${AGL_RUNTIME_DIR:?Set AGL_RUNTIME_DIR to an absolute dedicated host directory}"
    [[ "$AGL_RUNTIME_DIR" == /* ]] || fail "AGL_RUNTIME_DIR must be absolute."
    [[ "$AGL_RUNTIME_DIR" != *$'\n'* && "$AGL_RUNTIME_DIR" != *$'\r'* ]] \
        || fail "AGL_RUNTIME_DIR must not contain line breaks."
    command -v realpath >/dev/null || fail "realpath is required on the Linux host."
    local lexical resolved
    lexical="$(realpath -m -s -- "$AGL_RUNTIME_DIR")"
    resolved="$(realpath -m -- "$AGL_RUNTIME_DIR")"
    [[ "$lexical" == "$resolved" ]] \
        || fail "AGL_RUNTIME_DIR must not traverse a symlink. Choose a dedicated real directory."
    AGL_RUNTIME_DIR="$resolved"
    case "$AGL_RUNTIME_DIR" in
        /|/home|/root|/etc|/usr|/var|/tmp|/opt|/mnt|/srv|/run|/proc|/sys|/dev)
            fail "Choose a dedicated runtime subdirectory, not a system directory." ;;
    esac
    [[ "$AGL_RUNTIME_DIR" != "${HOME:-}" ]] \
        || fail "Do not mount your entire home directory."
    case "$AGL_RUNTIME_DIR/" in
        "$AGL_SOURCE_DIR/"*) fail "Runtime data must live outside the source checkout." ;;
    esac
    case "$AGL_SOURCE_DIR/" in
        "$AGL_RUNTIME_DIR/"*) fail "Runtime data must not be an ancestor of the source checkout." ;;
    esac
    [[ ! -e "$AGL_RUNTIME_DIR" || -d "$AGL_RUNTIME_DIR" ]] \
        || fail "AGL_RUNTIME_DIR exists and is not a directory."
    export AGL_RUNTIME_DIR
    AGL_GPU_DEVICE="${AGL_GPU_DEVICE:-0}"
    [[ "$AGL_GPU_DEVICE" =~ ^[0-9]+$ || "$AGL_GPU_DEVICE" =~ ^GPU-[A-Fa-f0-9-]+$ ]] \
        || fail "AGL_GPU_DEVICE must name exactly one GPU index or GPU UUID."
    export AGL_GPU_DEVICE
}

outside_container() {
    [[ "${WORKBENCH_CONTAINER:-}" != 1 && ! -f /.dockerenv && ! -f /run/.containerenv ]] \
        || fail "Run this helper on the GPU host, outside the Workbench/provider container."
    [[ "$(uname -s)" == Linux ]] || fail "This runtime requires a Linux host."
}

docker_cli() {
    command -v docker >/dev/null || fail "Docker CLI is missing on this host."
    docker compose version >/dev/null || fail "The Docker Compose plugin is required."
}

local_engine() {
    docker_cli
    local endpoint info
    if [[ -n "${DOCKER_HOST:-}" && "$DOCKER_HOST" != unix://* ]]; then
        fail "Select the local Unix-socket Docker daemon; remote Docker mounts use remote paths."
    fi
    endpoint="$(docker context inspect --format '{{.Endpoints.docker.Host}}')"
    [[ "$endpoint" == unix://* ]] || fail "The selected Docker context is not a local Unix socket."
    info="$(docker info --format '{{.OSType}} {{.Architecture}}')" \
        || fail "The local Docker daemon is not accessible."
    [[ "$info" == 'linux x86_64' || "$info" == 'linux amd64' ]] \
        || fail "This training image requires a native Linux amd64 Docker daemon."
}

check_gpu_host() {
    outside_container
    [[ "$(uname -m)" == x86_64 ]] \
        || fail "This pinned training runtime requires native x86_64. ARM64 Surface training is not qualified."
    [[ -z "${DOCKER_DEFAULT_PLATFORM:-}" || "$DOCKER_DEFAULT_PLATFORM" == linux/amd64 ]] \
        || fail "DOCKER_DEFAULT_PLATFORM conflicts with the native x86_64 training runtime."
    local_engine
    command -v nvidia-smi >/dev/null || fail "nvidia-smi is missing on the GPU host."
    local driver major
    driver="$(nvidia-smi --id="$AGL_GPU_DEVICE" --query-gpu=driver_version --format=csv,noheader)" \
        || fail "The selected NVIDIA GPU is not accessible."
    major="${driver%%.*}"
    [[ "$major" =~ ^[0-9]+$ ]] && (( major >= 580 )) \
        || fail "The CUDA 13 recipe needs a compatible R580+ host driver."
    nvidia-smi --id="$AGL_GPU_DEVICE" --query-gpu=name,memory.total,driver_version --format=csv
    printf 'Source: %s\nRuntime data: %s\n' "$AGL_SOURCE_DIR" "$AGL_RUNTIME_DIR"
    printf '%s\n' 'Host checks passed. GPU container access, memory fit, and RL are still runtime checks.'
}

compose() {
    # /dev/null prevents automatic .env loading. Credentials are entered later
    # in the explicit interactive shell, not in the build or Compose definition.
    docker compose --env-file /dev/null --project-name agl-training \
        --file "$AGL_COMPOSE_FILE" "$@"
}

action="${1:-help}"
case "$action" in
    help|-h|--help) usage; exit 0 ;;
    check|config|up|shell|down) ;;
    *) usage >&2; exit 2 ;;
esac
[[ "$#" -eq 1 ]] || fail "Supply exactly one action."
validate_runtime_path
case "$action" in
    check) check_gpu_host ;;
    config) docker_cli; compose config ;;
    up)
        check_gpu_host
        umask 077
        mkdir -p -- "$AGL_RUNTIME_DIR"
        compose config --quiet
        compose up -d --build training
        if ! compose exec -T training nvidia-smi; then
            fail "The runtime started but GPU visibility failed. Inspect it, then run this helper's down action when finished."
        fi
        printf '%s\n' 'Idle runtime ready. Enter shell and run the documented experiment bootstrap.'
        ;;
    shell)
        outside_container; local_engine
        compose exec training bash
        ;;
    down)
        outside_container; local_engine
        compose down
        printf '%s\n' 'Runtime removed. Bind-mounted data retained. Provider billing is separate.'
        ;;
esac
