# Optional: connect the existing Workbench to local inference

This is only for local inference clients on the Surface Ultra. The cloud
Agent Lightning training path works without this patch. It does not turn the
existing GGUF server into a trainable Agent Lightning backend.

This is an optional patch for a **different repository**, not a prerequisite for the generic [Modal client setup](../README.md#2-local-client-setup). Its `.devcontainer` paths refer to that external target; this lab has no container definition of its own. Historical machine names and patch instructions below identify that specific setup.

## What changes

The patch targets `amulyaparthasarathy/unix-ai-workbench` at commit
`3e3951396bc23e05c4d95084e76ae6f5a60052c3` and changes three files:

- `scripts/prepare-host.sh` ensures the local `llm` Docker bridge exists. It
  checks Docker availability and rejects an existing network with an unexpected
  driver, scope, or internal-only setting.
- `.devcontainer/devcontainer.json` joins that bridge and sets
  `LOCAL_LLM_BASE_URL=http://llama:9931/v1`.
- `workbench` joins the bridge for online and interactive sessions. Its offline
  verification container retains `--network none` and receives no endpoint
  variable from the helper.

The Dockerfile, mounts, GPU settings, existing security settings, and the
inference Compose files remain as supplied in the reviewed repository. No
Docker socket or host directory is added. There is no provider fallback.

## Apply from your WSL host terminal

Use the ordinary WSL terminal where Docker Engine runs, outside the Dev
Container. The path below assumes the proposed independent lab checkout.
Adjust only `AGL_NETWORK_PATCH` if you extracted the package elsewhere.

```bash
AGL_NETWORK_PATCH="$HOME/src/unix-ai-workbench/projects/repos/agent-lightning-lab/integration/optional-workbench-llm.patch"
cd "$HOME/src/unix-ai-workbench"
git status --short
git diff -- .devcontainer/devcontainer.json scripts/prepare-host.sh workbench
```

Preserve any existing edits before proceeding. In particular, if these three
files already implement the network integration locally, inspect those changes
instead of applying the same integration again. A Git checkpoint or dated
backup preserves an edited starting point. Do not reset or discard work to
make this patch apply.

For a clean starting point, create a new branch and apply the patch:

```bash
git switch -c integrate/local-llm
git apply --check "$AGL_NETWORK_PATCH"
git apply "$AGL_NETWORK_PATCH"
git diff --check
git diff -- .devcontainer/devcontainer.json scripts/prepare-host.sh workbench
./workbench check
```

Run each command only after the preceding command succeeds. If the branch
already exists, choose a new branch name. If `git apply --check` fails, compare
your current files with the patch; do not force it or use a reset.

In the existing VS Code Workbench window, use **Dev Containers: Rebuild
Container** so that the new network and environment settings take effect.
The image definition is unchanged and requires no CUDA additions.

The inference container remains managed by your existing
`~/src/local-ai-stack` Compose files. With that service running, verify from
inside the recreated Workbench:

```bash
curl -fsS --max-time 5 "$LOCAL_LLM_BASE_URL/models"
```

This checks connectivity and model discovery only. It does not execute a tool
loop or perform training. Commit and push the reviewed Workbench change through
your normal workflow when you choose to retain it.

## Two laptops

On the Surface Pro the patch creates an empty local `llm` bridge. The Workbench
can start normally, but `llama` will not resolve there unless that machine has
its own inference service. An endpoint check failing on that machine does not
trigger a cloud request.

Docker network names are local to a Docker daemon. A bridge named `llm` on the
Pro does not connect it to the Ultra. The Ultra Workbench reaches its own
service at `http://llama:9931/v1`; its WSL host uses
`http://localhost:9931/v1`. Cross-machine access would be a separate explicit
networking task and is not configured by this patch.

## Preparation checks

The patch applied cleanly to temporary copies whose Git blob hashes matched
the pinned repository files. Both shell scripts passed `bash -n`; the JSON
parsed; existing mounts and build configuration were preserved. Scripted
fixtures checked all three helper modes and five network cases, including a
missing network, an existing bridge, an unexpected network type, an internal
network, and an unavailable Docker daemon.

These were local fixture checks. No actual Docker command, container build,
network change, Workbench rebuild, or model request was performed while
preparing the patch.
