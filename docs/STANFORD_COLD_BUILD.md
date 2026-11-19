# Stanford cold image construction failure

The owner reported that Stanford image **`im-fc2RdjhSAGjfhNX3zuDS5W`** failed
during `Image.run_function(build_training_stack)`, before the CUDA/VERL installer
executed. Modal imported `/root/modal_app.py`, which imported
`experiment.ppo_contract`, and raised `ModuleNotFoundError: No module named
'experiment'`. This report is retained as owner-supplied evidence; no remote
logs, image or Volume were accessed for this fix.

The fix starts from merged `main` at
`a8893264fc5a16ad62b7107b445932935d6d4b3f` (PR #22), not the old PPO branch.
The previous final-image import correction remains necessary: it finds the
packaged contract under `/opt/agent-lightning-lab`. It cannot help construction
of an earlier image layer where that package has not been copied yet. The
[previous runtime failure](../evidence/ppo-smoke-01-import-failure-2026-11-14.json)
is a separate incident and is retained unchanged.

`modal_build.py` now defines both build functions. The recipe explicitly copies
it to `/root/modal_build.py` before either `run_function`, using `copy=True`.
Both builders use `include_source=False` because the source is already present.
Modal discovers the functions in `modal_build`, independently of `modal_app`.
The module imports only stdlib `subprocess`; its functions invoke the existing
shell scripts with `check=True`. It has no application/PPO imports, decorators,
Volume or Secret objects, or indirect runtime dependencies.
Submission source hashes and retained source copies now include the build module.

The CUDA digest, Python 3.12, uv 0.12.13, Agent Lightning commit, VERL,
FlashAttention build settings, installation scripts, requirements, lockfile,
H100 settings, host-memory limits (including PPO's 196608 MiB), and model
identities are unchanged. The builders retain 8 CPU / 65536 MiB / 7200 seconds
and 4 CPU / 16384 MiB / 1800 seconds respectively.

## Cache implications

Modal caches image layers and changes propagate to subsequent layers.
`copy=True` makes the module part of the built image before execution.
See [Modal's image and caching documentation](https://modal.com/docs/guide/images#image-caching-and-rebuilds).

- The existing CUDA/Python, apt and uv layers are unchanged and eligible for
  reuse in the same workspace/builder configuration; availability is not proven.
- The new module-copy layer changes the parent of the installer copy and heavy
  build. Budget for rebuilding CUDA/VERL/FlashAttention and all subsequent layers.
  The failed image ID does not establish a successful cached heavy layer.
- Once this recipe succeeds, runtime/PPO file edits remain after both builders.
  Agent requirements/finalizer shell edits remain after the heavy builder.
- Editing either function in the shared `modal_build.py` changes its early copy
  layer and can rebuild both stages. Keep that module stable. A workspace or
  image-builder change can also prevent reuse; no cache settings were changed.

## Independent staging blocker

The owner reports regular files at `restored/models/llama3b` and
`restored/data-small` on the current Stanford Volume, where directories are
required. This followed a Modal CLI directory download to a nonexistent local
destination. The Volume was not inspected, repaired, overwritten or deleted.

The PPO worker now explicitly requires both directories, then regular-file
`model_manifest.json` and `data_manifest.json` entries. Missing/wrong-type paths
fail before `nvidia-smi`, GPU checks or launch, with no fallback to legacy root
inputs and no download. This is a structural gate; downstream launch validation
still checks manifest contents and model/data identities. Volume input repair
and identity verification remain separately authorized work, even if the image
build succeeds.

## Offline evidence

`tests/test_modal_build_import.py` exercises the installed Modal SDK's
`FunctionSourceInfo` and container-side `import_single_function_service` for
both actual builder functions. Isolated subprocesses contain only the build
module, with no `modal_app` or `experiment` package. Network access and runtime
imports are denied, Volume/Secret lookup is trapped, and shell execution is
mocked. It checks exact installer dispatch and propagation of installer failure.
The recipe test checks source availability, ordering and unchanged resources.

The existing final-runtime module import regression is retained, with the build
module included as it is in the final image. Staging tests cover missing and
wrong-type directories/manifests, unchanged fixture files, no fallback/no
commands on rejection, and an accepted structural fixture with offline flags.
See [the test/evidence record](../evidence/stanford-cold-build-offline.json).
The same SDK discovery against the base commit failed for both builder names
with the reported missing-`experiment` exception before any installer ran.

## PR #23 documentation closeout — 2026-11-14

The code fix and offline verification are complete in implementation commit
[`66a752804c946a832de983236a01b893affe11e3`](https://github.com/amulyaparthasarathy/agent-lightning-lab/commit/66a752804c946a832de983236a01b893affe11e3).
[PR #23](https://github.com/amulyaparthasarathy/agent-lightning-lab/pull/23)
remains open as a draft. This closeout changes only this document; it does not
change the implementation, test fixtures, pinned inputs or retained evidence.

The implementation's full offline suite passed **192 tests in 8.140 seconds**
on Python 3.12.3 / Modal 1.6.3 / Linux aarch64. The exact command and eight tested
source hashes are in the [evidence record](../evidence/stanford-cold-build-offline.json).
The [retained test log](../evidence/stanford-cold-build-unittest.txt) has SHA-256
`0d80099655d054d3f66f0180eb795e1ca3b34658e05477bb37bebae342f8830b`.
For this documentation closeout, those source and log hashes were rechecked,
along with the 25 protected-file hashes and local document links;
`git diff --check` passed. The test suite was not rerun for this prose-only edit.

Cold construction and dependency/GPU/runtime qualification remain unperformed.
The owner-reported Stanford Volume staging blocker also remains unresolved:
`restored/models/llama3b` and `restored/data-small` are reported regular files,
and their repair plus manifest/identity verification need separate authorization.
Failed image `im-fc2RdjhSAGjfhNX3zuDS5W` remains the incident reference; this
closeout establishes no replacement image or successful training result.

No Modal jobs, image builds, GPU executions, infrastructure changes, Volume
access or repairs, file deletions, merges or training were performed for this
closeout. The single proposed qualification command below remains unexecuted
and subject to separate approval. Build success would resolve only the image
construction gate; it would not resolve the independent staging blocker.

## Proposed cold-build qualification — separate approval required

This command has **not** been run. From the reviewed commit in a provisioned
Linux checkout, replace `YOUR_STANFORD_PROFILE` with the approved profile. It
builds this exact image under an otherwise empty ephemeral app: no runtime
function invocation, GPU allocation, Volume/Secret resolution, model download
or training. Importing `modal_app` only constructs its lazy declarations; its
app is not run. Preserve terminal output and the printed app/image IDs.

```bash
timeout --signal=INT --kill-after=60s 3h env \
  MODAL_PROFILE=YOUR_STANFORD_PROFILE MODAL_IGNORE_CACHE=1 MODAL_FORCE_BUILD=0 \
  uv run --locked --offline python - <<'PY'
import modal
from modal_app import image

build_app = modal.App("stanford-cold-image-qualification")
with modal.enable_output(), build_app.run():
    print(f"qualification_app_id={build_app.app_id}", flush=True)
    image.build(build_app)
    print(f"qualification_image_id={image.object_id}", flush=True)
PY
```

`MODAL_IGNORE_CACHE=1` requests a genuinely cold build without replacing shared
cache entries; `MODAL_FORCE_BUILD=0` avoids that broader cache invalidation.
The heavy and finalizer steps retain their 2-hour and 30-minute server limits.
The outer 3-hour timeout bounds the local client, with a 60-second shutdown
grace; it is not proof of remote cancellation. On interruption, verify the app
has stopped before considering any retry. There is no automatic retry command.
`uv --offline` covers local client resolution only; the remote installers still
download their pinned sources/dependencies. Build success establishes image
construction only. Dependency audit, GPU/runtime qualification and restored
input validation remain separate gates, with no authorization implied here.
