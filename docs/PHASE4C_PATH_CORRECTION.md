# Phase 4C GRPO qualification correction and successor handoff

## The failed allocation

A01 remains closed. On November 25, 2026, the live baseline produced 24 diagnostic development episodes with 24 confirmed rewards, including 9 strict successes, 12 semantic failures and 3 evidence failures. Worker finalization failed before a completion receipt. The supervisor skipped probe, training and reload. Qualification admitted **0 episodes, 0 actor updates and 0 checkpoints**. Replaying the private terminal collection with the failed execution source produced a byte-identical independent report (SHA-256 `dce541a236f007923b0228a80cbf73e80d726410638194c5c0703ca94c3eabfb`). [Failed-attempt report](PHASE4C_GRPO_QUALIFICATION_A01.md) · [Merged evidence PR #8](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/pull/8)

| Source | Commit |
|---|---|
| Historical Phase 4B execution | `bc81c1fb0def27b15c56cdd8abb01098e7e1f8d5` |
| Tested runtime ancestor | `852e01d141b66232c73ca46e6e7bf4b9dfb0e854` |
| Failed A01 execution | `547053db6636fdc1475f0d53f8bf580fed751bbd` |
| Reviewed A01 evidence | `17fe351663d2203f2841902a67b2f1142c6c9c81` |
| Merged closeout | `f303fc143a815ade56bb62ce86b65aeaa35e44a5` |
| Corrected runtime proposed for a successor | `d66b5d112587f36ca1a2708577d9218f7db39e29` |

At 18:15 UTC on November 25, a read-only lookup of the closed 17:00–18:00 UTC interval confirmed **$0.62502184 before credits**. Three apps had stopped and zero owned containers remained. The original four calls were terminal. The ledger keeps $112.37497816 earmarked, including $7 for storage and $2 for delayed billing. Final invoicing and subsequent storage remain separate. A01 retained 6,445,405,176 regular-file bytes and has a December 2, 2026 UTC retention review. Phase 4B has its own holds. [Dated reconciliation](../evidence/phase4c-grpo-qualification-20261125/closed-hour-20261125.json) · [Original snapshot](../evidence/phase4c-grpo-qualification-20261125/exposure-and-retention.json)

## Why the old check failed

`reader_assets.owned_path` resolves the logical mount before `matched_remote.stage_worker` writes the request. The launcher takes its run directory from that request file's parent. The trainer therefore persisted a canonical absolute checkpoint directory and canonical Ray request, run and audit paths. On the same worker, `complete_stage` called `stage_report`, which rebuilt an expected run path by joining the literal `/data` to the stage output. Exact validation stopped at `trainer.default_local_dir`. The former fixture supplied the logical directory directly during configuration composition, then reviewed files in a separate temporary directory. It never used the worker's resolved request parent to produce the configuration.

The historical reproducer creates a logical symlink to a physical directory, passes the real owned run path through the content-bound request transport and launcher, then calls production completion. On the failed commit, launcher validation succeeded and completion raised `ValueError: Matched resolved configuration drift: trainer.default_local_dir`. To reproduce, set `FAILED` to a clean checkout of the failed source and run the following from the fix checkout. The temporary script picks up both runtime and test helpers from the failed checkout.

```bash
repro=$(mktemp /tmp/phase4c-repro-XXXXXX.py)
cp tests/reproduce_phase4c_a01.py "$repro"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$FAILED:$FAILED/tests" python "$repro"
```

## What now crosses the worker boundary

CPU preflight observes the mounted root and hydrated volume ID. For every stage it constructs exact paths from the validated plan, job and stage, composes the pinned Calc-X/Hydra configuration, and runs the same resolved-config checks used at training. Preflight only projects future stage paths. At worker entry the observed mount, full paths, plan and request digests must match the successful preflight before GPU checks or services. The worker writes the binding, then loads the canonical, content-hashed request. The probe routes both eager inputs to its bound stage-local Parquet file.

Completion validates the original resolved configuration. The supervisor records the preflight digest and volume ID. Each stage receipt commits its binding and report. Independent review can therefore reconstruct the producer's absolute path strings using the observed root and frozen relative locations after collection to another machine. Training binds the native checkpoint inventory to the original resolved-config hash and training binding. Reload verifies the declared sibling's terminal step, payload and inventory hashes before and after model work. Completion and review require distinct training and reload call and container identities.

The telemetry helper joins the emitting process's controller membership to its mount table, including subtree mounts. It reads v1 memory usage/high-water counters or v2 current/peak counters and records availability. A malformed or absent reading stays unavailable. CUDA allocated/reserved peaks remain scoped to the emitting process's allocator.

## Offline verification

The tests below used `d66b5d112587f36ca1a2708577d9218f7db39e29`. The reader and matched scripts used installed, locked CPU dependencies and cached pinned sources. The evidence tests used the locked reader-harness interpreter, loopback reward transport and scripted model responses. Provider identities, GPU checks, optimizer events and native checkpoint bytes were simulated. The four-stage flow used the real stage worker, canonical request-file launcher, recovery harness, stage completion, durable receipt construction and independent reviewer. Each scenario copied evidence elsewhere and removed the provider-mount fixture before review.

| Gate | Passed | Failed | Skipped |
|---|---:|---:|---:|
| Reader offline, three unittest groups | 39 + 15 + 148 = 202 | 0 | 0 |
| Pinned matched training gate | 73 outer tests, including a 9-test Hydra subprocess | 0 | 0 |
| Focused path, cgroup and request transport (also in reader gate) | 11 | 0 | 0 |
| Four end-to-end scenarios, both algorithms × both mount layouts | 4 | 0 | 0 |

Each end-to-end scenario exercised the 24/32/64/24 schedule, distinct simulated training/reload identities, the native inventory and hash comparison, and rejection of independently modified collected artifacts. All 576 episode slots in this matrix used scripted output.

```bash
bash scripts/test_reader_offline.sh
OMP_NUM_THREADS=1 bash scripts/test_matched_training.sh
PYTHONPATH=tests/offline_guard:.:tests python -m unittest \
  test_qualification_paths test_cgroup_memory test_request_transport -v
# Run each variant in its own credential-free local harness process.
for variant in grpo_plain grpo_aliased ppo_plain ppo_aliased; do
  env -i PATH="$PATH" HOME="$(mktemp -d)" MODAL_CONFIG_PATH=/dev/null HF_HUB_OFFLINE=1 \
    OMP_NUM_THREADS=1 PYTHONPATH="$PWD:$PWD/tests" "$TASK_TEST_PYTHON" -m unittest \
    "test_recovery_learning_evidence.RecoveryCollectionTests.test_${variant}_volume" -v
done
```

The pinned gate used `MATCHED_TEST_PYTHON`, `AGL_PINNED_SOURCES`, `AGL_TRANSFORMERS_WHEELS` and `AGL_VLLM_WHEEL` for existing local dependencies. Rejection tests change the root, volume, request/plan identity, stage-relative paths, Ray paths, eager inputs, base model, checkpoint step or inventory. They also exercise absolute and traversal substitutions, nested symlinks, altered configuration and missing evidence. Historical calculator, asset-status, zero-update recovery calibration, request transport and native checkpoint checks ran in the reader and matched regression suites.

## Successor allocation proposal

Request a **separate $113 before-credits ceiling**, with zero assumed credits and zero retries, for one GRPO qualification on the corrected runtime above. The private successor proposal passed offline plan generation and validation with the owner profile's unchanged placement, task, model and budget settings, a new experiment identity, proposed app/registry/volume names and disjoint paths. Its preparation-plan digest is `b3960194aa2f09bcc3b371261a64ed37b577b6d3bc4020632778cb458b29e0bf`. Ownership, name availability, prices, capacity and account headroom require verification at execution time. No resources or funds were reserved for it.

1. Upon separate authorization, use the frozen checkout for all operations on its plans. Check the exact authenticated workspace/environment, handles, spend headroom and component ceilings. Open a separate ledger and reserve $113 before dispatch. The recovery contract requires integer `attempt == 1` within the **new experiment**. Distinct experiment and allocation identities preserve A01's terminal registry keys, stage markers and W&B IDs.
2. Use the supported setup once. It verifies the existing secret handle and creates both a new volume and a new registry with `allow_existing=False`. A collision or partially successful setup requires inspection. Preserve existing resources and stop dependent dispatch. Preparation checks that all model, dataset and staging paths are absent. The current interface has no verified cross-volume import or prior-volume attachment route for immutable assets. Budget for one fresh preparation on the new volume. Historical manifests supply expected content. Only the new terminal manifest and on-volume verification establish the successor's assets.
3. Bind an execution plan to the actual new manifest, record its digest, and require the durable, same-digest CPU preflight. The existing submission reservations, call acknowledgments, restart guards, deadlines, dependency gates and original admission rules apply. Preserve failed and unknown outcomes and their exposure.
4. Run the unchanged `grpo-matched-v2` qualification for `recovery-readiness-v1` with seed 9001 and its frozen streams. Baseline runs 24 greedy development episodes, the probe samples 32 training episodes, fresh-base training samples 64 episodes across two actor updates, and a fresh worker evaluates the terminal step-2 checkpoint on 24 greedy development episodes. Total 144, with zero final-panel evaluations. Use one H100, 16 CPUs, 192 GiB host memory, concurrency one, four 3,600-second stages, 14,400 worker seconds and 15,000 supervisor seconds.
5. Collect and independently review complete or partial evidence. Reconcile exact calls and gross usage, retain the native inventory and evidence, and set a seven-day review from the new retention start. Check projected volume bytes and free worker scratch separately. A failed gate ends dependent execution.

| Component | Proposed ceiling | Basis |
|---|---:|---|
| Compute and supervision | $82 | (14,400 + 15,000) seconds at the contract's $10/h reservation rate = $81.67 |
| Build | $15 | Allow for a cold build |
| Preparation and CPU preflight | $5 | One fresh preparation up to 3,600 s and one same-plan preflight up to 900 s |
| Storage | $7 | Up to 300 GiB for seven days, $6.30 at the worksheet's dated rate |
| Collection | $2 | Up to 2 GiB |
| Delayed billing | $2 | Separate hold through reconciliation |
| **Total** | **$113** | **New authorization required** |

A new live run would be needed to measure training GPU use, process/cgroup peaks, host offload traffic, log-probability recomputation, update/refresh latency, simultaneous scratch demand, checkpoint bytes/write latency and fresh-worker reload time. The historical baseline's largest sampled GPU reading was 53.64 GiB and its late cgroup reading was 43.42 GiB. Those samples bound neither training nor reload. The offline tests establish software behavior only. The two-update qualification and longer study remain unqualified. [Original worksheet A and storage assumptions](PHASE4C_LEARNING_STUDY_PROPOSAL.md)
