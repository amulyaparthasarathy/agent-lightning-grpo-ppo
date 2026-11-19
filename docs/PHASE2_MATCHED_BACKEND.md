# Matched GRPO/PPO backend: offline implementation and execution gates

**Ten-way extension:** see [source-bound a02, campaign gates, admission probe and exact commands](PHASE2_TENWAY.md). The original two-job instructions below describe qualification/pilot; a01 is spent and must not be resubmitted.

The canonical specification is [PHASE2_MATCHED_STUDY_CONTRACT.md](PHASE2_MATCHED_STUDY_CONTRACT.md). The new executable protocol is `matched-rollout-mean-v2`, with algorithms `grpo-matched-v2` and `ppo-mc-matched-v2`. This implementation has CPU tests, **not live Stanford qualification**. No cloud job, Secret change, merge, or campaign was performed during implementation.

The original `modal_app.py` resource declarations, image build functions, PPO smoke restrictions, frozen base weights, and completed Stanford baseline are preserved. The new app adds source layers to the existing image recipe and requests no forced rebuild. No image build or replacement was executed during implementation; live reuse of Modal's existing CUDA-layer cache still needs confirmation. Historical GRPO remains a legacy comparator; the Stanford baseline remains 26/64. Do not run bootstrap preparation, a forced CUDA build, or the old `stanford-v2-base-01` evaluation again. The `baseline` stages below are required fresh controls under the new matched evaluation route, not reruns of that completed bootstrap job.

## Runtime and reductions

`experiment/matched_launch.py` builds the pinned Calc-X/VERL configuration and invokes actual training. `experiment/matched_runtime.py` is installed in every Ray process through `worker_process_setup_hook`. All overlaid upstream files are checked against `ppo_source_lock.json` and `matched_source_lock.json` before model construction. No installed upstream file is edited.

The pinned adapter retains original case, group, rollout, turn, generated-token, and fragment identities. Both arms collect eight questions and four real trajectories per question per iteration. All complete trajectories, including constant-reward groups, remain in the optimizer batch. A missing/nonbinary reward, missing trace, truncated response, invalid prompt, duplicate rollout, or incomplete four-trajectory group rejects the entire collection with an audit. No replacement sampling, reward-dependent dropping, floor-to-32 selection, padding trajectories, or transparent retry is allowed. Strict rejection may fail a live probe; those failures require inspection rather than automatic relaxation.

One collection produces exactly one actor optimizer call and, for PPO, one critic optimizer call. Each trajectory's generated-token losses sum with coefficient `1 / (real_trajectory_count * trajectory_generated_tokens)`. Observation and padding tokens have zero weight. The native clipped actor loss receives normalized advantages; the critic receives separate weight tensors and unchanged MC targets. Complete trajectories can be fragmented without changing clipping decisions, gradient accumulation, or optimizer-call count. The supported runtime topology is one H100 per job; concurrency two means two independent jobs.

GRPO computes its baseline/std from the four real rewards, never from repeated fragment rows. Constant groups have zero group-relative advantages. PPO uses terminal return `R` on every generated policy token, raw advantage `R - V`, and unbiased token-population whitening with epsilon `1e-8`. This is checked against pinned GAE at `gamma=lambda=1`. Critic predictions and predictive-utility diagnostics are recorded before updates, with an earlier-iteration constant predictor where available. Constant-target explained variance is undefined. The critic's rollout-mean training loss and token-population prediction diagnostics are distinct quantities.

Python, NumPy, Torch CPU/CUDA, vLLM engine sampling, optimizer initialization, row batching, and critic initialization have recorded independent streams. Both arms use the same separately seeded Torch data sampler for question order. Critic head construction uses a forked Torch RNG; optimizer initialization also owns a forked stream. `PYTHONHASHSEED` is set before subprocess/Ray startup. Ray request ordering, vLLM scheduling, and GPU kernels still prevent a bitwise reproducibility claim.

Every arm starts from the hash-verified prepared untrained Llama. Its stages are `baseline`, `training`, and `checkpoint-eval`, with disjoint output trees. Baseline and reload use the same frozen 64-case development panel and no updates. Reload requires the exact final step, algorithm, plan, configuration, role inventories, sizes, and SHA256s. Evaluation completion requires exactly one real completed, scored heldout-route episode per expected ID. Final and step-20 native saves are retained; there is no checkpoint deletion path. Campaign plans label legacy data exploratory; campaign launch requires the new holdout and operational gates.

## Observation and failure handling

W&B initializes inside the trainer's Ray process with explicit entity, project, group, stage ID/name, job type, `resume=never`, source/data/protocol hashes, seed streams, and resource settings. Every stage has `observability.json`, raw `metrics.jsonl`, separate trainer/controller/gateway logs, traces, and per-process audit files. Audits record original row maps, bounded token tensors, critic diagnostics, actual optimizer calls, pre/post-clipping norms, refresh events, and cgroup/process/GPU memory scope. Completion requires terminal global step, optimizer counts, refresh counts, native final checkpoint or complete evaluation, and `wandb.finish()`.

`matched_app.py` launches one detached CPU supervisor. The supervisor alone writes the aggregate index; each single-use GPU worker writes only its own stage. A durable Dict tracks attempt ownership, exact FunctionCall IDs, and worker heartbeats. A stage gets no retry. Missing startup heartbeats, lost heartbeats, absolute stage duration, supervisor duration, and owner stop requests are independent cancellation conditions. A lost spawn response is `unknown`, never resubmitted. The supervisor index and stage receipts remain on the working Volume.

`status` reconciles FunctionCall results, heartbeat age, Volume receipts/completion records, and remote W&B identity mappings. `collect` downloads evidence and explicitly reports missing/unavailable artifacts; native weight files remain on the Volume. Hosted W&B history is not independently queried, and the report says so. Local terminal metrics remain authoritative evidence if the hosted final row is absent. A supervisor that never starts has no aggregate index: `status` reports unknown and `stop-owned` can cancel its exact submitted call ID. It never stops an app by display name.

All resource bounds include initialization and checkpoint/receipt work. The owner supplies an **all-resource hourly ceiling** and dollar budget; the manifest conservatively reserves six stage lifetimes plus the supervisor lifetime at that ceiling. This is a runtime/resource bound and a cost estimate, not a provider billing lock or a capacity reservation. Status reports accumulated reserved-time cost. No campaign follows a qualification or pilot automatically.

## Local verification

The normal client suite deliberately has no Torch. Run both environments:

```bash
MODAL_CONFIG_PATH=/dev/null HF_HUB_OFFLINE=1 \
  uv run --locked python -m unittest discover -s tests

# First use only: downloads pinned code and CPU packages, never models.
scripts/test_matched_training.sh --prepare
# Subsequent verification is offline and refuses missing/drifted source/dependencies.
scripts/test_matched_training.sh
```

The CPU suite verifies actual pinned adapter, GAE, actor-loss dispatch, TensorDict minibatch iteration, FSDP backward accumulation, BaseEngine optimizer path, gradient clipping, and AdamW calls. It uses Torch 2.11.0 CPU, NumPy 1.26.4, TensorDict 0.10.0, and OmegaConf 2.3.0. A small differentiable CPU module replaces transformer construction; transport/telemetry seams replace Ray/CUDA infrastructure. Unequal fragment sizes and two-rank Gloo reductions are exercised. These tests do not qualify GPU memory, model loading, vLLM serving, hosted W&B, or Modal hydration.

## Separately authorized qualification and pilot

Do not execute launch commands until explicitly authorized. Commit the reviewed code first; executable launch rejects a dirty or changed source tree. The owner must set `STUDY_BUDGET_USD` and `ALL_RESOURCE_HOURLY_CEILING_USD`; no account price is assumed here. Use a new attempt ID after any uncertain or failed attempt.

Prepare the two-step new-protocol qualification plan offline:

```bash
uv run --locked python scripts/phase2_study.py plan \
  --phase qualification --study-id agl-matched-v2 --attempt 1 \
  --stage-seconds 3600 --startup-seconds 600 \
  --budget-usd "$STUDY_BUDGET_USD" \
  --all-resource-hourly-ceiling-usd "$ALL_RESOURCE_HOURLY_CEILING_USD" \
  --data-manifest .local/stanford-stage-v2/data-small/data_manifest.json \
  --plan .local/phase2/matched-qualification.json
```

Review the manifest and its printed authorization token. A separately approved qualification launches only this bounded pair:

```bash
MODAL_PROFILE=amulyasp uv run --locked python scripts/phase2_study.py launch \
  --plan .local/phase2/matched-qualification.json \
  --authorize '<reviewed qualification study/attempt:plan-sha256>' \
  --out .local/phase2/matched-qualification-submission.json
```

Use the same interface for observation, collection, or targeted cleanup:

```bash
MODAL_PROFILE=amulyasp uv run --locked python scripts/phase2_study.py status \
  --plan .local/phase2/matched-qualification.json
MODAL_PROFILE=amulyasp uv run --locked python scripts/phase2_study.py collect \
  --plan .local/phase2/matched-qualification.json --out .local/phase2/qualified-evidence
MODAL_PROFILE=amulyasp uv run --locked python scripts/phase2_study.py stop-owned \
  --plan .local/phase2/matched-qualification.json
```

After reviewing successful live qualification, prepare a **fresh 20-step pair**, seed 9001, offline. This bound allows at most two concurrent H100s, six stage calls, 24 cumulative GPU-hours, a 12-hour-10-minute supervisor, and no retries. Choose a budget covering the conservative bound before generating the plan:

```bash
uv run --locked python scripts/phase2_study.py plan \
  --phase pilot --study-id agl-matched-v2 --attempt 1 \
  --stage-seconds 14400 --startup-seconds 600 \
  --budget-usd "$STUDY_BUDGET_USD" \
  --all-resource-hourly-ceiling-usd "$ALL_RESOURCE_HOURLY_CEILING_USD" \
  --data-manifest .local/stanford-stage-v2/data-small/data_manifest.json \
  --plan .local/phase2/matched-pilot.json
uv run --locked python scripts/phase2_study.py validate --plan .local/phase2/matched-pilot.json
```

**Bounded pilot launch command, requiring separate owner authorization:**

```bash
MODAL_PROFILE=amulyasp uv run --locked python scripts/phase2_study.py launch \
  --plan .local/phase2/matched-pilot.json \
  --qualification .local/phase2/qualified-evidence/index.json \
  --authorize '<reviewed pilot study/attempt:plan-sha256>' \
  --out .local/phase2/matched-pilot-submission.json
```

The pilot gate also checks the qualification receipt against the durable remote supervisor index and requires identical source/data. CLI exit proves submission only. Inspect `status` and `collect` for completion. No live qualification or pilot command above was executed during implementation.

After terminal qualification, `collect` must return exit code zero. Check both files before treating `qualified-evidence/index.json` as qualification proof:

```bash
uv run --locked python - <<'PY'
import json
from pathlib import Path
root = Path('.local/phase2/qualified-evidence')
assert json.loads((root/'index.json').read_text())['status'] == 'completed'
assert json.loads((root/'collection.json').read_text())['evidence_complete'] is True
print('Qualification terminal receipts and collected evidence agree.')
PY
```

Collect only into a new output directory. An early collection can contain incomplete evidence and must not be reused as a successful qualification. There are six stage calls in the minimal supported qualification: two new-protocol baselines, two two-step training runs, and two native checkpoint reloads. No smaller qualifying mode exists in this implementation.

Pilot terminal collection and targeted shutdown use its own plan:

```bash
MODAL_PROFILE=amulyasp uv run --locked python scripts/phase2_study.py status \
  --plan .local/phase2/matched-pilot.json
MODAL_PROFILE=amulyasp uv run --locked python scripts/phase2_study.py collect \
  --plan .local/phase2/matched-pilot.json --out .local/phase2/pilot-evidence
# Only when stopping this attempt is intended:
MODAL_PROFILE=amulyasp uv run --locked python scripts/phase2_study.py stop-owned \
  --plan .local/phase2/matched-pilot.json
```

## Finalization review: ten-GPU campaign remains unsupported

The [latest owner concurrency requirement](https://github.com/amulyaparthasarathy/agent-lightning-lab/pull/24#issuecomment-6886443424) requires configurable concurrency 1..10 and one launch/supervisor for five GRPO plus five PPO jobs. This review finalizes the existing work without changing its runtime design. **There is no valid ten-GPU campaign launch command in this commit.** Do not substitute a loop of pilot or legacy smoke launches.

Concrete implementation blockers:

The isolated ten-way extension now implements campaign planning, configurable
1..10 scheduling, thirty stage receipts and a separately authorized admission
probe. [The ten-way runbook](PHASE2_TENWAY.md) supersedes the original implementation
gap list. Execution remains blocked by the new holdout, same-source live
qualification/pilot, measured admission, current headroom and owner authorization.
