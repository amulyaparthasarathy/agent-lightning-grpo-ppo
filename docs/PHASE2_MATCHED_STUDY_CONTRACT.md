# Phase 2: matched GRPO versus PPO-MC study, execution contract

**Status (2026-11-14):** PR #24 adds the `matched-rollout-mean-v2` backend and offline CPU/runtime-dispatch tests. **Live Stanford qualification and cloud execution approval remain separate; no pilot or campaign was launched.** The ordinary GRPO recipe and PPO two-step smoke retain their historical behavior. This document remains the canonical implementation contract; [backend operation and bounded commands](PHASE2_MATCHED_BACKEND.md) describe the new path and its remaining live gates.

## Canonical completed evidence

- Repository baseline for this design: `main` after PR #23, commit `0d84e57e2f72573a2b57fd3526f2e9d823cfb6fb` (before adding the Stanford baseline documentation).
- Historical GRPO `train-01`: all 20 iterations completed, native step-20 checkpoint, independent reload 35/64. W&B missed the final hosted row; the recorded step-20 trainer metrics remain valid. Never call this a 19-step crash; never treat historical GRPO under legacy processing as a matched comparator.
- PPO `ppo-mc-v1-smoke-02`: two-step actor/critic training, exit 0, native actor/critic checkpoint. Separate `ppo-mc-v1-reload-01`: exit 0, restored `global_step_2`, no new checkpoint. This establishes plumbing only, not learned critic quality or a fair 20-step PPO recipe.
- Stanford `amulyasp` / environment `main`: correctly staged `cs329z-agent-lightning-working-stanford-v2`; Secret `agent-lightning-credentials` exists. Cold image `im-qPPicZgz64ekqF9AR9UKz1` built successfully; `stanford-v2-base-01` subsequently evaluated 64/64, **26/64 correct (40.625%)**, no checkpoints. See [Stanford baseline receipt](PHASE2_STANFORD_BASELINE_2026-11-14.md) and [build closeout](STANFORD_COLD_BUILD.md). The historical `base-01` was 27/64, not a replacement result.
- No cloud job, model download, training, merge, or billed infrastructure action is authorized by this contract.

## Research objective and milestones

**Question:** How do GRPO's within-question, four-sample group-relative advantages compare with PPO-MC's token-level learned critic baseline for the same tool-using Llama 3.2 3B agent? Compare verified task completion, learning signal availability, critic predictive utility, tool/format failures, optimizer behavior, and cost. Do not infer a causal algorithm effect from unlike processing protocols or one unpaired historical run.

| Milestone | Scope | Acceptance gate |
|---|---|---|
| B0 implementation | Offline code/fixtures, staged data hashes, bounded launcher, observer, W&B propagation, new algorithm version | Tests pass; no execution; reviewer checks changes |
| B1 live integration | One *separately approved* tiny new-protocol probe per algorithm, without redefining old smoke | Correct row/group/mask/loss/seed/W&B/complete receipt; no retries |
| B2 development pilot | One **fresh pair** (run seed 9001), 20 steps GRPO and 20 steps PPO-MC, with a no-update baseline and native final reload for each | Both algorithms start from identical prepared base; verified case ID panels, checkpoint identity, optimizer/refresh, safety and cost |
| B3 campaign (separate authorization) | Five *new* matched seeds 1001..1005 per arm, 60 steps, 10 total fresh-base training jobs | New frozen untouched 256-case holdout, pilot review, storage/compute capacity, concurrency/lifetime qualification |
| B4 analysis | Paired seed/case-aware report including **all failures** | Model and task outcomes independently checked; limitations explicit |

The pilot and the campaign must **never chain automatically**. Development seeds 9001 and campaign seeds 1001..1005 are disjoint. Five pairs are exploratory; a perfect-direction two-sided five-pair sign-flip test has minimum p=0.0625. Multiple reused evaluations of the same 256 questions are not 1,280 independent cases.

## Common rollout/optimizer contract (mandatory before B1)

1. Same frozen model revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`, template, 256 training cases, binary scorer, agent tools, vLLM parser, inference temperature/settings, sequence limits, optimizer LR `1e-6` for actor, batch schedule and hardware; freeze source hashes. The existing 64 and 8-case panels are development only.
2. Keep **all valid real collected rollouts**, including constant-reward groups; never drop rows based on rewards to floor to a magic batch multiple. Define failure/truncation/exclusion rules and preserve original question/group IDs, trace lengths, fragment maps, and denominator accounting. Four different rollouts per GRPO group means four actual sampled agent trajectories, **not** four fragments.
3. Actor clipping, normalization, token masks and optimizer update count must be invariant to fragmentation (at fixed complete trajectories), to the maximum supported by the pinned worker. Average per real rollout's generated-token loss, not tool-output tokens or fragment count. Explicitly test distributed/microbatch loss reduction with uneven fragment sizes and complete worker optimizer path, not a separate idealized equation only. If padding is unavoidable, pad contributes zero to losses, group baselines, whitening, token counts and optimizer-step count.
4. For PPO-MC, set `gamma=lambda=1`; Monte Carlo terminal reward R is the return for every preceding policy action. Train the learned value head with recorded **pre-update** predictions/targets and the generated-token mask. Distinguish the prewhitening raw advantage `R-V(s_t)` from whitened policy advantage; critic targets and loss normalization are separate from actor normalization. Record the exact optimizer/evaluation iteration and value-head initial-seed provenance. Constant targets make explained variance undefined/N/A; positive critic gradients alone do not prove useful prediction.
5. For GRPO, advantage is calculated from the four real trajectories' rewards within each question. Preserve constant groups with zero estimated group-relative signal; do not inject successful samples to force variance. Report informative/constant groups as diagnostics without changing sampling or LR in response.
6. Record raw token-level masks/logprobs/rewards/advantages for a bounded subset, pre-/post-clipping gradient norm, policy KL/clip fraction, critic MSE and earlier-data constant-predictor comparison, actual optimizer calls, checkpoint/reload, weight-refresh events, memory (including cgroup-vs-system distinction), and wall-clock/cost.

## Seeds, baseline and evaluation

- Frozen dataset-selection seed **329** is not an RL run seed. Derive separate independent run streams for Python/NumPy/Torch CPU/CUDA, inference sampling, actor optimizer, critic/head initialization, data/order and row batching. Fix a **paired question-order schedule** for GRPO and PPO at a given run seed without conflating actor and critic randomness. Record resolved RNG streams and what vLLM/Ray cannot fully control.
- Each pilot arm starts independently from the prepared **untrained Llama** weights, not from `train-01` or the PPO smoke checkpoint. A no-update baseline under the *new exact evaluation route* is included per job/paired context; `stanford-v2-base-01` is an operational reference and not a substitute where protocols differ. No need to repeat the unchanged bootstrap evaluation merely for convenience.
- Before campaign runs, reserve **new** authorized 256-case untouched final evaluation panel, by ID and normalized-question deduplication, with a source/data manifest, exact selection rule, frozen hashes, near-duplicate/template contamination review, and independent grading rubric. Do not invent those 256 cases or use the historical 64 as final test.
- Predeclare final-step endpoint. No best-of-seeds or checkpoint cherry-picking; independent reloaded checkpoint evaluation, case wins/losses/ties, exact denominator, tool success vs task success, format/extraction errors, ambiguity labels. Preserve nondeterministic greedy case flips.

## Parallel Modal architecture and cost controls

- Target **Stanford** workspace `amulyasp` environment `main`, Volume `cs329z-agent-lightning-working-stanford-v2`, one `H100!` and 16 CPU cores and **196608 MiB** host RAM for **both** matched arms. Preserve existing historical GRPO defaults and existing PPO smoke restrictions outside the newly versioned study path.
- Pilot concurrency **2**; campaign concurrency must be configurable **1..10**, targeting **10 independent single-H100 jobs under one submission and supervisor**, per the [owner update in PR #24](https://github.com/amulyaparthasarathy/agent-lightning-lab/pull/24#issuecomment-6886443424), which supersedes the original proposed limit of four. Admission remains contingent on Stanford quota, remaining slots, H100 availability, and cost/storage bounds; unsupported admission must fail closed. **The isolated ten-way extension implements this scheduler and its offline simulations; actual ten-GPU capacity remains unproven. See [the ten-way runbook](PHASE2_TENWAY.md).** This is not a proven account capacity reservation. Require owner-chosen budget/cost ceiling, cumulative resource bound, max runtime including GPU initialization/validation/checkpoint commits, and independent stop conditions. Model/data read-only; all writing disjoint under `phase2/studies/<study-id>/<phase>/<job-id>/<stage>/`. Each worker has own Ray/temporary/cache directories. Preserve source/failed image/historical checkpoint Volumes.
- Approximate **additional** role-file storage if all 10 campaign jobs keep two full checkpoints: ~**1.1 TB** for two checkpoints, or **1,660.5 GiB** for the implemented three-checkpoint (20/40/60) schedule, using observed ~73.31 GiB PPO (actor+critic) and ~37.39 GiB GRPO (actor only) per checkpoint. This excludes new datasets, logs, artifacts and transients and is not a storage-quota assertion. Design bounded new-run-only retention, protect final/step-20 checkpoints, and measure actual bytes before starting all jobs.
- One durable remote supervisor, explicit per-job FunctionCall IDs, per-stage receipts, exact failure states and W&B mapping. *Detached local CLI exit is not job completion.* Startup import crash loops can occur despite `retries=0`; observe server-side missing heartbeats, bound absolute attempt time, and cancel **only owned** failed calls. Never stop an app by the shared display name, delete model Volumes, or transparently retry an attempt. An uncertain attempt requires investigation and a **new attempt ID** before any rerun.

## W&B identity, paths, completion proof

- Project `cs329z-agent-lightning`, entity `amulyap-stanford-university`. Shared group `<study-id>-<pilot|campaign>`; one job name `<study>-<phase>-<algorithm>-s<seed>-aNN`; unique **per-stage** W&B id/name, job_type `baseline|training|checkpoint-eval`, `resume=never`; explicit pair_id, dataset/commit and protocol hashes, full seed streams and resource settings in remote W&B config.
- Set W&B `group`, `id`, `name`, `job_type` **in the remote process that actually initializes W&B**, not only in the user's shell or an unrelated local process. Store immutable local `observability.json`, raw `metrics.jsonl`, stage receipts and unsampled metrics alongside independent `logs/trainer.log`, `logs/controller.log`, `logs/gateway.log`, traces, checkpoints and PPO audit. Check final global_step/optimizer count and `wandb.finish` or equivalent flush; preserve local terminal metrics if hosted history omits the final row (historical train-01 precedent).
- Status lifecycle: `planned -> submitted -> started -> completed|failed|cancelled|unknown`. No false success on CLI exit, missing receipt or incomplete cases. Supervisor is sole aggregate-index writer; worker writes its own tree. A per-study `status` and `collect` tool must reconcile actual Modal state, Volumes, W&B mappings, errors and incomplete/unavailable evidence.

## Implementation and approval gates

PR B must deliver:
- One stable **launch/status/collect/stop-owned** command whose `plan` and `validate` modes are offline and whose **billable** launch mode is separately explicit, bounded, and user authorized. No shell loop that merely invokes the existing two-step PPO smoke concurrently.
- Frozen versioned algorithm/retention/loss and evaluation contracts, clean source identity; tests of actual pinned adapter reductions, incomplete/constant-reward groups, unequal fragments, seed wiring and no dependency on critic RNG for paired actor schedule, trainer W&B remote config, parallel collision/ownership, startup hydration failure/disconnect and targeted stop, immutable checkpoint selection/reload, data/test leakage checks, and resource timeouts.
- One later, separately approved, bounded Stanford live qualification before **any** 20-step pair; test CPU image packaging and true H100/VERL run in current account, exact source and case manifests, and failure cleanup with no extra retries.

The included `scripts/phase2_study_plan.py` only renders a **planning-only, nonlaunchable** pilot or campaign manifest. Those proposal manifests remain contract fixtures and are not accepted for execution. The separate `scripts/phase2_study.py` interface wires seeds and remote W&B identities for versioned qualification/pilot/campaign manifests. Generating these JSON files causes no Modal activity:

```bash
uv run --locked python scripts/phase2_study_plan.py --phase pilot --out .local/phase2/pilot-plan.json
uv run --locked python scripts/phase2_study_plan.py --phase campaign --out .local/phase2/campaign-plan.json
```

Do not submit these planning-only manifests. The new pilot launcher requires a clean reviewed source tree, completed live qualification for the same source/data, owner-selected bounds, and separate authorization for the exact executable plan. Campaign execution remains disabled pending its new untouched holdout and separate review.
