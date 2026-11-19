# Phase 2 PR A: PPO-MC smoke contract and decision record

Date: 2026-11-14. Status: **offline implementation; runtime unqualified; no execution authorized**.
This implements the bounded part of the owner's reviewed handoff v2. It does not make a
matched pilot or multiseed study launchable. Earlier handoffs, Phase 1 evidence, failed
attempts, checkpoint/export records, and the preservation workflow remain unchanged.

## Decision and source identity

Use `--algorithm ppo-mc-v1`, protocol `ppo-mc-smoke-legacy-retention-v1`:

> PPO clipped actor objective with Monte Carlo terminal returns and a learned token-level value baseline; implemented through VERL GAE with gamma=lambda=1.

The default remains `grpo`, with the existing upstream GRPO settings and command shape.
PPO permits only `smoke` and `smoke-reload-eval`, the prepared Llama profile, and fresh
`ppo-mc-v1-…` names. No free-form config override, alternate model, warm-up, reward shaping,
LR multiplier, automatic chaining, fallback hardware, or retry policy is added.

PPO smoke and independent smoke reload request one `H100!`, 16 physical CPU cores,
and 192 GiB host RAM (`memory=196608` MiB) via the pinned Modal client's
`Function.with_options`; legacy GRPO keeps its 128 GiB request. Only memory is
overridden; existing timeouts, retry policy, image, mount, model and template stay fixed.
The client accepts this resource shape; server admission and PPO memory fit remain
unverified. The historical `actor/perf/cpu_memory_used_gb` is
`psutil.virtual_memory().used / 2**30`, not actor RSS or offloaded-state bytes.
Do not add a critic-state estimate to it. Capture cgroup and GPU evidence during a
separately authorized smoke. This resource change does not authorize execution.
The override is available only for the two validated PPO actions; GRPO and non-GPU
actions do not use it. No new resource selector or arbitrary override is exposed.

Main fetched at `bca9303bba69e8af0f56e393ed1d938716eb45f1`; no later changes or open PRs
were found at initial inspection. Work began on a clean new `feat/ppo-mc-smoke` branch.
Both November 14 closeout reports and their evidence indexes were read. The earlier
`agent_lightning_phase2_ppo_codex_handoff.txt` was not found in the inspected workspace;
no existing handoff was replaced.

| Source | Reviewed identity and evidence |
|---|---|
| Agent Lightning | `v1.0.3`, `cbc56e41e0eab643aa71d195e92e5bba652fa683`, clean local source clone |
| VERL | `v0.8.0`, `7aed6b230776f963fa09509c10d9c3a767d1102c` |
| VERL wheel | `verl-0.8.0-py3-none-any.whl`, SHA256 `4e32830aae80771ee3b5f0d26a2b794d793f3bcece67aff07c2a7c62e948714f`; common source files match the tag, except `version/version` |
| Transformers constructor | Historical package record says `5.22.2`; inspected wheel SHA256 `829c2820aff91730b8ec45a4b8ff4be07f586a140b68a9073998b36fa35297dc` |
| Actor base | `meta-llama/Llama-3.2-3B-Instruct`, revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`; preserved weight/tokenizer/config byte identities |

The [source lock](../experiment/ppo_source_lock.json) retains reviewed file hashes,
wheel locators, prepared-asset hashes, and frozen 16/8 smoke data identities. The wheels
were downloaded **as source evidence only**, with no installation. The remote installed
VERL revision is not newly observed: future execution must match these file hashes and
record installed packages before model construction. A version string alone is insufficient.

Primary reviewed code: [AGL trainer](https://github.com/microsoft/agent-lightning/blob/cbc56e41e0eab643aa71d195e92e5bba652fa683/agentlightning/verl/trainer.py),
[per-rollout loss](https://github.com/microsoft/agent-lightning/blob/cbc56e41e0eab643aa71d195e92e5bba652fa683/agentlightning/verl/per_rollout_loss.py),
[VERL GAE/value loss](https://github.com/verl-project/verl/blob/7aed6b230776f963fa09509c10d9c3a767d1102c/verl/trainer/ppo/core_algos.py),
[model engine](https://github.com/verl-project/verl/blob/7aed6b230776f963fa09509c10d9c3a767d1102c/verl/workers/engine/fsdp/transformer_impl.py),
[worker reductions](https://github.com/verl-project/verl/blob/7aed6b230776f963fa09509c10d9c3a767d1102c/verl/workers/utils/losses.py).
The local review also covered config composition, entrypoint, rollout adapter, rollout-level
advantage, model construction, padding/value alignment, seeds, checkpoint save/load and retention.

## Evidence packet and research limits

The owner packet ZIP has SHA256
`5f389b80ca5845f99110fc5afa889607ef10e7ea1bcd968c589096d47b404c90`.
Its ZIP inventory and all 15 manifest entries match the local packet. After reading both
bundled scripts, offline verification passed: 64 selected-batch attempts, eight complete
traces, 16 groups, two frozen question rows. A further check matched 71 **already local**
source files and the source-manifest hash. This is internal/local checksum consistency,
**not an independent source-archive download in this session**. Raw material stays private.

Iteration 1's selected question group has rewards `[0,0,0,1]`; iteration 2 has eight
all-failure groups, including its selected `[0,0,0,0]` group. This does not establish that
all six historical zero-contrast iterations were all-failure batches. The original report
records 30/160 varying groups and 114/1,394 dropped rows. Unmerged-rollout counts do not
count single-call rows. Nothing here changes historical rewards or re-scores a model.

The JSONL traces contain messages, tool results, answers, rewards, times, IDs and usage.
They lack historical token IDs, log probabilities, exact masks, retained-row mapping and
critic predictions. The new arithmetic fixtures are synthetic, not historical loss or
gradient replay. The research question concerns useful task learning, stability, behavior
and measured cost under a common recipe. Nonzero gradients, favorable advantages, lower
training loss, or statistical significance alone do not answer it. Negative or inconclusive
results are valid; PPO superiority is not an acceptance gate.

## Estimator and fragment semantics

Set `adv_estimator=gae`, `gamma=1`, `lam=1`, rollout-level advantage **false**, critic **true**.
On a valid generated token with zero terminal bootstrap, raw `A_t = R - V_old(h_t)` and
return target `G_t = R`. VERL evaluates values at the preceding token position through
`no_padding_2_padding`; the original prompt and generated prefix form `h_t`. Values are
computed before both critic and actor updates. The critic updates first, using saved old
values; the actor uses the already computed advantages. Returns are never actor-normalized.

AGL places the outcome at the last attended response position of each independently
retained fragment. Each copy is a fragment outcome target, not additive rewards on a
stitched episode. Observation gaps are context, not actions. The adapter merges only when
token prefixes match; the PPO audit checks included tokens against actual call prefixes
and records call indices. It does not invent contexts or establish causal tool-action credit.
Truncation and legacy dropping remain visible limitations; the equivalence claim excludes them.

Fixtures use identical generated-token multisets, conditioning prefixes and positions,
per-token old values, outcome, masks, and global whitening population. Merged rows with
masked observations and separate outcome-labeled fragments agree for raw advantages,
returns, and whitened valid advantages at lambda=1. Lambda<1 and changed-context/value
counterexamples delimit the claim. Padding and observation values do not affect valid
GAE results. VERL uses sample variance plus `1e-8`; at least two valid tokens are required.
Constant values center approximately to zero, including finite-precision residuals.
The runtime records the raw Monte Carlo residual and checks FP32 advantages/returns with
absolute tolerance `2e-4`, relative tolerance `2e-5`; short native fixtures use `2e-6`.
For all-zero rewards and varying values, whitening `-V` creates positive and negative
contrast; it does not demonstrate a useful baseline. Unit variance fails near zero variance.
Larger average advantages for successful rollouts are a diagnostic, not an invariant or pass gate.

## Critic contract and optimization semantics

The critic has a **separate backbone** loaded from the exact prepared local actor directory.
Offline resolution and `trust_remote_code=false` are enforced. No generic DeepSeek path or
fallback download is accepted. Tokenizer/template/config and both base weight shards are
checked against preservation hashes, rather than trusting a rewritten model manifest.

The pinned engine calls `load_valuehead_model`, normally trying token classification and
catching any exception before an optional TRL fallback. The lab overlay selects the same
Llama token-classification constructor but **refuses fallback** and requests loading info.
With inspected Transformers 5.22.2, `LlamaForTokenClassification` uses a separate `LlamaModel`
and `score = Linear(hidden_size, 1, bias=True)`. Only `score.weight` and `score.bias` may be
missing; an unused `lm_head.weight` is allowed, other unexpected/mismatched keys fail.
Transformers initializes the head weight with normal standard deviation `initializer_range`
and bias with zeros. These are constructor/loading expectations, not an observed model load.

Engine `seed` alone does not seed this constructor when `full_determinism=false`.
The overlay uses Torch `fork_rng`, seed **739391**, and restores prior CPU/CUDA RNG states.
It records actual loading keys, class, head shapes/dtypes/hashes, token IDs and trainable
parameter counts/dtypes separately for actor and critic. These runtime records do not yet
exist. Python/NumPy/data-order/inference seed control for paired studies remains PR B.
The value model predicts the same outcome; it is not an independent correctness verifier.

| Critic setting | Explicit smoke proposal |
|---|---|
| Update/forward microbatch | 1/1 per GPU; both legacy keys and effective `critic.fsdp.micro_batch_size_per_gpu` / `infer_micro_batch_size_per_gpu` |
| Minibatch/epochs | 8 questions × 4 rollout factor = 32 retained rows; one epoch, no shuffle |
| Optimizer | Torch AdamW, LR `1e-5`, betas `[0.9,0.999]`, weight decay `0.01`, constant LR, zero warm-up |
| Clip | Optimizer gradient norm 1; value clip 0.5; `critic_warmup=0` |
| Memory | Parameter/optimizer CPU offload; activation checkpointing; no activation offload; static microbatches |
| Dtypes | FP32 storage, BF16 compute, FP32 gradient reduction/buffers |
| Checkpoints | Native model/optimizer/extra for both roles; keep 2 each in the new run only |

The older critic microbatch keys are **not propagated** by the unified worker constructor;
setting only those would be insufficient. Critic configuration is instantiated and checked
in the future real-runtime arithmetic preflight. Pilot settings above are provisional,
not tuned against any final holdout. Actor LR stays `1e-6`, lower/upper clips `.2/.3`, dual
clip `3`, original `per_rollout_mean` objective, no KL or entropy penalty.

**Reduction finding:** let N be retained rows and T_r retained generated tokens in rollout r.
AGL scales actor advantages by `1/(N*T_r)`. Its loss sums tokens (times DP size); the pinned
FSDP engine calls backward for each microbatch without a further accumulation divisor.
Thus splitting rows can change actor scale and the number/placement of optimizer steps.
Equal advantages do not prove equal losses or updates. The unequal-length/fragment tests
include finite-difference actor derivatives and an explicit row-count counterexample.

The native critic loss masks observations/padding, takes `0.5*mean(max(squared error,
value-clipped squared error))` within each microbatch, and the engine sums these gradients.
At microbatch 1 this is **a sum of row token means per 32-row minibatch**. More fragments
increase a rollout's critic weighting. This is documented legacy worker behavior for the
first plumbing smoke, not a representation-invariant common study objective.

AGL floors rows to a multiple of 32 and preferentially drops constant-reward groups, then
random rows if needed. This is not neutral for PPO, nor literally cost-free for GRPO.
PR A preserves it only for smoke, recording marked/dropped/retained rows and reasons.
There is no padding policy or comparison qualification. Before matched pilots, PR B must
retain valid information, whole rollouts and groups with no reward-dependent selection,
or report a concrete backend blocker. Random independent dropping is not an accepted cure.

## Diagnostics, checkpoints, and runtime acceptance

The explicit `ppo_runtime.register_in_worker` Ray hook verifies source hashes and installs
process-local wrappers; it never rewrites upstream or installed files. GRPO does not install
it. The driver runs native Torch/VERL arithmetic assertions before training. Client tests
use inspected stdlib ports/static checks/mocks because Torch, VERL, NumPy and OmegaConf are
absent locally. None of this qualifies real Ray/FSDP/FlashAttention/model integration.

Private `ppo-audit/events-PID.jsonl` records:

- Actual GAE input to whitening, post-whitening and actor-normalized summaries, with
  token/rollout/prompt-group weighting distinguished; reward/turn distributions and row lengths.
- Fresh-rollout **pre-update** MSE/MAE/bias, value/return distributions, out-of-[0,1] values,
  target variance, raw upstream EV and qualified EV (`null`/N/A when variance ≤ `1e-8`).
  A wrong constant predictor can have raw EV=1; EV>0 is never a pass gate.
- A constant predictor fitted once per real rollout on **earlier iterations only**; N/A
  on the first batch. Predictions are stratified by turn and within-row generated-token
  position quartile. This is fresh-rollout evidence, not an independent fixed development set.
- GRPO diagnostics once per real rollout, separately pre/post-selection, on the PPO batch,
  without changing that batch. These are not a counterfactual GRPO run or performance result.
- Iteration, case/group/rollout, fragment/call indices, counts/rewards, selection status/reason,
  minibatch assignment and normalization coefficient. No padded rows exist in PR A.
- First four full retained rows per iteration with original IDs/positions/masks, available
  sampling and training log-probs, old values/rewards/returns/raw/effective advantages. Dumps
  are bounded by configured sequence lengths; raw tasks/tensors are not published to GitHub.
- Actual optimizer step hooks by role, preclip gradient norm and postclip local norm; skipped
  nonfinite updates remain visible. Native clip fractions/entropy are preserved. `actor/ppo_kl`
  is approximate sampled update KL; rollout/training log-prob mismatch is logged separately.
- Cgroup memory current/peak/events and membership scope, process-local CUDA allocated/reserved
  peaks, optimizer-state tensor bytes, trainable/storage bytes, elapsed steps and checkpoint bytes.

Cgroup coverage must be checked on the actual worker. CUDA allocator metrics exclude other
processes, vLLM KV allocation and driver overhead. Detailed offload transfers, activation/KV
attribution and serialization peaks still require runtime profiling; no capacity or price
claim follows from GRPO fitting. For an authorized smoke, retain cgroup membership,
current/peak memory and OOM events alongside stage timing, actor/critic optimizer-state
bytes and checkpoint sizes. Check whether the observed cgroup contains the complete
worker/process tree; label gaps explicitly. CUDA allocator peaks are cumulative within
each process, so whole-device use and cross-process peaks need separate runtime evidence.
Inspect construction, updates and checkpoint serialization, and report measured values
separately from estimates. The 192 GiB reservation is not evidence that PPO fits.
Local logs are required; no PPO hosted W&B/Weave readback exists because PPO has not run.
W&B receives the resolved algorithm/protocol metadata through existing tracking,
but delivery is not claimed. Old upstream `training/n_zero_adv_groups` means constant reward
groups; it must not be interpreted as actual zero PPO advantages.

Native save completion adds `algorithm_manifest.json` with both role inventories, byte
hashes, config hash, upstream lock identity and lab source hashes. A save failure preserves
its partial evidence and cannot qualify reload. Retention only targets actor/critic paths
inside this fresh run; deletion on load is disabled. Upstream retention tracks paths saved
by this manager instance, pruning older role directories at capacity. Source/historical
checkpoints are never registered for deletion. Campaign step-20/final protection is PR B.

Reload requires matching algorithm/protocol and base/data/template/source identities,
completed step 2, actor **and critic** model/optimizer/extra inventories, and content hashes.
Missing metadata is accepted only as historical GRPO when the requested algorithm is GRPO.
Native load returns and the subsequent upstream inference refresh must be observed in logs;
requested-path metadata alone is insufficient. Validation-only reload does not test optimizer
continuation, even when optimizer files are loaded. A resume-update test needs separate approval.

## Future command proposals — separate approval for each stage

The first submission, `ppo-mc-v1-smoke-01`, remains a failed attempt at source
`11dd5a261615f57b86213477f14eb7bc53bc0683`: the owner reported
`ModuleNotFoundError: No module named 'experiment'` during function hydration,
before training. App `ap-viboiu2YehpFAeMk35Opbo` and call
`fc-01M7DQYFAT0JBKN5QRG26DAFP4` are preserved in the
[failure record](../evidence/ppo-smoke-01-import-failure-2026-11-14.json).
The import fix uses the existing packaged `BUNDLE` namespace when the repository
package is absent beside `modal_app.py`. An isolated offline test imports the entire
module with the real client SDK and only the packaged contract available; this does
not validate actual Modal hydration, image resolution or training. No retry occurred.
Use the fresh name `ppo-mc-v1-smoke-02` only after separate submission approval.

**Do not execute these as part of this PR.** The implemented Modal signature and resulting
launcher argv are tested offline. The preserved plan binds working Volume
`cs329z-agent-lightning-working-20261114` / `vo-rVO9GricBLVN10azwFVoYP` to restored inputs:
historical `/work/restored/models` and `/work/restored/data-small` become
`/data/restored/models` and `/data/restored/data-small`. New outputs go to `/data/phase2/runs`.
The original source Volume is not mounted. The prior read-only review confirmed the
working Volume and input assets; storage headroom and current image readiness still
need owner review before execution. The worker
fails on missing inputs and never creates a new working Volume or downloads replacements.
The submission code requires a clean committed source tree.

Stage 1 proposal (fresh name must still be unused):

```bash
AGL_WORKING_VOLUME=cs329z-agent-lightning-working-20261114 \
  uv run --locked modal run --detach modal_app.py \
  --action smoke --name ppo-mc-v1-smoke-02 \
  --algorithm ppo-mc-v1 --protocol ppo-mc-smoke-legacy-retention-v1 \
  --data-layout restored --wandb-entity amulyap-stanford-university
```

One H100 (`H100!`), 16 CPU cores, 192 GiB RAM; existing image/heavy layers, no upgrade.
Launcher 45 minutes; launch subprocess 55 minutes, GPU preflight up to 10 minutes; existing
outer function cap 4 hours. This is an upper-bound configuration, not a runtime/cost forecast.
Future image resolution/builds are billable and require approval too. Check the unchanged
qualified image receipt and cached layers before launch. No automatic hardware/memory fallback.

Acceptance: real two-iteration 16-case training flow, eight separate smoke eval cases,
actual tool/reward activity, finite values/losses/gradients where expected, measured actor
and critic optimizer calls, inference refresh, and both native checkpoints. No accuracy gain
is required. Synthetic nonzero-gradient checks stay distinct from authentic measurements.
Submission acknowledgment is not completion. Inspect terminal receipt and all failures.

Only after Stage 1 review and separate Stage 2 approval, retrieve the actual step-2
`algorithm_manifest.json` through an approved evidence-retrieval route and save its exact
bytes at `.local/phase2/ppo-mc-v1-smoke-02-algorithm_manifest.json`. It does not exist yet.
The following command intentionally refuses to submit without that local identity file:

```bash
AGL_WORKING_VOLUME=cs329z-agent-lightning-working-20261114 \
  uv run --locked modal run --detach modal_app.py \
  --action smoke-reload-eval --name ppo-mc-v1-reload-01 \
  --algorithm ppo-mc-v1 --protocol ppo-mc-smoke-legacy-retention-v1 \
  --data-layout restored --wandb-entity amulyap-stanford-university \
  --checkpoint phase2/runs/ppo-mc-v1-smoke-02/checkpoints/global_step_2 \
  --checkpoint-identity .local/phase2/ppo-mc-v1-smoke-02-algorithm_manifest.json
```

Use the same resource bounds. Verify eight exact IDs/rewards, native actor/critic load,
refresh and no update/new checkpoint. These commands are implemented interfaces tied to
the recorded resource mapping, not evidence that runtime integration or live bindings pass.

## PR B: separately authorized study controls

PR B is **not implemented or authorized by this PR**. No 20/60-step PPO command is exposed.
Before matched pilots it must deliver:

1. One versioned common retention/batching and actor-loss scaling policy for both arms;
   preserve real rollout/group membership. If padding is needed it contributes to neither
   loss, whitening, group baselines nor normalization, and creates no phantom optimizer steps.
   Test unequal lengths/fragments through the complete worker reduction and step path.
2. Validated steps, run seed, approved GPU selector, walltime and compatible inner/function
   timeouts; requested/resolved settings and graceful shutdown. Remove relevant hidden
   20-step, two-step-resume, 100-epoch and 180-minute limits; verify saves/evals/reload at new counts.
3. Separate derived streams for Python, NumPy, Torch CPU/CUDA, actor engine, critic/head,
   data order, row/minibatch selection and supported inference sampling. Record uncontrollable
   RNGs. Pair question-order schedules; a label containing `s329` is not seed control and
   a common integer does not promise bitwise equality. Dataset-selection seed/hashes stay separate.
4. Exact evaluation ID/count manifests and disjointness by ID and normalized question.
   Historical 64 become legacy/development, eight smoke remain separate. Reserve a **new**
   256-case final holdout from authorized retained source data, with near-duplicate/template
   checks, selection rule/hashes and leakage limitations (including unknown pretraining).
   No silent source replacement and no hard-coded 64-case summarizer for a 256-case panel.
5. Freeze binary training scorer for both arms; separately predeclare independent grading
   for math, extraction/format, tool execution and ambiguous ground truth, with rubric and
   uncertainty labels. Never change historical scores. Same evaluation loading/serving route
   and generation settings for both arms; retain all predeclared repeated greedy evaluations.
6. Freeze study configuration, fixed-final-step endpoint, baseline/loading route, paired
   seed/case-aware analysis, resource/storage bounds and protected step-20/final checkpoints
   before final-test evaluation. No holdout-based tuning, seed replacement or best-checkpoint choice.

After that qualification, separately authorize **one fresh 20-step PPO and one matched
20-step GRPO development pilot from the same base**. Development seeds are excluded from
campaign seeds. Review stability, early-position critic skill against the earlier-data
constant baseline (or a fixed unused development trajectory set), target variation,
policy movement, clipping, gradient/optimizer counts, task behavior, capacity and timing.
Decreasing training loss or positive EV alone cannot promote the design. Poor baseline
performance or constant targets must be reported as poor/inconclusive skill; recommend one
bounded development-only revision for review, or an explicitly exploratory campaign.
Never train indefinitely to pass a metric or promote pilot checkpoints to final runs.

The proposed later campaign is five paired fresh-base seeds per arm, 60 outer iterations,
four rollouts/question, identical qualified hardware and fixed common initial LR. Report
per-seed scores/paired differences, case wins/losses/ties and uncertainty under a declared
seed/case sampling model. Five uses of the same 256 cases are not 1,280 independent cases.
A five-pair two-sided sign-flip test has minimum p=2/32=.0625; alternative tests carry other
assumptions. The historical 11/3 discordance yields exact two-sided McNemar p=.057373046875
for that model pair, not an algorithm-across-seeds conclusion. A 6–7 point gap sometimes
crossing .05 is not an 80%-power guarantee. Prespecify meaningful-effect/discordance/seed
sensitivity; five pairs are exploratory unless power is separately established.

No stage starts automatically. Recheck current prices only when execution is proposed.
The owner's November 14 nominal H100/H200 totals ($5.726736/$6.317136 per hour at 16 cores,
128 GiB) are reference arithmetic, not invoices or a PPO duration forecast. Do not infer
free RAM from checkpoint sizes or promise a fixed per-run price/checkpoint size.
Those reference totals do not describe the new 192 GiB PPO request.

## Review disposition

PR A is reviewable bounded plumbing with offline evidence. Remaining gates are real source
and config composition checks, constructor/key/seed observations, native arithmetic and
worker integration, model fit, authentic optimizer/refresh/checkpoint/reload evidence, live
resource/data availability, and full resource-scope qualification. Keep all failures.
Common-pipeline, seed, evaluation and study controls remain a separate PR B request.

## Offline verification receipt

The [offline evidence index](../evidence/ppo-mc-offline-2026-11-14.json) records the exact
command/environment, tested source-file hashes, source head, dates, log checksum, previous
attempts and protected-file comparison. The final precommit run passed **180 tests**,
exit **0**, in **6.943 seconds**. Log SHA256:
`d80c1ce5acaf830f0d7a416ef0eed8c1643391288201c4d048745ee8705830c3`.
This includes 32 new PPO tests and the existing 148 tests. A syntax error in the initial
new test file was corrected; earlier successful test logs remain private and indexed.

```bash
MODAL_CONFIG_PATH=/dev/null HF_HUB_OFFLINE=1 \
  uv run --locked --offline python -m unittest discover -s tests -v
```

Python 3.12.3, uv 0.12.13, aarch64 Linux client. No CUDA/PyTorch training dependencies were
installed. Source-lock verification checked 36 source/config files, including the Calc-X
entrypoint; it checked downloaded source bytes, not a remote installed environment.
Launcher `--help`, Python AST parsing, JSON parsing, local documentation links and
`git diff --check` passed. The draft PR records a separate verification against its exact
committed head. No model, native checkpoint, cloud runtime or scientific evaluation was tested.
