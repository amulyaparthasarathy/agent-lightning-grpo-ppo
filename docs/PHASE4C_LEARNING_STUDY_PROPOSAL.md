# Frozen GRPO recovery learning study, version 1

The proposed scientific question is whether a fixed 16-update GRPO schedule changes strict recovery reward on held-out cases from the same synthetic generator. There are no model results for this proposal. The reader explicitly rejects its schema. A future executor, its review, and separate authorization would be required. Two-update qualification alone cannot authorize or establish capacity for the longer experiment. [Machine-readable declaration](recovery-learning-study-v1.json)

## Policies and schedule

Use meta-llama/Llama-3.2-3B-Instruct at revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`, the locked tokenizer and chat template, llama3_json parser, recovery-readiness-v1 generator, tools, prompt, production agent and independent strict verifier. The declaration pins source and asset identities and every derived random stream. Each seed (2001, 2002, 2003) starts from the fresh base. Save only step 16. The selected checkpoints are all three terminal checkpoints. Exclude qualification warm starts, intermediate tuning, reward-based early stopping, seed replacement and best-checkpoint selection.

| Stage | Occurrences | Cases × samples × updates per occurrence | Episodes |
|---|---:|---:|---:|
| Fresh training, train_main | 3 | 8 × 4 × 16 | 1,536 |
| Development, dev_panel | 4 | 24 × 1 | 96 |
| Final, final_eval | 4 | 64 × 1 | 256 |
| Total | 11 | | 1,888 |

Training samples at temperature 1 with top-p 1 and top-k −1. Evaluation disables sampling with the same top-p and top-k settings. Each evaluation uses the validation route, greedy decoding, one response per case, seed 9001 and the fixed evaluation streams. Evaluate one base plus the three terminal checkpoints on development. Freeze checkpoint identities, all analysis choices and the final routing before releasing **any** final result. Then evaluate a fresh base and those same three checkpoints on final. Phase 4B's development baseline cannot supply the final base. The protocol provides zero sampled final episodes. For each prospective evaluation, the proposal binds both eager inputs to the same panel by filename, file SHA-256, ordered case-ID digest and row count. The prospective training stage binds its validation input to development. The final routing test uses deterministic dataset metadata only.

The upstream example eagerly reads both Parquet arguments. The qualification CPU replay uses that example, LoadedDataset, seeded RandomSampler, eight loader workers, StatefulDataLoader and the synchronous trainer iterator. LoadedDataset disables overlong-prompt filtering. With gen_batch_size absent, train_batch_size is eight, shuffle is enabled and drop_last is true. The fit loop stayed in epoch one through step 16, stopped on the declared step and reached its sole save boundary. All 128 distinct case IDs appeared exactly once per seed. The recorded orders and their digests are compared in tests. Only file access, loader and control flow were exercised for these schedules. [Ordered schedules](../experiment/recovery_schedules.json) · [Pinned loader replay](../tests/recovery_loader.py)

The numerical protocol is matched-rollout-mean-v2. Use unbiased within-group standard deviation and epsilon 1e-6, rollout-mean loss, one minibatch and one PPO epoch per update. AdamW uses lr 1e-6, betas 0.9/0.999, weight decay 0.01, constant scheduler, zero warmup and gradient clipping at 1.0. Lower/upper ratio clips are 0.2/0.3 and dual clip is 3. Entropy coefficient is zero. Disable critic, reward KL, actor KL, importance/rejection sampling and dynamic batching. Recompute old log probabilities. Actor and log-probability microbatches are one per GPU. Enable parameter and optimizer offload and gradient checkpointing. Serving utilization is 0.5, maximum concurrent sequences 32, serving context 4096 and output cap 1024. Trajectory adapter caps are 3072 prompt and 2048 response. [Expanded settings and evaluation streams](recovery-learning-study-v1.json)

## Paired analysis

For each seed subtract the fresh base strict binary reward from the terminal policy's reward on the same final case. Report all three means over the 64 cases, their arithmetic mean and descriptive sample standard deviation across seeds. Missing or unconfirmed rewards make the analysis incomplete. Reward zero requires an observed, independently confirmed outcome.

The generator has 32 training, six development and 16 final policy blocks, with four cases per block. Both same-asset temporal pairs stay within their block. For each of 10,000 replicates use random.Random(329) to draw 16 final policy blocks with replacement, jointly across all four policies. Keep both temporal pairs and all repeated samples inside each case. Average samples inside case, cases inside block and the sampled block differences across blocks. Report linearly interpolated 2.5th and 97.5th percentiles per seed and for the seed mean, labelled **conditional synthetic case-mix uncertainty**. Three training seeds support a descriptive account of seed variation. The interval addresses case mix only. Generalization here is within the synthetic generator. Real backup systems are outside this estimand. Phase 4B retains its original question-resampling analysis. [Synthetic analysis implementation](../scripts/recovery_study.py)

Secondary reporting should show strict reward by family with counts, decision and evidence correctness separately (correct / assessable, plus assessable / all), malformed final and typed action failures per episode, recovery rate among typed failures, and temporal pairs with both decisions correct / assessable pairs. Report tools called and independently confirmed observations per episode and per tool, informative groups / all sampled groups, generated and targeted tokens, stage and update latency, measured usage before credits, storage and delayed charges. Retain denominators and incomplete counts. A future PPO-MC proposal would also need pre-update critic prediction error, target variance, qualified explained variance, value clipping, policy clip fractions and KL by step.

## Cost worksheets

All dollar figures below are proposed ceilings or explicit planning allowances. No spending authority is attached. The dated Phase 4B all-resource worker rate was $6.2428/hour. Use a conservative $10/hour all-resource reservation here. The reader floor is $7/hour for the worker and $0.07/hour for the supervisor. Its supervisor reservation actually charges the worker ceiling for its entire deadline. All three worksheets use that policy, include the extra 600 supervisor seconds, allow one attempt, reserve cold build capacity and assume zero credits. [Calibration cost evidence](../evidence/phase4b-live-20261125/cost-and-retention.json)

| Independent worksheet | Worker stage timeouts | Worker hours | Supervisor hours | Compute reservation rounded up | Proposed ceiling |
|---|---|---:|---:|---:|---:|
| A. First GRPO qualification, 144 episodes | 24 baseline, 32 probe, 64 train, 24 reload. 3,600 s each | 4 | 4 + 1/6 | (4 + 4 + 1/6) × $10 = $81.67 → $82 | **$113** |
| B. Optional paired qualification, 288 episodes | Two four-stage routes as A, sequential. 3,600 s per stage | 8 | 8 + 1/6 | (8 + 8 + 1/6) × $10 = $161.67 → $162 | **$193** |
| C. Three-seed GRPO proposal, 1,888 episodes | 3 × 512 training at 14,400 s each, 4 × 24 development and 4 × 64 final at 1,800 s each | 16 | 16 + 1/6 | (16 + 16 + 1/6) × $10 = $321.67 → $322 | **$353** |

| Allowance per independent worksheet | A | B | C | Basis and stop condition |
|---|---:|---:|---:|---|
| Build | $15 | $15 | $15 | Cold build allowance. Stop if a reviewed cold estimate exceeds it. |
| Preparation and preflight | $5 | $5 | $5 | One preparation up to 3,600 s ($4), one preflight up to 900 s ($1), both estimates. |
| Storage | $7 | $7 | $7 | At most 300 GiB for seven days at dated $0.09/GiB-month ≈ $6.30. Inventory retained bytes under the owned volume paths and project additional volume writes. |
| Collection | $2 | $2 | $2 | Up to 2 GiB of receipts, logs and manifests. At dated $0.04/GiB, egress alone would be $0.08. |
| Delayed billing | $2 | $2 | $2 | Hold until reconciliation. Charges and final invoice are unknown. |
| Total outside execution | $31 | $31 | $31 | Add once to each worksheet's rounded compute reservation. |

A and B use a 600-second startup bound within each stage. C's proposed 4-hour training deadline is a prospective budget limit and would require a different executor. It is not an extrapolation of collection speed. The stopping rule for each worksheet is the first exhausted budget/time/storage bound, failed dependency or unacknowledged outcome. Zero automatic retries. Keep failed work and reconcile all prior exposure before any separate decision.

Historical Phase 3 native role inventories were 40,146,400,251 bytes for a GRPO actor and 78,716,827,965 bytes for PPO-MC actor plus critic. Comparable final roles alone would take about 37.39 GiB (A), 110.70 GiB (B), and 112.17 GiB (C). Each worksheet allocates 16 GiB for retained model/cache files and 2 GiB for collected logs. Add existing retained volume bytes, projected checkpoint and loader-state bytes, and any temporary files written on that volume. Compare the projected inventory with the 300-GiB limit and the retention cost with the $7 allowance. Separately, measure free bytes on the worker-local filesystem that will hold temporary writes and compare them with peak simultaneous scratch requirements during checkpoint creation or reload. Seven days is the proposed review horizon. Retention beyond it requires an explicit storage decision. Three paired seeds would need about 332 GiB for final roles alone. No historical checkpoint deletion or bound expansion is proposed. [Historical role inventories](PHASE3_CLOSEOUT_2026-11-24.md)

Before any spending, the owner would need to select a single worksheet, confirm source, exact resources, price and remaining headroom, and explicitly authorize that attempt and ceiling. The recommended first decision is A. A favorable or unfavorable result does not automatically change C's frozen design or unlock its final evaluations.
