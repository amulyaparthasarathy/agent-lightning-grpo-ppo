# Scientific analysis of Agent Lightning Phase 2 r03

The six jobs completed the intended matched 20-update protocol. GRPO finished at 36/64, 30/64 and 34/64; PPO-MC at 30/64, 34/64 and 34/64. The final-accuracy difference changes sign across seeds. Mean final accuracy differs by only +1.041667 percentage points in GRPO's direction, while mean difference in improvements is −0.520833 pp. These descriptive endpoints do not establish an algorithm effect.

## Provenance and scope

Executed source: `5fb6adfd48ee245509f7dff3ab1d849fcf707d86`, branch `feat/phase2-exploratory`. This is a publication copy of the completed local audit, not a claim that later documentation or merge commits generated the experiment. The PR stack was open/draft at audit time. Current integration status is tracked by the PRs and [closeout](../../docs/PHASE2_CLOSEOUT_2026-11-15.md).

Immediate frozen-source parents are `fa4dc93` (rollout seed configuration) and `608a3e5` (Transformers source fix); `5fb6adf` fixes IPC scratch paths. The old failed attempts remain preserved.

Canonical JSON digests (sorted keys, compact separators):

| Identity | SHA256 |
|---|---|
| Qualification plan | `00668b59c4943a1e5f450a41b733736ae858d591c4bdda99093bed5418747d55` |
| Study plan | `979046370f524f59b838c12dd93f3f3da31acd5b6c52cd7fb41e2e9dd617144a` |
| Source identity | `1ec04f20f3a4e4ad381acfab8ab6566005674c47e3538d1d92f8c61365153065` |
| Data manifest | `f0320992e0b650a73e3c069fbd745f7e024dc453ced7dd6f16aec7b059ccc33b` |
| Numerical contract document | `a7fbbd4cd581c3dbf3eb33114ae64b928cde475126c5754c16e9ef608ca6dd0c` |

All 44 plan source files match the frozen Git objects and current clean checkout. Plan equality holds across submissions, indices, requests, collection records and hosted matched-study metadata. The prepared Llama 3.2 3B revision is `0cb88a4f764b7a12671c53f0838cd831a0843b95`; Agent Lightning is pinned to `cbc56e41e0eab643aa71d195e92e5bba652fa683`, VERL to 0.8.0, and reviewed Transformers to 5.22.2. See `experiment/ppo_source_lock.json` keys `agentlightning_commit`, `verl_wheel`, `prepared_assets`; `experiment/matched_runtime.py:233` checks numerical source hashes before registration. Source/data/model identity is supported by these fail-closed runtime checks and matching evidence, not a new physical rehash of remote weights.

The namespace is `agl-explore-20-v1-r03`, the existing private Modal environment and working Volume (locators in EVIDENCE_INDEX.md). The environment name `main` is not a Git-source claim. Two single-H100 worker lanes, each 16 CPU and 196608 MiB, are recorded in every `resource-preflight.json`. The interval sweep peaks at two workers in both phases. Stage order is baseline → training → checkpoint-eval; one final checkpoint per job, no retries, replacement seeds or intermediate selection.

The manifest contains 256 training cases and 64 evaluation cases with disjoint IDs. All observed training cases belong to the frozen training list; every evaluation matches all 64 expected IDs exactly once. Source `matched_launch.py:122` additionally verifies parquet hashes, row order and normalized-question disjointness at execution. The existing local collection does not include the dataset parquet files, so a new independent byte-level/near-duplicate content audit is UNVERIFIED. Historical manifest `heldout` wording is retained for identity; the protocol explicitly labels this repeatedly used panel **exploratory-legacy-64** (`docs/PHASE2_EXPLORATORY.md:23`). It is not a fresh holdout.

Seed 329 selects data; it is not an RL seed. Qualification seed 9101 is excluded from every scientific average and figure. Paired seeds 2001/2002/2003 share `paired_question_order` but have algorithm-specific Python, NumPy, Torch, actor-sampling, actor-optimizer and row-batching streams, plus PPO critic initialization. Resolved configs and `matched_registered.streams` match each plan. The actual 20 training batches contain matching case multisets within each pair; sampling trajectories and asynchronous completion order are not paired RNG draws. No bitwise reproducibility claim follows.

## What was independently checked

`experiment/matched_launch.py:183` accepts a stage only with one trainer terminal event at the expected step and a W&B finish receipt. Training requires refreshes 0..N, actual optimizer events, a single final native checkpoint and hashed role inventories. Evaluation requires no optimizer calls, one refresh at 0 or N, no saved checkpoint, and complete real val episodes with posted binary rewards. `experiment/summarize.py:89` checks the case panel. The audit strengthens optimizer validation to the exact count sequence 1..N, not just event count/max.

All 24 rows in [stage_receipts.csv](stage_receipts.csv) pass: 6 qualification stages and 18 study stages. Durable receipt, supervisor stage result, completion.json and result.json agree; all exit codes are zero. Every study training job has actor counts 1..20 and refreshes 0..20, and PPO has critic counts 1..20. Gradient events are finite and report optimizer_called=true. The post-hook sits on the actual optimizer (`ppo_runtime.py:411`); metric-row counts alone were not used to infer updates. No nonfinite/skipped optimizer event is observed. Zero current gradients are discussed below.

Every study job saves exactly one `global_step_20`, with model/optimizer/extra-state files for actor and, for PPO, critic. Manifest hashes and file-byte sums match completion receipts; saved resolved-config hashes match local bytes. Available small checkpoint files also rehash correctly. Tensor weights were neither downloaded nor loaded. **The manifest is a recorded content-hash inventory, not a new independent verification of inaccessible tensor bytes.** Checkpoint-eval workers have different call IDs, the exact training checkpoint resume path/hash, terminal global_step=20 and validation_only=true. They perform zero actor/critic updates and no new checkpoint save. Source verifies the checkpoint before launch and after evaluation (`matched_launch.py:172`, `:236`).

All 16 evaluations (12 study, 4 qualification) contain exactly 64 complete real episodes, 64 confirmed binary reward posts, no duplicate/missing case, and zero optimizer events. All 3840 study training trajectories are likewise present: 32 real rollouts per step, eight groups of four. Fragment counts and generated-token totals reconcile to `collected_rows`; no padded, dropped or truncated trajectory is silently included. Bounded tensors confirm terminal returns, response-mask weights and raw advantage formulae. This evidence supports the implemented processing, not a new independent derivation of every unrecorded token.

The overnight timestamps in the request are verified in `overnight.log:1–22`. Poll observations are later than durable finishes: qualification supervisor ended at **06:59:58.154153**, observed completed at 07:01:00; study supervisor ended at **10:15:48.270791**, observed at 10:17:27. The runner reported success after collection at 11:11:41. These are different clocks/events, not contradictory completion claims.

## Frozen-scoring results

The scorer is unchanged: `observed_calc_agent.py:329` extracts the `### ANSWER:` span or falls back to the raw final response, then invokes original Calc-X scalar comparison with tolerance 1e-2. Ground truth is supplied to the scorer, not included in the constructed agent prompt (`:307`). This audit uses the **recorded posted reward**, does not rewrite the extractor, and does not independently rescore answers. Task correctness here means this recorded scalar endpoint, not a manual semantic assessment.

| Algorithm | Seed | Baseline | Final | Baseline % | Final % | Δ pp | Actor / critic updates | Wins / losses / ties |
|---|---|---|---|---|---|---|---|---|
| GRPO | 2001 | 27/64 | 36/64 | 42.1875 | 56.2500 | 14.0625 | 20 / 0 | 11 / 2 / 51 |
| PPO-MC | 2001 | 26/64 | 30/64 | 40.6250 | 46.8750 | 6.25 | 20 / 20 | 7 / 3 / 54 |
| GRPO | 2002 | 27/64 | 30/64 | 42.1875 | 46.8750 | 4.6875 | 20 / 0 | 7 / 4 / 53 |
| PPO-MC | 2002 | 25/64 | 34/64 | 39.0625 | 53.1250 | 14.0625 | 20 / 20 | 12 / 3 / 49 |
| GRPO | 2003 | 26/64 | 34/64 | 40.6250 | 53.1250 | 12.5 | 20 / 0 | 13 / 5 / 46 |
| PPO-MC | 2003 | 26/64 | 34/64 | 40.6250 | 53.1250 | 12.5 | 20 / 20 | 10 / 2 / 52 |

| Seed | GRPO − PPO final (pp) | GRPO − PPO improvement (pp) | GRPO-only / PPO-only correct | Both correct / both wrong |
|---|---|---|---|---|
| 2001 | 9.375 | 7.8125 | 9 / 3 | 27 / 25 |
| 2002 | -6.25 | -9.375 | 4 / 8 | 26 / 26 |
| 2003 | 0.0 | 0.0 | 7 / 7 | 27 / 23 |

| Arm / contrast | Metric | Mean | Median | Min | Max |
|---|---|---|---|---|---|
| GRPO | Final accuracy (%) | 52.083333 | 53.125000 | 46.875000 | 56.250000 |
| GRPO | Improvement (pp) | 10.416667 | 12.500000 | 4.687500 | 14.062500 |
| PPO-MC | Final accuracy (%) | 51.041667 | 53.125000 | 46.875000 | 53.125000 |
| PPO-MC | Improvement (pp) | 10.937500 | 12.500000 | 6.250000 | 14.062500 |
| GRPO − PPO | Paired final difference (pp) | 1.041667 | 0.000000 | -6.250000 | 9.375000 |
| GRPO − PPO | Difference in improvements (pp) | -0.520833 | 0.000000 | -9.375000 | 7.812500 |

Per-pair final disagreements are 12, 12 and 14 cases, respectively. Seed 2003 has equal totals but 7 cases correct only for each arm. Baselines differ within seeds (27 vs 26, 27 vs 25, 26 vs 26); equal base weights and greedy evaluation therefore do not imply identical observed outcomes under asynchronous hardware execution. Mean baseline accuracy is 41.666667% GRPO and 40.104167% PPO-MC, which explains why the mean final difference and mean improvement difference point in opposite directions.

Qualification is separate: seed 9101 GRPO 27/64 → 27/64, PPO-MC 27/64 → 30/64, with two actual actor steps each and two critic steps for PPO. All six qualification stages pass. These are integration checks, not selected scientific seeds.

## Training signal and critic behavior

| Algorithm | Seed | Mean sampled reward | Mean informative groups | Zero actor-gradient steps | Max |PG loss| | Mean actor preclip norm | Actor norm min–max |
|---|---|---|---|---|---|---|---|
| GRPO | 2001 | 0.135937 | 2.00 / 8 | 3 | 1.34e-08 | 4.0819 | 0.0000–13.3219 |
| PPO-MC | 2001 | 0.085938 | 1.20 / 8 | 0 | 0.424 | 11.2352 | 5.4623–26.9453 |
| GRPO | 2002 | 0.165625 | 1.70 / 8 | 2 | 1.42e-08 | 4.2586 | 0.0000–12.8977 |
| PPO-MC | 2002 | 0.142187 | 1.30 / 8 | 0 | 0.522 | 9.9797 | 5.6563–26.7510 |
| GRPO | 2003 | 0.156250 | 1.80 / 8 | 3 | 1.2e-08 | 3.7349 | 0.0000–8.0739 |
| PPO-MC | 2003 | 0.143750 | 1.50 / 8 | 0 | 0.524 | 9.8444 | 5.7404–18.5066 |

Training rewards are 32 sampled trajectory rewards per step, not repeated independent evaluation grades. Their means need not match the row/token-weighted `critic/rewards/mean` metrics; fragmentation changes those diagnostic denominators. They are not a substitute for the 64-case endpoint.

GRPO has only 110 informative groups out of 480 across the three runs (22.9167%). Its group-centered advantages sum approximately to zero. At the old-policy ratio of one, the scalar surrogate can cancel to near zero while its parameter derivative remains nonzero. This is exactly what the measured preclip gradients show. Zero scalar loss is therefore not evidence that optimization stopped.

GRPO zero-gradient steps are **2001: 4,10,13; 2002: 9,19; 2003: 10,13,17**. All eight have eight constant-reward groups, zero group advantages and optimizer_called=true, with proper sequential optimizer counts. These steps have zero current policy-gradient contribution. AdamW momentum/weight decay can still affect parameters; parameter-by-parameter changes were not measured, so the audit does not claim that each call changes weights. Across other steps, preclip norms reach 13.321857; clipping generally limits the postclip norm to about one.

PPO-MC uses a learned Llama token-classification value head with separately seeded initialization. Resolved `algorithm.adv_estimator=gae`, gamma=1, lambda=1 are consistent with the matched override: **every generated action gets terminal return R; raw advantage is R−V; whitening uses the full valid generated-token population** (`matched_runtime.py:87–134`). It does not mistakenly bootstrap at each fragment boundary. Per-rollout weighting gives each trajectory equal total weight, excludes observation/tool-output tokens, and is separate for actor and value loss (`:72`, `:137`). Bounded raw/return/weight tensors agree with this implementation. Only the first four fragment rows are retained for raw/policy/effective-advantage diagnostics; their curves are explicitly not full-batch statistics.

| PPO seed | Pre-update MSE: first → last | Value loss: first → last | Qualified EV range | Valid EV steps | Undefined EV steps | MSE beats earlier-data constant |
|---|---|---|---|---|---|---|
| 2001 | 4.073710 → 0.189495 | 2.377891 → 0.094438 | -83.883 to -1.120 | 14 | 4,6,10,13,17,19 | 0 / 19 |
| 2002 | 2.447244 → 0.151124 | 1.255584 → 0.106794 | -226.350 to -0.709 | 19 | 8 | 0 / 19 |
| 2003 | 3.163390 → 0.376402 | 2.077990 → 0.180897 | -178.340 to -1.665 | 18 | 13,17 | 0 / 19 |

All 51 defined explained-variance observations are negative. Nine constant-target steps are undefined, omitted from qualified-EV series and never zero-filled. Critic MSE improves substantially from initialization but loses to the recorded earlier-training constant predictor on **all 57 comparable steps**. That comparison is descriptive and uses the source's earlier rollout-mean predictor with token-weighted current error. Large critic gradients prove optimization activity, not useful value prediction. PPO can produce nonzero actor gradients even when group rewards are constant because its learned values vary; this is a mechanism difference, not demonstrated advantage.

`actor/ppo_kl` and actor clipping fractions are zero for all 120 study updates. With one full-collection optimizer update, one PPO epoch, and old/current policy evaluation before that update, this is consistent with ratio-one diagnostic evaluation. It does not prove no across-step drift. `rollout_corr/kl` and nonnegative `rollout_corr/k3_kl` quantify rollout-versus-recomputed old-policy log-probability differences; they are **not reference-model KL**. Reference KL is unavailable because no reference penalty is active. Entropy is not recorded (`calculate_entropy=false`, coefficient zero). No fabricated entropy/reference-KL curve is supplied.

## Tools, formatting and incomplete outcomes

| Algorithm | Seed | Baseline tool errors/calls | Final tool errors/calls | Training tool errors/calls | Missing answer markers: base → final | Training missing markers |
|---|---|---|---|---|---|---|
| GRPO | 2001 | 23/76 | 15/72 | 300/727 | 26 → 1 | 147/640 |
| PPO-MC | 2001 | 23/76 | 30/80 | 330/781 | 27 → 12 | 269/640 |
| GRPO | 2002 | 23/76 | 10/64 | 304/748 | 26 → 1 | 150/640 |
| PPO-MC | 2002 | 23/76 | 14/72 | 315/758 | 27 → 1 | 178/640 |
| GRPO | 2003 | 23/76 | 15/73 | 307/756 | 25 → 0 | 148/640 |
| PPO-MC | 2003 | 22/76 | 21/73 | 314/750 | 27 → 3 | 236/640 |

There are zero invalid/incomplete episodes, empty final answers, recorded model-span exceptions, or trace warnings. There are nevertheless many **tool-reported/exception spans**, and many final answers lack the prescribed marker. These are distinct from missing trajectories. All missing-marker episodes in both training and evaluation have recorded reward zero; no such episode was counted as a success. `malformed_or_missing` in the final CSVs means missing answer marker plus invalid/incomplete episode count (the latter is zero), and separate columns expose both quantities. Tool error counts are spans, not failed-case counts; one case may have multiple calls or recover after a tool error. Tool success rates use recorded spans and do not measure unrecorded transport retries. A separate exhaustive malformed tool-call-parser event stream is unavailable; zero model-span exceptions is not proof that no parser warning occurred anywhere.

GRPO's final marker compliance improves for all three seeds; PPO seed 2001 retains 12/64 missing markers and has the lowest PPO final score. This pattern is consistent with an output-format contribution to scores, but it is not a causal decomposition: the frozen scorer already includes extraction and numerical correctness, and no alternative rescore was performed. Tool success alone does not explain the differences: GRPO seed 2002 has fewer final tool errors but a lower score than PPO seed 2002.

## Resources, costs and hosted reconciliation

| Algorithm | Seed | Training worker seconds | All three worker-stage seconds | Checkpoint bytes | Checkpoint GiB |
|---|---|---|---|---|---|
| GRPO | 2001 | 2525.290 | 3331.937 | 40146400251 | 37.389 |
| PPO-MC | 2001 | 2085.664 | 2948.698 | 78716827965 | 73.311 |
| GRPO | 2002 | 1749.980 | 2516.082 | 40146400187 | 37.389 |
| PPO-MC | 2002 | 2329.205 | 3216.041 | 78716827965 | 73.311 |
| GRPO | 2003 | 1698.370 | 2526.124 | 40146400251 | 37.389 |
| PPO-MC | 2003 | 3675.539 | 4581.727 | 78716827965 | 73.311 |

Worker intervals use durable receipt started_at/finished_at and include startup, model loading, evaluation, training, hashing and receipt work; they are not GPU kernel-utilization time or billed seconds. Qualification sums to **2994.427137 worker-seconds**, study **19120.607965 worker-seconds**. Supervisor elapsed times are **1725.782357** and **11287.050669 seconds** respectively and overlap the workers; do not add them as wall-clock duration. The supervisor's submission-to-observed-finish bounds are larger (3091.617770 / 19336.909928 seconds). Log-delimited collection takes 378 seconds for qualification and 3254 seconds for the study, outside scientific training. Exact stage intervals are in the 24-row CSV.

Study checkpoint role inventories sum to **356589684584 bytes**; qualification plus study to **475452912736 bytes**. These are recorded role-file bytes, not total Volume occupancy (logs, identities, transients and earlier artifacts are excluded). GRPO seed 2002 has a 64-byte smaller role inventory than the other GRPO study seeds; byte sums and hashes remain internally consistent.

At the read-only provider snapshot **2026-11-15 14:12:18.783408 UTC**, hourly object-level billing reports attribute:

| Scope | Provider object | Metered USD |
|---|---|---:|
| r03 qualification | `ap-5ZH9grlMGxLOfI2582GWzH` | 5.76491115 |
| r03 study | `ap-tcefSddplIc3elCItYK2xZ` | 34.33868457 |
| r03 apps combined | Sum of the above | 40.10359572 |
| Whole workspace billing cycle | 2026-11-01 to 2026-12-01 | 43.32536871 |
| Whole workspace billed_cost after adjustments | Not attributable to r03 alone | 13.29 |

App IDs come from the preserved qualification/study launch logs, not a shared title search. Their hourly rows cover the execution hours (qualification 06–07 UTC; study 07–10 UTC). These are measured as-of provider app records, not a closed-cycle final invoice. There is no seed/job-level billing split in this evidence, and app charges include worker/supervisor resource use. Future Volume storage, exact r03-attributed shared storage/collection charges and later adjustments are **UNVERIFIED**. No provider pricing estimate is presented as an invoice.

The provider workspace snapshot itself is not exactly additive: reported category breakdown sums to **$43.33036198**, $0.00499327 above its metered_cost field. The final billed_cost field is rounded to cents; that may relate to the discrepancy, but the provider did not supply an explanation. Both raw fields are retained. A residual subtraction from the workspace total is not a measured per-study charge. Prior failed apps have separately preserved billed rows and are not included in the $40.10359572 r03 app sum.

The allocation remains **$50 cumulative qualification + $175 study**, with **$75 contingency locked**. The logged prelaunch workspace gates ($3.15125017 and $9.08142174) and scheduler upper-cost figures ($9.067211 and $56.848931) are not invoices. Their purposes differ from object-level metering.

W&B was queried read-only for exactly the 24 frozen IDs in the private project identified in EVIDENCE_INDEX.md. All names, groups, job types, source/protocol/data hashes, algorithms, seed streams and stage metadata match each plan/local observability.json. Each run is finished. The bounded [0,100) history query requested up to 1000 rows, well above the maximum 20, and returned **140 rows** total: 120 study training, 4 qualification training, 16 evaluation. The exact local step sets and every local scalar key/value are present; all final scalar summaries agree within 1e-12 relative/absolute tolerance. This includes step 20 for all six study runs.

Literal full config equality is not claimed. W&B's API wraps config fields in `value`; after decoding, actor and critic optimizer total_training_steps are 2/20, whereas pre-trainer local resolved_config.json stores −1. The other nonempty flattened values match; structured empty-container serialization also prevents blanket object equality. This is recorded runtime resolution, not a source/seed or observed-history discrepancy. W&B state is corroborative; the actual completion verdict comes from trainer/optimizer/evaluation evidence. The unrelated historical `train-01` crashed label and missing final hosted row do not invalidate its separately preserved Phase 1 receipts and were not used here. Weave verification is out of scope; `matched_app.py:107` explicitly sets weave=false.

## Interpretation and limitations

**Tested observations:** exact update counts and complete reload evaluations; sign-changing paired endpoints; sparse GRPO informative groups; zero current GRPO gradients on eight constant-group steps; weak PPO pre-update value prediction; substantial tool/format errors; and varying runtimes. GRPO mean training-stage duration is 1991.213 seconds versus PPO 2696.803 seconds, but PPO seed 2001 is faster than its GRPO pair. Extra critic compute does not alone explain observed stage duration; startup, serving, fragmentation and scheduling vary.

**Hypotheses, not tested causal findings:** sparse GRPO signal may limit learning; the initially inaccurate critic may inject noisy PPO advantages; improved format compliance may account for part of endpoint gains; asynchronous sampling and hardware behavior may explain some baseline and paired variability. No ablation, alternative scorer, unseen panel, or counterfactual run tests these hypotheses.

Three seed pairs are a small descriptive sample. The reused 64 questions are not 384 independent test cases. Separate algorithm-specific sampling/optimizer streams, stochastic training, nondeterministic greedy serving, full-token PPO whitening versus group normalization, fragment/context handling, response-only masking, extractor behavior, scheduler variation and critic compute all constrain causal interpretation. The matched implementation controls many processing differences and verifies complete rollouts, but it cannot turn this design into a confirmatory study. No p-values, significance claims or generalization claims are made.

## Negative findings and unavailable measurements

Earlier original/r01/r02 qualifications failed before successful training: original `cb246a0` hit a Transformers source-lock mismatch (installed 5.23.0 versus reviewed 5.22.2); r01 `608a3e5` hit unsupported RolloutConfig.seed; r02 `fa4dc93` hit 205/207-byte vLLM IPC paths exceeding 107 bytes. All six earlier baseline worker receipts are failed; all three supervisor calls returned failed study outcomes despite provider SUCCESS for the supervisor function. Their exact calls, sources, failed receipts and costs remain in `evidence/phase2-exploratory/failure-{a01,r01,r02}/`. No earlier failed seed was silently folded into r03 or replaced within r03. The empty pre-study app and unsubmitted a02 are not completed experiments.

Unavailable: full-batch raw/whitened token advantages beyond the bounded four rows; entropy; reference-policy KL; exhaustive parser diagnostics; an independent human semantic grade; new physical model/data/tensor checksum reads; final invoice and future storage. The smallest relevant existing evidence would be a fuller already-recorded tensor audit, structured parser-event log, prepared-data manifest/files, or closed provider billing export. If those were never recorded, they cannot be recovered honestly from scalar metrics. No rerun or download is required to use the verified endpoints.

The original local audit retains 49 SVGs and the 18,231-row training_metrics_long.csv, with source locations and population scope. This selected bundle includes only the [paired accuracy plot](figures/INDEX.md). The full atlas and long table are private local locators in EVIDENCE_INDEX.md, not Git-hosted artifacts. Missing qualified EV remains a gap; entropy/reference KL remain UNAVAILABLE. Omitted measurements were never zero-filled.
