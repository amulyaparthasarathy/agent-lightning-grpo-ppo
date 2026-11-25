# Phase 4C · GRPO qualification attempt 1

**Qualification failed during baseline finalization on November 25, 2026 UTC.** The trainer completed 24 greedy development episodes and exited successfully. Independent replay confirmed every reward. The worker then failed a configuration check before writing its stage completion record. The supervisor skipped the collection probe, training and reload. The evidence reviewer admitted **0 episodes and 0 actor updates**. Final-panel model evaluations totaled zero.

| Stage | Planned episodes | Observed | Confirmed rewards | Admitted | Outcome |
|---|---:|---:|---:|---:|---|
| Fresh-base development baseline | 24 | 24 | 24 | 0 | Failed after evaluation |
| Collection probe | 32 | 0 | 0 | 0 | Skipped |
| Fresh-base training | 64 | 0 | 0 | 0 | Skipped |
| Step-2 reload evaluation | 24 | 0 | 0 | 0 | Skipped |

## Failure and diagnostic reward

The runtime wrote the provider-resolved path to the owned volume in `trainer.default_local_dir`. The final check constructed its expected path from the logical `/data` mount. The two paths had the same volume-relative checkpoint directory and different absolute strings. Replaying the worker check against the collected configuration reproduced the exception. The missing completion record and failed worker receipt prevented stage admission. The registry and volume supervisor indexes agreed.

The 24 replayable baseline trajectories yielded **9 strict successes (37.5%)**, 12 semantic failures and 3 evidence failures. All 24 reached a final answer. All 144 model calls reported usage. The baseline recorded one policy refresh at step 0. The run produced zero sampled groups, actor optimizer calls or checkpoints. Loss, gradient, training-time log-probability recomputation, update latency, refresh latency and fresh-worker reload measurements are unavailable. The two-update protocol and the longer study remain unqualified. [Stage evidence summary](../evidence/phase4c-grpo-qualification-20261125/qualification-summary.json)

## Capacity

The baseline ran on one NVIDIA H100 with 16 CPUs and 192 GiB of host memory. The worker receipt spans 341.676 seconds. Four live device samples reached a maximum of **53.64 GiB**. A late cgroup sample measured **43.42 GiB** in use. The highest 60-second provider averages were 37.54 GiB of GPU memory and 35.09 GiB of host memory. Exact device and cgroup peaks are unavailable. Worker-local scratch had 512.00 GiB free at entry and at least 511.998 GiB at the sampled instants. Actor parameter and optimizer offload were configured. Evaluation supplied no measurement of training offload traffic, backward-pass capacity or checkpoint write latency.

## Provenance and collection

| Identity | Value |
|---|---|
| Execution source | `547053db6636fdc1475f0d53f8bf580fed751bbd` |
| Reviewed and tested runtime ancestor | `852e01d141b66232c73ca46e6e7bf4b9dfb0e854` |
| Historical Phase 4B execution source | `bc81c1fb0def27b15c56cdd8abb01098e7e1f8d5` |
| Preparation plan | `fa8ff85ef86e8778675f0cb8533392074a798561c83606f89cfcd37c6b6d9fcd` |
| Asset-bound execution plan | `8098ce37276f88860f2ae2a7530a8671edd8f8997e7e7d587b731a299c6eeb51` |

Fresh preparation verified the pinned model, tokenizer, template and frozen dataset. The CPU preflight passed on the exact asset-bound plan. Each submission was acknowledged once. Terminal collection returned 4,280,475 bytes with zero collection errors. The existing reviewer independently reproduced the incomplete qualification verdict from the private evidence. [Provenance, receipt commitments and collection](../evidence/phase4c-grpo-qualification-20261125/provenance-and-gates.json)

## Costs and retention

All four owned calls and their containers reached a terminal state. Preparation and preflight succeeded. The supervisor successfully returned the failed study outcome, and the baseline call raised an exception. All three apps stopped. Outstanding call exposure is **$0**.

The provider reported **$0.62502184 before credits** across the three app identities in the open 17:00–18:00 UTC billing interval. Preparation accounted for $0.01303687, preflight $0.00557058, and the supervisor and baseline together $0.60641439. No separate image-build or storage row had posted at the snapshot. The $113 ledger is closed to further dispatch. Its $112.37497816 unspent balance remains earmarked pending billing and retention review, including the $7 storage reserve and $2 delayed-billing hold. Phase 4B funds and holds remain separate.

The fresh volume retained **6,445,405,176 regular-file bytes**. The full listing, including directories and symlinks, totaled 6,445,413,763 bytes. Checkpoint bytes totaled zero. Seven days at $0.09/GiB-month on a 30-day basis projects **$0.1261** before credits. Retention review is due **December 2, 2026 UTC**. [Exposure and retained inventory](../evidence/phase4c-grpo-qualification-20261125/exposure-and-retention.json)
