# Phase 4C GRPO operational qualification

**A03 completed on November 25, 2026 UTC.** Independent review of the collected traces and durable receipts admitted 144 episodes, confirmed 144 rewards and verified two actor optimizer calls. Training saved the terminal step-2 checkpoint and a fresh worker reloaded it. Both greedy development evaluations had 9 strict successes out of 24. These results establish operational qualification for the two-update protocol. They establish no learning improvement or longer-study qualification. Final-panel model evaluations totaled zero. [Redacted A03 evidence](../evidence/phase4c-grpo-cumulative/A03.json)

| Stage | Admitted / confirmed | Strict successes | Failure counts | Actual actor updates | Worker receipt seconds |
|---|---:|---:|---|---:|---:|
| Fresh-base development baseline | 24 / 24 | 9 | 12 semantic, 3 evidence | 0 | 259.661 |
| Sampled training probe | 32 / 32 | 12 | 11 semantic, 7 format, 2 evidence | 0 | 267.159 |
| Fresh-base training | 64 / 64 | 16 | 29 semantic, 5 format, 14 evidence | 2 | 539.162 |
| Fresh-worker reload development | 24 / 24 | 9 | 12 semantic, 3 evidence | 0 | 343.334 |

## Optimizer, checkpoint and capacity

The probe had 5 informative groups out of 8. Training had 3 of 8 in each update. Actual actor optimizer calls were 1 and 2, with finite preclip gradient norms of 1.175673 and 0.907641. Actor policy-gradient losses were −3.0268e−9 and 8.6729e−9. The training worker refreshed weights at steps 0, 1 and 2. Stored and recomputed log probabilities were compared on 9,044 and 9,508 target tokens, with maximum absolute deltas of 0.364075 and 0.415974.

| Measured operation | Update 1 | Update 2 |
|---|---:|---:|
| Old log-probability recomputation | 13.025 s | 9.770 s |
| Actor update | 44.774 s | 37.152 s |
| Weight refresh | 2.026 s | 2.028 s |

The native step-2 inventory commits **40,146,407,567 hashed payload bytes**, including loader state. Its digest is `932f4e21aad6e9f347706193d5485851ddaa2ae16af942757f9d06771f9e7e1a`. The independent reviewer joined the inventory, training configuration hash, training receipt and reload receipt. The mounted volume inventory matched the recorded file sizes. Training and reload had distinct provider calls and containers. The interval between the last refresh and the checkpoint-save return was 55.562 s. An isolated write timer and an isolated reload timer were unavailable.

The lane was one H100 with 16 CPUs and 192 GiB host memory. Training's largest sampled whole-device reading was **67.73 GiB**. The trainer audit recorded per-process CUDA allocator peaks of 59.91 GiB allocated and 63.04 GiB reserved. The highest sampled v1 memory-controller reading was **81.44 GiB**. Neither the true whole-device peak nor the cgroup high-water counter was available. The highest complete 60-second provider averages during training were 57.07 GiB GPU and 76.06 GiB host. Actor parameter and optimizer offload were configured. Offload traffic was unmeasured. Training entered with 512 GiB of free worker-local scratch and the lowest sampled reading was 511.994 GiB. The checkpoint and retained evidence reside on the volume. [Stage measurements and identity commitments](../evidence/phase4c-grpo-cumulative/A03.json)

## Allocation lineage

| Allocation | Actual execution source | Preparation plan SHA-256 | Asset-bound plan SHA-256 | Outcome |
|---|---|---|---|---|
| A01 | `547053db6636fdc1475f0d53f8bf580fed751bbd` | `fa8ff85ef86e8778675f0cb8533392074a798561c83606f89cfcd37c6b6d9fcd` | `8098ce37276f88860f2ae2a7530a8671edd8f8997e7e7d587b731a299c6eeb51` | Baseline finalization failed. 24 diagnostic episodes, 24 confirmed rewards, 9 strict successes, zero admitted and zero updates |
| A02 | `d66b5d112587f36ca1a2708577d9218f7db39e29` | `97afeb534d1b4c2140a805ba36b6238e366dd703e8000218b3fd50aad81112d4` | `3d03a28a35fb903d380262a0254c3ebbd73f203fa891ac2f9c6820b5a3466e2d` | Fresh preparation passed. Dependency drift stopped CPU preflight before GPU dispatch |
| A03 | `7fdb2b1e0ce2a05f03980cfd3a1e07a31937ca6a` | `875d011a79a4ccce03b6c28cdac1e29a815381381cae3ae4a98f8faa8cc617b9` | `a899410f14650150917e0a68d9d9d2897f50d1c560106baf41e8528163cf6a39` | All four stages passed |

The original tested runtime ancestor was `852e01d141b66232c73ca46e6e7bf4b9dfb0e854`. Historical Phase 4B used `bc81c1fb0def27b15c56cdd8abb01098e7e1f8d5`. PR #9 merged the path correction at `6ef174eef411f172e2eb9bb5f0d18f963d53b731`. After the A02 cold build, preflight correctly rejected newer ml-dtypes and OpenCV wheels. PR #10 restored the exact reviewed wheels in the reader image and merged at `b7e2bc0d2714b5178d9ccab3f99eacaf166949d3`. Local tests installed the actual failed and corrected Linux wheels. Twenty focused wheel, dependency and cold-import tests and 202 reader-gate tests passed. Each successor had its own experiment, resources and internal attempt 1. A04 was never allocated. [A01 evidence](PHASE4C_GRPO_QUALIFICATION_A01.md) · [A02 evidence](../evidence/phase4c-grpo-cumulative/A02.json) · [Dependency repair](PHASE4C_A02_DEPENDENCY_REPAIR.md)

## Provider and financial closeout

Of 13 owned calls, 4 belonged to A01, 2 to A02, and 7 to A03. All reached terminal states. A03's preparation, preflight, supervisor and four workers each returned successfully. All eight apps across the campaign stopped, and the final inventory showed zero owned containers. The terminal supervisor index supplied the ownership join. Collection transferred 40,035,399 bytes with zero errors. The offline reviewer independently recomputed admission from the collected evidence. The private handoff preserves exact IDs and the public evidence provides their hash commitments.

| Allocation | Attributable usage before credits | Regular-file bytes retained |
|---|---:|---:|
| A01 | $0.62502184 | 6,445,405,176 |
| A02 | $0.84622284 | 6,435,944,452 |
| A03 | $2.52933395 | 46,633,520,042 |
| **Campaign** | **$4.00057863** | **59,514,869,670** |

The 21:00 UTC read covers the completed usage hours. Each provider object/hour row was attributed once. The cumulative ceiling was $113 before credits. Each successor reserved $89 in full before dispatch, including $58 compute/supervision and $31 ancillary allowances. The authenticated all-resource prices were $6.2428/hour for the worker and $0.0633/hour for supervision. The validator used $7/hour over both bounded deadlines. Phase 4B funds and holds stayed separate.

The ledger keeps **$45.61444321** for possible further billing and seven-day storage. This is a conservative reserve. Outstanding call exposure is **$0**. A02's stopped CPU apps used a $3 combined gross envelope. A03 retains the $15 build and $5 preparation/preflight ceilings plus a padded terminal compute estimate. Both combined envelopes deduct posted gross once. Retention, unposted collection and other delayed billing have separate holds. The ledger leaves **$63.38497816 unencumbered**. With qualification complete, further dispatch authority is **$0**. Review all three retained volumes and the checkpoint on **December 2, 2026 UTC**. [Cumulative accounting](../evidence/phase4c-grpo-cumulative/campaign-closeout.json)
