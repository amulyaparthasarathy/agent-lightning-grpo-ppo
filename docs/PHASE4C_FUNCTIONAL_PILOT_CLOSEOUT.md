# Seed 2001 functional pilot closeout

Allocation P02 completed the development-only GRPO pilot. Independent replay admitted 560 episodes and confirmed 560 rewards. The worker trained from the pinned base for 16 updates and saved the terminal checkpoint. A fresh worker reloaded it. The supervisor persisted the technical gate after the third stage. There were zero final-panel evaluations.

| Stage | Admitted | Confirmed | Strict successes | Worker seconds |
|---|---:|---:|---:|---:|
| baseline | 24 | 24 | 9 | 297.57 |
| training | 512 | 512 | 206 | 2021.86 |
| checkpoint-eval | 24 | 24 | 12 | 455.87 |

Across the 24 paired development cases, strict reward changed on 15, decision correctness on 18, evidence correctness on 12, raw final answer on 18, and tool sequence or arguments on 20. Nine cases moved from failure to success and six from success to failure. Three remained successes and six remained failures. Baseline failures comprised 12 semantic and three evidence errors. The reload had 12 semantic failures. The redacted evidence includes all 24 paired case records.

Offline replay of the raw answers confirms that the step-16 greedy policy answered `not_ready` on all 24 development cases. Decision correctness stayed at 12/24. The strict gain coincided with loss of every ready decision on this panel. The technical gate established sustained functional operation. [Paired cases, class-conditional training signal and evidence limits](PHASE4C_PILOT_LEARNING_DIAGNOSIS.md)

## Update evidence

Each collection admitted eight groups of four samples. Training refreshed step 0 before collection, then refreshed once after each optimizer call. Seventy-six of the 128 four-sample groups contained different rewards. All 16 losses and gradient norms were finite. The loss column is the reported policy-gradient loss. Step time includes collection and optimization.

| Step | Actor call | Refresh | Informative groups / 8 | PG loss | Preclip norm | Step s | Log-prob s | Actor s | Refresh s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 4 | -2.32831e-09 | 1.86056 | 133.89 | 13.81 | 60.20 | 2.78 |
| 2 | 2 | 2 | 5 | -2.21189e-09 | 1.57694 | 96.28 | 9.78 | 35.92 | 2.58 |
| 3 | 3 | 3 | 7 | -1.51922e-08 | 1.90053 | 96.48 | 9.70 | 36.17 | 2.99 |
| 4 | 4 | 4 | 4 | -5.58794e-09 | 1.03599 | 98.01 | 10.22 | 38.90 | 2.89 |
| 5 | 5 | 5 | 7 | 5.00586e-09 | 1.43319 | 101.81 | 10.43 | 40.06 | 2.87 |
| 6 | 6 | 6 | 6 | -1.12923e-08 | 1.50235 | 98.15 | 9.73 | 38.12 | 2.72 |
| 7 | 7 | 7 | 7 | -5.82077e-11 | 1.59966 | 96.31 | 10.27 | 39.19 | 2.55 |
| 8 | 8 | 8 | 3 | 1.86265e-09 | 0.729282 | 100.28 | 10.48 | 39.02 | 2.61 |
| 9 | 9 | 9 | 6 | -5.87897e-09 | 1.16648 | 100.82 | 10.86 | 40.26 | 2.64 |
| 10 | 10 | 10 | 4 | -1.18744e-08 | 1.1027 | 96.33 | 10.23 | 37.88 | 2.67 |
| 11 | 11 | 11 | 4 | -7.10133e-09 | 0.839341 | 99.18 | 10.53 | 38.45 | 2.86 |
| 12 | 12 | 12 | 2 | -1.68802e-09 | 0.708918 | 101.55 | 10.49 | 39.82 | 2.44 |
| 13 | 13 | 13 | 6 | -3.0268e-09 | 1.02095 | 100.76 | 11.19 | 39.67 | 2.51 |
| 14 | 14 | 14 | 6 | 7.79983e-09 | 1.1921 | 99.46 | 10.12 | 40.09 | 3.08 |
| 15 | 15 | 15 | 2 | 8.14907e-09 | 0.758199 | 97.45 | 10.09 | 37.85 | 2.75 |
| 16 | 16 | 16 | 3 | 7.21775e-09 | 0.707802 | 95.63 | 10.19 | 38.40 | 2.61 |

## Checkpoint, time and capacity

The step-16 native inventory digest is `0ff47e0e09dd070bb787b6885a1045f64fe05f7526fd884767b6935368738b7e`. Its 10 hashed files contain 40,146,407,567 bytes (37.39 GiB). The provider listing agreed with the inventory sizes. Training and reload report the same inventory digest before and after load, with distinct call and container identities.

The native checkpoint save took 76.05 s and the fresh-worker load took 30.59 s.

Worker receipts total 46.26 minutes. The supervisor took 47.09 minutes. Reservation through supervisor completion took 52.98 minutes including build, preparation and preflight. The median training step was 98.67 s.

Each model worker used one H100, 16 CPUs and 192 GiB host allocation. CUDA allocator peaks belong to the emitting process. Device samples and provider bucket averages do not establish a whole-device peak. Scratch space belongs to the worker and is excluded from retained volume bytes.

| Stage | Process CUDA allocated peak GiB | Process CUDA reserved peak GiB | Process cgroup peak GiB | Max sampled cgroup usage GiB | Maximum device sample GiB | Minimum sampled scratch free GiB |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.00 | 0.00 | unavailable | 45.49 | 54.11 | 512.00 |
| training | 59.91 | 63.04 | unavailable | 81.05 | 67.72 | 511.96 |
| checkpoint-eval | 0.00 | 0.00 | unavailable | 78.58 | 53.90 | 512.00 |

## Interrupted first allocation

P01 trained from base and admitted 24 baseline and 512 training episodes. Its native step-16 inventory is `6220187be4e3a692a8dd8f43ade7683c0ae1dd7bb1a086e47dd50ebac57c762b`. The provider recorded preemption of the supervisor during the development reload. On interruption the supervisor cancelled that exact worker. Re-entry on a second supervisor container stopped at the ownership guard. The reload admitted zero episodes and the first-seed gate was absent. P01 remains incomplete.

All six P01 calls terminated across seven containers and three stopped apps. Its checkpoint and evidence remain retained on a separate volume. After reconciling the first allocation, the operator reserved a fresh $103 for P02. The merged source differs from P01 only in documentation and has the same packaged runtime digest. The four local study execution tests passed again before replacement. Both runs used seed 2001, the same data, frozen streams and numerical settings. P02 started from the base policy.

[P01 partial evidence](../evidence/phase4c-functional-pilot/P01.json)

## Exposure and retention

Before P02 setup, P01 encumbered $30.80. Reserving another $103 brought exposure to $133.80 of the $150 cumulative ceiling, before credits. At 2026-11-26T03:22:33.894568+00:00, provider billing rows attributable to this allocation totaled USD 4.90468350 before credits, with no posted volume storage charge yet. Invoice timing leaves USD 26.23531650 in conservative holds.

| Unresolved or future obligation | USD |
|---|---:|
| remaining build preflight compute collection | 23.31531650 |
| retention | 0.92 |
| other delayed | 2 |

The holds cover remaining build, preflight, collection and padded terminal compute net of posted charges, seven days of retained storage and delayed billing. Unencumbered ceiling is USD 88.06000000. The campaign closed on first success with zero further dispatch authority. Historical qualification and Phase 4B holds remain on their own ledgers.

All 6 exact calls terminated, 3 owned apps stopped, and zero owned containers remained. The fresh volume retained 46,702,825,648 bytes (43.50 GiB). The collector recorded 92,937,591 bytes across preparation, preflight and terminal downloads. Additional status reads and billing queries have no separate transfer meter here. The pilot checkpoint and evidence have a 2026-12-03 UTC retention review. The separate historical review remains 2026-12-02 UTC.

Cumulative posted pilot charges were USD 9.50754026. Conservative cumulative holds were USD 52.43245974. The two volumes retained 93,397,672,417 bytes in aggregate. Campaign opening through the last supervisor finish took 119.11 minutes including P01 and its reconciliation. Across both allocations, the three-stage worker receipts total 89.61 minutes. P01 posted USD 4.60285676 and held USD 26.19714324. Total elapsed time from campaign reservation through independent review and financial reconciliation was 132.99 minutes. Both pilot volumes have a December 3 UTC review.

## Reproducibility

The tested implementation at `9d12ce32ff05b22bf7a3fd5295bb09819f5a6d8d` passed 202 reader tests, 74 pinned CPU tests and four study execution tests. PR #12 integrated it at `17889f37651d57653cfc0dbe12d06304ac27371a`. The four study execution tests ran again on that exact P02 source with zero failures or skips in 81.272 seconds. The fresh preparation result and durable CPU preflight bound the three-stage execution plan. The independent review replayed collected traces and verified the terminal technical gate. Exact identifying handles, billing rows and raw evidence are retained privately.

| Commitment | SHA-256 |
|---|---|
| runtime manifest | `d35194e68391533ac347614de3c15486031bc371c92ec148af38e945b935ea62` |
| owner profile | `f00e53a3de0114630fefcdb66ee3700e05b927fa5526810178aff0acebc55ca4` |
| preparation plan | `be0d4436cc4275a71494f89ba1fa1cbed58953dae965f344918a6c05b1adbd10` |
| plan | `238678dbf18c5659e77b556bc24220edfd13e5bb7c03b7b936a92fdef213017a` |
| assets | `e4c46350891b52f88ebe0ce0cd73be93e090b3c2b03c0315f0f01c5ab2d2df89` |
| data manifest | `c7d07942dda1ee81b8797ce9a59255e7f26bca712e668c18cd1a936d83d6e8ab` |
| model manifest | `af66f1551b619488d35e04a4d8cb87f556a619fb0c096d03e22c406ea2fb22e3` |
| first seed gate | `530445cb9b5039d1d9ff6d647efe5d7868e053a5b31daf152ee7b60ad738d149` |
| independent report | `839833c614474b494b348fd7b3fdb637b7da052ae7839ffce9aa1a81bf22bf8b` |

[Redacted stage, inventory, timing and accounting evidence](../evidence/phase4c-functional-pilot/P02.json) · [Implementation and local verification](PHASE4C_FUNCTIONAL_PILOT.md)
