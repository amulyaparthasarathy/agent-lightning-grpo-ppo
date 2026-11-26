# What seed 2001 learned on the development panel

**All 24 raw step-16 greedy answers say `not_ready` and cite no supporting runs.** Strict successes rose from 9 to 12. Decision accuracy stayed at 12/24. The base correctly identified nine ready cases and three negative cases. The reload correctly identified only the 12 negatives. The training samples show that valid `ready` answers existed throughout much of training and became less frequent. The only greedy measurements on this panel are fresh base and step 16.

## Greedy development replay

| Actual class | Cases | Base ready | Base not_ready | Reload ready | Reload not_ready |
|---|---:|---:|---:|---:|---:|
| ready | 12 | 9 | 3 | 0 | 12 |
| not_ready | 12 | 9 | 3 | 0 | 12 |

| Family | Actual | Cases | Base decisions correct | Reload decisions correct | Base strict | Reload strict |
|---|---|---:|---:|---:|---:|---:|
| recent_usable | ready | 6 | 3 | 0 | 3 | 0 |
| failed_newer_usable_older | ready | 6 | 6 | 0 | 3 | 0 |
| stale_success | not_ready | 4 | 0 | 4 | 0 | 4 |
| recent_incomplete | not_ready | 3 | 0 | 3 | 0 | 3 |
| recent_integrity_failed | not_ready | 3 | 1 | 3 | 1 | 3 |
| empty_history | not_ready | 2 | 2 | 2 | 2 | 2 |

All 48 answers parsed, named the correct target and time, cited the correct record identifiers, and followed observations of the target asset, policy and complete inventory. Evidence correctness conditional on the **answer's decision** rose from 12/24 to 24/24. A `ready` answer needs a nonempty, duplicate-free subset of usable runs. For `not_ready`, the observed complete inventory and an empty supporting list satisfy the run-evidence check. The verifier separately checks whether a usable run exists. Every one of the 12 false negatives passed the evidence check and earned **zero** strict reward. [Verifier](../experiment/recovery_task.py) · [All 24 paired records, including inventory ages, RPO, selections and trace hashes](../evidence/phase4c-functional-pilot/diagnosis/development.json)

The case suffixes below identify every transition. A recovery point must be within the inclusive RPO at the requested time and have succeeded with passed integrity.

| Base → reload strict reward | Cases | What changed |
|---|---|---|
| 1 → 0 | `529d506be0`, `96ea63c827`, `cf2994fed4` | Three recent-usable decisions lost valid support. The base cited a point exactly on the RPO boundary in the first case and valid witnesses in each multi-witness inventory. |
| 1 → 0 | `3c9386f800`, `6c1c74a0e6`, `8a187cb514` | The base selected an older usable point despite a newer failure. The reload rejected all three, including one boundary point. |
| 0 → 1 | `121050b58a`, `5db1a70832`, `803d4459c3`, `b5fac01e6c` | Four stale-success inventories. Each base selection included stale support. In the last case the point was 721 minutes old against a 720-minute limit. |
| 0 → 1 | `01f40b8f3c`, `9b22a6dd87`, `daf12ab574` | A recent run was incomplete in each inventory. The base cited older successful points outside the RPO in all three. |
| 0 → 1 | `6a7e0cc126`, `77fa9e56ce` | The base cited a run with pending integrity in the first and a stale point in the second. Neither inventory had a usable run. |
| 0 → 0 | `92516b687c`, `c25ac45067`, `e769799570` | **Three hidden decision regressions.** Each base answer correctly said ready and cited one usable point plus two, two and three stale points respectively. Each reload said not_ready despite observing the witness. |
| 0 → 0 | `aa26cb7656`, `d867a4a6e7`, `fb62402846` | Both policies missed a usable point. |
| 1 → 1 | `2c03f8daf4`, `2e641a2d24`, `87f449f853` | Both policies correctly rejected two empty inventories and one inventory with no integrity-qualified, in-window point. |

On the **single temporal sibling pair**, the same passed run was 436 minutes old at the earlier request and 1,131 minutes old at the later request, against a 720-minute RPO. In chronological order the truth was (ready, not_ready), base answered (not_ready, ready), and reload answered (not_ready, not_ready). Only the later sibling gained strict reward.

## Sampled training signal

P02 sampled four real trajectories per question, eight questions per collection, with equal class totals across the run. The audit joins collection 1–16 to behavior policy 0–15 and the subsequent actor call 1–16. The 128 questions are distinct. The table counts outcomes at temperature 1 on those questions. The class composition varies from one to six ready questions per collection. Family composition, exclusive failure classes, assessed denominators, positive advantages, generated-token totals and medians are available for **each class in every collection**. [Collection table](../evidence/phase4c-functional-pilot/diagnosis/collections.json) · [512 trajectories with reconstructed advantages](../evidence/phase4c-functional-pilot/diagnosis/training.csv)

| Collections | True class | Questions | Samples | Parsed ready / not_ready / invalid or absent | Strict | All-zero / mixed / all-one groups | Positive advantages |
|---|---|---:|---:|---|---:|---|---:|
| 1–8 | ready | 34 | 136 | 60 / 62 / 14 | 48 | 11 / 21 / 2 | 40 |
| 9–16 | ready | 30 | 120 | 35 / 80 / 5 | 19 | 19 / 11 / 0 | 19 |
| 1–8 | not_ready | 30 | 120 | 55 / 48 / 17 | 48 | 5 / 22 / 3 | 36 |
| 9–16 | not_ready | 34 | 136 | 39 / 92 / 5 | 91 | 1 / 22 / 11 | 47 |

Strict ready success fell from **48/136 (35.3%) to 19/120 (15.8%)**. Both positive families moved in this direction within their own denominators. Recent-usable went from 28/76 to 10/52, and failed-newer/usable-older from 20/60 to 9/68. On true negatives, strict success rose from 48/120 to 91/136. Ready all-zero groups rose from 11/34 to 19/30 and negative all-one groups from 3/30 to 11/34. These shifts are uneven. Collection 12 had zero strict ready successes, collection 13 had seven, and collection 15 had four. Collection 16 sampled five ready answers, one on a positive case and four on negative cases, with zero strict ready successes. Question difficulty, variants, rollout sampling and behavior-policy version all vary across collections. These data support a sampled shift toward the negative response. They leave its precise onset and step-16 stochastic support unresolved.

**Which trajectories supplied positive signal?** Across 512 samples, 67/256 true positives and 139/256 true negatives earned strict reward. Of 128 groups, 36 were all-zero, 76 mixed and 16 all-one. In the positive class the split was 30/32/2, in the negative class 6/44/14. Only the mixed groups supplied reward contrast. Of 142 positive-advantage trajectories, 59 were ready and 83 were not_ready. The other eight ready and 56 negative successes had zero advantage. Reconstructed sums of positive standardized advantages were 50.55 and 69.02 respectively. For example, collection 2, case `c-c1b77daafa`, trace `0fa63e28eb2e…`, cited one valid run and was the sole success among four (+1.499997). Median total generated tokens per successful trajectory were 276 for ready and 264 for not_ready. A shorter final evidence list does not imply that tool interactions were short or that token count determined reinforcement.

**What failed?** The verifier classified 206/512 as correct, 236/512 as semantic, 29/512 as evidence, 35/512 as format, and 6/512 as terminal action failures. Of 471 schema-valid answers, two supplied the wrong evaluation time. Their decision strings are retained and their decision/evidence correctness is unassessed. Among the remaining 469, 234 had a wrong decision and 124 failed the evidence check, with overlap. Of 35 format failures, 28 failed strict JSON syntax and seven parsed as JSON but violated the answer contract. The latter included three disallowed decision values, two string-valued supporting lists and two objects with different fields. All 41 format or terminal action outcomes have no parsed decision. The six terminal failures have null format diagnostics. Every admitted trajectory, including those six zero rewards, participates in its group.

- All 28 correct ready decisions that missed strict reward cited at least one stale point. Four also cited a failed-integrity point. In collection 2, `c-067ea89e85` (trace `3bc238d54575…`) observed the target records and listed four runs, including stale and integrity-failed support. Across all 122 unsuccessful parsed ready answers, 119 selected stale support and 12 selected failed or pending integrity, with overlap. These are observable eligibility errors. The model's internal time calculation is unknown.
- Of 469 assessed answers, the asset-record check failed zero times, policy-record twice and inventory-ID once. On the negative case `c-bc69e082f2` (collection 13, trace `2bbbcb29d474…`), the decision and empty list were correct, the complete inventory had been observed, and the inventory identifier was wrong. This was the sole decision-correct negative evidence failure.
- The collector recorded 53 typed tool errors across 47/512 trajectories, including 22 unknown policies and 20 unknown tools. There were 22 complete-recovered outcomes and six terminal policy failures. In collection 3, `c-2bfb25cedb` (trace `fc681e5bc3cd…`) ended after two invalid-as-of actions, with no assessable final answer. Overall, 491/512 trajectories observed all three required target records. Missing tool evidence does not explain the terminal development pattern.

**Mechanism suggested by the evidence.** A ready success requires locating a witness and excluding every ineligible citation. A negative success can reuse observed identifiers and give an empty list. The later samples offered fewer opportunities to reinforce valid ready behavior and more to reinforce the negative form. Constant groups supplied zero relative advantage on either side. This asymmetry is a plausible contributor to the terminal preference. Counts and scalar advantages cannot attribute a parameter update to a response form, isolate the effects of question mix, or establish forgetting outside this panel. The unchanged strict verifier rejected every wrong terminal decision.

## What the optimizer measurements mean

The reconstructed trajectory advantage is `(reward − group_mean) / (sample_sd + 1e-6)` with four binary rewards, unbiased sample SD and float32 arithmetic. Each generated policy token carries weight `1 / (32 × generated tokens in its real trajectory)`, across fragments. Observations and padding carry zero policy weight. With one minibatch and one epoch, the pre-step update ratio near one makes the scalar loss close to the negative mean advantage. Centering cancels that scalar. The gradient also involves each sampled token's log-probability derivative, which varies across trajectories and need not cancel. All 16 preclip gradient norms were positive (0.708–1.901 rounded). PG loss ranged from −1.52e−8 to +8.15e−9. [Numerical implementation](../experiment/matched_runtime.py) · [Recorded update sequence](PHASE4C_FUNCTIONAL_PILOT_CLOSEOUT.md#update-evidence)

Both recorded actor clipping fractions and the rollout-weighted `actor/ppo_kl` (negative clamped current-minus-old log-probability delta) were zero at every update. This diagnostic compares the actor with its recomputed old probabilities within the update. Separately, old trainer minus sampler log probabilities on 147,374 target tokens had per-collection means from −0.000320 to +0.000180 and per-collection maximum absolute differences from 0.249 to 0.818. Neither series measures cumulative movement from the base. Reference/base KL and entropy are unavailable. Actor KL loss and reward KL were off, and entropy calculation was disabled. [Probability comparison](../experiment/recovery_admission.py) · [Frozen numerical settings](PHASE4C_LEARNING_STUDY_PROPOSAL.md)

## Interrupted allocation compared separately

P01 executed `9d12ce32ff05b22bf7a3fd5295bb09819f5a6d8d`. P02 executed the PR #12 merge `17889f37651d57653cfc0dbe12d06304ac27371a`. Only the pilot documentation differs between these execution trees. Both plans record packaged-runtime digest `d35194e68391533ac347614de3c15486031bc371c92ec148af38e945b935ea62`. The frozen data manifest, plan study declaration, seed streams (including registrations), model revision and all nine base-model file size/hash pairs agree. The prepared model manifests differ only at `prepared_at`. Across the two available paired stages, parsed resolved configurations differ in 15 fields each, all associated with allocation paths, run names, request commitment or local service address. These include the mounted chat-template path. Its file bytes agree. The step-16 inventories have different actor model and optimizer file hashes. Serialized file differences establish different bytes without quantifying tensor distance. Only inventory metadata was read. [Comparison fields and source hashes](../evidence/phase4c-functional-pilot/diagnosis/summary.json)

Both greedy baselines made the same 24 decisions, with the same tool names, argument strings and returned observation contents on every pair. Their first-generation inputs matched on all 24. P01 had 8 strict successes and P02 had 9. Eight cases succeeded twice, 15 failed twice and one gained. On `529d506be0` both answered ready. P01 included two stale runs beside the valid boundary witness and P02 cited the witness alone. The raw final answer changed only there. Three cases differed in reported first-generation completion length, including that case (42 vs 130 tokens). This shows a generation difference before optimization in separately launched workers. Neither matching configured streams nor disabled full determinism identifies its cause.

P01 trained on the same 128 scheduled questions at seed 2001. Its 228/512 successes comprised 69/256 ready and 159/256 not_ready, with 75 mixed groups. P02 started afresh and recorded 206/512 (67 and 139), with 76 mixed groups. Supervisor preemption interrupted the P01 reload after training. Its 536 admitted episodes and separate checkpoint support a partial comparison. The sole completed campaign result is P02. [P01 receipt summary](../evidence/phase4c-functional-pilot/P01.json) · [P02 receipt summary](../evidence/phase4c-functional-pilot/P02.json)

| Allocation | Baseline strict | Training strict | Training strict on ready / not_ready (each of 256) | Mixed groups | Actor calls | Admitted reload |
|---|---:|---:|---|---:|---:|---|
| P01 | 8/24 | 228/512 | 69 / 159 | 75/128 | 16 | 0 episodes |
| P02 | 9/24 | 206/512 | 67 / 139 | 76/128 | 16 | 12/24 strict |

P01 admitted 536 episodes before supervisor preemption interrupted reload. P02 was the sole permitted replacement and started from base. Both runs used seed 2001. Full determinism was disabled, but the retained records do not identify the cause of the baseline or training differences. The campaign result uses P02 alone.

## Proposed next intervention

**Test a witness-selection warm start in a separately authorized development experiment.** Use a single small supervised pass from the same base on 32 training-only, public-tool-solver demonstrations, 16 ready and 16 not_ready. Include boundary, older-usable, stale and integrity cases. Use the existing prompt, tool observations, answer format and strict verifier. Positive examples show a minimal valid witness. Hard negatives preserve the rejection decision. This directly targets the observed selection errors and the scarcity of rewarded positives.

The smallest evaluation is one greedy replay of the same 24 development questions after that pass, paired against the retained base. Set a decision threshold before execution of **at least 8/12 strict in each class**, with all 24 answers assessable. Report both sibling decisions, per-family counts and gains/regressions even if the threshold fails. This screens whether the intervention produces both grounded decisions before paying for GRPO. It costs supervised examples and compute, risks imitation of the solver's action order, and cannot establish persistence through later RL or generalization from a reused development panel. This proposal has no execution authority.

## Offline reproduction and scope

From the evidence worktree, set `PY` to the existing reader CPU interpreter and `PILOT` to the retained directory containing P01 and P02. The original evidence commits are `194e9e8` and `2c9046e`. All outputs below are derived analysis.

```bash
export MODAL_CONFIG_PATH=/dev/null
export PYTHONPATH=tests/offline_guard:.:tests
out=$(mktemp -d)
"$PY" scripts/diagnose_functional_pilot.py "$PILOT/P01" "$PILOT/P02" "$out"
diff -qr evidence/phase4c-functional-pilot/diagnosis "$out"
"$PY" -m unittest -v test_functional_pilot_diagnosis test_recovery_task test_recovery_learning test_recovery_calibration
```

The script verifies the asset-bound plans and independent report hashes against the public summaries. It replays stage requests, conversation histories, served responses, tool observations, verifier results, admissions, refreshes and collection manifests. It compares each worker's stored stage report with that replay. The replay checks the completed episode’s raw answer against the last generation and its served-response evidence. A separate strict parse of that answer supplies the decision string. The public-tool solver cross-checks labels and qualifying runs. Case IDs join greedy pairs. Rollout IDs join actual collection slots to trace rows. The published trace SHA-256 values identify exact private JSONL bytes. Aggregate trace and audit commitments hash sorted lists of file hashes using canonical JSON. Answer fingerprints hash the canonical JSON encoding of the raw answer string. CSV blanks and JSON nulls preserve unavailable diagnostics. `selected_issues` describes the selected runs against the target inventory, and `observed` records which target observations the trajectory actually saw. The JSON collection counts use all trajectories for exclusive outcomes and separate assessed denominators for overlapping decision/evidence errors.

The P02 plan, assets and inventory commitments are `238678dbf18c5659e77b556bc24220edfd13e5bb7c03b7b936a92fdef213017a`, `e4c46350891b52f88ebe0ce0cd73be93e090b3c2b03c0315f0f01c5ab2d2df89` and `0ff47e0e09dd070bb787b6885a1045f64fe05f7526fd884767b6935368738b7e`. Fresh-worker reload and the stage-three technical gate completed the mandate. No inference, model load, training, provider access or new final-panel result was needed for this diagnosis. The closed $150 mandate has zero dispatch authority. At the recorded billing read, posted gross charges were $9.50754026, conservative holds $52.43245974 and unencumbered ceiling $88.06. The December 2, 2026 UTC historical and December 3, 2026 UTC pilot retention reviews, including both pilot checkpoints, stand as recorded. [Ledger timestamps and retention scope](PHASE4C_FUNCTIONAL_PILOT_CLOSEOUT.md#exposure-and-retention)
