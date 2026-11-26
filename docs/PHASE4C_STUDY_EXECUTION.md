# Three-seed GRPO recovery study executor

The reader supports the eleven-stage recovery study at implementation commit `b3912e9cb8382f3121ab069d370d23e829fa4d93`. The example profile disables spending. The study has no measured policy results. A03 remains the live operational evidence for two updates. [Offline profile](../examples/reader-profile.recovery-study.synthetic.json) · [A03 closeout](PHASE4C_GRPO_CAMPAIGN_CLOSEOUT.md)

## Protocol and execution

The execution phase is `recovery-learning-study-execution-v1`. It references the archived proposal bytes with SHA-256 `a49003098176e66c896a14879eb691209a16bbe30d6f65bc46bf228e9c61be5e`. The reader reconstructs the execution plan from the owner profile, source and asset manifest. It checks the proposal file and the frozen agent, generator, prompt, trace runtime, upstream lock and analysis source commitments. The implementation adds stage dispatch, evidence checks and checkpoint timers. The numerical functions and reviewed dependency wheels retain their qualification versions. [Frozen declaration](recovery-learning-study-v1.json) · [Execution binding](../experiment/study_execution.py)

| Order | Policy and input | Episodes | Technical boundary |
|---|---|---:|---|
| 1 | Fresh base, development | 24 | Greedy evaluation, both loaders on development |
| 2–3 | Fresh seed 2001 training, then terminal development | 512 + 24 | Sixteen collections, actor optimizer calls 1–16, refreshes 0–16, one step-16 checkpoint, distinct reload worker |
| 4–7 | Fresh seeds 2002 and 2003, each followed by terminal development | 2 × (512 + 24) | The supervisor admits the first replicate before dispatching seed 2002 |
| 8–11 | Fresh base and all three terminal policies, final | 4 × 64 | Commit final release first, bind both loaders to final, use greedy evaluation seed 9001 |

Each training seed consumes its frozen permutation of 128 questions, eight per update, with four samples per question. Every sampled group passes trajectory, token, mask, reward-delivery and fragment admission before optimization. Training binds its eager validation input to development and disables validation callbacks. Every evaluation binds both eager inputs to its selected panel. The final-only verifier checks the manifest, world and final Parquet. CPU preflight composes all eleven configurations against the observed volume. Every worker checks its request, preflight, canonical mount and reservation. [Schedules](../experiment/recovery_schedules.json) · [Launch configuration](../experiment/matched_launch.py)

The supervisor persists the release after the first seven admissions. The receipt commits the pinned base, task, model/data assets, execution source, three training workers and terminal inventory hashes, admitted receipt digests, analysis sources, four final run IDs and final-inclusive owner authority. The supervisor writes it to the volume and records its digest in the registry. The final worker checks both. The launcher checks the release before opening model-facing inputs and checks the mounted terminal policy against the released hash before reload. Collection checks each final worker's release digest, raw evidence, event sequences, native inventory, configuration and call identities. Large stage receipts stay on the volume. Unknown dispatch retains its full reservation. [Supervisor](../experiment/matched_supervisor.py) · [Worker](../experiment/matched_remote.py) · [Independent reviewer](../experiment/recovery_learning_evidence.py)

The final report joins each confirmed strict reward to its stage, policy and case. A complete table has 256 entries. The frozen analysis resamples sixteen four-case policy blocks jointly across the four policies, keeping temporal siblings and sample vectors together. It reports three paired effects, the seed mean, descriptive sample standard deviation and the 10,000-draw conditional synthetic case-mix intervals at seed 329. The secondary report retains episode and family counts, assessability denominators, temporal pairs, typed failures and recovery, sampled tools, confirmed observations, groups, tokens and stage/update timing. Billing uses attributable provider object-hours and separately states coverage of delayed charges. Generalization is within the synthetic generator. [Analysis declaration](PHASE4C_LEARNING_STUDY_PROPOSAL.md)

## Offline verification

The reader, pinned numerical and scripted study commands ran at the implementation commit above. The additional qualification, harness and source-preservation cases ran together with the study at the earlier tested commit, `28cc78eae9ca4c852416584248ad858cc22a8a94`. The repository's pinned tests require its existing CPU environment, upstream source checkout, reviewed wheel fixtures and hash-checked padding helper. The harness uses the locked reader CPU environment. The test inputs contain synthetic traces, native-format fixture files and simulated provider receipts. The numerical test performs real CPU backward and AdamW steps on a tiny module.

```bash
bash scripts/test_reader_offline.sh
OMP_NUM_THREADS=1 bash scripts/test_matched_training.sh
PYTHONPATH=.:tests "$TASK_TEST_PYTHON" -m unittest -v test_study_execution
# Additional preservation run at 28cc78e
PYTHONPATH=.:tests "$TASK_TEST_PYTHON" -m unittest -v test_study_execution \
  test_recovery_learning_evidence.RecoveryCollectionTests.test_grpo_aliased_volume \
  test_recovery_harness.QualificationHarnessTests test_phase3_source
```

| Verification gate | Passed | Failed | Skipped |
|---|---:|---:|---:|
| Reader offline (39 + 15 + 148) | 202 | 0 | 0 |
| Pinned matched training, including its nested nine-case configuration suite | 74 | 0 | 0 |
| Study flow, interrupted dispatch and gross billing | 3 | 0 | 0 |
| Earlier study and preservation run at `28cc78e` | 15 | 0 | 0 |

For the second command the environment supplied `MATCHED_TEST_PYTHON`, `AGL_PINNED_SOURCES`, `AGL_VLLM_WHEEL` and `AGL_TRANSFORMERS_WHEELS`. The nested pinned Hydra suite composed all eleven study stages without creating outputs. The pinned example actually opened both bound Parquet arguments for each of the eleven stage requests with its engine intercepted. The upstream loader replay checked every scheduled batch for 2001, 2002 and 2003, as well as the qualification seed. The numerical fixture kept one tiny actor and AdamW optimizer alive through sixteen successive admitted updates. The end-to-end test exercised the real scheduler, dispatch adapter, worker entry, file-bound request, launcher, stage completion, release persistence and independent review. It generated synthetic stage traces from one scripted development example and removed the provider mount before review. Final routing used synthetic records only. [Flow and failure tests](../tests/test_study_execution.py) · [Numerical tests](../tests/test_matched_training_pinned.py)

The first full flow exposed an incorrect reporting assumption that every evaluation asset has a temporal sibling. The frozen generator has one tagged pair on development and four on final. The corrected metric groups only the `time_sibling` cases, counts both decisions correct over jointly assessable pairs, and reports total pairs even when no decision is assessable. A separate pinned-entrypoint assertion initially indexed the upstream list of rows as a columnar table. Reading each returned row fixed the test and preserved the upstream code. The regression also exercises a missing release anchor, wrong training sibling, wrong global step, interrupted submission, underfunded profile, duplicate billing hour, missing/duplicate final cases and unconfirmed zero. [Redacted verification receipt](../evidence/recovery-study-offline/verification.json)

Generate the preparation template in a clean checkout of the tested source. The manifest-bound demonstration used the retained A03 asset manifest as offline input. A paid allocation requires its own preparation result, plan and matching durable preflight. Both demonstrated templates had `fixture=true`, `study_authority=disabled` and `training_admissible=false`. [Template identities](../evidence/recovery-study-offline/verification.json)

```bash
profile=examples/reader-profile.recovery-study.synthetic.json
python scripts/reader_training.py plan --profile "$profile" --out .local/study-preparation.json
python scripts/reader_training.py validate --profile "$profile" --plan .local/study-preparation.json
```

## A03 paired diagnostic

The offline comparison reverified the retained baseline and reload traces against A03 plan `a899410f14650150917e0a68d9d9d2897f50d1c560106baf41e8528163cf6a39`. The same nine cases succeeded and the same fifteen failed. All 24 failure categories and raw final answers matched. **One case changed its ordered tool names**. The comparison excludes generated tool-call IDs and also compares parsed argument values. This is an observation about two greedy evaluations on development. Run `python scripts/recovery_qualification_pairs.py "$A03_COLLECTED"` with the private collection directory. [Paired counts](../evidence/recovery-study-offline/A03-development-pairs.json)

## Capacity and cost worksheet

A03 used 539.162 seconds for two training updates. Total step times were 111.721 and 93.180 seconds. Actor updates took 44.774 and 37.152 seconds, log-probability recomputation 13.025 and 9.770, and weight refreshes 2.026 and 2.028. The native payload including loader state was 40,146,407,567 bytes (37.39 GiB). The last refresh to checkpoint-save return took 55.562 seconds. The new study records native save/load elapsed time. The model-facing sustained workload, actual sixteen-update checkpoint and evaluation throughput still require live measurement. [A03 capacity evidence](../evidence/phase4c-grpo-cumulative/A03.json)

A03 training sampled 67.73 GiB whole-device and 81.44 GiB host cgroup usage. Process CUDA allocator peaks were 59.91 GiB allocated and 63.04 GiB reserved. True whole-device peak, cgroup high-water mark, offload traffic and isolated live save/load times were unavailable. Free worker-local scratch started at 512 GiB and reached a sampled minimum of 511.994 GiB. The worker retains its 100-GiB entry minimum. Volume storage and local scratch have separate inventories.

| Prospective allocation, zero credits | Basis | USD |
|---|---|---:|
| Expected compute and supervisor | Three training workers at 539.162 + 14 × 102.4505 s; development at 259.661 + 3 × 343.334 s; scale the four development times by 64/24 for final. About 2.96 lane-hours at A03 rates of $6.2428 and $0.0633 per hour | 18.65 |
| Expected seven-day incremental storage | Three terminal payloads total 112.17 GiB; add 16 GiB model/cache and 2 GiB evidence | 2.73 |
| Expected total range | Rounded compute/storage estimate plus $0–15 build, $0–5 preparation/preflight and up to $0.08 egress. Startup, actual episode length and delayed attribution remain uncertain | **about 21–42** |
| Conservative compute/supervisor reservation | Three × 14,400 s training plus eight × 1,800 s evaluation = 57,600 worker s. Supervisor 58,200 s. Value both at $7/h, round $225.17 up | **226** |
| Ancillary reserves | Build $15, preparation/preflight $5, storage $7, collection $2, delayed billing $2 | **31** |
| Full reservation before each allocation | One H100, 16 CPUs, 192 GiB, one lane, 600 s startup inside each stage, one preparation up to 3,600 s and preflight up to 900 s | **257** |

The published on-demand prices checked for this worksheet were $0.001097/H100-second, $0.0000131/physical-core-second, $0.00000222/GiB-second, $0.09/GiB-month and $0.04/GiB egress. The future operator must verify the authenticated prices, placement, actual account headroom and projected owned storage before reservation. The seven-day estimate assumes equal checkpoint size and excludes historical storage already held. [Modal pricing](https://modal.com/pricing)

The qualification closeout retains **59,514,869,670 bytes** (55.43 GiB), $4.00057863 posted and $45.61444321 in unresolved build/usage, collection and retention allowances. Outstanding call exposure was zero. Phase 4B separately retained 6,531,187,952 bytes, $5.05898874 posted and $5 held. Together with the projected fresh study volume the inventory would be about **191.68 GiB** before temporary volume writes and later growth. Reinventory all retained resources before each allocation and preserve the original A03 checkpoint. The historical retention review is **December 2, 2026 UTC**. The new seven-day review date follows each future allocation's actual retention start. [Qualification ledger](../evidence/phase4c-grpo-cumulative/campaign-closeout.json) · [Phase 4B accounting](../evidence/phase4b-live-20261125/cost-and-retention.json)

## One future owner decision

**Authorize a cumulative $300 before credits for this study, including one initial allocation and at most one controlled engineering replacement.** The mandate would cover all fixed seeds, automatic continuation after the first technical admission and exactly the four frozen final evaluations after durable release. Only one allocation may be active. Paid authority for this assignment is zero. The $300 ceiling leaves $43 of room for first-allocation cost and remaining liabilities if a full replacement becomes necessary.

1. After integration, pin the reviewed implementation in a clean checkout. Use fresh experiment, volume, registry and app identities, internal attempt 1, and the existing owned secret. Record the owner mandate and its cumulative ceiling privately. Set the owner profile's `study_authority` to `all-eleven-stages-including-final`. Never set an admission flag in a plan.
2. Enter actual gross, outstanding liabilities and the full $257 allocation reservation in one append-only study ledger before setup/build. Carry historical liabilities separately in the account headroom calculation. Use supported fresh preparation, verify the actual manifest, bind the new plan and require its same-plan CPU preflight. Preserve the acknowledged call IDs. Launch the supervisor once. No new paid pilot is needed.
3. Stop dependent work on failed admission, ownership, deadline, budget or storage. Preserve all partial evidence. Reconcile exact calls and outstanding exposure, including interrupted submissions, before any replacement. Cancel only recorded owned calls. Technical recovery requires a documented transient or a reproduced, locally tested and reviewed fix, a fresh allocation and a full reservation. A $257 replacement fits only if prior posted charges, unresolved liabilities and retained storage within the study leave at least $257 of the $300 cap. Recheck aggregate storage through three-checkpoint coexistence.
4. Use the unchanged scientific design on recovery, with all three seeds starting from base in the replacement. Keep failed runs distinct. Never pool seeds across materially different implementations or use observed development/final reward, finite zero gradients or constant groups as restart criteria. Keep final outcomes sealed against tuning. Stop at the first operationally complete study, the allocation limit or any exhausted bound. Report incomplete evidence as incomplete.
