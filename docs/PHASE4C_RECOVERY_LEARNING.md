# Phase 4C recovery-learning qualification

**GRPO operational qualification completed on A03.** Independent review admitted 144 episodes and two actor updates, including a fresh-worker reload of the step-2 checkpoint. [Campaign closeout](PHASE4C_GRPO_CAMPAIGN_CLOSEOUT.md)

Recovery development calibration is closed. On each route the fresh base succeeded on 8 of 24 greedy development episodes and 38 of 96 sampled episodes. All 120 rewards per route were confirmed. GRPO had 13 semantic/evidence mixed questions and PPO-MC had 15, spanning five families each. All eight stages recorded zero advantages, gradients, optimizer calls and checkpoint saves. This supports use of development data for this task. Recovery training capacity, improvement and algorithm superiority remain open. Final-panel model evaluations to date total zero. The historical execution source is `bc81c1fb0def27b15c56cdd8abb01098e7e1f8d5`. The evidence/documentation head is `79b534f5fad7b11ca32a89a239a74bb53e4d7b00`. [Phase 4B results](PHASE4B_LIVE_CALIBRATION.md)

The first live Phase 4C GRPO allocation failed at baseline finalization. Its 24 diagnostic rewards remain available, with zero qualification-admitted episodes or updates. The offline correction and proposed fresh allocation are documented separately. [A01 evidence](PHASE4C_GRPO_QUALIFICATION_A01.md) · [Correction and successor worksheet](PHASE4C_PATH_CORRECTION.md)

## Four stages per selected route

The synthetic reader profile selects GRPO. The versioned phase also accepts an explicit PPO-MC selection or an ordered, sequential pair. It requires attempt 1. The plan derives its stage list, expected rewards, files, batches, policy steps and maximum worker/supervisor exposure. A profile cannot supply an update count or substitute a task. All four workers use seed 9001 and the existing algorithm-specific derived streams. [Example profile](../examples/reader-profile.recovery-learning.synthetic.json) · [Dispatch](../experiment/reader_modes.py) · [Declaration](../experiment/recovery_learning.py)

| Stage | Eager training input | Eager validation input | Policy and sampling | Episodes | Updates |
|---|---|---|---|---:|---:|
| Base baseline | dev_panel, 24 | dev_panel, 24 | Fresh base, one greedy | 24 | 0 |
| Collection probe | Ordered probe, 8 | Same ordered probe, 8 | Fresh base, four at temperature 1 | 32 | 0 |
| Training | train_main, 128 | dev_panel, 24 | Fresh base, eight × four at temperature 1 per update | 64 | 2 |
| Checkpoint evaluation | dev_panel, 24 | dev_panel, 24 | Fresh worker, step 2, one greedy | 24 | 0 |

The probe materializes the first eight frozen training IDs in order to a run-local Parquet file. The first update reuses that predetermined question batch with a fresh initialization and collection. Batch two is disjoint. Selection used the seeded upstream sampler before observing any probe rewards. Probe output includes the admitted collection and excludes advantages, backward, optimizer and checkpoint operations. A full single route has 144 episodes. A pair has 288. Only the question and fixed interaction protocol enter the user prompt. Case metadata, labels and verifier diagnostics stay with grading. Model-visible tool observations come from the production harness.

Both file arguments, scoped asset inspection and the materialized row order are checked against immutable commitments. Training disables validation callbacks. All-split consistency checks still run during deterministic dataset preparation. Final data has no qualification stage. The full IDs include the prefix `recovery-readiness-v1/train/c-`. [Ordered IDs and hashes](../experiment/recovery_schedules.json)

| Batch | ID suffixes in loader order |
|---|---|
| Probe and update 1 | 8a57194009, 3bfc2b6ea4, c605de2476, c257ee0d75, c52d879870, 92941931c9, 2c4d925efa, 1784f158e3 |
| Update 2 | 27ade11485, dda94ea016, e561e8fc75, faf21938ea, ec873962cc, e1245dedba, 59e0cab660, 49edc1a869 |

## What counts as qualification

Each collection needs eight groups of four distinct, completed rollout IDs. The reviewer independently replays tool observations, labels, terminal state, reward post and conversation against the frozen case. At the adapter boundary the controller attempt, episode, served prompt, action IDs and per-action log probabilities must agree. A confirmed zero for a wrong or malformed final answer, a typed policy termination or a recovered episode is a legitimate observation. Constant groups count. Infrastructure failures, corrupted observations, missing rewards, truncation, omitted rows and incomplete groups terminate admission. No resampling or substitution is permitted. [Admission code](../experiment/recovery_admission.py)

The same matched-rollout-mean-v2 optimizers consume admitted batches. GRPO standardizes reward within each group. PPO-MC retains its learned critic, true-terminal Monte Carlo returns, gamma and lambda of one, token whitening and clipping. Both give equal weight to real rollouts across fragments. Old probabilities are recomputed and compared with serving probabilities on target tokens. Observations, reminders and padding have zero policy-target weight.

Operational completion requires two actor optimizer calls, two additional critic calls for PPO-MC, finite per-update losses and gradient evidence, and policy refreshes at 0, 1 and 2. Training saves only the terminal step-2 native checkpoint, with model, optimizer and extra state for each required role. The evaluation worker hashes the native inventory before loading and after evaluation. Its provider call and container must differ from training. Evaluation stages run zero updates.

The collected report joins persisted supervisor state, stage request, source/configuration, durable receipt, worker identity, raw trace, audit and W&B finish. It reconstructs the stage result and retains available slot diagnostics from incomplete work. It reports operational completion, informative groups and numerical signals, and strict performance separately. Finite zero loss or gradient is admissible. A performance decrease or absence of informative groups provides no grounds for a retry. [Evidence reviewer](../experiment/recovery_learning_evidence.py)

## Single-GRPO launch review

The offline planner and validator agreed on plan SHA-256 `c2e903b66e107824cb001ad9526cf932f5867ab1a8da41d5ebd510793e395713` from clean corrected runtime `852e01d141b66232c73ca46e6e7bf4b9dfb0e854`. The synthetic GRPO profile binds the retained preparation manifest. The validation receipt records `training_admissible: false` and unverified remote ownership and availability. The full bound plan and receipt are in ignored Workbench storage.

| Attempt | Baseline | Probe | Training | Reload | Total |
|---|---|---|---|---|---:|
| 1, GRPO, seed 9001 | 24 greedy development | 32 sampled train, zero updates | 64 sampled train, two updates from fresh base | 24 greedy development, step 2 | 144 episodes |

The plan permits one lane, four stages, 3,600 seconds per stage, 14,400 worker seconds, 15,000 supervisor seconds and zero automatic retries. The proposed ceiling is $113 including the $82 compute reservation. Evaluation uses development only. Fresh owner authorization is required before any paid qualification. Revalidate from the exact execution checkout when preparing a future authorized attempt. [Verification commands](PHASE4C_VERIFICATION.md#corrected-supervisor-verification)

## Future operation and capacity

One operator owns the mutable source worktree and ledger. Use a clean committed source, create the bound plan in ignored storage, and keep that checkout fixed through launch acknowledgement. Reserve exposure before dispatch. Keep the submission client alive until acknowledgement. On a failed, rejected or unknown outcome, keep the reservation, logs and receipts. Reconcile the exact owned call and its exposure before a separate retry decision. Stop only an exactly owned call. The proposal allows one initial attempt, zero automatic retries, concurrency one. Never clear a reservation to create retry capacity. [Phase 3 engineering checklist](PHASE3_ENGINEERING_CHECKLIST.md)

If dispatch is interrupted before acknowledgement, the persisted stage receipt records an unknown outcome with no call ID. After acknowledgement, interruption cleanup targets the exact recorded call and marks cancellation requested. A returned cancellation request may close the supervisor stage as `cancelled`. Failed cancellation leaves it `unknown`. Both retain the full 3,600-second worker reservation ($10) pending independent terminal evidence and exposure reconciliation. The supervisor's finish timestamp records local cleanup time.

Inventory logical bytes under the owned retained volume paths and project the bytes that checkpoint writes will add there. Use those figures for the 300-GiB retained-volume bound and seven-day storage allowance. Measure free bytes separately on the worker-local scratch filesystem against the peak temporary writes that must coexist during save or reload. [Cost worksheets](PHASE4C_LEARNING_STUDY_PROPOSAL.md#cost-worksheets)

The candidate is one exact H100, 16 CPUs and 192 GiB host memory. Qualification would have to measure GPU peaks during log-probability recomputation and backward, memory outside the per-process allocator, host/cgroup peaks and offload, update and refresh latency, checkpoint size, write latency and scratch headroom, and fresh-worker load/refresh. Sustained 16-update behavior requires separate evidence. Collection-only time and CPU fixture size supply no backward throughput estimate. Changes to capacity, duration or budget require a revised frozen proposal before execution.

Historical Phase 4B usage was $5.05898874 before credits, with final invoicing outstanding. The registry and 6.083-GiB volume remain retained, with $3 storage and $2 delayed-billing holds and a December 2, 2026 UTC retention review. Its unused allowance is unavailable for a new study. This milestone performed no resource mutation or retention review. [Historical cost and retention](../evidence/phase4b-live-20261125/cost-and-retention.json) · [Separate cost worksheets](PHASE4C_LEARNING_STUDY_PROPOSAL.md#cost-worksheets) · [Offline verification](PHASE4C_VERIFICATION.md)

The separate three-seed executor, offline verification, cost worksheet and proposed owner mandate are documented in the [study execution report](PHASE4C_STUDY_EXECUTION.md).
