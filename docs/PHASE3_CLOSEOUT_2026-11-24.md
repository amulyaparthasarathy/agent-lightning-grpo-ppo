# Phase 3 closeout: review, preservation and accounting — 2026-11-24

The asset-task milestone is closed as **operationally qualified**. This record
covers the final review of the branch, what is preserved and where, the campaign
accounting and the cleanup that is proposed but not executed. No training,
inference, GPU validation or deletion ran during this closeout.

Qualification itself is described in the
[qualification report](PHASE3_QUALIFICATION_2026-11-24.md). Reusable lessons are in
the [engineering checklist](PHASE3_ENGINEERING_CHECKLIST.md).

## What the evidence does and does not show

| Claim | Status |
|---|---|
| Operational qualification: baseline → two updates → native checkpoint → fresh-worker reload, both algorithms | **Shown** by a07, all six stages at first submission |
| Better baseline behavior from the harness corrections | **Shown**: 0/64 strict-correct in a04, 64/64 in a07. The model-visible context changed, so this is a harness result, not a policy change |
| Informative reinforcement-learning signal | **Barely present**: 1 informative group in 16 for GRPO, one update with exactly zero gradient; PPO-MC updated under a poorly fitted critic |
| Learning improvement | **Not shown.** 64/64 before and 64/64 after leaves nothing to measure |

Recovery by a later successful call and the two-error terminal limit were never
sampled live. They are covered offline with scripted inference only. That is a
documented limitation, not a reason to rerun.

## Review against the Phase 3 objectives

| Objective | Where it is implemented and evidenced | Result |
|---|---|---|
| Reader-owned fork provenance | `experiment/reader_contract.py`; real-Git tests in `tests/test_phase3_source.py` (forks, clones, shallow history, replacement objects, grafts) | Met |
| Calculator backward compatibility | Calculator agent body pinned to the public baseline by test; calculator plan, stdio probe and CPU preflight rerun in this closeout | Met; the calculator was not retrained, by instruction |
| Complete alternate MCP task | `experiment/asset_task.py`, `asset_server.py`, `asset_agent.py`, `asset_probe.py` | Met |
| Task-specific preparation, execution, verification, evidence | `scripts/reader_training.py`; a06 and a07 preparation, preflight and stage records | Met |
| Collection-only sampled development validation | `"purpose": "collection-only"`; a06 | Met |
| GRPO and PPO-MC training, checkpoints, fresh-worker reload | a07 stage records and checkpoint manifests | Met |
| Reader adaptation documentation | [Task adaptation](TASK_ADAPTATION.md), [reader procedure](READER_TRAINING.md), the checklist | Met |

### Consistency checks run

A read-only script compared the committed reports with the committed evidence and
with the private originals. All 68 checks passed:

- The a06 and a07 source inventories equal the Git blobs at execution commit
  `7f1e13d7bacb477c9e5e95a16868a2b0f2484ee6`. Commits after it change only
  documentation and evidence.
- Each of the six a07 stage records carries the same source commit, stage plan
  `2a9b1cc811421c6eca122e344e1cc3a60cd210a3776603c7b8c9a4bd2d44bf21` and task
  hash. Supervisor, receipt and provider call IDs agree, and every call is
  terminal at the provider.
- Counts in the six-row table were recomputed from the stage records: 64/64
  complete, 64 confirmed rewards, 64/63/64 correct, optimizer events 0, 2, 0 for
  GRPO and 0, 2+2, 0 for PPO-MC, refreshes `[0]`, `[0, 1, 2]`, `[2]`.
- Each checkpoint manifest hashes to the identity in its training completion and
  in its evaluation completion. Evaluation ran in a different container.
- All six hosted runs are present, finished and configuration-matched, with every
  local numeric metric equal to its hosted value.
- Cost totals, holds and retained bytes add up.
- Every long identifier quoted in the Phase 3 reports resolves to committed
  Phase 3 evidence or a Git commit. Four identifiers that the reader guides carry
  over from earlier phases (the calculator plan hash, two private-history commits
  and the VERL pin) belong to those phases' own reports.

### Documentation corrections made here

- The README and learning chapter 8 still said alternate-task work "remains
  open". They now say the asset task is operationally qualified and the alternate
  model is still open.
- The learning index said the asset checks were local only.
- The qualification report called the a05 stop marker durable without
  qualification. The provider entry expires after seven idle days; see below.
- No relative link or anchor in the 58 tracked Markdown files is broken.

### Documented commands rerun

Run locally from commit `010dde92fb3528cc1384982a4509a3af7ee34719`, with fresh
output paths and no credentials: asset `generate`, `verify`, `probe` and
`scripted`; the calculator `probe` and CPU preflight; `reader_training.py --help`;
and `plan` plus `validate` for the asset profile, the calculator profile and a
collection-only profile. All exited 0. The three offline gates were rerun on the
final documentation tree; see [PHASE3_VERIFICATION](PHASE3_VERIFICATION.md). The
billable commands were not rerun; they are evidenced by a06 and a07.

## Preservation inventory

Machine-readable detail is in
[`closeout/preservation-inventory.json`](../evidence/phase3-qualification-20261124/closeout/preservation-inventory.json).

### Evidence

| Attempt | Local collected evidence (git-ignored) | Files / bytes | Owned Modal volume | Volume size |
|---|---|---:|---|---:|
| a01 | `.local/phase3-live-20261121/failed-a01-evidence` | 74 / 2,527,969 | `reader-asset-20261121-a01-volume` (`vo-KtIYfz0ZCf5RVNpWjB6LMB`) | 6.00 GiB |
| a02 | `.local/phase3-live-20261121/a02/stopped-evidence` | 233 / 3,416,284 | `reader-asset-20261121-a02-volume` (`vo-KtzGn1oPG8LfXOztBi04Jr`) | 6.00 GiB |
| a03 | `.local/phase3-live-20261121/a03/final-evidence` | 382 / 5,612,556 | `reader-asset-20261121-a03-volume` (`vo-WFtwoXPWkiS6yOwCoFfSz8`) | 6.01 GiB |
| a04 | `.local/phase3-live-20261121/a04/final-evidence` | 602 / 9,345,130 | `reader-asset-20261121-a04-volume` (`vo-1DponfTcYKI9q3oxLt3NIn`) | 6.02 GiB |
| a05 | `.local/phase3-live-20261121/a05` (plan, receipt, logs, stop result) | 8 files | `reader-asset-20261122-a05-volume` (`vo-WqMMuCiJaLx8qWg8NiyqGj`) | empty |
| a06 | `.local/phase3-live-20261121/a06/final-evidence` | 254 / 4,846,314 | `reader-asset-20261122-a06-volume` (`vo-IiG0se8E2kgzVyvte5pHAR`) | 6.01 GiB |
| a07 | `.local/phase3-live-20261121/a07/final-evidence` | 1,098 / 74,288,336 | `reader-asset-20261122-a07-volume` (`vo-aNf9E4heLEnwhGuv6pFxDD`) | 116.75 GiB |

Each attempt also has a registry named like its volume with the suffix
`-registry`. Their identifiers are in
[retained resources](../evidence/phase3-qualification-20261124/retained-resources.json).

Verification done in this closeout:

- Every local file was re-hashed against the committed
  `collected-file-manifest.json` of its attempt. All six sets match exactly, with
  no missing and no extra file.
- Every collected file still exists on its volume with the same size. The only
  local-only file is the collector's own `collection.json`.
- What the volumes hold beyond the collected files was listed read-only:
  compiler and dataset caches under `temporary/`, the prepared Llama snapshot
  (6.43 GB per attempt, re-acquirable from the pinned revision), the synthetic
  data (regenerable, hashes committed), the a07 checkpoint tensors, and a few
  small generated files the collector skips by suffix. Those small files, 82 in
  all and 51 KB, were read into
  `.local/phase3-live-20261121/volume-supplement-20261124` with a hash manifest.

Backup status. The collected evidence exists twice: the volume original and the
verified local copy. The private campaign directory (about 6,000 files, 258 MB: ledger,
plans, receipts, logs, reconciliations, helpers and the collected evidence) exists
once, on the workbench disk. Reviewed aggregates are in Git. Hosted W&B keeps the
metrics for all present runs.

### Checkpoints

| Algorithm | Files and bytes | Identity SHA-256 | Location |
|---|---|---|---|
| GRPO | actor, 9 files, 40,146,400,251 | `2e6cf21b255a0bc46f4629deed908c887c1349987115dae69cd35b065a45bdd7` | a07 volume, `phase3-runs/reader-asset-20261122-r07/a07/grpo-matched-v2/training/checkpoints/global_step_2` |
| PPO-MC | actor 9 files, 40,146,400,251; critic 8 files, 38,570,427,714 | `f8ad498fa245d84e66fd95cfb06fbdf57190a4aa60ff98de5904d3047f42ec3d` | a07 volume, `phase3-runs/reader-asset-20261122-r07/a07/ppo-mc-matched-v2/training/checkpoints/global_step_2` |

Per-file sizes and SHA-256 values are in the two a07 training stage records. A
read-only listing on 2026-11-24 found all 26 inventoried files present with their
manifest sizes. Each checkpoint was hashed by its training worker and verified
again by the fresh evaluation worker.

**Neither checkpoint has an independent backup.** Both exist only on the a07
volume. No tensor was downloaded in this closeout.

### Backup recommendation

These are two-update operational checkpoints with no demonstrated learning value.
Their identities, inventories and reload proof are already preserved. A backup
protects the ability to re-inspect or reload the exact tensors, nothing more.

Options checked:

| Option | Capacity and cost | Assessment |
|---|---|---|
| Keep only the a07 volume | US$10.51 per month | Current state; single copy |
| Pull to the workbench disk and verify against the manifests | 118.86 GB needed, 843 GB free; one-time egress of about US$4.43 if the listed US$0.04/GiB applies | **Recommended** if a second copy is wanted. Uses existing storage and existing per-file hashes |
| Existing private Hugging Face backup repository from the November 14 preservation workflow | That workflow is complete; its remaining byte allowance (about 65 GB) is below the 119 GB needed | Only if off-machine durability is required. Needs a new approval and adapting a workflow bound to other resources |

The workbench disk is one virtual disk on one machine. A local copy is an
independent copy, not an off-site one. Transfer speed was not measured.

### The a05 record

The reconciliation and stop record are preserved in three places: the local
`a05/stop-owned-01.json`, the committed
[`a05/reconciliation.json`](../evidence/phase3-qualification-20261124/a05/reconciliation.json)
and the provider registry entry `stop:reader:522a9dff…`, confirmed present on
2026-11-24. The provider entry is a Modal Dict entry, and those expire after
seven days without a read or write. It will lapse on its own unless it is read
again. That would not let a05 run: its app never registered a function, nothing
was spawned, and no client will resubmit that plan.

## Proposed cleanup — not executed

Nothing below was run. Each step needs explicit approval.

1. Optional second copy of the private evidence (258 MB), outside the repository:

   ```bash
   tar -C .local -czf /workspace/projects/phase3-live-20261121-evidence.tar.gz phase3-live-20261121
   sha256sum /workspace/projects/phase3-live-20261121-evidence.tar.gz
   ```

2. If a checkpoint backup is approved, pull and verify before any deletion:

   ```bash
   MODAL_PROFILE=amulyasp modal volume get --env main reader-asset-20261122-a07-volume \
     phase3-runs/reader-asset-20261122-r07/a07/grpo-matched-v2/training/checkpoints \
     /workspace/projects/phase3-a07-checkpoints/grpo-matched-v2
   MODAL_PROFILE=amulyasp modal volume get --env main reader-asset-20261122-a07-volume \
     phase3-runs/reader-asset-20261122-r07/a07/ppo-mc-matched-v2/training/checkpoints \
     /workspace/projects/phase3-a07-checkpoints/ppo-mc-matched-v2
   ```

   Then hash every file and compare with `native_checkpoint_manifests` in the two
   training stage records.

3. Retire the volumes that hold nothing unique. This frees 30.04 GiB, about
   US$2.70 per month:

   ```bash
   for n in 20261121-a01 20261121-a02 20261121-a03 20261121-a04 20261122-a05 20261122-a06; do
     MODAL_PROFILE=amulyasp modal volume delete --env main reader-asset-$n-volume
   done
   ```

4. Retire the matching registries, except a05 while its stop marker is wanted and
   a07 while its volume is kept:

   ```bash
   for n in 20261121-a01 20261121-a02 20261121-a03 20261121-a04 20261122-a06; do
     MODAL_PROFILE=amulyasp modal dict delete --env main reader-asset-$n-registry
   done
   ```

5. Keep `reader-asset-20261122-a07-volume` and its registry. Delete them only
   after a verified checkpoint backup, or after a decision that the tensors are
   not needed.

Not proposed: any change to the stopped apps (they hold no resources), cached
images, the shared credentials secret, hosted W&B runs, or any resource from the
calculator reference qualification and earlier phases.

Retention review dates are unchanged: a01–a04 by 2026-11-29 01:54:46 UTC,
a05–a07 by 2026-12-01 03:34:30 UTC. Nothing is scheduled to delete itself.

## Campaign accounting

Reconciled read-only at 2026-11-24 05:23:07 UTC. All 36 known calls are terminal.
No app in the workspace has a running task. Nothing changed since the 04:02 UTC
reconciliation. Detail is in
[`closeout/campaign-closeout.json`](../evidence/phase3-qualification-20261124/closeout/campaign-closeout.json).

| Category | Amount | Notes |
|---|---:|---|
| Authorized aggregate ceiling | US$300.00 | One ceiling for all of Phase 3 |
| Observed charges | US$12.31 | Completed-hour usage: a01–a04 5.76, a05 0.00, a06 1.39, a07 5.16. Every row is attributed to an owned object |
| Unresolved billing | not known | The usage report is not an invoice and applies no credits. Volume storage has no per-object rows in it; accrued storage so far is estimated at US$0.19. Delayed rows remain possible |
| Continuing storage | US$13.21 per month | 146.79 GiB at US$0.09 per GiB-month, about US$3.08 per week. The a07 volume alone is US$10.51 per month |
| Conservative holds | US$74.00 | Includes the observed US$12.31; US$61.69 is margin against unresolved billing and retention |
| Unused execution authorization | US$226.00 | Not reserved and not spent |

The unused balance is not a budget for further experiments. The private ledger
now records execution as closed. The holds stay in place until billing settles
and the retention decisions are made. This closeout spent nothing beyond
read-only provider queries and a 51 KB evidence read.

## Merge readiness

No implementation, evidence or documentation blocker remains. Open items:

- **Owner approval.** PR #4 stays draft and unmerged until it is given.
- **PR title and description.** The text on GitHub still describes the failed
  a04 state. The command-line client here is not authenticated, so the
  replacement text is supplied with the closeout for the owner to paste.
- **Retention decisions** above. They do not block the merge.

Standing limitations travel with the merge: no learning claim, two episode
branches without live observation, the six reviewed dependency exceptions, no
cold-build claim, and no qualification of resumed training, other models or more
than two updates.
