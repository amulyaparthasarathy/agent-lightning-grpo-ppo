# Current project continuation

Read [AGENTS.md](../AGENTS.md), then the [project plan](PROJECT_PLAN.md). This file is the mutable handoff for ChatGPT, Codex, and other agents. Update it with each completed implementation slice. Historical execution handoffs retain their original scopes.

**Last repository verification:** November 27, 2026, at approximately 00:09 UTC (November 26 in America/Los_Angeles). This snapshot precedes PR #15's authorized merge; inspect its live merge state before continuing. No cloud/account state was queried for this update.

## Resume here

**Next work item: M1, the entry path and one finished trajectory lesson.**

Implement the brief contrast between conventional response/chat training and agent training; explain harness ownership; polish the existing synthetic calculator trajectory; establish one default reading path. Reuse the existing content and experiments. Deliver finished learner-facing prose rather than another planning report.

Primary files:

- `README.md`
- `docs/learning/README.md`
- `docs/learning/01-agent-training-and-frameworks.md`
- `docs/learning/02-follow-one-calculator-task.md`

Also correct the clearly current recovery-status summaries in `docs/READER_TRAINING.md`, `docs/TASK_ADAPTATION.md`, and `docs/learning/source-map.md`. GRPO A03/P02 are completed; recovery PPO-MC live training/reload remains unqualified; the full recovery study remains unexecuted. Preserve dated historical receipts and the original source map.

M1 acceptance: a basic-Python reader can explain what is trained, how the harness controls interaction, how a completed episode earns reward, and which lesson follows. Use the practical lesson contract: purpose, prediction, command, expected output, visual/table, explanation, optional alteration, and evidence boundary. Link an existing credential-free example with its checked expected output. Keep the foundations brief and agent-specific. M1 does not build the full figure catalog or a new backend. Do not change the numerical runtime or rerun model experiments.

## Repository snapshot and delivery

| Item | Verified state |
|---|---|
| Canonical public repository | `amulyaparthasarathy/agent-lightning-grpo-ppo` |
| Main inspected for this plan | `e463dc146db3a4be7e156a06109a749db5f8649e` |
| Main tree | `84ddaeb53045192aba5fd0338bce94d4f0b9011b` |
| PR #13 | Merged; pilot closeout and offline behavioral diagnosis are on main |
| PR #14 | Open, draft, unmerged; deferred by the current direction |
| PR #14 tested implementation | `eeafcc10099a3a8cc2aef575e959f4f391864677` |
| PR #14 head | `cd2829fc72c0ff7ade5c8d0dab4e9ec660295cd7`; specification-only change after the tested implementation |
| Plan delivery branch | `docs/agent-training-project-plan` |
| Plan delivery PR | [PR #15](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/pull/15); owner authorized the updated plan's merge after final review; live GitHub records integration |

Inspect PR #15's current merge state before starting implementation. After it is merged, use fetched main and proceed directly to M1; do not attempt a second merge or restart planning. If it is still open, the owner's instruction authorizes ordinary integration only after final review of the exact head. Do not bypass required checks or reviews. The source SHA above is the inspected pre-merge baseline, not the eventual documentation merge or an experiment execution identity.

## Decisions to preserve

1. Documentation quality comes first, for a first-year CS/basic-Python audience.
2. Foundations are a short bridge to **training models for agent use**, not a generic LLM training course.
3. The twelve logical modules reuse the existing eight chapters, exact mathematics, code, and completed experiments. No physical chapter migration has been performed yet.
4. The next capability extension is one bounded recovery-specific PPO-MC qualification, after the main documentation work.
5. Witness SFT PR #14, additional recovery GRPO seeds, long studies, and final-panel model evaluation are deferred.
6. A functional pipeline is not a promise of useful learning, fresh-holdout generalization, or arbitrary-seed reproducibility.
7. Provide a complete copyable reference path and optional challenges; reproduce figures from existing public evidence. A notebook, if added, shares the command-line calculations.
8. Explain the Tinker API/backend/infrastructure distinction now. Runnable hosted-Tinker GRPO remains optional future work after the core guide and disposition of bounded recovery PPO; it is not added to the required experiment backlog.

## Milestone ledger

| ID | Status at this handoff | Next acceptance evidence |
|---|---|---|
| M0: Shared plan and continuation | Plan v1.1 delivered in PR #15; merge authorized after exact-head final review | Verify live integration, then start M1 |
| M1: Entry and representative lesson | Next; not started | Finished opening/trajectory lesson and reconciled mutable status summaries |
| M2: Deep training mechanics | Planned; reuse existing Chapters 1 and 3–6 | Checked examples/derivations and source-linked learner exercises |
| M3: Execution and adaptation | Planned | Supported procedure explained step by step with expected evidence and cost boundary |
| M4: Experimental teaching | Planned | Reproducible evidence figures, expected counts, and interpretation exercises for calculator/recovery |
| M5: Recovery PPO-MC preparation | Planned; no new reservation proposed | Exact four-stage plan, relevant local gates, capacity/cost/retention proposal |
| M6: Recovery PPO-MC execution | Conditional; not authorized | Actual qualification or documented failure and complete closeout |
| M7: Reader walkthrough and closeout | Planned; can finish without funded M6 | Checked learner path and an honest disposition of remaining capability work |

## PR #14 disposition

Keep [PR #14](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/pull/14) open and draft, with its branch and evidence preserved. It is not a dependency of M1–M4 or the existing recovery PPO-MC route. Its fourteen changed files are isolated additions. The authorization key is unset, and dispatch rejects before Modal access. This focused source review identified no concrete code blocker, but did not rerun its recorded test suites.

The missing full-size GPU result limits execution and behavioral claims; it is not automatically a ban on merging gated code. Integration is deferred because it is unnecessary for the selected work. Merging PR #15 does not merge or activate PR #14. Do not spend time rerunning the SFT suites during M1 or merge it merely to clear the PR queue.

## Evidence that is already complete

Use the [plan's evidence reuse table](PROJECT_PLAN.md#4-existing-experiments-to-teach-from) and original reports. Do not treat these as pending reruns:

- Calculator r03: six runs, three seeds per algorithm, 20 actor updates each, PPO's 20 critic updates, independent final reloads.
- Public reader calculator and asset-status: both GRPO and PPO-MC completed their short operational workflows.
- Recovery calibration: 120 admitted episodes per route, zero optimization.
- Recovery GRPO A03: 144 admitted episodes, two actor updates, fresh-worker step-2 reload.
- Recovery GRPO P02: 24 baseline + 512 training + 24 reload, all 560 admitted/reward-confirmed, sixteen actor updates, fresh-worker step-16 reload.
- P02 diagnosis: all 24 terminal greedy decisions were `not_ready`; strict 9/24 to 12/24 did not establish improved decision accuracy. P01 is separate partial evidence.

Key public execution identities: reader calculator `5daff64cefd348fb731094cf465b223ddceb5fb2`; asset-status `7f1e13d7bacb477c9e5e95a16868a2b0f2484ee6`; A03 `7fdb2b1e0ce2a05f03980cfd3a1e07a31937ca6a`; P02 `17889f37651d57653cfc0dbe12d06304ac27371a`. Historical calculator source/publication mapping remains in the [source map](learning/source-map.md#public-source-representations).

## Authority, cost, and retention

The owner explicitly requested creation, the CMU-inspired plan update, and merge of PR #15 after final review, followed by Codex's implementation of the aligned plan. This is a bounded documentation-write/integration exception to ChatGPT's previous read-only repository role. No cloud execution or new experimental run was authorized. Future agents must follow their actual assigned implementation scope and any still-applicable explicit authorization.

**New paid dispatch authority: zero.** The recovery PPO preference selects a direction, not a numeric budget. M5 must produce the concrete proposal before the execution decision. Do not ask for that decision during the documentation work.

The closed pilot's recorded accounting is historical, not a new billing query:

| Pilot closeout snapshot | USD before credits |
|---|---:|
| Posted gross | 9.50754026 |
| Conservative outstanding holds | 52.43245974 |
| Unencumbered portion of the closed $150 ceiling | 88.06 |
| Further dispatch authority | 0 |

The unencumbered balance does not fund a new run. PR #14's proposed $25 and the older GRPO qualification worksheets are also not PPO authority or a current PPO quote. See [pilot accounting and retention](PHASE4C_FUNCTIONAL_PILOT_CLOSEOUT.md#exposure-and-retention).

- Historical retention review: **December 2, 2026 UTC**.
- Both pilot volumes/checkpoints: **December 3, 2026 UTC**.
- A future PPO allocation must record its own retention review horizon.
- A review date is not permission to delete. Preserve the existing holds and private ledgers until their separately authorized review.

The retained private pilot continuation is `.local/phase4c-functional-pilot/CONTINUE.md` in the operator's existing environment. Its exact resource inventory is not part of this public handoff and was not inspected during plan authoring. The current documentation work does not need it.

## Latest work and validation

This documentation change adds the shared plan, current continuation, and root agent instructions, plus navigation links. Plan v1.1 incorporates the agreed practical lesson contract, three reader paths, figure catalog, CMU numerical distinctions, and optional Tinker comparison. M1 remains the next implementation task. The change does not implement M1–M7 or alter runtime, experiment data, rewards, frozen evidence, or cloud resources.

Authoring verification included live main/PR reads, inspection of the existing guide and evidence reports, and separate reviews of the curriculum reuse map and recovery-PPO scope. Both focused reviews found no blockers. A local Markdown/file-scope check verified 132 relative link targets across the five changed files and eight anchors whose target content was loaded, with zero errors. The file delta contains exactly the three new documents and two README navigation insertions. Both modified READMEs matched their original bytes plus the intended insertion. The PR description records delivery and the resulting commit.

The CMU comparison recomputed the P02 class/group counts from committed public JSON and ran the existing synthetic CPU accounting example, confirming row means 5 to 3.5 and equal-rollout means 5 to 5. These are offline teaching checks, not new experiments. No training, inference, full-size model load/download, model-provider or cloud-compute call, or historical regression suite was run. Old CPU test counts remain in their original evidence; they are not new validation results for this documentation change. The PR description records the updated exact-head final review and delivery checks.

## Handoff update template

Replace the current work/status sections with concise current facts instead of appending a new roadmap. Keep evidence in its owning reports.

```text
Verified at: <date/time and timezone>
Inspected main: <SHA>
Working branch / PR / tested implementation: <identities>
Milestone and deliverable: <ID and concrete scope>
Completed: <accepted content or behavior, with links>
Validation performed: <commands/checks and actual results>
Material limits / not run: <only relevant omissions>
Module/file migration: <if any>
Next action: <one bounded task and its files>
Blockers or decisions: <specific unresolved item, or none>
Authority: <what remains authorized; exact dispatch scope if separately granted>
Retention/ledger changes: <if any, with safe references>
```
