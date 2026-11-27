# Current project continuation

Read [AGENTS.md](../AGENTS.md), then the [project plan](PROJECT_PLAN.md). This file is the mutable handoff for ChatGPT, Codex, and other agents. Update it with each completed implementation slice. Historical execution handoffs retain their original scopes.

**Repository verification, November 27, 2026 UTC.** Fetched main matched the supplied continuation revision, `c8e48c102bde9b01a6c428f4f90d2e758ff6e736`. PR #15 was merged and PR #14 was open and draft at the heads below. M1 was implemented in a separate worktree on that main.

## Resume here

**Next action:** Review the finished learner path in [M1 PR #16](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/pull/16). Follow its default entry into Chapters 1 and 2 and check that the learner can explain why changing only the final answer changes the reward.

M1 delivers the [root entry](../README.md), [learning index](learning/README.md), [agent-training foundations](learning/01-agent-training-and-frameworks.md) and [calculator trajectory](learning/02-follow-one-calculator-task.md). It also reconciles current recovery status in the [reader procedure](READER_TRAINING.md), [task adaptation](TASK_ADAPTATION.md) and [source-map footer](learning/source-map.md#current-adaptation-implementation). The twelve logical modules retain their destinations in the eight chapters and supporting guides. No files moved.

## Repository snapshot and delivery

| Item | Verified state |
|---|---|
| Canonical public repository | `amulyaparthasarathy/agent-lightning-grpo-ppo` |
| Planning baseline (preserved) | `e463dc146db3a4be7e156a06109a749db5f8649e` |
| PR #15 reviewed head | `bab5688141ec5df8bbbfd83006731b2122ea0f1d` |
| Plan integration commit, verified on main | `44fc49d93563d7048c73ffc18e87a18f6d1d89ac` |
| Merged tree, equal to the reviewed head's tree | `5faf46ed3c1bd290ffc06f0cc2227d41e234bee5` |
| PR #13 | Merged; pilot closeout and offline behavioral diagnosis are on main |
| PR #14 | Open, draft, unmerged; deferred by the current direction |
| PR #14 tested implementation | `eeafcc10099a3a8cc2aef575e959f4f391864677` |
| PR #14 head | `cd2829fc72c0ff7ade5c8d0dab4e9ec660295cd7`; specification-only change after the tested implementation |
| Plan delivery branch | `docs/agent-training-project-plan` |
| Plan delivery PR | [PR #15](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/pull/15); merged after final review of the exact head; plan v1.1 integrated |
| M1 base on main | `c8e48c102bde9b01a6c428f4f90d2e758ff6e736` |
| M1 delivery | `m1-learner-entry`, [PR #16](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/pull/16) against main |
| Validated M1 learner content | `9bc04f254b797cdd61034a764d3d978787779305`. The subsequent handoff commit records the PR and validation. |

## Decisions to preserve

1. Documentation quality comes first, for a first-year CS/basic-Python audience.
2. Foundations are a short bridge to **training models for agent use**, not a generic LLM training course.
3. The twelve logical modules reuse the existing eight chapters, exact mathematics, code, and completed experiments. M1 retains their existing paths.
4. The next capability extension is one bounded recovery-specific PPO-MC qualification, after the main documentation work.
5. Witness SFT PR #14, additional recovery GRPO seeds, long studies, and final-panel model evaluation are deferred.
6. A functional pipeline is not a promise of useful learning, fresh-holdout generalization, or arbitrary-seed reproducibility.
7. Provide a complete copyable reference path and optional challenges; reproduce figures from existing public evidence. A notebook, if added, shares the command-line calculations.
8. Explain the Tinker API/backend/infrastructure distinction now. Runnable hosted-Tinker GRPO remains optional future work after the core guide and disposition of bounded recovery PPO; it is not added to the required experiment backlog.

## Milestone ledger

| ID | Status at this handoff | Next acceptance evidence |
|---|---|---|
| M0: Shared plan and continuation | Complete; plan v1.1 merged through PR #15 | Reviewed head and merged tree verified |
| M1: Entry and representative lesson | Implemented in PR #16, awaiting review | Checked learner commands, reconciled status summaries and verified navigation as recorded below |
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

The M1 assignment covers documentation, relevant local CPU checks and delivery of a reviewable PR. Review is the stopping point. PR #14 remains deferred and independent of the recovery PPO-MC route. This slice required no change to budgets, retention, gates or experimental resources.

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

The opening now follows one question through the harness before introducing framework names. The trajectory lesson distinguishes the requested calculation, returned observation, generated answer, independent verifier and attributed terminal reward. It links to the existing chapters on training tensors, advantages, losses and optimizer steps.

| M1 check | Observed result |
|---|---|
| Extracted and ran the Chapter 2 shell blocks from the repository root with Python 3.12.3 | For the marked answer 36, the observation and answer both printed 36 and reward was 1.0. Changing only the answer to 35 kept the observation at 36 and gave reward 0.0. |
| Called `calculate` through the existing CPU MCP workbench | The tool advertised a required string `expression` and returned text `36` for `(7+5)*3`. The lesson supplies the two model actions. |
| Compared the teaching account with source | Inspected the pinned calculator harness, launcher, matched worker, upstream scalar comparator and completed-rollout builder for prompt construction, tool limits, reward extraction and attribution to the last call and its rollout. |
| Ran `python3 experiment/cpu_accounting_demo.py` and `python3 scripts/check_phase2_results.py` | Synthetic row means 5.0 and 3.5, equal-rollout means 5.0 and 5.0. The separate evidence checker passed 12 artifact hashes, 24 stage summaries, six runs and three contrasts. |
| Recomputed recovery totals from the [A03](../evidence/phase4c-grpo-cumulative/A03.json) and [P02](../evidence/phase4c-functional-pilot/P02.json) stage records and [paired cases](../evidence/phase4c-functional-pilot/diagnosis/development.json) | A03 had 144 admitted and confirmed episodes and two actor calls. P02 had 560 and 16. All 24 terminal greedy decisions were `not_ready`. Strict success rose from 9/24 to 12/24 while decision accuracy stayed at 12/24. |
| Checked Markdown and navigation | Validated 168 relative references across the eight files, including 22 with heading anchors, plus 12 incoming heading links. Checked chapter order, onward links, fences, disclosures and `git diff --check`. Checked the four learner pages in GitHub's rendered HTML for tables, disclosures and onward links. Both Mermaid blocks match their previously rendered sources byte for byte. |
| Reviewed diff and historical boundaries | Root historical operations, the learning index's dated authoring receipts and the original source mappings retain their baseline text. The changes cover eight Markdown files. |

No implementation blocker remains for M1. The example illustrates one possible exchange. The CPU tool invocation and scoring exercise supply no model behavior or learning measurement. No model execution, historical regression suites or new diagram rendering ran for M1. The [planning handoff at the M1 base](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/blob/c8e48c102bde9b01a6c428f4f90d2e758ff6e736/docs/PROJECT_CONTINUATION.md#latest-work-and-validation) preserves the separate PR #15 validation receipt.

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
