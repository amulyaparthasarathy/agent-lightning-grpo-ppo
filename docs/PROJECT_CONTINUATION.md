# Current project continuation

Read [AGENTS.md](../AGENTS.md), then the [project plan](PROJECT_PLAN.md). This file is the mutable handoff for ChatGPT, Codex, and other agents. Update it with each completed implementation slice. Historical execution handoffs retain their original scopes.

**Repository verification, November 27, 2026 UTC.** PR #16 was open at its independently reviewed head, with a clean merge state and no outstanding GitHub requirements. The ordinary merge matched that exact head. Fetched main contains the merge and has the same tree as the reviewed M1 revision.

## Resume here

**Next action.** Review [M2 PR #17](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/pull/17). Walk across the [Chapter 3 shift table](learning/03-trajectories-tokens-and-fragments.md#line-up-each-prediction-with-its-target) and run the first command with both histories. The second prompt's changed prefix requires another fragment while preserving four generated targets for the episode.

The M1 entry path and scored calculator lesson are on main. This first M2 slice adds the two prompt histories, next-token shift, response masks, fragment identity and equal-episode weights in Chapter 3. The remaining M2 lessons cover advantages, losses and the complete update. The twelve modules still use the eight existing chapters and supporting guides.

### Parallel hosted-Tinker track

Documentation remains the highest priority. The owner reports a $250 credit grant that may expire December 15, 2026. Its remaining balance, exact expiry and grant-specific eligibility are unverified. Public prices can inform a proposal but cannot confirm an account balance. The companion workstream prepares one runnable hosted-Tinker GRPO implementation, local validation and one exact execution proposal. Hosted execution awaits a separate decision. [Track scope](PROJECT_PLAN.md#parallel-hosted-tinker-companion)

**Companion branch and PR:** none identified in local worktrees, local or remote branch names, or open PRs at this handoff. The documentation workstream owns edits to the shared plan and continuation and records the companion identity here when available.

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
| M1 delivery | `m1-learner-entry`, [PR #16](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/pull/16) merged into main |
| Validated M1 learner content | `9bc04f254b797cdd61034a764d3d978787779305`. The subsequent handoff commit records the PR and validation. |
| M1 reviewed head | `c1555bfe315d2e1afa33557ad4f245c7685e1db7`. The independent review reproduced the 36/35 exercise and fragment accounting. |
| M1 merge and M2 base | `f2c20bb66b3c7b7dca16cb9dfe8426837a2b34d1`. Both M1 trees are `94edef83b1d02b193c01a1d6647d6e2b300f1a44`. |
| M2 delivery | `m2-calculator-token-alignment`, [PR #17](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/pull/17). Validated Chapter 3 and scheduling revision `9ba28878608eb5c7ed00c77e5cb3db564cfb8b19`. The handoff commit records the checks and PR identity. |

## Decisions to preserve

1. Documentation quality comes first, for a first-year CS/basic-Python audience.
2. Foundations are a short bridge to **training models for agent use**, not a generic LLM training course.
3. The twelve logical modules reuse the existing eight chapters, exact mathematics, code, and completed experiments. M1 retains their existing paths.
4. The bounded recovery-specific PPO-MC qualification remains planned after the main documentation work.
5. Witness SFT PR #14, additional recovery GRPO seeds, long studies, and final-panel model evaluation are deferred.
6. A functional pipeline is not a promise of useful learning, fresh-holdout generalization, or arbitrary-seed reproducibility.
7. Provide a complete copyable reference path and optional challenges; reproduce figures from existing public evidence. A notebook, if added, shares the command-line calculations.
8. Prepare the bounded hosted-Tinker companion offline in parallel. The guide and recovery PPO-MC remain independent of its execution decision.

## Milestone ledger

| ID | Status at this handoff | Next acceptance evidence |
|---|---|---|
| M0: Shared plan and continuation | Complete; plan v1.1 merged through PR #15 | Reviewed head and merged tree verified |
| M1: Entry and representative lesson | Complete and merged in PR #16 | Independently reviewed head integrated, with matching trees |
| M2: Deep training mechanics | First Chapter 3 slice implemented in PR #17, awaiting review | Shift, masks, merge/split and weighting checked on CPU. Remaining M2 mechanics are open |
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

The assigned integration was the reviewed M1 head. M2 ends at a reviewable PR. The parallel companion has offline preparation authority. PR #14 remains deferred and independent of recovery PPO-MC. The reported Tinker grant supplies no spending authority.

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

| First M2 slice check | Observed result |
|---|---|
| Ran both Chapter 3 shell blocks from the repository root with Python 3.12, then changed the second prompt as instructed | The selected pairs were `(11,20), (20,21)` and `(30,40), (40,41)`. The optional alteration changed the prefix result from true to false and preserved all four pairs. Checked the printed values against the shift table. |
| Ran `python3 experiment/cpu_accounting_demo.py` | The synthetic fixture gave row means 5.0 and 3.5. Both equal-rollout means were 5.0 and both weights for A were 0.5. |
| Fed the two histories to the hash-verified Agent Lightning adapter and VERL response slice in the existing pinned CPU environment | Four fixture episodes produced four rows with matching prefixes and eight with the changed prefix. Both had 16 selected targets. Verified the prompts, response masks, previous-position outputs, sampled log probabilities and total matched weight of 0.25 per episode. Used input IDs as position markers in place of model outputs. |
| Ran the existing `PinnedTrainingTests.test_real_adapter_constant_groups_and_fragment_counts` | One CPU test passed, covering merged and fragmented rows. The optional telemetry package was absent. All assertions were local. |
| Checked documentation and scope | Checked relative links and anchors, incoming chapter navigation, fenced blocks, the displayed formula and table values, and `git diff --check`. GitHub-rendered Chapter 3 HTML contains all three tables with the expected dimensions and its onward link. The existing Mermaid source matches the merged version. |

The examples use invented IDs and predefined losses. No model execution, experimental cloud operation or credit use occurred. Historical evidence and retention remain in their owning records. The [M1 handoff](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/blob/c1555bfe315d2e1afa33557ad4f245c7685e1db7/docs/PROJECT_CONTINUATION.md#latest-work-and-validation) preserves the earlier validation.

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
