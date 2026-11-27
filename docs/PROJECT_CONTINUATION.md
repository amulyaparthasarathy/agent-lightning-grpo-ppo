# Current project continuation

Read [AGENTS.md](../AGENTS.md), then the [project plan](PROJECT_PLAN.md). This file is the mutable handoff for ChatGPT, Codex, and other agents. Update it with each completed implementation slice. Historical execution handoffs retain their original scopes.

**Verified baseline, November 27, 2026 UTC.** [PR #20](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/pull/20) merged at `6228222015f2595e220f7248c8825e28448f720d`. The parents are the preceding main `c74d0f971bfed408d5e7bfc2509a73f0fe93344b` and the reviewed head `a94597e845e2c7a45358dc89d940fb1bd59a87d4`. On the fetched merge, Chapter 5 and its source map match the validated content commit. The SDK dependency and lockfile match the preceding main.

## Resume here

**Next action.** Review [Chapter 6 in PR #21](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/pull/21). Follow D's cached advantage past the critic update into the actor update. Run the [CPU accounting exercise](learning/06-one-complete-training-update.md#count-the-work-on-a-cpu) with and without D's split. Use the [receipt exercise](learning/06-one-complete-training-update.md#check-that-the-saved-actor-served-the-independent-evaluation) to explain how the independent evaluation can report global step 20 with zero new optimizer calls.

Chapters 3–5 are on main. Chapter 6 and its source-map references are ready for review. The worked example covers collection, both loss reductions, gradient accumulation, the serving refresh, the terminal save and the evaluation-only reload. No files moved.

### Parallel hosted-Tinker track

Documentation remains the highest priority. The owner reports a $250 credit grant that may expire December 15, 2026. Its remaining balance, exact expiry and grant-specific eligibility are unverified. Public prices can inform a proposal but cannot confirm an account balance. The companion workstream prepares one runnable hosted-Tinker GRPO implementation, local validation and one exact execution proposal. Hosted execution awaits a separate decision. [Track scope](PROJECT_PLAN.md#parallel-hosted-tinker-companion)

**Companion implementation pending.** [PR #19](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/pull/19) added and locked Tinker SDK 0.40.1 in the optional `tinker` dependency group. Select it with uv's `--group tinker` option. The dependency branch was `chore/tinker-sdk`, implementation `c3197ca34b24ae7452113d2cec6e69252ef77ca3`. The owner has not started the separate implementation prompt and will share the branch and PR when available. At this handoff the only Tinker branch on the canonical remote is the merged dependency branch. That session owns the runnable implementation, local validation and exact execution proposal. This documentation workstream owns the shared plan and continuation and will integrate the companion into the guide after implementation and validation. CMU's separate SkyRL deployment uses Tinker-compatible APIs on Modal. [Provider comparison](PROJECT_PLAN.md#cmu-and-tinker-references)

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
| M2 Chapter 3 | [PR #17](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/pull/17) merged. Validated content `9ba28878608eb5c7ed00c77e5cb3db564cfb8b19`. Independently reviewed head `b0f0d6579c7f8f3bcff9546c59a8ba4edffd0f7a`. |
| PR #17 merge and Chapter 4 base | `acc4a50c4691efb57c726d61a44e4c6167abfe82`. Both trees are `6063636db998ae6ed3ef3fb1fe043575030f1e84`. |
| M2 Chapter 4 | [PR #18](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/pull/18) merged after final review. The lesson blob at the merge is `5bc7ca70b10a3083024ab452598fe337353da9a6` and its handoff blob is `3cd5ca623613cbef5eeb4c6f3ccea9f136cfce94`. |
| SDK files at the Chapter 4 merge | `pyproject.toml` blob `ca4728cbbea44d1c4e29f532c7c914452702535b` and `uv.lock` blob `49cbb72cfb9918d7073bd45ba3cdfbc2c9cc8985`. All previously locked package/version pairs were present after #19. |
| M2 Chapter 5 | `m2-ppo-mc-calculator-lesson`, [PR #20](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/pull/20). Merged reviewed head `a94597e845e2c7a45358dc89d940fb1bd59a87d4`. Validated content `217c201eaa2b7cb4493c81bd86b30501cf69346e`. |
| M2 Chapter 6 | `m2-complete-training-update`, [PR #21](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/pull/21). Validated learner content and source map at `d97e6cea6bb513a7da9ac9c4d3a6af29ee96ae13`. |

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
| M2: Deep training mechanics | Chapters 3–5 merged. Chapter 6 ready for review in PR #21 | Check the iteration order and both CPU exercises against the cited implementation and stage receipts |
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

The owner authorized the ordinary merge of the reviewed Chapter 5 head and Chapter 6 delivery for review. Chapter 6 merge awaits a separate decision. The owner coordinates the pending offline companion. PR #14 remains open, draft and deferred.

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

| Check performed | Observed result |
|---|---|
| Final review and integration of #20 | ChatGPT's bounded independent review reproduced the Chapter 5 commands and optional alteration. At final review we reran those examples, checked the unchanged head, clean GitHub state, requirements and threads, and verified the ordinary merge against both parents. The [Chapter 5 handoff](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/blob/a94597e845e2c7a45358dc89d940fb1bd59a87d4/docs/PROJECT_CONTINUATION.md#latest-work-and-validation) records the earlier pinned backend and rendering checks. |
| Ran both Chapter 6 Python shell blocks and the `split_d = False` alteration | 32 synthetic episodes, 88 generated targets and weight 1/32 for D in both representations. Splitting D gave 33 backward passes per trained role, joining it gave 32. Both have one optimizer call per trained role. The recorded PPO-MC seed-2001 stage rows matched all printed values. |
| Exercised the hash-verified adapter and matched trainer step on the lesson's 32-episode CPU fixture | Both algorithms admitted the 33- and 32-row representations. Instrumented worker stand-ins recorded the prediction, weighting, role-update and refresh order. The PPO-MC actor received the original weighted advantages and old values after the critic callback changed its simulated prediction. |
| Exercised the pinned engine and outer trainer loop | Scalar autograd losses in the one-row microbatches produced 33 or 32 backwards before one hooked AdamW step, with accumulated preclip norm 1 in both cases. The outer loop with stubbed stage operations scheduled the final save after step 20. In validation-only mode it loaded step 20, refreshed and evaluated, then returned. This exercised control flow without reading model weights. |
| Ran three focused `PinnedTrainingTests` | `test_incomplete_truncated_duplicate_and_missing_reward_fail_closed`, `test_runtime_update_dispatch_uses_actual_row_count` and `test_pinned_trainer_step_retains_nonmultiple_and_constant_real_rollouts` passed on CPU. |
| Checked recorded results | For each of the six r03 study jobs, compared the committed stage rows, episode and update counts, terminal step, distinct training/reload worker calls and matching recorded checkpoint digest. Cross-checked refreshes and the single save against the verification ledger. The separate recovery P02 closeout records step 16. |
| Checked publication and navigation | Verified the links and heading targets in the changed files, incoming heading links, Chapter 5–7 navigation, fences and `git diff --check`. The committed GitHub HTML matched the eight math expressions, event table and five fenced blocks. Rendered the Mermaid diagram locally with mmdr 0.3.1 and inspected the branches, labels and arrows. |

The synthetic counts and instrumented CPU callbacks describe teaching fixtures. The study measurements are from published evidence. No model execution, hosted model call, experiment rerun, cloud dispatch, credit use or retention change occurred.

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
