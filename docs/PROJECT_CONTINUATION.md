# Current project continuation

Read [AGENTS.md](../AGENTS.md), then the [project plan](PROJECT_PLAN.md). This file is the mutable handoff for ChatGPT, Codex, and other agents. Update it with each completed implementation slice. Historical execution handoffs retain their original scopes.

**Verified main.** [PR #23](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/pull/23) integrated M3 by ordinary merge at `cdaac834255e951d8199c2c478588a329d4d4717`. Its parents are main `3aacc79610be9a6557af9f664b702f4367d859c8` and the independently reviewed head `1f903b71afd37d31da6a8d9b253ea50d7c9e3cf4`. The fetched merge and reviewed head have the same tree, `b1f9ad0c257ea1e5cb678a19c3089900b77ed84f`. Only the continuation differed between M3 learner source `49d3e7fc90160eafa777a89fe8949a52de253458` and that head.

## Resume here

**Next action.** Review [M4 in PR #24](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/pull/24), beginning with [Chapter 7](learning/07-read-the-experiment-results.md). Check the calculator contrasts, follow one of the three hidden decision regressions, and regenerate the four P02 figures from the public checkout. After M4 review and integration, the next documentation milestone is M7’s reader walkthrough and editorial closeout.

M4 branch `docs/m4-read-experiment-results`, validated learner commit `bb01858257f1b47a3bd72fd41a7f001bd22cac27`. The subsequent handoff commit records the PR and its checks. M4 stops at review. Modules 11–12 share the existing Chapter 7 path and use the [public plotting helper](../scripts/plot_p02_lesson.py). No implementation blocker remains for review.

### Hosted-Tinker companion paused

The owner selected one active documentation workstream until the core guide is finished. Preserve `hosted-tinker-grpo` in [PR #22](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/pull/22). Executable source `111ed8518ce28a26bf641376e09ab343b6aa17ce`, PR head `7ef13551d27b66a75310a3dc35ae5476406ee919`. The companion reports nine local tests with real tokenizer/SDK inputs and mocked service operations. It has no hosted execution. Its proposed $0.02 gross maximum is unauthorized. The reported grant balance, expiry and eligibility remain unverified. The optional Tinker SDK 0.40.1 is on main through PR #19. CMU’s SkyRL deployment on Modal uses Tinker-compatible APIs. [Companion scope and provider comparison](PROJECT_PLAN.md#cmu-and-tinker-references)

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
| M2 Chapter 6 | `m2-complete-training-update`, [PR #21](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/pull/21). Merged at `3aacc79610be9a6557af9f664b702f4367d859c8`. Validated learner content and source map at `d97e6cea6bb513a7da9ac9c4d3a6af29ee96ae13`. |
| M3 run and adapt | [PR #23](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/pull/23) merged at `cdaac834255e951d8199c2c478588a329d4d4717`. The verified parents and tree are above. |
| M4 experimental results | [PR #24](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/pull/24) open for review. Learner content and figures at `bb01858257f1b47a3bd72fd41a7f001bd22cac27`. |

## Decisions to preserve

1. Documentation quality comes first, for a first-year CS/basic-Python audience.
2. Foundations are a short bridge to **training models for agent use**, not a generic LLM training course.
3. The twelve logical modules reuse the existing eight chapters, exact mathematics, code, and completed experiments. M1 retains their existing paths.
4. The bounded recovery-specific PPO-MC qualification remains planned after the main documentation work.
5. Witness SFT PR #14, additional recovery GRPO seeds, long studies, and final-panel model evaluation are deferred.
6. A functional pipeline is not a promise of useful learning, fresh-holdout generalization, or arbitrary-seed reproducibility.
7. Provide a complete copyable reference path and optional challenges; reproduce figures from existing public evidence. A notebook, if added, shares the command-line calculations.
8. Finish the core learner documentation before returning to the locally tested hosted-Tinker companion. Keep the longer-term recovery PPO-MC goal.

## Milestone ledger

| ID | Status at this handoff | Next acceptance evidence |
|---|---|---|
| M0: Shared plan and continuation | Complete; plan v1.1 merged through PR #15 | Reviewed head and merged tree verified |
| M1: Entry and representative lesson | Complete and merged in PR #16 | Independently reviewed head integrated, with matching trees |
| M2: Deep training mechanics | Complete. Chapters 3–6 merged in PRs #17, #18, #20 and #21 | Reviewed Chapter 6 tree verified on main |
| M3: Execution and adaptation | Complete and merged in PR #23 | Reviewed M3 tree verified on main |
| M4: Experimental teaching | Delivered for review in PR #24 | Review Chapter 7’s calculator comparisons and public P02 figure path |
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

The owner authorized the ordinary M3 merge and delivery of one offline M4 PR for review. Both actions are complete. M4 integration and further milestone work await the owner after review. ChatGPT retains its read-only review role. PR #22 stays paused, with its proposed $0.02 hosted execution unauthorized. PR #14 remains open, draft and deferred.

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
| M3 integration | Main and PR head matched the reviewed identities. GitHub showed a clean, mergeable PR, no applicable branch rules or protection, and no checks, formal reviews or review threads. ChatGPT’s bounded review reproduced the synthetic means, calculator checker and asset fixture outcomes including cleanup. M3’s ten focused tests and rendering checks belong to the [original handoff](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/blob/1f903b71afd37d31da6a8d9b253ea50d7c9e3cf4/docs/PROJECT_CONTINUATION.md#latest-work-and-validation). Verified both parents and an exact tree match after merge. |
| Chapter 7 public path | Ran all seven shell blocks from a clean Git archive of the validated M4 learner commit with Python 3.12 and cached script dependencies (`UV_OFFLINE=1`). All four P02 PNGs regenerated byte-for-byte from public tables. Client and training dependencies remain separate. |
| Calculator | The existing checker passed 12 artifact hashes, 24 stage summaries, six runs and three contrasts. Checked critic values against the owning analysis, including all 57 earlier-data comparisons, 51 negative defined EV observations and nine undefined. |
| P02 development | Verified all three public table hashes against the diagnosis summary. Joined all 24 case identities to the receipt and checked baseline/reload decision correctness, evidence correctness and reward. The nine strict gains, six regressions, three retained successes and six retained failures reconcile. Three retained failures lost correct ready decisions. All 24 terminal greedy decisions were `not_ready`. |
| P02 training and updates | Recounted 512 CSV trajectories in 128 groups. Checked four attempts per unique question and matched all 32 class/collection cells against JSON. Ready and not-ready strict totals were 67 and 139, positive advantages 59 and 83. Checked all sixteen collection, sampled policy and actor-call identities, refreshes 0–16, scalar losses and preclip norms against the P02 receipt and rounded closeout. |
| Reading and navigation | Inspected the historical SVG and all four new figures at 760 px width. The calculator table supplies exact values where historical endpoint labels overlap. Checked changed Markdown relative links and heading targets, incoming Chapter 7 anchors and whitespace. GitHub-rendered Chapter 7 contains seven shell blocks, seven tables and five figures. Verified no diff to frozen evidence, client manifest or lockfile. |

The figures describe the calculator’s reused panel and P02’s one development-only seed. Only baseline and terminal greedy recovery measurements exist. There is no entropy or reference/base KL series. Training questions and class mix varied. These checks used public derived data without private replay, a new model run, a first-time reader trial or any cloud dispatch. Historical holds and retention decisions are unchanged.

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
