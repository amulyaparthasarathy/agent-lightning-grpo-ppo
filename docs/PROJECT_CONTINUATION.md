# Current project continuation

Read [AGENTS.md](../AGENTS.md), then the [project plan](PROJECT_PLAN.md). This file is the mutable handoff for ChatGPT, Codex, and other agents. Update it with each completed implementation slice. Historical execution handoffs retain their original scopes.

**Verified main.** [PR #24](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/pull/24) integrated M4 by ordinary merge at `4d24d99cab74088fa9bf05b14d18bc7bfd758ae2`. Its parents are the M3 merge `cdaac834255e951d8199c2c478588a329d4d4717` and the independently reviewed M4 head `f4022e5e55257ba1bd0388b74a07de3434401b0d`. The fetched merge and reviewed head have identical tree `4e19d5e28dfb882855f30756335686a404329973`. Only the continuation differed between M4 learner source `bb01858257f1b47a3bd72fd41a7f001bd22cac27` and that head. Evidence and dependency locks carry through the merge.

## Resume here

**Next action.** Review [M7 in PR #25](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/pull/25). Follow the [published chapter order](learning/README.md#read-the-chapters-in-order), check the change from episode scores to token rewards in Chapter 5, and try the [calculator resource exercise](learning/07-read-the-experiment-results.md#what-did-those-six-calculator-runs-use). M7 stops at review.

Branch `docs/m7-reader-walkthrough`. Validated learner commit `17123c154f9de323396fb1853bfe21140e23749b`. The final head includes the continuation-only handoff. Its full SHA is recorded in the [PR description](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/pull/25) and is available from GitHub as `refs/pull/25/head`.

### Core guide coverage at review

| Reader question | Where the walkthrough answers it |
|---|---|
| Who acts and who supplies the score? | Chapters 1–2 follow the synthetic calculator request, tool observation, final answer and terminal verifier reward. The reader can change the answer while holding the observation fixed. |
| What reaches the optimizer? | Chapters 3–5 trace the first answer token through its exact context, generated-target mask and fragments. Four attempts give GRPO its comparison group. Within one PPO-MC attempt the terminal return is shared, old critic predictions vary, and whitening uses the collection’s generated tokens. |
| When does a new policy answer? | Chapter 6 follows the cached advantages, equal-rollout loss, accumulated backward passes, critic then actor step, serving refresh and terminal save. A receipt exercise identifies the reloaded actor and zero evaluation updates. |
| What did the agent learn and what did the run use? | Chapter 7 pairs baselines and reloads, splits recovery outcomes by actual class, and works out study worker time and role-file bytes. The dated app charges have no measured per-seed split. |
| What can a reader reproduce or adapt? | Chapter 8 links to the public arithmetic, asset harness and fixture plan. The first online setup is marked in the workflow and the [reader procedure](READER_TRAINING.md) owns the operational commands. Its locked client is separate from the CPU harness and remote GPU stack. |

This covers the [completion standard](PROJECT_PLAN.md#1-goal-and-completion-standard) within the documentation scope. The agent walkthrough found no further blocking documentation gap. A first-time human trial has not been conducted in this slice. Recovery PPO-MC remains an explicitly deferred capability.

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
| M3 run and adapt | [PR #23](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/pull/23) merged at `cdaac834255e951d8199c2c478588a329d4d4717`. Both parents and the merged tree were verified at integration. |
| M4 experimental results | [PR #24](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/pull/24) merged at `4d24d99cab74088fa9bf05b14d18bc7bfd758ae2`. Learner content and figures at `bb01858257f1b47a3bd72fd41a7f001bd22cac27`. |

## Decisions to preserve

1. Documentation quality comes first, for a first-year CS/basic-Python audience.
2. Foundations are a short bridge to **training models for agent use**, not a generic LLM training course.
3. The twelve logical modules reuse the existing eight chapters, exact mathematics, code, and completed experiments. M1 retains their existing paths.
4. The bounded recovery-specific PPO-MC proposal (M5) and its execution (M6) are deferred. Each needs a separate owner decision.
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
| M4: Experimental teaching | Complete and merged in PR #24 | Reviewed M4 tree verified on main |
| M5: Recovery PPO-MC preparation | Deferred pending separate preparation authority | Exact four-stage plan, relevant local gates, capacity/cost/retention proposal |
| M6: Recovery PPO-MC execution | Deferred pending an exact M5 proposal and separate execution authority | Actual qualification or documented failure and complete closeout |
| M7: Reader walkthrough and closeout | Delivered for review in PR #25 | Review the core guide walkthrough, calculator resource exercise and deferred capability disposition |

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

The owner authorized the ordinary M4 merge and one offline M7 PR. Both actions are complete. M7 merge and any further work await the owner. ChatGPT retains its read-only review role. PR #22 remains open on its preserved branch with the proposed $0.02 execution unauthorized. PR #14 remains open, draft and deferred. Additional recovery seeds, the full recovery study and final-panel evaluation remain deferred.

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

| Current-session check | Result |
|---|---|
| Final M4 integration | Exact base and head, clean working tree, clean merge status, no applicable protection or rules, checks, formal reviews or review threads. After merge, verified both parents and compared the full tree. The [M4 handoff](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/blob/f4022e5e55257ba1bd0388b74a07de3434401b0d/docs/PROJECT_CONTINUATION.md#latest-work-and-validation) owns the prior clean-archive, figure and rendering checks. ChatGPT’s bounded independent M4 review reproduced all seven then-current Chapter 7 shell blocks, figure blob hashes, paired cases, groups and update metrics, and inspected the four figures. |
| Beginner walkthrough | Read the introduction, index and Chapters 1–8 in order. Added definitions at first use, clarified what the two four-reward lists index, matched notation to the mask and return examples, and connected the four-attempt loss shares to the 32-attempt update. Chapter 8 now ends by asking what evidence answers a behavioral question. |
| Technical walkthrough | Resolved 31 distinct line anchors across 16 pinned files used by the chapters. Traced prefix matching and mask gaps, next-token alignment, matched weights, GRPO and PPO-MC calculations, old value/advantage storage, role ordering, backward accumulation, serving refresh, terminal save and validation-only reload. Checked source and numerical claims separately from the readability pass. |
| Public chapter exercises | Ran 15 shell blocks in chapter order with Python 3.12, plus the Chapter 7 training-only alteration. Checked the synthetic 36/1 endpoint, target pairs, 5 → 3.5 row mean versus 5 → 5 rollout mean, centered GRPO loss and slope, PPO-MC return/whitening/loss outputs, the 32-attempt/33-row example and independent step-20 receipt. The existing calculator checker passed 12 hashes, 24 summaries, six runs and three contrasts. Unchanged P02 plot commands use the bounded M4 validations above. |
| Resource and charge interpretation | The new CSV exercise gave 19,120.608 total worker-seconds, 14,064.047 for training stages and 356,589,684,584 checkpoint role-file bytes (332.100 GiB). Cross-checked against the 18 study receipt rows. Checked the two historical app charges sum to USD 40.10359572 and the supervisor wall time against the published resource analysis. |
| Operational and adaptation branches | With cached packages and uv offline, ran the reader procedure’s client sync/help and fixture plan/validate in the existing environment. Calculator and asset templates each returned matching plan hashes, unverified ownership/availability and false training admission. Ran the adaptation guide’s CPU sync, data generate/verify (256/16/64/8), both probes and one scripted chain. The wrong-answer and missing-answer exercise produced `[0.0]` and `[]`. The corrected alteration produced `[1.0]` and `[]`. |
| Navigation and rendering | Checked relative files and heading links on the reading path and incoming links to edited headings. GitHub-rendered Chapters 1–8 matched all 32 fenced blocks, 100 math expressions and 25 table structures. Rendered and visually inspected the changed Chapter 8 workflow. Ran `git diff --check` and confirmed a Markdown-only diff. |

This validation used public derived evidence and local scripted responses. The measurements still belong to their historical execution sources. The agent walkthrough adds no private replay, new model result, cold-build qualification, provider or billing check, or evidence of useful learning on fresh cases. The historical holds, retention inventory and execution authority are unchanged.

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
