# Phase 4B live development calibration

On November 25, 2026 (UTC), both base-policy collection routes passed the predeclared recovery-readiness screen in one complete attempt. Each route admitted 120 development episodes. Both earned reward on 38 of 96 sampled rollouts, but that average hides the result that matters for a learning study. On the same question, the base policy sometimes succeeded and sometimes made a substantive decision or evidence error. Thirteen GRPO-route questions and fifteen PPO-MC-route questions showed that pattern. Calibration ends here. A learning study would require a separate design and budget.

## What ran

The upstream Phase 4A implementation was tested offline at `e5806b15d0d4da49045d2d495457a905562034ad`. The reviewed PR head and verified merged main shared tree `116472bb10de7ddb1175e91bf0aec72d6f38b448`. Codex took sole ownership of a separate Phase 4B worktree, the private campaign ledger, and all submissions on branch `phase4b-live-calibration` from main `92b276516e93c305cd0a29c61983cb0f798042a9`. Both inherited checkouts were clean. Phase 3, including its earlier allocation and execution source `7f1e13d7bacb477c9e5e95a16868a2b0f2484ee6`, remained under its existing ownership.

The tested and executed Phase 4B commit was `bc81c1fb0def27b15c56cdd8abb01098e7e1f8d5`. Two report defects reproduced on merged main. A list of job IDs raised an attribute error. A collected audit file with a different W&B finish ID still admitted the packet. The corrections validate the jobs mapping and check the raw finish ID against the planned run during consumer recomputation. Four affected report tests passed, including a complete simulated collected directory, and 13 contract tests passed, including existing calculator and asset launch behavior. Subsequent documentation commits describe this execution.

The actual configuration used the pinned Llama 3.2 3B Instruct revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`, Agent Lightning `cbc56e41e0eab643aa71d195e92e5bba652fa683`, VERL `7aed6b230776f963fa09509c10d9c3a767d1102c` (wheel 0.8.0), the pinned llama3_json parser and fixed-date served template. Both jobs started from the base model with seed 9001 and used recovery-episode-v1, recovery-calibration-v1 and recovery-context-v1. The development panel was `rr1-dev-ac7fd952343b0c7d`. Each route ran one 24-episode greedy stage at temperature 0 and three temperature-1 packets, each covering eight questions with four rollouts per question. The GPU lane was sequential.

The fresh preparation plan was non-executable. Its CPU worker acquired the locked model and generated the recovery dataset. The executable plan was regenerated and validated against the collected manifests. CPU preflight in the actual training image rechecked the inputs, package locks and reviewed dependency exceptions, W&B project and gated-model access, tokenizer and template, and all 13 MCP probe checks. The largest prompt in the preflight scenarios was 2,739 tokens, leaving 333 tokens below the trajectory prompt limit. The model saw only development questions. Preparation generated and preflight verified the frozen final files as assets. The final panel `rr1-final-a76397cabf99d31a` had zero model evaluations.

| Clean-source plan | SHA-256 |
|---|---|
| Preparation, no assets | `f74683842896769c3430352e875b868547fc19bb48725572f2a6c1e0a13530e3` |
| Executable, verified assets | `9d38cee73e4e682ff55ac7014f0fe5bca0f456ae80b9657a0015abdcfe280646` |

The package inventory contained 65 runtime files. The public evidence includes hashes for the complete source identity, package, configuration, task, environment, model manifest, dataset manifest and bound asset bundle. The private plans and receipts carry the exact attempt, stage, provider and W&B identities.

## Admission and scientific result

| Measure | grpo-matched-v2 | ppo-mc-matched-v2 |
|---|---:|---:|
| Episodes expected / attempted / completed / admitted | 120 / 120 / 120 / 120 | 120 / 120 / 120 / 120 |
| Confirmed / missing rewards | 120 / 0 | 120 / 0 |
| Rejected / unattempted / unknown | 0 / 0 / 0 | 0 / 0 / 0 |
| Greedy ones / zeros | 8 / 16 | 8 / 16 |
| Greedy zero classes (semantic / evidence / format / action) | 12 / 4 / 0 / 0 | 12 / 4 / 0 / 0 |
| Sampled ones / zeros | 38 / 58 | 38 / 58 |
| Sampled zero classes (semantic / evidence / format / action) | 39 / 10 / 9 / 0 | 37 / 8 / 12 / 1 |
| Constant zero / constant one / mixed groups | 6 / 4 / 14 | 5 / 2 / 17 |
| Mixed groups with semantic or evidence errors | 13 | 15 |
| Informative families | 5 | 5 |
| Interface share of sampled zeros | 9/58 (15.5%) | 13/58 (22.4%) |
| recovery-screen-v1 | screen_passed | screen_passed |

Each route independently exceeded six informative groups across two families and stayed below the 50% interface limit. The five informative families were newer failed run with older usable backup (3 GRPO, 2 PPO-MC), incomplete recent backup (2, 2), failed integrity check (3, 3), recent usable backup (2, 4), and stale success (3, 4). The sampled packet vector for every question, its failure classes, greedy reward, stage audit and token accounting are in the reviewed evidence.

The statistical unit for comparing case performance is the question. The 96 sampled answers on each route cover 24 selected questions. Four draws share each question, and both routes reuse the same panel. These data support a decision about this development panel. They do not establish generalization, improvement, or an algorithm ranking.

## Serving and trajectory evidence

On both routes every greedy episode invoked all three tools successfully. Nevertheless, 16 of 24 received zero reward. On sampled episodes, the full required evidence was observed in 92/96 on GRPO and 93/96 on PPO-MC. Ten GRPO answers and eight PPO-MC answers made the correct decision with invalid evidence. Retrieval alone left the model with two tasks, applying the readiness rule and selecting evidence that proved the answer.

- On `c-5db1a70832` (stale success), two GRPO samples made the same sequence of five successful calls. Both obtained the required evidence. One answered `ready` and earned 0. The other answered `not_ready` with valid support and earned 1. The trace digests begin `183458606bdb` and `97b734df2fba`.
- On `c-c25ac45067` (newer failure with an older usable backup), a GRPO sample correctly answered `ready`. Its asset, policy and inventory references checked out. Its supporting run IDs failed verification, so it earned 0. Trace `c422f80c92a7`.
- On `c-77fa9e56ce`, a PPO-MC sample called `list_backup Runs`. The harness retained the space and uppercase R. The next recorded assistant history contained the exact sampled ID, name and arguments, and the gateway returned a generation with 1,519 prompt and 62 completion tokens. The model subsequently called `list_backup_runs` and earned 1. Trace `bfbd03958b11`.
- On `c-fb62402846`, a PPO-MC sample used an invalid as-of argument, made successful asset and policy calls, then repeated the timestamp error. That second typed error reached the two-error terminal limit and posted reward 0. Trace `b0c42fd2d5e6`.

All 14 unknown-tool calls (10 GRPO, 4 PPO-MC) appeared verbatim in the next model input. This includes the single sampled name that the pinned client would have rewritten. For generations without a tool call, all 239 returned nonempty content. The remaining episode ended at the typed-error limit. The campaign therefore supplies no observation of how this stack represents an empty text generation without a tool call.

GRPO recorded 12 typed errors and two episodes with a later successful action. PPO-MC recorded seven typed errors and two episodes with a later successful action, including the eventual terminal failure above. A recovered action and a rewarded episode answer different questions.

| Served or adapter measurement | GRPO route | PPO-MC route |
|---|---:|---:|
| Served calls across four stages | 694 | 700 |
| Prompt token counts reproduced using the locked tokenizer/template | 694 | 700 |
| Largest observed served prompt / trajectory limit | 1,900 / 3,072 | 1,900 / 3,072 |
| Sampled generated tokens / target-mask tokens | 28,841 / 28,841 | 28,249 / 28,249 |
| Sampled fragment rows / masked context tokens | 550 / 0 | 556 / 0 |
| Truncated / dropped prompt rows | 0 / 0 | 0 / 0 |
| Advantage / gradient / optimizer / checkpoint events | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| Weight refreshes | Initial step 0 once per stage | Initial step 0 once per stage |

Each of the eight raw W&B finish IDs matched its planned stage ID, alongside the worker completion, receipt, supervisor entry, trace membership and reward joins. No stage ran SFT, a backward pass, a warm start, or a checkpoint reload. For each sampled rollout, the original generated-token count equaled the sum of the fragment target masks. The stage records also give fragment prompt sums, actual prompt ranges, and generated lengths. Longer, unobserved actions could still exceed the declared context limits.

Resampling all four rollouts together by question gives case-mix sensitivity ranges of 26.0–54.2% (GRPO) and 28.1–51.0% (PPO-MC) around the 39.6% sampled reward on each route. These percentile ranges describe resampling the 24 selected development cases. They do not quantify uncertainty on a new holdout or a trained policy. The evidence file includes the resampling method and per-question vectors. [Reviewed execution evidence](../evidence/phase4b-live-20261125/calibration.json)

## Resources and continuation

| Stage | GRPO worker seconds | PPO-MC worker seconds |
|---|---:|---:|
| Greedy development | 322.9 | 302.5 |
| Sampled packet 1 | 325.3 | 406.5 |
| Sampled packet 2 | 413.9 | 429.4 |
| Sampled packet 3 | 314.4 | 320.0 |

The supervisor recorded 2,956.2 seconds across its eight dispatch-to-finish intervals and 2,991.2 seconds overall. Every stage passed the GPU kernel probe on an NVIDIA H100 80GB HBM3 with CUDA 13.0, vLLM 0.22.0 and FlashAttention 2.8.3. Each worker exited with code 0, saved its resolved configuration and completed owned-process cleanup. The collected launcher arguments bind both dataset paths to the materialized development files. All trace case IDs and split identities matched that stage's declaration. Temperatures were verified in the gateway launch configuration. Hosted W&B history was not independently queried.

The private ledger reserved $95 before setup against the fresh $200 authorization. The first-attempt ceilings were $70 for GPU workers and supervisor, $15 for builds, $5 for CPU preparation and preflight, $3 for storage, and $2 for collection and reconciliation. One exact H100, 16 physical CPU cores and 192 GiB cost $6.2428/hour at the verified workspace rates. No placement or nonpreemptible multiplier applied. The estimate valued credits, cache hits and included allowances at zero. The preparation client stayed connected through 59 newly saved image layers until provider acknowledgement. Preparation and preflight both returned terminal results under their own recorded call IDs. [Modal pricing](https://modal.com/pricing) · [Billing report semantics](https://modal.com/docs/guide/billing)

The final reader download returned 39,157,714 bytes with zero collection errors. Two earlier collections returned 323,156 and 743,119 bytes. The strict standalone report on the final directory agreed with the collector. Private artifacts include the original traces, audit streams, input manifests, submission and worker receipts, supervisor index, resolved configuration, process records and operational logs. The public JSON contains sorted file-hash commitments for each stage's traces, audit files, configuration, completion, receipt and observability record.

All 11 provider calls returned completed (prepare, preflight, supervisor, eight stages). All three campaign apps stopped with zero tasks and zero remaining containers. The retained registry and volume belong to this attempt. The volume inventory measured 6,531,187,952 logical bytes (6.083 GiB) across 1,949 entries with zero checkpoint files. The prepared model, development data, logs and traces remain available. The existing credential handle and W&B project were used under their existing ownership. There are eight new planned-and-finished stage run identities. No historical allocation was modified.

Modal's two closed UTC billing intervals showed **$5.05898874** of attributable usage before credits. The preparation app accounted for $0.00536315, preflight $0.00496688, and the launch app including supervisor $5.04865871. The report attributed $3.16072490 to H100, $0.65114346 to CPU, $1.24632091 to memory and $0.00079947 to network egress. The reconciliation included all 59 observed built-image IDs. The export returned no separate build-object charge or per-volume storage row. Delayed attribution and invoice settlement remain open. [Cost and retention record](../evidence/phase4b-live-20261125/cost-and-retention.json)

At the workspace storage rate of $0.09/GiB-month and the measured inventory, seven days of retention cost an estimated $0.13, or $0.55 per month while the inventory stays constant. Estimated total through the retention review is **$5.19** before credits, subject to delayed charges. Once all calls were terminal the operator closed the execution reservation and retained $3 for storage and $2 for collection or late billing. Of $200 authorized, $194.94 remains after recorded usage and **$189.94 is unencumbered** after both holds. The optional second attempt and $10 contingency were unused.

**Operator handoff.** Keep the private campaign ledger, raw evidence, registry and volume. Review retention on **December 2, 2026 UTC**. Recheck the provider billing export against the owned app and image IDs, check volume size, and reconcile the two remaining holds. Retention continues pending the owner's decision. The private handoff includes the read-only reconciliation command and exact evidence directory for repeating the report. Branch `phase4b-live-calibration` and draft PR #6 contain the reviewable source and redacted results. The documentation head postdates execution.

**Decision.** Stop calibration after attempt one. Both routes passed with complete evidence and the same frozen base model. A separate learning study could now test whether optimization improves readiness decisions and evidence selection. That proposal should freeze development choices, seeds, training budget, checkpoint-selection rule and held-out evaluation before any update. Training and final-panel evaluation require new approval.
