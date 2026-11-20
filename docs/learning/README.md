# Learn how a calculator experience becomes a weight update

The question for this guide is: **How does one calculator-agent experience become a model-weight update, and what changes between GRPO and this lab's PPO-MC?**

You need Python and basic machine learning. The public educational snapshot needs no account to read; the private historical repository is not a prerequisite. The exercises use Python's standard library, with no model, cloud service or dataset download.

## Reading paths and chapter contracts

Read Chapters 1–5 then 7 for the mechanisms and findings. Read all eight for the implementation path. Start at Chapter 8 if you only want to check committed evidence. Each chapter moves from a question and intuition to an example, mathematics where needed, exact source, evidence limits and an offline exercise with an answer.

| Chapter | Opening question | Contract |
|---|---|---|
| [01](01-agent-training-and-frameworks.md) | What is being trained, and why use a real harness? | Define workflows, compound systems, agents, SFT and offline/online RL. Map the harness, gateway/controller/adapter, vLLM, VERL and matched overlay. |
| [02](02-follow-one-calculator-task.md) | How does one question become a recorded reward? | Follow a synthetic question through per-call prompts, tool action, observation, continuation, extraction, scoring and reward attribution. |
| [03](03-trajectories-tokens-and-fragments.md) | Which tokens are actions, and why can one rollout occupy several rows? | Preserve exact sampled token IDs and prompts; distinguish group, rollout and fragment identities; derive real-rollout loss weights. |
| [04](04-grpo-step-by-step.md) | How can four attempts teach an actor without a critic? | Derive sample-standard-deviation GRPO advantages, constant-group behavior, the shared clipped surrogate and zero-loss/nonzero-gradient example. |
| [05](05-ppo-mc-step-by-step.md) | What changes when a learned value supplies the baseline? | Derive general GAE and terminal MC; explain direct matched residuals, full-token whitening, separate critic initialization, value loss and diagnostic utility. |
| [06](06-one-complete-training-update.md) | When do collected experiences actually change model weights? | Trace 32 real rollouts through fragments, old log probabilities, losses, accumulation, optimizer calls, weight refresh, save and independent reload. |
| [07](07-read-the-experiment-results.md) | What do the six runs establish? | Preserve denominators, separate baselines and paired contrasts; interpret weak critic, clipping, formatting and tool errors with explicit limits. |
| [08](08-reproduce-and-operate.md) | What can a reader check offline? | Run existing arithmetic and derived-evidence checks; separate raw verification, paid cold builds, licensing and publication gates. |

Use the [glossary](glossary.md) for shared notation and the [source map](source-map.md) to distinguish execution, evidence, editorial sources and historical protocols.

## Scope and identity

The guide explains `agl-explore-20-v1-r03`, numerical protocol `matched-rollout-mean-v2`, study protocol `exploratory-matched-20-v1`. The freshly inspected main on 2026-11-18 is `e2b57a590b9ea687fbb4c9001465eddd5b65ef70`. The actual executed source is `5fb6adfd48ee245509f7dff3ab1d849fcf707d86`. Later documentation and merge commits did not train these models.

The prepared model is `meta-llama/Llama-3.2-3B-Instruct`. The upstream starter's Qwen model and the paper's benchmark experiments are separate. Chapters 2–6 share a **synthetic** `(7+5)*3` task, expected numeric result `36`, and four illustrative rewards `[0,0,0,1]`. These examples are not recorded r03 trajectories.

The three seed pairs improved their frozen-scorer endpoints on a reused 64-question development panel. They establish neither superiority, equivalence, statistical significance nor fresh-holdout generalization. Read the [closeout](../PHASE2_CLOSEOUT_2026-11-15.md) and [qualified evidence](../../evidence/phase2-r03/VERIFICATION.md).

This guide is retained in an independently attributed publication snapshot. Historical runbooks and frozen evidence stay at their original paths. The [source map](source-map.md#public-source-representations) explains public source links and private archival locators; [publication status](../PUBLIC_RELEASE_PLAN.md) records the separate distribution identity.


## Documentation verification, 2026-11-18

**Historical authoring receipts from the private original follow; publication verification is recorded in the release plan.**

Checkpoint A is commit `03f87a81202298ddc2e08d54d3d6c1cf5896c257`; completed lesson content is commit `31c64c9` (the subsequent documentation commit records this verification). Checks ran from a fresh local Git clone of the content commit with Python **3.12.3**, without copying the private `.local` packet or installing packages.

| Check actually run | Observed result and boundary |
|---|---|
| `python3 experiment/cpu_accounting_demo.py` | Exit 0. Synthetic row mean 5 to 3.5; equal-rollout token mean 5 in both representations. |
| `python3 scripts/check_phase2_results.py` | Exit 0. PASS: 12 artifact hashes, 24 stage summaries, six runs, three contrasts; both reported algorithm means reproduced. Derived-evidence consistency only. |
| Chapter 4 inline Python | Exit 0. Mean 0.25, sample std 0.5; epsilon-adjusted advantages matched; synthetic zero loss and derivative -0.5. |
| Chapter 5 inline Python | Exit 0. Raw residuals `[-0.1,-0.2,-0.6,0.6]`; three-action GAE example `[0.9,0.8,0.6]`. |
| Temporary standard-library authoring validator | PASS: 71 local links/heading anchors, 83 pinned source line anchors, 63 source symbols/constants in 22 pinned files, table columns, headings, fenced blocks, formula examples, answer keys and all six result rows. |
| Source evidence preparation | Exact upstream files were read from GitHub before the offline validation. Nine available source-lock hashes matched; lab source bytes matched pinned Git objects. No package/model installation or runtime import. |
| Diagram/figure checks | Seven Mermaid diagrams checked for structure and declared node/participant references; existing paired SVG parsed as XML. Browser rendering was not run because no Mermaid renderer/Node runtime was installed. |
| `git diff --check` and scope inspection | PASS. Only the eleven learning Markdown files and narrow root README navigation change; no runtime, mathematics implementation, scorer, source lock, historical report or evidence changes. |

Optional CPU backend integration tests were skipped; no prepared compatible environment was established for this task, and no dependencies were downloaded to create one. GPU training, inference, paid services, raw-trace/tensor verification, cold-build reproduction and publication checks were not run. The packet's manifest verification and the larger report-package discrepancy are recorded in the [source map](source-map.md).


## Final teaching and presentation review

This pass starts from reviewed PR head `442fe5a9a8b8bbe7d3e60a0c70aa95e6cfd3d924`. Local and remote heads matched and the working tree was clean before editing; main remained `e2b57a590b9ea687fbb4c9001465eddd5b65ef70`. Execution identity remains `5fb6adfd48ee245509f7dff3ab1d849fcf707d86`. The earlier receipt above describes the earlier content revision, including its then-unavailable browser rendering.

The pass clarifies sample standard deviation and distinct rollout rewards, adds three synthetic actor-clipping cases, and completes a four-rollout, six-valid-token PPO-MC arithmetic fixture. The latter separates token-population whitening from equal-total-weight-per-rollout loss reduction and includes one weighted clipped-value-loss calculation. No new empirical measurements are introduced.

| Check run again on the revised documentation | Observed result |
|---|---|
| `python3 experiment/cpu_accounting_demo.py` | Exit 0; row mean 5 to 3.5, equal-rollout mean stays 5. |
| `python3 scripts/check_phase2_results.py` | Exit 0; all 12 hashes, 24 stage summaries, six runs and three contrasts pass; reported means unchanged. |
| All four inline Python blocks in Chapters 4–5, extracted and executed with `python3` | Exit 0; both original answer keys still match. New actor losses: -1.3, 0.8, 3. New valid-token mean: -0.066666667; sample variance: 0.174666667; all whitening/weight rows match; clipped value contribution: 0.00125. |
| Reused temporary standard-library authoring validator | PASS: 72 local links/anchors, 83 pinned source line anchors, 63 symbols/constants across 22 pinned files, nine source-lock hashes, formulas, answer keys, result rows, headings, tables and fences. Adapted for the new snippets, Mermaid configuration and current documentation edits; cached source evidence reused. This temporary path is not a reader dependency. |
| Local browser rendering and visual inspection | All seven Mermaid diagrams rendered with Mermaid 11.12.0 in Chromium 159.0.8154.6 through Playwright 1.65.2. Markdown-it-py 4.3.0 and KaTeX 0.16.22 rendered Markdown and math. Inspected at a 1024-pixel viewport with an 880-pixel reading column. |
| Rendered presentation and navigation | Inspected diagram labels, arrow directions, branches, layout, all display equations, tables, chapter navigation and the existing results SVG. Changed three horizontal flows to vertical layouts and wrapped the sequence diagram. The onward navigation links in all eight chapters worked in the local render; Markdown targets were also checked by the validator. |
| `git diff --check` and diff-scope inspection | PASS. This pass edits learning Markdown only. The complete PR still contains eleven learning files and the existing narrow root README navigation change. |

The authoring-validator command was `python3 /tmp/agl-learning-review-20261118-8h_gso7m/verify.py /workspace/projects/repos/agent-lightning-lab`.

Rendering tools, downloaded browser libraries, assets and screenshots stayed outside the repository under a temporary directory. No project dependency declarations or lockfiles changed. The browser rendered locally; private guide content was not sent to an external rendering service. This completes the previously missing rendered check, but is not a claim of pixel-identical GitHub rendering.

The wide source-map tables require horizontal scrolling. The frozen accuracy SVG has crowded endpoint labels; Chapter 7 directs readers to the exact-count table. Neither prevents documentation review. The SVG and all frozen evidence remain unchanged.

Optional CPU backend integration tests, raw-trace/tensor verification, training/inference, paid cold builds and public-release checks remain unrun. The repository remains **PRIVATE**; licensing, privacy, redistribution and other publication gates are separate, unverified requirements. The missing metric series and chart collection remain unavailable as recorded in the source map.

For an implemented second task and reader-owned forks, see [Task adaptation](../TASK_ADAPTATION.md). Its asset-status checks are local; they do not add learning results to this guide.
