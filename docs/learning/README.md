# From a calculator request to a model update

**Start with [Chapter 1 · Why train through the agent harness?](01-agent-training-and-frameworks.md)**

All you need to start is basic Python. In Chapter 2 you'll trace one attempt at `(7+5)*3`, predict its reward and check the result in a terminal. Have Python 3.12 and a [local copy of the repository](../../README.md#credential-free-offline-quick-start) for that exercise. No packages, model or account are required.

## Read the chapters in order

| Chapter | What you'll do |
|---|---|
| [1 · Model and harness](01-agent-training-and-frameworks.md) | Identify who chooses the calculation, who performs it and what training can change. Compare learning from a demonstration with learning from an outcome. |
| [2 · One calculator task](02-follow-one-calculator-task.md) | Follow two model calls in one attempt. Separate the tool's result from the model's final answer and score that answer. |
| [3 · Trajectories, tokens and fragments](03-trajectories-tokens-and-fragments.md) | Mark which pieces of text the model generated. See why one attempt can occupy several training rows. |
| [4 · GRPO](04-grpo-step-by-step.md) | Compare four attempts at the same question and work out which actions the trainer should encourage. |
| [5 · PPO-MC](05-ppo-mc-step-by-step.md) | Compare an outcome with a second model's prediction of that outcome. |
| [6 · One complete training update](06-one-complete-training-update.md) | Follow the calculations that change the model's weights and see when the next attempt can use them. |
| [7 · Read the results](07-read-the-experiment-results.md) | Compare each calculator run with its own baseline. Use paired recovery cases to find the decision errors hidden by a higher score. |
| [8 · Reproduce and operate](08-reproduce-and-operate.md) | Reproduce the calculator means, [change the final asset answer](../TASK_ADAPTATION.md#a-successful-lookup-and-two-different-outcomes), or [plan a reader-owned run](../READER_TRAINING.md#try-a-plan-without-an-account). |

**What counts as evidence?** The calculator dialogue and four rewards `[0, 0, 0, 1]` in the worked examples are invented to explain the mechanics. Chapter 7 uses measured results from three seed pairs on the same 64 development questions. That comparison leaves open how the policies would perform on fresh questions or with other seeds. A new measurement requires separate model execution. [Calculator closeout](../PHASE2_CLOSEOUT_2026-11-15.md)

The [glossary](glossary.md) defines terms as you need them. For task adaptation, see the [calculator and asset walkthrough](../TASK_ADAPTATION.md). For an example where a higher score hid a policy that missed every ready case, see the [recovery diagnosis](../PHASE4C_PILOT_LEARNING_DIAGNOSIS.md).

## Source and guide history

The calculator study `agl-explore-20-v1-r03` used `matched-rollout-mean-v2` and `exploratory-matched-20-v1`. Its historical execution commit was `5fb6adfd48ee245509f7dff3ab1d849fcf707d86`. The [source map](source-map.md#public-source-representations) identifies the public representations of that code. The [project plan](../PROJECT_PLAN.md#3-agreed-learning-structure-and-reuse-map) maps twelve teaching modules onto the eight chapters and their supporting guides. The [current continuation](../PROJECT_CONTINUATION.md) tracks implementation.

<details>
<summary>Historical authoring and verification receipts from the original repository</summary>

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

</details>
