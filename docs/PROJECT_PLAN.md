# Project plan: training language models for agent use

**Version 1.2, November 27, 2026 UTC.** This is the shared implementation plan for the owner, ChatGPT, Codex, and other contributing agents. [Current continuation](PROJECT_CONTINUATION.md) records progress and the next action. [Agent instructions](../AGENTS.md) provide the session entry point.

## 1. Goal and completion standard

Build a rigorous, readable, step-by-step guide that explains how a language model becomes better at operating inside an agent harness. A first-year CS student who can read basic Python should be able to follow the main narrative, run the credential-free exercises, understand what the lab actually demonstrated, and find a supported procedure for their own separately funded experiment.

The guide's central question is:

> How do we train a language model for agent use, and what changes when training data comes from interactions with tools and an environment?

The guide should explain the path from an agent's experience to the exact tensors, advantages, losses, optimizer calls, inference refresh, checkpoint, and independent evaluation. It should connect those mechanics to the existing calculator, asset-status, and recovery experiments.

This project already has substantial implementation, mathematical explanations, real training, and reviewed evidence. **Use that work. Do not repeat completed experiments to populate the new structure.** Reorganize and explain what exists, and undertake only the explicitly identified remaining capability work.

The educational revision is complete when:

- One clear reading path leads from a brief agent-training introduction through the deep mechanics, practical execution, adaptation, and interpretation modules below.
- The main path works for a basic-Python reader without requiring a prior graduate RL course. Technical depth remains available in worked derivations and source references.
- Each substantial lesson includes a concrete example, a learner action, an expected result or answer, and a link to the actual implementation or evidence.
- A reader can distinguish a tool call from an episode, a fragment from a rollout, a reward from an advantage, and an optimizer call from a claim of useful learning.
- Existing results retain their task, algorithm, seed, panel, source, and evidence boundaries.
- Instructions distinguish free local checks from model execution and show where paid execution begins.
- A new agent can resume from the continuation without reconstructing the project from old conversations.
- The bounded recovery PPO-MC qualification is either completed and documented, or explicitly recorded as awaiting authorization, blocked, or deferred. Its execution does not block finishing the documentation.

The repository is already public. This is an educational revision of that publication, not a new general publication-readiness project. See the existing [publication record](PUBLIC_RELEASE_PLAN.md).

## 2. Decisions that govern the work

| Decision | Implementation consequence |
|---|---|
| Documentation quality is the highest priority | Deliver finished learner-facing lessons before more experiment planning. |
| The subject is training models for agents | Keep generic pretraining, transformer architecture, and general ML background brief and optional. |
| Foundations should establish a useful contrast | Compare conventional response/chat SFT and RL with agent SFT and RL, then move into harness-aware training. |
| Preserve technical depth | Retain the correct mathematics, exact code references, and implementation-specific qualifications already in the guide. Improve their teaching sequence. |
| Reuse completed experiments | Treat the evidence map in Section 4 as source material, not work to rerun. |
| Add task-specific breadth through recovery PPO-MC | Prepare one small existing-route qualification after the documentation work. Do not rebuild PPO. |
| Defer the proposed SFT experiment | PR #14 stays a separate optional intervention. Its proposed budget and CPU results are not full-size execution authority or evidence. |
| Keep new execution bounded | No automatic extra seeds, longer runs, final-panel evaluation, retries, or changes to the learning algorithm. |
| Maintain durable continuity | Keep stable direction here and mutable progress in `PROJECT_CONTINUATION.md`. Update both only when their respective content changes. |
| Adopt CMU-inspired practical teaching | Supply complete runnable examples, expected outputs, reproducible figures, and interpretation exercises within the existing structure. |
| Prepare one hosted-Tinker companion in parallel | The documentation path stays first. A separate track builds one small GRPO companion offline and returns runnable code, local validation and one exact execution proposal. [Scope and decision](#parallel-hosted-tinker-companion) |

The owner authorized creation, revision, and merge of PR #15 after final review. It does not turn every future milestone into an unrestricted implementation or spending mandate. An agent continues the work assigned by the owner and preserves any applicable standing authorization.

### The conceptual bridge

Use one task to explain four training situations:

| Situation | What supplies the learning material? | What the explanation should emphasize |
|---|---|---|
| Conventional response/chat SFT | Demonstrated responses conditioned on prompts or conversations | Learn to predict the selected response tokens. |
| Conventional response/chat RL | Sampled responses with outcome feedback | Use reward and an advantage estimate to update the response policy. |
| Agent SFT | Demonstrated model actions and answers, conditioned on the histories the harness constructs | Tool calls can be targets; tool observations are conditioning context. The chosen supervision mask determines what is actually trained. |
| Agent RL | Model actions produce tool/environment observations and an eventual task outcome | Collect and attribute the interaction, preserve its exact model inputs/actions, and construct valid training samples and credit assignments. |

Do not define the contrast as single-turn versus multi-turn. Chat training can be multi-turn. Agent training need not use a different model architecture or fundamentally new objective. Explain the changed behavior, data-generation process, context construction, credit assignment, and evaluation.

Then introduce the progression from response-oriented RL to trainer-owned agent interaction to harness-owned interaction. Agent Lightning's responsibilities become meaningful after the reader understands why a harness can construct a different prompt at each model call.

## 3. Agreed learning structure and reuse map

Working title: **Training Language Models for Agents: A Hands-On Guide to Harness-Aware Reinforcement Learning**.

The twelve modules below are a logical teaching structure. The repository currently has eight substantive chapters. Keep those files and their inbound links while developing the new organization. A module may be a section in an existing chapter. Split or rename files only when the finished content benefits, update inbound links, and record the final destination in the continuation. Do not create twelve empty replacement chapters.

### Part I. From language models to trainable agents

**1. Training an LLM for agent use**

- Question: What is being trained when a model operates tools?
- Reuse: [current Chapter 1](learning/01-agent-training-and-frameworks.md).
- Add the concise SFT/RL contrast above. Explain model, policy, action, observation, trajectory, and outcome through the example before introducing framework names.
- Distinguish a deterministic workflow, a compound AI system, and an autonomous agent. Explain that SFT is a learning objective while full training, selective parameter training, and LoRA are different parameter-update choices.
- Learner outcome: identify the trainable behavior and the fixed environment in a small interaction.

**2. Why the harness matters**

- Question: Who constructs the next prompt, executes the tool, handles an error, and decides whether to continue?
- Reuse: Chapter 1's existing responsibility discussion and the primary references in the [source map](learning/source-map.md).
- Explain control flow, tool observations, context changes, termination, and trainer-owned versus harness-owned interaction.
- Learner outcome: trace which component owns each decision and explain why training must respect the histories seen by the model.

Keep this opening short. Introduce probability, log probability, gradients, and expected return when the later worked example needs them. Do not add a preliminary generic neural-network course.

### Part II. Understanding harness-aware agent training

**3. Inside Agent Lightning**

- Question: How does an existing agent connect to a trainer without moving all interaction logic into the training loop?
- Reuse: Chapter 1's architecture table, [integration reference](INTEGRATION.md), and source map.
- Develop the map progressively: harness and tools; captured model calls and reward attribution; adapter; training backend; inference copy. Identify the roles of AutoGen/MCP, Agent Lightning, VERL, vLLM, and this lab's matched overlay.
- Explain retokenization/sample merging, advantages over dynamically produced samples, normalization by real rollout, and scheduling variable row counts as the problems the later modules solve.
- Learner outcome: locate the source component responsible for one observed behavior. Distinguish this pinned lab implementation from the paper's broader results and current upstream releases.

**4. Follow one agent trajectory**

- Question: How does one request become a completed, independently scored experience?
- Reuse: [current Chapter 2](learning/02-follow-one-calculator-task.md).
- Keep the synthetic `(7+5)*3` example, result `36`, as the continuous thread. Show the generated tool request, observation, continuation, final answer, verifier, and attributed reward.
- Add concrete schemas or shapes only after checking them against the pinned code. Label illustrative transcripts as synthetic, not recovered historical traces.
- Learner outcome: follow an episode and distinguish successful tool execution from a correct final answer.

**5. From agent experiences to training tensors**

- Question: Which tokens contribute to training, and why can one rollout produce multiple rows?
- Reuse: [current Chapter 3](learning/03-trajectories-tokens-and-fragments.md) and the existing CPU accounting example.
- Introduce concrete token/array views before notation. Preserve exact sampled IDs and log probabilities, chat-template boundaries, response masks, exact-prefix merging, fragments, and real-rollout identity.
- Derive equal-rollout weighting across generated tokens and fragments. Explain why observations can affect predictions while carrying zero direct policy-target weight.
- Learner outcome: mark action versus observation tokens and verify that splitting a rollout does not increase its total training weight.

**6. GRPO for agents**

- Question: How do several attempts on the same task produce a learning signal without a learned critic?
- Reuse: [current Chapter 4](learning/04-grpo-step-by-step.md).
- Retain the four-reward example, sample-standard-deviation calculation, constant groups, token broadcasting, probability ratio, clipping, and equal-rollout reduction.
- Preserve the zero-scalar-loss/nonzero-gradient example. Explain the meaning of each diagnostic before interpreting it.
- Learner outcome: compute one group's advantages and explain why an all-zero or all-one group supplies no relative reward contrast.

**7. PPO for agents: this lab's PPO-MC**

- Question: What changes when a learned value model supplies the baseline?
- Reuse: [current Chapter 5](learning/05-ppo-mc-step-by-step.md).
- Start with the critic's prediction task. Then derive general GAE and the lab's true-terminal Monte Carlo specialization with gamma and lambda equal to one.
- Retain valid-token whitening, its distinction from rollout loss weighting, the separate critic, clipped value loss, and actor/critic optimizers.
- Learner outcome: compute a small return/advantage example and explain why a trained critic may still predict poorly. Do not generalize this configuration to all PPO.

**8. One complete training iteration**

- Question: When do collected actions become an actual model update and a new serving policy?
- Reuse: [current Chapter 6](learning/06-one-complete-training-update.md).
- Trace the existing collection of eight questions times four real rollouts through dynamic fragments, old log probabilities, advantages, microbatches, backward accumulation, optimizer calls, inference refresh, checkpoint, and fresh-worker evaluation.
- Show current actor, behavior/old probabilities, inference weights, and optional reference policy as distinct roles. State what the recorded diagnostics actually compare.
- Learner outcome: count collections, fragments, backward passes, and optimizer calls without confusing them. Distinguish evaluation reload from resumed training.

This is the core of the guide. Deep explanations belong here, tied to the real implementation rather than a new miniature training framework.

### Part III. Run and adapt the training system

**9. Run your first real training experiment**

- Question: What can I check locally, and what does a separately funded model run require?
- Reuse: [current Chapter 8](learning/08-reproduce-and-operate.md), the [reader procedure](READER_TRAINING.md), and its [live qualification](READER_QUALIFICATION_2026-11-19.md).
- Begin with the two existing credential-free commands. Then guide the reader through local client setup, their own resources, plan/prepare/preflight/launch/status/collect/stop, baseline, training, and independent reload.
- For each command, state where it runs, its inputs, expected evidence, and the specific failure that stops progression. Keep local Python/client dependencies separate from remote CUDA/training dependencies.
- Learner outcome: understand and reproduce the supported workflow within their own authorization. Do not make private operator artifacts a prerequisite.

**10. Train an agent for a different task**

- Question: Which parts change when the agent's job changes?
- Reuse: [task adaptation](TASK_ADAPTATION.md), [asset closeout](PHASE3_CLOSEOUT_2026-11-24.md), and the [recovery task](PHASE4A_RECOVERY_TASK.md).
- Follow calculator to asset-status to recovery. Identify changes to task data, tool schemas, prompt, harness, answer contract, independent verifier, and evaluation panels while retaining the training infrastructure.
- Use the asset episode-contract corrections to explain policy mistakes, recoverable tool errors, terminal failures, and invalid infrastructure evidence.
- Learner outcome: specify a small new task and its verification contract without changing the trainer to fit an example.

### Part IV. What our experiments teach us

**11. GRPO versus PPO-MC in practice**

- Question: What do the completed comparisons establish, and what remains uncertain?
- Reuse: [current Chapter 7](learning/07-read-the-experiment-results.md) and the six-run calculator evidence.
- Preserve each baseline, seed pair, denominator, critic diagnostic, and reused-panel limitation. Present cost/time measurements with their original scope and measurement date.
- Learner outcome: interpret a real result without turning descriptive changes into superiority, equivalence, or generalization claims.

**12. When reinforcement learning goes wrong**

- Question: How can the training system work while the resulting policy becomes less useful?
- Reuse: the [recovery pilot closeout](PHASE4C_FUNCTIONAL_PILOT_CLOSEOUT.md) and [learning diagnosis](PHASE4C_PILOT_LEARNING_DIAGNOSIS.md).
- Explain the P02 terminal all-`not_ready` policy, class-specific strict results, unchanged decision accuracy, evidence checks, reward opportunities, and limits on causal inference.
- Contrast functional qualification with behavioral learning. Present witness-selection SFT only as a deferred, locally implemented proposal with its actual supervision scope and unmeasured full-size outcome.
- Learner outcome: choose the right next diagnostic or bounded intervention rather than interpreting an aggregate reward increase as success.

### Pattern for each substantial lesson

Use: a concrete question; a small running example; intuition; exact mechanics and defined symbols; the relevant code; a short exercise or inspection task; an expected result; and the scope of the evidence. Avoid mechanically repeating headings when connected prose works better.

Diagrams should explain ownership, event order, or tensor relationships. Keep labels short and introduce one relationship at a time. Tables should retain exact comparisons and denominators. Equations and examples must be checked against the pinned implementation. No screenshots or new experiments are required merely to make a page attractive.


### Three ways to use the guide

| Path | Reader activity | Requirements | Evidence produced |
|---|---|---|---|
| Understand the mechanics | Run synthetic reward, mask, advantage, and weighting examples | Python; no model or account | Checked arithmetic and implementation invariants |
| Analyze completed experiments | Reproduce tables/figures and inspect public paired cases | Committed evidence and the documented plotting environment | Derived analysis of existing execution |
| Run a new experiment | Follow the supported baseline, training, checkpoint, and reload workflow | Separately authorized model access, compute, and budget | A new run with its own evidence |

The first two paths should provide a complete learning experience. Do not make a GPU run or an account a requirement for understanding the core material. Keep raw-evidence replay, public derived-data checks, and new execution distinct.

### Practical lesson contract

Each practical lesson supplies:

1. A one-sentence purpose.
2. A small prediction before execution.
3. A copyable command with its working directory, prerequisites, and required files.
4. Expected output, including exact values or an explicit tolerance/invariant.
5. A figure or table showing those same values.
6. An explanation connecting the result to the training mechanism.
7. One optional alteration or implementation challenge, with an answer.
8. The evidence boundary: synthetic illustration, recorded-data analysis, or new model execution.

Provide a complete working reference path before optional exercises. Required instructions must not depend on unfinished functions or unexplained placeholders. Do not present proposed helper commands as available until they exist and have been checked. Show meaningful setup placeholders clearly when a reader must supply their own values.

Include the example inputs, figure-generation code, rendered figures, and explanations in the public learning path. Keep one implementation of each calculation. An optional notebook should call the same helpers as the command-line path. Use a small documented plotting environment; do not add another training framework or a mandatory notebook service.

### Figure plan from existing evidence

| Figure | What the reader should learn | Existing source |
|---|---|---|
| Synthetic token/action-mask strip | Generated actions and observations play different roles in the loss | Current Chapters 2–3; verify concrete schemas against the pinned code |
| Four-attempt reward/advantage bars | Relative reward, scaling, and constant-group behavior | Current Chapter 4's synthetic example |
| Fragment-weight comparison | Splitting a rollout must preserve its intended total weight | [CPU accounting example](../experiment/cpu_accounting_demo.py): row mean 5 to 3.5, equal-rollout mean 5 to 5 |
| Per-seed baseline/reload points | Means can hide differing baselines and seed outcomes | [Calculator per-seed results](../evidence/phase2-r03/per_seed_results.csv) |
| P02 strict success by actual class | Aggregate reward can conceal class collapse | [Paired development cases](../evidence/phase4c-functional-pilot/diagnosis/development.json): ready 6/12 to 0/12, negative 3/12 to 12/12 |
| Paired-case outcome grid | Net change hides gains and regressions | Same paired cases: nine strict gains, six regressions, three retained successes, six retained failures; show decision correctness separately |
| P02 reward-group composition | Successes and positive relative advantages are different counts | [Collections](../evidence/phase4c-functional-pilot/diagnosis/collections.json): ready all-zero/mixed/all-one 30/32/2; negative 6/44/14 |
| Per-collection class-specific strict rate | Sampled behavior varies with policy, questions, and class denominators | Same collections; show denominator and sampled policy version |
| Separate scalar-loss and gradient panels | A near-zero scalar policy loss can coexist with a nonzero gradient | Same sixteen recorded actor updates |

Implement figures in their owning M2/M4 lessons, reusing existing plots where appropriate. The P02 class and group figures are the first experimental examples to finish; do not regenerate every historical figure for stylistic consistency. Label axes, units, denominators, sources, and synthetic values. Keep calculations reproducible and preserve original evidence bytes.

P02 collection k samples policy k-1. Training questions and class composition change, so these observations are not repeated evaluations of a fixed panel. Only baseline and terminal greedy development evaluations are available. Do not invent intermediate development scores, missing raw transcripts, entropy, base-reference KL, or the unavailable historical full metric atlas. Use separate scales/panels for loss and gradient norm rather than implying a direct magnitude comparison.

### CMU and Tinker references

The pedagogical reference is [CMU 11-768 Assignment 3](https://github.com/cmu-agents/assignment-3/blob/88d1b1fa2aea58938d6d251847ce5559c4ac6558/ASSIGNMENT.md), inspected at commit `88d1b1fa2aea58938d6d251847ce5559c4ac6558`. Borrow its progression from data to updates and its use of diagnostics and interpretation prompts. This guide supplies a complete reference path, with optional challenges, for its broader beginner audience.

Do not silently import its numerical choices. The assignment's [advantage contract](https://github.com/cmu-agents/assignment-3/blob/88d1b1fa2aea58938d6d251847ce5559c4ac6558/scottyrl/advantages.py) uses mean-centering without reward-standard-deviation normalization; its [batch contract](https://github.com/cmu-agents/assignment-3/blob/88d1b1fa2aea58938d6d251847ce5559c4ac6558/scottyrl/train/common.py) omits constant-reward groups from training sequences; its [adapter](https://github.com/cmu-agents/assignment-3/blob/88d1b1fa2aea58938d6d251847ce5559c4ac6558/scottyrl/tinker_adapter.py) uses a global action-token normalization. Preserve this lab's standardized advantages, declared rollout admission, and equal-real-rollout weighting.

Introduce the following ownership comparison in Modules 3/9, and link the [sequence-extension tutorial](https://tinker-docs.thinkingmachines.ai/tutorials/advanced/sequence-extension/) from Module 5:

| Stack | Agent/data/learning logic | Training and sampling backend | Infrastructure |
|---|---|---|---|
| Existing lab | Harness, Agent Lightning adapters, matched GRPO/PPO-MC | VERL and vLLM | Modal |
| CMU assignment | scottyrl harness and student training/data code | Tinker-compatible API, SkyRL, Megatron, vLLM | Modal |
| Hosted-Tinker companion (planned) | Our adapted harness, verifier, advantages, loss, and evaluation | Thinking Machines' service through the Tinker API | Managed by Thinking Machines |

CMU's [deployment guide](https://github.com/cmu-agents/assignment-3/blob/88d1b1fa2aea58938d6d251847ce5559c4ac6558/skyrl_tinker_modal/README.md) uses Tinker-compatible APIs on Modal. These are different layers, not interchangeable providers. Hosted Tinker's [current overview](https://tinker-docs.thinkingmachines.ai/tinker/) describes LoRA training; its [model catalog](https://tinker-docs.thinkingmachines.ai/tinker/models.json) did not establish support for this lab's exact pinned Llama checkpoint at the November 26 review. Recheck model availability when preparing the companion proposal.

A loss named `ppo` is an actor surrogate, not proof that this lab's learned-critic PPO-MC has been ported. A sampling client reload does not by itself prove the distinct physical worker/container identity measured on Modal. Compare the supported model/revision, parameterization, objective, masks, weighting, checkpoint guarantees, and all relevant costs before making parity claims. Hosted token-based charges and Modal resource-time charges require different accounting. No lower-cost outcome is established here.

### Parallel hosted-Tinker companion

The owner reports a $250 credit grant that may expire December 15, 2026. The remaining balance, exact expiry and grant-specific eligibility are unverified. Published prices can support a cost estimate. Only account-specific evidence can establish available credit and applicable terms.

Prepare one small hosted-Tinker GRPO companion alongside the guide. Deliver a runnable implementation, local credential-free validation of its data and update inputs, and one exact proposed run. Identify the source revision, command, model, task, sampling and update counts, token limits, expected evidence, maximum exposure and stop condition. Separate local results from any behavior that requires the hosted service. The owner decides on credit-consuming execution after reviewing those deliverables and the grant terms. Offline preparation has zero dispatch authority.

The documentation workstream maintains the shared plan and continuation and records the companion branch/PR when known. The core chapters and planned recovery PPO-MC qualification proceed independently. Use the completed studies as teaching evidence.

## 4. Existing experiments to teach from

The following is a reuse index, verified against main `e463dc146db3a4be7e156a06109a749db5f8649e`. The linked reports own the detailed evidence. This plan does not supersede their measurements.

| Existing work | Established evidence | Use in the guide | Claim boundary |
|---|---|---|---|
| [Calculator r03](PHASE2_CLOSEOUT_2026-11-15.md), GRPO and PPO-MC | Three seeds per algorithm; 20 actor calls per run; 20 critic calls per PPO run; 640 training trajectories per run; independent final reloads | Modules 5–8 and 11: mechanics and actual seeded comparison | Reused 64-case development panel; no superiority, equivalence, or fresh-holdout generalization |
| [Public reader calculator](READER_QUALIFICATION_2026-11-19.md), both algorithms | Real baseline, two-update training, native checkpoint, distinct-worker reload | Module 9: supported reader execution | GRPO had zero informative groups; operational completion is not useful learning |
| [Asset-status qualification](PHASE3_QUALIFICATION_2026-11-24.md), both algorithms | Baseline 64, training 64, reload 64 per route; two actor calls and PPO's two critic calls | Modules 9–10: integration and adaptation | Both baseline and reload were 64/64; harness correction is not weight-learning improvement |
| [Recovery calibration](PHASE4B_LIVE_CALIBRATION.md), both routes | 120 admitted episodes per route, all rewards confirmed, zero optimization | Modules 10 and 12: task difficulty and reward variance | Collection does not qualify PPO backward, checkpoint, or reload |
| [Recovery GRPO A03](PHASE4C_GRPO_CAMPAIGN_CLOSEOUT.md) | 144 admitted episodes, two actor calls, independent step-2 reload | Modules 8–10: functional gates | Development strict result stayed 9/24; short qualification does not establish sustained learning |
| [Recovery GRPO P02](PHASE4C_FUNCTIONAL_PILOT_CLOSEOUT.md) | 560 admitted and reward-confirmed episodes; 16 actor calls; refreshes 0–16; distinct-worker step-16 reload | Modules 8 and 12: sustained operation, resource use, and failure analysis | Strict 9/24 to 12/24 masked all-negative terminal decisions; decision accuracy stayed 12/24 |
| Recovery P01 partial allocation, in the pilot closeout | 536 admitted episodes and completed training before preemption interrupted reload | Optional operations example | Keep its checkpoint, cost, and partial observations separate from completed P02 |
| [Witness SFT PR #14](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/pull/14) | Frozen 32 demonstrations and tiny-CPU training/reload verification | Module 12: optional targeted intervention | Draft/unmerged at verification; no full-size GPU fit, serving, or behavioral result; deferred |

Historical calculator execution commit `5fb6adfd48ee245509f7dff3ab1d849fcf707d86` belongs to the original history. Its public source representation is not a replacement execution identity. Preserve the [source map](learning/source-map.md#public-source-representations) and [publication provenance](../PUBLICATION_PROVENANCE.json).

The completed three-seed calculator study is separate from the unexecuted full recovery study. The latter is three training seeds plus the base policy, eleven stages and 1,888 episodes. P02 ran only the explicit seed-2001, three-stage, development-only pilot. Do not describe seeds 2002/2003 or final-panel evaluation as the remaining automatic steps.

### Reuse without manufacturing new evidence

- Preserve historical reports, frozen evidence, hashes, and exact execution identities. Link to them from new teaching prose.
- Correct clearly current summaries that have become stale. Do not globally replace dated `not run`, `PRIVATE`, or approval statements in historical records.
- Derive teaching tables or calculations from existing public evidence when helpful. Label them as derived and retain the source and denominators.
- Label new conceptual traces, token IDs, probabilities, and diagrams as synthetic when they are not recorded observations.
- Do not re-download models, rerun old campaigns, reconstruct missing private traces, or regenerate immutable evidence to improve presentation.
- Existing CPU examples and bounded offline validations may be run when they verify a changed lesson. They do not create new model-execution evidence.

## 5. Implementation milestones

Use one focused PR per coherent slice. M2 may need several slices because it contains the central technical material. Record progress by milestone and concrete deliverable, not by number of pages written. Do not mark a milestone complete because an outline exists.

### M0. Publish the shared plan and continuation

Deliver this plan, the root agent entry point, the current continuation, and small links from the root README and learning index.

Acceptance: another agent can determine the objective, completed evidence, next task, and authority from the repository alone. The change is documentation only. It does not claim that the educational revision or new qualification has already been completed.

### M1. Deliver the entry path and a finished representative lesson

Start with the existing README, learning index, and Chapters 1–2. Add the brief agent-training contrast, harness bridge, and a polished end-to-end trajectory lesson. Establish one default reading route and place advanced/archival material behind purposeful links.

In the same bounded documentation work, reconcile the mutable current recovery summaries in:

- `docs/READER_TRAINING.md`: separate completed GRPO A03/P02 from unqualified recovery PPO-MC and the unexecuted full study.
- `docs/TASK_ADAPTATION.md`: update the opening recovery status that still describes only the earlier offline/calibration phase.
- `docs/learning/source-map.md`: correct the current adaptation footer without changing the historical public-source mapping.

Acceptance: a basic-Python reader can explain what changes during training, who owns the interaction, how an episode earns reward, and where the next lesson begins. The representative lesson is finished content, not another proposed outline. Apply the practical lesson contract to its reader actions, linking an existing credential-free example with its actual expected output. Keep optional challenges after the working reference path. New factual claims link to existing evidence. M1 does not need to implement the entire figure catalog or a new backend.

### M2. Develop the deep training mechanics

Work through Modules 3–8 using existing Chapters 1 and 3–6. Suggested slices: architecture/tensors; GRPO; PPO-MC/complete update. Keep the same continuous calculator example and consistent notation. Include runnable token-alignment, mask, advantage, fragment-weight, and update-accounting examples, using the practical lesson contract and relevant synthetic figures.

Acceptance: examples and formulas match the actual masks, normalization, advantages, clipping, and optimizer ordering. Each lesson has a checked exercise and a source pointer. The reader can work from an interaction through the loss to the actual optimizer and serving refresh. Preserve the runtime and numerical backend. A small teaching helper is acceptable only when existing examples cannot demonstrate the concept clearly; it must stay separate from production training and be labeled synthetic.

### M3. Make execution and task adaptation teachable

Develop Modules 9–10 from the existing reader and adaptation procedures. Keep the canonical commands in one maintained procedure and link from the narrative instead of creating conflicting copies. Show the three reader paths explicitly, and explain backend ownership. Keep companion implementation in its parallel track.

Acceptance: steps identify the terminal/environment, prerequisites, exact input, expected output/evidence, cost boundary, and relevant failure diagnosis. Public readers can complete the offline route without private resources. Paid commands are explained and locally checked where applicable; running them is not a documentation acceptance gate. The local client and remote GPU dependencies remain separate.

### M4. Integrate the experiments into the teaching story

Develop Modules 11–12 from the evidence index. Show the completed calculator comparison and recovery failure as different experiments answering different questions. Use asset-status to illustrate how harness fixes can change behavior without training. Supply the relevant figure-generation helpers, rendered figures, exact expected counts, and interpretation exercises from the figure plan. Prioritize the P02 class/group figures and existing calculator per-seed results.

Acceptance: results have source, task, algorithm, denominator, panel, and execution identity. The all-negative recovery policy is visible alongside aggregate strict reward. Observations, mechanistic explanations supported by code, hypotheses, and proposed interventions are distinguished. No new model run is needed.

### M5. Prepare the bounded recovery PPO-MC qualification

After the main educational content is in place, reuse the existing recovery-learning route and prepare the exact proposal in Section 6. Inspect implementation changes since the last qualified source. Run only the relevant local contract/admission/numerical checks needed by that proposal. Do not reopen a broad readiness review or rebuild the framework.

Acceptance: a reviewable source/profile and exact four-stage plan exist; expected evidence, capacity assumptions, current prices, maximum exposure, retention horizon, and replacement policy are explicit. Distinguish historical measurements from new estimates and identify any unmeasured recovery-PPO fit. The agent completes this preparation before requesting the separate paid authorization.

### M6. Execute and close recovery PPO-MC, only if authorized

Carry out exactly the approved plan, reconcile its evidence, and report functional and behavioral outcomes separately. Preserve any granted standing authority for its routine steps. If authorization is not granted, record that state and continue to M7.

Acceptance: the Section 6 gates are met or the failure is documented with the admitted/partial evidence, resource disposition, charges, holds, and remaining authority. A failed behavioral result does not justify another attempt. A technical failure permits only the diagnosis or replacement explicitly covered by the new mandate. Update the teaching material with the result without recasting it as an algorithm-comparison study.

### M7. Complete the reader walkthrough and editorial closeout

Walk through the learner path using the public documentation and the changed offline examples. Verify local links, relevant pinned source references, rendered diagrams/math, prerequisites, and expected outputs. Review technical correctness and beginner readability separately. Keep historical provenance accessible without making it the opening narrative.

Acceptance: the Section 1 completion standard is met; the continuation records the final documentation commit/PR, actual checks and material limitations, and the disposition of recovery PPO-MC. No generic release audit, new model campaign, framework migration, or final-panel run is required. A real first-time reader trial is useful if available; do not claim one occurred merely because an agent reviewed the prose.

## 6. Bounded recovery PPO-MC qualification

This work answers: **Does the existing PPO-MC route complete real actor/critic training and a fresh-worker checkpoint evaluation on the recovery task?** PPO has already functioned on calculator and asset-status. The remaining gap is recovery-specific integration and capacity.

Use the implemented `ppo-mc` selection in [recovery_learning.py](../experiment/recovery_learning.py), the [qualification contract](PHASE4C_RECOVERY_LEARNING.md), and existing admission/evidence machinery. Qualification seed is **9001**. It is not a study seed or a continuation of P02's checkpoint.

| Stage | Starting policy and operation | Episodes | Optimizer calls |
|---|---|---:|---|
| Baseline | Fresh base, greedy development | 24 | 0 |
| Collection probe | Fresh base, eight predetermined training questions times four samples | 32 | 0 |
| Training | Fresh base, two collections of eight questions times four samples | 64 | 2 actor and 2 critic |
| Checkpoint evaluation | Distinct worker loads terminal step 2, greedy development | 24 | 0 |
| **Total** | One route, four stages, one lane | **144** | **2 actor and 2 critic** |

The probe uses the first predetermined training question batch. Training starts afresh and recollects it; update two uses the disjoint next batch. Keep the declared identities and order. Do not select or replace questions based on observed rewards.

### Functional gate

The existing contracts remain authoritative. The implementation and review must establish:

1. All 144 declared episodes are admitted and every reward is independently confirmed, including valid zeros.
2. Each training collection contains eight complete groups of four distinct real rollouts. Prompt, action-token, sampled-log-probability, controller, and episode identities reconcile at admission.
3. Two actual actor and two actual critic optimizer calls are recorded, with finite losses and gradient evidence. Finite zero values are admissible under the existing qualification contract.
4. Policy refreshes occur at steps 0, 1, and 2. Evaluation stages perform no optimization.
5. The terminal native step-2 checkpoint has the required role inventories and source/model/tokenizer/data/configuration bindings.
6. Reload uses a distinct provider call and container, with matching inventory commitments before loading and after evaluation.
7. Class/family results, assessability, gains/regressions, and relevant critic diagnostics are reported separately from technical completion. There is no improvement threshold for this functional gate.
8. Capacity, timing, observed usage, outstanding holds, retained bytes, resource closeout, and the allocation's own UTC retention review date are recorded. Sampled memory maxima must not be labeled true peaks.

### Proposal and spending boundary

The proposal must specify exact source and resources; worker and startup limits; actor/critic memory and checkpoint assumptions; cold-build/preflight allowance; current provider pricing; full cumulative reservation; delayed-billing/storage holds; and explicit replacement authority, if any. Do not assume a replacement is permitted.

The older `$113`/`$82` worksheet in the recovery qualification document is a **GRPO** worksheet. The closed pilot's `$150` ceiling and PR #14's proposed `$25` do not fund PPO. A learned critic changes resource requirements. Do not quote a new recovery-PPO cost as measured before execution.

Prepare the concrete proposal first. Ask for paid authorization only for the unresolved execution decision, identifying the exact scope and ceiling. Once granted, preserve that authority across sessions and do not ask again for routine steps inside it. New retained artifacts need their own review horizon; historical review dates do not authorize deletion.

## 7. Technical and editorial invariants

These rules protect the meaning of the guide during simplification:

- A tool observation is context, not automatically a supervised target or a policy action. Zero direct target weight does not mean it cannot affect predictions through attention.
- A response mask identifies target positions; it is not a reward label. A fragment is not another independent rollout.
- Group-relative advantages, token whitening, and loss weighting are separate operations. Use this implementation's actual conventions.
- A zero scalar policy-gradient loss can have a nonzero gradient. An optimizer call does not prove every parameter changed or that the change was useful.
- The critic predicts return. The independent verifier checks task correctness. They are different components with different evidence.
- Within-update actor/old-policy diagnostics do not establish cumulative KL from the original base. Unavailable entropy or base-reference KL stays unavailable.
- Fresh-worker evaluation reload is not proof of resumed optimizer-state training, arbitrary export, or all future environments.
- A correct aggregate score can hide class collapse. Always preserve the recovery decision/evidence distinction and assessment denominators.
- Configured seeds and pinned inputs do not guarantee identical GPU outputs across every future allocation. Do not promise that an arbitrary number of seeded runs will succeed or improve.
- Explain SFT PR #14 precisely if mentioned: final-answer tokens conditioned on demonstrated conversations, selective dense query/value projection updates, tiny-CPU verification, and an unmeasured full-size proposal. Do not call it demonstrated end-to-end action imitation or completed LoRA training.
- Introduce terms before abbreviations, define every mathematical symbol, and make the example earn the notation. Keep operational receipts in reference material where possible.

## 8. Agent workflow, validation, and continuity

### Start a session

1. Read `AGENTS.md`, this plan, and the current continuation.
2. Inspect live main/PR state and the existing working tree. Reconcile concrete differences since the recorded revision. Preserve unrelated changes; do not reset another session's work.
3. Identify the first unfinished work item within the owner's current assignment. Read its existing material and evidence before editing.
4. State the bounded deliverable and proceed. Ask only when a real unresolved choice or authorization blocks the next action.

Use the existing Unix AI Workbench and compatible environments. Do not create a new development container or local CUDA stack just to author documentation. Parallel reviewers can inspect pedagogy and technical claims; use one accountable writer per file or coordinate ownership before edits.

### Validate the changed surface

For documentation-only slices, check links/anchors, examples, claims against the cited evidence, and diagrams/math when they change. Use the existing offline exercises where relevant. Do not rerun every historical regression suite or model campaign simply because a chapter changed.

For a changed teaching script, check its specific educational invariant and expected output. For a runtime change needed by M5, use the relevant existing contract and numerical tests; record the exact tested source and distinguish local checks from GPU qualification. Do not describe old test counts as freshly run.

PR descriptions should explain the learner's problem, the changed experience, evidence reused, and actual validation. Record a remaining limit when it affects interpretation. Keep scientific evidence and runtime changes separate from editorial changes whenever practical.

### End a session

Update `PROJECT_CONTINUATION.md` in the same implementation change with:

- Verified baseline, working branch/PR, and applicable implementation commit.
- Milestone/deliverable completed and acceptance evidence, or the precise incomplete state.
- Actual commands/checks and results, plus material checks not run.
- File moves or final module destinations, if any.
- The single next bounded action and relevant files.
- Blockers, unresolved decisions, and authorization still applicable.
- Retention or execution-ledger references if that session changed them.

Keep detailed private calls, container/app IDs, credentials, and retained-resource inventories in their existing private operator notes. The public continuation contains safe summaries and references, not copied private inventories. If a later authorized execution requires an unavailable private handoff, recover that specific input rather than inventing it.

The plan owns intent and acceptance criteria. The continuation owns present progress. Existing reports own experimental evidence. Avoid a second competing roadmap or a new handoff file for every chat.

### Copyable continuation instruction

```text
Continue amulyaparthasarathy/agent-lightning-grpo-ppo using AGENTS.md,
docs/PROJECT_PLAN.md, and docs/PROJECT_CONTINUATION.md.
Inspect live GitHub and local changes first, then implement the next unfinished
milestone within my current authorization. Reuse the completed experiments and
existing chapters. Do not restart planning or repeat old runs. Preserve the
documentation priority, parallel offline Tinker scope, planned recovery PPO-MC
qualification and all deferred-work boundaries.
Update the continuation with actual work, checks, authority, and one next action.
```

That instruction points an agent to the plan. It does not itself authorize paid execution or expand a read-only role.

## 9. Deferred work and reopening conditions

Witness SFT execution or merge, additional recovery GRPO seeds, the full recovery study, extended PPO studies, final-panel model evaluation, other alternative backends, resumed-training qualification and new deployment or export campaigns remain deferred. The [bounded hosted-Tinker companion](#parallel-hosted-tinker-companion) has a parallel offline implementation track. Hosted execution requires a separate owner decision on its exact proposal.

Do not keep these as mandatory unfinished tasks in the educational release. Mention them as possible future work with their uncertainties. Reopen one only when its learning or engineering value justifies the time/cost and the owner selects it.

If the owner changes a priority, update the decision table and continuation with the reason and affected milestones. Preserve already-completed evidence. The purpose of the plan is to carry the project forward, not to force every earlier proposal to be executed.
