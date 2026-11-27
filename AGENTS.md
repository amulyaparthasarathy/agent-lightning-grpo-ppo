# Instructions for project contributors and AI agents

## Start every session here

This is `amulyaparthasarathy/agent-lightning-grpo-ppo`, the public educational repository. The older `agent-lightning-lab` repository is historical.

Read these files in order before proposing or implementing the next change:

1. [Project plan](docs/PROJECT_PLAN.md): purpose, agreed structure, reuse map, milestones, and acceptance criteria.
2. [Current continuation](docs/PROJECT_CONTINUATION.md): verified state, active milestone, next action, and authority.
3. The existing source and evidence linked by that milestone.

Inspect current GitHub main, relevant PRs, and the local working tree. Reconcile the changes since the continuation's recorded revision. Preserve unrelated work and reuse the existing Workbench and compatible environments. Do not restart planning when the next authorized task is already clear.

## Working priorities

- First, develop the step-by-step guide to **training language models for agent use**. Keep general foundations brief. Explain agent interaction, harness responsibilities, trajectories, training tensors, GRPO, this lab's PPO-MC, and the complete update lifecycle in depth.
- Use the existing experiments as teaching evidence. Completed calculator, asset-status, recovery calibration, A03, P01, and P02 runs are not a backlog to repeat.
- Give practical lessons a complete reference path: purpose, prediction, copyable command, expected output, figure or table, explanation, one optional small alteration, and evidence boundary. Reproduce teaching plots from public committed evidence; keep optional implementation challenges separate.
- Keep one active implementation workstream. Finish the core learner documentation before returning to the hosted-Tinker companion. PR #22 is implemented, locally tested and paused before hosted execution. Preserve its branch and proposal. [Companion disposition](docs/PROJECT_PLAN.md#hosted-tinker-companion)
- The documentation workstream owns the shared plan and continuation. M3 is integrated. Review M4 in Chapter 7 before M7’s reader walkthrough and editorial closeout.
- After the core guide, prepare the bounded **recovery-specific PPO-MC qualification** described in the plan. PPO already has live calculator and asset-status evidence.
- Witness-selection SFT PR #14, additional recovery GRPO seeds, long studies, and final-panel model evaluation are deferred.

## Scope and authority

Follow the user's current instructions and any still-applicable authorization in the session. The plan records direction; it does not grant a spending mandate or override the user's scope.

The owner authorized the ordinary merge of M3 in PR #23 and delivery of M4 for review. M4 stops at the review boundary. ChatGPT otherwise retains its read-only repository/cloud role unless the owner extends it. For Codex and other implementation agents, work only within the task actually assigned. Complete already-authorized local work without repeated permission requests.

There is currently **zero further cloud dispatch authority**. Historical ceilings, unused balances, proposals, and `YES` instructions from closed mandates do not fund new execution. Prepare an exact proposal before requesting new paid authorization; preserve that authority once granted. No cloud execution, resource mutation, deletion, or experiment expansion follows automatically from a documentation task. Do not merge unrelated PRs.

`docs/PORTABLE_HANDOFF.md` and dated execution documents are historical records. Use the current continuation for today's priorities and authorization status.

## Implementation and reporting

- Preserve historical evidence bytes, source identities, numerical behavior, and recorded limitations. Correct mutable status summaries by linking the later evidence.
- Treat the twelve planned modules as a teaching structure. Reuse the existing eight chapters; do not rename or replace them wholesale.
- Label synthetic examples, offline checks, CPU fixtures, live GPU execution, and measured behavior distinctly. Do not convert successful execution into a claim of reliable learning or arbitrary-seed reproducibility.
- Validate the changed surface. Documentation does not require rerunning model experiments or every historical test suite. Record only checks actually performed in the current work.
- Keep public documentation free of credentials and private operator inventories. Public links and redacted evidence are sufficient for the learning path.
- Before ending a session, update `docs/PROJECT_CONTINUATION.md` with completed work, the relevant commit/PR, actual validation, blockers, authority, and one concrete next action.

Use plain, precise prose and concrete examples. Define unfamiliar terms before relying on them. Avoid generic promotional language and em dashes as sentence separators.
