# Engineering checklist from the Phase 3 attempts

Seven attempts (a01–a07) were needed to qualify the asset task. Four failed, one
was stranded before it ran, one was a development check and one qualified. Each
item below is a check that would have saved an attempt. Run through the list
before adapting a task, changing a prompt or harness, or paying for a GPU stage.

The full story is in the [failed campaign report](PHASE3_QUALIFICATION_2026-11-22.md)
and the [qualification report](PHASE3_QUALIFICATION_2026-11-24.md). The
[adaptation guide](TASK_ADAPTATION.md#episode-outcomes-asset-episode-v2) explains
the mechanisms.

## Before changing a prompt or harness

1. **Read the prompt the model actually receives.** Render the served chat
   template with the real messages and tool list, for every model call in the
   episode. Source prompt strings are not enough.
   *What happened:* the template prepends "respond with a JSON for a function
   call" whenever tools are advertised and drops it at the tool-less reflection
   call. The task's own JSON instruction sat beside one and disappeared from view
   at the other. Three prompt edits (a02–a04) were made without seeing this.
   *Where:* the recorded `model_input` of each call in a private trace, and
   `experiment/templates/tool_chat_template_llama3.1_json.jinja`.

2. **Separate policy mistakes from infrastructure failures.** A wrong action the
   model sampled is training data. A crashed server, a timeout or a truncated
   generation is an untrusted execution. Decide which is which from typed,
   explicit evidence, and fail closed on anything unrecognized.
   *What happened:* a04 treated five extra-argument tool calls as incomplete
   executions and rejected two whole 32-rollout collections.
   *Where:* `EPISODE_POLICY` in `experiment/asset_task.py`; the classifier in
   `experiment/asset_agent.py`.

3. **Keep legitimate failed and recovered trajectories.** A complete episode with
   a wrong answer earns a confirmed zero. An episode that erred and recovered is
   scored on its final answer, with the error retained. Never filter, fill or
   resample to make a group pass. A missing reward stays missing.
   *What happened:* four of the five rejected a04 rollouts had recovered. In a03
   the hosted `val/reward=0` hid 64 episodes that had no reward at all.

4. **Score format and correctness separately.** Record whether the final response
   parsed, whether it had the contract's shape, whether it named the right target
   and whether its values were right. Do this after the reward, as diagnostics
   only.
   *What happened:* a04's 0/64 was read as a task failure. Every final answer was
   prose; none was a wrong fact inside a well-formed object.

## Before paying for optimization

5. **Test the sampled path, not only the greedy one.** Baselines run at
   temperature 0 and training samples at temperature 1. Run a collection-only
   development plan on training cases first and review every rollout.
   *What happened:* a04 passed its baseline integration gate with 64/64 complete
   episodes and then lost both first training collections to actions the greedy
   policy never takes. a06 was the check that should have come first.
   *Where:* `"purpose": "collection-only"` in the profile; see
   [the guide](TASK_ADAPTATION.md#greedy-evaluation-does-not-test-sampled-training).

6. **Look at the reward distribution and group informativeness.** Count rewards
   per group before reading any loss. A group of four equal rewards carries no
   group-relative signal.
   *What happened:* the qualifying run had 1 informative group in 16. GRPO's
   second update had an exactly zero gradient and still counted as an update.
   With a 64/64 baseline there is no headroom to show learning.

7. **Leave sampled tokens and observation masking alone.** Fragment token totals
   must equal the originally generated totals for every rollout. Tool
   observations and harness reminders are model inputs, never loss targets. Do
   not rewrite, strip or repair a sampled action to make it valid.
   *What happened:* the non-object-argument fix returns a typed error for the
   call as sampled. It does not coerce the arguments.
   *Where:* `instrument_adapter` in `experiment/matched_runtime.py`, which rejects
   a collection when a rollout's fragment tokens differ from its sampled tokens.

## Reading results

8. **Count optimizer events, not W&B history rows.** A hosted step is a logging
   row. Collection artifacts create one without any update.
   *What happened:* a04's training runs and both a06 development runs show hosted
   step 1 with zero optimizer calls.
   *Where:* `optimizer_step` and `gradient_step` events in each stage's `audit/`
   files; `observed_optimizer_events` in the reviewed stage records.

9. **Operational completion is not learning.** State separately what completed,
   what the harness correction changed, whether any informative signal existed,
   and whether accuracy moved. Claim only the ones the evidence supports.

## Operating paid work

10. **Reconcile an unknown submission before any retry.** A receipt without a
    call ID is not proof that nothing ran. Query the provider read-only (apps,
    tasks, registry rows, volume contents, billing), place a stop with
    `stop-owned`, record the finding, and use a fresh attempt.
    *What happened:* a05's `prepare` client was killed while the image resolved.
    Keep the client alive until `prepare`, `preflight` and `launch` return.
    Note that the provider-side stop marker is a Modal Dict entry, which expires
    after seven days without a read or write. The lasting record is the local
    receipt and the committed reconciliation.

11. **Hand over ownership explicitly when switching operators or coding
    assistants.** Assistants do not share conversation history or private memory.
    The outgoing session should leave no client process mid-submission, push or
    clearly mark every commit, record every reservation in the ledger before
    dispatch, and write down what is in flight. The incoming session should
    verify the remote branch head rather than trust the brief, compare the ledger
    with provider state read-only, and review inherited commits against raw
    evidence before building on them.
    *What happened:* the continuation found the branch one commit ahead of the
    reviewed head, a US$175 reservation the ledger did not show, and the a05
    receipt above. None was in the handoff instructions.

## Small operating facts

- Every CLI command regenerates the plan from the working tree and compares.
  After a change under `experiment/`, older plans fail validation even for
  `status`, `collect` and `stop-owned`. Operate on an old plan from a temporary
  `git worktree` at its source commit. Do not commit between planning an attempt
  and its launch acknowledgement.
- Collecting a six-stage attempt downloads about 1,100 small files and takes
  about ten minutes. Run it as one detached process into a fresh directory.
- The collector keeps `.json`, `.jsonl`, `.log` and `.txt` files and skips
  `temporary/`. Checkpoint tensors and a few small generated files stay on the
  owned volume, so a volume is not redundant until those are accounted for.
- Reserve before dispatch, reduce a reservation only after every owned call is
  terminal, and never release unresolved exposure to make another attempt fit.
