# Phase 4A offline verification

Phase 4A adds the [`recovery-readiness-v1`](PHASE4A_RECOVERY_TASK.md) task and a
[development calibration](PHASE4A_CALIBRATION.md) workflow. This report records
what was executed, on what, and what was not.

**Everything here ran locally on CPU.** No model was called, no GPU was used, no
image was built, no cloud resource was created, and no remote preparation,
preflight or launch was run. No Phase 3 branch, evidence or resource was changed.
The [paid calibration proposal](PHASE4A_CALIBRATION_PROPOSAL.md) is unapproved.

## Source state

| Item | Value |
|---|---|
| Repository | `amulyaparthasarathy/agent-lightning-grpo-ppo` |
| Base | `main` at `daed9a51cc14892c3521cbb0b449fb2c7bd0db6f`, the merge of PR #4 |
| Branch | `phase4a-recovery-readiness`, in its own worktree |
| Tested implementation commit | `664a5a45e6648b298449c7a2a70c205ee683de39` |
| Later commits | Documentation and evidence only; no file under `experiment/`, `scripts/` or `tests/` |

The merge commit was verified before work started: its tree is identical to the
final Phase 3 PR head `42390a3`, and its parents are `5c069f7` and `42390a3`.
There is no dependency on an unmerged branch.

Source-bound plans were generated and validated at the tested commit from a clean
tree. Their identities are in
[`candidate-identities.json`](../evidence/phase4a-offline-20261124/candidate-identities.json).
A plan names the commit it was generated from, so plans must be regenerated after
the documentation commits; the delivery report gives the hashes at the branch
head.

## Gates

The three documented gates were run before any change, on unmodified `main`, and
again at the tested commit. No test was skipped in any run.

```bash
bash scripts/test_task_local.sh
bash scripts/test_reader_offline.sh
PYTHONDONTWRITEBYTECODE=1 \
AGL_PINNED_SOURCES=/workspace/projects/repos/agent-lightning-lab/.local/ppo-review \
AGL_VLLM_WHEEL=/workspace/projects/repos/agent-lightning-exploratory/.local/failure-r02/vllm-0.22.0-cp38-abi3-manylinux_2_28_x86_64.whl \
AGL_TRANSFORMERS_WHEELS=/workspace/projects/repos/agent-lightning-exploratory/.local/failure-a01 \
MATCHED_TEST_PYTHON=/workspace/projects/repos/agent-lightning-lab/.local/matched-test-venv/bin/python \
bash scripts/test_matched_training.sh
```

| Gate | Environment | `main` | Tested commit |
|---|---|---:|---:|
| Task gate | Locked CPU harness group: real MCP, real subprocesses | 26 | **79** |
| Reader gate: PPO-MC | Project environment, offline | 39 | 39 |
| Reader gate: matched study | Project environment, offline | 15 | 15 |
| Reader gate: reader suites | Project environment with PyArrow, offline | 86 | **126** |
| Matched gate | Cached CPU PyTorch and hash-verified pinned sources | 63 | **68** |
| Matched gate: pinned-config subprocess | Real Hydra over the pinned configuration builder | 6 | **7** |

All passed. The absolute paths name this workbench's reused caches, as in Phase 3;
a fresh workbench uses `scripts/test_matched_training.sh --prepare`. No CUDA stack
was installed anywhere.

The locked harness group gained `jinja2==3.1.6` and `tokenizers==0.23.3`, the
versions recorded in the training image, for prompt inspection. The lock diff is
purely additive. A new `.venv-reader-cpu` was created in the worktree; the main
checkout's environments were not touched.

### New suites

| Suite | Tests | Gate | What it executes |
|---|---:|---|---|
| `test_recovery_task` | 18 | Task, reader | Generation, manifests, splits, shortcuts, the verifier, the solver, mutations |
| `test_recovery_harness` | 13 | Task | The production harness over real stdio MCP with scripted model responses; prompts; the preflight task check |
| `test_recovery_calibration` | 22 | Task, reader | Plans, closed dispatch, dataset routing, completion rules, the report, mocked provider dispatch, package admission |
| `test_matched_recovery_calibration` | 5 | Matched | The real pinned collection adapter under production instrumentation, both routes |
| `pinned_config_checks` (one new test) | 1 | Matched | Eight calibration configurations through real Hydra and the pinned builder |

### What the tests cover

**Data and verifier**

- Byte-identical regeneration; manifest validation; refusal to overwrite.
- Split isolation of assets, policies and runs; sibling pairs kept together;
  disjoint packets that cover the panel and all six families.
- No label, family or split information in rows, case IDs, questions or any tool
  output; twelve shortcut rules measured on every split.
- Rehashed corruption: duplicated, relabelled, reordered, leaked and re-split rows
  are rejected even when file hashes are updated.
- The reference solver uses only public tools and agrees with the verifier on all
  216 cases.
- The inclusive RPO boundary, a just-stale point, a future point, completion
  after `as_of`, and integrity known only after `as_of`, both on explicit records
  and through the generated world.
- A newer failed run with an older usable run; every nonempty witness subset in
  every order; a complete empty history; a `not_ready` proof without the
  inventory.
- Wrong, fabricated, duplicate, unobserved, other-asset and other-time evidence,
  each with a correct decision and a zero reward.
- Strict JSON: duplicate keys, extra and missing fields, wrong types, fences,
  prose, a second object, `NaN`.
- World mutations: five ways of spoiling a usable run each flip the solver and the
  verifier together.

**Production harness, real subprocesses and real MCP, scripted inference**

- Three sequential calls and an answer; continuation prefixes; five rounds and the
  tool-less reflection with its reminder.
- Each of the nine typed errors, first and mid-episode, followed by recovery.
- Non-object arguments of six shapes, typed without contacting the server.
- The two-error terminal limit in three orders, including error, recovery, error.
- Early and ungrounded answers: five ways of skipping evidence, each a complete
  zero with a correct decision.
- Seven completed wrong or malformed answers.
- Thirteen incomplete generations: truncated answer, truncated action at the
  token cap, truncated reflection, content filter, empty answer, two calls in one
  generation, backend failure at three points, timeout, cancellation at two
  points. Each is asserted to fail for its own stated reason.
- Lost observations, an untyped backend tool error, altered observations and a
  failed reward post.
- Exact call-to-observation joins, identity checks against the stage, and cleanup
  of every owned MCP process with only two environment variables in the child.
- No label, credential, case ID, family or diagnostic in any model input.

**Reader and calibration**

- Development membership, order and split in both modes; plan tampering in
  sixteen ways.
- The final-holdout sentinel: with the final and training files unreadable, all
  eight stages are verified, routed, materialized and validated, and the record of
  opened files contains none of them.
- Both algorithm routes through the real adapter: 32 rollouts, 8 groups, token
  conservation, observation and reminder tokens masked, merged and fragmented.
- Constant groups retained; missing rewards, duplicate rollout IDs, wrong packets,
  altered counts, token loss, cross-source and cross-episode evidence rejected.
- No advantage, backward, optimizer or checkpoint event; one refresh at step 0.
- Scripted evidence rejected as live, in the stage row, the report and the durable
  collection.
- The nine new runtime files admitted; a changed, missing or extra packaged file
  rejected; test-only code excluded from the image.
- Calculator and asset plans and their fourteen launch configurations pinned to
  `main` by digest.

## Real, scripted, mocked and unexecuted

| Evidence | Class |
|---|---|
| Data generation, verification, hashing | Real, local |
| Three-tool MCP server over stdio, in an isolated owned subprocess | Real, local |
| Harness: AutoGen request building, response parsing, tool loop, tracing, reward post to a loopback receiver | Real, local |
| Model responses | **Scripted** at the OpenAI SDK call. No model |
| Reference solver on the development panel through real MCP | Real tools; a solver, not a model |
| Collection adapter and its instrumentation | Real pinned code on CPU; **synthetic** token sequences |
| Trainer-level events (return, refresh, W&B finish) | **Simulated** in tests; no trainer process ran |
| Configuration composition | Real Hydra and the pinned builder; model assets faked at the declared seam |
| Prompt rendering and token counts | Real template and real pinned tokenizer, offline |
| Renderer check against live calls | Replay of **recorded** Phase 3 calls; no new inference |
| Provider calls (`setup`, `prepare`, `preflight`, `launch`) | **Mocked** boundary only |
| Remote image, remote MCP version, GPU, vLLM serving, W&B | **Not executed** |
| How the model behaves on this task | **Unknown** |

The scripted calibration demonstration ran all 240 episodes through the
production harness. Its
[report](../evidence/phase4a-offline-20261124/scripted-calibration-summary.md) is
labelled as not a live policy measurement. Its numbers come from an arbitrary
fixed script and describe no policy.

## Commands exercised

Run at the tested commit from a clean tree, with fresh output paths under the
ignored `outputs/phase4a/`.

| Command | Result |
|---|---|
| `task_local.py generate` and `verify` | Environment `0c384ed9…`; 8 files; verified |
| `task_local.py probe` | Passed: 13 schema and typed-error checks, 24 of 24 development cases solved and verified, cleanup complete |
| `task_local.py scripted` | Passed: three joined chains, reward delivered; `scripted-inference` |
| `task_local.py prompts` | 15 rendered prompts over three paths |
| `task_local.py prompts --tokenizer` | Token counts and budget; all three limit checks hold |
| `task_local.py calibration-demo` | 240 of 240 scripted episodes admitted |
| `task_local.py report` | Report rebuilt; 240 episode rows rechecked against traces |
| `reader_training.py plan` and `validate` | Fixture and candidate plans valid; source clean; not training-admissible (no assets) |
| Asset and calculator probes, asset scripted demonstration | Still pass |
| A repeated `generate` | Refused, as designed |

## Identities

| Item | Value |
|---|---|
| Task | `recovery-readiness-v1` |
| Recipe | `recovery-readiness-v1/2026-11-24/329` |
| Environment SHA-256 | `0c384ed90027f7fa0fb0aeb097220e6e83876dfdd85094353f03c8dd035b1ca3` |
| World file SHA-256 | `230ddc529061f4fd3a6a57a646b566730c7485940a56ff8bcf7fab82ac15b5d8` |
| Splits | 128 train, 24 development, 64 final |
| Development panel | `rr1-dev-ac7fd952343b0c7d` |
| Packets | `rr1-dev-p1-98576e24a30d5457`, `rr1-dev-p2-292a56c798a2b41f`, `rr1-dev-p3-ee1d7ab06b3bf9ec` |
| Final panel | `rr1-final-a76397cabf99d31a` |
| Episode policy | `recovery-episode-v1` |
| Context limits | `recovery-context-v1` |
| Calibration schema and screen | `recovery-calibration-v1`, `recovery-screen-v1` |
| Plan schema | `reader-task-v2`, phase `development-calibration` |
| Template SHA-256 (prepared) | `a7186748…` in [`prompt-budget.json`](../evidence/phase4a-offline-20261124/prompt-budget.json) |
| Model, Agent Lightning, VERL | Unchanged pins; see the [proposal](PHASE4A_CALIBRATION_PROPOSAL.md#what-would-run) |

**Holdout policy.** The final panel was read only by dataset-consistency checks:
generation, hashing, the reference solver and the verifier. No prompt or
difficulty decision used it, and no calibration path can open it.

## What is preserved

**Numerical runtime.** These files are byte-identical to `main`: `ppo_math.py`,
`ppo_runtime.py`, `ppo_contract.py`, `matched_contract.py`, `launch.py`,
`observed_runtime.py`, `owned_processes.py`, `request_transport.py`,
`matched_supervisor.py`, `matched_remote.py`, both source locks, the chat
template, `reader_app.py` and `scripts/reader_training.py`.

`matched_runtime.py` changed in two places only. Mode selection now asks
`reader_modes` whether a stage is a collection, which gives the same answer as
before for every existing plan. One evidence event, `calibration_collection_tokens`,
is emitted for calibration stages after the batch is admitted; it reads the batch
and changes nothing. Advantage construction and normalization, rollout and
fragment weighting, the policy and value losses, clipping, optimizer semantics,
critic initialization, masks, token conservation, policy refresh and checkpoint
handling are untouched. The existing 63 matched tests still pass.

**Existing tasks.** `asset_task.py`, `asset_agent.py`, `asset_server.py`,
`asset_probe.py` and `observed_calc_agent.py` are byte-identical to `main`. For a
fixed source identity, the calculator, asset and asset collection-only plans and
all fourteen launch configurations and command sets have the same digests as on
`main`. The asset collection-only evidence event keeps its exact fields.

**Qualification rules.** The two-update expectations for qualification plans are
unchanged and still explicit. Calibration has its own rules and cannot satisfy
them.

**History.** No historical plan, identity or evidence file was edited. New plans
carry new source hashes, as they must.

## Problems found and corrected

Found by executing, not by reading.

1. **Truncation hidden by the serving stack.** Reading the pinned vLLM and AutoGen
   sources showed both report a parsed tool call as a tool-call finish even when
   generation hit the token cap. The harness now rejects a tool-call generation
   whose reported completion tokens reach the cap.
2. **The shared prompt limit did not hold the bounded worst case.** Measured at
   2,739 tokens against 2,048. The Phase 3 traces showed every call becomes its own
   fragment, so this was a real rejection risk. The task now binds a 3,072-token
   limit in its identity.
3. **Panel identifiers did not bind content.** Case IDs are derived from position,
   so the first identifiers survived a content change. They now hash each row's ID,
   request and label.
4. **The development panel lacked the completed-recently trap**, and same-asset
   sibling pairs arose only by chance. Pairs are now built explicitly and the panel
   includes the trap.
5. **`NaN` passed the JSON syntax diagnostic.** The reward was already zero; the
   diagnostic now uses the same strict parser.
6. **A scripted report named a next step** as if it were a result. Non-live
   evidence now yields no decision.
7. **Tests that could pass for the wrong reason.** Several incomplete-episode
   scripts ended exactly where the failure was injected, so an exhausted script
   would also have failed them. Each now continues past the fault and asserts the
   specific reason and request count.
8. **Three wrong test expectations**: a hand-computed Wilson interval, a fixture
   that overwrote model configuration, and a manifest check applied to mutations
   it is not meant to catch.

The reader gate fails while the new runtime files are untracked, because source
admission requires tracked files. That is the intended behavior.

### Found by an independent review

A separate read-only review of the implementation diff found no wrong reward from
the verifier, no leaking tool, no final-panel read and no change to existing
plans. It did find the following, each confirmed by running it and each now fixed
with a test.

9. **A rewritten tool name could be executed and rewarded.** AutoGen replaces
   invalid characters in a sampled tool name, so `get asset` ran as `get_asset`
   and could earn reward 1. The harness now compares against the name the server
   returned and treats a name rewritten into a valid tool as an untrusted
   execution.
10. **Untransportable arguments hung the MCP client.** An argument with a lone
    surrogate cannot cross the stdio transport and waited for the 120-second read
    timeout. It now gets the server's own typed validation error at once.
11. **Evidence class was pooled across routes** and depended on job order: a live
    route beside a scripted one could be reported as not live, or the reverse at
    the top level. It is now decided per route.
12. **A near-saturated route was labelled `interface_dominated`** when its only
    zero was a format failure. Saturation is now checked first.
13. **Trace review trusted recorded classifications.** A trace whose error
    observations had been altered was still admitted with its original codes, and
    a successful call did not have to be joined to a model input. Both are now
    re-derived and required.
14. **The fully merged token bound subtracted mismatched prompts.** The reflection
    prompt has no tool preamble and never merges with earlier turns. The bound now
    ends at the last tool-enabled turn (1,924 tokens against a 2,048 limit).
15. **No comparison of trace tokens with adapter tokens.** Server-reported
    completion tokens now must equal the adapter's sampled tokens when reported.
    They were equal for all 192 Phase 3 training rollouts.

Two review points were kept as they are and are stated as limitations below: an
empty answer stays incomplete, and the asset harness has the same tool-name
behavior.

## The eleven Phase 3 lessons, applied

| # | Lesson | Where it is applied |
|---:|---|---|
| 1 | Effective input | `recovery_prompt.py` renders the initial, continuation, error-recovery and reflection prompts from the exact wire requests. The renderer reproduced 1,343 of 1,343 recorded server token counts. Token budgets are tested against the prompt, model and trajectory limits, and re-measured in remote preflight |
| 2 | Policy versus infrastructure | Nine typed codes, each re-derived from the retained sampled call. Untyped errors, lost or altered observations, truncation, cancellation and backend failures are tested as incomplete, each for its own reason |
| 3 | Keep wrong and recovered trajectories | Confirmed zeros for wrong, malformed and ungrounded answers; recovery scored on the final answer; a missing reward stays missing; no group is filtered, filled or resampled |
| 4 | Format versus correctness | Separate syntax, contract, target, time, decision and evidence diagnostics; malformed answers are not assessable rather than wrong |
| 5 | The sampled path | Sampled stages are the production training collection with its existing stop, exercised on CPU through the real adapter for both routes |
| 6 | Reward distributions | A reward vector per question, constant-zero and constant-one groups, and mixed groups split by whether a failure is semantic or only format/action |
| 7 | Tokens and masks | Mask tokens equal sampled tokens per rollout; observation and reminder tokens are context; the loss-target masks are unchanged |
| 8 | Optimizer accounting | Completion counts audit events; any optimizer, gradient, advantage or checkpoint event fails the stage. W&B history rows are not used |
| 9 | Claims | The report keeps local operation, harness effects, useful signal and learning improvement apart, and assesses signal only on live evidence |
| 10 | Unknown submissions | The [reconcile-before-retry procedure](PHASE4A_CALIBRATION.md#an-unknown-submission-reconcile-before-any-retry): reserve first, never resubmit, reconcile read-only, stop owned calls, fresh attempt |
| 11 | Ownership handoff | The [handoff checklist](PHASE4A_CALIBRATION.md#handing-the-work-over) and the state below |

No earlier attempt is reinterpreted under a newer episode policy. Traces without a
recognized policy keep their historical treatment.

## Limitations and unverified assumptions

- **Model behavior is unknown.** Whether the base model produces contract-valid
  answers, how often it repeats calls, and whether rewards vary are all unmeasured.
- **Conflicting instructions on tool-enabled turns.** The template asks for a
  function call while the task asks for a final JSON object. Phase 3 showed the
  greedy policy keeps calling while tools are advertised. The five-round limit and
  the reflection reminder allow for that, but the effect here is not known.
- **Remote MCP version.** Local tests pin MCP 1.26.0; Phase 3 recorded a different
  version in the image. The typed error payloads are checked by the remote
  preflight probe, which has not run.
- **Stage times and cost** are estimates from asset-task timings.
- **Trainer-level events** were simulated in tests. The wrapper that emits them is
  unchanged and was qualified live in Phase 3 for the collection-only stop, but
  not with this task.
- **The larger prompt limit** has no measured training cost, because calibration
  does not train.
- **Renderer validation** used the asset task. It confirms the rendering and
  tokenization, not the recovery prompts' effect.
- **Generalization.** The final panel tests unseen assets from the same
  generator. It cannot establish real-world generalization.
- **An empty answer is incomplete, not a zero.** The pinned client cannot
  distinguish an empty generation from missing content. One such sample at
  temperature 1 would reject its collection and end that route's later stages.
- **The qualified asset harness is unchanged**, including the tool-name rewriting
  that the new harness now guards against. Changing it would change a qualified
  identity; it is noted for any future asset work.
- **The bounded worst-case prompt assumes schema-sized calls.** Very long sampled
  arguments can exceed it; the adapter then rejects the collection.
- **Truncation of a tool call** is detected from the server's finish reason and
  from reported completion tokens reaching the cap. If a server reported neither,
  it would not be detected.

## Handoff state

| Item | State |
|---|---|
| Checkout | `/workspace/projects/repos/agent-lightning-lab-public-phase4a`, a linked worktree; branch `phase4a-recovery-readiness` |
| Main checkout | `/workspace/projects/repos/agent-lightning-lab-public`, still on `phase3-reader-task-adaptation`, unmodified |
| Local environments | `.venv` and `.venv-reader-cpu` in the worktree; the matched CPU environment and pinned sources are reused read-only from the earlier lab checkout |
| Active local processes | None |
| Plans | Ignored `outputs/phase4a/`: a fixture plan and a candidate preparation plan, bound to the tested commit. Regenerate before use |
| Submission receipts | None. Nothing was submitted |
| Cloud resources | None created. Phase 3 resources, ledger and retention dates are untouched and remain the owner's decisions |
| Spending ledger | None for Phase 4A yet. One must be created and reserved before any dispatch |
| Completed | Offline implementation, tests, documentation, proposal |
| Waiting on the owner | Review of the draft PR; approval or change of the calibration ceiling; the Phase 3 retention decisions, unchanged |
