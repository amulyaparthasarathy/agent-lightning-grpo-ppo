# Phase 4A task: recovery-readiness-v1

`recovery-readiness-v1` is a small synthetic multi-tool task added for Phase 4A.
It exists to answer one question before any further training is paid for:

> Can this task produce reliable, semantically meaningful reward variation that
> would suit a later controlled learning study?

Phase 3 qualified both algorithms operationally on `asset-status-v1`, but the
baseline was already 64/64 and only one group in sixteen carried any
group-relative signal. This task asks for more: three tool calls that must be
joined, time arithmetic over a short inventory, and an answer that names its
evidence.

Nothing here has been run against a model. Offline scripted execution shows that
the mechanics work and that every case is solvable. It says nothing about whether
the base model will find the task easy, hard or impossible. That is what the
[development calibration](PHASE4A_CALIBRATION.md) is for, and it needs a separate
budget approval. What was actually executed is in
[offline verification](PHASE4A_VERIFICATION.md).

The calculator default and the qualified asset-status workflow are unchanged.

## The request

Each case names one asset and one explicit UTC evaluation time:

```text
As of 2026-08-22T17:35Z (UTC), does asset ast-02eb9990 meet its recovery point objective?
```

Three phrasings are used. The harness appends a fixed instruction that states the
rule in full, so nothing needed to answer is hidden:

- A backup run is usable only if its status is `succeeded`, its integrity is
  `passed`, and its `recovery_point_at` is no later than `as_of` and no more than
  the policy RPO before `as_of`.
- A recovery point exactly one RPO old still counts.
- The age is measured from `recovery_point_at`, not `completed_at`.
- The decision is `ready` if at least one run is usable, otherwise `not_ready`.

The task never reads the wall clock. The served chat template prints its own fixed
date (`Today Date: 11 Nov 2026`) in every prompt; the system message and the
instruction both say that the request's `as_of` is the only current time. The
development evaluation times fall on both sides of that header date and none is
within a day of it.

## Tools

The real stdio MCP server exposes exactly three read-only tools.

| Tool | Arguments | Returns |
|---|---|---|
| `get_asset` | `asset_id` | `record_id`, `asset_id`, `policy_id` |
| `get_policy` | `policy_id` | `record_id`, `policy_id`, `rpo` as `{"value", "unit"}` with unit `minutes`, `hours` or `days` |
| `list_backup_runs` | `asset_id`, `as_of` | `inventory_id`, `asset_id`, `as_of`, `complete: true`, and up to six `runs` |

Each run has `run_id`, `recovery_point_at`, `completed_at`, `status`
(`succeeded`, `failed`, `running`) and `integrity` (`passed`, `failed`, `pending`,
`not_applicable`). Timestamps are `YYYY-MM-DDTHH:MMZ`.

The inventory shows only what was known at the supplied `as_of`:

- A run that had not started is not listed.
- A run still in progress shows `running` with `completed_at: null`, even if the
  world records that it later succeeded.
- A finished run whose integrity check had not finished shows `pending`, even if
  the check later passed.

The inventory is complete and unpaginated for that asset and time, and
`inventory_id` is bound to both. Asking with a different `as_of` returns a
different inventory with a different ID.

No tool returns readiness, run eligibility, an expected answer, a reward, a case
family or a verifier explanation. Records are listed in a fixed hash order, not by
time. Identifiers are opaque hashes.

Invalid calls return a typed error observation,
`{"code": ..., "error_type": "recovery_policy_action_v1", "instruction": ...}`,
with one of `invalid_arguments`, `invalid_asset_id`, `unknown_asset`,
`invalid_policy_id`, `unknown_policy` or `invalid_as_of`. Extra arguments are
rejected, never stripped. The server child starts with a cleared environment, as
in the asset task.

## The answer

The whole final response must be one JSON object with exactly these keys:

```json
{"asset_id": "ast-02eb9990", "as_of": "2026-08-22T17:35Z", "decision": "ready",
 "asset_record_id": "rec-010485234c", "policy_record_id": "rec-130ab845b8",
 "inventory_id": "inv-56d62e00c7ff", "supporting_run_ids": ["run-ffb79681"]}
```

Duplicate keys, extra or missing keys, wrong types, code fences, surrounding
prose, a second object and `NaN` are all rejected. None of the keys is `name`, so
the served `llama3_json` tool parser returns the answer as text.

## The verifier

The verifier is independent of the tools and of the reference solver. It
regenerates the world, enumerates every raw record of the asset and decides
eligibility itself. It does not trust a tool's output, a model-selected run or the
row's label. The label is checked against the regenerated world before use.

Reward is 1 only if all of the following hold, and 0 for any other complete
answer:

1. The response satisfies the contract.
2. `asset_id` and `as_of` are the case's.
3. `decision` is correct.
4. `asset_record_id`, `policy_record_id` and `inventory_id` are the true ones,
   **and** each was observed in this episode: the asset record through
   `get_asset` for this asset, the policy record through `get_policy` for the
   policy that asset names, and the inventory through `list_backup_runs` for this
   asset and this exact time.
5. For `ready`: `supporting_run_ids` is a nonempty, duplicate-free subset of the
   qualifying runs, in any order. One nonqualifying, fabricated, other-asset or
   unobserved run makes it wrong.
6. For `not_ready`: `supporting_run_ids` is `[]`. The observed complete inventory
   is the evidence. Citing one failed run does not show that no usable run exists.

"Observed" means the call's result was joined to the sampled call in a later model
input. A record that exists in the world but was never fetched is not evidence, so
a correct guess earns zero. The harness also checks each successful observation
against what the environment returns for that call; a difference is an untrusted
execution, not a zero.

A wrong, malformed or ungrounded answer is a real completed outcome and earns a
confirmed zero. There is no partial credit, no shaping term, no LLM judge, no
answer repair and no hint.

### Diagnostics are not rewards

After the reward is computed, each episode records separate diagnostics: JSON
syntax, contract format, target, time, decision, each evidence field, which
required observations were made, and one failure class.

| Class | Meaning |
|---|---|
| `correct` | Reward 1 |
| `semantic` | Contract-valid, but wrong target, time or decision |
| `evidence` | Right decision, but wrong, unobserved or mis-scoped evidence |
| `format` | Not contract-valid. Decision and evidence are **not assessable** and are recorded as unknown, never as wrong |
| `action` | Ended at the policy-error limit with no answer |

Diagnostics never enter a prompt and never change a reward.

## Worked examples

All three are from the development panel. The answers shown are the reference
solver's, not model output.

**A newer failed run with an older usable point.** RPO 36 hours, evaluated at
2026-08-22T17:35Z.

```text
get_asset        -> {"asset_id":"ast-02eb9990","policy_id":"pol-03985359","record_id":"rec-010485234c"}
get_policy       -> {"policy_id":"pol-03985359","record_id":"rec-130ab845b8","rpo":{"unit":"hours","value":36}}
list_backup_runs -> inventory inv-56d62e00c7ff, five runs:
  run-eeb147b5  point 08-22T16:01  failed
  run-cb636a45  point 08-21T03:33  succeeded, passed   (38 h 02 min old: stale)
  run-5f593ac2  point 08-20T16:32  succeeded, integrity failed
  run-9f47c915  point 08-21T05:26  succeeded, passed   (36 h 09 min old: stale by nine minutes)
  run-ffb79681  point 08-22T12:20  succeeded, passed   (5 h 15 min old: usable)
answer: ready, supporting_run_ids ["run-ffb79681"]
```

**Completed recently, recovery point stale.** RPO 8 hours, evaluated at
2026-09-20T13:26Z. The newest run completed at 05:33, which is inside the RPO.
Its recovery point is 05:15, eight hours and eleven minutes old. Nothing
qualifies: `not_ready` with `supporting_run_ids: []`.

**The inclusive boundary.** RPO 36 hours, evaluated at 2026-10-22T03:39Z. The
newest recovery point is 2026-10-20T15:39Z, exactly 36 hours old. It counts:
`ready`.

## Data and splits

The recipe is `recovery-readiness-v1/2026-11-24/329`. Every value is derived by
SHA-256 from the recipe and a position, so generation needs no random number
generator and is byte-reproducible.

| Split | Cases | Use |
|---|---:|---|
| `train` | 128 | Reserved for a later study |
| `development` | 24 | Calibration, worked examples and all model-based diagnostics |
| `final` | 64 | Frozen. No model evaluation during calibration |

The world has 203 assets, 54 policies and 618 runs. Every split is balanced: half
`ready`, half `not_ready`.

| Family | Decision | Train | Dev | Final |
|---|---|---:|---:|---:|
| Recent usable recovery point | ready | 32 | 6 | 16 |
| Newer failed run, older usable point | ready | 32 | 6 | 16 |
| Stale successful backups | not ready | 22 | 4 | 11 |
| Recent incomplete backup | not ready | 16 | 3 | 8 |
| Recent integrity-failed backup | not ready | 16 | 3 | 8 |
| Empty history | not ready | 10 | 2 | 5 |

Variants inside the families cover the exact RPO boundary, a point one minute
past it, several valid witnesses, an integrity check still pending at the
evaluation time, and the completed-recently trap above.

**What stays together.** Cases are built in blocks of four that share one policy.
A policy, its assets and their runs belong to one split. Thirteen assets appear
twice, evaluated once while the newest point is usable and once after it has gone
stale; both cases of a pair are in the same split (8 pairs in train, 1 in
development, 4 in final). Paraphrases cannot cross splits because each case has
its own asset and time.

**The development panel is frozen.** Its 24 cases and their order are fixed, and
it is partitioned into three disjoint eight-case packets with four `ready` and
four `not_ready` each. Identifiers bind membership, order and content:

| Panel | Identifier |
|---|---|
| Development | `rr1-dev-ac7fd952343b0c7d` |
| Packet 1 | `rr1-dev-p1-98576e24a30d5457` |
| Packet 2 | `rr1-dev-p2-292a56c798a2b41f` |
| Packet 3 | `rr1-dev-p3-ee1d7ab06b3bf9ec` |
| Final evaluation | `rr1-final-a76397cabf99d31a` |

Environment SHA-256:
`0c384ed90027f7fa0fb0aeb097220e6e83876dfdd85094353f03c8dd035b1ca3`.

**No shortcut solves it.** Tests measure simple rules on every split: always one
decision, any run present, any succeeded-and-passed run, the newest run alone,
the first listed run, run count, ID parity and policy length all score below 80%.
The closest shortcut, measuring from `completed_at`, scores 84–92%: it is wrong
wherever a backup finished inside the RPO but captured an older point. Row order, case IDs, row fields and tool outputs carry no label, family or
split information.

**Files.** `train_main.parquet`, `dev_panel.parquet`, `dev_panel.jsonl`,
`dev_packet_1..3.parquet`, `final_eval.parquet`, `final_eval.jsonl`,
`world.json` and `data_manifest.json`. The final file is deliberately not named
`eval_main`, so no launcher default can reach it. Rows keep the shared transport
fields `question`, `result`, `case_id`, `split`; only `question` enters a prompt.

### What the final panel can and cannot show

The final panel tests whether behavior carries to **unseen assets, policies and
histories drawn from the same generator**. It is new and is not the Phase 3
64-case panel. It cannot show generalization to real backup systems, other
schemas, other tool designs or other instructions. Synthetic split separation is
not real-world generalization.

Dataset-consistency checks (generation, hashing, the reference solver, the
verifier) may read every split. Prompt and difficulty decisions use development
only. If a final-panel result ever influences a redesign, that panel becomes
development and regression evidence, and a new final holdout is required.

## The harness: recovery-episode-v1

`experiment/recovery_agent.py` reuses the production observed client, recording
workbench, stdio isolation, process ownership, cleanup and reward delivery.
`asset-episode-v2` and its harness are not modified.

| Setting | Value |
|---|---|
| Tool calls | Sequential, `parallel_tool_calls=False` |
| Tool rounds | Five: three required, plus one recovery and one spare |
| Terminal error limit | Two typed, attributable policy errors |
| Final generation | Tool-less reflection with one fixed reminder after the fifth round |
| Output cap | 1,024 tokens per generation, unchanged |

### Every terminal outcome

Defined from the pinned AutoGen 0.7.5 loop and the pinned vLLM 0.22.0 server.

| What happened | Outcome | Reward |
|---|---|---|
| Text answer with finish reason `stop`, on a tool-enabled turn or at reflection | `complete_final` | 0 or 1 by the verifier |
| The same, after a typed error and a later successful call | `complete_recovered` | 0 or 1; the error stays in the trajectory |
| An answer given before the required observations | `complete_final` | 0 (a policy choice) |
| Second typed policy error | `terminal_policy_failure` | One confirmed 0; nothing more is generated |
| Untyped tool error, lost or altered observation, more than one call in a generation, a tool name the framework rewrote into a valid tool, a call at reflection, truncation, empty answer, backend failure, timeout, cancellation, unconfirmed reward post | incomplete | None. The collection is rejected |

Choosing not to fetch evidence is the policy's own behavior and earns a zero.
Losing the result of a call that was issued is an execution failure and earns
nothing. The two are never confused.

A policy error is recognized only from typed evidence that matches the sampled
call: the harness re-derives the same code from the retained arguments. An error
string alone is not enough, because the MCP workbench turns every exception into
an error result. The nine recognized codes are the six server codes above,
`non_object_arguments`, `invalid_argument_json` and `unknown_tool`.

An empty or whitespace-only answer stays incomplete, as in `asset-episode-v2`.
The pinned client substitutes an empty string for absent content, so an empty
generation cannot be told apart from a missing one. This is the one place where a
possible policy output is treated as an untrusted execution.

### Four facts read from the pinned source

- **One call at most.** With `parallel_tool_calls=False`, vLLM keeps only the
  first parsed call. The harness therefore treats more than one call in a
  generation as an untrusted execution, not a policy error.
- **Truncation can be hidden.** Both vLLM and AutoGen report a parsed tool call as
  a tool-call finish even when generation stopped at the token limit. The harness
  rejects a tool-call generation whose reported completion tokens reach the cap.
- **Answers are text.** The `llama3_json` parser returns any JSON object without a
  `name` key as content, so a contract answer is never mistaken for a call.
- **Tool names are rewritten.** AutoGen replaces every character outside
  `[A-Za-z0-9_-]` in a sampled tool name before the harness sees it, so
  `get asset` would be executed as `get_asset`. The harness compares the name
  with what the server returned and never scores a name rewritten into a valid
  tool. A rewritten name that is still not a tool remains an ordinary
  `unknown_tool` error.

## The prompt the model actually receives

Source strings are not the served prompt. `experiment/recovery_prompt.py` renders
the prepared template for the exact request the harness sends, and
`scripts/task_local.py prompts` writes every prompt of an episode.

```bash
.venv-reader-cpu/bin/python scripts/task_local.py prompts --task recovery-readiness-v1 \
  --out outputs/phase4a/prompts.json
```

What the rendering shows:

- **Initial call.** The system block carries `Environment: ipython`, the fixed
  date header and the system message. The template then rewrites the first user
  turn: it prepends "please respond with a JSON for a function call", the three
  tool schemas, and only then the request and the task instruction.
- **Continuations.** Each earlier call is re-rendered as
  `{"name": ..., "parameters": ...}` and each observation as `{"output": "..."}`
  in an `ipython` turn. The function-call preamble is still present, so the model
  is asked for a call and for a final JSON object on the same turn.
- **Typed error and recovery.** The error payload appears as an ordinary
  observation and the next call follows it.
- **Reflection.** No tools are sent. The preamble and the `ipython` environment
  line disappear, and the fixed reminder is the last user turn.

The renderer was checked against the Phase 3 live traces, which record the
server's own prompt token count for each call. It reproduced all **1,343 of
1,343** counts exactly (192 in a06, 1,151 in a07). That check used the asset task
under the same model, template and serving stack. It does not show how the model
responds to these prompts.

### Token budget

Measured with the pinned tokenizer (`tokenizer.json`, SHA-256 `79e3e522…`),
which is a gated model file and is not committed.

| Prompt | Tokens |
|---|---:|
| Initial call, three schemas | 984–988 |
| Reference path, last call | up to 1,518 |
| Five rounds on the development panel, largest | 1,900 |
| Worst bounded case: five rounds of the largest inventory | 2,739 |

The worst case assumes each sampled call is about the size its schema requires.
A model that writes very long arguments can exceed any fixed bound; the adapter
then rejects the collection rather than cutting anything.

In the Phase 3 live collections every model call became its own training
fragment. Each call's whole prompt must therefore fit the trajectory prompt
limit. The shared limit is 2,048 tokens, which holds the reference path but not
the worst bounded case. A policy that repeats the inventory lookup, as the greedy
Phase 3 policy repeated its lookup, could exceed it and reject a collection.

Payloads were kept compact first: minute timestamps, no indentation, no derived
fields. That was not enough, so this task binds its own limits in its task
identity:

| Limit | Shared | This task |
|---|---:|---:|
| Trajectory prompt length | 2,048 | **3,072** |
| Trajectory response length | 2,048 | 2,048 |
| Model context | 4,096 | 4,096 |
| Tokens per generation | 1,024 | 1,024 |

3,072 is the model context minus the generation cap, the largest prompt the server
accepts with a full generation budget. Calculator and asset settings are
unchanged, and a test pins their launch configurations to `main`.

Resource implications: collection-only calibration makes no forward or backward
pass, so the larger limit only widens padded index tensors there. A later
learning study would train on rows up to about 5,100 positions instead of 4,096.
Its memory and time on one H100 are **unverified** and belong to that study's own
qualification.

Truncation is never hidden. A longer generation is not cut or repaired, and the
production adapter rejects any collection in which a rollout's fragment tokens
differ from its sampled tokens. The remote CPU preflight repeats this measurement
with the tokenizer and template actually in the image and fails if a bound is
exceeded.

## Run the local checks

All of these are local and free. They use the locked CPU harness group.

```bash
UV_PROJECT_ENVIRONMENT=.venv-reader-cpu uv sync --locked --only-group reader-harness
bash scripts/test_task_local.sh
.venv-reader-cpu/bin/python scripts/task_local.py generate --task recovery-readiness-v1 --data outputs/phase4a/data
.venv-reader-cpu/bin/python scripts/task_local.py verify   --task recovery-readiness-v1 --data outputs/phase4a/data
.venv-reader-cpu/bin/python scripts/task_local.py probe    --task recovery-readiness-v1 --out outputs/phase4a/probe.json
.venv-reader-cpu/bin/python scripts/task_local.py scripted --task recovery-readiness-v1 --out outputs/phase4a/scripted.json
.venv-reader-cpu/bin/python scripts/task_local.py prompts  --task recovery-readiness-v1 --out outputs/phase4a/prompts.json
```

Every command refuses to overwrite its output; choose a fresh path to repeat one.
`probe` runs the real server over stdio, checks the three schemas and every typed
error, and runs the reference solver on the development panel. `scripted` runs
one episode through the production harness with scripted model responses; its
receipt and traces say `scripted-inference`, and live evidence validation rejects
them. Add `--tokenizer PATH` to `prompts` to measure tokens; the file is verified
against the committed model lock.
