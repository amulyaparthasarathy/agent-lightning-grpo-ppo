# r01 rollout configuration failure and r02 recovery

> Historical failure/recovery record. The later [r03 closeout](PHASE2_CLOSEOUT_2026-11-15.md) records completed execution at `5fb6adfd48ee245509f7dff3ab1d849fcf707d86`; prepared/not-executed statements below describe this earlier repair. Original failed-attempt identities and evidence remain unchanged.


Source investigated: `608a3e505767eb6bd32915920baa8820abecbfce`, draft PR #26.
Failed identity: `agl-explore-20-v1-r01/exploratory-qualification/a01`.
No GPU jobs, image builds, retries, remote mutations, credential changes or merges
were performed during this correction.

## Demonstrated failure

Both downloaded trainer logs fail in installed VERL
`workers/engine_workers.py:592`, calling
`omega_conf_to_dataclass(self.config.rollout)`. The converter invokes Hydra
`instantiate(config, _convert_="partial")` in `utils/config.py:50`:

```text
TypeError: RolloutConfig.__init__() got an unexpected keyword argument 'seed'
full_key: actor_rollout_ref.rollout
```

The resolved configs contain the unsupported `actor_rollout_ref.rollout.seed`
introduced by `matched_launch.configuration()`. The failed source guard already
passed, and both runs reached actor model/FSDP initialization. This is a config
schema error, not a CUDA, model-weight, Transformers, Modal or admission failure.
Both complete downloaded resolved configs reproduce the exception on CPU, then
instantiate successfully after the seed-field relocation. Raw downloads remain
under `.local/failure-r01/phase2/`; their original paths, timestamps and SHA256s
are in [the inventory](../evidence/phase2-exploratory/failure-r01/volume-inventory.json).
Exception excerpts retain original trainer-log line numbers.

## Minimal correction and meaningful seeding

The only training implementation change is:

```text
actor_rollout_ref.rollout.seed
    -> actor_rollout_ref.rollout.engine_kwargs.vllm.seed
```

Both keys receive the **same existing `seed_streams.actor_sampling` integer**.
No seed stream, paired run seed, optimizer setting, algorithm or dataset changes.

Evidence from the exact VERL 0.8.0 wheel (SHA256
`4e32830aae80771ee3b5f0d26a2b794d793f3bcece67aff07c2a7c62e948714f`):

- `workers/config/rollout.py:161`: `RolloutConfig` has no `seed` constructor
  parameter. Its supported `engine_kwargs` field is a dictionary.
- `workers/rollout/vllm_rollout/vllm_async_server.py:215`: `launch_server()` reads
  `engine_kwargs[self._get_engine_kwargs_key()]`; the native method returns `vllm`.
  The default engine seed at line 271 is followed by `**engine_kwargs` at line 277.
  Thus the configured seed replaces the default, and `build_cli_args_from_config`
  at line 360 emits `--seed <actor_sampling>` for the vLLM engine. Each study job
  has one H100/one replica, so the original replica-rank offset is zero.
- Tests execute this **unmodified launch_server body** and the native CLI
  serializer for all 24 stage configurations, intercepting the CLI boundary
  before vLLM parser/server/GPU construction. All 24 emit exactly the declared,
  non-default sampling seed once. The engine seed is not merely stored in metadata.
- `matched_runtime.seed_process()` sets Python, NumPy, Torch CPU and CUDA RNGs;
  it does **not** use `actor_sampling`. It runs in the trainer and through
  `matched_runtime.register_in_worker`, the retained Ray worker setup hook.
  Removing the invalid key without replacing it would leave the vLLM engine at
  its default seed, rather than preserve the experiment's seed control.
- Question-order seeds remain shared within each paired run; algorithm-specific
  actor-sampling streams remain distinct, exactly as in the original plan.
  Actor/FSDP optimizer, critic initialization and row-batching streams are retained.
  This controls seeds, not bitwise determinism of asynchronous scheduling/GPU kernels.

## CPU schema validation

The CPU checks compare **all 371 extracted VERL Python/YAML files** with the
hash-verified released wheel, verify the pinned Agent Lightning config and Calc-X
entrypoint hashes, and execute the original `build_config()` with real Hydra
1.3.2 composition, OmegaConf 2.3.0 merging/interpolation and the full generated
overrides (including Ray environment settings).

All **24 configurations** are covered: two qualification jobs plus six report
jobs, each with baseline/training/checkpoint-evaluation stages. Checks construct
native actor, critic (even when disabled for GRPO), rollout and nested sampling,
FSDP engine, optimizer, profiler and checkpoint configs; model configs through
both the implicit Hydra and explicit dataclass conversion paths; reference
configs after the worker's field renames; and actor/critic TrainingWorkerConfig.

Schema definitions, constructors and `__post_init__` bodies are unmodified.
Only package initializers that pull in distributed/accelerator services and
model/tokenizer/config **asset I/O** are substituted. This tests schemas on CPU;
it is not a model-loading, vLLM parser, Ray cluster or GPU execution test.
Uninstantiated free-form application settings are not asserted to be dataclasses.

Five internal regression checks cover the complete matrix, original exception,
native seed CLI handoff, paired/Ray seed preservation, and rejection of injected
unknown actor/critic/rollout/model and nested fields. The existing client seed
assertion now checks the supported key and absence of the old one. The CPU test
preparation script pins Hydra 1.3.2, matching the retained installed-package record.

Validation: **284 client tests, 274 passed and 10 expected CPU-environment skips;
59 pinned suite tests passed with no skips**, including the five internal schema
checks. Both actual failed resolved configs separately reproduced the TypeError
and passed all tested constructors after relocation. `git diff --check` passed.
Logs are in [failure-r01](../evidence/phase2-exploratory/failure-r01/).

## Owned calls and accumulated costs

Read-only provider call graphs show all six calls terminal:

| Attempt / role | FunctionCall | Provider state |
|---|---|---|
| original supervisor | `fc-01M7EMCA0TS6QJTK72CXE19GHQ` | SUCCESS, returned study `failed` |
| original GRPO | `fc-01M7EMCQASD8JS0D65F4VYHA4R` | FAILURE |
| original PPO | `fc-01M7EMCY7KTTJR4GWH64NHX9XB` | FAILURE |
| r01 supervisor | `fc-01M7ETB3CY40QRKMRRBTR3CA0W` | SUCCESS, returned study `failed` |
| r01 GRPO | `fc-01M7ETBCWRDXTYFS8QDAQNNFYD` | FAILURE |
| r01 PPO | `fc-01M7ETBFP8KWVT8GQNW80J3007` | FAILURE |

Original/r01 owner, submission and qualification-allocation records remain intact.
Neither six-job report was submitted. No r02 owner, submission, allocation or
Volume output collision was found. See the committed terminal/reservation records.

Provider metered app attribution at the recorded read-only observation:

| Qualification | App | Metered USD |
|---|---|---:|
| Original | `ap-wBSCIZIwZ4MU9KlMgmZEQS` | 0.11218301 |
| r01 | `ap-UOvpKoVWEyNwJIV1FVi1qS` | 0.55511538 |
| Combined | | **0.66729839** |

For admission, count the larger **entire workspace gross metered total,
$2.22124808**, against qualification. The reviewed remaining qualification bound
is $45 ($26.42 compute + the full $13.40 storage/collection allowance + $5.18
incidental/build allowance). Therefore **$2.22124808 + $45 = $47.22124808 <= $50**.
Funding is demonstrable at this snapshot; the remaining conservative allowance
is $47.77875192. These are metered snapshots, not final invoices; no credits are
counted. The runner rechecks immediately before any new submission and refuses
if the cumulative bound exceeds $50 or billing cannot be established. Before
report submission it requires qualification charges within $50 and total charges
plus the separate $175 report allowance within $225. The **$75 reserve stays locked**.
See [cost reconciliation](../evidence/phase2-exploratory/failure-r01/cost-reconciliation.json).

## Recovery package

After the fix is committed, `.local/recovery-r02/` holds new clean-source plans
for `agl-explore-20-v1-r02/exploratory-qualification/a01` and
`agl-explore-20-v1-r02/exploratory/a01`. Source/plan SHA256s and exact commands are
recorded in `recovery.json`. The qualification retains seed 9101/two steps; the
report retains six 20-step jobs, paired seeds 2001/2002/2003, two GPU lanes and the
same `exploratory-legacy-64` evaluation protocol. Only the seed key and new
source/output identities differ.

The local r02 runner is copied from the reviewed r01 runner, with recovery paths,
accepted source/identity and the conservative prior-charge floor updated. It has
its own persistent 12-hour ledger; old ledgers, logs, reservations and receipts
are neither reused nor cleared. Existing runner regression tests and an offline
dry run are run against the new copy. No launch occurs during preparation.

The exact **manual qualification-only** command is in
`.local/recovery-r02/qualification-command.txt`. Its local wrapper uses the same
reviewed runner source checks, lock, persistent deadline, submission intent and
fresh cumulative budget gate. It submits only qualification, never the report.
The exact overnight command in `overnight-command.txt` can attach to that receipt
and continue, or start qualification itself if not submitted. It waits for
completed qualification, independently validates the collected native checkpoint
and reload evidence, then permits the report. Both commands require explicit
execution flags; preparation does not grant permission to run them.
