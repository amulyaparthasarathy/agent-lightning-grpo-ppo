# Exploratory qualification a01: demonstrated failure and bounded recovery

> Historical failure/recovery record. The later [r03 closeout](PHASE2_CLOSEOUT_2026-11-15.md) records completed execution at `5fb6adfd48ee245509f7dff3ab1d849fcf707d86`; prepared/not-executed statements below describe this earlier repair. Original failed-attempt identities and evidence remain unchanged.


Failed source: `cb246a0c64c5eef0bea6994bbc876e1b9597e56f`, draft PR #26.
Study: `agl-explore-20-v1/exploratory-qualification/a01`.
Evidence was read from the existing Stanford `main` Volume
`cs329z-agent-lightning-working-stanford-v2`; no remote writes or executions occurred.
All times below are **2026-11-15 UTC**, as recorded by Modal, not the host's supplied date.

## 1. Root cause

Both baseline **trainer** processes stopped in `trainer_worker.py:11`, calling
`ppo_runtime.source_identity()`, before `seed_process`, configuration construction,
model loading, Ray training, evaluation or optimizer updates. Both trainer logs end with:

```text
ValueError: Unreviewed runtime source: transformers/models/llama/modeling_llama.py: expected=46d35513266b2a232549531b9cefce1b87dc1632479e2497c36bf4c5ecef12f0, actual=97296b66ed591429b0870e4bd00b9fee62b7fd8422cc0c8a4d4bc3ff2680ac8f
```

The source lock's four Transformers hashes match its declared **5.22.2** wheel,
whose full SHA256 is `829c2820aff91730b8ec45a4b8ff4be07f586a140b68a9073998b36fa35297dc`.
The actual Llama hash exactly matches the official **5.23.0** wheel. Stanford's
retained `image-build/{base-,}installed-packages.txt` and baseline
`resolved_packages.json` independently report `transformers==5.23.0`.
The qualification itself did not save a package inventory; its observed Llama
file identity, rather than an assumed image version, ties this failure to 5.23.0.

`install_modal_training.sh` invokes upstream `setup_verl.sh`; that script installs
vLLM/VERL without pinning Transformers. The later agent overlay also does not pin
it. `matched_app.py` previously reused that environment while the matched worker
unconditionally checked the older source lock. This is dependency drift against
the reviewed model-loading implementation, shared by GRPO and PPO, not a loss,
critic, gateway, CUDA, resource-admission or dataset failure. The lock is correct;
accepting a new hash would silently broaden the reviewed runtime. The four files
have a 658-line diff, including model-loading/checkpointing changes, so numerical
equivalence of 5.23.0 is **not** assumed.

GRPO gateway readiness succeeded at 03:35:53; PPO at 03:36:01.
Both controllers repeatedly received HTTP 200, and gateways shut down normally
after trainer failure. Trainer log mtimes are 03:36:04 and 03:36:11 respectively.
`result.json` records `trainer_failed`, exit 1, no checkpoints, and
`resolved_config_saved=false`. `run_services()` previously returned that code
without writing `error.txt` for a trainer exit; `matched_launch.main()` raised
`SystemExit(1)`. Thus an empty launcher log hid an existing service traceback.

## 2. Complete baseline inventory

These are all 11 files per baseline, plus explicitly missing diagnostics.
Remote integer-second mtimes and SHA256s, including directories, are retained in
[volume-inventory.json](../evidence/phase2-exploratory/failure-a01/volume-inventory.json).

| File | GRPO bytes / UTC mtime | PPO bytes / UTC mtime |
|---|---|---|
| `completion.json` | **missing** | **missing** |
| `error.txt` | **missing** | **missing** |
| `launcher.log` | 0 / 03:35:43 | 0 / 03:35:49 |
| `logs/controller.log` | 1800 / 03:36:05 | 1800 / 03:36:12 |
| `logs/gateway.log` | 1684 / 03:36:05 | 1684 / 03:36:12 |
| `logs/trainer.log` | 681 / 03:36:04 | 683 / 03:36:11 |
| `observability.json` | 1877 / 03:35:52 | 1890 / 03:35:59 |
| `receipt.json` | 507 / 03:36:05 | 511 / 03:36:12 |
| `request.json` | 48065 / 03:35:42 | 48140 / 03:35:48 |
| `resolved_config.json` | **missing** | **missing** |
| `resolved_packages.json` | **missing** | **missing** |
| `resource-preflight.json` | 149 / 03:35:43 | 149 / 03:35:49 |
| `result.json` | 317 / 03:36:05 | 319 / 03:36:12 |
| `run_plan.json` | 101151 / 03:35:52 | 101334 / 03:35:59 |
| `trainer_worker.py` | 1062 / 03:35:52 | 1062 / 03:35:59 |

The paths are beneath `phase2/studies/agl-explore-20-v1/exploratory-qualification/`
with job IDs `agl-explore-20-v1-exploratory-qualification-grpo-matched-v2-s9101-a01`
and `agl-explore-20-v1-exploratory-qualification-ppo-mc-matched-v2-s9101-a01`.
The `traces` and temporary cache directories contain no files. No training or
checkpoint-evaluation stage was launched. Supervisor `plan.json` and `index.json`
were also downloaded. Every existing file was read; both requests validate and
both run plans exactly reproduce committed plan generation. Generated workers
match the committed worker string. Resource preflight reports 549750591488 scratch
bytes; observability records the correct distinct W&B identities. Those records
are declarations, not evidence that a W&B run or model evaluation occurred.

Complete unmodified downloads remain at `.local/failure-a01/phase2/`, with original
remote timestamps in the inventory. Small raw service logs, receipts and results
are committed under [failure-a01](../evidence/phase2-exploratory/failure-a01/).

## 3. Minimal correction and offline proof

- `matched_app.py` adds one matched-only pure-Python wheel install after the
  existing image: the URL and SHA256 already in `ppo_source_lock.json`, with
  `--no-deps --no-build`, targeting `/opt/agent-lightning/.venv/bin/python`.
  This restores the reviewed 5.22.2 implementation; it does not accept 5.23.0 or
  modify source guards, algorithms, CUDA/VERL layers, weights, data or evaluation.
- On trainer exit, `launch.run_services()` writes `error.txt` and prints the final
  16 KiB of `logs/trainer.log` to the parent launcher while retaining exit 1.
- Regression tests execute the actual declared install command in a disposable
  local venv, using independently hash-verified offline wheels. Real 5.23.0
  reproduces the exact guard failure; 5.22.2 passes all **36** source checks and
  installs no dependencies. Another test runs real CPU service subprocesses and
  proves the traceback reaches launcher stderr and preserves the failed result.
- **283 client tests: 274 passed, 9 expected CPU-environment skips. 58 pinned CPU
  tests passed, no skips.** `git diff --check` passed. Logs and 17 unchanged runtime,
  model/data-lock and CUDA-build source identities are committed beside the logs.

Reproduce with the existing client interpreter, `AGL_PINNED_SOURCES` pointing to
prepared Agent Lightning/VERL code and `AGL_TRANSFORMERS_WHEELS` pointing to the
local directory containing the two official wheels. The corrected wheel identity
is in `ppo_source_lock.json`; the failing wheel URL/hash is in the committed
`transformers-5.23.0-wheel.json`. Tests never download or contact Modal:

```bash
python -m unittest tests.test_matched_transformers tests.test_launcher_failure -v
python -m unittest discover -s tests -v
MATCHED_TEST_PYTHON=/path/to/prepared-cpu-python bash scripts/test_matched_training.sh
```

## 4. Verified terminal state and preserved ownership

Read-only `FunctionCall.get(timeout=0)` and `get_call_graph()` at approximately
04:49 UTC agree:

| FunctionCall | Provider state | Application outcome |
|---|---|---|
| `fc-01M7EMCA0TS6QJTK72CXE19GHQ` | SUCCESS, terminal | Supervisor returned `failed` |
| `fc-01M7EMCQASD8JS0D65F4VYHA4R` | FAILURE, terminal | GRPO baseline exited 1 |
| `fc-01M7EMCY7KTTJR4GWH64NHX9XB` | FAILURE, terminal | PPO baseline exited 1 |

There are exactly two children, no pending owned calls. The old qualification
owner, submission, allocation, stop record, receipts and Volume contents remain
untouched. The original report has no owner, submission, index or allocation.
See [terminal-and-reservations.json](../evidence/phase2-exploratory/failure-a01/terminal-and-reservations.json).

## 5. Recovery preparation, budget and blockers

Prepared **as a local plan only**, the noncolliding identity is
`agl-explore-20-v1-r01/exploratory-qualification/a01`. The frozen exploratory
contract only permits attempt 1, so a new study ID avoids altering that contract.
Neither new phase has an owner, submission, allocation, index or Volume path as
of the recorded check. This is not permission or a new spending allocation.

After committing the fix, clean-source qualification and report plans are generated
under `.local/recovery-r01/`, together with validation receipts, source SHA, plan
digests and the exact future CLI command in `recovery.json`. The original `.local/exploratory/`
plans, runner, submission intents and logs are preserved. Do not restart that runner:
it is bound to the old source and identity.

Qualification stays two jobs/two steps, seed 9101, 85-minute supervisor bound.
Report stays six jobs/20 steps, paired seeds **2001, 2002, 2003**, two GPU lanes,
460-minute supervisor bound, final checkpoints and `exploratory-legacy-64` evaluation.
Seed streams, numerical settings, model/data hashes, timeout limits and funding
are compared against the original plans during regeneration.

**$225 total execution allocation remains $50 qualification + $175 report;
$75 reserve remains locked, released $0.** Qualification modeled compute ceiling
is $26.4167; report modeled compute/storage/collection is $171.0667. These are
estimates under fixed bounds, not provider billing guarantees. The failed attempt's
spend must be charged to the same $50 qualification envelope; a new study ID does
not authorize another $50. Its old reservation is deliberately retained.

Remaining blockers: owner authorization for any new paid qualification; reconcile
a01's actual charges/remaining qualification allowance before accepting recovery
within the same allocation, without deleting its reservation; fresh owner-confirmed
gross headroom for bounded admission. No live image build or GPU validation was
performed. Cached CUDA reuse and end-to-end model/evaluation behavior still require
the separately authorized qualification. The report must wait for completed,
collected qualification on the new source. No missing log prevents this diagnosis;
no numerical result is claimed from these failed baselines.
