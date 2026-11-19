# r02 vLLM IPC path failure and bounded recovery

> Historical failure/recovery record. The later [r03 closeout](PHASE2_CLOSEOUT_2026-11-15.md) records completed execution at `5fb6adfd48ee245509f7dff3ab1d849fcf707d86`; prepared/not-executed statements below describe this earlier repair. Original failed-attempt identities and evidence remain unchanged.


Failed source: `fa4dc93e973344c87fb914b15b5a9c07ce1ee4fc` (PR #26).
Attempt: `agl-explore-20-v1-r02/exploratory-qualification/a01`.

## Demonstrated cause and minimal fix

The GRPO trainer log at line 5061 and PPO log at line 5087 report:

```text
zmq.error.ZMQError: ipc path "/data/phase2/studies/agl-explore-20-v1-r02/.../baseline/temporary/tmpdir/<uuid>" is longer than 107 characters
```

The actual pathnames are 205 and 207 bytes. The exception occurs in installed
vLLM `utils/network_utils.py:338`, `socket.bind(path)`, reached from the Ray
vLLM server's `AsyncLLM`/`EngineCoreClient` initialization. Both algorithms share
this cause. Original line-numbered excerpts and downloaded-file inventories are
in [failure-r02 evidence](../evidence/phase2-exploratory/failure-r02/).

`launch.run_services()` previously assigned `TMPDIR` beneath the persistent
stage directory. vLLM 0.22.0's lazy `VLLM_RPC_BASE_PATH` getter defaults to
`tempfile.gettempdir()` (`envs.py:672`); `get_open_zmq_ipc_path()` appends a UUID
(`utils/network_utils.py:141`). Ray's already-short `RAY_TMPDIR` did not affect
that getter.

The only runtime change removes TMPDIR from the persistent cache-directory
loop and sets both `TMPDIR` and `VLLM_RPC_BASE_PATH` to the launcher's existing,
unique `tempfile.mkdtemp(..., dir="/tmp")` result, alongside `RAY_TMPDIR`.
The path is absolute, independent of the full study identity, and unique per
worker. All three variables are passed to `Popen` before gateway/controller/
trainer initialization, so fresh Python/Ray/vLLM processes see them before
initializing temporary-directory or environment caches. The explicit vLLM
variable also avoids reliance on a previously cached tempfile fallback.

No training implementation, seed, algorithm, model/data, checkpoint or evaluation
path is changed. Logs, traces, metrics, W&B files, checkpoints, receipts and
study identities continue using the existing Volume paths. Other caches retain
their previous paths. No CUDA rebuild, dependency upgrade in the training image,
remote mutation, retry or paid execution was performed.

## Real pinned socket and process regression

The retained installed-package record identifies vLLM **0.22.0**, Ray **2.60.0**,
pyzmq **27.3.0**, and psutil **7.2.2**. The test pins match those versions.
The vLLM release wheel is downloaded as code only (not installed); its complete
SHA256 is `c387a977e35795e8f77b009e019e69722963819c26b55e4a679e09d4279ae35d`.
`tests/pinned_ipc_lock.json` additionally hashes the exact source files used.

The regression executes the whole native vLLM environment module and unchanged
endpoint/socket function bodies from that wheel. It uses real pyzmq/libzmq
bind/connect and a message exchange across real OS processes:

- Both original Volume pathnames reproduce the exact 107-character-limit error.
- The real launcher starts two simultaneous CPU trainer processes under long
  GRPO/PPO stage names. Its inherited TMPDIR and cached tempfile directory are
  deliberately long, testing the explicit `dir="/tmp"` behavior.
- Each trainer starts real Ray 2.60.0 and a nested Ray server actor, using the
  actual pinned VERL job defaults merged with the generated matched environment,
  and the server `runtime_env.env_vars` extracted from native `launch_servers`.
  Those overrides do not replace the three scratch variables.
- The Ray server executes pinned vLLM endpoint generation and `make_zmq_socket`
  bind. A real exec child inherits its environment, executes pinned socket
  connect, and sends its PID, TMPDIR/tempfile and durable-output paths back.
  Both **67-byte** endpoints bind/connect successfully, using different scratch
  directories. Driver, server and child observe the same per-worker scratch.
- A deliberately stale tempfile cache inside the server cannot defeat the
  explicit RPC base. The native RPC getter is cached using vLLM's own cache
  wrapper mechanism; unrelated CUDA-specific environment getters are not eagerly
  evaluated. The fixture verifies durable paths remain beneath the long stage.

Sockets and Ray processes are not mocked. These are CPU tests of the actual
pinned IPC path and environment inheritance; they do not construct an AsyncLLM
model, invoke the training worker setup hook, or validate full GPU initialization.
Those boundaries are deliberately excluded from this offline regression.

Validation: **61 pinned CPU tests passed, no skips; 286 client tests, 274 passed
and 12 expected CPU-environment skips.** Evidence: `socket-tests.txt`,
`pinned-tests.txt`, and `client-tests.txt`.
The CPU preparation script adds only the matching CPU test packages and retrieves
the pinned vLLM code artifact; it changes no remote environment or dependency pin.

## Provider state, preservation and cumulative budget

Read-only inspection confirms all nine owned calls from the original, r01 and
r02 qualifications are terminal: all three supervisors are provider `SUCCESS`
with failed study outcomes, and all six baseline workers are `FAILURE`.
The r02 supervisor is `fc-01M7EWQFMCVQX2YHVNF22FQAAV`; its children are
`fc-01M7EWQR7Z7Y5RQBAF80A9SHPT` and `fc-01M7EWQSX6NCP0TS1MVDKTVMGA`.
All original/r01/r02 reservations and remote evidence remain intact. The r02 local
files are hashed in `r02-local-preservation.json` and retained without edits.

Provider metered qualification app attribution at 2026-11-15 06:18 UTC:

| Attempt | Metered USD |
|---|---:|
| Original | 0.11218301 |
| r01 | 0.55511538 |
| r02 | 0.79442062 |
| Cumulative | **1.46171901** |

For admission, conservatively count the **whole workspace $3.15125017** against
the original $50 qualification allowance, leaving **$46.84874983**. The reviewed
remaining bound is $45 ($26.42 compute + $13.40 full storage/collection allowance
+ $5.18 incidental/build allowance), giving **$48.15125017 <= $50**. Billing is a
metered snapshot, not a final invoice. The runner rechecks live gross metered
charges before submission and refuses if the bound no longer fits or billing
is unavailable. Credits never expand the allocation. The report allowance is
still $175; the $75 reserve stays locked. No new study identity creates a new
spending allocation.

## Recovery procedure, preserving the existing deadline

After commit, source-bound plans and the existing reviewed runner are prepared
under `.local/recovery-r03/`, identity `agl-explore-20-v1-r03`. Qualification
retains two jobs/two updates; the report retains six 20-step jobs, paired seeds
2001/2002/2003, two H100 lanes and exploratory-legacy-64 evaluation.

The runner uses new paths for receipts, collection and logs. It preserves exact
source checks, qualification-before-report, independent checkpoint validation,
no retries and targeted shutdown. Its execution ledger is capped at the existing
r02 deadline **2026-11-15 18:00:37.472249 UTC**, never a fresh 12 hours. Old ledgers
are neither reused as new plan state nor edited; their absolute time bound is
retained. An expired or extended deadline is refused. A new execution window
requires explicit owner approval and is not granted by preparing this package.

The single ready command is in `.local/recovery-r03/overnight-command.txt`, with
source/plan/runner hashes and offline checks in `recovery.json`. The command is
prepared for separate approval, not executed. If approved within the remaining
window, it starts qualification once and submits the report only after completed
qualification evidence passes the existing independent checks. The new execution
ledger and submission receipts are not created during preparation.
