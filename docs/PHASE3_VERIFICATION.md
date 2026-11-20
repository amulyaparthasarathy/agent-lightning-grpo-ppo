# Phase 3 offline verification

Implementation base: public main `5c069f7e9cc23d03050d13360ce43e27e47e54aa`.
The checkout's origin was verified as the public `agent-lightning-grpo-ppo`
repository; GitHub reported no open PRs at implementation start. PR #3 is merged.
No private Git history was imported. Only hash-verified cached upstream package
sources/wheels and an existing isolated CPU interpreter were reused from local
workbench caches. No CUDA training stack was installed in the client environment.

The draft PR's final commit and the generated candidate receipt identify this
implementation. This report is an offline engineering record, not new experimental
results. The completed reference source/plan and r03 results remain unchanged.

## PR #4 correction pass (2026-11-20)

Started from reviewed head `8bac110184b4e5e102256cb1004386f8823dad38` on
`phase3-reader-task-adaptation`. The public origin, live PR head/base, draft/open
status and clean checkout were rechecked before editing. The sections below this
correction record preserve the prior candidate's verification history; their earlier
counts are not attributed to this correction candidate.

Implemented corrections:

- **Raw Git provenance:** temporary repositories reproduced all three admission
  failures against the prior implementation: replacement A→B admitted B's runtime
  bytes under A; a local graft and inherited `GIT_GRAFT_FILE` each admitted an unrelated
  root. The new regression run initially failed with three “ValueError not raised”
  assertions. All Git operations now use one controlled invocation, disabling
  replacement objects, graft files and cached commit-graph ancestry. Raw working
  bytes are checked against recorded HEAD. The regressions now reject all three;
  linked worktrees, normal forks/direct clones and shallow remediation still pass.
  No reader Git metadata is modified.
- **Crash-safe ownership:** the prior implementation reproduced `JSONDecodeError`
  before reaching a later valid owned process. Registration now flushes/fsyncs a
  private `.pending` file and atomically publishes JSON. Cleanup validates each
  receipt, rejects duplicate/ambiguous ownership and continues independently after
  malformed or interrupted receipts. PID/start ticks/group ownership remain required.
  Real subprocess tests terminate a later separately created group and explicitly
  reap it. Invalid/stale/symlink/duplicate receipts never authorize signaling.
  Group termination, requested signals and unresolved anomalies are reported
  separately. Launcher errors and logs survive cleanup anomalies; probe failure
  receipts and their ownership directory remain available even after timeout or
  cleanup failure. No global process-killing operation is used.
- **Baseline dependency:** `baseline-live-integration-v2` gates dependent training for
  reader v2 qualification. Completed baseline evidence is persisted before the gate;
  an unqualified report fails only that algorithm's dependency and records skipped
  training/reload reasons. Mocked worker-completion tests show one failed algorithm
  submits only baseline while the other runs all three stages. Valid zero-reward
  integration permits both algorithms to continue. Missing/inconsistent/scripted or
  cross-environment reports fail; historical/v1 scheduling remains unchanged. This
  gate does not require accuracy, informative groups, gradients or improvement.

Executed correction commands:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tests .venv-reader-cpu/bin/python -m unittest test_phase3_source test_phase3_corrections -v
bash scripts/test_task_local.sh
bash scripts/test_reader_offline.sh
# Full cached CPU/pinned-source invocation: exactly the environment command below.
PYTHONDONTWRITEBYTECODE=1 .venv-reader-cpu/bin/python scripts/task_local.py probe --out outputs/phase3/correction-asset-probe.json
PYTHONDONTWRITEBYTECODE=1 .venv-reader-cpu/bin/python scripts/task_local.py scripted --out outputs/phase3/correction-scripted.json
```

Results: **19 targeted tests**, **22 task/source tests**, **39 + 15 + 81 reader/offline
tests**, and **61 matched tests plus six pinned-config subprocess tests** passed,
with no skips. The reader gate includes the new correction suite; historical evidence
and accounting checks also passed. The matched gate again exercises 12 reader and
24 historical configurations, unchanged native CPU numerics and real Ray/ZeroMQ.
Expected MCP startup-error diagnostics and missing-hosted-W&B warnings in isolated
fixtures are not hidden. The standalone real asset probe and scripted harness CLI
passed with cleanup receipts. Scripts and local results are not remote qualification.

After committing intended source, both offline fixture plans were regenerated and
validated using the exact commands below. Prior candidate receipt and plans remain
under ignored `outputs/phase3/`; the new `candidate.json` records the final commit,
source/task/environment/configuration and example-plan hashes. No source-bound plan
is added to Git. The final delivery and PR description supply the exact commit/hash
values, avoiding a document that claims to identify its own containing commit.

```bash
uv run --locked python scripts/reader_training.py plan --profile examples/reader-profile.asset.synthetic.json --out outputs/phase3/correction-asset-plan.json
uv run --locked python scripts/reader_training.py validate --profile examples/reader-profile.asset.synthetic.json --plan outputs/phase3/correction-asset-plan.json
uv run --locked python scripts/reader_training.py plan --profile examples/reader-profile.synthetic.json --out outputs/phase3/correction-calculator-plan.json
uv run --locked python scripts/reader_training.py validate --profile examples/reader-profile.synthetic.json --plan outputs/phase3/correction-calculator-plan.json
```

The final tree/index/staged patch, all branch-introduced blobs (including deleted and
intermediate versions), commit metadata and updated PR text undergo the redacted
publication audit described below. Only the existing intentional synthetic credential
and unsafe-URL rejection fixtures require adjudication. Historical reports, source
locks, licenses, model/data locks and numerical algorithms are unchanged.

**No paid work ran.** Real local Git/process/MCP execution, scripted model responses,
simulated trace validation and mocked cloud completion are separate evidence classes.
Local MCP **1.26.0** differs from reference remote **1.30.0**. Fresh remote package
inventory and the real task probe remain mandatory. The unchanged tool-error policy
keeps errors incomplete; this pass changes no task/reward semantics. The single
**$300** aggregate proposal remains unapproved, with no change to its estimate or
contingency. Neither the closed reference authorization nor historical results are
reused or attributed to this candidate.

## Prior candidate verification (8bac110)

### Executed verification

```bash
UV_PROJECT_ENVIRONMENT=.venv-reader-cpu uv sync --locked --only-group reader-harness
bash scripts/test_task_local.sh
bash scripts/test_reader_offline.sh
```

Results: **19 Phase 3 tests passed**, covering real temporary Git histories, origins,
ancestry, shallow history, source/package tampering, closed runtime inventory,
deterministic datasets, parser/verifier behavior, actual MCP, actual harness with
scripted inference, child environment/cleanup, errors and cross-task evidence.
The reader/offline suite passed **39 PPO-MC tests, 15 matched-study tests, and 71
reader/build/import/dependency tests**. Its accounting demo and historical evidence
checker also passed (12 artifact hashes, 24 derived stage summaries, six study runs,
three paired contrasts). No tests skipped in these runs.

The full matched gate used existing isolated CPU dependencies and upstream caches:

```bash
PYTHONDONTWRITEBYTECODE=1 \
AGL_PINNED_SOURCES=/workspace/projects/repos/agent-lightning-lab/.local/ppo-review \
AGL_VLLM_WHEEL=/workspace/projects/repos/agent-lightning-exploratory/.local/failure-r02/vllm-0.22.0-cp38-abi3-manylinux_2_28_x86_64.whl \
AGL_TRANSFORMERS_WHEELS=/workspace/projects/repos/agent-lightning-exploratory/.local/failure-a01 \
MATCHED_TEST_PYTHON=/workspace/projects/repos/agent-lightning-lab/.local/matched-test-venv/bin/python \
bash scripts/test_matched_training.sh
```

**61 tests passed**, with **six additional checks in the pinned-config subprocess**,
and no skips. This executes the unchanged pinned configuration builder and native
CPU numerical paths, real local Ray/ZeroMQ subprocesses, and the offline
Transformers-wheel mismatch/correction regression. Twelve reader configurations
cover two tasks × two algorithms × baseline/training/reload; all 24 historical
exploratory configs also pass. Actual model construction and accelerator imports are
explicit test boundaries; this is not GPU execution. vLLM wheel code is read and
hash-checked, never installed into the client. W&B upload warnings in numerical
fixtures reflect that no hosted W&B operation is performed.

These absolute paths describe this workbench's reused caches, not a reader dependency
on private source. A fresh workbench uses `scripts/test_matched_training.sh --prepare`
and the public pinned source/wheel preparation tools. The optional historical
Transformers-wheel replay needs both public wheels named by its test; without them
that test reports a skip. Essential Phase 3 tests use the locked CPU group and fail
with setup diagnostics rather than silently skipping.

Additional executed local commands (fresh paths were used):

```bash
.venv-reader-cpu/bin/python scripts/task_local.py generate --data outputs/phase3/documented-assets
.venv-reader-cpu/bin/python scripts/task_local.py verify --data outputs/phase3/documented-assets
.venv-reader-cpu/bin/python scripts/task_local.py probe --out outputs/phase3/asset-probe.json
.venv-reader-cpu/bin/python scripts/task_local.py probe --task calculator-v1 --out outputs/phase3/documented-calculator.json
.venv-reader-cpu/bin/python scripts/task_local.py scripted --out outputs/phase3/documented-scripted.json
uv run --locked python scripts/reader_training.py --help
.venv-reader-cpu/bin/python scripts/task_local.py --help
```

The real asset probe passed all six cases, and the calculator probe retained its
valid/invalid expression checks. The scripted harness demonstration passed a causal
join with actual stdio transport and actual loopback HTTP reward delivery. Each
algorithm/stage label is additionally covered by a scripted harness test; none is
relabeled as live-model evidence. Plans for both profiles were generated and validated
after committing source, under ignored `outputs/phase3/`; changing source makes them
stale. No source-bound plan is committed.

## Debugging and compatibility findings

- The existing test expected any single stage to qualify integration. V2 now requires
  all three stages; the test checks baseline alone remains unqualified. A separate
  read-only v1 aggregation regression preserves the historical any-stage semantics.
- Two historical tests required private Git objects. Formal-plan equivalence now uses
  the public baseline; the missing-builder regression uses explicitly labeled fault
  injection through the real SDK. It does not claim to replay private source.
- Import regression checks now inspect the lazy shared lifecycle module as well as
  the app, matching the actual reader-era refactoring.
- Cached IPC wheel lookup honors explicit source/wheel locations. A short-lived
  ZeroMQ PUSH test peer previously used zero linger and could discard its only
  queued message; bounded linger now lets it flush. Production numerical and IPC
  configuration is unchanged.
- A hanging-MCP test initially imported its full test harness before writing the
  ownership receipt. The minimal test server now registers before hanging, allowing
  actual timeout and separately created process-group cleanup to be checked.
- Calculator `Agent` AST matches the public baseline. Its prompt, scorer, data and
  reward behavior are retained; common tracing was extracted with attribution.

## Evidence boundaries

| Category | What ran |
|---|---|
| Real local execution | Git, deterministic Arrow/JSON data I/O, MCP discovery/calls, stdio child processes, loopback reward receiver, owned cleanup, CPU numerical code, Ray/ZeroMQ, package install regression |
| Scripted inference | Production asset harness and tracing with test-only scripted model responses; explicit `scripted-inference` traces |
| Mocked boundaries | Existing cloud submission/ownership tests; numerical model/accelerator construction; test-only small model-file fixture for preparation routing |
| Simulated traces | Explicit `simulated-trace` parser/identity fixtures; rejected by live reporting |
| Unexecuted | Modal image build, remote CPU preparation/preflight, live inference, GPU training, native asset-task checkpoints/reload, hosted W&B reconciliation |

A successful MCP call with a wrong/malformed/wrong-target final answer gets reward
zero. A deliberately corrupted observation cannot change independent verifier truth.
Missing terminal output, truncation, model transport errors, timeout, cancellation or
failed reward delivery remain incomplete. No production test bypass was added.

## Publication review

`python3 scripts/audit_phase3_publication.py` examines tracked/current nonignored
files, index blobs, staged patch, every branch-introduced blob (including intermediate
and deleted versions), commit metadata and optional prepared PR text. It prints rule,
path and line metadata only, never detected values. Its token/private-key/credential-
URL/credential-literal/large-file checks are heuristic, not a guarantee against all
secrets or restricted content.

Manual review adjudicated the existing synthetic credential literal in
`tests/test_image04_import_failure.py` and the deliberate credential-bearing URL
rejection fixture in `tests/test_phase3_source.py`. These are test data, not credentials.
Review also covers the final diff, staged file list, source inventory, PR title/body,
licenses and attribution. No private dotenv was sourced, no model/checkpoint tensors
or raw private traces were added, and historical evidence and source locks were not
modified. Local generated data/plans/receipts stay ignored. The final delivery records
the actual review outcome and PR head.

Live qualification remains proposed under a separate [US$300 aggregate
envelope](PHASE3_QUALIFICATION_PROPOSAL.md). No provider spending cap is claimed.
