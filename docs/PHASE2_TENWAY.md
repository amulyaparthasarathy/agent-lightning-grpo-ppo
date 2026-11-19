# Phase 2: source-bound a02 and ten-way execution preparation

No billable execution, probe, merge, Secret change, Volume deletion, CUDA rebuild,
or historical Stanford baseline repeat is authorized or performed by this work.

## Two isolated sources

PR #24 contains the import repair at
`62f4de281737dc4e769a26339895a597b4aac32d`. Its a02 plan is
[`evidence/phase2-a02/qualification-plan.json`](../evidence/phase2-a02/qualification-plan.json),
with digest `534a9e1024d93a5cacb7fd21797ab29adb7eb317524886c7922b81c52bc4e8b6`.
Run that qualification only from the clean fix checkout, not the ten-way checkout.
The new runtime module uses only standard-library helpers; remote discovery does
not import `modal_app` or `modal_build`. Local image construction appends sources
after the unchanged CUDA/VERL layers. `modal.is_local()` selects local image
construction versus container discovery. The regression uses Modal **1.6.3**'s
actual `import_single_function_service`, isolated `/root` and `/opt` layouts,
network-denying audit hooks, and both functions; it reproduces the original
`ModuleNotFoundError` and tests missing packaged runtime dependencies. This is
real SDK **offline container-import evidence**, not a newly executed Modal job.

Preserved a01 evidence is under `evidence/phase2-a01`. Stopped App
`ap-IZ5DyjATiXCKQ24RriasXN` and zero tasks are owner-confirmed facts, not a fresh
cloud query. No a01 plan, run path, W&B ID or submission receipt is reused.

The exact a02 command, **only after separate authorization**, from the original
clean checkout is:

```bash
MODAL_PROFILE=amulyasp uv run --locked python scripts/phase2_study.py launch \
  --plan .local/phase2/a02/qualification-plan.json \
  --authorize agl-matched-v2/qualification/a02:534a9e1024d93a5cacb7fd21797ab29adb7eb317524886c7922b81c52bc4e8b6 \
  --out .local/phase2/a02/submission.json
```

a02 explicitly carries `scientific_classification: exploratory-legacy-64`; the
historical manifest is preserved verbatim and does not establish a new holdout.

Bounds carried forward as a **proposal**, not renewed spending authorization:
two concurrent H100s; 600-second startup watchdog; 3,600 seconds per stage;
11,400-second supervisor; six GPU-hours worst case; $100 budget, $10 per-resource
hour ceiling yields $91.67 conservative aggregate compute bound. Those rates are
owner assumptions, not current provider prices or a provider-enforced billing cap.

## Ten-way implementation

`feat/phase2-tenway` is an isolated worktree extending `matched-rollout-mean-v2`.
The CLI accepts campaign and concurrency 1..10; campaign contains fresh seeds
1001..1005 for each algorithm, 60 updates, and 30 stages. A rolling supervisor
fills available slots immediately without waiting for a whole batch. A single
command spawns one detached supervisor. Ten independent single-use H100 workers
can run at once; each owns one stage, Ray instance, checkpoint tree, metrics,
optimizer state, traces and W&B identity. No algorithm/retention/loss/seed math is
changed. Qualification/pilot seed 9001 remains outside campaign seeds.

W&B entity/project remain `amulyap-stanford-university/cs329z-agent-lightning`.
Group is `agl-tenway-v2-campaign`. Job IDs are
`agl-tenway-v2-campaign-<grpo-matched-v2|ppo-mc-matched-v2>-s<1001..1005>-a01`.
Stage names are `baseline`, `training`, `checkpoint-eval`; exact W&B IDs are the
first 24 hex characters of SHA256 of the JSON-encoded full stage name. The plan
contains every exact ID, source hash, pair ID, seed stream, stage type, data hash
and path. Resume is forbidden. Baselines use the matched route and per-job seed;
the historical Stanford bootstrap baseline is never repeated. Final evaluations
load only that job's final native checkpoint. Completion requires audited step,
optimizer/refresh count, W&B flush and plan-bound durable receipts.

Stage writes are disjoint under `phase2/studies/.../<job>/<stage>`; the supervisor
alone writes its aggregate index. Prepared model/datasets are used only as read
inputs and verified by hash. Offline HF/Transformers mode prevents downloads;
all writable caches are stage-specific. The pinned SDK rejects mounting the same
Volume twice; therefore input read-only behavior is an application invariant,
not an independently read-only filesystem mount. Existing Volume v2 is preserved.
Ray sockets use short unique container-local `/tmp/agl-<id>-...` paths; persistent
logs/traces/checkpoints remain in the stage tree.

The supervisor counts in-flight calls, not accepted submissions, against the
configured limit; the Modal worker pool uses the same max_containers value.
Thirty receipts include explicit skipped dependent stages after failure.
Submission reservations and stage entry markers are atomic. Ambiguous submission
is never retried; it stops new submissions. Workers record their call IDs to
recover ownership if a submission response is lost. Completion is checked before
heartbeat expiry to handle races. Lost heartbeat, startup deadline, stage timeout,
budget, owner stop and interrupted supervisor cause targeted cancellation.
A durable supervisor lease stops trainer subprocesses after control-plane loss;
hard function timeouts provide a further bound. SIGKILL can leave stale indexes:
`status` reports uncertainty and `stop-owned` reconciles known and worker-recorded
IDs; it never stops by a shared display name. Provider preemption/restarts are
refused by atomic stage entry markers even with retries=0.

## Evaluation gate

**Blocker: no suitable new untouched 256-case holdout was found in the prepared
local assets.** The only available evaluation data is historical 64/8-case
development data. The prepared campaign proposal explicitly says
`exploratory-legacy-64`; launch fails closed. It is not a final scientific campaign.
No invented questions or fabricated approval records are included.

Supply a separately sourced, authorized 256-case panel and preserve raw source
URI/hash and deterministic selection rule before seeing campaign outputs. Freeze
IDs, order, parquet/JSONL hashes and normalized question hashes; exclude all
training and historical development IDs, exact normalized duplicates, number
substitution templates and near duplicates (SequenceMatcher >=0.90). Human review
must adjudicate flagged pairs, semantic paraphrases, benchmark exposure and
source provenance. Automated checks cannot establish absence of model pretraining
contamination. A reviewer distinct from the training author must attest that no
panel case was used for tuning, early stopping, prompt/scorer design or pilot.
Qualification and pilot are forbidden from using this holdout.

The offline contamination command is:

```bash
uv run --locked python scripts/phase2_evaluation.py audit \
  --training TRAINING.jsonl --development ALL_HISTORICAL_DEVELOPMENT.jsonl \
  --cases NEW_HOLDOUT_256.jsonl --out NEW_CONTAMINATION_REPORT.json
```

Publish a reviewed `data_manifest.json` containing the unchanged training file
and the new eval parquet/JSONL; `holdout` includes `volume_path` under
`phase2/holdouts/<id>`, `source_uri`, `source_sha256`, `selection_rule`,
`untouched_attested`, `training_author`, `independent_reviewer`,
`contamination_review: passed`, `contamination_report_sha256`,
`normalized_question_sha256` (ordered list), `grading_rubric_sha256` and
`grading_protocol: blind-independent-numeric-v1`. Staging is a separately
reviewed data operation; this work does not write to the Volume.

Freeze a rubric for case-level task correctness, tool correctness, format errors,
unit/rounding tolerance and ambiguous targets. Independently extract final answers
from traces with algorithm/seed/outcome labels blinded. The independent grader
imports no training reward scorer, ignores reward bits, compares exact rational
numeric answers and flags ambiguous extraction/units for adjudication. A second
reviewer adjudicates all disagreements and a predeclared random sample of matches;
retain raw traces, grader output, decisions and source hashes. Non-numeric targets
require a reviewed rubric extension before accepting the panel.

```bash
uv run --locked python scripts/phase2_evaluation.py grade \
  --cases NEW_HOLDOUT_256.jsonl --predictions BLINDED_FINAL_ANSWERS.jsonl \
  --out INDEPENDENT_GRADES.json
```

Baseline and final holdout results are embargoed until all predetermined training
runs finish. No best-checkpoint/seed selection, retry for better results, or
interim holdout-driven changes. Include failures in paired seed/case analysis;
five paired seeds still limit inference. A completed execution receipt is not
independent scientific acceptance.

## Admission probe, separately authorized

`admission_app.py` uses a small Python image, no language model, no CUDA/VERL
installation, no Secret and no Volume. Ten workers request **H100!, 16 CPUs,
196608 MiB each**. The detached supervisor submits all ten concurrently, checks
GPU UUID/name, records fresh heartbeats and complete worker allocation intervals,
requires ten distinct GPUs overlapping for 30 seconds, then releases workers.
Startup window is 180 seconds, worker timeout 300 seconds, supervisor timeout
420 seconds. Any uncertainty fails without retry. Requests are approximately
160 CPUs and 1,920 GiB RAM, plus a 1-CPU/2-GiB supervisor.

The offline plan command is:

```bash
uv run --locked python scripts/phase2_probe.py plan \
  --plan .local/tenway/admission-plan.json --probe-id agl-tenway-admission-a01 \
  --rate 10 --budget 15
```

The $10 assumed all-resource hourly ceiling gives **$12.83** maximum modeled
compute cost (1.167 GPU-hours plus conservative supervisor allowance); $15 is a
proposed budget. Actual provider billing includes provisioning/other resource
usage and must be reconciled from the workspace usage export. The probe reports
interval measurements and explicitly leaves `billing_reconciled=false`. Retain
the usage export/reference and owner billing sign-off in the collected receipt
as `billing_evidence_reference` and `billing_reviewer` before setting that
supplement true. The admission measurement must be no more than 24 hours old. This is not a provider billing hard limit.

After exact-plan authorization and fresh capacity evidence:

```bash
MODAL_PROFILE=amulyasp uv run --locked python scripts/phase2_probe.py launch \
  --plan .local/tenway/admission-plan.json \
  --authorize "$(uv run --locked python -c 'import json; from experiment.matched_contract import digest; p=json.load(open(".local/tenway/admission-plan.json")); print(p["probe_id"]+":"+digest(p))')" \
  --capacity .local/tenway/capacity.json --out .local/tenway/admission-submission.json
```

Use `phase2_probe.py status|collect|stop-owned --plan ...` with `--out` for collect.
Ten accepted submissions, ten simulated workers or a pool limit of ten **do not
prove ten simultaneous GPU allocations**. Probe success only establishes capacity
at its measurement time; fresh headroom checks still apply to campaign launch.

## Campaign bounds, operational gates and commands

Prepare the clean-source **blocked proposal** offline:

```bash
uv run --locked python scripts/phase2_study.py plan \
  --plan .local/tenway/campaign-plan.json --study-id agl-tenway-v2 \
  --phase campaign --attempt 1 --concurrency 10 --stage-seconds 7200 \
  --startup-seconds 600 --budget-usd 700 --all-resource-hourly-ceiling-usd 10 \
  --data-manifest evidence/phase2-prepared-data-manifest.json
```

This proposal bounds 30 stage calls at two hours each: **60 GPU-hours**, plus a
22,200-second supervisor; conservative all-resource compute estimate **$661.67**,
proposed budget **$700**. Actual prices are not asserted. Validate the ceiling
against current H100/CPU/RAM/storage rates and owner spending limits. Long-lived
Volume storage charges are separate and depend on retention duration. At observed
role-file sizes, three checkpoints per job consume approximately **1,660.5 GiB**;
reserve at least **3,072 GiB** free Volume space and **100 GiB** measured worker
scratch. No historical artifacts are pruned. Workloads at lower concurrency must
fit the Modal 24-hour supervisor limit; invalid lifetime combinations are rejected.

A fresh (<=15 minutes), attributable capacity JSON must include:
`workspace: amulyasp`, `environment: main`, `observed_at` (Unix seconds),
`evidence_reference`, exact `resources`, `volume_version: 2`,
`gpu_limit/gpu_used`, `cpu_limit/cpu_used`, `memory_mib_limit/memory_mib_used`,
`container_limit/containers_used`, `volume_free_gib`, `worker_scratch_free_gib`.
Use current workspace limits/usage and storage evidence; never fill unknowns with
assumed capacity. Ten workers need headroom for 10 GPUs, 161 CPUs, 1,968,128 MiB
RAM and 11 containers. No live capacity evidence was obtained in this offline task.
The runtime gate checks values and their age, not the authenticity of an arbitrary
local attestation. Owner/reviewer must inspect the cited evidence.

Before campaign, require same-source live qualification, completed 20-step pilot,
reviewed new holdout, same-source measured admission and reconciled billing.
**a02 on the isolated fix source does not qualify a changed ten-way source.**
Qualify the final ten-way source separately using the existing legacy development
panel; no repeated standalone Stanford bootstrap evaluation is needed.

Once these blockers are resolved, regenerate the campaign plan with the reviewed
holdout manifest and new source commit if necessary, then obtain exact-plan
spending authorization. The one-command launch is:

```bash
MODAL_PROFILE=amulyasp uv run --locked python scripts/phase2_study.py launch \
  --plan .local/tenway/campaign-plan.json \
  --authorize "$(uv run --locked python -c 'import json; from experiment.matched_contract import digest; from experiment.matched_supervisor import key; p=json.load(open(".local/tenway/campaign-plan.json")); print(key(p)+":"+digest(p))')" \
  --qualification .local/tenway/qualification/index.json \
  --pilot .local/tenway/pilot/index.json \
  --admission .local/tenway/admission-collected.json \
  --capacity .local/tenway/capacity.json --out .local/tenway/campaign-submission.json
```

```bash
MODAL_PROFILE=amulyasp uv run --locked python scripts/phase2_study.py status --plan .local/tenway/campaign-plan.json
MODAL_PROFILE=amulyasp uv run --locked python scripts/phase2_study.py collect --plan .local/tenway/campaign-plan.json --out .local/tenway/campaign-collected
MODAL_PROFILE=amulyasp uv run --locked python scripts/phase2_study.py stop-owned --plan .local/tenway/campaign-plan.json
```

Evidence collection saves all stage JSON/JSONL/log/text artifacts, aggregate index,
receipt consistency and W&B mapping. Native weights remain on the Volume. Hosted
W&B history and billing require independent readback; a local flush receipt alone
cannot prove hosted history completeness.

Modal references checked during implementation: [resource requests/billing](https://modal.com/docs/guide/resources),
[dynamic pool limits](https://modal.com/docs/guide/dynamic-function-config),
[Volume v2 concurrency](https://modal.com/docs/guide/volumes), and
[targeted FunctionCall cancellation](https://modal.com/docs/sdk/py/latest/FunctionCall).
