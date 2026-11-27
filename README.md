# Agent Lightning: GRPO and PPO-MC

**Start with [Chapter 1 · Why train through the agent harness?](docs/learning/01-agent-training-and-frameworks.md)**

If you can read basic Python, you can follow a calculator request from the model's first choice to the reward for its final answer. In [Chapter 2](docs/learning/02-follow-one-calculator-task.md) you'll predict the score, check it locally and identify which parts of the exchange can train the model.

Continue through the chapters in order. The [learning index](docs/learning/README.md) maps the rest of the route from recorded actions to a model update, then to interpreting a real experiment. This independent lab compares two reinforcement learning methods, group relative policy optimization (GRPO) and a Monte Carlo configuration of proximal policy optimization (PPO-MC).

## Credential-free offline quick start

For the exercises, use a terminal and Python 3.12. If you need a checkout, run these two lines once in a directory where you keep projects. The lesson commands run from the resulting repository root, the directory containing `experiment/` and `scripts/`.

```bash
git clone https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo.git
cd agent-lightning-grpo-ppo
```

| Activity | What you do | What you learn |
|---|---|---|
| Synthetic mechanics | Work through the [calculator trajectory](docs/learning/02-follow-one-calculator-task.md). Then [change the answer after a successful asset lookup](docs/TASK_ADAPTATION.md#a-successful-lookup-and-two-different-outcomes) in the local CPU harness. | Which choices belong to the model, how the verifier scores an attempt, and how training keeps track of that attempt. |
| Committed experimental evidence | Check the [six calculator runs and their baselines](docs/learning/07-read-the-experiment-results.md#preserve-every-baseline-and-paired-outcome). Plot the [recovery results by actual class](docs/learning/07-read-the-experiment-results.md#recovery-readiness-on-the-same-24-cases). | How a higher score on a development panel can hide the loss of every success on one class of cases. |
| New model execution | The [reader procedure](docs/READER_TRAINING.md) covers collecting new answers, training and reloading with separately authorized model access, compute and budget. | A new run produces its own measurements. Both offline activities above work without accounts, model weights or GPUs. |

## What happened on the recovery task?

Recovery GRPO completed a two-update qualification (A03) and a 16-update development pilot (P02), each with a checkpoint loaded by a fresh worker. P02's final greedy policy answered `not_ready` on all 24 development cases. Strict success rose from 9/24 to 12/24, while decision accuracy stayed at 12/24. Recovery-specific PPO-MC live training and reload remain unqualified. The full recovery study remains unexecuted, and the proposed witness-selection supervised fine-tuning experiment is deferred. [A03 qualification](docs/PHASE4C_GRPO_CAMPAIGN_CLOSEOUT.md) · [P02 closeout](docs/PHASE4C_FUNCTIONAL_PILOT_CLOSEOUT.md) · [Paired decision analysis](docs/PHASE4C_PILOT_LEARNING_DIAGNOSIS.md) · [Study protocol](docs/PHASE4C_LEARNING_STUDY_PROPOSAL.md) · [Draft PR #14](https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo/pull/14)

## Current project direction

Contributors can find the agreed milestones in the [project plan](docs/PROJECT_PLAN.md) and the current handoff in the [continuation](docs/PROJECT_CONTINUATION.md). The [task adaptation guide](docs/TASK_ADAPTATION.md) explains what changes when the calculator is replaced by another tool and verifier.

<details>
<summary>Calculator study results, publication provenance and execution limits</summary>

## Completed exploratory comparison

Study `agl-explore-20-v1-r03`, protocol `matched-rollout-mean-v2`, used the **[Phase 2 execution source](docs/learning/source-map.md#public-source-representations)** (exact historical revision recorded in the [publication provenance](PUBLICATION_PROVENANCE.json)). Each of the six runs started from prepared `meta-llama/Llama-3.2-3B-Instruct`, used 32 real rollouts per collection, and completed 20 actor optimizer calls. PPO-MC also completed 20 critic calls per run. Seeds 2001, 2002 and 2003 form three algorithm pairs; qualification seed 9101 is excluded from the means.

| Algorithm | Mean baseline accuracy | Mean final-reload accuracy |
|---|---:|---:|
| GRPO | 41.666667% | 52.083333% |
| PPO-MC | 40.104167% | 51.041667% |

These are frozen-scorer results on the **same reused 64-case development panel**. Baselines differ and are retained. Paired contrasts change sign across seeds. The study establishes neither superiority, equivalence, statistical significance nor fresh-holdout generalization. Answer formatting and tool errors affect the score. Read the [closeout and limitations](docs/PHASE2_CLOSEOUT_2026-11-15.md) before interpreting the means.

Phase 1, the legacy PPO smoke, completed r03, and the **unexecuted/unqualified ten-job campaign** are separate records. Later documentation commits did not produce the original results.

## Publication and execution boundaries

This is the standalone educational publication of Agent Lightning Lab. The original `amulyaparthasarathy/agent-lightning-lab` and its complete Git history remain **PRIVATE**. This repository contains a corrected source snapshot and a new distribution history, not the original execution commits. See the [public source map](docs/learning/source-map.md#public-source-representations) and [publication manifest](PUBLICATION_PROVENANCE.json). The [release plan](docs/PUBLIC_RELEASE_PLAN.md), [readiness findings and portability matrix](docs/PUBLIC_READINESS_2026-11-19.md), and [rights inventory](docs/THIRD_PARTY_RIGHTS.md) record the scoped decisions, verification and publication status. Original software, configuration and executable examples use [MIT](LICENSE); original educational prose, diagrams, reports and selected evidence presentation use [CC BY 4.0](LICENSES/CC-BY-4.0.txt). See [license scope and third-party exceptions](LICENSING.md); upstream MIT/Apache notices are preserved, and underlying datasets/models are not relicensed.

Historical training code is available for inspection here. The historical matched route remains bound to owner-specific Modal/W&B resources, frozen inputs, exact source admission and prior qualification. The separately versioned [reader procedure](docs/READER_TRAINING.md) supplies reader-owned resource profiles. The reference Llama/calculator workflow passed live operational, integration, checkpoint and fresh-worker evaluation-reload checks on the [recorded configuration](docs/READER_QUALIFICATION_2026-11-19.md). Later documentation and merge commits did not produce those live results.

This short reference run established procedural execution, not reliable learning improvement. GRPO had 0/16 informative reward groups and no observed task-directed policy-gradient signal. PPO-MC made nonzero updates without establishing useful critic convergence or reliable improvement. Evaluation-only reload does not qualify resumed training. The alternate asset task was later [operationally qualified](docs/PHASE3_QUALIFICATION_2026-11-24.md), again without a learning claim; alternate-model work remains open, and future environments or cold builds are not universally qualified. The first release includes no model weights, full datasets, private editorial ZIP, raw trajectory collection or missing chart atlas. The local exercises use the files in this repository.

</details>

## Historical operations and retained records

The notes below are retained historical records from the private original. Their private-state statements, PR links and external artifact locators are dated provenance, not public reader dependencies. Public code equivalents are in the source map. They preserve earlier scientific assertions, useful links and operational context. They are not part of the offline quick start. Commands involving cloud services require separate authorization and may incur costs. The matched-study portability limits above take precedence over any generic client-setup wording below.

<details>
<summary>Expand historical results, client setup, runtime and preservation notes</summary>

**Phase 2 completed — November 15, 2026:** [r03 closeout](docs/PHASE2_CLOSEOUT_2026-11-15.md) records six study runs, 20 actual actor optimizer calls each, 20 critic calls for each PPO-MC run, and six independent final-checkpoint reload evaluations. GRPO mean baseline → final is **41.666667% → 52.083333%** (+10.416667 pp); PPO-MC is **40.104167% → 51.041667%** (+10.937500 pp). These are three seed pairs on the reused 64-case development panel, not evidence of superiority, equivalence or fresh-holdout generalization. Qualification seed 9101 is excluded. Executed source: **[Phase 2 historical implementation](docs/learning/source-map.md#public-source-representations)**, not later documentation/merge commits. The exact original execution commit is retained in the [publication provenance](PUBLICATION_PROVENANCE.json).

Read the [selected evidence](evidence/phase2-r03/EVIDENCE_INDEX.md), or run `python3 scripts/check_phase2_results.py` for credential-free CSV/hash checks. The implemented ten-job scheduler remains **UNEXECUTED/UNQUALIFIED** for its formal ten-GPU campaign and untouched-holdout route; its gates are unchanged. At that November 15 closeout, the original repository remained private and public-readiness work was next. The [release plan](docs/PUBLIC_RELEASE_PLAN.md) records this snapshot’s later publication disposition.

**Historical Phase 2 handoff — November 14, 2026:** the [Stanford bootstrap report](docs/PHASE2_STANFORD_BASELINE_2026-11-14.md), [matched-study contract](docs/PHASE2_MATCHED_STUDY_CONTRACT.md), and dated failure reports preserve the pre-r03 state. Their prepared/not-launched language applies at those dates and is superseded for r03 by the completed closeout. Historical source identities remain unchanged.

**Preservation closeout, November 14, 2026:** existing workflow `preserve-20261114-01` is complete. Original and replication exports have immutable private releases and independent checksum/model-loading receipts. The [closeout report](docs/PRESERVATION_CLOSEOUT_2026-11-14.md) records all revisions, execution commits, checkpoint identities, CPU reconciliation and remaining generation differences. Only replication publication/retrieval ran during closeout; scientific results below remain historical and unchanged.

The [GitHub-only closeout review](docs/GITHUB_CLOSEOUT_REVIEW_2026-11-14.md) documents the evidence checks and offline tests for the repository changes. It performs no new scientific or preservation execution.

**Historical experiment status, November 13, 2026:** the pinned Llama 3.2 3B Instruct adaptation ran on one H100 through real vLLM inference, MCP calculator continuation, GRPO training, native checkpoint save and evaluation-only reload. See the [smoke evidence](evidence/grpo-smoke-01.md) and [20-step results](evidence/grpo-main-01.md), including provenance, all-step metrics and limitations.

The historical Phase 1 result is **success/accuracy under the frozen answer extractor and reward**: baseline and train-initial 27/64, train-final 34/64, independent reload 35/64. Baseline versus reload has 11 wins, 3 losses and 50 ties (24 both correct, 26 both incorrect), a descriptive +8/64 or 12.5 percentage points. This is not independent semantic grading of all answers or evidence of statistically significant/general learning improvement. The documented `svamp__chal-520` difference is formatting-sensitive: both runs calculate 6, but the train-final extractor selects a quoted placeholder. The scoring mechanism is explained; the cause of changed wording is not.

The [evidence index](evidence/README.md) is current. [Root verification](VERIFICATION.json), [experiment verification](experiment/VERIFICATION.json), [validation verification](validation/VERIFICATION.json), [source provenance](SOURCE_PROVENANCE.json) and dated records retain their original scopes/counters; their earlier “not run” statements are historical, not the current status. New documentation checks are recorded separately in the [November 13 closeout note](docs/CLOSEOUT_2026-11-13.md).

## 1. Inspect the published result without cloud access

For completed Phase 2, read the [r03 closeout](docs/PHASE2_CLOSEOUT_2026-11-15.md) and run `python3 scripts/check_phase2_results.py`. The command below is retained specifically for **Phase 1**.

Read [the report](evidence/grpo-main-01.md) and [sanitized 64-case CSV](evidence/grpo-main-01-cases.csv). The CSV includes four rewards per case, baseline/post call/error/token/duration measures and trace IDs, without full prompts, datasets or raw trajectories. From the repository root, this standard-library-only snippet works with Python 3.12; it needs no Modal, W&B, CUDA or authentication:

```bash
python3 - <<'PY'
import csv
from collections import Counter
from pathlib import Path

rows = list(csv.DictReader(Path('evidence/grpo-main-01-cases.csv').open()))
assert len(rows) == len({r['case_id'] for r in rows}) == 64
assert all(r['case_id'] for r in rows)
phases = ('base', 'train_step0', 'train_step20', 'post')
for r in rows:
    for phase in phases:
        key = phase + '_reward'
        assert r[key] in ('0', '1'), (r['case_id'], key)
        r[key] = int(r[key])
totals = {p: sum(r[p + '_reward'] for r in rows) for p in phases}
assert totals == dict(base=27, train_step0=27, train_step20=34, post=35)
assert all(r['base_reward'] == r['train_step0_reward'] for r in rows)
pairs = Counter((r['base_reward'], r['post_reward']) for r in rows)
outcomes = dict(wins=pairs[0, 1], losses=pairs[1, 0],
                ties=pairs[0, 0] + pairs[1, 1],
                both_correct=pairs[1, 1], both_incorrect=pairs[0, 0])
assert outcomes == dict(wins=11, losses=3, ties=50,
                       both_correct=24, both_incorrect=26)
flips = [(r['case_id'], r['train_step20_reward'], r['post_reward'])
         for r in rows if r['train_step20_reward'] != r['post_reward']]
assert flips == [('svamp__chal-520', 0, 1)]
for phase, expected in [('base', (140, 76, 23, 44546, 5468, 13)),
                        ('post', (134, 70, 12, 42765, 5633, 10))]:
    measures = ('model_calls', 'tool_calls', 'tool_errors',
                'prompt_tokens', 'completion_tokens')
    sums = tuple(sum(int(r[phase + '_' + k]) for r in rows) for k in measures)
    affected = sum(int(r[phase + '_tool_errors']) > 0 for r in rows)
    assert sums + (affected,) == expected
    print(phase, dict(zip(measures, sums)), 'episodes_with_tool_errors', affected)
print('correct/64:', totals)
print('baseline -> post:', outcomes)
print('train-final -> post flips:', flips)
print('PASS: published CSV internal consistency only')
PY
```

This checks the published CSV's internal consistency, not the private raw traces, completeness against an independently obtained held-out manifest, or tensor identity. The earlier artifact audit is documented in the report. Missing raw data cannot be reconstructed from the CSV.

## 2. Local client setup

An independent client needs Git, a Unix shell, Python 3.12, uv (the reviewed client uses 0.12.13), network access and authorized accounts. Run directly in suitable Linux/WSL or use an existing compatible Linux container. The original historical repository required private access. This educational snapshot does not require that access; cloud execution still requires independently authorized accounts. Never embed tokens in clone URLs.

The author's architecture is **Windows 11 host → WSL2 Linux → existing Linux Dev Container/client terminal → separate Modal runtime**. This lab has no `.devcontainer/` definition; cloning it does not supply a container or a repository-configured “Reopen in Container” workflow. If you choose a container, Docker and VS Code attachment belong to your host setup. Run Git, uv and Modal commands in the selected Linux client terminal. No local CUDA, Torch, inference service, Docker-in-Docker or Docker-socket mount is required for the Modal client. Optional self-managed GPU Compose files are a different runtime path.

Prefer a reader-owned Linux filesystem checkout when using WSL, as described in [Microsoft's filesystem guidance](https://learn.microsoft.com/en-us/windows/wsl/filesystems). Host and container paths can differ. Authentication/cache persistence mounts are specific to your environment; `/state/modal.toml` and `/state/uv-cache` are historical examples, not required locations. Use normal private credential storage. See [VS Code's container guidance](https://code.visualstudio.com/docs/devcontainers/containers) if using an existing container.

The following optional client sequence describes the original environment. Full test discovery includes historical-Git checks whose objects are deliberately absent from this snapshot. It is not the supported offline quick start above; see the release receipt for the bounded 54-test verification actually performed. Use a directory you own:

```bash
git clone https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo.git
cd agent-lightning-grpo-ppo
git status --short
git rev-parse HEAD
uv --version
uv sync --locked --python 3.12
uv run --locked python --version
uv run --locked python -m unittest discover -s tests -v
```

[PR #3](https://github.com/amulyaparthasarathy/agent-lightning-lab/pull/3) was merged in the private original; this is an archival discussion locator. Its original `main` recorded that closeout. The dated preservation and Phase 2 reports identify their own source commits; consult those records when choosing an archived execution source. Existing checkouts should inspect status and use `git pull --ff-only` on the intended branch, preserving unrelated changes. The historical base/train/post experiments used the **[Phase 1 source revision](evidence/grpo-main-01.md#sources-and-frozen-inputs)**, not a later documentation commit. Record the clean revision used for any new submission. The adapter records a dirty flag but does not refuse all dirty checkouts.

`uv sync --locked --python 3.12` manages this project's `.venv` and refuses a stale lockfile. `uv run --locked` selects it without activation. Do not use `--active` to target a container-wide environment. **`uv.lock` locks local clients, not the entire remote GPU dependency closure.** No local Torch/VERL/CUDA installation is required for this path.

Do not copy credential stores or `.venv` between machines or architectures. Recreate the repository environment with uv. Historical aarch64 offline checks qualify only the client tests. The remote installer requires Linux x86-64 and H100/SM90; ARM64 GPU training and an independent cold build remain unqualified.

## 3. Accounts and runtime boundaries

Use your own authorized Modal workspace/environment, W&B account/entity and gated Llama access. No prior course account is assumed. Reuse existing authentication; if none exists, the owner can use `uv run --locked modal token new` interactively. `modal profile current` shows local selection, not proof of remote authentication. Set `MODAL_ENVIRONMENT` to the intended environment name for a non-default environment, consistently for validation, experiment and Volume commands.

In that environment, configure the named Secret **`agent-lightning-credentials`** with **`HF_TOKEN`** and **`WANDB_API_KEY`** using the dashboard. Reuse an existing authorized Secret; do not overwrite it. The HF token needs approved read access to `meta-llama/Llama-3.2-3B-Instruct`; W&B needs write access to the reader's entity slug (`YOUR_WANDB_TEAM` in examples). The runtime needs no GitHub key: Modal uploads allowlisted source files from the local checkout.

[VALIDATION.md](VALIDATION.md#2-credentials-optional-local-env-named-remote-secret) explains optional private `.env` staging. A local `.env` is not the runtime Secret; the helpers and trainer do not automatically load it. Do not source it or expose values. Dashboard entry avoids a local copy. The documented optional dotenv import is an explicit Secret creation operation, not a required setup step for an already configured account.

| Location | Persistent state / role |
|---|---|
| GitHub | Source, build recipe, provenance, reviewed reports and sanitized CSV |
| Local client | `.venv`, uv/client caches, ignored `.local` downloads; not a backup |
| Modal image | Separate GPU dependencies and `/opt/agent-lightning/.venv/bin/python` |
| Modal Secret | Runtime HF/W&B credentials; not image contents |
| `agent-lightning-validation` Volume | Synthetic validation evidence at `runs/NAME/` |
| `cs329z-agent-lightning` Volume | Models, frozen data, receipts, traces, native checkpoints and evidence |
| W&B | Trainer metrics in `cs329z-agent-lightning`; validation in `agent-lightning-validation` |
| Optional hosted Weave | Agent views only with explicit `--weave`; local JSONL is recorded regardless |

The training app, Volume and Secret defaults are `cs329z-agent-lightning`, `cs329z-agent-lightning` and `agent-lightning-credentials`; validation uses its separate `agent-lightning-validation` Volume. These names are scoped to the selected account/environment. The default training app can create its Volume if missing, so do not import/run it as a source-preservation preflight. The new [isolated working-Volume mode](docs/INTEGRATION.md) requires an existing distinct Volume and never creates it implicitly. No generic app/Volume/Secret override flags are implied; the implementation guide documents the specific non-secret working-Volume environment setting. A fresh example name does not isolate existing model/data assets inside a shared Volume; use an owner-selected environment with the intended state. Do not delete old evidence to make room for names.

All measured runs used **`weave=false`**. Primary reproduction commands omit `--weave`. Adding it sends public-task prompts, visible outputs, tool calls and observations to hosted Weave. W&B console “synced” output alone is not hosted readback proof. Phase 1 training-run W&B readback and Weave delivery remain unverified in that historical record; the r03 audit separately reconciled all 24 planned W&B identities and recorded scalar histories. Earlier synthetic validation readback has its own narrower evidence.

## 4. Reproducibility levels and limits

| Level | What is available |
|---|---|
| Published-result arithmetic | CSV and the offline command above; reproducible without private cloud access |
| Executable procedure | Existing validation and experiment runbooks, pinned source/model/template and data manifests |
| Historical execution | Reviewed smoke, 20-step and reload artifacts; H100 fit demonstrated for these exact settings |
| New cold-build reproduction | **Not performed by this closeout**; cached-image success does not qualify a fresh dependency resolution or another machine |

The [two-stage image recipe](experiment/README.md#image-recipe-and-rebuild-limits) preserves GCC selection, H100 build targets, interpreter isolation and PyArrow compatibility. Six exact metadata conflicts remain under a diagnostic exception; changed versions/conflicts fail the validation gate. Scoped versus ambient Protobuf inventories do not establish the actual imported version in the experiment processes. No gate is weakened here.

Model/upstream revisions, prepared tokenizer/template hashes and frozen data hashes are in the [results provenance table](evidence/grpo-main-01.md#sources-and-frozen-inputs). Seed 329 alone cannot identify the data: [preparation controls](experiment/README.md#frozen-input-identity) also record source Parquet hashes, selected IDs and output hashes. Upstream availability and a future cold build remain gaps. Neither temperature zero nor a native reload promises identical generation or identical scores elsewhere.

The retained limitations include sparse signal (30/160 informative groups), six zero-signal iterations, 114 dropped rows, parser fallbacks, trace splitting and tool errors. Zero gradients do not prove unchanged AdamW parameters. Latency comparisons are descriptive, not a controlled speedup. Operational restoration is corroborated without independent tensor-identity proof.

## 5. Prepare on Modal, then pass the smoke checks

For a new reproduction, first follow the **[current clean-start validation path](VALIDATION.md#current-clean-start-path)**, then the **[ordered experiment actions](experiment/README.md#action-order-and-acceptance-evidence)**. Those are billable commands for the reader's separately authorized resources; none is needed to inspect the historical result. They cover services → image → GPU validation → prepare → smoke → smoke-reload-eval → baseline-eval → main → reload-eval → summarize → export.

Run actions serially, inspect terminal evidence before advancing, and use a fresh prefix throughout (the runbook uses `repro-a-`; replace it if occupied). Historical `base-01`, `train-01`, `post-01` and other names are already used in the author's environment. Names are preserved even on failure. A guard is not a distributed lock; detachment does not queue later commands. Exact submission behavior, resource caps and failure diagnostics are in the runbooks.

The GPU phases request one `H100!`, 16 physical CPU cores and 128 GiB RAM. Inner limits are 45 minutes for smoke/evaluation and 180 minutes for main; the GPU function cap is four hours. CPU preparation/collection and image builds can also be billable. The November 11, 2026 planning estimate of approximately **$5.73 per active GPU-phase hour** includes requested CPU/RAM but excludes builds, storage and other usage. It is not a measured bill or current price guarantee. No account-specific storage quota or availability is promised.

## 6. Persistence and remaining work

The [preservation handoff](docs/PORTABLE_HANDOFF.md) is a historical proposal. Its completed named Phase 1 workflow is documented in the [November 14 preservation closeout](docs/PRESERVATION_CLOSEOUT_2026-11-14.md). That completed scope does not establish an r03 checkpoint backup.

For the completed experiment, see [A1 preservation preparation and the A2 approval block](docs/PRESERVATION.md): an offline manifest/comparison utility, fixture tests, three preservation groups and a proposed cloud-to-cloud verification sequence. Its pre-execution status is historical; use the November 14 closeout for completed Phase 1 preservation identities.

Native VERL/FSDP checkpoints are resume artifacts with actor, optimizer and extra state, not ready-to-deploy Hugging Face weights. For historical `train-01`, steps **15 and 20** retain actor checkpoints; step-5/10 directories retain only `data.pt`. Directory existence alone does not identify a complete checkpoint. See the [inventory and reload limits](evidence/grpo-main-01.md#post-restoration-and-native-checkpoint).

The [evidence exporter](experiment/README.md#8-export-evidence-and-finish-cloud-work) excludes model and native checkpoint tensors. It includes run logs/traces and frozen data, so inspect/redact before sharing; it is **not a model backup**. No independently verified **r03** archive/download backup or model publication is claimed by this closeout. A later explicitly authorized archive/download/verification step should record archive size/hash, verify the downloaded copy and inventory, and store it durably; checkpoint backup requires a separate tensor-aware scope. `.local` and a single Volume are not independent backups.

Deferred work: a separately versioned extractor fix with quoted-placeholder/repeated-marker fixtures; remote dependency-closure and cold-rebuild hardening; ARM64 training qualification; validated native-checkpoint conversion and private model publication under the base-model license. None requires another run to finish this documentation closeout.

## 7. Navigation

- [Experiment runbook](experiment/README.md): exact actions, inputs, artifacts, failure handling, optional CPU accounting example.
- [Validation guide](VALIDATION.md): current environment gates, then clearly separated historical diagnostics.
- [Evidence index](evidence/README.md): smoke/final reports and sanitized case records.
- [Workbench review](docs/WORKBENCH_REVIEW.md): dated environment decisions; optional [local inference integration](integration/README.md) is separate.
- [Closeout checks](docs/CLOSEOUT_2026-11-13.md): this documentation revision's checks and remaining limits.

The original course schedule in the runbook is historical context. A fresh reader needs their own repository/service access, not the author's course repositories or filesystem.

</details>
