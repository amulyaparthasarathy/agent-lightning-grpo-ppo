# Proposed asset-task live qualification — not authorized

Propose one new **US$300 aggregate Phase 3 ceiling**, covering the initial campaign
and evidence-driven corrections. This is a proposal only. No build, remote CPU
preparation/preflight, GPU stage or other billable Phase 3 work has been executed.
The completed reference campaign's $250 authorization is closed and is not reused.
Approval of this numerical envelope is required before any paid action.

## Candidate and prerequisites

Use the final implementation commit on the Phase 3 draft PR, based on canonical
public main `5c069f7e9cc23d03050d13360ce43e27e47e54aa`. The delivery report supplies
its exact full commit. After checkout, regenerate source-bound plans from that clean
commit. The candidate source/task hashes are also recorded in ignored
`outputs/phase3/candidate.json`; this document cannot identify the commit containing
itself in an executable plan.

- Task: `asset-status-v1`; recipe `asset-status-v1/2026-11-20/329`.
- Environment SHA-256:
  `d683b4ca993f4d014de260fe8ba04afa901b1488be036a633c37b4fec7ef4089`.
- Generator, parser, verifier, harness, server and tracing hashes: the candidate
  plan's `task_identity.source_sha256`, bound to the exact committed files.
- Model: `meta-llama/Llama-3.2-3B-Instruct`, revision
  `0cb88a4f764b7a12671c53f0838cd831a0843b95`, existing `llama3_json` and immutable
  prepared-asset locks. No new model compatibility claim.
- Agent Lightning: `cbc56e41e0eab643aa71d195e92e5bba652fa683`.
- VERL source: `7aed6b230776f963fa09509c10d9c3a767d1102c`, wheel 0.8.0.
- Upstream trainer SHA-256:
  `a0ddd84022896cd34f1d2e30bf6b35897497df62847f94ade98703c22e02b368`.

After approval, bind an explicit owned profile and fresh resource/path/attempt
identities, verify ownership and existing credential handles, and reserve the exact
plan before image resolution. Retain local and remote submission receipts. Generated
remote Parquet bytes and actual package inventory must be verified; local byte hashes
are not assumed to prove a future remote artifact. No calculator training is proposed.

## Campaign and acceptance

One H100! lane, 16 CPU cores, 196608 MiB, one worker per stage. Both algorithms use
paired seed 9001, their existing independent sampling streams and the unchanged
matched protocol. For each algorithm run:

1. Baseline on all 64 frozen heldout assets, exactly one completed rewarded attempt
   per case, with no optimizer calls.
2. Two updates: eight questions × four rollouts each collection, complete generated
   tokens and observation masks, equal-rollout weighting. GRPO: two actor calls;
   PPO-MC: two actor and two critic calls, terminal MC targets/residual advantages
   and whitening. Require finite gradients/losses per update, actor refresh 0/1/2,
   and retained group/critic diagnostics. Zero informative GRPO groups are reported
   as absent task-directed signal, not hidden by optimizer-call success.
3. Native `global_step_2` model/optimizer/extra-state files for required roles,
   content inventory and source/config identity. Evaluate that exact checkpoint on
   all 64 heldout assets in a distinct provider container/process with no optimizer
   step and refresh at step 2. This does not qualify resumed training.

For **each of six algorithm/stage pairs**, require at least one strict live causal
join: unique model-emitted call → matching arguments/ID → successful MCP execution →
exact observation in subsequent model input → continuation → terminal answer →
confirmed binary reward delivery. Report full integration coverage separately from
terminal accuracy and training completion. Test malformed outputs, wrong targets,
truncation and failed delivery against the offline-established semantics; preserve
failed/incomplete attempts rather than zero-filling them. No partial-credit shaping.
The v2 collection rule cannot be satisfied by baseline-only evidence.

Reconcile all six W&B IDs, source/task/environment configuration, history steps,
finished status and local metrics. Collect stage receipts, resolved configs, traces,
package/build identities, checkpoint inventories, learning diagnostics, supervisor
index and cancellation/failed-attempt evidence. Independently query hosted W&B after
collection; the local collector explicitly does not claim that query occurred.
Report operational completion, tool invocation, integration, correctness, informative
signal and observed accuracy changes separately. Two updates are qualification, not
reliable evidence of general learning improvement.

## Deadlines, estimates and aggregate allocation

Each GPU stage has a 3600-second bound, 600-second startup limit, shortened dispatch
clock reserve, durable heartbeat and no automatic retries. Six stages bound initial
GPU exposure at six hours; supervisor at 22200 seconds. The shared builder retains
8 CPU / 64 GiB / 7200 seconds, finalizer 4 CPU / 16 GiB / 1800 seconds, reader overlay
4 CPU / 16 GiB / 900 seconds. Preparation is 4 CPU / 16 GiB / 3600 seconds; remote
preflight 4 CPU / 16 GiB / 900 seconds. Single-use stage workers and the existing
cancellation/ownership guards stay enabled.

[Modal's pricing page](https://modal.com/pricing), checked 2026-11-20, lists H100 at
$0.001097/second, CPU at $0.0000131/core-second, memory at
$0.00000222/GiB-second, Volume storage at $0.09/GiB-month and egress at $0.04/GiB.
At the requested shape this implies about $6.24/worker-hour before premiums and
other costs. Budget with the existing conservative **$10/worker-hour** assumption,
not the GPU-only price. The shared supervisor's current conservative accounting
also charges its elapsed time at the worker ceiling; the initial training allocation
covers that overestimate. No region pinning, premium execution or subscription upgrade
is assumed. Verify rates/options before submission; credits/cache hits/free allowances
are assigned zero value in this budget.

| Aggregate category | Allocation |
|---|---:|
| Initial six GPU stages plus conservative supervisor accounting | $125 |
| Build/image resolution, dependency layers and build overhead | $30 |
| Initial model/data preparation and remote CPU preflight | $10 |
| Up to 300 GiB, seven days retention | $7 |
| Up to 2 GiB collection, CPU reconciliation and observability overhead | $3 |
| Correction contingency, including new builds/preflight/attempts and additional retained evidence | $125 |
| **Total new ceiling** | **$300** |

The contingency is a shared remaining balance, not permission for unlimited retries.
Before each correction reserve its full maximum exposure, including existing retained
storage and unresolved calls, against the $300 ledger. Use new committed source,
plan hashes and fresh attempt/allocation identities where required; preserve prior
failed evidence and reservations. Ordinary corrections can proceed within an approved
envelope after reconciliation, without a separate approval per correction. A correction
that cannot fit the remaining balance stops. Do not change numerical mathematics,
model scope or safeguards to make it fit.

The 300 GiB/seven-day policy applies to new Phase 3 outputs; track retained failed
attempts in the aggregate. At seven days review/archive only new Phase 3 artifacts
under explicit ownership. Do not delete historical resources or infer deletion
permission for them. Reconcile storage before continued retention.

## Stop conditions and uncertainty

Stop on lost ownership/lease, heartbeat/startup/deadline failure, identity/source or
package drift, incomplete rollouts/evaluation, nonfinite training evidence, failed
reward delivery, wrong checkpoint/worker identity, storage bound or insufficient
unreserved balance. Preserve evidence and request cancellation only for known owned
calls. A lost submission acknowledgement remains **unknown**: reserve its full possible
exposure, look up durable reservations/provider calls, and never blindly resubmit or
clear a reservation. Do not restart until exposure is reconciled.

This ledger and the estimates are conservative operating controls, **not guaranteed
provider spending caps**. Build/startup/storage/telemetry can outlive estimates and
provider billing may lag. Use observed charges plus worst-case unresolved exposure;
never count credits as a hard cap. No paid action follows from publishing this proposal.
