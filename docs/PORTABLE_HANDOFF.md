# Portable client, preservation, release and one replication

**Historical Phase 0 approval: PENDING.** The subsequent [worker continuation](PRESERVATION_EXECUTION.md) records the owner’s later YES approval and current execution gates. The following paragraph records the Phase 0 boundary.  Documentation, safe local implementation/tests, branch pushes and draft PRs may proceed. No merge, paid execution, resource creation, cloud-artifact or Secret write is authorized yet. The owner must explicitly change `REMOTE_EXECUTION_APPROVED` to `YES` for the complete scope below. Namespace and credential location are not spending/publication approval.

## Independent reader path

Follow [README client setup](../README.md#2-local-client-setup): obtain access to this private GitHub repository or an authorized fork, choose a directory in your Linux filesystem, clone, then run `uv sync --locked --python 3.12` and `uv run --locked python -m unittest discover -s tests -v` from the repository root. Git, a Unix shell, Python 3.12, uv and network access are prerequisites. Suitable Linux/WSL is enough; a compatible existing Linux container is optional. The lab has no `.devcontainer/` configuration. Host Docker/VS Code attachment and container authentication/cache persistence are environment-specific.

Use your own Modal account/workspace/environment, gated HF access to the pinned Llama base and W&B account/entity. The fixed training app/Volume/Secret names in current source apply within that selected environment, not globally. No flags overriding those names are implied by this guide. The optional [external local-inference patch](../integration/README.md) and self-managed GPU Compose files are separate from the Modal client.

Use `YOUR_HF_NAMESPACE`, `YOUR_MODAL_PROFILE`, `YOUR_MODAL_ENVIRONMENT` and `YOUR_WANDB_TEAM` in reusable configuration. Non-secret destination configuration must remain separate from `.env`, whose existing schema accepts exactly `HF_TOKEN` and `WANDB_API_KEY`. No local Torch/CUDA installation, local inference service, Docker-in-Docker or Docker-socket mount is needed. Never copy `.venv` across architectures or target a container-wide environment with `--active`.

## Owner-specific proposed resources

The author's [HF profile](https://huggingface.co/amulyaparthasarathy) identifies the namespace only. These planned resources have **not** been checked for existence, availability, privacy or capacity:

| Purpose | Proposed resource |
|---|---|
| Private backup | `amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup` |
| Private inference release | `amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-GRPO` |
| Original source | Existing `cs329z-agent-lightning` Volume; account/profile/environment/ID confirmation pending |
| Validation evidence | Existing `agent-lightning-validation` Volume in its confirmed source namespace |
| Restore/replication | One isolated Modal working Volume, exact unused name/ID pending |
| Publishing credential | Reuse a suitable narrowly scoped Secret, otherwise one project-specific publisher Secret after approval |

Prior CS224R models are unrelated and outside scope. Inaccessible resources are unknown, not unused. Reuse matching project resources safely. A unique suffix is allowed only within the approved owner/purpose and existing credential scope. Never broaden credentials, change privacy, invite readers or overwrite unrelated data automatically.

## Credential and capacity preflight after approval

Documentation needs no credential access. For authorized execution, use the existing [safe checker](../scripts/check_validation_env.py) before consuming only `HF_TOKEN` through a non-executing parser into process memory. Do not print, source, shell-expand, commit, archive or upload `.env`. No raw account responses, headers, debug traces, environment dumps or secret-bearing exceptions may enter logs. Authenticate explicitly with the installed locked HF client; local `.env` is not automatically loaded.

Verify sanitized identity `amulyaparthasarathy`, exact destination privacy and operation-specific scope using supported read-only checks, never dummy writes. Base-model read access does not establish publish access. Preserve `agent-lightning-credentials` and its gated-model credential. Only workers requiring publication receive the publisher Secret, containing only the selected HF credential. Prefer read-only access for retrieval. If no existing credential is suitable, request one narrowly scoped credential through the owner's local/Secret workflow, never its value in chat.

Determine private storage capacity through supported non-secret account information. Ask once if capacity cannot be established or owner subscription action is required; do not purchase/change a subscription. Current public prices/allowances are not proof of available account capacity. Check installed APIs before using current documentation.

## Conditional execution envelope

| Limit | Maximum authorized only after `YES` |
|---|---|
| Incremental one-time cost | USD 100 including builds, workers, startup, failed attempts, transfer/readback and contingency |
| Additional recurring cost | USD 30/month total for new project storage and any explicitly approved HF subscription |
| GPU | One H100 concurrently; 16 requested CPU cores and 128 GiB RAM; eight cumulative GPU worker hours |
| CPU-only/build | Ten cumulative hours; each worker at most eight requested cores and 128 GiB RAM |
| Outbound transfer | 500 GiB including retries |
| New retained storage | 300 GiB HF project artifacts; 200 GiB Modal working artifacts; measure scratch/caches separately |
| Recovery retries | At most two diagnosed infrastructure/implementation retries per affected phase, within every global limit; stop repeated identical failures |
| Replication | Exactly one completed 20-iteration replication from the pinned base; no preferred-score reruns |

Before every paid phase, record remaining budgets, conservative worst-case reservation plus contingency, actual/estimated spend and uncertainty. Set explicit timeouts and bounded retries. Delayed billing/credits are not permission to exceed limits. Recheck [Modal pricing](https://modal.com/pricing), [egress billing](https://modal.com/docs/guide/network-egress-billing), [HF pricing](https://huggingface.co/pricing) and [storage limits](https://huggingface.co/docs/hub/en/storage-limits). No new charges or capacity measurements are reported by this documentation change.

## Ordered gates and retained receipts

1. **Local integration:** qualify complete approved backup scope, handle a confirmed platform-managed root without weakening member-path protection, add truthful export evaluation, and retain detached job receipts. A1 file equality alone is insufficient if both copies omit required files. Complete offline tests/review before remote work.
2. **Original backup:** inventory all relevant non-checkpoint evidence, actual prepared base, frozen inputs, every retained smoke/train checkpoint directory including incomplete remnants, original dataset ZIP if present within limits, available validation records in a distinct namespace, and a verified Git bundle containing execution/closeout/integration history. Step 20 and its marker are mandatory. Measure all scope; do not omit artifacts silently to fit a cap. Keep raw evidence private and retained notices intact. Hash remotely on CPU, transfer cloud to cloud into a unique snapshot with create-only/expected-revision protection, and independently read destination bytes in a fresh verifier. Record full coverage, hashes, final Hub commit and a terminal receipt outside the source Volume. Partial attempts remain incomplete and retained.
3. **Restore and export:** restore verified bytes into fresh remote working storage. Evaluation/conversion workers must not mount the original Volume. Perform one 64-case native reload with no updates/saves. Inspect installed VERL merger source; the stock BF16 cast is not acceptable for preservation. Implement a small FSDP1/world-size-1 CPU export preserving every tensor key/shape/dtype/value, with validated equal-alias handling, Safetensors shards/index, tokenizer/template/config semantics and all-tensor readback checks. Unequal aliases/unexpected layouts are stops. Run one clean 64-case export evaluation with explicit phase filters and retain every difference from native reload. Conversion alone needs no H100.
4. **Original private release:** publish only validated inference files, required license/Notice, model card and small reviewed manifests. Review [Meta's primary terms](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE), use Built with Llama attribution and a model name beginning with Llama, and confirm disclosure rights. Private visibility is not an exemption. Use a separate upload action. Record an immutable Hub revision, then retrieve and verify it in a fresh remote cache/process without the source Volume or old export directory. Perform model/tokenizer load and a small fixed generation check. Another full evaluation requires a specific unresolved retrieval/runtime risk.
5. **One replication:** after original backup/release gates pass, run fresh baseline → 20-iteration main → separate native reload → paired analysis, using restored pinned base/data and new names. Keep scientific settings and Weave-disabled behavior unchanged; use the existing approved W&B project. Preserve evidence/final native state in a distinct independently verified backup snapshot, then reuse validated export/evaluation/retrieval under a distinct release branch/revision. Preserve the original release; never choose lineage by score. Existing within-run retention may affect only newly created replication directories. Record retained checkpoints and every score separately.

Keep all original and independently archived artifacts intact. No standalone destructive cleanup is authorized. Streaming remote reads and disposable worker scratch avoid laptop tensor staging. Once approved, successful internal gates permit automatic continuation within this scope; disconnected clients do not cause unsubmitted phases to run. Retain job IDs and durable phase receipts so a different machine can inspect/resume at the next uncompleted gate. A local script is not a durable coordinator.

Stop for ambiguous authority/privacy/scope, source corruption or missing required originals, unsupported semantics, unresolved license/disclosure rights, projected budget/storage excess, exhausted retries or substantive repository review/protection findings. Preserve failures and continue unaffected safe preparation. Do not weaken checks or change accounts/scorer/model/protocol to bypass a stop.

## Retrieval and runtime status

**No release revision exists yet.** A portable remote retrieval recipe will use the verified private repository, immutable commit and release hash manifest, with explicit read authentication. The shape is `snapshot_download(repo_id="YOUR_HF_NAMESPACE/Llama-3.2-3B-AgentLightning-Lab-GRPO", revision="VERIFIED_HUB_COMMIT", local_dir="FRESH_REMOTE_DIRECTORY", token=...)` in a remote CPU process. This is a future API shape, not a laptop download instruction or tested release. Do not copy a base `model_manifest.json` onto trained exports.

The qualified runtime remains Linux x86-64/H100, Python 3.12, VERL 0.8.0, vLLM 0.22.0, PyTorch 2.11.0+cu130, FlashAttention 2.8.3, AutoGen 0.7.5 and MCP calculator 0.2.1. Keep the original prepared tokenizer, fixed 11 Nov 2026 template, frozen 256/64 data, harness/scorer/hyperparameters and sampling recipe. Record effective compute dtype separately from storage dtype. Reuse qualified cached layers; no independent cold build or new machine qualification is claimed.

## Provenance and reporting

Historical main/base/post execution is `7cfddf58a0dece6a63ca67d18b103db2940e4aca`; smoke/reload is `ee120371429a4bc23d11b9da53d166d2fa4843fd`; closeout is `377b721cb0719ef1bee3521fadf4621675f8e4ec`; Agent Lightning is `cbc56e41e0eab643aa71d195e92e5bba652fa683`; A1 implementation is `ab93992daf21999ba658bd038ffe9ba565621431`. Record current integration/execution/converter revisions separately. Run-manifest `source_commit` identifies upstream; submission `request.source.commit` identifies lab execution.

Historical scores and limitations in the [final report](../evidence/grpo-main-01.md) remain unchanged: 27/27/34/35 out of 64; 11 wins, 3 losses, 50 ties; the formatting-sensitive `svamp__chal-520` flip; sparse signal, dropped rows, tool/parser/trace errors, six dependency conflicts, Protobuf ambiguity and historical tensor/W&B readback gaps. New checks cannot retroactively prove historical tensor identity. No claim of general/causal improvement, significance, mathematical audit, convergence, controlled speedup, clean dependencies, ARM64 GPU training or optimizer continuation is warranted.

The final remote report must list PR/commit identities, independently verified backup/restore receipts, all-tensor export checks, original and replication Hub revisions, every measured score/difference, resource durations/bytes/cost uncertainty, recurring storage estimate and the tested retrieval recipe. Until then those stages remain **not executed**, not PASS.

## Phase 0 verification

The documentation branch starts from A1 PR #4 at `ab93992daf21999ba658bd038ffe9ba565621431`, with clean local state and main still at `377b721cb0719ef1bee3521fadf4621675f8e4ec`. PR #4 was open/draft/mergeable with no returned reviews or comments. Only current guidance is edited; dated closeout/evidence, `docs/WORKBENCH_REVIEW.md`, pins, templates and scores remain unchanged. Fresh local check results are recorded in this PR's description, separately from historical 74/85-test records. No credential was accessed for documentation, no container was changed, and no remote phase was launched.

Fresh Phase 0 execution: `MODAL_CONFIG_PATH=/dev/null HF_HUB_OFFLINE=1 uv run --locked --offline python -m unittest discover -s tests -v` passed **85 tests in 6.082 seconds** on CPython 3.12.3/aarch64. A standard-library check parsed **78 local links/anchors and 48 Bash/JSON blocks** in the six changed guides, using `bash -n`/`json.loads` without executing examples; `git diff --check` passed. The installed HF 1.33.0 `snapshot_download` signature was inspected locally, without calling it. No new prose tests were added, and no cold build or independent-client execution was performed.
