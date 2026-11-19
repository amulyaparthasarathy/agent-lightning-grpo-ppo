# A1 preservation preparation — November 13, 2026

**Status: `not_executed`; `backup_verified=false`.** This PR prepares an offline inventory and the A2 decision. No real backup transfer, Modal account request, image build, remote process, conversion, inference or publication was executed. No destination is selected. All source artifacts remain untouched; successful future backup does not authorize deletion.

The [current owner handoff](PORTABLE_HANDOFF.md) expands the planned backup scope and defines conditional full-execution approval. Its status is **PENDING**. Once approved, its global gates replace separate per-stage approval requests within that scope; the A1 record below remains historical preparation, not execution evidence.

## Scope and identity

Paths below are relative to the existing **`cs329z-agent-lightning`** Volume's logical `/data` mount. Never use an internal Modal mount prefix as a portable location. Recommend step 20, all required non-checkpoint evidence and the actual prepared base for the first backup, in three separate manifest groups:

| Group | Required scope |
|---|---|
| `evidence` | All non-checkpoint `runs/`, `evidence/`, `submissions/` and frozen `data-small/`, including logs, raw traces, resolved configuration/packages, observability, source snapshots/hashes, receipts and build evidence. Explicitly include `smoke-01`, `smoke-reload-01`, `base-01`, `train-01`, `post-01`, `evidence/paired-01/` and its submission. Preserve a separate immutable repository snapshot containing execution, closeout and preservation revisions. |
| `native_state` | The **entire** `runs/train-01/checkpoints/global_step_20/` directory, plus `runs/train-01/checkpoints/latest_checkpointed_iteration.txt` (historically `20`). Full-parameter FSDP1/world size 1, not LoRA. |
| `prepared_base` | Actual `models/llama3b/` model files and shard index, tokenizer, config, generation config, fixed template, model manifest and every retained license/notice; exclude incidental caches. Do not redownload or reconstruct these inputs. |

Inventory every run and submission, including empty/failed/partial ones, before approving scope. Use the manifest's `directories`, `missing_selections`, file list and exclusions to reconcile a ledger of **name, source Volume, recorded status, selected paths, measured bytes, and any omission reason/owner decision**. Do not infer success from a directory or silently omit extra runs. `prepare-01` and any other submissions belong in evidence. Raw records remain private; this utility is not a content sanitizer.

The separate existing `agent-lightning-validation` Volume holds validation records. Account explicitly for the historical image-01/02 build failures, image-03/04/05 worker failures, image-06 and gpu-01, and any other discovered validation/service attempts. Some builds failed before a worker/Volume receipt existed. Repository historical records and copied submission build evidence are required; identify any additional raw validation records, their availability and measured bytes in the A2 ledger, using a separate source manifest and namespace. Do not create a missing Volume or claim absent records were backed up.

Optional additions require measured bytes and owner selection: step-15 native state, smoke checkpoint tensors, and `data-source/calc-x-data.zip`. Steps 5/10 historically contain only `data.pt`; account for these incomplete remnants explicitly, never count them as complete checkpoints. Inventory existing `exports/`, staging remnants and other top-level paths and record their disposition; they are not implicitly selected. All optional/unselected source material stays in place.

The [historical ten-file step-20 table](../evidence/grpo-main-01.md#post-restoration-and-native-checkpoint) totals **40,146,407,568 bytes (~37.389 GiB)**. This excludes the marker and other groups and is neither a fresh inventory nor the entire backup size. Native tensor hashes remain unknown. The prepared-model manifest records weight names/total bytes and tokenizer/template hashes, **not independent hashes of weight files**. Refresh the inventory rather than using this subtotal as measured capacity.

| Provenance field | Recorded identity; keep distinct |
|---|---|
| `lab_execution_commit` | Base/main/post: `7cfddf58a0dece6a63ca67d18b103db2940e4aca`. Smoke/reload receipts instead identify `ee120371429a4bc23d11b9da53d166d2fa4843fd`. Read each `submissions/NAME/receipt.json` → `request.source.commit`. |
| `documentation_commit` | Rechecked `origin/main`: `377b721cb0719ef1bee3521fadf4621675f8e4ec`; PR #3 reviewed head `33cdaa137a72e05853b2a8762d6e669cf7515a8e` is **not** an execution revision. |
| `agent_lightning_commit` | v1.0.3, `cbc56e41e0eab643aa71d195e92e5bba652fa683`. This is what run-manifest `source_commit` identifies. |
| `preservation_implementation_commit` | Record the exact reviewed A1 PR head from Git in the A2 receipt; do not substitute it for execution provenance. |

PR #3's reviewed tree and current main's tree are `868772e629f398e705170835ac31149bf83dfef1`. The starting checkout was clean and no open PR existed at the A1 recheck. Preserve [historical verification records](CLOSEOUT_2026-11-13.md) as dated records. For the repository snapshot, later create a Git bundle containing the required commit histories, verify its object integrity and ability to restore those exact commits in a fresh directory, and inventory/hash the bundle under `evidence` in a separate explicit root. Do not scan a working `.git` directory or include credentials/unrelated local files.

Model identity is `meta-llama/Llama-3.2-3B-Instruct` at `0cb88a4f764b7a12671c53f0838cd831a0843b95`. Recorded prepared `tokenizer_config.json` SHA256 is `7fe399d0739ac28d562a04f0d294093d5d38ad3f00e6ea0e624f932ccf397116`; template SHA256 is `a718674808440b537a553c15e0557b5b45d6d76e49ad9ddbc1a9e92447843e2c` and fixes **11 Nov 2026**. These are historical expectations, not A1 measurements. Preserve exact data hashes from the [six-file frozen-input table](../evidence/grpo-smoke-01.md#provenance-and-evidence-locations) and original `data-small/data_manifest.json`. The [64-case CSV](../evidence/grpo-main-01-cases.csv) retains published identities, including `svamp__chal-520`; it cannot reconstruct frozen inputs or replace the original manifest. Seed 329 alone is insufficient.

## What the existing code does and does not preserve

Source inspection of [modal_app.py](../modal_app.py) confirms `export_evidence()` includes `runs/`, `evidence/`, `data-small/`, prior `submissions/`, and `models/*/model_manifest.json`. It skips checkpoint subtrees, its own submission directory, symlinks and hardlinks. It excludes model tensors and does not comprehensively retain prepared tokenizer/config files. Raw logs/traces and frozen questions/answers make it unsuitable as a sanitized public package.

The archive is created exclusively under `exports/` **on the same Volume**. Exclusive creation and its source SHA256 do not establish an independent verified backup. Interruption can leave a partial archive; preserve the export's terminal receipt separately. CPU export uses the existing heavy training image; CPU build charges remain possible and cache availability is not assumed. The new utility does not import that app, construct a Modal app or invoke its exporter.

The native `actor/huggingface/` directory is metadata, not an inference-ready model. [launch.py](../experiment/launch.py) accepts native checkpoints only for `reload-eval` and `smoke-reload-eval`; main training starts from base weights. Retained optimizer state does not establish supported or validated continued training. A tensor backup and an inference release have different contents.

## Offline utility contract

[scripts/preservation_manifest.py](../scripts/preservation_manifest.py) uses only Python 3.12's standard library on Linux/WSL. It has no default source root and no network operation. A1 runs it only on tiny synthetic fixtures. Future measured scans require separate authorization and quiescent artifacts. The filesystem root must be explicit, with a JSON selection such as this **future Volume scope**, not a command to run in A1:

```json
{
  "provenance": {
    "lab_execution_commit": "7cfddf58a0dece6a63ca67d18b103db2940e4aca",
    "documentation_commit": "377b721cb0719ef1bee3521fadf4621675f8e4ec",
    "agent_lightning_commit": "cbc56e41e0eab643aa71d195e92e5bba652fa683",
    "preservation_implementation_commit": null
  },
  "selections": [
    {"group": "evidence", "path": "runs"},
    {"group": "evidence", "path": "evidence"},
    {"group": "evidence", "path": "submissions"},
    {"group": "evidence", "path": "data-small"},
    {"group": "native_state", "path": "runs/train-01/checkpoints/global_step_20"},
    {"group": "native_state", "path": "runs/train-01/checkpoints/latest_checkpointed_iteration.txt"},
    {"group": "prepared_base", "path": "models/llama3b"}
  ]
}
```

This selection covers the primary Volume only; repository/validation evidence needs separately inventoried roots. Add per-run provenance from receipts rather than applying base/main/post provenance to smoke. Missing selections are findings, never implicit permission to create them. Evidence traversal records and skips checkpoint subtrees; the explicit native selection includes every file in step 20. All other files under selected directories are included unless a recorded name-policy exclusion applies. Reconcile every exclusion and the prepared manifest's weight list/license inventory before accepting scope.

The following CLI forms are **fixture examples**. `FIXTURE_ROOT`/`COPY_ROOT` must be explicitly prepared tiny directories; JSON inputs and new output files must be outside both source trees. The tests construct these automatically.

```bash
python scripts/preservation_manifest.py scan --root FIXTURE_ROOT --selection selection.json \
  --observation-class synthetic_observation --origin fixture-source --output source.json
python scripts/preservation_manifest.py scan --root COPY_ROOT --selection selection.json \
  --observation-class synthetic_observation --origin fixture-destination --output copy.json
python scripts/preservation_manifest.py compare --source source.json --destination copy.json --output comparison.json
python scripts/preservation_manifest.py inventory --input supplied-inventory.json --output normalized.json
```

Schema version 1 retains timestamp, provenance, file group/role, logical path, exact known bytes, SHA256, observation class/origin, hash basis, exclusions, directories, required-native-file findings and totals by group/class. With the same inputs and explicit timezone-bearing `--timestamp`, serialization is sorted and deterministic. Without that flag, it records current UTC. Use `historical_expectation`, `synthetic_observation` or future `measured_observation` honestly; origin identifies the actual report or source/destination readback. `scan` allows only the latter two classes and sets `hash_basis=bytes_read`.

Supplied inventory JSON is `{"schema_version":1,"timestamp":"...","provenance":{},"files":[...]}`. Each file requires `group`, `path`, `observation_class` and `observation_origin`; `bytes` and `sha256` may be null/absent. For example, a historical actor row has path `runs/train-01/checkpoints/global_step_20/actor/model_world_size_1_rank_0.pt`, group `native_state`, bytes `14427143047`, SHA256 null, class `historical_expectation`, origin `evidence/grpo-main-01.md`. Reported hashes are preserved but normalized as `hash_basis=reported`; they never become freshly measured. Unknown values stay null; `known_bytes` is only a subtotal, split by observation class.

Hashing streams 1 MiB buffers and treats `.pt` as opaque bytes. It checks file identity/type/size/mtime/ctime before and after reads, rejects detected mutation, and checks traversed directories for changes. This is **not an atomic snapshot** or protection against all concurrent changes. Only marker, FSDP config and model architecture metadata are interpreted for native structure. All ten recorded files must be nonempty, the marker must equal 20, and metadata must identify FSDP1/world size 1 and `LlamaForCausalLM`. Filename/structure checks do not establish tensor integrity, semantic equality or restore success.

Absolute/traversing/noncanonical member paths, duplicate paths, overlapping selections (except evidence checkpoint exclusion plus explicit native selection), symlinks (including ancestors), hardlinks and nonregular files are refused. Before reading or descending, a conservative name policy excludes `.env*`, common credential stores/config directories, tokens, keys and caches. Renamed secrets or sensitive content within ordinary logs are not detected. Keep raw evidence private and review the selection. Output parents must already exist; outputs are private, create-only and outside the source tree. Interrupted output writes may remain partial and must be retained/diagnosed with a new attempt name.

Comparison emits missing/extra paths, size/hash/group differences, unknown fields and unestablished independence. Exit codes: **0** produced inventory or fully known content match; **1** comparison mismatch/unknown/absent destination; **2** invalid input, unsafe path, mutation or output refusal. A scan exit 0 does not mean scope/native findings passed. Self-comparison with the same origin and reported/historical inventories cannot yield `content_match=true`. Different origins are labels, not proof of independence: a future verifier must read destination bytes itself, never copy a source manifest and relabel it. Every A1 output, including a successful synthetic comparison, keeps `backup_status=not_executed` and `backup_verified=false`. A later independent receipt must establish real verification.

## A2 sequence — proposed, not executed

1. **Confirm identity without provisioning.** After separate approval, confirm exact account/workspace, profile, environment and existing Volume identity with the owner/dashboard and read-only listings. Reuse suitable existing resources. Missing/wrong resources are a stop condition, never a reason to create them. Keep the existing gated-model read credential unchanged; do not inspect local credential stores or `.env` contents.
2. **Refresh metadata and scope.** Check `exports/` and `submissions/` before reserving a unique attempt name. Enumerate selected directory levels explicitly and reconcile the run/disposition ledger, required files, completion marker, manifests, retained license/notices and all scope totals. Only retrieve explicit small files to the client. Never pass a checkpoint/model directory or `/` to `volume get` on a laptop.
3. **After separately approved compute, hash source bytes remotely on CPU.** Record full selected-path inventories and provenance; preserve each source manifest independently. Use a small approved CPU process and bounded memory; no H100 is needed for ordinary hashing/copying. Confirm quiescence and available temporary space. Do not assume the heavy training image is cached or run its exporter as a shortcut. Check the installed Modal 1.6.3 API/source before choosing a read-only mount mechanism; do not assume current-doc options work or upgrade dependencies to obtain them.
4. **Transfer cloud to cloud.** Use the approved private, unique destination prefix with create-only/conditional writes and bounded timeout/retries. No provider implementation is included in A1. Copy only approved scope, preserve source bytes, and record partial transfers as incomplete. If an archive is chosen, budget its duplicate storage and record its source hash and safe member inventory.
5. **Read back destination bytes remotely.** Independently recompute per-file hashes, paths and exact sizes from destination storage in a fresh verifier. Compare against source manifests, including separately scoped repository/validation evidence. Check archive readability, safe/nonduplicate member inventory and member-byte hashes if applicable; this utility does not unpack archives. Copied hash metadata, ETags without a byte-hash contract, or upload-success responses are insufficient.
6. **Record terminal evidence.** Retain destination account/service/prefix, immutable object versions or repository commits, both inventories, exclusions, comparisons, archive checks, attempt IDs/timestamps/costs and terminal receipt outside the source Volume as well. Do not mark complete after partial transfer, missing scope, unknown hashes or mismatched readback. Keep failed-attempt records; no automatic destructive cleanup.
7. **Keep all sources unchanged.** No deletion is authorized, even after successful verification. No training, rescoring, inference, conversion or release is part of A2 backup approval.

Local-only CLI compatibility was checked with Modal **1.6.3** using `MODAL_CONFIG_PATH=/dev/null uv run --locked --offline modal volume {list,ls,get} --help` (each help command separately), without reading credentials or making account requests. The [current official Volume CLI](https://modal.com/docs/cli/latest/volume) was also consulted. The pinned help supports `--profile`, `--env`, `--json` for listing, and explicit file retrieval; **`volume ls` has no recursive option**. Help says a directory passed to `get` downloads recursively. Proposed read-only examples below require A2 authorization and exact owner-confirmed replacements:

```bash
uv run --locked modal volume list --profile APPROVED_PROFILE --env APPROVED_ENV --json
uv run --locked modal volume ls --profile APPROVED_PROFILE --env APPROVED_ENV cs329z-agent-lightning exports --json
uv run --locked modal volume ls --profile APPROVED_PROFILE --env APPROVED_ENV cs329z-agent-lightning submissions --json
uv run --locked modal volume ls --profile APPROVED_PROFILE --env APPROVED_ENV cs329z-agent-lightning runs --json
uv run --locked modal volume ls --profile APPROVED_PROFILE --env APPROVED_ENV cs329z-agent-lightning runs/train-01/checkpoints/global_step_20 --json
uv run --locked modal volume get --profile APPROVED_PROFILE --env APPROVED_ENV cs329z-agent-lightning runs/train-01/checkpoints/latest_checkpointed_iteration.txt -
uv run --locked modal volume get --profile APPROVED_PROFILE --env APPROVED_ENV cs329z-agent-lightning models/llama3b/model_manifest.json -
uv run --locked modal volume get --profile APPROVED_PROFILE --env APPROVED_ENV cs329z-agent-lightning data-small/data_manifest.json -
uv run --locked modal volume get --profile APPROVED_PROFILE --env APPROVED_ENV cs329z-agent-lightning submissions/train-01/receipt.json -
```

Continue listings one explicit subdirectory at a time (including `actor/`, `actor/huggingface/`); the example list is not a complete inventory. Retrieve other known small receipts/configs explicitly only after checking their sizes. Preserve the export terminal receipt separately if an old export exists.

### A2 approval block — fill before execution

Approval is staged: first authorize exact read-only metadata scope; then authorize CPU hashing with bounds; then approve transfer/readback using the resulting measured inventory. Unknowns below are **not authorization**.

| Required field | Current proposal / unresolved value |
|---|---|
| Exact source account owner, workspace, profile, environment, Volume IDs | **Unknown**; primary existing Volume name `cs329z-agent-lightning`; identify any secondary validation scope explicitly |
| Destination account owner and service | **Unknown; no destination selected** |
| Existing private destination and exact unique prefix; intended readers | **Unknown**; approve privacy and create-only behavior; no storage/repository creation or permission changes implied |
| Retention scope and duration | Recommend three required groups above; **owner choice pending** for optional step 15, smoke tensors, original ZIP and additional records; sources never deleted |
| Measured bytes, file counts, hash coverage, inventory identities | **Unknown per group and optional scope**; historical 40,146,407,568-byte subtotal is not a measurement |
| Temporary disk and CPU RAM/cores | **Unknown**; account for archive/copy/readback duplicates separately; CPU only |
| One-time build/compute/transfer/readback budget and currency | **Unknown**, including retries and verifier costs |
| Recurring storage budget, billing owner and duration | **Unknown**, including existing source storage and selected destination |
| Per-process/overall timeouts, maximum retries and stop limits | **Unknown**; no unbounded retries or automatic cleanup |
| Approved Secret references and exact scopes | **Unknown names**, no values; narrow source-read/destination-create/readback access; do not replace `agent-lightning-credentials` or its gated-model read credential |
| Approval record and terminal receipt location | **Unknown**; identify approver, authorized stage, immutable manifests and independent receipt destination |

Recheck [Modal pricing](https://modal.com/pricing), [network egress billing](https://modal.com/docs/guide/network-egress-billing) and the selected destination's current terms before approval. CPU, memory, builds, storage, egress and readback can all cost money. Published allowances are not evidence of remaining account capacity. No total cost can be justified until destination, measured scope and budgets are known.

### Failure diagnostics

| Finding | Required response |
|---|---|
| Wrong/missing workspace, environment, Volume or destination | Stop; confirm identities/authorization, create nothing |
| Missing/empty actor, optimizer, extra state, data, metadata or marker; marker not 20; wrong FSDP/world size | Retain inventory/findings; no structural-completeness claim; do not substitute step 15 or a data-only directory |
| Mutation, unreadable file, unsupported link/type or unsafe/duplicate path | Stop; retain diagnostic, establish quiescence or review selection; never follow links or deserialize `.pt` to bypass refusal |
| Missing prepared weights/license/input files or unexpected frozen hashes | Preserve mismatch; reconcile original manifests and exact files; no download, seed reconstruction or replacement |
| Existing output/prefix or partial archive/receipt | Preserve old attempt; inspect it and choose a fresh approved name, never overwrite/delete |
| Missing/extra files, differing size/hash, unknown hashes or absent destination | Incomplete/unverified; save both manifests and comparison and diagnose exact paths |
| Timeout, quota/budget exceeded, failed archive member readback | Stop within approved limits; retain partial/failed records and report outstanding bytes, never claim completion |

## Conversion and private release remain separate

The public VERL v0.8.0 source identity is `7aed6b230776f963fa09509c10d9c3a767d1102c`. The later supported entry point is `python -m verl.model_merger merge --backend fsdp`, with the native **actor directory** as `--local_dir`. Recheck the installed merger source identity before any separately authorized conversion. The pinned [FSDP merger](https://github.com/verl-project/verl/blob/7aed6b230776f963fa09509c10d9c3a767d1102c/verl/model_merger/fsdp_model_merger.py) casts tensors to BF16; the [base merger](https://github.com/verl-project/verl/blob/7aed6b230776f963fa09509c10d9c3a767d1102c/verl/model_merger/base_model_merger.py) handles tied aliases and tokenizer saving/normalization. These are source findings, not conversion validation.

Inspect actual native dtypes, keys/shapes, tying and token/template semantics first. If conversion changes precision or semantics, stop for a reviewed preservation-compatible path. Use a separate fresh output directory, an explicit Safetensors output contract, independently planned CPU RAM and temporary duplicate storage, and **omit `hf_upload_path`**. File hashes do not prove semantic tensor equivalence. Inference weights cannot replace optimizer/native-state backup.

Before a separately authorized private release, verify the retained license against [primary Meta Llama 3.2 terms](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE). Include the agreement and required Notice, display **Built with Llama**, use a model name beginning with **Llama**, and review acceptable use and dataset disclosure rights. Do not automatically put raw traces or frozen questions/answers in an inference repository.

A release needs its own approved Hub owner/repository/privacy, narrow publishing credential, exact export hashes, native-checkpoint/Git provenance, runtime recipe, clean-process inference/evaluation and immutable Hub commit. Validate retrieval into a fresh remote process/cache **without source Volume access**. Reuse the frozen harness/scorer for comparable evaluation and retain every output/score difference. Recheck [Hub storage](https://huggingface.co/docs/hub/en/storage-limits), [token scopes](https://huggingface.co/docs/hub/en/security-tokens) and [revision-specific downloads](https://huggingface.co/docs/huggingface_hub/en/guides/download); no Hub operation is performed here.

## Frozen result and fresh A1 checks

The [existing report](../evidence/grpo-main-01.md) remains authoritative: baseline/train-initial **27/64**, train-final **34/64**, post **35/64**; baseline/post **11 wins, 3 losses, 50 ties** (24 both correct, 26 both incorrect), with 64 completed rewarded episodes in each evaluation. The observed gain is **12.5 percentage points under the unchanged formatting-sensitive scorer**. Both `svamp__chal-520` outputs calculated 6; the train-final placeholder was selected by the first-match extractor. The mechanism is explained; changed wording remains unexplained. No extractor repair, rescoring or preferred-score rerun.

Retain 30/160 informative groups, six zero-signal iterations, 1394 collected/1280 trained/114 dropped rows, tool/parser errors and trace splitting, six dependency conflicts, experiment Protobuf-import ambiguity, unverified independent tensor identity and hosted training-run W&B readback, and descriptive timing limits. No claim of independent mathematical correctness, convergence, significance, causal/general improvement, validated training resume or completed portable release follows from preservation.

The historical qualified runtime remains Linux x86-64, one H100, 16 requested CPU cores and 128 GiB RAM, with VERL 0.8.0, vLLM 0.22.0, PyTorch 2.11.0+cu130, FlashAttention 2.8.3, AutoGen 0.7.5 and MCP calculator 0.2.1. The Windows 11/WSL2 Workbench/client setup and all pins are unchanged; local aarch64 tests do not qualify ARM64 GPU training.

A1 checks used CPython **3.12.3**, GCC 13.3.0, **aarch64**, `/workspace/projects/repos/agent-lightning-lab/.venv/bin/python3`, in the existing uv environment. The expected warning that Workbench's `/opt/venv` is ignored was emitted. The closeout's **74 tests are historical** and are not reused as A1 evidence.

| Fresh command | Final result |
|---|---|
| `uv run --locked --offline python -m unittest discover -s tests -p test_preservation_manifest.py -v` | **11 tests, 0.473 s, OK** |
| `MODAL_CONFIG_PATH=/dev/null HF_HUB_OFFLINE=1 uv run --locked --offline python -m unittest discover -s tests -v` | **85 tests, 5.888 s, OK** |
| `git diff --check` and documentation checks | Passed; local link targets, JSON example and Bash syntax checked without executing cloud examples |

The focused tests exercise production-shaped tiny fixtures: all three groups, failed-run evidence, each missing/empty native file, bad metadata/marker, data-only checkpoints, unsafe paths/links/types, duplicate selections, overwrite refusal, mutation, independent copy discrepancies, exact totals/order and unchanged source content/metadata. CLI tests also run with `python -S`, without site packages. They establish offline behavior only, not real artifact integrity, remote compatibility or backup independence.

Development history: the first focused run was interrupted when its mutation fixture appended indefinitely; the fixture now mutates once and the reader rejects growth beyond starting size. An intermediate focused run passed 11 tests in 0.506 s and full run passed 85 in 5.884 s. Final review preserved supplied exclusion records and tightened architecture-metadata typing; the final executions above include those changes. No further test expansion or cloud qualification followed. Historical evidence, scorer, pins, execution code and frozen inputs are unchanged.
