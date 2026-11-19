# Preservation qualification and export evaluation

PR #6 introduced the gates and runtime interfaces described below. The owner has since approved the [worker continuation](PRESERVATION_EXECUTION.md), which adds transfer, verification, restoration, conversion and durable continuation code plus three review corrections. Read-only preflight has begun; no backup, tensor conversion, inference, release or replication has executed. The existing training Secret remains unchanged.

## Complete-scope backup qualification

The A1 byte comparator remains available and never sets `backup_verified=true`. [backup_qualification.py](../scripts/backup_qualification.py) adds a separate gate:

```bash
python scripts/backup_qualification.py --scope approved-scope.json \
  --sources source-manifests.json --destinations destination-manifests.json \
  --retrieval independent-retrieval-receipt.json --output qualification.json
```

Inputs must be explicit, trusted small JSON records. Output is create-only and outside source roots. `source-manifests.json` and `destination-manifests.json` map the approved namespace names to complete A1 manifests, with no namespace dropped. They are not source filesystem scans. The scope record requires:

| Field | Contract |
|---|---|
| `schema_version`, `approval_id`, `approved_by`, `remote_execution_approved` | Version 1; concrete scope approval identity; real verification requires `YES` |
| `primary_namespace`, `validation_namespace`, `repository_namespace` | Names explicitly present in `namespaces`; validation availability/omissions must be reconciled before approval |
| `native_checkpoint` | Defaults to the mandatory original `runs/train-01/checkpoints/global_step_20`; replication supplies its own fresh run's step-20 path explicitly |
| `namespaces.NAME.source_identity` | Concrete source identity, identical to that source manifest's provenance field; use confirmed account/environment/Volume IDs or repository snapshot identity |
| `namespaces.NAME.selections` | Exact approved group/path selections, matched on both sides |
| `namespaces.NAME.approved_files`, `required_files` | Reconciled full expected path inventory and required subset, not regenerated from a damaged copy |
| `namespaces.NAME.reviewed_exclusions` | Exact reviewed exclusion records; any change fails |
| `namespaces.NAME.expected_sha256` | Historical/approved identity anchors; primary scope must include all six frozen input files and prepared tokenizer-config/template hashes from retained evidence |
| `retained_notices`, `required_git_commits` | Required notice paths and execution/closeout/preservation/integration commits recovered from the selected Git bundle |
| `destination` | Exact private `repo_id`, completed immutable 40-hex `revision`, and unique `snapshot_prefix`; bind the actual uploaded revision into the final reconciled scope record |

The original primary selection must cover all non-checkpoint evidence and **all retained smoke/train checkpoints**, frozen data and prepared base. Select native `runs/smoke-01/checkpoints` and `runs/train-01/checkpoints` explicitly in addition to evidence `runs`. Include the source ZIP if present and within approved limits, plus other approved roots/records. A1 records every file in these trees. Only the chosen step-20 set receives the mandatory structural check; other directories must have their completeness/incomplete-remnant disposition recorded in the scope ledger. Directory existence never implies completeness. Original smoke names are not reused for the new replication.

The gate checks full approved file coverage, missing selections, reconciled exclusions, group/path/hash equality, per-namespace provenance, nonempty step-20 native files and structure/marker findings. It checks actual prepared weight filenames/byte totals against the model manifest and shard index; required metadata, retained notices, both tokenizer/template hashes and all six frozen data-file hashes must be covered. A required file absent on both sides fails even if A1 reports the remaining files match.

The independent retrieval receipt requires `method=destination_bytes`, `fresh_process=true`, `original_source_mounted=false`, a concrete `verifier_id` and independently retained `receipt_location`, and the same concrete destination. `source_manifests`/`destination_manifests` map namespace to the canonical JSON SHA256 from `backup_qualification.digest`; `scope_sha256` binds the scope. `git_bundle` contains selected bundle `path`/`sha256`, `verified=true`, and `restored_commits`. A later verifier must actually check bundle restoration and read destination bytes before producing these records.

Exit 0 means all checks passed on **measured byte observations with execution approved**. Exit 1 means incomplete/unverified or synthetic-only evidence; inspect `failures` and `scope_checks_passed`. Exit 2 means invalid input/path/output. Even a fully matching synthetic fixture cannot set `backup_verified=true`. The gate cannot authenticate a fabricated receipt or infer independent reads from labels alone. Do not copy source metadata into destination records. Real qualification depends on a trusted independent remote verifier, which is still to be run after approval. The gate is a coverage/receipt validator, not a new source of observed evidence.

## Platform-managed source roots

During approved preflight, inspect the actual `/data` mount and record its exact resolved target. A1 still refuses symlinked roots by default. If `/data` is a confirmed platform link, pass its runtime target explicitly:

```bash
python scripts/preservation_manifest.py scan --root /data \
  --approved-root-target CONFIRMED_RUNTIME_TARGET --selection approved-selection.json \
  --observation-class measured_observation --origin SOURCE_INVENTORY_ID \
  --output /approved-scratch/source-manifest.json
```

This is a future **remote CPU** command, not a client download. The final root component is resolved once and must equal the supplied target. Its binding is checked again after scanning. Symlinked ancestors and member symlinks/escapes remain rejected. All artifact paths stay logical and relative; `runtime_root_binding` is audit metadata, never a portable destination path. No internal Volume ID is hardcoded. Sources must be quiescent; these checks are not an atomic snapshot guarantee. The selection may include `native_checkpoint` for a fresh replication step 20; the original default remains unchanged.

## Trained export contract

[export_artifact.py](../experiment/export_artifact.py) validates a separately produced `export_manifest.json`. It does not load tensors or perform conversion. A base `model_manifest.json` inside an export is refused. The export contract contains:

- Version 1, `artifact_type=trained_full_parameter_export`, `profile=llama3b`, exact `base_repo_id` and `base_revision`, and the byte-inventory hash `prepared_base_manifest_sha256`.
- `source_checkpoint.path`, native actor `model_sha256`, and source native-inventory `manifest_sha256`. Original training is step 20; a replication has its own fresh step-20 path.
- `training_execution_commit` (historical original execution or actual replication commit), `integration_execution_commit`, `converter_commit`, and `converter_source_sha256`. Do not substitute upstream Agent Lightning's commit for any of these.
- Exact `harness_sha256`, three main frozen-file hashes in `frozen_data_sha256`, and prepared tokenizer-config/template hashes.
- `files`: relative inference filenames mapped to exact `bytes` and `sha256`, including all Safetensors shards, an index when sharded, required model/tokenizer/template/generation metadata and retained notices. Unknown non-inference members are refused. Raw traces and credentials do not belong here.
- `tensor_validation`: `all_tensors_equal=true`, `dtype_preserved=true`, positive `tensor_count`, explicit `validated_alias_map`, matching `source_model_sha256`, and `export_files_sha256` (SHA256 of UTF-8 `json.dumps(files, sort_keys=True, separators=(",", ":"))`). This must come from actual all-tensor reload/comparison in the trusted converter, not from a hand-authored success flag.

Byte validation ties evaluation to that converter receipt. It does not independently establish tensor equality, supported layouts or model behavior. The [worker continuation](PRESERVATION_EXECUTION.md) now implements a dtype-preserving CPU converter, installed-merger source inspection, all-tensor validation and private uploader/verifier. Those remote paths remain unqualified until actual execution; fixture tests are not tensor evidence. Frozen model/template semantics, licensing and budget gates still apply. No stock BF16 cast or merger upload shortcut is approved by this interface.

## Isolated evaluation and durable submissions

`modal_app.py` supports the explicit non-secret environment setting `AGL_WORKING_VOLUME=EXISTING_DISTINCT_VOLUME`. It must name an existing Volume distinct from `cs329z-agent-lightning`; implicit creation is disabled. The setting is propagated into the worker image, and a mismatch with the submitted Volume identity fails before storage access. Use this mode for restored-native validation, export evaluation and the fresh replication so the original source Volume is not mounted. `prepare` is refused in working mode: restore and verify the original inputs instead of downloading replacements.

After execution approval, resource/capacity checks, verified backup/restore and validated conversion, the export action has this form:

```bash
AGL_WORKING_VOLUME=APPROVED_EXISTING_WORKING_VOLUME uv run --locked modal run --detach modal_app.py \
  --action export-eval --name NEW_EXPORT_EVAL_NAME --model-profile llama3b \
  --export-path exports/VALIDATED_EXPORT_NAME --wandb-entity YOUR_WANDB_TEAM
```

`export-eval` requires that working mode. The launcher verifies export bytes/lineage, uses `val_only=true`, `resume_mode=disable` and no native checkpoint. It writes `export_manifest.json` into the run, sets policy identity `export:sha256:MANIFEST_HASH`, and preserves dynamic ports, service cleanup, harness, parser, generation bounds and the frozen held-out files. Effective runtime dtypes remain in the resolved configuration and must be checked in actual execution; storage dtype is a separate converter contract. CLI flags for direct local launcher execution are available via `python experiment/launch.py --help`, without importing Torch.

Pair recovered native and exported evaluations using explicit filters, for example `--phase export-eval --baseline-phase reload-eval --route-mode val` with [summarize.py](../experiment/summarize.py). The Modal `summarize` action exposes `--post-phase` and `--baseline-phase` and passes them explicitly. Its defaults remain reload-eval/baseline-eval for the original experiment. Preserve original per-case output traces and all score differences; no scorer repair or score-driven retry.

Every `modal_app.py` submission now requires `--detach`, calls `.spawn()` and immediately prints App/FunctionCall IDs. A local `.local/experiment-submissions/NAME-TIMESTAMP.json` records the source commit, requested Volume and expected durable receipt path. The worker claims/commits `submissions/NAME/receipt.json` before the operation, retains its FunctionCall ID, and writes terminal status on ordinary success/failure. Existing names remain refused. If local receipt writing fails after `SUBMITTED`, inspect that call instead of resubmitting.

From another authorized machine, use the saved profile/environment, App/FunctionCall IDs and explicit small receipt path to inspect completion. Modal 1.6.3 exposes `FunctionCall.from_id`; this API and `current_function_call_id` were inspected locally. No call was recovered remotely here. Hard termination can leave only `started`/partial evidence, which is not a pass. Already submitted detached work can survive client disconnect; no future phase is automatically submitted by a local script. This change intentionally introduces no multi-phase coordinator or unbounded retry loop. Resume at the next uncompleted gate only after checking durable receipts and the remaining budget ledger.

## Local checks and remaining execution

Tests use tiny synthetic filesystem copies, opaque fake weight bytes and mocked Modal dispatch. They cover matching-but-incomplete backups, namespace/exclusion/hash/receipt failures, approved root alias handling, explicit replication checkpoint identity, trained-export provenance and damaged/unlisted shards, validation-only/no-resume plans, export policy identity, explicit summary phase filters and durable detached submission behavior. They do not validate cloud accounts, real tensors, GPU inference or publication.

Fresh local execution on 13 November 2026 used CPython 3.12.3 on aarch64:

```bash
MODAL_CONFIG_PATH=/dev/null HF_HUB_OFFLINE=1 uv run --locked --offline python -m unittest discover -s tests -v
```

Result: **99 tests passed in 6.167 seconds**. The same command passed 85 tests for the separate documentation change before implementation. These are new local runs, distinct from the historical closeout's 74 tests and A1's previously recorded 85 tests. No ARM64 GPU qualification or independent cold build was performed. Original evidence is unchanged. Remote-stage receipts, resource durations/bytes/costs and Hub revisions remain absent. That earlier verification predates the newly approved worker continuation; consult its current status before execution.
