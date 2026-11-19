#!/usr/bin/env python3
"""Qualify explicit approved scope against independently read byte inventories.

No SDK or source reads. Receipts are evidence inputs, not cryptographic proof of
their origin. A trusted remote verifier must produce them from destination bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
import re

try:
    from scripts import preservation_manifest as pm
except ModuleNotFoundError:
    import preservation_manifest as pm

BASE = "models/llama3b/"
DATA = "data-small/"
BASE_METADATA = ("config.json", "generation_config.json", "tokenizer.json",
                 "tokenizer_config.json", "chat_template.jinja", "model_manifest.json")
DATA_FILES = ("train_main.parquet", "eval_main.parquet", "eval_main.jsonl",
              "train_smoke.parquet", "eval_smoke.parquet", "eval_smoke.jsonl")
PROVENANCE = ("lab_execution_commit", "documentation_commit", "agent_lightning_commit",
              "preservation_implementation_commit")
EXPECTED_BASE = {"profile": "llama3b", "repo_id": "meta-llama/Llama-3.2-3B-Instruct",
                 "revision": "0cb88a4f764b7a12671c53f0838cd831a0843b95",
                 "architecture": "LlamaForCausalLM"}
ORIGINAL_LINEAGE = {f"submissions/{run}/receipt.json": commit for run, commit in (
    ("base-01", "7cfddf58a0dece6a63ca67d18b103db2940e4aca"),
    ("train-01", "7cfddf58a0dece6a63ca67d18b103db2940e4aca"),
    ("post-01", "7cfddf58a0dece6a63ca67d18b103db2940e4aca"),
    ("smoke-01", "ee120371429a4bc23d11b9da53d166d2fa4843fd"),
    ("smoke-reload-01", "ee120371429a4bc23d11b9da53d166d2fa4843fd"))}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def selections(value):
    return sorted((item["group"], pm.member(item["path"])) for item in value)


def qualify(scope, sources, destinations, retrieval, timestamp):
    """Return diagnostics; synthetic success never qualifies a real backup."""
    failures, comparisons = [], {}
    def need(condition, reason):
        if not condition:
            failures.append(reason)

    need(scope.get("schema_version") == 1, "unsupported scope schema")
    need(bool(scope.get("approval_id")) and bool(scope.get("approved_by")), "missing scope approval identity")
    approved = scope["namespaces"]
    need(scope.get("expected_base") == EXPECTED_BASE, "approved base identity missing or unsupported")
    need(bool(approved), "empty scope")
    need(set(approved) == set(sources) == set(destinations), "source/destination namespace mismatch")
    need(scope.get("primary_namespace") in approved, "missing primary namespace")
    need(scope.get("repository_namespace") in approved, "missing repository namespace")
    need(scope.get("validation_namespace") in approved, "unreconciled validation namespace")
    real = True
    for namespace, specification in sorted(approved.items()):
        pm.member(namespace)
        source, destination = sources.get(namespace), destinations.get(namespace)
        if source is None or destination is None:
            continue
        expected = {pm.member(p) for p in specification["approved_files"]}
        need(len(expected) == len(specification["approved_files"]), f"{namespace}: duplicate approved paths")
        required = {pm.member(p) for p in specification["required_files"]}
        need(bool(required) and required <= expected, f"{namespace}: required scope omitted")
        for label, manifest in (("source", source), ("destination", destination)):
            rows = pm.normalize(manifest["files"])
            indexed = {r["path"]: r for r in rows}
            need(set(indexed) == expected, f"{namespace}/{label}: approved file coverage differs")
            need(not manifest.get("missing_selections", ["unknown"]), f"{namespace}/{label}: missing selections")
            need(selections(manifest.get("selections", [])) == selections(specification["selections"]),
                 f"{namespace}/{label}: selections differ")
            need(bool(specification["selections"]), f"{namespace}: empty selections")
            need(manifest.get("exclusions") == specification["reviewed_exclusions"],
                 f"{namespace}/{label}: exclusions not reconciled")
            need(all(manifest.get("provenance", {}).get(k) for k in PROVENANCE),
                 f"{namespace}/{label}: incomplete provenance")
            expected_provenance = specification.get("expected_provenance", {})
            need(all(k in expected_provenance for k in PROVENANCE), f"{namespace}: expected provenance missing")
            need(all(manifest.get("provenance", {}).get(k) == v for k, v in expected_provenance.items()),
                 f"{namespace}/{label}: provenance identity differs")
            for path, commit in specification.get("run_lineage", {}).items():
                receipt = manifest.get("checked_metadata", {}).get(path, {})
                need(path in indexed and receipt.get("request", {}).get("source", {}).get("commit") == commit,
                     f"{namespace}/{label}: run execution lineage differs: {path}")
            need(bool(rows) and all(r["bytes"] is not None and r["sha256"] is not None
                 and r.get("hash_basis") == "bytes_read" and r["observation_class"] != "historical_expectation"
                 for r in rows), f"{namespace}/{label}: missing byte-derived observations")
            real = real and bool(rows) and all(r["observation_class"] == "measured_observation" for r in rows)
            for path, sha in specification.get("expected_sha256", {}).items():
                need(indexed.get(path, {}).get("sha256") == sha, f"{namespace}/{label}: frozen identity differs: {path}")
        need(bool(specification.get("source_identity")), f"{namespace}: missing source identity")
        need(source.get("provenance", {}).get("source_identity") == specification.get("source_identity"),
             f"{namespace}: source identity mismatch")
        comparison = pm.compare(source, destination, timestamp)
        comparisons[namespace] = comparison
        need(comparison["content_match"], f"{namespace}: independent byte comparison failed")
        need(retrieval.get("source_manifests", {}).get(namespace) == digest(source),
             f"{namespace}: source manifest receipt binding differs")
        need(retrieval.get("destination_manifests", {}).get(namespace) == digest(destination),
             f"{namespace}: destination manifest receipt binding differs")

    primary = scope.get("primary_namespace")
    checkpoint = pm.member(scope.get("native_checkpoint", pm.CHECKPOINT))
    need(bool(re.fullmatch(r"runs/[A-Za-z0-9][A-Za-z0-9._-]*/checkpoints/global_step_20", checkpoint)),
         "approved checkpoint is not step 20")
    marker = checkpoint.rsplit("/", 1)[0] + "/latest_checkpointed_iteration.txt"
    if checkpoint == pm.CHECKPOINT:
        lineage = approved.get(primary, {}).get("run_lineage", {})
        need(all(lineage.get(path) == commit for path, commit in ORIGINAL_LINEAGE.items()),
             "original run lineage expectations incomplete or changed")
    expected_inputs = set(DATA + n for n in DATA_FILES) | {
        BASE + "tokenizer_config.json", BASE + "chat_template.jinja"}
    need(expected_inputs <= set(approved.get(primary, {}).get("expected_sha256", {})),
         "historical frozen-input expectations missing from approved scope")
    for label, collection in (("source", sources), ("destination", destinations)):
        manifest = collection.get(primary, {})
        rows = {r["path"]: r for r in pm.normalize(manifest.get("files", []))}
        need(manifest.get("required_file_findings", {}).get("structurally_complete") is True,
             f"{label}: selected step-20 structure/marker invalid")
        # Independently require paths even when a copied finding claims completeness.
        need(manifest.get("native_checkpoint", pm.CHECKPOINT) == checkpoint,
             f"{label}: native finding is for a different run")
        native = [f"{checkpoint}/{p}" for p in pm.NATIVE_FILES] + [marker]
        for path in native:
            need(rows.get(path, {}).get("group") == "native_state" and (rows.get(path, {}).get("bytes") or 0) > 0,
                 f"{label}: required native file absent/empty: {path}")
        metadata = manifest.get("checked_metadata", {})
        model = metadata.get(BASE + "model_manifest.json", {})
        for key in EXPECTED_BASE:
            need(model.get(key) == EXPECTED_BASE[key], f"{label}: prepared base identity differs: {key}")
        need(metadata.get(BASE + "config.json", {}).get("architectures") == [EXPECTED_BASE["architecture"]],
             f"{label}: prepared base architecture differs")
        data = metadata.get(DATA + "data_manifest.json", {})
        weights = model.get("weights_files", [])
        need(bool(weights) and len(weights) == len(set(weights)), f"{label}: missing/duplicate base shard list")
        observed_weights = {p[len(BASE):] for p in rows if p.startswith(BASE) and p.endswith(".safetensors")}
        need(observed_weights == set(weights), f"{label}: unlisted/missing prepared weight shard")
        need(type(model.get("weight_bytes")) is int and model["weight_bytes"] ==
             sum(rows.get(BASE + n, {}).get("bytes") or 0 for n in weights), f"{label}: prepared weight bytes differ")
        for name in (*BASE_METADATA, *weights):
            path = BASE + pm.member(name)
            need(rows.get(path, {}).get("group") == "prepared_base" and (rows.get(path, {}).get("bytes") or 0) > 0,
                 f"{label}: prepared-base coverage missing: {path}")
        index = metadata.get(BASE + "model.safetensors.index.json")
        if len(weights) > 1 or index is not None:
            need(bool(index) and set(index.get("weight_map", {}).values()) == set(weights),
                 f"{label}: base shard index coverage differs")
            need(BASE + "model.safetensors.index.json" in rows, f"{label}: base shard index absent")
        for name, field in (("tokenizer_config.json", "prepared_tokenizer_config_sha256"),
                            ("chat_template.jinja", "template_sha256")):
            need(bool(model.get(field)) and rows.get(BASE + name, {}).get("sha256") == model.get(field),
                 f"{label}: prepared tokenizer/template identity differs")
        need(DATA + "data_manifest.json" in rows, f"{label}: frozen manifest absent")
        for name in DATA_FILES:
            entry = data.get("files", {}).get(name, {})
            need(bool(entry.get("sha256")) and rows.get(DATA + name, {}).get("sha256") == entry.get("sha256"),
                 f"{label}: frozen data coverage/hash differs: {name}")
        need(bool(scope.get("retained_notices")), "retained notices not reconciled")
        for path in scope.get("retained_notices", []):
            need(pm.member(path) in rows, f"{label}: required notice absent: {path}")

    target = retrieval.get("destination", {})
    need(target == scope.get("destination"), "destination differs from approved identity")
    need(target.get("private") is True and bool(target.get("repo_id"))
         and bool(re.fullmatch(r"[0-9a-f]{40}", target.get("revision", "")))
         and bool(target.get("snapshot_prefix")), "destination identity/privacy/immutable revision missing")
    need(retrieval.get("method") == "destination_bytes" and retrieval.get("fresh_process") is True
         and retrieval.get("original_source_mounted") is False, "readback not isolated independent destination reads")
    need(bool(retrieval.get("verifier_id")) and bool(retrieval.get("receipt_location")), "retrieval evidence missing")
    need(retrieval.get("scope_sha256") == digest(scope), "approved scope receipt binding differs")
    bundle = retrieval.get("git_bundle", {})
    need(bundle.get("verified") is True and bool(bundle.get("restored_commits")), "repository recovery unverified")
    repo_rows = {r["path"]: r for r in sources.get(scope.get("repository_namespace"), {}).get("files", [])}
    need(bool(bundle.get("sha256")) and repo_rows.get(bundle.get("path"), {}).get("sha256") == bundle.get("sha256"),
         "repository bundle verification not bound to selected bytes")
    need(set(scope.get("required_git_commits", [])) <= set(bundle.get("restored_commits", []))
         and bool(scope.get("required_git_commits")), "required repository history not restored")
    passed = not failures
    verified = passed and real and scope.get("remote_execution_approved") == "YES"
    return dict(schema_version=1, timestamp=timestamp, scope_sha256=digest(scope),
                scope_checks_passed=passed, failures=sorted(set(failures)), comparisons=comparisons,
                observation_class="measured_observation" if real else "synthetic_or_unknown",
                backup_verified=verified, backup_status="verified" if verified else "not_verified",
                approval=scope.get("approval_id"), destination=target, retrieval_evidence=retrieval,
                provenance={n: m.get("provenance") for n, m in sources.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scope", "sources", "destinations", "retrieval", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    try:
        values = [pm.read_json(getattr(args, key)) for key in ("scope", "sources", "destinations", "retrieval")]
        result = qualify(*values, datetime.now(timezone.utc).isoformat())
        roots = [m["source_root"] for collection in values[1:3] for m in collection.values() if m.get("source_root")]
        pm.write_output(args.output, result, roots)
        return 0 if result["backup_verified"] else 1
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as exc:
        parser.exit(2, f"qualification: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
