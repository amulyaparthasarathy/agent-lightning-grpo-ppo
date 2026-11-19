"""Validate a trained export's byte inventory/lineage without loading tensors.

All-tensor validation must be produced separately by the trusted CPU converter.
This validates that the export matches that receipt, not tensor semantics itself.
"""
import hashlib
import json
from pathlib import Path
import re

BASE_REVISION = "0cb88a4f764b7a12671c53f0838cd831a0843b95"
METADATA = {"config.json", "generation_config.json", "tokenizer.json",
            "tokenizer_config.json", "chat_template.jinja"}
OPTIONAL_FILES = {"model.safetensors.index.json", "special_tokens_map.json", "added_tokens.json",
                  "tokenizer.model", "LICENSE", "Notice", "NOTICE", "README.md", "USE_POLICY.md"}


def sha256(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            if size > before.st_size:
                raise ValueError("Export file grew while hashing")
            digest.update(chunk)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("Export file changed while hashing")
    return digest.hexdigest()


def regular(root, name):
    if (not isinstance(name, str) or not name or "/" in name or "\\" in name
            or name in {".", ".."} or ":" in name):
        raise ValueError("Export members must be plain file names")
    path = root / name
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError(f"Export member missing or linked: {name}")
    return path


def verify_export(root, data_manifest, harness_sha256):
    root = Path(root)
    manifest = json.loads(regular(root, "export_manifest.json").read_text())
    if (manifest.get("schema_version") != 1 or manifest.get("artifact_type") != "trained_full_parameter_export"
            or manifest.get("base_revision") != BASE_REVISION or manifest.get("profile") != "llama3b"
            or manifest.get("base_repo_id") != "meta-llama/Llama-3.2-3B-Instruct"):
        raise ValueError("Expected a trained Llama export manifest, not prepared-base metadata")
    if (root / "model_manifest.json").exists():
        raise ValueError("Trained export must not masquerade as a prepared base")
    source = manifest.get("source_checkpoint", {})
    if (not re.fullmatch(r"runs/[A-Za-z0-9][A-Za-z0-9._-]*/checkpoints/global_step_20", source.get("path", ""))
            or not re.fullmatch(r"[0-9a-f]{64}", source.get("model_sha256", ""))
            or not re.fullmatch(r"[0-9a-f]{64}", source.get("manifest_sha256", ""))
            or not re.fullmatch(r"[0-9a-f]{64}", manifest.get("prepared_base_manifest_sha256", ""))):
        raise ValueError("Missing native checkpoint/model identity")
    for key in ("training_execution_commit", "integration_execution_commit", "converter_commit"):
        if not re.fullmatch(r"[0-9a-f]{40}", manifest.get(key, "")):
            raise ValueError(f"Missing export provenance: {key}")
    if not re.fullmatch(r"[0-9a-f]{64}", manifest.get("converter_source_sha256", "")):
        raise ValueError("Missing converter source identity")
    receipt = manifest.get("tensor_validation", {})
    if (receipt.get("all_tensors_equal") is not True or receipt.get("dtype_preserved") is not True
            or receipt.get("source_model_sha256") != source["model_sha256"]
            or type(receipt.get("tensor_count")) is not int or receipt["tensor_count"] <= 0
            or not isinstance(receipt.get("validated_alias_map"), dict)):
        raise ValueError("Missing all-tensor/dtype/alias validation receipt")
    if manifest.get("harness_sha256") != harness_sha256:
        raise ValueError("Export and evaluation harness differ")
    for name in ("train_main.parquet", "eval_main.parquet", "eval_main.jsonl"):
        expected = data_manifest["files"][name]["sha256"]
        if manifest.get("frozen_data_sha256", {}).get(name) != expected:
            raise ValueError(f"Export frozen-data identity differs: {name}")
    files = manifest["files"]
    actual = {p.name for p in root.iterdir()}
    if actual != set(files) | {"export_manifest.json"}:
        raise ValueError("Unlisted or missing export member")
    for name in actual:
        regular(root, name)
    if not METADATA <= files.keys():
        raise ValueError("Export metadata inventory incomplete")
    weights = {name for name in files if name.endswith(".safetensors")}
    if not weights:
        raise ValueError("Export has no Safetensors weights")
    if set(p.name for p in root.glob("*.safetensors")) != weights:
        raise ValueError("Unlisted or missing export shard")
    for name, expected in files.items():
        if name not in METADATA | OPTIONAL_FILES and not name.endswith(".safetensors"):
            raise ValueError(f"File outside inference export contract: {name}")
        path = regular(root, name)
        if (type(expected.get("bytes")) is not int or expected["bytes"] <= 0
                or path.stat().st_size != expected["bytes"] or sha256(path) != expected.get("sha256")):
            raise ValueError(f"Export byte identity differs: {name}")
    if weights != {"model.safetensors"} or "model.safetensors.index.json" in files:
        if "model.safetensors.index.json" not in files:
            raise ValueError("Export shard index missing from inventory")
        index = json.loads(regular(root, "model.safetensors.index.json").read_text())
        if set(index.get("weight_map", {}).values()) != weights:
            raise ValueError("Export shard index coverage differs")
    if receipt.get("export_files_sha256") != hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest():
        raise ValueError("All-tensor receipt is not bound to these exported bytes")
    tokenizer = json.loads(regular(root, "tokenizer_config.json").read_text())
    template = regular(root, "chat_template.jinja").read_text()
    if (tokenizer.get("chat_template") != template
            or files["chat_template.jinja"]["sha256"] != manifest.get("template_sha256")
            or files["tokenizer_config.json"]["sha256"] != manifest.get("prepared_tokenizer_config_sha256")):
        raise ValueError("Export tokenizer/template semantics differ")
    if "LlamaForCausalLM" not in json.loads(regular(root, "config.json").read_text()).get("architectures", []):
        raise ValueError("Export architecture differs")
    return manifest
