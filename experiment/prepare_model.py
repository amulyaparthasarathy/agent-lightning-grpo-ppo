#!/usr/bin/env python3
"""Download a pinned trainable checkpoint, or check gated access without weights.

No inference or GPU allocation. Credentials use HF_TOKEN or the normal local HF
credential store. The Llama working copy gets one fixed, versioned chat template.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

PROFILES = {
    "llama3b": {
        "repo_id": "meta-llama/Llama-3.2-3B-Instruct",
        "revision": "0cb88a4f764b7a12671c53f0838cd831a0843b95",
        "architecture": "LlamaForCausalLM",
    },
    "reference-qwen": {
        "repo_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "revision": None,
        "architecture": "Qwen2ForCausalLM",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def effective_template() -> str:
    path = Path(__file__).resolve().parent / "templates/tool_chat_template_llama3.1_json.jinja"
    # The upstream template otherwise inserts the current date, which would make
    # the prompt differ across Wednesday/Thursday/Friday evaluations.
    return '{%- set date_string = "11 Nov 2026" %}\n' + path.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="llama3b")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-access", action="store_true")
    args = parser.parse_args()
    if not args.check_access and args.output_dir is None:
        parser.error("--output-dir is required unless --check-access is used")

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    spec = PROFILES[args.profile]
    revision = spec["revision"] or HfApi().model_info(spec["repo_id"]).sha
    if not revision:
        raise RuntimeError("The model repository did not return an immutable revision")
    config_path = Path(hf_hub_download(spec["repo_id"], "config.json", revision=revision))
    config = json.loads(config_path.read_text())
    if spec["architecture"] not in config.get("architectures", []):
        raise RuntimeError("Unexpected model architecture; do not continue with this profile")
    if args.check_access:
        print(json.dumps({"status": "config_access_verified", "repo_id": spec["repo_id"],
                          "revision": revision, "architecture": spec["architecture"],
                          "weights_downloaded": False}, indent=2))
        return

    target = args.output_dir.expanduser().resolve()
    manifest_path = target / "model_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get("profile") != args.profile or old.get("revision") != revision:
            raise RuntimeError("Output contains a different prepared model; choose a new directory")
        missing = [name for name in old.get("weights_files", []) if not (target / name).is_file()]
        if missing:
            raise RuntimeError("Prepared model is incomplete; use a new directory")
        if sha256(target / "tokenizer_config.json") != old.get("prepared_tokenizer_config_sha256"):
            raise RuntimeError("Prepared tokenizer configuration changed; use a new directory")
        if args.profile == "llama3b":
            wanted = hashlib.sha256(effective_template().encode()).hexdigest()
            if sha256(target / "chat_template.jinja") != wanted or old.get("template_sha256") != wanted:
                raise RuntimeError("Prepared model template changed; use a new directory")
            if json.loads((target / "tokenizer_config.json").read_text()).get("chat_template") != effective_template():
                raise RuntimeError("Tokenizer and serving templates differ")
        print(f"Existing prepared model verified: {target}")
        return
    if target.exists() and any(target.iterdir()):
        raise RuntimeError("Output directory is nonempty without a manifest; choose a fresh directory")

    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=spec["repo_id"], revision=revision, local_dir=target,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja", "*.txt",
                        "LICENSE", "USE_POLICY.md", "README.md"],
        ignore_patterns=["original/*"],
    )
    tokenizer_path = target / "tokenizer_config.json"
    original_tokenizer_hash = sha256(tokenizer_path)
    template_hash = None
    if args.profile == "llama3b":
        tokenizer = json.loads(tokenizer_path.read_text())
        template = effective_template()
        tokenizer["chat_template"] = template
        tokenizer_path.write_text(json.dumps(tokenizer, ensure_ascii=False, indent=2) + "\n")
        (target / "chat_template.jinja").write_text(template, encoding="utf-8")
        template_hash = sha256(target / "chat_template.jinja")
    weights = sorted(str(path.relative_to(target)) for path in target.glob("*.safetensors"))
    if not weights:
        raise RuntimeError("No trainable Safetensors weights downloaded")
    manifest = {
        "profile": args.profile, "repo_id": spec["repo_id"], "revision": revision,
        "architecture": spec["architecture"], "prepared_at": datetime.now(timezone.utc).isoformat(),
        "original_tokenizer_config_sha256": original_tokenizer_hash,
        "prepared_tokenizer_config_sha256": sha256(tokenizer_path),
        "template_sha256": template_hash, "weights_files": weights,
        "weight_bytes": sum((target / name).stat().st_size for name in weights),
        "template_modification": "Fixed date and vLLM 0.22 Llama JSON template" if template_hash else None,
        "gpu_compatibility_verified": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"prepared_model": str(target), **manifest}, indent=2))


if __name__ == "__main__":
    main()
