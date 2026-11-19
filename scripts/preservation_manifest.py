#!/usr/bin/env python3
"""Offline, opaque-byte preservation inventory/comparison. See docs/PRESERVATION.md.

No cloud SDK, credentials, Torch, deserialization, transfer or backup attestation.
Sources must be quiescent; metadata checks are not an atomic snapshot guarantee.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat

GROUPS = ("evidence", "native_state", "prepared_base")
CLASSES = ("historical_expectation", "synthetic_observation", "measured_observation")
CHECKPOINT = "runs/train-01/checkpoints/global_step_20"
MARKER = "runs/train-01/checkpoints/latest_checkpointed_iteration.txt"
NATIVE_FILES = (
    "actor/model_world_size_1_rank_0.pt", "actor/optim_world_size_1_rank_0.pt",
    "actor/extra_state_world_size_1_rank_0.pt", "data.pt", "actor/fsdp_config.json",
    "actor/huggingface/config.json", "actor/huggingface/generation_config.json",
    "actor/huggingface/tokenizer.json", "actor/huggingface/tokenizer_config.json",
    "actor/huggingface/chat_template.jinja",
)
METADATA = {MARKER, f"{CHECKPOINT}/actor/fsdp_config.json",
            f"{CHECKPOINT}/actor/huggingface/config.json",
            "models/llama3b/model_manifest.json", "models/llama3b/model.safetensors.index.json",
            "models/llama3b/config.json",
            "data-small/data_manifest.json"}
BUFFER = 1024 * 1024


def member(value):
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or any(ord(c) < 32 for c in value)
            or any(p in ("", ".", "..") for p in value.split("/"))
            or PurePosixPath(value).is_absolute()):
        raise ValueError(f"unsafe member path: {value!r}")
    return value


def exclusion(path):
    # Conservative name-based exclusions, before stat/open/descent. Not sanitization.
    for part in PurePosixPath(path).parts:
        name = part.lower()
        if (name.startswith(".env") or name in {
                ".git", ".ssh", ".aws", ".azure", ".config", ".local", ".gnupg",
                ".kube", ".docker", ".netrc", "_netrc", ".git-credentials",
                "modal.toml", ".modal.toml", "cache", ".cache", "__pycache__",
                ".venv", "venv", "token", "tokens", "stored_tokens",
            } or "credential" in name or "secret" in name
                or name.startswith(("token.", "token_", "access_token", "refresh_token"))
                or name.endswith((".pem", ".key"))):
            return "credential/token/cache exclusion (name policy; bytes not read)"
    return None


def signature(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


@contextmanager
def directory(path):
    """Open each directory component without following symlinks (Linux/WSL)."""
    path = Path(path).absolute()
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            if part == "..":
                raise ValueError("parent traversal in filesystem root")
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


def read_file(parent, name, capture=False, limit=None):
    before = os.stat(name, dir_fd=parent, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError(f"unsupported file type or hardlink: {name}")
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    with os.fdopen(fd, "rb") as handle:
        if signature(before) != signature(os.fstat(handle.fileno())):
            raise ValueError(f"source mutation before read: {name}")
        digest, size, chunks = hashlib.sha256(), 0, []
        while chunk := handle.read(BUFFER):
            size += len(chunk)
            if size > before.st_size:
                raise ValueError(f"source mutation during read: {name}")
            if limit is not None and size > limit:
                raise ValueError(f"metadata too large: {name}")
            digest.update(chunk)
            if capture:
                chunks.append(chunk)
        if (size != before.st_size or signature(before) != signature(os.fstat(handle.fileno()))
                or signature(before) != signature(os.stat(name, dir_fd=parent, follow_symlinks=False))):
            raise ValueError(f"source mutation during read: {name}")
    return size, digest.hexdigest(), b"".join(chunks)


def read_json(path):
    path = Path(path).absolute()
    if exclusion(path.as_posix()):
        raise ValueError("excluded JSON input path")
    with directory(path.parent) as fd:
        return json.loads(read_file(fd, path.name, capture=True, limit=64 * BUFFER)[2])


def normalize(entries):
    rows, seen = [], set()
    for entry in entries:
        row = dict(entry)
        path = member(row["path"])
        if path in seen:
            raise ValueError(f"duplicate path: {path}")
        seen.add(path)
        if row["group"] not in GROUPS or row["observation_class"] not in CLASSES:
            raise ValueError(f"invalid group/observation class: {path}")
        if not isinstance(row["observation_origin"], str) or not row["observation_origin"].strip():
            raise ValueError(f"missing observation origin: {path}")
        size, sha = row.get("bytes"), row.get("sha256")
        if size is not None and (type(size) is not int or size < 0):
            raise ValueError(f"invalid byte count: {path}")
        if sha is not None and (not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha)):
            raise ValueError(f"invalid SHA256: {path}")
        row.update(bytes=size, sha256=sha)
        rows.append(row)
    return sorted(rows, key=lambda row: row["path"])


def native_findings(rows, metadata, checkpoint=CHECKPOINT):
    marker = checkpoint.rsplit("/", 1)[0] + "/latest_checkpointed_iteration.txt"
    indexed = {row["path"]: row for row in rows if row["group"] == "native_state"}
    required = sorted([f"{checkpoint}/{p}" for p in NATIVE_FILES] + [marker])
    missing = [p for p in required if p not in indexed]
    unknown = [p for p in required if p in indexed and indexed[p]["bytes"] is None]
    empty = [p for p in required if p in indexed and indexed[p]["bytes"] == 0]
    checks = {"marker_is_20": None, "fsdp1_world_size1": None, "llama_architecture": None}
    try:
        if marker in metadata:
            checks["marker_is_20"] = metadata[marker].decode().strip() == "20"
        path = f"{checkpoint}/actor/fsdp_config.json"
        if path in metadata:
            config = json.loads(metadata[path])
            version = config.get("FSDP_version", config.get("fsdp_version"))
            checks["fsdp1_world_size1"] = (type(version) is int and version == 1
                and config.get("fsdp_version", version) == version and type(config.get("world_size")) is int
                and config["world_size"] == 1)
        path = f"{checkpoint}/actor/huggingface/config.json"
        if path in metadata:
            architectures = json.loads(metadata[path]).get("architectures", [])
            checks["llama_architecture"] = isinstance(architectures, list) and "LlamaForCausalLM" in architectures
    except (ValueError, AttributeError, TypeError):
        checks["invalid_metadata"] = True
    complete = (not missing and not unknown and not empty and "invalid_metadata" not in checks
                and all(value is True for value in checks.values()))
    return dict(required=required, missing=missing, unknown_sizes=unknown, empty=empty,
                metadata_checks=checks, structurally_complete=complete,
                tensor_correctness="unverified", continued_training="unsupported_by_lab_launcher")


def manifest(entries, provenance, timestamp, exclusions=(), metadata=None, **extra):
    rows = normalize(entries)
    totals = {}
    for group in GROUPS:
        selected = [row for row in rows if row["group"] == group]
        totals[group] = dict(files=len(selected), known_bytes=sum(r["bytes"] or 0 for r in selected),
                             unknown_sizes=sum(r["bytes"] is None for r in selected),
                             unknown_hashes=sum(r["sha256"] is None for r in selected))
        totals[group]["by_observation_class"] = {
            kind: dict(files=sum(r["observation_class"] == kind for r in selected),
                       known_bytes=sum(r["bytes"] or 0 for r in selected if r["observation_class"] == kind))
            for kind in CLASSES}
    return dict(schema_version=1, timestamp=timestamp, provenance=provenance, files=rows,
                exclusions=sorted(exclusions, key=lambda item: item["path"]), totals_by_group=totals,
                required_file_findings=native_findings(rows, metadata or {}, extra.get("native_checkpoint", CHECKPOINT)),
                backup_status="not_executed", backup_verified=False, **extra)


def scan(root, selection, observation_class, origin, timestamp, approved_root_target=None):
    if observation_class not in CLASSES[1:]:
        raise ValueError("filesystem reads must be synthetic or measured observations")
    checkpoint = member(selection.get("native_checkpoint", CHECKPOINT))
    if not re.fullmatch(r"runs/[A-Za-z0-9][A-Za-z0-9._-]*/checkpoints/global_step_20", checkpoint):
        raise ValueError("native_checkpoint must explicitly select a run's global_step_20")
    marker = checkpoint.rsplit("/", 1)[0] + "/latest_checkpointed_iteration.txt"
    metadata_paths = METADATA | {marker, f"{checkpoint}/actor/fsdp_config.json",
                                f"{checkpoint}/actor/huggingface/config.json"}
    metadata_paths |= set(selection.get("provenance_records", []))
    for path in metadata_paths:
        member(path)
    selections = selection["selections"]
    excluded_links = {member(p) for p in selection.get('excluded_symlinks', [])}
    if not selections:
        raise ValueError("explicit nonempty selection required")
    chosen = []
    for item in selections:
        path, group = member(item["path"]), item["group"]
        if group not in GROUPS:
            raise ValueError(f"invalid group: {group}")
        for previous, previous_group in chosen:
            if path == previous:
                raise ValueError(f"duplicate selection: {path}")
            ancestor, descendant, ancestor_group = (previous, path, previous_group)
            if previous.startswith(path + "/"):
                ancestor, descendant, ancestor_group = path, previous, group
            if descendant.startswith(ancestor + "/") and not (
                    ancestor_group == "evidence"
                    and "checkpoints" in PurePosixPath(descendant[len(ancestor) + 1:]).parts):
                raise ValueError(f"overlapping selection: {path}")
        chosen.append((path, group))
    rows, excluded, metadata, visited, missing_selections = [], [], {}, [], []

    def walk(parent, name, path, group):
        reason = exclusion(path)
        if group == "evidence" and "checkpoints" in PurePosixPath(path).parts:
            reason = "checkpoint subtree outside evidence group; select native state separately"
        if reason:
            excluded.append(dict(path=path, reason=reason))
            return
        info = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if path in excluded_links:
            if group != 'evidence' or not stat.S_ISLNK(info.st_mode):
                raise ValueError(f"reviewed exclusion is not an evidence symlink: {path}")
            excluded.append(dict(path=path, reason='reviewed evidence symlink excluded; target not read'))
            return
        if stat.S_ISDIR(info.st_mode):
            fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            try:
                if signature(info) != signature(os.fstat(fd)):
                    raise ValueError(f"source directory mutation: {path}")
                visited.append(path)
                for child in sorted(os.listdir(fd)):
                    walk(fd, child, member(f"{path}/{child}"), group)
                if (signature(info) != signature(os.fstat(fd))
                        or signature(info) != signature(os.stat(name, dir_fd=parent, follow_symlinks=False))):
                    raise ValueError(f"source directory mutation: {path}")
            finally:
                os.close(fd)
        else:
            size, sha, content = read_file(parent, name, path in metadata_paths,
                                           BUFFER if path in metadata_paths else None)
            if path in metadata_paths:
                metadata[path] = content
            rows.append(dict(group=group, path=path, bytes=size, sha256=sha,
                             observation_class=observation_class, observation_origin=origin,
                             hash_basis="bytes_read"))

    # Each selected parent is opened component by component, never through a link.
    logical_root = Path(root).absolute()
    root = logical_root
    if exclusion(root.as_posix()):
        raise ValueError("excluded source root")
    # Only an explicitly confirmed final mount component may be a platform link.
    # Member links and symlinked ancestors remain forbidden.
    with directory(root.parent):
        root_before = root.lstat()
    if root.is_symlink():
        if approved_root_target is None:
            raise ValueError("source root symlink requires approved runtime target")
        resolved = root.resolve(strict=True)
        if resolved != Path(approved_root_target).absolute():
            raise ValueError("source root differs from approved runtime target")
        root = resolved
    elif approved_root_target is not None and root != Path(approved_root_target).absolute():
        raise ValueError("source root differs from approved runtime target")
    if exclusion(root.as_posix()):
        raise ValueError("excluded runtime root target")
    with directory(root):
        for item in selections:
            path = item["path"]
            if exclusion(path):
                excluded.append(dict(path=path, reason=exclusion(path)))
                continue
            target = root / path
            try:
                with directory(target.parent) as fd:
                    os.stat(target.name, dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError:
                missing_selections.append(path)
                continue
            with directory(target.parent) as fd:
                walk(fd, target.name, path, item["group"])
    if signature(root_before) != signature(logical_root.lstat()):
        raise ValueError("source root binding changed during scan")
    checked_metadata = {}
    for path, content in metadata.items():
        if path.endswith(".json"):
            try:
                checked_metadata[path] = json.loads(content)
            except ValueError:
                pass  # Invalid metadata cannot qualify coverage later.
    return manifest(rows, selection.get("provenance", {}), timestamp, excluded, metadata,
                    source_root=str(logical_root), selections=selections, directories=sorted(visited),
                    checked_metadata=checked_metadata, native_checkpoint=checkpoint,
                    runtime_root_binding=dict(logical=str(logical_root), resolved=str(root)),
                    missing_selections=sorted(missing_selections))


def supplied(inventory, timestamp):
    """Keep reported hashes, but never present supplied claims as bytes read here."""
    if inventory.get("schema_version") != 1:
        raise ValueError("unsupported inventory schema")
    rows, excluded = [], []
    for item in inventory.get("exclusions", []):
        excluded.append(dict(path=member(item["path"]), reason=str(item["reason"])))
    for entry in normalize(inventory["files"]):
        reason = exclusion(entry["path"])
        if reason:
            excluded.append(dict(path=entry["path"], reason=reason))
        else:
            rows.append(dict(entry, hash_basis="reported"))
    return manifest(rows, inventory.get("provenance", {}), timestamp, excluded,
                    supplied_inventory_timestamp=inventory.get("timestamp"))


def compare(source, destination, timestamp):
    for value in (source, destination):
        if value is not None and value.get("schema_version") != 1:
            raise ValueError("unsupported manifest schema")
    left = {r["path"]: r for r in normalize(source["files"])}
    right = {r["path"]: r for r in normalize(destination["files"])} if destination else {}
    result = dict(schema_version=1, timestamp=timestamp, missing=sorted(left.keys() - right.keys()),
                  extra=sorted(right.keys() - left.keys()), size_differences=[], hash_differences=[],
                  group_differences=[], unknown=[], independence_unestablished=[],
                  destination_present=destination is not None,
                  backup_status="not_executed", backup_verified=False)
    for path in sorted(left.keys() & right.keys()):
        a, b = left[path], right[path]
        for field, key in (("bytes", "size_differences"), ("sha256", "hash_differences"),
                           ("group", "group_differences")):
            if a[field] is None or b[field] is None:
                result["unknown"].append(dict(path=path, field=field))
            elif a[field] != b[field]:
                result[key].append(dict(path=path, source=a[field], destination=b[field]))
        if (a["observation_origin"] == b["observation_origin"]
                or any(r["observation_class"] == CLASSES[0] or r.get("hash_basis") != "bytes_read"
                       for r in (a, b))):
            result["independence_unestablished"].append(path)
    result["content_match"] = bool(left and destination) and not any(result[key] for key in (
        "missing", "extra", "size_differences", "hash_differences", "group_differences",
        "unknown", "independence_unestablished"))
    result["provenance"] = {"source": source.get("provenance"),
                            "destination": destination.get("provenance") if destination else None}
    return result


def check_output(output, roots=()):
    output = Path(output).absolute()
    if exclusion(output.as_posix()):
        raise ValueError("excluded output path")
    if os.path.lexists(output):
        raise FileExistsError(f"output already exists: {output}")
    if any(output.resolve().is_relative_to(Path(root).resolve()) for root in roots):
        raise ValueError("output must be outside source tree")
    with directory(output.parent):
        pass


def write_output(output, value, roots=()):
    check_output(output, roots)
    output = Path(output).absolute()
    with directory(output.parent) as parent:
        fd = os.open(output.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=parent)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    scan_parser = commands.add_parser("scan", help="hash explicit selections only")
    scan_parser.add_argument("--root", required=True, type=Path)
    scan_parser.add_argument("--selection", required=True, type=Path)
    scan_parser.add_argument("--observation-class", required=True, choices=CLASSES[1:])
    scan_parser.add_argument("--origin", required=True)
    scan_parser.add_argument("--approved-root-target", type=Path,
                             help="exact platform mount target confirmed by approved preflight")
    inventory_parser = commands.add_parser("inventory", help="normalize supplied observations without source reads")
    inventory_parser.add_argument("--input", required=True, type=Path)
    compare_parser = commands.add_parser("compare", help="compare separately generated manifests")
    compare_parser.add_argument("--source", required=True, type=Path)
    compare_parser.add_argument("--destination", type=Path)
    for command in (scan_parser, inventory_parser, compare_parser):
        command.add_argument("--output", required=True, type=Path)
        command.add_argument("--timestamp", help="explicit observation timestamp for reproducible serialization")
    args = parser.parse_args()
    timestamp = args.timestamp or datetime.now(timezone.utc).isoformat()
    try:
        if datetime.fromisoformat(timestamp.replace("Z", "+00:00")).tzinfo is None:
            raise ValueError("timestamp requires an explicit timezone")
        roots = [args.root] if args.command == "scan" else []
        check_output(args.output, roots)
        if args.command == "scan":
            value = scan(args.root, read_json(args.selection), args.observation_class, args.origin,
                         timestamp, args.approved_root_target)
        elif args.command == "inventory":
            inventory = read_json(args.input)
            roots = [inventory["source_root"]] if inventory.get("source_root") else []
            value = supplied(inventory, timestamp)
        else:
            source = read_json(args.source)
            destination = read_json(args.destination) if args.destination else None
            roots = [m["source_root"] for m in (source, destination) if m and m.get("source_root")]
            value = compare(source, destination, timestamp)
        write_output(args.output, value, roots)
        if args.command == "compare" and not value["content_match"]:
            return 1
        return 0
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        parser.exit(2, f"preservation: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
