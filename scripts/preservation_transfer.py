"""Bounded HF byte transport. No Torch, credentials on disk, or implicit resources.

SDK imports are lazy. All paths come from approved manifests. No automatic
application retry: ambiguous commits require readback/reconciliation first.
"""
from __future__ import annotations
import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
from threading import Event, Lock

from scripts import preservation_manifest as pm
from scripts.backup_qualification import digest

READERS = 8  # At most eight bounded byte streams on the eight-core CPU worker.


def parallel_read(items, read):
    """Keep input order; stop other streams after the first authentic failure."""
    items = list(items)
    results = [None] * len(items)
    stopped, lock, errors = Event(), Lock(), []
    def run(index, item):
        if stopped.is_set():
            return
        try:
            results[index] = read(item, stopped)
        except Exception as exc:
            with lock:
                if not errors:
                    errors.append(exc)
            stopped.set()
    with ThreadPoolExecutor(max_workers=READERS) as pool:
        futures = [pool.submit(run, i, item) for i, item in enumerate(items)]
        for future in futures:
            future.result()
    if errors:
        raise errors[0]
    return results


def now():
    return datetime.now(timezone.utc).isoformat()


class GateFailure(ValueError):
    """Sanitized application diagnostic, never an SDK response."""


def require(condition, message):
    if not condition:
        raise GateFailure(message)


def identity(api, repo):
    require(repo in {"amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup",
                     "amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-GRPO"}, "unapproved destination")
    info = api.repo_info(repo_id=repo, repo_type="model")
    require(info.id == repo and info.private is True, "destination identity/privacy mismatch")
    return info.sha


def byte_stream(api, repo, revision, path):
    """Fresh HTTP response, no HF cache. Redirect auth stripping is HTTPX's policy."""
    import httpx
    from huggingface_hub import hf_hub_url
    require(len(revision) == 40, "immutable destination revision required")
    url = hf_hub_url(repo, pm.member(path), revision=revision, repo_type="model")
    with httpx.Client(follow_redirects=True, timeout=60, trust_env=False) as client:
        with client.stream("GET", url, headers={"Authorization": "Bearer " + api.token}) as response:
            response.raise_for_status()
            yield from response.iter_bytes(chunk_size=pm.BUFFER)


def read_small(api, repo, revision, path):
    data = bytearray()
    for chunk in byte_stream(api, repo, revision, path):
        data.extend(chunk)
        require(len(data) <= 8 * pm.BUFFER, "small record exceeds limit")
    return json.loads(data)


def commit_files(api, repo, revision, additions, message):
    from huggingface_hub import CommitOperationAdd
    operations = [CommitOperationAdd(path_in_repo=pm.member(p), path_or_fileobj=v) for p, v in additions]
    require(0 < len(operations) <= 50, "bounded commit required")
    return api.create_commit(repo_id=repo, repo_type="model", revision="main", parent_commit=revision,
                             operations=operations, commit_message=message, num_threads=1).oid


def put_record(api, repo, revision, path, record):
    return commit_files(api, repo, revision, [(path, json.dumps(record, sort_keys=True, allow_nan=False).encode())],
                        "Preservation receipt " + path.rsplit("/", 1)[-1])


def inventory(roots, selections, origin):
    result = {}
    for namespace, root in roots.items():
        root = Path(root)
        # The caller binds these approved mounts by Volume ID. Resolve only this root.
        result[namespace] = pm.scan(root, selections[namespace], "measured_observation", origin + "/" + namespace,
                                    now(), approved_root_target=root.resolve(strict=True))
    return result


def upload_snapshot(api, repo, prefix, roots, manifests, expected_revision, record_progress, resume=False):
    """Create a new namespace. Failed attempts stay intact; never restart over it."""
    prefix = pm.member(prefix)
    require(identity(api, repo) == expected_revision, "destination advanced before snapshot claim")
    existing = list(api.list_repo_files(repo, revision=expected_revision))
    occupied={p for p in existing if p == prefix or p.startswith(prefix + "/")}
    if resume and occupied:
        require(prefix+"/transfer.json" not in occupied, "only incomplete attempts can resume")
        attempt=read_small(api,repo,expected_revision,prefix+"/attempt.json")
        require(attempt.get("source_manifests_sha256")==digest(manifests), "resume source identity differs")
        allowed={f"{prefix}/{n}/{r['path']}" for n,m in manifests.items() for r in m['files']}
        require(occupied<=allowed|{prefix+"/attempt.json",prefix+"/source-manifests.json"}, "foreign files in incomplete prefix")
        revision=expected_revision
    else:
        require(not occupied, "snapshot prefix already exists")
        revision = put_record(api, repo, expected_revision, prefix + "/attempt.json",
                              {"status": "incomplete", "started_at": now(), "source_manifests_sha256": digest(manifests)})
    record_progress({"status": "incomplete", "revision": revision, "uploaded_bytes": 0})
    reused = [(f"{prefix}/{n}/{r['path']}", r) for n, m in sorted(manifests.items())
              for r in m['files'] if f"{prefix}/{n}/{r['path']}" in occupied]
    checked = parallel_read(reused, lambda item, stopped:
        download_file(byte_stream(api, repo, expected_revision, item[0]), item[1], stopped=stopped))
    total = sum(value[0] for value in checked)
    record_progress({'status':'incomplete','revision':revision,'uploaded_bytes':total,'validated_reused_bytes':total})
    pending = []
    def flush():
        nonlocal revision, total
        if not pending:
            return
        for key,path,before,size in pending:
            require(pm.signature(path.lstat()) == before, 'source changed before batch upload')
        revision = commit_files(api, repo, revision, [(key,str(path)) for key,path,_,_ in pending],
                                'Preserve source files (bounded batch)')
        for key,path,before,size in pending:
            require(pm.signature(path.lstat()) == before, 'source changed during upload')
            total += size
        pending.clear()
        record_progress({'status':'incomplete','revision':revision,'uploaded_bytes':total})
    for namespace, manifest in sorted(manifests.items()):
        root = Path(roots[namespace]).resolve(strict=True)
        # Re-hash each source before upload, including metadata mutation checks.
        for row in manifest["files"]:
            key=f"{prefix}/{namespace}/{row['path']}"
            if key in occupied:
                # All committed parts were independently rehashed before new writes.
                continue
            path = root / pm.member(row["path"])
            before = pm.signature(path.lstat())
            with pm.directory(path.parent) as parent:
                size, sha, _ = pm.read_file(parent, path.name, False, None)
            require((size, sha) == (row["bytes"], row["sha256"]), "source changed after inventory")
            require(pm.signature(path.lstat()) == before, 'source changed during validation')
            if pending and (len(pending)>=50 or sum(p[3] for p in pending)+size>8*pm.BUFFER):
                flush()
            pending.append((key,path,before,size))
            if size>8*pm.BUFFER:
                flush()
    flush()
    if prefix + "/source-manifests.json" not in occupied:
        revision = put_record(api, repo, revision, prefix + "/source-manifests.json", manifests)
    else:
        require(digest(read_small(api,repo,expected_revision,prefix+"/source-manifests.json"))==digest(manifests),
                "resume manifest differs")
    # This status explicitly does NOT qualify an independent backup.
    revision = put_record(api, repo, revision, prefix + "/transfer.json",
                          {"status": "uploaded_not_verified", "bytes": total, "completed_at": now()})
    return {"revision": revision, "bytes": total, "snapshot_prefix": prefix, "repo_id": repo, "private": True}


def download_file(chunks, expected, output=None, capture=False, stopped=None):
    digestor, size, data = hashlib.sha256(), 0, bytearray()
    handle = None
    try:
        require(stopped is None or not stopped.is_set(), 'parallel read cancelled after another failure')
        if output is not None:
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            with pm.directory(output.parent):
                handle = output.open("xb")
        for chunk in chunks:
            require(stopped is None or not stopped.is_set(), 'parallel read cancelled after another failure')
            size += len(chunk)
            require(size <= expected["bytes"], "destination exceeds expected bytes")
            digestor.update(chunk)
            if capture:
                require(size <= pm.BUFFER, "metadata exceeds bounded capture")
                data.extend(chunk)
            if handle:
                handle.write(chunk)
        require(size == expected["bytes"] and digestor.hexdigest() == expected["sha256"],
                "independent destination byte mismatch")
        return size, digestor.hexdigest(), bytes(data)
    finally:
        if handle:
            handle.close()  # Partial files are retained and never qualify restoration.
        close = getattr(chunks, 'close', None)
        if close:
            close()


def verify_bundle(path, commits):
    with tempfile.TemporaryDirectory(prefix="bundle-check-") as scratch:
        subprocess.run(["git", "clone", "--bare", str(path), scratch + "/repo"],
                       check=True, capture_output=True, timeout=120)
        for commit in commits:
            require(len(commit) == 40 and all(c in "0123456789abcdef" for c in commit), "invalid required Git commit")
            subprocess.run(["git", "-C", scratch + "/repo", "cat-file", "-e", commit + "^{commit}"],
                           check=True, capture_output=True, timeout=30)
    return {"verified": True, "restored_commits": list(commits)}


def verify_snapshot(api, target, sources, selections, bundle_path, required_commits, verifier_id, require_bundle=True):
    """Run in a fresh source-free worker. Hash all bytes, not LFS metadata."""
    repo, revision, prefix = target["repo_id"], target["revision"], target["snapshot_prefix"]
    identity(api, repo)
    expected = {f"{prefix}/{n}/{r['path']}" for n, m in sources.items() for r in m["files"]}
    actual = {p for p in api.list_repo_files(repo, revision=revision) if p.startswith(prefix + "/")}
    controls = {prefix + "/" + n for n in ("attempt.json", "source-manifests.json", "transfer.json")}
    require(actual == expected | controls, "destination namespace has missing/extra files")
    destinations, bundle = {}, None
    with tempfile.TemporaryDirectory(prefix="readback-") as scratch:
        for namespace, source in sorted(sources.items()):
            rows, metadata = [], {}
            capture = pm.METADATA | set(selections[namespace].get("provenance_records", []))
            checkpoint = source.get("native_checkpoint", pm.CHECKPOINT)
            capture |= {checkpoint + "/actor/fsdp_config.json", checkpoint + "/actor/huggingface/config.json",
                        checkpoint.rsplit("/", 1)[0] + "/latest_checkpointed_iteration.txt"}
            def read_row(row, stopped):
                key = f"{prefix}/{namespace}/{row['path']}"
                is_bundle = (namespace + "/" + row["path"]) == bundle_path
                output = Path(scratch) / "repository.bundle" if is_bundle else None
                size, sha, content = download_file(byte_stream(api, repo, revision, key), row, output,
                                                   row["path"] in capture, stopped)
                recovered = dict(verify_bundle(output, required_commits), path=row['path'], sha256=sha) if is_bundle else None
                return row, size, sha, content, recovered
            for row, size, sha, content, recovered in parallel_read(source['files'], read_row):
                rows.append(dict(group=row["group"], path=row["path"], bytes=size, sha256=sha,
                                 observation_class="measured_observation", observation_origin=verifier_id + "/" + namespace,
                                 hash_basis="bytes_read"))
                if row["path"] in capture:
                    metadata[row["path"]] = content
                if recovered is not None:
                    bundle = recovered
            checked = {p: json.loads(v) for p, v in metadata.items() if p.endswith(".json")}
            destinations[namespace] = pm.manifest(rows, source["provenance"], now(), source["exclusions"], metadata,
                selections=source["selections"], missing_selections=[], checked_metadata=checked, native_checkpoint=checkpoint)
    require(not require_bundle or bundle is not None, "Git bundle absent from independent readback")
    return destinations, bundle


def restore(api, target, rows, root):
    """Restore an approved subset into a new directory; existing content refused."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    rows = list(rows)
    paths = []
    for row in rows:
        path = pm.member(row["path"])
        require(not pm.exclusion(path), "excluded restore path")
        require(path not in paths, 'duplicate restore path')
        paths.append(path)
    def read_row(row, stopped):
        path = row['path']
        key = target["snapshot_prefix"] + "/" + row["namespace"] + "/" + path
        size, sha, _ = download_file(byte_stream(api, target["repo_id"], target["revision"], key), row,
                                     root / path, stopped=stopped)
        return {"path": path, "bytes": size, "sha256": sha}
    restored = parallel_read(rows, read_row)
    return {"status": "restored_bytes_verified", "destination": target, "files": restored, "at": now()}
