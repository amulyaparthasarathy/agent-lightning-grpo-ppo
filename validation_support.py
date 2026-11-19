"""Small shared helpers. Importing this module performs no network requests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

VOLUME_NAME = "agent-lightning-validation"
PROJECT = "agent-lightning-validation"
SECRET_NAME = "agent-lightning-credentials"
ROOT = Path("/validation")
RUNTIME = "/opt/validation/runtime.py"


def request(kind: str, name: str, entity: str, hold_seconds: int = 0) -> dict:
    if kind not in {"services", "image", "gpu"}:
        raise ValueError("kind must be services, image, or gpu")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", name):
        raise ValueError("Use a new run name: 1-80 letters/digits/._-, starting with a letter/digit")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", entity):
        raise ValueError("Use the W&B entity/team slug, not an email or project path")
    if type(hold_seconds) is not int or not 0 <= hold_seconds <= 900:
        raise ValueError("hold_seconds must be an integer between 0 and 900")
    if kind != "services" and hold_seconds:
        raise ValueError("Only the CPU services test supports a disconnect-test hold")
    return dict(kind=kind, name=name, entity=entity, hold_seconds=hold_seconds,
                project=PROJECT, synthetic=True)



def require_detached() -> None:
    if "--detach" not in sys.argv and "-d" not in sys.argv:
        raise ValueError("Use scripts/validate_modal.sh or modal run --detach; attached submission refused")


def source_identity() -> dict:
    root = Path(__file__).resolve().parent
    paths = ("validation_support.py", "validation_services.py", "validation_training.py",
             "validation/runtime.py", "validation/compatibility.py", "modal_app.py", "modal_build.py",
             "scripts/install_modal_training.sh", "scripts/finalize_modal_training.sh",
             "experiment/requirements-agent.txt", "experiment/check_gpu.py",
             "pyproject.toml", "uv.lock")
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in paths if (root / name).is_file()}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                         stderr=subprocess.DEVNULL, text=True, timeout=10).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                             cwd=root, text=True, timeout=10).strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    return dict(commit=commit, tracked_dirty=dirty, source_sha256=hashes)

def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def submitted(app, call, req: dict) -> None:
    """Called ONLY after .spawn returns. No remote success claim is made."""
    receipt = dict(request=req, app_id=app.app_id, function_call_id=call.object_id,
                   submitted_at=datetime.now(timezone.utc).isoformat(),
                   status="submitted_not_validated", volume=VOLUME_NAME,
                   result_path=f"runs/{req['name']}/result.json")
    local = Path(".local/validation-submissions")
    local.mkdir(parents=True, exist_ok=True)
    # Timestamp avoids overwriting a previous local submission receipt.
    destination = local / f"{req['name']}-{time.time_ns()}.json"
    write_json(destination, receipt)
    print("SUBMITTED: " + json.dumps(receipt), flush=True)
    print(f"Receipt: {destination}. With --detach, this job no longer needs your laptop.", flush=True)
    print("Submission is not completion. Inspect result.json and the W&B run.", flush=True)


def execute(req: dict, python: str, volume, *, timeout_seconds: int = 1800) -> dict:
    """Runs entirely on Modal. Explicit commits preserve evidence between ticks."""
    request(req["kind"], req["name"], req["entity"], req["hold_seconds"])
    volume.reload()
    record = ROOT / "runs" / req["name"]
    record.mkdir(parents=True, exist_ok=False)
    receipt = dict(request=req, status="started", started_at=datetime.now(timezone.utc).isoformat())
    write_json(record / "launcher.json", receipt)
    volume.commit()  # Claim before requests. Run serially; this is not a distributed lock.
    command = [python, RUNTIME, "--kind", req["kind"], "--name", req["name"],
               "--entity", req["entity"], "--hold-seconds", str(req["hold_seconds"]),
               "--output", str(record)]
    proc = None
    try:
        with (record / "runtime.log").open("w") as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + timeout_seconds
            last_commit = time.monotonic()
            with (record / "runtime.log").open() as reader:
                while proc.poll() is None:
                    print(reader.read(), end="", flush=True)
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Validation exceeded its bounded runtime")
                    if time.monotonic() - last_commit >= 30:
                        volume.commit()
                        last_commit = time.monotonic()
                    time.sleep(1)
                print(reader.read(), end="", flush=True)
        receipt["exit_code"] = proc.returncode
        if proc.returncode:
            raise RuntimeError("Validation failed; inspect runtime.log and result.json")
        result = json.loads((record / "result.json").read_text())
        result_status = result.get("status")
        if result_status == "diagnostic_passed_with_reviewed_exceptions":
            training = result.get("checks", {}).get("training", {})
            if (req["kind"] not in {"image", "gpu"}
                    or result.get("dependency_closure_consistent") is not False
                    or result.get("reviewed_dependency_conflicts") != 6
                    or result.get("qualification") != "diagnostic_with_reviewed_dependency_exceptions"
                    or training.get("cpu_imports_checked") is not True):
                raise RuntimeError("Child diagnostic exception status is not independently corroborated")
            receipt["dependency_qualification"] = result["qualification"]
        elif result_status != "passed":
            raise RuntimeError("Child did not produce an acceptable validation result")
        receipt["status"] = "completed"
        return result
    except BaseException as exc:
        receipt.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(record / "launcher.json", receipt)
        volume.commit()
