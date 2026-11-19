"""One bounded Agent Lightning phase per Modal invocation. See experiment/README.md.

The local environment needs only the locked Modal client dependencies. All training
packages live in a separate image venv, and all durable state lives on the Volume.
Run invocations serially: the committed name marker is a retry guard, not a lock
across independently submitted Modal apps.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
import zipfile

import modal
from modal_build import build_training_stack, finalize_training_stack

LOCAL = Path(__file__).resolve().parent
STATE = Path("/data")
REPO = Path("/opt/agent-lightning")
PYTHON = REPO / ".venv/bin/python"
BUNDLE = Path("/opt/agent-lightning-lab/experiment")
if not (LOCAL / "experiment/ppo_contract.py").is_file():
    # Remote /root/modal_app.py uses the already packaged experiment namespace.
    sys.path.insert(0, str(BUNDLE.parent))
from experiment import ppo_contract

BUILD_EVIDENCE = Path("/opt/agl-build/evidence")
SOURCE_VOLUME_NAME = "cs329z-agent-lightning"
WORKING_VOLUME = os.environ.get("AGL_WORKING_VOLUME", "")
if WORKING_VOLUME and (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", WORKING_VOLUME)
                       or WORKING_VOLUME == SOURCE_VOLUME_NAME):
    raise ValueError("AGL_WORKING_VOLUME must identify a distinct existing working Volume")
VOLUME_NAME = WORKING_VOLUME or SOURCE_VOLUME_NAME
SECRET_NAME = "agent-lightning-credentials"
DATA_ID = "1FQMyKLLd6hP9dw9rfZn1EZOWNvKaDsqw"
CUDA_IMAGE = "nvidia/cuda:13.0.0-devel-ubuntu24.04@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d"
PHASES = {"smoke", "smoke-reload-eval", "baseline-eval", "main", "reload-eval", "export-eval"}
ACTIONS = PHASES | {"prepare", "preflight", "summarize", "export"}
NAME_RE = r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}"
RUNTIME_FILES = (
    "ppo_contract.py", "ppo_math.py", "ppo_runtime.py", "ppo_source_lock.json",
    "check_gpu.py", "launch.py", "export_artifact.py", "observed_calc_agent.py", "prepare_model.py",
    "prepare_data.py", "summarize.py", "requirements-agent.txt", "LICENSE-AgentLightning.txt",
    "templates/tool_chat_template_llama3.1_json.jinja", "templates/PROVENANCE.json",
    "templates/LICENSE-vllm.txt",
)


image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .apt_install("build-essential", "git", "curl", "unzip", "pkg-config", "ca-certificates")
    .uv_pip_install("uv==0.12.13", uv_version="0.12.13")
    # Present before either builder is discovered; no experiment/app source here.
    .add_local_file(LOCAL / "modal_build.py", "/root/modal_build.py", copy=True)
    .add_local_file(LOCAL / "scripts/install_modal_training.sh", "/opt/agl-build/install_modal_training.sh", copy=True)
    .run_function(build_training_stack, cpu=8, memory=65536, timeout=7200, include_source=False)
    # Changes below this line reuse a successfully built FlashAttention layer.
    .add_local_file(LOCAL / "scripts/finalize_modal_training.sh", "/opt/agl-build/finalize_modal_training.sh", copy=True)
    .add_local_file(LOCAL / "experiment/requirements-agent.txt", "/opt/agl-build/requirements-agent.txt", copy=True)
    .run_function(finalize_training_stack, cpu=4, memory=16384, timeout=1800, include_source=False)
)
# Only named runtime inputs are copied; credentials, local venvs, and datasets are not uploaded.
for relative in RUNTIME_FILES:
    image = image.add_local_file(LOCAL / "experiment" / relative, str(BUNDLE / relative), copy=True)
if WORKING_VOLUME:
    # Preserve the explicit non-secret binding when the module is imported remotely.
    image = image.env({"AGL_WORKING_VOLUME": WORKING_VOLUME})

app = modal.App("cs329z-agent-lightning")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=not bool(WORKING_VOLUME))
credentials = modal.Secret.from_name(SECRET_NAME, required_keys=["HF_TOKEN", "WANDB_API_KEY"])


def request(action, name, profile="llama3b", checkpoint="", wandb_entity="", weave=False,
            baseline_name="", post_name="", export_path="",
            baseline_phase="baseline-eval", post_phase="reload-eval", algorithm="grpo", protocol="",
            data_layout="root", checkpoint_identity=""):
    """Reject malformed or ambiguous requests locally, before a remote GPU call."""
    if action not in ACTIONS:
        raise ValueError(f"--action must be one of {', '.join(sorted(ACTIONS))}")
    if not re.fullmatch(NAME_RE, name):
        raise ValueError("--name must be 1–80 letters/digits/._-, beginning with a letter or digit")
    if profile not in {"llama3b", "reference-qwen"}:
        raise ValueError("--model-profile must be llama3b or reference-qwen; profiles never switch automatically")
    algorithm_identity = ppo_contract.validate_selection(algorithm, action, profile, name, protocol)
    if data_layout not in {"root", "restored"}:
        raise ValueError("--data-layout must be root or restored")
    if algorithm == ppo_contract.PPO and data_layout != "restored":
        raise ValueError("PPO smoke requires explicit --data-layout restored on the working Volume")
    reload_phase = action in {"smoke-reload-eval", "reload-eval"}
    if reload_phase:
        prefix = "phase2/" if algorithm == ppo_contract.PPO else ""
        if not re.fullmatch(rf"{prefix}runs/{NAME_RE}/checkpoints/global_step_\d+", checkpoint):
            raise ValueError("--checkpoint must be relative to /data: runs/NAME/checkpoints/global_step_N")
        if action == "smoke-reload-eval" and not checkpoint.endswith("/global_step_2"):
            raise ValueError("smoke-reload-eval requires global_step_2")
    elif checkpoint:
        raise ValueError("--checkpoint is accepted only for reload evaluation")
    identity_sha256 = ""
    if algorithm == ppo_contract.PPO and reload_phase:
        if not checkpoint.split('/')[2].startswith('ppo-mc-v1-') or not checkpoint_identity:
            raise ValueError("PPO reload requires its new namespace and --checkpoint-identity local JSON")
        identity_file = Path(checkpoint_identity)
        ppo_contract.validate_checkpoint(json.loads(identity_file.read_text()), algorithm)
        identity_sha256 = sha256(identity_file)
    elif checkpoint_identity:
        raise ValueError("--checkpoint-identity is only accepted for PPO smoke reload")
    if action == "export-eval":
        if profile != "llama3b" or not re.fullmatch(rf"exports/{NAME_RE}", export_path):
            raise ValueError("export-eval requires llama3b and --export-path exports/NAME")
    elif export_path:
        raise ValueError("--export-path is accepted only for export-eval")
    if baseline_phase not in PHASES or post_phase not in PHASES:
        raise ValueError("Summary phase filters must be explicit known experiment phases")
    if action in PHASES and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", wandb_entity):
        raise ValueError("Training/evaluation requires --wandb-entity with your W&B user/team slug")
    if weave and action not in PHASES:
        raise ValueError("--weave is available only for training/evaluation phases")
    if action == "summarize" and not all(re.fullmatch(NAME_RE, value) for value in (baseline_name, post_name)):
        raise ValueError("summarize requires --baseline-name and --post-name")
    return dict(action=action, name=name, profile=profile, checkpoint=checkpoint,
                **algorithm_identity, data_layout=data_layout, checkpoint_identity_sha256=identity_sha256,
                wandb_entity=wandb_entity, weave=weave, baseline_name=baseline_name, post_name=post_name,
                export_path=export_path, baseline_phase=baseline_phase, post_phase=post_phase)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def environment(req):
    env = dict(os.environ, HF_HOME=str(STATE / "cache/huggingface"), PYTHONUNBUFFERED="1",
               PYTHONDONTWRITEBYTECODE="1")
    # PATH is changed only for subprocesses, never for the Modal worker's interpreter.
    # The upstream agent launches calculator and other console scripts from this venv.
    env["PATH"] = f"{PYTHON.parent}:{env.get('PATH', '')}"
    env.pop("AGL_WEAVE_PROJECT", None)
    if req["wandb_entity"]:
        env["WANDB_ENTITY"] = req["wandb_entity"]
    if req["weave"]:
        env["AGL_WEAVE_PROJECT"] = req["wandb_entity"] + "/cs329z-agent-lightning"
    return env


def run_command(command, log, env, timeout, follow_files=()):
    """Close subprocesses/logs before Volume commit, including on command timeout."""
    command = [str(arg) for arg in command]
    print(f"Running {Path(command[0]).name}; log: {log}", flush=True)
    with Path(log).open("w") as output:
        proc = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env,
                                stdin=subprocess.DEVNULL)
        tails = {}

        def stream():
            # Poll existing log files on this thread; no extra worker or tracing service.
            for path in (Path(log), *follow_files):
                if path not in tails and path.is_file():
                    tails[path] = path.open(errors="replace")
                if path in tails:
                    chunk = tails[path].read(262144)
                    if chunk:
                        print(f"[{path.name}] {chunk}", end="" if chunk.endswith("\n") else "\n", flush=True)

        deadline = time.monotonic() + timeout
        try:
            while proc.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                stream()
                try:
                    proc.wait(timeout=min(1, remaining))
                except subprocess.TimeoutExpired:
                    pass
            code = proc.returncode
        except BaseException:
            # launch.py handles SIGTERM and stops its own gateway/controller/trainer groups.
            proc.terminate()
            try:
                proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
        finally:
            stream()
            for handle in tails.values():
                handle.close()
    if code:
        raise RuntimeError(f"Command exited {code}; inspect {log}")


def recorded(req, operation):
    if req.get("volume", VOLUME_NAME) != VOLUME_NAME:
        raise ValueError("Worker Volume binding differs from submitted request")
    if req["action"] == "export-eval" and not WORKING_VOLUME:
        raise ValueError("export-eval must not mount the original source Volume")
    if req.get("algorithm") == ppo_contract.PPO:
        if not WORKING_VOLUME or req.get("data_layout") != "restored":
            raise ValueError("PPO must use restored inputs on an existing working Volume")
        ppo_contract.validate_selection(req['algorithm'], req['action'], req['profile'], req['name'], req['protocol'])
    volume.reload()
    record = STATE / "submissions" / req["name"]
    record.mkdir(parents=True, exist_ok=False)
    receipt = dict(request=req, cuda_base_image=CUDA_IMAGE, status="started",
                   volume=VOLUME_NAME, function_call_id=modal.current_function_call_id(),
                   started_at=datetime.now(timezone.utc).isoformat())
    write_json(record / "receipt.json", receipt)
    # Commit before any model download or GPU computation. A retry with the same name refuses to start.
    volume.commit()
    try:
        shutil.copytree(BUNDLE, record / "source/experiment")
        shutil.copy2(__file__, record / "source/modal_app.py")
        shutil.copy2(LOCAL / "modal_build.py", record / "source/modal_build.py")
        shutil.copytree(BUILD_EVIDENCE, record / "image-build")
        write_json(record / "source-sha256.json", {
            str(p.relative_to(record / "source")): sha256(p)
            for p in (record / "source").rglob("*") if p.is_file()
        })
        if req.get("algorithm") == ppo_contract.PPO:
            expected = req.get('source', {}).get('runtime_files_sha256', {})
            if set(expected) != set(RUNTIME_FILES) or any(sha256(BUNDLE / f) != expected[f] for f in RUNTIME_FILES):
                raise ValueError("PPO worker runtime files differ from the submitted source identity")
        result = operation(req, record)
        receipt.update(status="completed", result=result)
        return result
    except BaseException as exc:
        receipt.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(record / "receipt.json", receipt)
        volume.commit()


def verify_data(directory):
    manifest = json.loads((directory / "data_manifest.json").read_text())
    for name, item in manifest["files"].items():
        if Path(name).name != name or sha256(directory / name) != item["sha256"]:
            raise ValueError(f"Frozen dataset differs from its manifest: {name}")


def extract_parquets(archive, target):
    """Copy only the two expected members, including when the ZIP has a top folder."""
    with zipfile.ZipFile(archive) as zipped:
        for name in ("train.parquet", "test.parquet"):
            matches = [member for member in zipped.infolist() if Path(member.filename).name == name]
            if len(matches) != 1:
                raise ValueError(f"Dataset ZIP must contain exactly one {name}")
            with zipped.open(matches[0]) as source, (target / name).open("xb") as destination:
                shutil.copyfileobj(source, destination)


def prepare(req, record):
    env = environment(req)
    staging = STATE / "staging" / req["name"]
    staging.mkdir(parents=True)
    model = STATE / "models" / req["profile"]
    model.parent.mkdir(parents=True, exist_ok=True)
    model_target = model if model.exists() else staging / "model-download"
    run_command([PYTHON, BUNDLE / "prepare_model.py", "--profile", req["profile"],
                 "--output-dir", model_target], record / "prepare-model.log", env, 2700)
    if model_target != model:
        model_target.rename(model)
    data = STATE / "data-small"
    if not data.exists():
        source = STATE / "data-source"
        source.mkdir(parents=True, exist_ok=True)
        archive = source / "calc-x-data.zip"
        if not archive.exists():
            downloaded = staging / "calc-x-data.zip"
            run_command([PYTHON, "-m", "gdown", DATA_ID, "-O", downloaded],
                        record / "download-data.log", env, 600)
            if not zipfile.is_zipfile(downloaded):
                raise ValueError("Download is not a ZIP. Upload the released ZIP to data-source/calc-x-data.zip")
            downloaded.rename(archive)
        extract_parquets(archive, staging)
        run_command([PYTHON, BUNDLE / "prepare_data.py", "--train-parquet", staging / "train.parquet",
                     "--eval-parquet", staging / "test.parquet", "--out", data, "--seed", "329"],
                    record / "prepare-data.log", env, 180)
    verify_data(data)
    return dict(model=str(model), data=str(data), gpu_used=False)


def launch_command(req):
    inputs = STATE / "restored" if req.get("data_layout", "root") == "restored" else STATE
    output = STATE / "phase2/runs" if req.get("algorithm") == ppo_contract.PPO else STATE / "runs"
    command = [PYTHON, BUNDLE / "launch.py", "--repo", REPO, "--python", PYTHON,
               "--model-path", (STATE / req["export_path"] if req["action"] == "export-eval"
                                else inputs / "models" / req["profile"]), "--data-dir", inputs / "data-small",
               "--output-root", output, "--name", req["name"], "--profile", req["profile"],
               "--phase", req["action"], "--walltime-minutes", "180" if req["action"] == "main" else "45"]
    if req.get("algorithm") == ppo_contract.PPO:
        command.extend(["--algorithm", req["algorithm"], "--protocol", req["protocol"]])
        if req.get("checkpoint_identity_sha256"):
            command.extend(["--checkpoint-identity-sha256", req["checkpoint_identity_sha256"]])
    if req["checkpoint"]:
        command.extend(["--checkpoint", STATE / req["checkpoint"]])
    return [str(arg) for arg in command]


def train_or_check(req, record):
    env = environment(req)
    if req.get("algorithm") == ppo_contract.PPO:
        inputs = STATE / "restored"
        for relative in ("models/llama3b", "data-small"):
            directory = inputs / relative
            if not directory.is_dir():
                raise ValueError(f"Missing restored PPO directory or wrong path type: {directory}; no fallback/download")
        for relative in ("models/llama3b/model_manifest.json", "data-small/data_manifest.json"):
            if not (inputs / relative).is_file():
                raise ValueError(f"Missing restored PPO input: {inputs / relative}; no fallback/download")
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    run_command(["nvidia-smi"], record / "nvidia-smi.txt", env, 60)
    run_command([PYTHON, BUNDLE / "check_gpu.py", "--output", record / "gpu_preflight.json"],
                record / "gpu-preflight.log", env, 600)
    if req["action"] == "preflight":
        return json.loads((record / "gpu_preflight.json").read_text())
    minutes = 190 if req["action"] == "main" else 55
    runs = STATE / "phase2/runs" if req.get("algorithm") == ppo_contract.PPO else STATE / "runs"
    service_logs = [runs / req["name"] / "logs" / f"{service}.log"
                    for service in ("trainer", "controller", "gateway")]
    run_command(launch_command(req), record / "launch.log", env, minutes * 60, service_logs)
    return json.loads((runs / req["name"] / "result.json").read_text())


def export_evidence(name):
    destination = STATE / "exports" / f"{name}.tgz"
    destination.parent.mkdir(parents=True, exist_ok=True)

    def keep(info):
        parts = Path(info.name).parts
        if "checkpoints" in parts or parts[:2] == ("submissions", name) or info.issym() or info.islnk():
            return None
        return info

    # Cache, original data, staging downloads, and model/checkpoint weights are excluded by construction.
    with tarfile.open(destination, "x:gz") as archive:
        for folder in ("runs", "evidence", "data-small", "submissions"):
            if (STATE / folder).exists():
                archive.add(STATE / folder, arcname=folder, filter=keep)
        for manifest in sorted((STATE / "models").glob("*/model_manifest.json")):
            archive.add(manifest, arcname=str(manifest.relative_to(STATE)), filter=keep)
    return dict(archive=str(destination), volume_path=str(destination.relative_to(STATE)),
                bytes=destination.stat().st_size, sha256=sha256(destination), gpu_used=False)


def collect(req, record):
    if req["action"] == "export":
        return export_evidence(req["name"])
    output = STATE / "evidence" / req["name"]
    run_command([PYTHON, BUNDLE / "summarize.py", "--run-dir", STATE / "runs" / req["post_name"] / "traces",
                 "--baseline-dir", STATE / "runs" / req["baseline_name"] / "traces",
                 "--heldout-file", STATE / "data-small/eval_main.jsonl", "--output-dir", output,
                 "--phase", req["post_phase"], "--baseline-phase", req["baseline_phase"], "--route-mode", "val"],
                record / "summarize.log", environment(req), 300)
    return dict(summary=str(output / "summary.json"), gpu_used=False)


@app.function(image=image, volumes={"/data": volume}, secrets=[credentials], cpu=4, memory=16384,
              timeout=3600, retries=0, max_containers=1, single_use_containers=True)
def prepare_assets(req: dict):
    return recorded(req, prepare)


@app.function(image=image, volumes={"/data": volume}, secrets=[credentials], gpu="H100!", cpu=16,
              memory=131072, timeout=14400, retries=0, max_containers=1, single_use_containers=True)
def gpu_phase(req: dict):
    return recorded(req, train_or_check)


@app.function(image=image, volumes={"/data": volume}, cpu=4, memory=16384, timeout=3600,
              retries=0, max_containers=1, single_use_containers=True)
def collect_evidence(req: dict):
    return recorded(req, collect)


@app.local_entrypoint()
def main(action: str = "", name: str = "", model_profile: str = "llama3b", checkpoint: str = "",
         wandb_entity: str = "", weave: bool = False, baseline_name: str = "", post_name: str = "",
         export_path: str = "", baseline_phase: str = "baseline-eval", post_phase: str = "reload-eval",
         algorithm: str = "grpo", protocol: str = "", data_layout: str = "root", checkpoint_identity: str = ""):
    """Submit one phase. Use a new --name for every call; never run calls concurrently."""
    req = request(action, name, model_profile, checkpoint, wandb_entity, weave, baseline_name, post_name,
                  export_path, baseline_phase, post_phase, algorithm, protocol, data_layout, checkpoint_identity)
    if "--detach" not in sys.argv and "-d" not in sys.argv:
        raise ValueError("Use modal run --detach; attached submission refused")
    if action == "export-eval" and not WORKING_VOLUME:
        raise ValueError("export-eval requires AGL_WORKING_VOLUME; original Volume must be absent")
    if WORKING_VOLUME and action == "prepare":
        raise ValueError("Working Volume inputs must be restored, never redownloaded by prepare")
    if algorithm == ppo_contract.PPO and not WORKING_VOLUME:
        raise ValueError("PPO requires AGL_WORKING_VOLUME; original Phase 1 Volume must be absent")
    req["volume"] = VOLUME_NAME
    req["app_id"] = app.app_id
    try:
        req["source"] = {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=LOCAL, text=True,
                                               stderr=subprocess.DEVNULL, timeout=10).strip(),
            "tracked_dirty": bool(subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"], cwd=LOCAL, text=True, timeout=10).strip()),
        }
    except (OSError, subprocess.SubprocessError):
        req["source"] = {"commit": None, "tracked_dirty": None}
    if algorithm == ppo_contract.PPO:
        if (not re.fullmatch(r"[0-9a-f]{40}", req["source"]["commit"] or "")
                or req["source"]["tracked_dirty"] is not False):
            raise ValueError("PPO submission requires an identifiable clean committed source tree")
        untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"],
                                            cwd=LOCAL, text=True, timeout=10).strip()
        if untracked:
            raise ValueError("Commit reviewed PPO source before submission; untracked files remain")
        req["source"]["runtime_files_sha256"] = {relative: sha256(LOCAL / "experiment" / relative)
                                                  for relative in RUNTIME_FILES}
    function = prepare_assets if action == "prepare" else collect_evidence if action in {"summarize", "export"} else gpu_phase
    if algorithm == ppo_contract.PPO:
        function = function.with_options(memory=196608)
    call = function.spawn(req)
    receipt = dict(status="submitted_not_validated", request=req, app_id=app.app_id,
                   function_call_id=call.object_id, volume=VOLUME_NAME,
                   receipt_path=f"submissions/{name}/receipt.json",
                   submitted_at=datetime.now(timezone.utc).isoformat())
    # Print IDs before local I/O: a local write failure must not trigger resubmission.
    print("SUBMITTED: " + json.dumps(receipt), flush=True)
    local = LOCAL / ".local/experiment-submissions"
    local.mkdir(parents=True, exist_ok=True)
    with (local / f"{name}-{time.time_ns()}.json").open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print("Submission is not completion; inspect the durable Volume receipt. No later phase is queued.", flush=True)
