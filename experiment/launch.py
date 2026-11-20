#!/usr/bin/env python3
"""Bounded, isolated launcher for the source-reviewed Agent Lightning experiment.

Requires a prepared local model, prepared data, and the pinned AGL environment.
No model download, package installation, or GPU reservation occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .export_artifact import verify_export
    from . import ppo_contract
else:
    from export_artifact import verify_export
    import ppo_contract

AGL_COMMIT = "cbc56e41e0eab643aa71d195e92e5bba652fa683"
LOCAL_KEY = "agl-course-local-placeholder"
PROFILES = {
    "llama3b": {"repo_id": "meta-llama/Llama-3.2-3B-Instruct", "model_type": "llama", "parser": "llama3_json", "status": "source-reviewed port; not GPU validated"},
    "reference-qwen": {"repo_id": "Qwen/Qwen2.5-1.5B-Instruct", "model_type": "qwen2", "parser": "hermes", "status": "upstream model; this bounded configuration remains unvalidated"},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def read_json(path: Path):
    return json.loads(path.read_text())


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def dot(key: str, value) -> str:
    # OmegaConf.from_dotlist parses JSON-compatible values, including quoted paths.
    return key + "=" + json.dumps(value)


def absolute_python_path(path: Path) -> Path:
    """Preserve a venv interpreter symlink so Python finds its pyvenv.cfg."""
    return path.expanduser().absolute()


def validate_inputs(args, bundle: Path):
    algorithm = getattr(args, "algorithm", ppo_contract.GRPO)
    ppo_contract.validate_selection(algorithm, args.phase, args.profile, getattr(args, "name", None),
                                    getattr(args, "protocol", None))
    repo, model, data = (p.expanduser().resolve() for p in (args.repo, args.model_path, args.data_dir))
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, timeout=15).strip()
    if head != AGL_COMMIT:
        raise ValueError(f"Expected Agent Lightning {AGL_COMMIT}; checkout is {head}")
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True, timeout=15).strip()
    if dirty:
        raise ValueError("Pinned Agent Lightning tracked files were modified. Use a separate clean checkout.")
    python = absolute_python_path(args.python or repo / ".venv/bin/python")
    if not python.is_file():
        raise ValueError(f"Prepared Python executable not found: {python}")
    data_manifest = read_json(data / "data_manifest.json")
    exported = args.phase == "export-eval"
    if exported and args.profile != "llama3b":
        raise ValueError("export-eval requires the frozen llama3b profile")
    model_manifest = (verify_export(model, data_manifest, sha256_file(bundle / "observed_calc_agent.py"))
                      if exported else read_json(model / "model_manifest.json"))
    profile = PROFILES[args.profile]
    if not exported and (model_manifest.get("profile") != args.profile or model_manifest.get("repo_id") != profile["repo_id"]):
        raise ValueError("Model manifest does not match the explicit selected profile")
    if read_json(model / "config.json").get("model_type") != profile["model_type"]:
        raise ValueError("Model architecture does not match the selected profile")
    if algorithm == ppo_contract.PPO:
        if model_manifest.get("revision") != ppo_contract.BASE_REVISION:
            raise ValueError("PPO requires the frozen prepared Llama base revision")
        if args.hooks:
            raise ValueError("PPO does not accept arbitrary rollout hooks")
        output = args.output_root.expanduser().resolve()
        if output.parts[-2:] != ("phase2", "runs"):
            raise ValueError("PPO outputs must use a fresh phase2/runs namespace")
        # Includes both weight shards, token IDs, config and prepared template.
        # These are preserved byte identities, not a self-asserted model revision.
        locked_assets = read_json(bundle / "ppo_source_lock.json")
        for filename, expected in locked_assets['prepared_assets'].items():
            if (sha256_file(model / filename) != expected['sha256']
                    or (model / filename).stat().st_size != expected['bytes']):
                raise ValueError(f"PPO frozen prepared asset mismatch: {filename}")
        for filename, expected in locked_assets['smoke_data'].items():
            if (sha256_file(data / filename) != expected['sha256']
                    or len(data_manifest['files'][filename]['case_ids']) != expected['count']):
                raise ValueError(f"PPO frozen smoke data mismatch: {filename}")
    if not any(model.glob("*.safetensors")):
        raise ValueError("Prepared model directory contains no safetensors weights")
    tokenizer_config = model / "tokenizer_config.json"
    if not tokenizer_config.is_file() or sha256_file(tokenizer_config) != model_manifest.get("prepared_tokenizer_config_sha256"):
        raise ValueError("Prepared tokenizer configuration is missing or differs from its manifest; prepare the model again")
    template = model / "chat_template.jinja"
    if args.profile == "llama3b":
        if not template.is_file() or sha256_file(template) != model_manifest.get("template_sha256"):
            raise ValueError("Prepared Llama chat template is missing or differs from its manifest")
        if read_json(tokenizer_config).get("chat_template") != template.read_text():
            raise ValueError("Llama actor tokenizer and rollout chat template differ")
    data_manifest = read_json(data / "data_manifest.json")
    suffix = "smoke" if args.phase in ("smoke", "smoke-reload-eval") else "main"
    for name in (f"train_{suffix}.parquet", f"eval_{suffix}.parquet", f"eval_{suffix}.jsonl"):
        entry = data_manifest["files"][name]
        if sha256_file(data / name) != entry["sha256"]:
            raise ValueError(f"Dataset changed since preparation: {name}")
    train_ids = set(data_manifest["files"][f"train_{suffix}.parquet"]["case_ids"])
    eval_ids = set(data_manifest["files"][f"eval_{suffix}.parquet"]["case_ids"])
    if train_ids & eval_ids:
        raise ValueError("Training/evaluation case IDs overlap")
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else None
    if args.phase in ("reload-eval", "smoke-reload-eval"):
        if checkpoint is None or not re.fullmatch(r"global_step_\d+", checkpoint.name) or not (checkpoint / "actor").is_dir():
            raise ValueError(f"{args.phase} requires --checkpoint pointing to a saved global_step_N directory")
        if args.phase == "smoke-reload-eval" and checkpoint.name != "global_step_2":
            raise ValueError("smoke-reload-eval requires the completed two-step smoke checkpoint, global_step_2")
        latest = checkpoint.parent / "latest_checkpointed_iteration.txt"
        if not latest.is_file() or int(latest.read_text().strip()) < int(checkpoint.name.rsplit("_", 1)[1]):
            raise ValueError("Checkpoint has no completed native checkpoint receipt")
        source_run = checkpoint.parent.parent
        source_manifest = read_json(source_run / "manifest.json")
        ppo_contract.verify_identity(source_manifest, algorithm)
        if algorithm == ppo_contract.PPO:
            identity_path = checkpoint / "algorithm_manifest.json"
            if sha256_file(identity_path) != getattr(args, "checkpoint_identity_sha256", None):
                raise ValueError("Checkpoint identity differs from the pre-dispatch reviewed identity")
            native_identity = read_json(identity_path)
            ppo_contract.validate_checkpoint(native_identity, algorithm, checkpoint.name)
            if sha256_file(source_run / "resolved_config.json") != native_identity['resolved_config_sha256']:
                raise ValueError("Checkpoint source resolved configuration changed")
            for filename, expected in native_identity.get("bundle_source_sha256", {}).items():
                if sha256_file(bundle / filename) != expected:
                    raise ValueError(f"PPO reload source differs from training: {filename}")
            ppo_contract.verify_inventory(checkpoint, native_identity)
        expected_source_phase = "smoke" if args.phase == "smoke-reload-eval" else "main"
        if source_manifest.get("phase") != expected_source_phase or source_manifest.get("profile") != args.profile:
            raise ValueError(f"{args.phase} expects a {expected_source_phase}-run checkpoint from the same model profile")
        original_agent_hash = source_manifest.get("bundle_source_sha256", {}).get("observed_calc_agent.py")
        if original_agent_hash and sha256_file(bundle / "observed_calc_agent.py") != original_agent_hash:
            raise ValueError("Agent harness changed after the source training run; restore the recorded harness or run a fresh experiment")
        source_model = read_json(source_run / "model_manifest.json")
        if any(source_model.get(key) != model_manifest.get(key) for key in ("revision", "template_sha256", "prepared_tokenizer_config_sha256")):
            raise ValueError("Checkpoint source model/template differs from the prepared model")
        source_data = read_json(source_run / "data_manifest.json")
        for name in (f"train_{suffix}.parquet", f"eval_{suffix}.parquet"):
            if source_data["files"][name]["sha256"] != data_manifest["files"][name]["sha256"]:
                raise ValueError("Checkpoint and evaluation use different frozen data")
    elif checkpoint is not None:
        raise ValueError("--checkpoint is only accepted for reload-eval or smoke-reload-eval; other phases start from base weights")
    if not (bundle / "observed_calc_agent.py").is_file():
        raise ValueError("observed_calc_agent.py is missing from the starter bundle")
    return repo, model, data, python, model_manifest, data_manifest, checkpoint


WORKER = '''import json, sys
from importlib.metadata import distributions
from pathlib import Path
from omegaconf import OmegaConf
import ray
plan = json.loads(Path(sys.argv[1]).read_text())
packages = {dist.metadata["Name"]: dist.version for dist in distributions() if dist.metadata["Name"]}
Path(plan["resolved_config"]).with_name("resolved_packages.json").write_text(json.dumps(packages, indent=2, sort_keys=True) + "\\n")
sys.path.insert(0, plan["calc_dir"])
from train_calc_agent import build_config, train
kwargs = dict(model=plan["model"], agl_base_url=plan["gateway"],
              agl_key=plan["local_key"], run_name=plan["run_name"],
              config_overrides=plan["overrides"])
if plan.get("algorithm") == "ppo-mc-v1":
    from ppo_contract import runtime_env_overrides
    from ppo_runtime import source_identity
    source = source_identity()
    from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
    kwargs["config_overrides"] = runtime_env_overrides(plan["overrides"], get_ppo_ray_runtime_env())
config = build_config(**kwargs)
Path(plan["resolved_config"]).write_text(json.dumps(OmegaConf.to_container(config, resolve=True), indent=2) + "\\n")
if plan.get("algorithm") == "ppo-mc-v1":
    from ppo_contract import validate_resolved
    from ppo_runtime import runtime_assertions
    validate_resolved(OmegaConf.to_container(config, resolve=True), plan["model"])
    Path(plan["resolved_config"]).with_name("runtime_source_identity.json").write_text(json.dumps(source, indent=2) + "\\n")
    runtime_assertions()
try:
    train(train_file=plan["train_file"], val_file=plan["eval_file"], **kwargs)
finally:
    ray.shutdown()
'''


def make_plan(args, bundle: Path, repo: Path, model: Path, data: Path, python: Path, run: Path, checkpoint: Path | None, port: int):
    algorithm = getattr(args, "algorithm", ppo_contract.GRPO)
    identity = ppo_contract.validate_selection(algorithm, args.phase, args.profile, run.name,
                                               getattr(args, "protocol", None))
    profile = PROFILES[args.profile]
    is_smoke = args.phase in ("smoke", "smoke-reload-eval")
    steps = 2 if is_smoke else 20
    suffix = "smoke" if is_smoke else "main"
    config = {
        "data.train_batch_size": 8, "data.val_batch_size": None, "data.seed": 329,
        "data.max_prompt_length": 2048, "data.max_response_length": 1024,
        "actor_rollout_ref.rollout.n": 4,
        "actor_rollout_ref.rollout.val_kwargs.n": 1,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.max_model_len": 4096,
        "actor_rollout_ref.rollout.max_num_seqs": 32,
        "actor_rollout_ref.rollout.gpu_memory_utilization": 0.5,
        "actor_rollout_ref.rollout.enforce_eager": True,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice": True,
        "actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser": profile["parser"],
        "actor_rollout_ref.actor.ppo_mini_batch_size": 8,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.fsdp_config.param_offload": True,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "agentlightning.trace_aggregator.trajectory_max_prompt_length": 2048,
        "agentlightning.trace_aggregator.trajectory_max_response_length": 2048,
        "agentlightning.local.agent_class": "observed_calc_agent.Agent",
        "agentlightning.local.env_map": {"QUESTION": "input.question", "RESULT": "input.result", "AGL_CASE_ID": "input.case_id", "AGL_DATA_SPLIT": "input.split", "EVAL_SPLIT": "input.split"},
        "agentlightning.rollout_timeout_seconds": 300,
        "agentlightning.async_rollout.enabled": False,
        "trainer.project_name": "cs329z-agent-lightning",
        "trainer.experiment_name": run.name,
        "trainer.default_local_dir": str(run / "checkpoints"),
        "trainer.total_training_steps": steps,
        "trainer.total_epochs": 100,
        "trainer.n_gpus_per_node": 1,
        "trainer.nnodes": 1,
        "trainer.val_before_train": True,
        "trainer.val_only": args.phase in ("baseline-eval", "reload-eval", "smoke-reload-eval", "export-eval"),
        # Smoke validates at both steps; main evaluates only initially and finally.
        "trainer.test_freq": 1 if is_smoke else 20,
        "trainer.save_freq": 1 if is_smoke else 5,
        "trainer.max_actor_ckpt_to_keep": 2,
        "trainer.resume_mode": "resume_path" if checkpoint else "disable",
        "ray_kwargs.ray_init.address": "local",
        "ray_kwargs.ray_init.include_dashboard": False,
        "ray_kwargs.ray_init.dashboard_host": "127.0.0.1",
    }
    if args.profile == "llama3b":
        config["actor_rollout_ref.rollout.engine_kwargs.vllm.chat_template"] = str(model / "chat_template.jinja")
    if checkpoint:
        config["trainer.resume_from_path"] = str(checkpoint)
    if args.hooks:
        hooks = args.hooks.expanduser().resolve()
        if not hooks.is_file():
            raise ValueError(f"Hook file does not exist: {hooks}")
        config["agentlightning.hooks"] = str(hooks)
    if algorithm == ppo_contract.PPO:
        if args.hooks:
            raise ValueError("PPO does not accept arbitrary rollout hooks")
        config.update(ppo_contract.overrides(model))
        config["lab_protocol"] = {**identity, "description": ppo_contract.DESCRIPTION,
                                  "comparison_qualified": False, "critic_init_seed": ppo_contract.CRITIC_SEED}
        config["ray_kwargs.ray_init.runtime_env.worker_process_setup_hook"] = "ppo_runtime.register_in_worker"
        config["ray_kwargs.ray_init.runtime_env.env_vars"] = {
            "AGL_PPO_RUN_DIR": str(run), "AGL_PPO_AUDIT_DIR": str(run / "ppo-audit"),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    gateway = f"http://127.0.0.1:{port}"
    plan = {"run_name": run.name, "phase": args.phase, "profile": args.profile, "source_commit": AGL_COMMIT,
            **identity,
            "calc_dir": str(repo / "examples/calc_x"), "model": str(model), "gateway": gateway,
            "local_key": LOCAL_KEY, "train_file": str(data / f"train_{suffix}.parquet"),
            "eval_file": str(data / f"eval_{suffix}.parquet"), "resolved_config": str(run / "resolved_config.json"),
            "overrides": [dot(k, v) for k, v in config.items()]}
    commands = {
        "gateway": [str(python), "-m", "agentlightning.server", "host=127.0.0.1", f"port={port}", dot("key", LOCAL_KEY), dot("default_proxy.model_name", str(model)), "default_proxy.train.temperature=1.0", "default_proxy.val.temperature=0.0"],
        "controller": [str(python), "-m", "agentlightning.controller", "runner_type=local", dot("agl_server.url", gateway), dot("agl_server.key", LOCAL_KEY), "local_runner.maximum_size=8", "local_runner.poll_interval=1"],
        "trainer": [str(python), "-u", str(run / "trainer_worker.py"), str(run / "run_plan.json")],
    }
    return plan, commands


def stop_owned(processes: list[subprocess.Popen]) -> None:
    """Signal only process groups created by this launcher; never global pkill/ray stop."""
    for sig, seconds in ((signal.SIGINT, 12), (signal.SIGTERM, 5), (signal.SIGKILL, 2)):
        for proc in reversed(processes):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + seconds
        while any(proc.poll() is None for proc in processes) and time.monotonic() < deadline:
            time.sleep(0.2)


def run_services(commands, plan, run: Path, repo: Path, bundle: Path, walltime: float):
    env = dict(os.environ)
    matched = plan.get("protocol") == "matched-rollout-mean-v2"
    run_id = plan["request"]["stage"]["wandb"]["id"] if matched else secrets.token_hex(8)
    env.update({"PYTHONUNBUFFERED": "1", "RAY_ADDRESS": "local", "WANDB_RUN_ID": run_id,
                "WANDB_RESUME": "never", "AGL_EXPERIMENT_ID": run.name, "AGL_WANDB_RUN_ID": run_id,
                "AGL_TRACE_DIR": str(run / "traces"), "AGL_PHASE": plan["phase"],
                "AGL_MAX_TOKENS": "1024", "AGL_POLICY_VERSION": plan["policy_version"],
                "AGL_MODEL_PROFILE": plan["profile"]})
    env["PYTHONPATH"] = os.pathsep.join([str(bundle), str(repo), plan["calc_dir"], env.get("PYTHONPATH", "")])
    if plan.get("algorithm") == ppo_contract.PPO:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", AGL_PPO_RUN_DIR=str(run),
                   AGL_PPO_AUDIT_DIR=str(run / "ppo-audit"))
    if matched:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", AGL_PPO_RUN_DIR=str(run),
                   AGL_PPO_AUDIT_DIR=str(run / "audit"), AGL_MATCHED_REQUEST=json.dumps(plan["request"]))
        for key in ("XDG_CACHE_HOME", "TRITON_CACHE_DIR", "VLLM_CACHE_ROOT", "HF_HOME", "WANDB_DIR"):
            directory = run / "temporary" / key.lower()
            directory.mkdir(parents=True)
            env[key] = str(directory)
    if matched:
        env["PYTHONHASHSEED"] = str(plan["request"]["job"]["seed_streams"]["python"])
        env['WANDB_ENTITY'] = plan['request']['stage']['wandb']['entity']
        env['WANDB_PROJECT'] = plan['request']['stage']['wandb']['project']
        if plan['request']['plan'].get('schema') == 'reader-reference-v1':
            env['AGL_EXPERIMENT_ID'] = plan['request']['job']['job_id']
    # Ray and vLLM Unix sockets must fit sockaddr_un.sun_path. Set these before
    # exec so children initialize tempfile/vLLM caches with short local paths.
    # Explicit /tmp also avoids an inherited or cached long tempfile directory.
    ray_tmp = Path(tempfile.mkdtemp(prefix=("agl-"+run_id[:12]+"-") if matched else "agl-ray-", dir="/tmp"))
    env["RAY_TMPDIR"] = str(ray_tmp)
    env["TMPDIR"] = env["VLLM_RPC_BASE_PATH"] = str(ray_tmp)
    dump_json(run / "observability.json", {"experiment_id": env['AGL_EXPERIMENT_ID'], "wandb_run_id": run_id,
                                          "wandb_project": plan['request']['stage']['wandb']['project'] if matched else "cs329z-agent-lightning", "ray_temp_directory": str(ray_tmp),
                                          **({"wandb": plan["request"]["stage"]["wandb"], "request_sha256":
                                              hashlib.sha256(json.dumps(plan["request"], sort_keys=True).encode()).hexdigest()}
                                             if matched else {})})
    (run / "traces").mkdir()
    processes: list[subprocess.Popen] = []
    log_handles = []
    status, exit_code = "failed", None
    deadline = time.monotonic() + walltime * 60

    def start(name):
        handle = (run / "logs" / f"{name}.log").open("w")
        log_handles.append(handle)
        proc = subprocess.Popen(commands[name], cwd=plan["calc_dir"], env=env, stdin=subprocess.DEVNULL,
                                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(proc)
        return proc

    try:
        server = start("gateway")
        ready_deadline = min(deadline, time.monotonic() + 90)
        while True:
            if server.poll() is not None:
                raise RuntimeError("Gateway exited; inspect logs/gateway.log")
            try:
                with urllib.request.urlopen(plan["gateway"] + "/healthz", timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() >= ready_deadline:
                raise TimeoutError("Gateway readiness timed out")
            time.sleep(0.5)
        controller = start("controller")
        trainer = start("trainer")
        while trainer.poll() is None:
            if server.poll() is not None or controller.poll() is not None:
                raise RuntimeError("Gateway/controller exited while training; inspect service logs")
            if time.monotonic() >= deadline:
                raise TimeoutError("Configured wall-clock limit reached")
            time.sleep(1)
        exit_code = trainer.returncode
        status = "completed" if exit_code == 0 else "trainer_failed"
        if exit_code:
            error = f"Trainer exited {exit_code}; inspect {run / 'logs/trainer.log'}\n"
            (run / "error.txt").write_text(error)
            print(error, file=sys.stderr, end="", flush=True)
            # The parent launcher captures stderr, while the worker traceback
            # lives in a separate file. Surface its tail before SystemExit.
            with (run / "logs/trainer.log").open("rb") as log:
                log.seek(0, os.SEEK_END)
                log.seek(max(0, log.tell() - 16384))
                print(log.read().decode(errors="replace"), file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        status, exit_code = "interrupted", 130
    except TimeoutError as exc:
        status, exit_code = "walltime_or_readiness_timeout", 124
        (run / "error.txt").write_text(str(exc) + "\n")
    except Exception as exc:
        status, exit_code = "failed", 1
        (run / "error.txt").write_text(str(exc) + "\n")
    finally:
        stop_owned(processes)
        for handle in log_handles:
            handle.close()
        checkpoints = sorted((run / "checkpoints").glob("global_step_*"), key=lambda p: int(p.name.rsplit("_", 1)[1]))
        tracker = run / "checkpoints/latest_checkpointed_iteration.txt"
        latest_completed = None
        if tracker.is_file() and tracker.read_text().strip().isdigit():
            candidate = tracker.parent / f"global_step_{int(tracker.read_text().strip())}"
            if (candidate / "actor").is_dir():
                latest_completed = candidate
            if plan.get("algorithm") == ppo_contract.PPO and not (candidate / "algorithm_manifest.json").is_file():
                latest_completed = None
        dump_json(run / "result.json", {"phase": plan["phase"], "status": status, "exit_code": exit_code,
                                       **({"algorithm": plan["algorithm"], "protocol": plan["protocol"]} if matched
                                          else ppo_contract.identity(plan.get("algorithm", ppo_contract.GRPO))),
                                       "finished_at": datetime.now(timezone.utc).isoformat(),
                                       "checkpoints": [str(p) for p in checkpoints],
                                       "latest_checkpoint": str(latest_completed) if latest_completed else None,
                                       "restored_checkpoint": next((json.loads(v.split("=", 1)[1]) for v in plan["overrides"] if v.startswith("trainer.resume_from_path=")), None),
                                       "resolved_config_saved": (run / "resolved_config.json").is_file()})
    return exit_code or 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--name", help="Exact new run subdirectory, 1-80 letters/digits/dots/underscores/hyphens")
    parser.add_argument("--profile", choices=PROFILES, default="llama3b")
    parser.add_argument("--algorithm", choices=ppo_contract.ALGORITHMS, default=ppo_contract.GRPO)
    parser.add_argument("--protocol", help="Optional exact algorithm protocol identity; mismatches rejected")
    parser.add_argument("--checkpoint-identity-sha256", help="Required PPO reload manifest digest, checked before loading")
    parser.add_argument("--phase", choices=("smoke", "smoke-reload-eval", "baseline-eval", "main", "reload-eval", "export-eval"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--python", type=Path, help="Defaults to REPO/.venv/bin/python")
    parser.add_argument("--hooks", type=Path, help="Optional rollout hook; none is installed by default")
    parser.add_argument("--walltime-minutes", type=float)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and write the plan; start no services")
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("Use Linux or WSL, not native Windows")
    if args.name is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", args.name):
        parser.error("--name must start with a letter/digit and use only letters, digits, dots, underscores, or hyphens (1-80 characters)")
    walltime = args.walltime_minutes if args.walltime_minutes is not None else (180 if args.phase == "main" else 45)
    if walltime <= 0:
        parser.error("Wall-clock limit must be positive")
    bundle = Path(__file__).resolve().parent
    try:
        repo, model, data, python, model_manifest, data_manifest, checkpoint = validate_inputs(args, bundle)
        output = args.output_root.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        name = args.name or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{args.phase}-{args.profile}-" + secrets.token_hex(3))
        run = output / name
        if run.exists():
            raise FileExistsError(f"Run directory already exists; choose a new --name: {run}")
        run.mkdir()
        (run / "logs").mkdir()
        plan, commands = make_plan(args, bundle, repo, model, data, python, run, checkpoint, free_port())
        plan["model_revision"] = model_manifest.get("revision")
        if args.phase == "export-eval":
            plan["artifact_type"] = "trained_full_parameter_export"
            plan["base_model_revision"] = model_manifest["base_revision"]
        plan["policy_version"] = ("base:" + str(model_manifest.get("revision")) if args.phase == "baseline-eval"
                                  else str(checkpoint) if checkpoint else "training-policy-version-not-recorded")
        if args.phase == "export-eval":
            plan["policy_version"] = "export:sha256:" + sha256_file(model / "export_manifest.json")
        dump_json(run / "run_plan.json", plan)
        dump_json(run / "commands.json", commands)
        dump_json(run / ("export_manifest.json" if args.phase == "export-eval" else "model_manifest.json"), model_manifest)
        dump_json(run / "data_manifest.json", data_manifest)
        snapshot = run / "code"
        snapshot.mkdir()
        source_hashes = {}
        for source in sorted(bundle.glob("*.py")):
            shutil.copy2(source, snapshot / source.name)
            source_hashes[source.name] = sha256_file(source)
        if args.algorithm == ppo_contract.PPO:
            source = bundle / "ppo_source_lock.json"
            shutil.copy2(source, snapshot / source.name)
            source_hashes[source.name] = sha256_file(source)
        if (model / "chat_template.jinja").is_file():
            shutil.copy2(model / "chat_template.jinja", snapshot / "prepared_chat_template.jinja")
        dump_json(run / "manifest.json", {"source_commit": AGL_COMMIT, "profile": args.profile,
                                          **ppo_contract.identity(args.algorithm),
                                          "plan_sha256": sha256_file(run / "run_plan.json"),
                                          "profile_status": PROFILES[args.profile]["status"], "phase": args.phase,
                                          "launcher_sha256": sha256_file(Path(__file__)), "bundle_source_sha256": source_hashes,
                                          "walltime_minutes": walltime,
                                          "dry_run": args.dry_run, "python": str(python)})
        (run / "trainer_worker.py").write_text(WORKER)
        print(f"Run directory: {run}", flush=True)
        if args.dry_run:
            dump_json(run / "result.json", {"phase": args.phase, "status": "dry_run_prepared", "services_started": False,
                                           **ppo_contract.identity(args.algorithm)})
            print("Plan saved. No services or GPU work started.")
            return
        print(f"Logs: {run / 'logs'}", flush=True)
        def interrupted(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
        code = run_services(commands, plan, run, repo, bundle, walltime)
        print(f"Status: {read_json(run / 'result.json')['status']}. Result: {run / 'result.json'}")
        raise SystemExit(code)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
