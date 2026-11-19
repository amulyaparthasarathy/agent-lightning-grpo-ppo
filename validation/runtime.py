#!/usr/bin/env python3
"""Synthetic validation, executed in the selected REMOTE interpreter only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import secrets
import subprocess
import sys
import tempfile
import time

PROJECT = "agent-lightning-validation"
BUILD_EVIDENCE = Path("/opt/agl-build/evidence")
MODEL = "meta-llama/Llama-3.2-3B-Instruct"
REVISION = "0cb88a4f764b7a12671c53f0838cd831a0843b95"
PINS = {"agentlightning": "1.0.3", "verl": "0.8.0", "vllm": "0.22.0",
        "flash-attn": "2.8.3", "autogen-agentchat": "0.7.5", "autogen-ext": "0.7.5",
        "openai": "2.54.0", "weave": "0.54.1", "mcp-server-calculator": "0.2.1",
        "googleapis-common-protos": "1.75.0", "gdown": "5.2.0"}
CPU_MODULES = ("torch", "transformers", "numpy", "pyarrow", "datasets",
               "ray", "agentlightning", "verl",
               "wandb", "weave", "autogen_agentchat", "autogen_ext.models.openai",
               "autogen_ext.tools.mcp", "mcp", "openai", "sympy", "httpx",
               "omegaconf", "pandas", "huggingface_hub",
               "ml_dtypes", "cv2", "xgrammar", "google.api_core", "google.protobuf",
               "tensorboard", "proto")


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def inventory_training_environment(output: Path) -> dict[str, str]:
    """Use only dist-info in the training venv and retain metadata provenance."""
    site = (Path(sys.prefix) / "lib"
            / f"python{sys.version_info[0]}.{sys.version_info[1]}"
            / "site-packages")

    def normalize(name):
        return name.lower().replace("_", "-").replace(".", "-")

    def describe(dist):
        return {"version": dist.version,
                "metadata_path": str(getattr(dist, "_path", "<unknown>"))}

    scoped = {}
    for dist in metadata.distributions(path=[str(site)]):
        name = dist.metadata.get("Name")
        if name:
            scoped.setdefault(normalize(name), []).append(describe(dist))

    visible_protobuf = []
    for dist in metadata.distributions():
        if normalize(dist.metadata.get("Name", "")) == "protobuf":
            visible_protobuf.append(describe(dist))

    ambiguous = {name: records for name, records in scoped.items()
                 if len(records) != 1}
    save(output / "package-inventory-details.json", {
        "venv_site_packages": str(site),
        "protobuf_in_selected_venv": scoped.get("protobuf", []),
        "protobuf_on_interpreter_search_path": visible_protobuf,
        "ambiguous_venv_distributions": ambiguous,
    })
    packages = {name: records[0]["version"] for name, records in scoped.items()
                if len(records) == 1}
    save(output / "packages.json", packages)
    if ambiguous:
        raise RuntimeError("Ambiguous package metadata inside training virtual environment")
    return packages


def record_import_failure(output: Path, module: str, exc: ImportError) -> dict:
    """Save a bounded and credential-redacted import error, never an env dump."""
    def scrub(value: object) -> str | None:
        if value is None:
            return None
        message = str(value)
        for key in ("HF_TOKEN", "WANDB_API_KEY"):
            secret = os.environ.get(key)
            if secret:
                message = message.replace(secret, "[REDACTED]")
        return message[:800]

    record = {
        "attempted_module": module,
        "exception_type": type(exc).__name__,
        "missing_name": scrub(getattr(exc, "name", None)),
        "exception_path": scrub(getattr(exc, "path", None)),
        "message": scrub(exc),
        "cause_type": type(exc.__cause__).__name__ if exc.__cause__ else None,
        "cause_message": scrub(exc.__cause__) if exc.__cause__ else None,
    }
    save(output / "import-failure.json", record)
    print("IMPORT_FAILED module=" + module + " exception=" + type(exc).__name__, flush=True)
    return record


def audit_arrow_dataset_roundtrip(output: Path) -> dict:
    """Exercise synthetic Datasets/Arrow/Parquet/NumPy IO on the CPU, locally."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import Dataset
    from importlib.metadata import version

    expected_values = np.array([3, 7, 11], dtype=np.int64)
    expected_labels = ["alpha", "beta", "gamma"]
    try:
        # Separate from the frozen research dataset and model cache.
        with tempfile.TemporaryDirectory(prefix="agl-data-smoke-", dir=output) as temp:
            path = Path(temp) / "synthetic.parquet"
            original = Dataset.from_dict({
                "value": expected_values.tolist(),
                "label": expected_labels,
            })
            original.to_parquet(str(path))
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError("Data smoke produced no Parquet bytes")

            arrow_values = pq.read_table(str(path)).column("value").to_numpy()
            if not np.array_equal(arrow_values, expected_values):
                raise RuntimeError("PyArrow/NumPy Parquet round-trip changed values")

            restored = Dataset.from_parquet(
                str(path), keep_in_memory=True, cache_dir=str(Path(temp) / "cache")
            )
            if (len(restored) != len(expected_values)
                    or list(restored["value"]) != expected_values.tolist()
                    or list(restored["label"]) != expected_labels):
                raise RuntimeError("Datasets Parquet round-trip changed values")
    except Exception as exc:
        message = str(exc)
        for key in ("HF_TOKEN", "WANDB_API_KEY"):
            token = os.environ.get(key)
            if token:
                message = message.replace(token, "[REDACTED]")
        save(output / "data-smoke-failure.json", {
            "step": "datasets_arrow_numpy_parquet_roundtrip",
            "exception_type": type(exc).__name__,
            "message": message[:800],
        })
        raise

    evidence = {
        "verified": True,
        "synthetic": True,
        "rows": len(expected_values),
        "numpy": np.__version__,
        "pyarrow": pa.__version__,
        "datasets": version("datasets"),
        "numpy_values_roundtripped": True,
        "datasets_parquet_roundtrip": True,
        "model_loaded": False,
    }
    save(output / "data-smoke.json", evidence)
    return evidence


def audit_training(output: Path, gpu: bool) -> dict:
    if sys.version_info[:2] != (3, 12) or sys.prefix != "/opt/agent-lightning/.venv":
        raise RuntimeError("Not running inside the declared Python 3.12 training venv")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from compatibility import classify_conflicts
    packages = inventory_training_environment(output)
    for name, expected in PINS.items():
        save(output / "dependency-progress.json", {"checking_package": name, "expected": expected})
        actual = metadata.version(name).split("+", 1)[0]
        if actual != expected:
            save(output / "dependency-progress.json", {"package": name, "expected": expected, "actual": actual})
            raise RuntimeError(f"Version mismatch for {name}; expected {expected}")
    source_path = BUILD_EVIDENCE / "base-agent-lightning-commit.txt"
    source_commit = source_path.read_text().strip() if source_path.is_file() else "<missing>"
    check = subprocess.run(["uv", "pip", "check", "--python", sys.executable],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, timeout=120, check=False)
    (output / "dependency-check.txt").write_text(check.stdout)
    decision = classify_conflicts(
        check.returncode, check.stdout, packages,
        python_minor=f"{sys.version_info[0]}.{sys.version_info[1]}",
        machine=platform.machine(), source_commit=source_commit)
    save(output / "dependency-conflicts.json", decision)
    if not decision["accepted"]:
        raise RuntimeError(
            "Unreviewed, changed or unparseable dependency conflicts or runtime identity; "
            "inspect dependency-conflicts.json and dependency-check.txt")
    imports = {}
    for module in CPU_MODULES:
        save(output / "dependency-progress.json", {"importing_module": module})
        try:
            imported = importlib.import_module(module)
        except ImportError as exc:
            record_import_failure(output, module, exc)
            raise
        if module == "google.protobuf":
            actual_runtime = getattr(imported, "__version__", None)
            expected_runtime = packages.get("protobuf")
            save(output / "protobuf-import.json", {
                "imported_version": actual_runtime,
                "venv_distribution_version": expected_runtime,
            })
            if actual_runtime != expected_runtime:
                raise RuntimeError("Imported google.protobuf disagrees with training-venv metadata")
        imports[module] = "imported"
        save(output / "imports.json", imports)
    save(output / "dependency-progress.json", {"checking": "data_interop"})
    data_interop = audit_arrow_dataset_roundtrip(output)
    result = dict(dependencies_checked=True, cpu_imports_checked=True,
                  data_interop_checked=True, data_interop=data_interop,
                  dependency_closure_consistent=decision["dependency_closure_consistent"],
                  reviewed_dependency_conflicts=decision["reviewed_dependency_conflicts"],
                  qualification=decision["qualification"],
                  native_gpu_imports_checked=False, gpu_kernels_checked=False)
    if gpu:
        importlib.import_module("vllm")
        importlib.import_module("flash_attn")
        subprocess.run([sys.executable, "/opt/agent-lightning-lab/experiment/check_gpu.py",
                        "--output", str(output / "gpu_preflight.json")], check=True, timeout=600)
        result.update(native_gpu_imports_checked=True, gpu_kernels_checked=True)
    return result


def check_hugging_face(output: Path) -> dict:
    from huggingface_hub import HfApi, hf_hub_download
    token = os.environ["HF_TOKEN"]
    HfApi().whoami(token=token)  # Deliberately do not record the identity/token response.
    config = hf_hub_download(MODEL, "config.json", revision=REVISION, token=token,
                             local_dir=str(output / "hf-config"), force_download=True,
                             etag_timeout=30)
    if json.loads(Path(config).read_text()).get("model_type") != "llama":
        raise RuntimeError("Unexpected gated model configuration")
    return dict(authenticated=True, gated_config_downloaded=True, model=MODEL,
                revision=REVISION, model_weights_downloaded=False, hub_write_tested=False)


def verify_wandb(wandb, path: str, expected: int, attempts: int = 6, qualification: dict | None = None) -> dict:
    """Server-side readback, not merely a successful SDK log() call."""
    for attempt in range(attempts):
        try:
            remote = wandb.Api(timeout=20).run(path)
            summary = dict(remote.summary)
            qualification_ok = (
                qualification is None or (
                    summary.get("dependency_closure_consistent") is qualification["dependency_closure_consistent"]
                    and summary.get("reviewed_dependency_conflicts") == qualification["reviewed_dependency_conflicts"]
                    and summary.get("qualification") == qualification["qualification"]
                )
            )
            if (remote.state == "finished" and summary.get("validation_completed") is True
                    and summary.get("heartbeats_expected") == expected
                    and summary.get("validation/heartbeat") == expected - 1
                    and qualification_ok):
                return dict(finished_state_verified=True, summary_verified=True,
                            last_heartbeat_verified=True, full_history_verified=False)
        except Exception:
            pass  # Do not echo potentially sensitive HTTP errors. Retry reads only.
        if attempt + 1 < attempts:
            time.sleep(5)
    raise RuntimeError("W&B server readback not confirmed; inspect the run before retrying")


def run(kind: str, name: str, entity: str, hold_seconds: int, output: Path) -> dict:
    report = dict(status="started", kind=kind, name=name, synthetic=True,
                  started_at=datetime.now(timezone.utc).isoformat(),
                  python=platform.python_version(), executable=sys.executable,
                  modal_image_id=os.environ.get("MODAL_IMAGE_ID"),
                  model_loaded=False, rl_update_verified=False, checkpoint_reload_verified=False,
                  weave_delivery_verified=False, disconnect_observed=False, checks={})
    run_handle = None
    finished = False
    stage = "credentials"
    save(output / "result.json", report)
    try:
        for key in ("HF_TOKEN", "WANDB_API_KEY"):
            if not os.environ.get(key):
                raise RuntimeError(f"Required runtime Secret key missing: {key}")
        stage = "training_dependencies" if kind != "services" else "service_dependencies"
        if kind != "services":
            report["checks"]["training"] = audit_training(output, kind == "gpu")
        else:
            save(output / "packages.json", {n: metadata.version(n) for n in ("wandb", "huggingface-hub")})
        training = report["checks"].get("training", {})
        if training:
            report.update(dependency_closure_consistent=training["dependency_closure_consistent"],
                          reviewed_dependency_conflicts=training["reviewed_dependency_conflicts"],
                          qualification=training["qualification"])
            save(output / "result.json", report)
        stage = "wandb_init"
        import wandb
        os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"
        launcher = output / "launcher.json"
        source = json.loads(launcher.read_text())["request"].get("source", {}) if launcher.is_file() else {}
        wandb_config = {"synthetic": True, "kind": kind, "hold_seconds": hold_seconds,
                        "source": source}
        if training:
            wandb_config.update({
                "dependency_closure_consistent": training["dependency_closure_consistent"],
                "reviewed_dependency_conflicts": training["reviewed_dependency_conflicts"],
                "qualification": training["qualification"],
            })
        run_handle = wandb.init(entity=entity, project=PROJECT, name=name,
                                id=secrets.token_hex(8), resume="never", mode="online",
                                job_type=f"validation-{kind}", tags=["synthetic", "validation"],
                                config=wandb_config,
                                dir=str(output), settings=wandb.Settings(
                                    init_timeout=60, disable_git=True, disable_code=True))
        if run_handle is None:
            raise RuntimeError("W&B did not return an online run")
        path = f"{entity}/{PROJECT}/{run_handle.id}"
        report.update(wandb_run_path=path, wandb_url=run_handle.url)
        save(output / "result.json", report)
        print(f"REMOTE_STARTED {name}; W&B: {run_handle.url}", flush=True)
        stage = "hugging_face"
        report["checks"]["hugging_face"] = check_hugging_face(output)
        stage = "heartbeat"
        start = time.monotonic()
        step = 0
        while True:
            elapsed = time.monotonic() - start
            tick = {"validation/heartbeat": step, "validation/elapsed_seconds": elapsed,
                    "validation/unix_time": time.time()}
            run_handle.log(tick, step=step)
            with (output / "heartbeat.jsonl").open("a") as log:
                log.write(json.dumps(tick) + "\n")
            print(f"HEARTBEAT {step} elapsed={elapsed:.1f}s", flush=True)
            step += 1
            if elapsed >= hold_seconds:
                break
            time.sleep(min(10, hold_seconds - elapsed))
        summary_values = {"validation_completed": True, "heartbeats_expected": step,
                          "synthetic": True}
        if training:
            summary_values.update({
                "dependency_closure_consistent": training["dependency_closure_consistent"],
                "reviewed_dependency_conflicts": training["reviewed_dependency_conflicts"],
                "qualification": training["qualification"],
            })
        run_handle.summary.update(summary_values)
        stage = "wandb_finish"
        run_handle.finish(exit_code=0)
        finished = True
        stage = "wandb_readback"
        report["checks"]["wandb"] = verify_wandb(wandb, path, step, qualification=training or None)
        status = ("diagnostic_passed_with_reviewed_exceptions"
                  if training and not training["dependency_closure_consistent"] else "passed")
        report.update(status=status, heartbeats=step)
    except Exception as exc:
        report.update(status="failed", failed_stage=stage, error_type=type(exc).__name__)
        # Error types/stages only; never serialize exception objects, tokens, or the environment.
        print(f"VALIDATION_FAILED stage={stage} type={type(exc).__name__}", flush=True)
        if run_handle is not None and not finished:
            try:
                run_handle.finish(exit_code=1)
            except Exception:
                pass
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save(output / "result.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def validation_exit_code(report: dict) -> int:
    """A diagnostic-only qualification may exit successfully only with evidence."""
    if report.get("status") == "passed":
        return 0
    training = report.get("checks", {}).get("training", {})
    if (report.get("status") == "diagnostic_passed_with_reviewed_exceptions"
            and report.get("dependency_closure_consistent") is False
            and report.get("reviewed_dependency_conflicts") == 6
            and report.get("qualification") == "diagnostic_with_reviewed_dependency_exceptions"
            and training.get("cpu_imports_checked") is True
            and report.get("checks", {}).get("wandb", {}).get("summary_verified") is True
            and report.get("checks", {}).get("hugging_face", {}).get("authenticated") is True):
        return 0
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("services", "image", "gpu"), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--hold-seconds", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.hold_seconds <= 900:
        parser.error("--hold-seconds must be 0..900")
    report = run(args.kind, args.name, args.entity, args.hold_seconds, args.output)
    raise SystemExit(validation_exit_code(report))
