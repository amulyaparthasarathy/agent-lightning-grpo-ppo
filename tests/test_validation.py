"""Offline tests only: no Modal RPC, GPU, Hub requests, or hosted W&B writes."""
from __future__ import annotations
import ast
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import validation_support as support
sys.path.insert(0, str(ROOT / "validation"))
import compatibility as compat


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime = load("validation_runtime", ROOT / "validation/runtime.py")
envcheck = load("validation_env", ROOT / "scripts/check_validation_env.py")


class Requests(unittest.TestCase):
    def test_valid(self):
        req = support.request("services", "services-01", "amulyap-stanford-university", 180)
        self.assertEqual(req["project"], "agent-lightning-validation")
        self.assertTrue(req["synthetic"])

    def test_bad_requests(self):
        for args in [("wrong", "ok", "team", 0), ("services", "../escape", "team", 0),
                     ("services", "", "team", 0), ("services", "ok", "team/project", 0),
                     ("services", "ok", "team", 901), ("services", "ok", "team", -1),
                     ("services", "ok", "team", True), ("gpu", "ok", "team", 1)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                support.request(*args)

    def test_detach_required(self):
        with patch.object(sys, "argv", ["modal", "run", "file.py"]), self.assertRaises(ValueError):
            support.require_detached()
        for flag in ("--detach", "-d"):
            with patch.object(sys, "argv", ["modal", "run", flag, "file.py"]):
                support.require_detached()

    def test_atomic_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            support.write_json(path, {"status": "passed"})
            self.assertEqual(json.loads(path.read_text())["status"], "passed")
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_source_hash_allowlist(self):
        identity = support.source_identity()
        self.assertIn("validation/runtime.py", identity["source_sha256"])
        self.assertIn("validation/compatibility.py", identity["source_sha256"])
        self.assertIn("scripts/finalize_modal_training.sh", identity["source_sha256"])
        self.assertIn("modal_build.py", identity["source_sha256"])
        self.assertFalse(any(".env" in name for name in identity["source_sha256"]))


class Runtime(unittest.TestCase):
    def fake_wandb(self):
        run = Mock(id="fake123", url="https://wandb.example/test")
        run.summary = {}
        return types.SimpleNamespace(init=Mock(return_value=run), Settings=Mock(), Api=Mock()), run

    def run_fake(self, *, hf_error=False, readback_error=False):
        wandb, run = self.fake_wandb()
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            output = Path(directory)
            stack.enter_context(patch.dict(os.environ, {"HF_TOKEN": "fake-secret-hf", "WANDB_API_KEY": "fake-secret-wb"}))
            stack.enter_context(patch.dict(sys.modules, {"wandb": wandb}))
            stack.enter_context(patch.object(runtime.metadata, "version", return_value="test"))
            stack.enter_context(patch.object(runtime, "check_hugging_face",
                side_effect=RuntimeError("fake-secret-hf") if hf_error else None,
                return_value={"authenticated": True}))
            stack.enter_context(patch.object(runtime, "verify_wandb",
                side_effect=RuntimeError("fake-secret-wb") if readback_error else None,
                return_value={"summary_verified": True}))
            captured = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            report = runtime.run("services", "test-01", "team", 0, output)
            on_disk = json.loads((output / "result.json").read_text())
            text = captured.getvalue()
            self.assertNotIn("fake-secret-hf", text)
            self.assertNotIn("fake-secret-wb", text)
            self.assertEqual(report, on_disk)
            return report, run, wandb

    def test_services_success_is_synthetic_not_rl(self):
        result, run, wandb = self.run_fake()
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["synthetic"])
        for key in ("model_loaded", "rl_update_verified", "checkpoint_reload_verified",
                    "weave_delivery_verified", "disconnect_observed"):
            self.assertFalse(result[key])
        self.assertEqual(result["heartbeats"], 1)
        self.assertEqual(wandb.init.call_args.kwargs["mode"], "online")
        self.assertEqual(wandb.init.call_args.kwargs["project"], support.PROJECT)
        run.finish.assert_called_once_with(exit_code=0)

    def test_hf_failure_marks_failed_wandb_run(self):
        result, run, _ = self.run_fake(hf_error=True)
        self.assertEqual(result["failed_stage"], "hugging_face")
        self.assertEqual(result["status"], "failed")
        run.finish.assert_called_once_with(exit_code=1)

    def test_readback_failure_not_success(self):
        result, run, _ = self.run_fake(readback_error=True)
        self.assertEqual(result["failed_stage"], "wandb_readback")
        self.assertEqual(result["status"], "failed")
        run.finish.assert_called_once_with(exit_code=0)

    def test_missing_secret_no_sdk_call(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()):
            result = runtime.run("services", "test", "team", 0, Path(directory))
        self.assertEqual(result["failed_stage"], "credentials")

    def test_readback_requires_server_finished_and_final_tick(self):
        remote = types.SimpleNamespace(state="finished", summary={"validation_completed": True,
                                       "heartbeats_expected": 3, "validation/heartbeat": 2})
        api = Mock()
        api.run.return_value = remote
        wandb = types.SimpleNamespace(Api=Mock(return_value=api))
        result = runtime.verify_wandb(wandb, "team/project/id", 3, attempts=1)
        self.assertTrue(result["summary_verified"])
        remote.summary["validation/heartbeat"] = 1
        with self.assertRaises(RuntimeError):
            runtime.verify_wandb(wandb, "team/project/id", 3, attempts=1)
        remote.summary["validation/heartbeat"] = 2
        remote.state = "running"
        with self.assertRaises(RuntimeError):
            runtime.verify_wandb(wandb, "team/project/id", 3, attempts=1)

    def test_wandb_readback_verifies_exception_metadata(self):
        qualification = {
            "dependency_closure_consistent": False,
            "reviewed_dependency_conflicts": 6,
            "qualification": "diagnostic_with_reviewed_dependency_exceptions",
        }
        summary = {
            "validation_completed": True, "heartbeats_expected": 1,
            "validation/heartbeat": 0, **qualification,
        }
        remote = types.SimpleNamespace(state="finished", summary=summary)
        api = Mock()
        api.run.return_value = remote
        wandb = types.SimpleNamespace(Api=Mock(return_value=api))
        result = runtime.verify_wandb(wandb, "team/project/id", 1,
                                     attempts=1, qualification=qualification)
        self.assertTrue(result["summary_verified"])
        for changed in ("qualification", "dependency_closure_consistent", "reviewed_dependency_conflicts"):
            with self.subTest(field=changed):
                original = summary[changed]
                summary[changed] = "wrong"
                with self.assertRaises(RuntimeError):
                    runtime.verify_wandb(wandb, "team/project/id", 1,
                                         attempts=1, qualification=qualification)
                summary[changed] = original

    def test_cpu_audit_defers_native_gpu_imports(self):
        fixture = (
            "Checked 285 packages in 12ms\nFound 6 incompatibilities\n"
            "The package `ml-dtypes` requires `numpy>=2.0.0`, but `1.26.4` is installed\n"
            "The package `opencv-python-headless` requires `numpy>=2 ; python_full_version >= '3.9'`, but `1.26.4` is installed\n"
            "The package `google-api-core` requires `protobuf>=6.33.5,<8.0.0`, but `5.29.6` is installed\n"
            "The package `tensorboard` requires `protobuf>=6.31.1,<8.0.0`, but `5.29.6` is installed\n"
            "The package `xgrammar` requires `transformers>=4.38.0,<5`, but `5.22.2` is installed\n"
            "The package `proto-plus` requires `protobuf>=6.33.5,<8.0.0`, but `5.29.6` is installed\n"
        )
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            directory = Path(directory)
            (directory / "base-agent-lightning-commit.txt").write_text(compat.UPSTREAM_COMMIT)
            stack.enter_context(patch.object(sys, "version_info", (3, 12)))
            stack.enter_context(patch.object(sys, "prefix", "/opt/agent-lightning/.venv"))
            stack.enter_context(patch.object(runtime, "BUILD_EVIDENCE", directory))
            stack.enter_context(patch.object(runtime.platform, "machine", return_value="x86_64"))
            dists = [types.SimpleNamespace(metadata={"Name": name}, version=value)
                     for name, value in compat.EXPECTED_VERSIONS.items()]
            stack.enter_context(patch.object(runtime.metadata, "distributions", return_value=dists))
            stack.enter_context(patch.object(runtime.metadata, "version", side_effect=lambda n: runtime.PINS[n]))
            stack.enter_context(patch.object(runtime.subprocess, "run", return_value=types.SimpleNamespace(
                returncode=1, stdout=fixture)))
            imports = stack.enter_context(patch.object(
                runtime.importlib, "import_module",
                side_effect=lambda name: types.SimpleNamespace(__version__="5.29.6")
                if name == "google.protobuf" else Mock()))
            data_smoke = stack.enter_context(patch.object(
                runtime, "audit_arrow_dataset_roundtrip", return_value={"verified": True}))
            result = runtime.audit_training(directory, False)
            data_smoke.assert_called_once_with(directory)
            self.assertTrue(result["data_interop_checked"])
            self.assertTrue(result["data_interop"]["verified"])
            self.assertTrue(result["cpu_imports_checked"])
            self.assertFalse(result["dependency_closure_consistent"])
            self.assertEqual(result["reviewed_dependency_conflicts"], 6)
            self.assertFalse(result["gpu_kernels_checked"])
            self.assertEqual((directory / "dependency-check.txt").read_text(), fixture)
            called = [c.args[0] for c in imports.call_args_list]
            self.assertIn("xgrammar", called)
            self.assertIn("cv2", called)
            self.assertIn("google.protobuf", called)
            self.assertNotIn("vllm", called)
            self.assertNotIn("flash_attn", called)

    def test_training_audit_rejects_wrong_interpreter(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(sys, "prefix", "/wrong"), self.assertRaises(RuntimeError):
            runtime.audit_training(Path(directory), False)


class Launcher(unittest.TestCase):
    def test_real_child_success_and_duplicate_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / "child.py"
            child.write_text('import sys,json\nfrom pathlib import Path\nout=Path(sys.argv[sys.argv.index("--output")+1])\n(out/"result.json").write_text(json.dumps({"status":"passed"}))\n')
            volume = Mock()
            with patch.object(support, "ROOT", root), patch.object(support, "RUNTIME", str(child)), contextlib.redirect_stdout(io.StringIO()):
                req = support.request("services", "real-child", "team")
                self.assertEqual(support.execute(req, sys.executable, volume)["status"], "passed")
                with self.assertRaises(FileExistsError):
                    support.execute(req, sys.executable, volume)
            receipt = json.loads((root / "runs/real-child/launcher.json").read_text())
            self.assertEqual(receipt["status"], "completed")
            self.assertGreaterEqual(volume.commit.call_count, 2)

    def test_diagnostic_status_requires_correlated_exception_evidence(self):
        for valid in (True, False):
            with self.subTest(valid=valid), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                child = root / "child.py"
                report = {
                    "status": "diagnostic_passed_with_reviewed_exceptions",
                    "dependency_closure_consistent": False,
                    "reviewed_dependency_conflicts": 6 if valid else 5,
                    "qualification": "diagnostic_with_reviewed_dependency_exceptions",
                    "checks": {"training": {"cpu_imports_checked": True}},
                }
                child.write_text(
                    'import sys, json\nfrom pathlib import Path\n'
                    'out = Path(sys.argv[sys.argv.index("--output") + 1])\n'
                    'out.joinpath("result.json").write_text(json.dumps(' + repr(report) + '))\n'
                )
                volume = Mock()
                req = support.request("image", "diag-test", "team")
                with patch.object(support, "ROOT", root), patch.object(support, "RUNTIME", str(child)), contextlib.redirect_stdout(io.StringIO()):
                    if valid:
                        result = support.execute(req, sys.executable, volume)
                        self.assertEqual(result["status"], "diagnostic_passed_with_reviewed_exceptions")
                    else:
                        with self.assertRaises(RuntimeError):
                            support.execute(req, sys.executable, volume)
                receipt = json.loads((root / "runs/diag-test/launcher.json").read_text())
                self.assertEqual(receipt["status"], "completed" if valid else "failed")

    def test_nonzero_child_failure_committed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / "child.py"
            child.write_text('raise SystemExit(7)\n')
            volume = Mock()
            with patch.object(support, "ROOT", root), patch.object(support, "RUNTIME", str(child)), contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                support.execute(support.request("services", "bad-child", "team"), sys.executable, volume)
            receipt = json.loads((root / "runs/bad-child/launcher.json").read_text())
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(receipt["exit_code"], 7)
            self.assertGreaterEqual(volume.commit.call_count, 2)

    def test_shell_enforces_detach_for_all_tests(self):
        with tempfile.TemporaryDirectory() as directory:
            fake_uv = Path(directory) / "uv"
            fake_uv.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            fake_uv.chmod(0o700)
            env = dict(os.environ, PATH=f"{directory}:{os.environ['PATH']}")
            for kind in ("services", "image", "gpu"):
                result = subprocess.run(["bash", str(ROOT / "scripts/validate_modal.sh"), kind, "test-01"],
                                        capture_output=True, text=True, check=True, env=env)
                args = result.stdout.splitlines()
                self.assertEqual(args[:5], ["run", "--locked", "modal", "run", "--detach"])
            bad = subprocess.run(["bash", str(ROOT / "scripts/validate_modal.sh"), "bad", "test"],
                                 capture_output=True, env=env)
            self.assertNotEqual(bad.returncode, 0)

    def test_image_boundaries_and_remote_spawn(self):
        light = (ROOT / "validation_services.py").read_text()
        heavy = (ROOT / "validation_training.py").read_text()
        self.assertNotIn("from modal_app", light)
        self.assertNotIn("gpu=", light)
        self.assertIn("from modal_app import image as training_image, PYTHON", heavy)
        self.assertIn("str(PYTHON)", heavy)
        self.assertIn('add_local_python_source("validation_support", "modal_app")', heavy)
        self.assertIn('"validation/compatibility.py"', heavy)
        for source in (light, heavy):
            ast.parse(source)
            self.assertIn(".spawn(req)", source)
            self.assertIn("require_detached()", source)
            self.assertNotIn("add_local_dir", source)
            self.assertNotIn("from_dotenv", source)


class CredentialFile(unittest.TestCase):
    def prepare(self, root, text):
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        (root / ".gitignore").write_text('.env\n.env.*\n!.env.example\n')
        path = root / ".env"
        path.write_text(text)
        path.chmod(0o600)
        return path

    def test_valid_ignored_private_file_and_no_value_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.prepare(root, 'HF_TOKEN=fake_hf\nWANDB_API_KEY="fake_wb"\n')
            with patch.object(envcheck, "__file__", str(root / "scripts/check_validation_env.py")), contextlib.redirect_stdout(io.StringIO()) as output:
                envcheck.check(path)
            self.assertNotIn("fake_hf", output.getvalue())
            self.assertNotIn("fake_wb", output.getvalue())

    def test_bad_values_and_duplicate_keys(self):
        cases = ['HF_TOKEN=replace_with_your_token\nWANDB_API_KEY=fake\n',
                 'HF_TOKEN=${SECRET}\nWANDB_API_KEY=fake\n',
                 'HF_TOKEN=a\nHF_TOKEN=b\nWANDB_API_KEY=fake\n',
                 'HF_TOKEN=a\nEXTRA_KEY=fake\n', 'HF_TOKEN=\nWANDB_API_KEY=fake\n']
        for text in cases:
            with self.subTest(text=text), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path = self.prepare(root, text)
                with patch.object(envcheck, "__file__", str(root / "scripts/check_validation_env.py")), self.assertRaises(ValueError):
                    envcheck.check(path)

    def test_world_readable_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.prepare(root, 'HF_TOKEN=a\nWANDB_API_KEY=b\n')
            path.chmod(0o644)
            with self.assertRaises(ValueError):
                envcheck.check(path)

    def test_tracked_secret_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.prepare(root, 'HF_TOKEN=a\nWANDB_API_KEY=b\n')
            subprocess.run(["git", "add", "-f", ".env"], cwd=root, check=True)
            with patch.object(envcheck, "__file__", str(root / "scripts/check_validation_env.py")), self.assertRaises(ValueError):
                envcheck.check(path)


if __name__ == "__main__":
    unittest.main()
