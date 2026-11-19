"""Offline checks of cost/persistence boundaries. No Modal RPCs or GPU dependencies."""
import json
from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import modal_app as runner


class RequestTests(unittest.TestCase):
    def test_reload_cannot_escape_volume_or_choose_unfinished_smoke(self):
        for checkpoint in ("/data/runs/a/checkpoints/global_step_2", "../secret", "runs/../checkpoints/global_step_2",
                           "runs/a/checkpoints/global_step_1"):
            with self.subTest(checkpoint=checkpoint), self.assertRaises(ValueError):
                runner.request("smoke-reload-eval", "check-01", checkpoint=checkpoint, wandb_entity="team")
        req = runner.request("smoke-reload-eval", "check-01", checkpoint="runs/smoke-01/checkpoints/global_step_2",
                             wandb_entity="team")
        command = runner.launch_command(req)
        self.assertEqual(command[command.index("--checkpoint") + 1], "/data/runs/smoke-01/checkpoints/global_step_2")
        self.assertEqual(command[command.index("--python") + 1], "/opt/agent-lightning/.venv/bin/python")

    def test_invalid_request_never_dispatches_remote_call(self):
        with patch.object(runner.gpu_phase, "remote") as remote:
            for kwargs in ({"action": "unknown", "name": "x"}, {"action": "smoke", "name": "../x"},
                           {"action": "main", "name": "x"}, {"action": "prepare", "name": "x", "weave": True}):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    runner.main(**kwargs)
            remote.assert_not_called()

    def test_weave_requires_explicit_opt_in(self):
        req = runner.request("smoke", "smoke-01", wandb_entity="team")
        with patch.dict(os.environ, {"AGL_WEAVE_PROJECT": "inherited/project"}):
            self.assertNotIn("AGL_WEAVE_PROJECT", runner.environment(req))
        req["weave"] = True
        self.assertEqual(runner.environment(req)["AGL_WEAVE_PROJECT"], "team/cs329z-agent-lightning")


class PersistenceTests(unittest.TestCase):
    def test_failed_phase_commits_receipt_and_name_cannot_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, build = root / "bundle", root / "build"
            bundle.mkdir()
            build.mkdir()
            (bundle / "example.py").write_text("# fixture\n")
            (build / "packages.txt").write_text("fixture==1\n")
            req = runner.request("prepare", "failed-01")
            committed = []

            class Volume:
                def reload(self):
                    pass

                def commit(self):
                    committed.append(json.loads((root / "submissions/failed-01/receipt.json").read_text())["status"])

            calls = []

            def failure(request, record):
                self.assertEqual(committed, ["started"])
                calls.append(request["name"])
                (record / "failure.log").write_text("retained diagnostic\n")
                raise RuntimeError("simulated training failure")

            with patch.multiple(runner, STATE=root, BUNDLE=bundle, BUILD_EVIDENCE=build, volume=Volume()):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    runner.recorded(req, failure)
                self.assertEqual(committed, ["started", "failed"])
                with self.assertRaises(FileExistsError):
                    runner.recorded(req, failure)
            self.assertEqual(calls, ["failed-01"])
            self.assertTrue((root / "submissions/failed-01/failure.log").is_file())
            source = root / "submissions/failed-01/source"
            self.assertEqual((source / "modal_build.py").read_bytes(),
                             (runner.LOCAL / "modal_build.py").read_bytes())
            hashes = json.loads((source.parent / "source-sha256.json").read_text())
            self.assertEqual(hashes["modal_build.py"], runner.sha256(source / "modal_build.py"))

    def test_export_keeps_evidence_and_excludes_weights_credentials_and_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = {
                "runs/train/traces/trajectory.jsonl": "{}\n",
                "runs/train/checkpoints/global_step_20/actor/model.pt": "weights",
                "models/llama3b/model_manifest.json": "{}",
                "models/llama3b/model.safetensors": "weights",
                "staging/failed/model-download/partial.safetensors": "partial weights",
                "cache/huggingface/token": "secret fixture",
                "evidence/comparison/summary.json": "{}",
                "submissions/failed/receipt.json": "{}",
                "data-small/data_manifest.json": "{}",
            }
            for relative, data in files.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(data)
            (root / "runs/unsafe-link").symlink_to(root / "cache/huggingface/token")
            with patch.object(runner, "STATE", root):
                result = runner.export_evidence("export-01")
            with tarfile.open(result["archive"]) as archive:
                names = set(archive.getnames())
            self.assertIn("runs/train/traces/trajectory.jsonl", names)
            self.assertIn("models/llama3b/model_manifest.json", names)
            self.assertIn("evidence/comparison/summary.json", names)
            self.assertFalse(any("checkpoints" in name or name.startswith(("cache/", "staging/")) for name in names))
            self.assertNotIn("models/llama3b/model.safetensors", names)
            self.assertNotIn("runs/unsafe-link", names)

    def test_command_timeout_allows_child_to_close_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stopped = root / "stopped"
            service_log = root / "trainer.log"
            script = (
                "import pathlib, signal, sys, time; "
                "signal.signal(signal.SIGTERM, lambda *_: (pathlib.Path(sys.argv[1]).write_text('closed'), sys.exit(0))); "
                "pathlib.Path(sys.argv[2]).write_text('trainer ready\\n'); print('launcher ready', flush=True); "
                "time.sleep(60)"
            )
            streamed = StringIO()
            with redirect_stdout(streamed), self.assertRaises(subprocess.TimeoutExpired):
                runner.run_command([sys.executable, "-c", script, stopped, service_log], root / "command.log",
                                   dict(os.environ), 1, [service_log])
            self.assertEqual(stopped.read_text(), "closed")
            self.assertIn("launcher ready", streamed.getvalue())
            self.assertIn("trainer ready", streamed.getvalue())

    def test_dataset_archive_copies_only_expected_members(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "extracted"
            target.mkdir()
            archive = root / "data.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("release/train.parquet", "train")
                zipped.writestr("release/test.parquet", "test")
                zipped.writestr("../unexpected.txt", "untrusted")
            runner.extract_parquets(archive, target)
            self.assertEqual({p.name for p in target.iterdir()}, {"train.parquet", "test.parquet"})
            self.assertFalse((root / "unexpected.txt").exists())


if __name__ == "__main__":
    unittest.main()
