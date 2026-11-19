"""Export lineage and evaluation plans, with tiny opaque files and no Torch."""
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiment import export_artifact as export
from experiment import launch
import modal_app as runner


class ExportEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model, self.data, self.bundle = (self.root / n for n in ("model", "data", "bundle"))
        for p in (self.model, self.data, self.bundle):
            p.mkdir()
        (self.bundle / "observed_calc_agent.py").write_text("# fixture harness")
        self.harness = export.sha256(self.bundle / "observed_calc_agent.py")
        template = 'fixed fixture template: 11 Nov 2026'
        content = {"config.json": json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
                   "generation_config.json": "{}", "tokenizer.json": "{}",
                   "tokenizer_config.json": json.dumps({"chat_template": template}),
                   "chat_template.jinja": template, "model.safetensors": "opaque fixture only"}
        for name, text in content.items():
            (self.model / name).write_text(text)
        files = {name: {"bytes": (self.model / name).stat().st_size, "sha256": export.sha256(self.model / name)}
                 for name in content}
        self.data_manifest = {"files": {}}
        for name in ("train_main.parquet", "eval_main.parquet", "eval_main.jsonl"):
            (self.data / name).write_text(name)
            self.data_manifest["files"][name] = {"sha256": export.sha256(self.data / name),
                                                "case_ids": ["train" if name.startswith("train") else "eval"]}
        (self.data / "data_manifest.json").write_text(json.dumps(self.data_manifest))
        self.manifest = {"schema_version": 1, "artifact_type": "trained_full_parameter_export",
                         "profile": "llama3b", "base_revision": export.BASE_REVISION,
                         "base_repo_id": "meta-llama/Llama-3.2-3B-Instruct",
                         "prepared_base_manifest_sha256": "e" * 64,
                         "source_checkpoint": {"path": "runs/train-01/checkpoints/global_step_20",
                                               "manifest_sha256": "f" * 64,
                                               "model_sha256": hashlib.sha256(b"fixture native").hexdigest()},
                         "training_execution_commit": "a" * 40, "integration_execution_commit": "b" * 40,
                         "converter_commit": "c" * 40, "converter_source_sha256": "d" * 64,
                         "files": files, "harness_sha256": self.harness,
                         "template_sha256": files["chat_template.jinja"]["sha256"],
                         "prepared_tokenizer_config_sha256": files["tokenizer_config.json"]["sha256"],
                         "frozen_data_sha256": {n: r["sha256"] for n, r in self.data_manifest["files"].items()},
                         "tensor_validation": {"all_tensors_equal": True, "dtype_preserved": True,
                                               "tensor_count": 1, "validated_alias_map": {},
                                               "source_model_sha256": hashlib.sha256(b"fixture native").hexdigest(),
                                               "export_files_sha256": hashlib.sha256(json.dumps(files, sort_keys=True,
                                                                                    separators=(",", ":")).encode()).hexdigest()}}
        self.save()

    def save(self):
        (self.model / "export_manifest.json").write_text(json.dumps(self.manifest))

    def args(self):
        return SimpleNamespace(repo=self.root, model_path=self.model, data_dir=self.data,
                               python=Path(sys.executable), profile="llama3b", phase="export-eval",
                               checkpoint=None, hooks=None)

    def test_export_validates_identity_and_uses_validation_only_without_resume(self):
        self.assertEqual(export.verify_export(self.model, self.data_manifest, self.harness), self.manifest)
        args = self.args()
        with patch.object(launch.subprocess, "check_output", side_effect=[launch.AGL_COMMIT, ""]):
            validated = launch.validate_inputs(args, self.bundle)
        self.assertEqual(validated[-3]["artifact_type"], "trained_full_parameter_export")
        plan, commands = launch.make_plan(args, self.bundle, self.root, self.model, self.data,
                                          Path(sys.executable), self.root / "export-evaluation", None, 12345)
        config = dict((key, json.loads(value)) for key, value in (v.split("=", 1) for v in plan["overrides"]))
        self.assertIs(config["trainer.val_only"], True)
        self.assertEqual(config["trainer.resume_mode"], "disable")
        self.assertNotIn("trainer.resume_from_path", config)
        self.assertIn("12345", plan["gateway"])
        self.assertEqual(config["actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser"], "llama3_json")
        self.assertEqual(config["actor_rollout_ref.rollout.max_model_len"], 4096)

    def test_export_refuses_base_masquerade_changed_shard_and_missing_receipt(self):
        (self.model / "model_manifest.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "masquerade"):
            export.verify_export(self.model, self.data_manifest, self.harness)
        (self.model / "model_manifest.json").unlink()
        self.manifest["tensor_validation"]["dtype_preserved"] = False
        self.save()
        with self.assertRaisesRegex(ValueError, "all-tensor"):
            export.verify_export(self.model, self.data_manifest, self.harness)
        self.manifest["tensor_validation"]["dtype_preserved"] = True
        self.save()
        (self.model / "model.safetensors").write_text("damaged")
        with self.assertRaisesRegex(ValueError, "byte identity"):
            export.verify_export(self.model, self.data_manifest, self.harness)

    def test_export_refuses_data_harness_or_template_drift_and_unlisted_shards(self):
        with self.assertRaisesRegex(ValueError, "harness"):
            export.verify_export(self.model, self.data_manifest, "wrong")
        self.manifest["frozen_data_sha256"]["eval_main.parquet"] = "bad"
        self.save()
        with self.assertRaisesRegex(ValueError, "frozen-data"):
            export.verify_export(self.model, self.data_manifest, self.harness)
        self.manifest["frozen_data_sha256"]["eval_main.parquet"] = self.data_manifest["files"]["eval_main.parquet"]["sha256"]
        self.manifest["template_sha256"] = "bad"
        self.save()
        with self.assertRaisesRegex(ValueError, "semantics"):
            export.verify_export(self.model, self.data_manifest, self.harness)
        self.manifest["template_sha256"] = self.manifest["files"]["chat_template.jinja"]["sha256"]
        self.save()
        (self.model / "extra.safetensors").write_text("extra")
        with self.assertRaisesRegex(ValueError, "Unlisted"):
            export.verify_export(self.model, self.data_manifest, self.harness)

    def test_export_inventory_cannot_read_credential_or_raw_trace_members(self):
        for name in (".env", "trajectory-private.jsonl", "../model.safetensors"):
            with self.subTest(name=name):
                self.manifest["files"][name] = {"bytes": 1, "sha256": "a" * 64}
                self.save()
                with self.assertRaises(ValueError):
                    export.verify_export(self.model, self.data_manifest, self.harness)
                del self.manifest["files"][name]

    def test_closed_inventory_and_standard_shard_discovery(self):
        for name in ("added_tokens.json", "model.safetensors.index.json", "nested"):
            extra = self.model/name
            extra.mkdir() if name == "nested" else extra.write_text("{}")
            with self.assertRaisesRegex(ValueError, "Unlisted"):
                export.verify_export(self.model,self.data_manifest,self.harness)
            extra.rmdir() if name == "nested" else extra.unlink()
        (self.model/"model.safetensors").rename(self.model/"unusual.safetensors")
        self.manifest["files"]["unusual.safetensors"] = self.manifest["files"].pop("model.safetensors")
        self.save()
        with self.assertRaisesRegex(ValueError, "index missing"):
            export.verify_export(self.model,self.data_manifest,self.harness)
        index = self.model/"model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map":{"x":"unusual.safetensors"}}))
        self.manifest["files"][index.name] = {"bytes":index.stat().st_size,"sha256":export.sha256(index)}
        self.manifest["tensor_validation"]["export_files_sha256"] = hashlib.sha256(json.dumps(
            self.manifest["files"],sort_keys=True,separators=(",",":")).encode()).hexdigest()
        self.save()
        export.verify_export(self.model,self.data_manifest,self.harness)

    def test_changed_actual_jsonl_rejected_before_evaluation(self):
        (self.data/"eval_main.jsonl").write_text("changed after manifest")
        with patch.object(launch.subprocess,"check_output",side_effect=[launch.AGL_COMMIT, ""]):
            with self.assertRaisesRegex(ValueError, "Dataset changed.*eval_main.jsonl"):
                launch.validate_inputs(self.args(), self.bundle)

    def test_dry_run_records_export_policy_and_never_labels_base(self):
        argv = ["launch.py", "--repo", str(self.root), "--model-path", str(self.model),
                "--data-dir", str(self.data), "--output-root", str(self.root / "runs"),
                "--name", "export-check", "--phase", "export-eval", "--dry-run"]
        validated = (self.root, self.model, self.data, Path(sys.executable), self.manifest, self.data_manifest, None)
        with patch.object(sys, "argv", argv), patch.object(launch, "validate_inputs", return_value=validated), \
                patch.object(launch, "run_services") as services, redirect_stdout(StringIO()):
            launch.main()
        services.assert_not_called()
        run = self.root / "runs/export-check"
        plan = json.loads((run / "run_plan.json").read_text())
        self.assertEqual(plan["policy_version"], "export:sha256:" + export.sha256(self.model / "export_manifest.json"))
        self.assertTrue((run / "export_manifest.json").is_file())
        self.assertFalse((run / "model_manifest.json").exists())


class DetachedRunnerTests(unittest.TestCase):
    def test_export_request_and_summary_filters_are_explicit(self):
        req = runner.request("export-eval", "check", export_path="exports/original", wandb_entity="team")
        command = runner.launch_command(req)
        self.assertEqual(command[command.index("--model-path") + 1], "/data/exports/original")
        self.assertNotIn("--checkpoint", command)
        for path in ("../escape", "/data/export", "exports/a/b"):
            with self.assertRaises(ValueError):
                runner.request("export-eval", "check", export_path=path, wandb_entity="team")
        req = runner.request("summarize", "compare", baseline_name="native", post_name="exported",
                             baseline_phase="reload-eval", post_phase="export-eval")
        with patch.object(runner, "run_command") as command, patch.object(runner, "environment", return_value={}):
            runner.collect(req, Path("/fixture"))
        argv = command.call_args.args[0]
        self.assertEqual(argv[argv.index("--phase") + 1], "export-eval")
        self.assertEqual(argv[argv.index("--baseline-phase") + 1], "reload-eval")

    def test_detach_required_and_original_volume_export_refused_before_spawn(self):
        with patch.object(runner.gpu_phase, "spawn") as spawn:
            with patch.object(sys, "argv", ["modal", "run"]), self.assertRaisesRegex(ValueError, "detach"):
                runner.main(action="main", name="fixture", wandb_entity="team")
            with patch.object(sys, "argv", ["modal", "run", "--detach"]), \
                    patch.object(runner, "WORKING_VOLUME", ""), self.assertRaisesRegex(ValueError, "original Volume"):
                runner.main(action="export-eval", name="fixture", export_path="exports/a", wandb_entity="team")
            spawn.assert_not_called()

    def test_spawn_ids_saved_without_waiting_for_result(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(runner, "LOCAL", Path(temporary)), \
                patch.object(sys, "argv", ["modal", "run", "--detach"]), \
                patch.object(runner.subprocess, "check_output", side_effect=["a" * 40, ""]), \
                patch.object(runner.gpu_phase, "spawn", return_value=Mock(object_id="fc-fixture")) as spawn, \
                patch.object(runner.gpu_phase, "remote") as remote, redirect_stdout(StringIO()) as output:
            runner.main(action="main", name="new-fixture", wandb_entity="team")
            receipts = list((Path(temporary) / ".local/experiment-submissions").glob("*.json"))
            self.assertEqual(len(receipts), 1)
            receipt = json.loads(receipts[0].read_text())
            self.assertEqual(receipt["function_call_id"], "fc-fixture")
            self.assertEqual(receipt["status"], "submitted_not_validated")
            self.assertEqual(receipt["receipt_path"], "submissions/new-fixture/receipt.json")
            self.assertIn("SUBMITTED", output.getvalue())
            spawn.assert_called_once()
            remote.assert_not_called()

    def test_worker_refuses_mismatched_mount_before_source_access(self):
        req = runner.request("main", "fixture", wandb_entity="team")
        req["volume"] = "different-work-volume"
        with patch.object(runner.volume, "reload") as reload, self.assertRaisesRegex(ValueError, "binding"):
            runner.recorded(req, Mock())
        reload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
