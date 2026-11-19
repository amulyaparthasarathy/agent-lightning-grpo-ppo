"""Tiny opaque fixtures only: no cloud, credentials, weights or training imports."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import preservation_manifest as pm

STAMP = "2026-11-13T00:00:00+00:00"


class PreservationTests(unittest.TestCase):
    def test_explicit_evidence_link_exclusion_never_reads_target(self):
        path='runs/failed-02/wandb/latest-run'
        link=self.root/path;link.parent.mkdir(parents=True)
        link.symlink_to('/nonexistent-outside-fixture')
        with self.assertRaises(ValueError):
            pm.scan(self.root,self.selection,'synthetic_observation','fixture',STAMP)
        self.selection['excluded_symlinks']=[path]
        report=pm.scan(self.root,self.selection,'synthetic_observation','fixture',STAMP)
        self.assertIn({'path':path,'reason':'reviewed evidence symlink excluded; target not read'},report['exclusions'])
        self.assertNotIn(path,{r['path'] for r in report['files']})
        link.unlink();link.write_text('real evidence must not be skipped')
        with self.assertRaisesRegex(ValueError,'not an evidence symlink'):
            pm.scan(self.root,self.selection,'synthetic_observation','fixture',STAMP)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.parent = Path(self.temporary.name)
        self.root = self.parent / "source"
        self.root.mkdir()
        self.selection = {
            "provenance": {"lab_execution_commit": "fixture-execution",
                           "documentation_commit": "fixture-closeout",
                           "agent_lightning_commit": "fixture-upstream",
                           "preservation_implementation_commit": "fixture-implementation"},
            "selections": [{"group": "evidence", "path": p} for p in
                           ("runs", "evidence", "submissions", "data-small", "repository")]
                          + [{"group": "native_state", "path": pm.CHECKPOINT},
                             {"group": "native_state", "path": pm.MARKER},
                             {"group": "prepared_base", "path": "models/llama3b"}],
        }
        self.files = {}
        for run in ("smoke-01", "smoke-reload-01", "base-01", "train-01", "post-01", "failed-02"):
            for name in ("result.json", "logs/trainer.log", "traces/trajectory-example.jsonl",
                         "resolved_config.json", "observability.json", "code/launch.py"):
                self.files[f"runs/{run}/{name}"] = b"fixture evidence\n"
            self.files[f"submissions/{run}/receipt.json"] = b'{"status":"fixture"}'
        self.files.update({"evidence/paired-01/summary.json": b"{}",
                           "submissions/paired-01/source/experiment/launch.py": b"# fixture",
                           "submissions/train-01/image-build/packages.txt": b"fixture==1",
                           "data-small/data_manifest.json": b"{}",
                           "data-small/eval_main.jsonl": b'{"case_id":"fixture-only"}',
                           "repository/execution-closeout.bundle": b"opaque fixture snapshot",
                           pm.MARKER: b"20\n"})
        for name in pm.NATIVE_FILES:
            self.files[f"{pm.CHECKPOINT}/{name}"] = b"opaque fixture\x00"
        self.files[f"{pm.CHECKPOINT}/actor/fsdp_config.json"] = b'{"fsdp_version": 1, "world_size": 1}'
        self.files[f"{pm.CHECKPOINT}/actor/huggingface/config.json"] = b'{"architectures":["LlamaForCausalLM"]}'
        for name in ("model-00001-of-00001.safetensors", "model.safetensors.index.json",
                     "model_manifest.json", "tokenizer.json", "tokenizer_config.json",
                     "config.json", "generation_config.json", "chat_template.jinja", "LICENSE", "Notice"):
            self.files[f"models/llama3b/{name}"] = b"prepared fixture"
        for path, content in self.files.items():
            self.put(path, content)

    def put(self, path, content):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    def scan(self, root=None, origin="fixture-source"):
        return pm.scan(root or self.root, self.selection, "synthetic_observation", origin, STAMP)

    def test_inclusion_totals_order_provenance_and_source_unchanged(self):
        before = {p: ((self.root / p).read_bytes(), pm.signature((self.root / p).stat())) for p in self.files}
        self.put("runs/train-01/checkpoints/global_step_15/actor/model.pt", b"optional")
        self.put("runs/train-01/checkpoints/global_step_5/data.pt", b"incomplete")
        result = self.scan()
        self.assertEqual([r["path"] for r in result["files"]], sorted(self.files))
        self.assertTrue(result["required_file_findings"]["structurally_complete"])
        self.assertEqual(result["provenance"], self.selection["provenance"])
        self.assertIn("runs/failed-02", result["directories"])
        self.assertEqual(result, self.scan())
        self.assertTrue(any(e["path"] == "runs/train-01/checkpoints" for e in result["exclusions"]))
        for group in pm.GROUPS:
            entries = [r for r in result["files"] if r["group"] == group]
            self.assertEqual(result["totals_by_group"][group]["known_bytes"],
                             sum(len(self.files[r["path"]]) for r in entries))
        for row in result["files"]:
            self.assertEqual(row["sha256"], hashlib.sha256(self.files[row["path"]]).hexdigest())
        for path, expected in before.items():
            self.assertEqual(((self.root / path).read_bytes(), pm.signature((self.root / path).stat())), expected)

    def test_each_required_file_missing_or_empty(self):
        for path in [f"{pm.CHECKPOINT}/{p}" for p in pm.NATIVE_FILES] + [pm.MARKER]:
            with self.subTest(path=path):
                target = self.root / path
                content = target.read_bytes()
                target.unlink()
                finding = self.scan()["required_file_findings"]
                self.assertIn(path, finding["missing"])
                self.assertFalse(finding["structurally_complete"])
                target.write_bytes(b"")
                self.assertIn(path, self.scan()["required_file_findings"]["empty"])
                target.write_bytes(content)

    def test_data_only_checkpoint_and_invalid_metadata_never_complete(self):
        for path, content in ((pm.MARKER, b"15"),
                              (f"{pm.CHECKPOINT}/actor/fsdp_config.json", b'{"fsdp_version":2,"world_size":1}'),
                              (f"{pm.CHECKPOINT}/actor/fsdp_config.json", b'{"fsdp_version":1,"world_size":2}'),
                              (f"{pm.CHECKPOINT}/actor/fsdp_config.json", b"invalid json"),
                              (f"{pm.CHECKPOINT}/actor/huggingface/config.json", b'{"architectures":"LlamaForCausalLM"}'),
                              (f"{pm.CHECKPOINT}/actor/huggingface/config.json", b'{"architectures":[]}')):
            with self.subTest(path=path, content=content):
                self.put(path, content)
                self.assertFalse(self.scan()["required_file_findings"]["structurally_complete"])
                self.put(path, self.files[path])
        shutil.rmtree(self.root / pm.CHECKPOINT / "actor")
        self.assertFalse(self.scan()["required_file_findings"]["structurally_complete"])

    def test_unsafe_paths_duplicate_and_overlap_rejected(self):
        for path in ("/absolute", "../escape", "runs/../escape", "runs//x", "runs/./x", "C:\\x", "x/", ""):
            with self.subTest(path=path), self.assertRaises(ValueError):
                pm.member(path)
        for item in ({"group": "evidence", "path": "runs"},
                     {"group": "prepared_base", "path": "runs/train-01"}):
            self.selection["selections"].append(item)
            with self.assertRaises(ValueError):
                self.scan()
            self.selection["selections"].pop()
        result = self.scan()
        result["files"].append(result["files"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            pm.supplied(result, STAMP)

    def test_symlink_ancestor_escape_hardlink_and_fifo_rejected(self):
        outside = self.parent / "outside"
        outside.write_bytes(b"outside fixture")
        target = self.root / "runs/link"
        target.symlink_to(outside)
        with self.assertRaises(ValueError):
            self.scan()
        target.unlink()
        target.symlink_to(self.parent, target_is_directory=True)
        self.selection["selections"] = [{"group": "evidence", "path": "runs/link/outside"}]
        with self.assertRaises(OSError):
            self.scan()
        target.unlink()
        self.selection["selections"] = [{"group": "evidence", "path": "runs"}]
        os.link(outside, target)
        with self.assertRaises(ValueError):
            self.scan()
        target.unlink()
        os.mkfifo(target)
        with self.assertRaises(ValueError):
            self.scan()

    def test_exclusions_before_read_and_descent(self):
        excluded = ("runs/.env", "runs/.env.local", "runs/.aws/credentials",
                    "models/llama3b/.cache/huggingface/token", "runs/token.json",
                    "runs/credentials.json", "runs/.ssh/id_rsa", "runs/modal.toml")
        for path in excluded:
            self.put(path, b"synthetic excluded content")
        real_read = pm.read_file
        def guarded(parent, name, *args):
            self.assertNotIn(name, {Path(p).name for p in excluded})
            return real_read(parent, name, *args)
        with patch.object(pm, "read_file", side_effect=guarded):
            result = self.scan()
        self.assertEqual({r["path"] for r in result["files"]}, set(self.files))
        self.assertTrue(result["exclusions"])
        with self.assertRaisesRegex(ValueError, "excluded"):
            pm.read_json(self.root / "runs/.env")

    def test_existing_output_and_output_in_source_refused(self):
        output = self.parent / "manifest.json"
        pm.write_output(output, self.scan(), [self.root])
        before = output.read_bytes()
        with self.assertRaises(FileExistsError):
            pm.write_output(output, {}, [self.root])
        self.assertEqual(output.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "outside source"):
            pm.write_output(self.root / "new.json", {}, [self.root])
        link = self.parent / "output-link"
        link.symlink_to(self.parent / "absent")
        with self.assertRaises(FileExistsError):
            pm.write_output(link, {})

    def test_mutation_while_hashing_is_detected(self):
        target = self.root / pm.CHECKPOINT / "data.pt"
        real_sha = hashlib.sha256
        class MutatingHash:
            def __init__(self):
                self.digest = real_sha()
                self.mutated = False
            def update(self, chunk):
                if not self.mutated:
                    with target.open("ab") as handle:
                        handle.write(b"mutation")
                    self.mutated = True
                self.digest.update(chunk)
            def hexdigest(self):
                return self.digest.hexdigest()
        with pm.directory(target.parent) as parent, patch.object(pm.hashlib, "sha256", MutatingHash):
            with self.assertRaisesRegex(ValueError, "mutation"):
                pm.read_file(parent, target.name)

    def test_independent_copy_and_missing_changed_extra_files(self):
        source = self.scan()
        destination = self.parent / "copy"
        shutil.copytree(self.root, destination)
        copied = self.scan(destination, "fixture-destination")
        result = pm.compare(source, copied, STAMP)
        self.assertTrue(result["content_match"])
        self.assertFalse(result["backup_verified"])
        self.assertEqual(result["backup_status"], "not_executed")
        (destination / "evidence/paired-01/summary.json").unlink()
        (destination / pm.CHECKPOINT / "data.pt").write_bytes(b"changed")
        (destination / "data-small/eval_main.jsonl").write_bytes(b"x" * len(self.files["data-small/eval_main.jsonl"]))
        (destination / "evidence/extra.txt").write_bytes(b"extra")
        result = pm.compare(source, self.scan(destination, "fixture-destination"), STAMP)
        self.assertFalse(result["content_match"])
        self.assertEqual(result["missing"], ["evidence/paired-01/summary.json"])
        self.assertEqual(result["extra"], ["evidence/extra.txt"])
        self.assertEqual(len(result["size_differences"]), 1)
        self.assertEqual(len(result["hash_differences"]), 2)

    def test_historical_unknown_absent_and_self_comparison_never_verify(self):
        source = self.scan()
        for destination in (None, source):
            result = pm.compare(source, destination, STAMP)
            self.assertFalse(result["content_match"])
            self.assertFalse(result["backup_verified"])
        historical = copy.deepcopy(source)
        for row in historical["files"]:
            row.update(observation_class="historical_expectation", observation_origin="historical report",
                       bytes=None, sha256=None)
        historical = pm.supplied(historical, STAMP)
        self.assertFalse(historical["required_file_findings"]["structurally_complete"])
        self.assertGreater(historical["totals_by_group"]["native_state"]["unknown_sizes"], 0)
        result = pm.compare(source, historical, STAMP)
        self.assertTrue(result["unknown"])
        self.assertFalse(result["content_match"])
        self.assertFalse(result["backup_verified"])
        reported = pm.supplied(source, STAMP)
        self.assertEqual(reported["exclusions"], source["exclusions"])
        self.assertEqual(reported["files"][0]["sha256"], source["files"][0]["sha256"])
        self.assertEqual(reported["files"][0]["hash_basis"], "reported")
        self.assertFalse(pm.compare(source, reported, STAMP)["content_match"])

    def test_cli_scan_inventory_compare_exit_codes_without_site_packages(self):
        selection = self.parent / "selection.json"
        selection.write_text(json.dumps(self.selection))
        script = Path(pm.__file__).resolve()
        def run(*args):
            return subprocess.run([sys.executable, "-S", str(script), *map(str, args)],
                                  capture_output=True, text=True, timeout=10)
        output = self.parent / "source.json"
        result = run("scan", "--root", self.root, "--selection", selection,
                     "--origin", "fixture-source", "--observation-class", "synthetic_observation",
                     "--timestamp", STAMP, "--output", output)
        self.assertEqual(result.returncode, 0, result.stderr)
        comparison = self.parent / "comparison.json"
        result = run("compare", "--source", output, "--output", comparison)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse(json.loads(comparison.read_text())["backup_verified"])
        result = run("inventory", "--input", output, "--output", self.parent / "reported.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        result = run("inventory", "--input", output, "--output", output)
        self.assertEqual(result.returncode, 2, result.stderr)


if __name__ == "__main__":
    unittest.main()
