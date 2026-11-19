"""Offline regressions for image-03: uv banner, scoped metadata, and exit code."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

compat = load("image03_compatibility", ROOT / "validation/compatibility.py")
runtime = load("image03_runtime", ROOT / "validation/runtime.py")

REPORT = """Using Python 3.12.1 environment at: /opt/agent-lightning/.venv
Checked 285 packages in 142ms
Found 6 incompatibilities
The package §ml-dtypes§ requires §numpy>=2.0.0§, but §1.26.4§ is installed
The package §opencv-python-headless§ requires §numpy>=2 ; python_full_version >= '3.9'§, but §1.26.4§ is installed
The package §google-api-core§ requires §protobuf>=6.33.5,<8.0.0§, but §5.29.6§ is installed
The package §tensorboard§ requires §protobuf>=6.31.1,<8.0.0§, but §5.29.6§ is installed
The package §xgrammar§ requires §transformers>=4.38.0,<5§, but §5.22.2§ is installed
The package §proto-plus§ requires §protobuf>=6.33.5,<8.0.0§, but §5.29.6§ is installed
""".replace("§", chr(96))

def classify(report=REPORT, versions=None):
    return compat.classify_conflicts(
        1, report, compat.EXPECTED_VERSIONS if versions is None else versions,
        python_minor="3.12", machine="x86_64", source_commit=compat.UPSTREAM_COMMIT)


class Image03Diagnostics(unittest.TestCase):
    def test_exact_uv_report_with_header_is_qualified_diagnostic(self):
        result = classify()
        self.assertTrue(result["report_parsed"])
        self.assertTrue(result["identity_matches"])
        self.assertTrue(result["reviewed_exception_qualified"])
        self.assertEqual(result["reviewed_dependency_conflicts"], 6)
        self.assertFalse(result["dependency_closure_consistent"])

    def test_headerless_report_remains_supported(self):
        self.assertTrue(classify(REPORT.split("\n", 1)[1])["accepted"])

    def test_unknown_or_changed_banner_is_rejected(self):
        for report in (
            REPORT.replace("Python 3.12.1", "Python 3.13.0"),
            REPORT.replace("/opt/agent-lightning/.venv", "/opt/venv"),
            "WARNING: unexpected\n" + REPORT,
            REPORT.replace("Checked 285", "Something else\nChecked 285"),
            REPORT.replace("Checked 285", REPORT.splitlines()[0] + "\nChecked 285"),
        ):
            with self.subTest(report=report.splitlines()[0]):
                self.assertFalse(classify(report)["accepted"])

    def test_protobuf_drift_still_fails(self):
        changed = dict(compat.EXPECTED_VERSIONS, protobuf="6.31.1")
        result = classify(versions=changed)
        self.assertFalse(result["identity_matches"])
        self.assertIn("protobuf", result["environment_identity"]["version_drift"])

    def test_venv_scoping_excludes_unrelated_ambient_protobuf(self):
        def dist(version, path):
            return types.SimpleNamespace(metadata={"Name": "protobuf"},
                                         version=version, _path=Path(path))
        venv = [dist("5.29.6", "/opt/agent-lightning/.venv/lib/python3.12/site-packages/protobuf-5.29.6.dist-info")]
        ambient = [dist("6.31.1", "/usr/local/lib/python3.12/site-packages/protobuf-6.31.1.dist-info")]

        def discover(*, path=None):
            if path is not None:
                self.assertEqual(path, ["/opt/agent-lightning/.venv/lib/python3.12/site-packages"])
                return venv
            return ambient

        with tempfile.TemporaryDirectory() as name:
            with patch.object(sys, "prefix", "/opt/agent-lightning/.venv"):
                with patch.object(runtime.metadata, "distributions", side_effect=discover):
                    packages = runtime.inventory_training_environment(Path(name))
            self.assertEqual(packages["protobuf"], "5.29.6")
            info = json.loads((Path(name) / "package-inventory-details.json").read_text())
            self.assertEqual(info["protobuf_in_selected_venv"][0]["version"], "5.29.6")
            self.assertEqual(info["protobuf_on_interpreter_search_path"][0]["version"], "6.31.1")

    def test_ambiguous_venv_protobuf_fails_closed(self):
        def dist(version):
            return types.SimpleNamespace(metadata={"Name": "protobuf"}, version=version)
        with tempfile.TemporaryDirectory() as name:
            with patch.object(sys, "prefix", "/opt/agent-lightning/.venv"):
                with patch.object(runtime.metadata, "distributions",
                                  return_value=[dist("5.29.6"), dist("6.31.1")]):
                    with self.assertRaisesRegex(RuntimeError, "Ambiguous package metadata"):
                        runtime.inventory_training_environment(Path(name))
            info = json.loads((Path(name) / "package-inventory-details.json").read_text())
            self.assertIn("protobuf", info["ambiguous_venv_distributions"])

    def test_diagnostic_cli_exit_requires_correlated_evidence(self):
        report = {
            "status": "diagnostic_passed_with_reviewed_exceptions",
            "dependency_closure_consistent": False,
            "reviewed_dependency_conflicts": 6,
            "qualification": "diagnostic_with_reviewed_dependency_exceptions",
            "checks": {
                "training": {"cpu_imports_checked": True},
                "wandb": {"summary_verified": True},
                "hugging_face": {"authenticated": True},
            },
        }
        self.assertEqual(runtime.validation_exit_code(report), 0)
        for key in ("training", "wandb", "hugging_face"):
            altered = dict(report, checks={k: v for k, v in report["checks"].items() if k != key})
            self.assertEqual(runtime.validation_exit_code(altered), 1)
        self.assertEqual(runtime.validation_exit_code({"status": "failed"}), 1)
        self.assertEqual(runtime.validation_exit_code({"status": "passed"}), 0)


if __name__ == "__main__":
    unittest.main()
