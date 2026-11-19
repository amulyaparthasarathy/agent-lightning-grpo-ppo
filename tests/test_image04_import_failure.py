"""Offline regression tests for the image-04 datasets import-error evidence.

No network access, model loading, or cloud resources are used.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "validation/runtime.py"
spec = importlib.util.spec_from_file_location("runtime_image04_import", FILE)
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


class ImportFailureEvidenceTests(unittest.TestCase):
    def test_reports_attempted_datasets_import_and_missing_symbol(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            error = ImportError("cannot import name 'missing_symbol' from 'huggingface_hub'",
                                name="huggingface_hub")
            record = runtime.record_import_failure(Path(temp), "datasets", error)
            persisted = json.loads((Path(temp) / "import-failure.json").read_text())
            self.assertEqual(persisted, record)
            self.assertEqual(persisted["attempted_module"], "datasets")
            self.assertEqual(persisted["exception_type"], "ImportError")
            self.assertEqual(persisted["missing_name"], "huggingface_hub")
            self.assertIn("missing_symbol", persisted["message"])
            self.assertNotIn("environment", persisted)

    def test_redacts_secrets_in_message_and_metadata(self):
        token = "synthetic-test-HF-secret-value-not-a-real-key"
        api_key = "synthetic-test-WB-secret-value-not-a-real-key"
        error = ImportError("broken " + token + " " + api_key, name=token, path="/tmp/" + api_key)
        with tempfile.TemporaryDirectory() as temp, patch.dict(
                os.environ, {"HF_TOKEN": token, "WANDB_API_KEY": api_key}), contextlib.redirect_stdout(io.StringIO()):
            record = runtime.record_import_failure(Path(temp), "datasets", error)
            persisted = (Path(temp) / "import-failure.json").read_text()
        self.assertNotIn(token, persisted)
        self.assertNotIn(api_key, persisted)
        self.assertIn("[REDACTED]", record["message"])
        self.assertIn("[REDACTED]", record["missing_name"])
        self.assertIn("[REDACTED]", record["exception_path"])

    def test_limits_large_exception_message(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            record = runtime.record_import_failure(Path(temp), "datasets", ImportError("X" * 20000))
        self.assertEqual(len(record["message"]), 800)

    def test_audit_still_propagates_import_error(self):
        source = FILE.read_text()
        self.assertIn('save(output / "dependency-progress.json", {"importing_module": module})', source)
        self.assertIn("except ImportError as exc:", source)
        self.assertIn("record_import_failure(output, module, exc)", source)
        self.assertIn("raise\n        if module", source)
        self.assertIn('save(output / "imports.json", imports)', source)


if __name__ == "__main__":
    unittest.main()
