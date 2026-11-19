"""Offline, dependency-free tests for the new PyArrow 25 training overlay.

These use stand-in modules. Only image-06 on Modal can verify the real PyArrow
binary ABI, Hugging Face Datasets implementation and Parquet round-trip.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
INSTALL = ROOT / "scripts/finalize_modal_training.sh"
HEAVY = ROOT / "scripts/install_modal_training.sh"

def load(name, location):
    spec = importlib.util.spec_from_file_location(name, location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

runtime = load("arrow_runtime_test", ROOT / "validation/runtime.py")
policy = load("arrow_policy_test", ROOT / "validation/compatibility.py")


class Array:
    def __init__(self, values):
        self.values = list(values)

    def tolist(self):
        return list(self.values)

    def __len__(self):
        return len(self.values)


def fake_modules(*, corrupt_arrow=False, corrupt_dataset=False, error=None):
    np = types.ModuleType("numpy")
    np.__version__ = "1.26.4"
    np.int64 = "int64"
    np.array = lambda seq, dtype=None: Array(seq)
    np.array_equal = lambda one, two: one.tolist() == two.tolist()

    class FakeDataset:
        def __init__(self, data):
            self.data = data

        @classmethod
        def from_dict(cls, data):
            return cls(dict(data))

        def to_parquet(self, path):
            if error is not None:
                raise error
            Path(path).write_text(json.dumps(self.data))
            return Path(path).stat().st_size

        @classmethod
        def from_parquet(cls, path, **kwargs):
            data = json.loads(Path(path).read_text())
            if corrupt_dataset:
                data["label"][0] = "wrong"
            return cls(data)

        def __len__(self):
            return len(self.data["value"])

        def __getitem__(self, key):
            return self.data[key]

    ds = types.ModuleType("datasets")
    ds.Dataset = FakeDataset
    pa = types.ModuleType("pyarrow")
    pa.__version__ = "25.0.1"
    pa.__path__ = []

    pq = types.ModuleType("pyarrow.parquet")
    def read_table(path):
        data = json.loads(Path(path).read_text())

        class Column:
            def to_numpy(self):
                return Array([999, 999, 999] if corrupt_arrow else data["value"])

        class Table:
            def column(self, name):
                assert name == "value"
                return Column()

        return Table()

    pq.read_table = read_table
    pa.parquet = pq
    return {"numpy": np, "datasets": ds, "pyarrow": pa, "pyarrow.parquet": pq}


class ArrowDataInteropTests(unittest.TestCase):
    def test_exact_pyarrow_pin_is_after_upstream_install_but_before_freeze(self):
        shell = INSTALL.read_text()
        self.assertLess(shell.index('"$AGL_BUILD/requirements-agent.txt"'),
                        shell.index("pyarrow==25.0.1"))
        self.assertLess(shell.index("pyarrow==25.0.1"),
                        shell.index('installed-packages.txt'))
        self.assertIn("--no-deps --no-build", shell)
        self.assertNotIn("pyarrow==25.0.1", HEAVY.read_text())
        subprocess.run(["bash", "-n", str(INSTALL)], check=True)

    def test_scoped_inventory_requires_exact_arrow_and_datasets(self):
        self.assertEqual(policy.EXPECTED_VERSIONS["pyarrow"], "25.0.1")
        self.assertEqual(policy.EXPECTED_VERSIONS["datasets"], "5.2.0")
        self.assertEqual(policy.EXPECTED_VERSIONS["numpy"], "1.26.4")

    def test_synthetic_roundtrip_and_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.dict(sys.modules, fake_modules()), patch(
                    "importlib.metadata.version", return_value="5.2.0"):
                actual = runtime.audit_arrow_dataset_roundtrip(Path(temp))
            self.assertTrue(actual["verified"])
            self.assertEqual(actual["rows"], 3)
            self.assertEqual(actual["pyarrow"], "25.0.1")
            self.assertEqual(actual["numpy"], "1.26.4")
            self.assertEqual(actual["datasets"], "5.2.0")
            persisted = json.loads((Path(temp) / "data-smoke.json").read_text())
            self.assertEqual(persisted, actual)
            self.assertFalse((Path(temp) / "data-smoke-failure.json").exists())

    def test_rejects_corrupted_pyarrow_values(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.dict(sys.modules, fake_modules(corrupt_arrow=True)):
                with self.assertRaisesRegex(RuntimeError, "Parquet round-trip"):
                    runtime.audit_arrow_dataset_roundtrip(Path(temp))
            fail = json.loads((Path(temp) / "data-smoke-failure.json").read_text())
            self.assertEqual(fail["exception_type"], "RuntimeError")
            self.assertFalse((Path(temp) / "data-smoke.json").exists())

    def test_rejects_corrupted_datasets_values(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.dict(sys.modules, fake_modules(corrupt_dataset=True)):
                with self.assertRaisesRegex(RuntimeError, "Datasets Parquet"):
                    runtime.audit_arrow_dataset_roundtrip(Path(temp))
            fail = json.loads((Path(temp) / "data-smoke-failure.json").read_text())
            self.assertIn("Datasets Parquet", fail["message"])

    def test_does_not_serialize_secrets_on_roundtrip_error(self):
        secret = "synthetic-credential-not-real"
        with tempfile.TemporaryDirectory() as temp:
            with patch.dict(sys.modules, fake_modules(error=RuntimeError(secret))):
                with patch.dict("os.environ", {"HF_TOKEN": secret}):
                    with self.assertRaises(RuntimeError):
                        runtime.audit_arrow_dataset_roundtrip(Path(temp))
            contents = (Path(temp) / "data-smoke-failure.json").read_text()
            self.assertNotIn(secret, contents)
            self.assertIn("[REDACTED]", contents)

    def test_cpu_audit_requires_data_interop_before_reporting_success(self):
        code = (ROOT / "validation/runtime.py").read_text()
        self.assertLess(code.index("data_interop = audit_arrow_dataset_roundtrip"),
                        code.index("cpu_imports_checked=True"))
        self.assertIn('save(output / "data-smoke.json", evidence)', code)


if __name__ == "__main__":
    unittest.main()
