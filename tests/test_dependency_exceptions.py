"""Offline tests for exact, version-qualified dependency exception policy."""
from pathlib import Path
import importlib.util
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / "validation/compatibility.py"
spec = importlib.util.spec_from_file_location("dependency_compatibility", MODULE_PATH)
compat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compat)

SIX = """Checked 285 packages in 12ms
Found 6 incompatibilities
The package `ml-dtypes` requires `numpy>=2.0.0`, but `1.26.4` is installed
The package `opencv-python-headless` requires `numpy>=2 ; python_full_version >= '3.9'`, but `1.26.4` is installed
The package `google-api-core` requires `protobuf>=6.33.5,<8.0.0`, but `5.29.6` is installed
The package `tensorboard` requires `protobuf>=6.31.1,<8.0.0`, but `5.29.6` is installed
The package `xgrammar` requires `transformers>=4.38.0,<5`, but `5.22.2` is installed
The package `proto-plus` requires `protobuf>=6.33.5,<8.0.0`, but `5.29.6` is installed
"""


def classify(report=SIX, rc=1, versions=None, **overrides):
    return compat.classify_conflicts(
        rc, report, versions if versions is not None else compat.EXPECTED_VERSIONS,
        python_minor=overrides.get("python_minor", "3.12"),
        machine=overrides.get("machine", "x86_64"),
        source_commit=overrides.get("source_commit", compat.UPSTREAM_COMMIT))


class DependencyExceptionTests(unittest.TestCase):
    def test_image_02_exact_six_are_diagnostic_only(self):
        result = classify()
        self.assertTrue(result["accepted"])
        self.assertTrue(result["reviewed_exception_qualified"])
        self.assertFalse(result["dependency_closure_consistent"])
        self.assertEqual(result["reviewed_dependency_conflicts"], 6)
        self.assertEqual(result["qualification"], "diagnostic_with_reviewed_dependency_exceptions")
        self.assertEqual(len(result["observed_conflicts"]), 6)

    def test_genuinely_clean_report(self):
        result = classify("Checked 285 packages in 12ms\nAll installed packages are compatible\n", 0)
        self.assertTrue(result["accepted"])
        self.assertTrue(result["dependency_closure_consistent"])
        self.assertFalse(result["reviewed_exception_qualified"])

    def test_version_drift_of_target_package_is_rejected(self):
        self.assertFalse(classify(SIX.replace("`1.26.4`", "`1.26.5`"))["accepted"])

    def test_requirement_range_drift_rejected(self):
        self.assertFalse(classify(SIX.replace("numpy>=2.0.0", "numpy>=2.1.0"))["accepted"])

    def test_marker_drift_rejected(self):
        self.assertFalse(classify(SIX.replace("python_full_version >= '3.9'", "python_full_version >= '3.10'"))["accepted"])

    def test_offending_package_version_drift_rejected(self):
        versions = dict(compat.EXPECTED_VERSIONS, **{"xgrammar": "0.2.6"})
        self.assertFalse(classify(versions=versions)["accepted"])

    def test_torch_cuda_suffix_drift_rejected(self):
        versions = dict(compat.EXPECTED_VERSIONS, **{"torch": "2.11.0+cu129"})
        self.assertFalse(classify(versions=versions)["accepted"])

    def test_missing_package_rejected(self):
        versions = dict(compat.EXPECTED_VERSIONS)
        versions.pop("vllm")
        self.assertFalse(classify(versions=versions)["accepted"])

    def test_python_and_machine_and_commit_drift_rejected(self):
        for override in ({"python_minor":"3.13"}, {"machine":"aarch64"}, {"source_commit":"different"}):
            with self.subTest(override=override):
                self.assertFalse(classify(**override)["accepted"])

    def test_new_conflict_rejected(self):
        modified = SIX.replace("Found 6 incompatibilities", "Found 7 incompatibilities") + (
            "The package `vllm` requires `torch==2.11.0`, but `2.12.0` is installed\n")
        self.assertFalse(classify(modified)["accepted"])

    def test_removed_conflict_also_requires_review(self):
        lines = SIX.splitlines()
        del lines[2]
        self.assertFalse(classify("\n".join(lines).replace("Found 6", "Found 5"))["accepted"])

    def test_mismatched_count_rejected(self):
        self.assertFalse(classify(SIX.replace("Found 6", "Found 7"))["accepted"])

    def test_duplicate_conflict_rejected(self):
        self.assertFalse(classify(SIX.replace("Found 6", "Found 7") + SIX.splitlines()[2] + "\n")["accepted"])

    def test_malformed_output_rejected(self):
        for rc, report in [(1,"failed to read metadata"), (2,SIX), (0,SIX), (1,""), (0,""),
                           (0,"All installed packages are compatible")]:
            with self.subTest(rc=rc, report=report):
                self.assertFalse(classify(report, rc)["accepted"])

    def test_unexpected_warning_rejected(self):
        self.assertFalse(classify("warning: unexpected\n" + SIX)["accepted"])


if __name__ == "__main__":
    unittest.main()
