"""Offline contracts for the cached CUDA image and its diagnostic overlay.

These checks never install training dependencies or allocate a Modal worker.
"""
from pathlib import Path
import ast
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "scripts/install_modal_training.sh"
FINAL = ROOT / "scripts/finalize_modal_training.sh"
APP = ROOT / "modal_app.py"
RUNTIME = ROOT / "validation/runtime.py"
VALIDATION = ROOT / "validation_training.py"
SUPPORT = ROOT / "validation_support.py"


class TrainingCompatibilityTests(unittest.TestCase):
    def test_both_shell_scripts_are_syntactically_valid(self):
        for script in (BASE, FINAL):
            with self.subTest(script=script.name):
                subprocess.run(["bash", "-n", str(script)], check=True)

    def test_heavy_layer_keeps_gcc_source_pin_cuda_and_flashattention(self):
        shell = BASE.read_text()
        self.assertIn("AGL_PIN=cbc56e41e0eab643aa71d195e92e5bba652fa683", shell)
        self.assertIn("bash scripts/setup_verl.sh 0.8.0 cu130", shell)
        self.assertIn("export CC=/usr/bin/gcc", shell)
        self.assertIn("export CXX=/usr/bin/g++", shell)
        self.assertIn("export FLASH_ATTN_CUDA_ARCHS=90", shell)
        self.assertNotIn("uv pip check --python", shell)
        self.assertNotIn("requirements-agent.txt", shell)

    def test_overlay_preserves_versions_and_captures_status(self):
        shell = FINAL.read_text()
        self.assertIn('-r "$AGL_BUILD/requirements-agent.txt"', shell)
        self.assertIn("pip-check-exit-code.txt", shell)
        self.assertIn("check_rc=$?", shell)
        self.assertIn("uv pip check --python .venv/bin/python", shell)
        self.assertNotIn("|| true", shell)
        self.assertIn("set -euo pipefail", shell)
        self.assertIn("build-inputs.sha256", shell)

    def test_modal_uses_two_ordered_image_build_layers(self):
        code = APP.read_text()
        ast.parse(code)
        markers = (".run_function(build_training_stack",
                   '"scripts/finalize_modal_training.sh"',
                   '"experiment/requirements-agent.txt", "/opt/agl-build/requirements-agent.txt"',
                   ".run_function(finalize_training_stack")
        locations = [code.index(m) for m in markers]
        self.assertEqual(locations, sorted(locations))
        self.assertIn('gpu="H100!"', code)
        self.assertNotIn("requirements-compat.txt", code)

    def test_remote_validation_carries_pure_exception_policy(self):
        source = VALIDATION.read_text()
        ast.parse(source)
        self.assertIn('"validation/compatibility.py"', source)
        self.assertIn('"validation/runtime.py"', source)
        self.assertIn(".spawn(req)", source)

    def test_diagnostic_is_distinct_from_clean_and_sync_to_wandb(self):
        source = RUNTIME.read_text()
        ast.parse(source)
        self.assertIn("diagnostic_passed_with_reviewed_exceptions", source)
        self.assertIn('"dependency_closure_consistent"', source)
        self.assertIn('"qualification"', source)
        self.assertIn("dependency-conflicts.json", source)
        self.assertIn("wandb_config.update", source)
        self.assertIn("summary_values.update", source)
        self.assertIn("cv2", source)
        self.assertIn("xgrammar", source)

    def test_submission_only_accepts_correlated_diagnostic(self):
        source = SUPPORT.read_text()
        ast.parse(source)
        self.assertIn("diagnostic_passed_with_reviewed_exceptions", source)
        self.assertIn('result.get("reviewed_dependency_conflicts") != 6', source)
        self.assertIn('training.get("cpu_imports_checked") is not True', source)


if __name__ == "__main__":
    unittest.main()
