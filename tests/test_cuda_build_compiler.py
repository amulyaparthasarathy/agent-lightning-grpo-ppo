"""Regression guards for the FlashAttention CPU image-build host compiler.

These checks inspect the *recipe*, not a Modal build, nvcc, or GPU kernel.
"""
from pathlib import Path
import subprocess
import unittest

INSTALLER = Path(__file__).resolve().parents[1] / "scripts/install_modal_training.sh"


class CudaImageBuildCompilerTests(unittest.TestCase):
    def test_shell_syntax(self):
        subprocess.run(["bash", "-n", str(INSTALLER)], check=True)

    def test_gcc_is_selected_before_upstream_flashattention_build(self):
        script = INSTALLER.read_text()
        setup = script.index("bash scripts/setup_verl.sh 0.8.0 cu130")
        for setting in ("export CC=/usr/bin/gcc", "export CXX=/usr/bin/g++",
                        "export CUDAHOSTCXX=/usr/bin/g++"):
            with self.subTest(setting=setting):
                self.assertIn(setting, script)
                self.assertLess(script.index(setting), setup)
        self.assertIn('[[ ! -x "$compiler" ]]', script)
        self.assertIn('"$CXX" -dumpfullversion -dumpversion', script)

    def test_pinned_stack_and_sm90_are_preserved(self):
        script = INSTALLER.read_text()
        for required in ("AGL_PIN=cbc56e41e0eab643aa71d195e92e5bba652fa683",
                         "export TORCH_CUDA_ARCH_LIST=9.0", "export FLASH_ATTN_CUDA_ARCHS=90",
                         "bash scripts/setup_verl.sh 0.8.0 cu130",
                         "build-settings.txt", "CXX_VERSION=%s"):
            with self.subTest(required=required):
                self.assertIn(required, script)

    def test_heavy_layer_is_independent_of_agent_requirements(self):
        script = INSTALLER.read_text()
        self.assertIn("bash scripts/setup_verl.sh 0.8.0 cu130", script)
        self.assertNotIn("requirements-agent.txt", script)
        self.assertNotIn("uv pip check --python", script)
        self.assertIn("base-installed-packages.txt", script)


if __name__ == "__main__":
    unittest.main()
