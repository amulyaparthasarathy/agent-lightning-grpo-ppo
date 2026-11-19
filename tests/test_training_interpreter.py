"""Check interpreter isolation without importing the GPU training stack."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import venv

from experiment.launch import absolute_python_path


class TrainingInterpreterTests(unittest.TestCase):
    def test_symlinked_python_keeps_virtual_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            environment = Path(temporary) / "training-venv"
            venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
            interpreter = environment / "bin/python"
            self.assertTrue(interpreter.is_symlink())

            selected = absolute_python_path(interpreter)
            result = subprocess.check_output(
                [str(selected), "-c",
                 "import json, sys; print(json.dumps([sys.prefix, sys.base_prefix]))"],
                text=True,
                timeout=10,
            )
            prefix, base_prefix = json.loads(result)
            self.assertEqual(Path(prefix).resolve(), environment.resolve())
            self.assertNotEqual(Path(prefix).resolve(), Path(base_prefix).resolve())


if __name__ == "__main__":
    unittest.main()
