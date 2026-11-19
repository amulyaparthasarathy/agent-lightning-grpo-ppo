"""Real pinned Hydra/dataclass checks in an isolated CPU subprocess."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest


@unittest.skipUnless(importlib.util.find_spec('hydra') and importlib.util.find_spec('torch'),
                     'requires explicit pinned CPU environment (including hydra-core)')
class PinnedConfigTests(unittest.TestCase):
    def test_complete_generated_worker_configs_and_sampling_seed(self):
        result = subprocess.run([sys.executable, str(Path(__file__).with_name('pinned_config_checks.py'))],
                                text=True, capture_output=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)
        print(result.stdout, end='')
        print(result.stderr, end='')
