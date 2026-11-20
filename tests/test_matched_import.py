"""Real pinned SDK container discovery in an isolated, image-equivalent layout."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MatchedImportTests(unittest.TestCase):
    def discover(self, function, missing=None, original=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'root'; root.mkdir()
            bundle = Path(tmp)/'opt/agent-lightning-lab/experiment'
            bundle.mkdir(parents=True)
            source = (ROOT/'matched_app.py').read_text()
            if original:
                # Explicit fault injection reproduces the missing-builder import mechanism;
                # it is not a claim to execute the private historical source.
                source = source.replace('import modal', 'import modal\nimport modal_app', 1)
            source = source.replace("Path('/opt/agent-lightning-lab/experiment')", f'Path({str(bundle)!r})')
            (root/'matched_app.py').write_text(source)
            for name in ('matched_container.py', 'matched_contract.py', 'matched_supervisor.py', 'matched_exploratory.py', 'matched_data_lock.json', 'matched_remote.py'):
                if name != missing:
                    shutil.copyfile(ROOT/'experiment'/name, bundle/name)
            script = r'''
import sys, json, importlib.util
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
def deny(event, args):
    if event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto'):
        raise AssertionError('Network forbidden')
sys.addaudithook(deny)
import modal
assert modal.__version__ == '1.6.3'
from modal._runtime.user_code_imports import import_single_function_service
from modal_proto import api_pb2
assert importlib.util.find_spec('modal_app') is None
assert importlib.util.find_spec('experiment') is None
# Container lifecycle sets is_local=False before discovery. Keep real decorators,
# SDK module loading, service construction and callable lookup.
with patch.object(modal, 'is_local', return_value=False), patch.object(modal.Image, 'from_registry', side_effect=AssertionError('build side effect')):
    service = import_single_function_service(api_pb2.Function(module_name='matched_app', function_name=sys.argv[2]), None)
assert service._user_defined_callable.__name__ == sys.argv[2]
# Common lifecycle modules are loaded lazily at function entry after discovery.
from experiment import matched_remote
assert 'modal_app' not in sys.modules and 'modal_build' not in sys.modules
assert not any(n in sys.modules for n in ('torch', 'ray', 'transformers'))
print(json.dumps({'function': sys.argv[2], 'sdk': modal.__version__, 'import': 'passed', 'network': 'blocked', 'builders': 'absent'}))
'''
            env = {k:v for k,v in os.environ.items() if not k.startswith(('MODAL_', 'AGL_', 'PYTHON', 'HF_', 'WANDB_'))}
            env['MODAL_CONFIG_PATH'] = '/dev/null'
            return subprocess.run([sys.executable, '-I', '-c', script, str(root), function],
                                  cwd=root, env=env, text=True, capture_output=True, timeout=30)

    def test_fault_injected_builder_import_fails_under_real_sdk(self):
        for function in ('supervise', 'stage_worker'):
            with self.subTest(function=function):
                result = self.discover(function, original=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("ModuleNotFoundError: No module named 'modal_app'", result.stderr)

    def test_both_remote_functions_import_without_build_application(self):
        for function in ('supervise', 'stage_worker'):
            with self.subTest(function=function):
                result = self.discover(function)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)['function'], function)

    def test_missing_runtime_dependencies_fail_closed_for_both_functions(self):
        for function in ('supervise', 'stage_worker'):
            for missing in ('matched_container.py', 'matched_contract.py', 'matched_supervisor.py', 'matched_exploratory.py'):
                with self.subTest(function=function, missing=missing):
                    result = self.discover(function, missing=missing)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(missing[:-3], result.stderr)
