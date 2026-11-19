"""Isolated module-import checks using the real client SDK, with no Modal RPC."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RemoteImportTests(unittest.TestCase):
    def import_remote_layout(self, include_contract=True):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            remote_root = root/'root'
            bundle = root/'opt/agent-lightning-lab/experiment'
            remote_root.mkdir()
            bundle.mkdir(parents=True)
            # Relocate only the absolute runtime path into the isolated fixture.
            # Keep all module imports, image declarations and decorators intact.
            source = (ROOT/'modal_app.py').read_text()
            binding = 'BUNDLE = Path("/opt/agent-lightning-lab/experiment")'
            self.assertEqual(source.count(binding), 1)
            (remote_root/'modal_app.py').write_text(source.replace(binding, f'BUNDLE = Path({str(bundle)!r})'))
            shutil.copyfile(ROOT/'modal_build.py', remote_root/'modal_build.py')
            if include_contract:
                shutil.copyfile(ROOT/'experiment/ppo_contract.py', bundle/'ppo_contract.py')
            script = r'''
import importlib.util
import json
from pathlib import Path
import sys

def deny_network(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo', 'socket.sendto'}:
        raise AssertionError('Network access forbidden during offline import')

sys.addaudithook(deny_network)
remote_root, bundle = map(Path, sys.argv[1:])
sys.path.insert(0, str(remote_root))
assert importlib.util.find_spec('experiment') is None
assert not (remote_root/'experiment').exists()
assert not (remote_root/'scripts').exists()
assert str(bundle.parent) not in sys.path
from modal._runtime.user_code_imports import import_single_function_service
from modal_proto import api_pb2

# Exercise the installed SDK's container-side import/lookup/dependency path.
# This constructs local declarations only: no resolver, client RPC or invocation.
service = import_single_function_service(
    api_pb2.Function(module_name='modal_app', function_name='gpu_phase'), None)
import modal_app
import modal
assert service._user_defined_callable.__name__ == 'gpu_phase'
assert service._user_defined_callable.__globals__['ppo_contract'] is modal_app.ppo_contract
assert Path(modal_app.ppo_contract.__file__).resolve() == bundle/'ppo_contract.py'
assert modal_app.LOCAL == remote_root
assert modal_app.BUNDLE == bundle
assert modal_app.WORKING_VOLUME == 'working-import-fixture'
assert modal_app.VOLUME_NAME == 'working-import-fixture'
# Reaching the final entrypoint proves the whole module body imported, including
# lazy image/Volume/Secret declarations and all real SDK function decorators.
assert callable(modal_app.main)
for name in ('prepare_assets', 'gpu_phase', 'collect_evidence'):
    assert isinstance(getattr(modal_app, name), modal.Function)
    imported = import_single_function_service(
        api_pb2.Function(module_name='modal_app', function_name=name), None)
    assert imported._user_defined_callable.__name__ == name
request = modal_app.request('smoke', 'ppo-mc-v1-import-fixture', wandb_entity='team',
                           algorithm='ppo-mc-v1', data_layout='restored')
assert request['protocol'] == 'ppo-mc-smoke-legacy-retention-v1'
assert modal_app.ppo_contract.identity()['algorithm'] == 'grpo'
print(json.dumps({'module_import': 'complete', 'contract_source': 'packaged-runtime',
                  'network': 'blocked'}))
'''
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith(('MODAL_', 'HF_', 'WANDB_', 'AGL_', 'PYTHON'))}
            env.update(MODAL_CONFIG_PATH='/dev/null', HF_HUB_OFFLINE='1',
                       AGL_WORKING_VOLUME='working-import-fixture')
            return subprocess.run([sys.executable, '-I', '-c', script, str(remote_root), str(bundle)],
                                  cwd=remote_root, env=env, text=True, capture_output=True, timeout=30)

    def test_complete_module_import_with_only_packaged_contract(self):
        result = self.import_remote_layout()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {'module_import': 'complete',
                         'contract_source': 'packaged-runtime', 'network': 'blocked'})

    def test_missing_packaged_contract_fails_closed(self):
        result = self.import_remote_layout(include_contract=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertRegex(result.stderr, r'(ModuleNotFoundError|ImportError)')
        self.assertIn('experiment', result.stderr)
