"""Offline regression for exploratory a01's Transformers source mismatch.

The wheel test requires explicitly prepared local 5.23.0 and locked 5.22.2
wheels in AGL_TRANSFORMERS_WHEELS, and the usual AGL_PINNED_SOURCES tree.
It installs no dependencies, imports no model stack, and makes no API calls.
"""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
import venv

from experiment.ppo_runtime import source_identity

ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads((ROOT/'experiment/ppo_source_lock.json').read_text())
FAILED_WHEEL_SHA256 = '018e82fd07490ad5bb58414f65e31576033dc7f36bed10c574843f69b90991bf'
FAILED_LLAMA_SHA256 = '97296b66ed591429b0870e4bd00b9fee62b7fd8422cc0c8a4d4bc3ff2680ac8f'


@lru_cache
def image_command():
    # Observe the actual local image declaration with the pinned SDK. No app.run,
    # build, remote call or connection is permitted.
    script = '''
import json, sys
from unittest.mock import patch
def deny(event, args):
    if event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto'):
        raise AssertionError('Network forbidden')
sys.addaudithook(deny)
import modal
commands = []
def record(self, *args, **kwargs):
    commands.extend(args)
    return self
with patch.object(modal.Image, 'run_commands', record):
    import matched_app
print(json.dumps(commands))
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=ROOT,
                            env=dict(os.environ, MODAL_CONFIG_PATH='/dev/null'),
                            capture_output=True, text=True, timeout=30, check=True)
    commands = json.loads(result.stdout)
    assert len(commands) == 1, commands
    return shlex.split(commands[0])


class MatchedTransformersImageTests(unittest.TestCase):
    def test_only_locked_wheel_is_installed_in_training_venv(self):
        command = image_command()
        wheel = LOCK['transformers_wheel']
        self.assertEqual(command, ['uv', 'pip', 'install', '--python',
            '/opt/agent-lightning/.venv/bin/python', '--no-deps', '--no-build',
            wheel['url']+'#sha256='+wheel['digests']['sha256']])


@unittest.skipUnless(os.environ.get('AGL_TRANSFORMERS_WHEELS') and
                     os.environ.get('AGL_PINNED_SOURCES'), 'requires prepared offline source/wheel fixtures')
class MatchedTransformersWheelTests(unittest.TestCase):
    def test_actual_failed_wheel_then_image_install_passes_unchanged_guard(self):
        wheels = Path(os.environ['AGL_TRANSFORMERS_WHEELS'])
        sources = Path(os.environ['AGL_PINNED_SOURCES'])
        failed = wheels/'transformers-5.23.0-py3-none-any.whl'
        corrected = wheels/LOCK['transformers_wheel']['filename']
        self.assertEqual(hashlib.sha256(failed.read_bytes()).hexdigest(), FAILED_WHEEL_SHA256)
        self.assertEqual(hashlib.sha256(corrected.read_bytes()).hexdigest(),
                         LOCK['transformers_wheel']['digests']['sha256'])
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'venv'
            venv.EnvBuilder(with_pip=False).create(target)
            python = target/'bin/python'
            site = Path(subprocess.check_output([str(python), '-c',
                'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True).strip())
            roots = {'agentlightning': sources/'agent-lightning/agentlightning',
                     'verl': sources/'verl/verl', 'transformers': site/'transformers'}
            # Execute the real image's install command locally, substituting only
            # the isolated target and hash-verified offline wheel transport.
            command = image_command().copy()
            command[0] = shutil.which('uv')
            command[4] = str(python)
            command += ['--no-index', '--offline']
            def install(wheel):
                command[7] = str(wheel)
                subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
            install(failed)
            llama = roots['transformers']/'models/llama/modeling_llama.py'
            self.assertEqual(hashlib.sha256(llama.read_bytes()).hexdigest(), FAILED_LLAMA_SHA256)
            with self.assertRaisesRegex(ValueError, 'Unreviewed runtime source: transformers/models/llama/modeling_llama.py'):
                source_identity(roots)
            install(corrected)
            identity = source_identity(roots)
            self.assertEqual(identity['files_verified'], 1+sum(map(len, LOCK['files'].values())))
            # --no-deps left the isolated environment with exactly one package.
            self.assertEqual([p.name for p in site.glob('*.dist-info')],
                             ['transformers-5.22.2.dist-info'])


if __name__ == '__main__':
    unittest.main()
