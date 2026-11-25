"""Reproduce A02's dependency drift and apply the real reader overlay to local wheels."""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit
import venv

import modal_reader_build
from test_dependency_exceptions import classify, compat

ROOT = Path(__file__).resolve().parents[1]
TRANSFORMERS = json.loads((ROOT/'experiment/ppo_source_lock.json').read_text())['transformers_wheel']


def command():
    with patch('subprocess.run') as run:
        modal_reader_build.install_matched_transformers(TRANSFORMERS['url'], TRANSFORMERS['digests']['sha256'])
    return run.call_args.args[0]


class ReaderDependencyTests(unittest.TestCase):
    def test_overlay_targets_the_exact_training_environment_and_reviewed_wheels(self):
        cmd = command()
        self.assertEqual(cmd[:7], ['uv','pip','install','--python','/opt/agent-lightning/.venv/bin/python',
                                   '--no-deps','--no-build'])
        self.assertEqual(cmd[7], TRANSFORMERS['url']+'#sha256='+TRANSFORMERS['digests']['sha256'])
        self.assertEqual(len(cmd), 10)
        for package, url in zip(('ml-dtypes','opencv-python-headless'),cmd[8:], strict=True):
            parsed = urlsplit(url)
            self.assertEqual(parsed.hostname,'files.pythonhosted.org')
            self.assertRegex(parsed.fragment,r'^sha256=[a-f0-9]{64}$')
            self.assertTrue(Path(parsed.path).name.startswith(package.replace('-','_')+'-'+compat.EXPECTED_VERSIONS[package]+'-'))

    @unittest.skipUnless(os.environ.get('AGL_READER_WHEELS'), 'requires four local Linux wheels and the pinned Transformers wheel')
    def test_cold_versions_rejected_then_real_wheel_overlay_restores_exact_classifier(self):
        fixtures = Path(os.environ['AGL_READER_WHEELS'])
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'venv'
            venv.EnvBuilder(with_pip=False).create(target)
            python = target/'bin/python'
            def versions():
                code = "from importlib.metadata import version;import json;print(json.dumps({p:version(p) for p in ('ml-dtypes','opencv-python-headless')}))"
                return json.loads(subprocess.check_output([python,'-c',code],text=True))
            cmd = command()
            cmd[4] = str(python)
            # The gate reads distribution metadata. Stage the real target wheels even on ARM hosts.
            prefix = cmd[:7]+['--offline','--python-platform','x86_64-manylinux_2_28']
            failed = [next(fixtures.glob(package+'-*.whl')) for package in ('ml_dtypes-0.7.0','opencv_python_headless-5.1.0.95')]
            subprocess.run(prefix+failed,check=True)
            self.assertFalse(classify(versions={**compat.EXPECTED_VERSIONS, **versions()})['accepted'])
            selected = []
            for url in cmd[7:]:
                parsed = urlsplit(url)
                file = fixtures/Path(parsed.path).name
                self.assertEqual(sha256(file.read_bytes()).hexdigest(), parsed.fragment.removeprefix('sha256='))
                selected.append(file)
            subprocess.run(prefix+selected,check=True)
            self.assertTrue(classify(versions={**compat.EXPECTED_VERSIONS, **versions()})['accepted'])
