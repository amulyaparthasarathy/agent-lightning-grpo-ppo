"""Real process argument-limit regression and fail-closed file transport checks."""
import errno
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from experiment import request_transport as transport


class RequestTransportTests(unittest.TestCase):
    def test_large_reader_request_crosses_real_process_boundary_without_inline_payload(self):
        request = {'plan': {'schema': 'reader-task-v2'}, 'identity_payload': 'x'*200000}
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp).resolve()
            (run/'request.json').write_text(json.dumps(request))
            env = {**transport.environment(request, run), 'AGL_PPO_RUN_DIR': str(run)}
            self.assertLess(len(json.dumps({'env_vars': env})), 4096)
            # The old representation fails even without Ray/model imports.
            with self.assertRaises(OSError) as caught:
                subprocess.run(['/bin/true', json.dumps({'env_vars': {'AGL_MATCHED_REQUEST': json.dumps(request)}})], check=True)
            self.assertEqual(caught.exception.errno, errno.E2BIG)
            subprocess.run(['/bin/true', json.dumps({'env_vars': env})], env={**os.environ, **env}, check=True)
            self.assertEqual(transport.load(env), request)

    def test_tamper_missing_ambiguous_and_unowned_paths_are_rejected(self):
        request = {'plan': {'schema': 'reader-task-v2'}, 'binding': 'original'}
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp).resolve(); path = run/'request.json'
            env = {**transport.environment(request, run), 'AGL_PPO_RUN_DIR': str(run)}
            with self.assertRaises(ValueError): transport.load(env)
            path.write_text(json.dumps(request))
            for changes in ({'AGL_MATCHED_REQUEST': json.dumps(request)},
                            {'AGL_MATCHED_REQUEST_SHA256': '0'*64},
                            {'AGL_MATCHED_REQUEST_FILE': str(run/'other.json')},
                            {'AGL_PPO_RUN_DIR': '/different-owned-stage'}):
                with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                    transport.load({**env, **changes})
            path.write_text(json.dumps({**request, 'binding': 'changed'}))
            with self.assertRaises(ValueError): transport.load(env)
            path.rename(run/'other.json'); path.symlink_to(run/'other.json')
            with self.assertRaises(ValueError): transport.load(env)
            with self.assertRaises(ValueError): transport.load({'AGL_MATCHED_REQUEST': json.dumps(request)})

    def test_historical_inline_contract_is_unchanged(self):
        request = {'plan': {'schema': 'matched-study-v2'}}
        env = transport.environment(request, Path('/historical'))
        self.assertEqual(env, {'AGL_MATCHED_REQUEST': json.dumps(request)})
        self.assertEqual(transport.load(env), request)


if __name__ == '__main__':
    unittest.main()
