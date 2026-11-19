"""Exercise service-to-launcher error propagation using real CPU subprocesses."""
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

from experiment import launch


class LauncherFailureTests(unittest.TestCase):
    def test_trainer_traceback_reaches_launcher_without_error_masking(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run/'logs').mkdir()
            port = launch.free_port()
            gateway = '''
from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
HTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
'''.replace('PORT', str(port))
            commands = {
                'gateway': [sys.executable, '-c', gateway],
                'controller': [sys.executable, '-c', 'import time; time.sleep(30)'],
                'trainer': [sys.executable, '-c',
                    "raise ValueError('Unreviewed runtime source: transformers/models/llama/modeling_llama.py')"],
            }
            plan = dict(phase='baseline', policy_version='test', profile='llama3b',
                        calc_dir=tmp, gateway=f'http://127.0.0.1:{port}', overrides=[])
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                code = launch.run_services(commands, plan, run, run, run, 0.5)
            self.assertEqual(code, 1)
            self.assertIn('Trainer exited 1', (run/'error.txt').read_text())
            self.assertIn('logs/trainer.log', stderr.getvalue())
            self.assertIn('Traceback (most recent call last)', stderr.getvalue())
            self.assertIn('ValueError: Unreviewed runtime source:', stderr.getvalue())
            result = json.loads((run/'result.json').read_text())
            self.assertEqual(result['status'], 'trainer_failed')
            self.assertEqual(result['exit_code'], 1)
            self.assertFalse(result['resolved_config_saved'])
            self.assertEqual(result['checkpoints'], [])


if __name__ == '__main__':
    unittest.main()
