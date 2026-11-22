"""Real Ray 2.60 CPU workers; model construction/optimizer execution is absent."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class RayRequestTransportTests(unittest.TestCase):
    def test_reader_request_larger_than_single_argument_limit_in_real_ray_worker(self):
        script = '''
import json, os, sys
from pathlib import Path
import ray
from experiment import request_transport as t
run = Path(sys.argv[1])
request = {'plan': {'schema': 'reader-task-v2'}, 'full_identity': 'x'*250000}
(run/'request.json').write_text(json.dumps(request))
env = dict(t.environment(request, run), AGL_PPO_RUN_DIR=str(run))
ray.init(num_cpus=2, include_dashboard=False, object_store_memory=100*1024**2,
         runtime_env={'env_vars': env})
try:
    @ray.remote
    def read_request():
        from experiment import request_transport as transport
        import os
        value = transport.load()
        return transport.digest(value), len(value['full_identity']), os.getpid()
    digest, size, pid = ray.get(read_request.remote(), timeout=60)
    assert digest == t.digest(request) and size == 250000 and pid != os.getpid()
finally:
    ray.shutdown()
'''
        with tempfile.TemporaryDirectory(prefix='rq-') as tmp:
            env = {k: v for k, v in os.environ.items() if not k.startswith('AGL_MATCHED_REQUEST')}
            env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])
            env['RAY_TMPDIR'] = tmp
            result = subprocess.run([sys.executable, '-c', script, tmp], env=env,
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr[-6000:])
