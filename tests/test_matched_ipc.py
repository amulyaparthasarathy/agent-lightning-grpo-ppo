"""Pinned vLLM + real ZeroMQ/Ray subprocess regression; CPU only."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiment import launch, matched_contract as c

ROOT=Path(__file__).resolve().parents[1]
HELPER=ROOT/'tests/pinned_ipc_support.py'


@unittest.skipUnless(importlib.util.find_spec('ray') and importlib.util.find_spec('zmq'),
                     'requires explicit pinned CPU environment with Ray and pyzmq')
class IPCTests(unittest.TestCase):
    def test_old_native_endpoint_bind_fails_for_both_algorithms(self):
        for algorithm in c.ALGORITHMS:
            stage=f'/data/phase2/studies/agl-explore-20-v1-r02/exploratory-qualification/agl-explore-20-v1-r02-exploratory-qualification-{algorithm}-s9101-a01/baseline'
            env={**os.environ,'TMPDIR':stage+'/temporary/tmpdir'}
            env.pop('VLLM_RPC_BASE_PATH',None)
            # No /data mutation: prime tempfile's cached value with the observed
            # existing container path in this isolated old-code subprocess.
            code='import tempfile,runpy,sys; tempfile.tempdir=sys.argv.pop(); sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name="__main__")'
            proc=subprocess.run([sys.executable,'-c',code,str(HELPER),'old',env['TMPDIR']],
                                env=env,text=True,capture_output=True,timeout=30)
            self.assertEqual(proc.returncode,0,proc.stderr)
            result=json.loads(proc.stdout)
            self.assertIn('longer than 107 characters',result['old_error'])
            self.assertIn(result['path_bytes'],(205,207))
            print(json.dumps(result),flush=True)

    def test_long_stages_isolated_real_launcher_ray_and_ipc_children(self):
        bounds=c.exploratory.bounds('exploratory-qualification')
        plan=c.make_plan('agl-explore-20-v1-r02','exploratory-qualification',1,bounds['stage_seconds'],
            bounds['startup_seconds'],50,10,json.loads((ROOT/'evidence/phase2-prepared-data-manifest.json').read_text()),
            {'commit':'b'*40,'dirty':False,'files':{'docs/PHASE2_MATCHED_STUDY_CONTRACT.md':'c'*64}})
        with tempfile.TemporaryDirectory(dir='/tmp') as temp:
            parent=Path(temp)
            stale=parent/('inherited-and-cached-long-temp-'*5);stale.mkdir()
            # Both inherited environment AND already cached launcher tempfile
            # are long. Only mkdtemp(dir='/tmp') may choose local scratch.
            with patch.dict(os.environ,{'TMPDIR':str(stale),'VLLM_RPC_BASE_PATH':str(stale)}),patch.object(tempfile,'tempdir',str(stale)):
                def run_job(job):
                    run=parent/job['stages'][0]['output'];(run/'logs').mkdir(parents=True)
                    port=launch.free_port()
                    gateway="from http.server import HTTPServer,BaseHTTPRequestHandler\nclass H(BaseHTTPRequestHandler):\n def do_GET(self): self.send_response(200);self.end_headers()\nHTTPServer(('127.0.0.1',"+str(port)+"),H).serve_forever()"
                    commands={'gateway':[sys.executable,'-c',gateway],
                              'controller':[sys.executable,'-c','import time;time.sleep(240)'],
                              'trainer':[sys.executable,str(HELPER),'ray']}
                    req=c.stage_request(plan,job,job['stages'][0])
                    rp=dict(protocol=c.PROTOCOL,algorithm=job['algorithm'],request=req,phase='baseline',
                            policy_version=c.PROTOCOL,profile='llama3b',calc_dir=str(ROOT),
                            gateway=f'http://127.0.0.1:{port}',overrides=[])
                    code=launch.run_services(commands,rp,run,ROOT,ROOT/'tests',3)
                    self.assertEqual(code,0,(run/'logs/trainer.log').read_text()[-10000:])
                    proof=json.loads((run/'socket-proof.json').read_text())
                    scratch=proof['env']['TMPDIR']
                    try:
                        self.assertTrue(Path(scratch).is_absolute());self.assertEqual(Path(scratch).parent,Path('/tmp'))
                        self.assertLessEqual(proof['path_bytes'],107)
                        for snap in (proof,proof['peer'],proof['driver']):
                            for name in ('TMPDIR','RAY_TMPDIR','VLLM_RPC_BASE_PATH'):
                                self.assertEqual(snap['env'][name],scratch)
                            self.assertEqual(snap['tempfile'],scratch)
                            self.assertEqual(snap['env']['AGL_TRACE_DIR'],str(run/'traces'))
                            self.assertEqual(snap['env']['WANDB_DIR'],str(run/'temporary/wandb_dir'))
                        self.assertNotEqual(proof['pid'],proof['peer']['pid'])
                        for path in ('logs/trainer.log','observability.json','result.json','traces'):
                            self.assertTrue((run/path).exists())
                        self.assertFalse((run/'temporary/tmpdir').exists())
                        print(json.dumps(proof),flush=True)
                        return scratch
                    finally: shutil.rmtree(scratch)
                with ThreadPoolExecutor(max_workers=2) as pool:
                    scratches=list(pool.map(run_job,plan['jobs']))
                self.assertNotEqual(*scratches)

if __name__=='__main__':unittest.main(verbosity=2)
