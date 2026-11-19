"""Container discovery for the probe, with only its explicitly mounted module."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

class AdmissionImportTests(unittest.TestCase):
    def test_both_functions_import_without_training_checkout_or_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'admission_app.py').write_text((Path(__file__).resolve().parents[1]/'admission_app.py').read_text())
            script='''import sys
sys.path.insert(0,sys.argv[1])
def deny(event,args):
    if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):raise AssertionError('network')
sys.addaudithook(deny)
import modal
assert modal.__version__=='1.6.3'
from modal._runtime.user_code_imports import import_single_function_service
from modal_proto import api_pb2
for name in ('worker','supervise'):
    service=import_single_function_service(api_pb2.Function(module_name='admission_app',function_name=name),None)
    assert service._user_defined_callable.__name__==name
assert 'modal_app' not in sys.modules
assert 'experiment' not in sys.modules
'''
            env={k:v for k,v in os.environ.items() if not k.startswith(('MODAL_','PYTHON','AGL_'))}
            env['MODAL_CONFIG_PATH']='/dev/null'
            result=subprocess.run([sys.executable,'-I','-c',script,tmp],cwd=tmp,env=env,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)

class AdmissionControllerTests(unittest.TestCase):
    def simulate(self, mode='complete'):
        import admission_app as app
        from scripts.phase2_probe import make_plan
        from types import SimpleNamespace as NS
        from unittest.mock import Mock, patch, AsyncMock
        now=[1000.];store={};calls={};cancelled=[]
        def put(key,value,skip_if_exists=False):
            if skip_if_exists and key in store:return False
            store[key]=value;return True
        def read(key):
            if ':worker:' in key:
                slot=int(key.rsplit(':',1)[1]);fid=f'fc-{slot}';start=1000+slot*2
                if now[0]<start or mode=='nine' and slot==9:return None
                return {'slot':slot,'call_id':fid,'started_at':start,'heartbeat_at':now[0],
                        'status':'started','gpu':f'GPU-{0 if mode=="same_gpu" else slot}, H100, 81920 MiB'}
            return store.get(key)
        async def spawn(plan,slot):
            if mode=='ambiguous' and slot==9:raise ConnectionError('response lost')
            fid=f'fc-{slot}';calls[fid]=slot;return NS(object_id=fid)
        def from_id(fid):
            def get(timeout=0):
                slot=calls[fid]
                if mode=='nine' and slot==9:raise TimeoutError('not started')
                return dict(read(f'probe:worker:{slot}'),status='completed',finished_at=now[0])
            return NS(get=get,cancel=lambda **_:cancelled.append(fid))
        registry=NS(put=put,get=read)
        plan=make_plan('probe',{'commit':'b'*40})
        with (patch.object(app,'registry',registry),patch.object(app,'time',NS(time=lambda:now[0],sleep=lambda n:now.__setitem__(0,now[0]+n))),
              patch.object(app,'worker',NS(spawn=NS(aio=spawn))),
              patch.object(app.modal,'current_function_call_id',return_value='fc-supervisor'),
              patch.object(app.modal.FunctionCall,'from_id',side_effect=from_id)):
            result=app.supervise.local(plan)
        return result,calls,cancelled

    def test_measurement_requires_ten_distinct_overlapping_gpu_intervals(self):
        result,calls,cancelled=self.simulate()
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['max_simultaneous'],10)
        self.assertEqual(result['interval_max_simultaneous'],10)
        self.assertGreaterEqual(result['overlap_seconds'],30)
        self.assertEqual(len(calls),10);self.assertEqual(len(cancelled),10)
        self.assertFalse(result['billing_reconciled'])
        self.assertEqual(result['workers']['9']['startup_latency_seconds'],18)

    def test_accepted_calls_or_reused_gpu_are_not_admission_proof(self):
        for mode in ('nine','same_gpu','ambiguous'):
            with self.subTest(mode=mode):
                result,calls,cancelled=self.simulate(mode)
                self.assertEqual(result['status'],'failed')
                self.assertLessEqual(len(calls),10)
                self.assertEqual(set(calls),set(cancelled))
