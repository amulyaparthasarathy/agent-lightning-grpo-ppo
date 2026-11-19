"""Ten-worker control-plane simulation, including real concurrent CPU workers."""
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch

from experiment import matched_contract as c, matched_supervisor as s, matched_capacity as capacity
from experiment.matched_holdout import audit_rows
from scripts import phase2_study as cli, phase2_probe as probe

ROOT=Path(__file__).resolve().parents[1]


def campaign(concurrency=10):
    return c.make_plan('tenway-fixture','campaign',1,300,60,100,10,
        json.loads((ROOT/'experiment/matched_data_lock.json').read_text()),
        {'commit':'b'*40,'dirty':False,'files':{'docs/PHASE2_MATCHED_STUDY_CONTRACT.md':'c'*64}},concurrency)


def simulation(behavior='complete', concurrency=10):
    plan=campaign(concurrency);now=[0];active={};calls={};records=[];cancelled=[];peak=[0]
    def dispatch(job,stage):
        if behavior=='ambiguous':raise ConnectionError('response lost after submission')
        fid=f'fc-{len(calls)}';row=(job,stage,now[0]);active[fid]=row;calls[fid]=row
        peak[0]=max(peak[0],len(active));return fid
    def get(fid):
        job,stage,start=calls[fid]
        if behavior=='interrupt':raise KeyboardInterrupt()
        if behavior=='import':raise ModuleNotFoundError('modal_app')
        if behavior in ('startup','lost','stop','cancel_error','stage_timeout','budget'):raise TimeoutError()
        delay=60 if behavior in ('queue','completion_race') else 10
        if now[0]-start < delay:raise TimeoutError()
        active.pop(fid,None)
        return {'status':'completed','call_id':fid,'output':stage['output'],
                'completion':{'plan_sha256':c.digest(plan),'stage':stage['name']}}
    def heartbeat(fid):
        start=calls[fid][2]
        if behavior in ('startup','completion_race','cancel_error') or behavior=='queue' and now[0]-start<50:return None
        if behavior=='lost':return {'call_id':fid,'time':start} if now[0]-start<10 else None
        return {'call_id':fid,'time':now[0]}
    def cancel(fid):
        if behavior=='cancel_error':raise ConnectionError('cancel lost')
        cancelled.append(fid);active.pop(fid,None)
    if behavior=='budget':plan['bounds']['budget_usd']=.1  # fault injection after validation
    try:
        result=s.run(plan,dispatch,get,cancel,heartbeat,lambda:behavior=='stop' and now[0]>=10,
            records.append,clock=lambda:now[0],sleep=lambda n:now.__setitem__(0,now[0]+n))
    except KeyboardInterrupt:
        result=records[-1]
    return result,{'max_active':peak[0],'calls':len(calls),'cancelled':len(cancelled),'elapsed':now[0],
                   'receipts':sum(len(x['stages']) for x in result['jobs'].values()),'status':result['status']},records


class TenWayTests(unittest.TestCase):
    def test_all_concurrency_limits_and_30_distinct_receipts(self):
        for n in range(1,11):
            with self.subTest(concurrency=n):
                p=campaign(n);c.validate(p)
                result,metrics,records=simulation(concurrency=n)
                self.assertEqual(metrics['max_active'],n)
                self.assertEqual((metrics['calls'],metrics['receipts'],metrics['status']),(30,30,'completed'))
                stages=[x for j in result['jobs'].values() for x in j['stages']]
                self.assertEqual(len({x['wandb']['id'] for x in stages}),30)
                self.assertEqual(len({x['output'] for x in stages}),30)
                self.assertEqual(len({x['call_id'] for x in stages}),30)
                self.assertEqual(len({j['job_id'] for j in p['jobs']}),10)
                self.assertEqual({j['run_seed'] for j in p['jobs']},set(range(1001,1006)))
                self.assertEqual(sum(j['algorithm']==c.ALGORITHMS[0] for j in p['jobs']),5)
                # All initial submissions occur before simulated time advances.
                if n==10:self.assertEqual(len({x['stages'][0]['submitted_at'] for x in result['jobs'].values()}),1)
                self.assertTrue(all(x['wandb']['config']['evaluation']=='exploratory-legacy-64' for x in stages))

    def test_queue_delays_and_completion_races_win_over_missing_heartbeat(self):
        for mode in ('queue','completion_race'):
            index,m,_=simulation(mode)
            self.assertEqual(m['status'],'completed');self.assertEqual(m['cancelled'],0)

    def test_failures_cancel_without_retry_and_leave_all_stage_receipts(self):
        for mode,status in [('startup','failed'),('lost','failed'),('import','failed'),('stop','cancelled'),
                            ('cancel_error','unknown'),('ambiguous','unknown'),('interrupt','cancelled'),
                            ('stage_timeout','failed'),('budget','failed')]:
            with self.subTest(mode=mode):
                index,m,_=simulation(mode)
                self.assertEqual(m['status'],status)
                self.assertEqual(m['receipts'],30)
                self.assertLessEqual(m['calls'],10)
                self.assertLessEqual(m['elapsed'],300)
                if mode not in ('ambiguous','cancel_error'):self.assertEqual(m['calls'],m['cancelled'])

    def test_ten_real_cpu_threads_reach_barrier_and_complete_thirty_stages(self):
        p=campaign();pool=ThreadPoolExecutor(max_workers=10);barrier=threading.Barrier(10)
        lock=threading.Lock();active=[0];peak=[0];calls={};heartbeats={};records=[]
        def dispatch(job,stage):
            fid=f'fc-{len(calls)}'
            def worker():
                with lock:
                    active[0]+=1;peak[0]=max(peak[0],active[0]);heartbeats[fid]={'call_id':fid,'time':time.time()}
                barrier.wait(timeout=10)
                time.sleep(.01)
                with lock:active[0]-=1
                return {'status':'completed','call_id':fid,'output':stage['output'],
                        'completion':{'plan_sha256':c.digest(p),'stage':stage['name']}}
            calls[fid]=pool.submit(worker);return fid
        try:
            index=s.run(p,dispatch,lambda fid:calls[fid].result(timeout=0),lambda fid:calls[fid].cancel(),
                heartbeats.get,lambda:False,records.append,sleep=lambda _:time.sleep(.002))
        finally:pool.shutdown()
        self.assertEqual(index['status'],'completed');self.assertEqual(peak[0],10);self.assertEqual(len(calls),30)

    def test_no_relaunch_after_uncertain_atomic_reservation(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        from unittest.mock import Mock
        import tempfile
        p=c.make_plan('reservation','qualification',1,300,60,100,10,
            campaign()['data_manifest'],campaign()['source'])
        lock=threading.Lock();store={};barrier=threading.Barrier(10)
        def put(key,value,skip_if_exists=False):
            with lock:
                if skip_if_exists and key in store:return False
                store[key]=value;return True
        registry=Mock();registry.put.side_effect=put;registry.get.side_effect=lambda k:store.get(k)
        cloud=Mock();cloud.Dict.from_name.return_value=registry;cloud.enable_output.side_effect=lambda:nullcontext()
        app=Mock();app.app.run.side_effect=lambda **kw:nullcontext()
        app.supervise.with_options.return_value.spawn.return_value=SimpleNamespace(object_id='fc-only-one')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'plan.json';path.write_text(json.dumps(p))
            def submit(i):
                barrier.wait(timeout=3)
                try:
                    return cli.main(['launch','--plan',str(path),'--authorize',s.key(p)+':'+c.digest(p),
                                     '--out',str(Path(tmp)/f'receipt-{i}.json')])==0
                except ValueError:return False
            with patch.dict('sys.modules',{'modal':cloud,'matched_app':app}), patch.dict('os.environ',MODAL_PROFILE='amulyasp'), patch.object(c,'source_identity',return_value=p['source']):
                with ThreadPoolExecutor(max_workers=10) as pool:
                    self.assertEqual(sum(pool.map(submit,range(10))),1)
        app.supervise.with_options.return_value.spawn.assert_called_once()

    def test_campaign_legacy_holdout_and_capacity_fail_closed(self):
        p=campaign()
        with self.assertRaisesRegex(ValueError,'untouched'):capacity.validate_campaign_gate(p,None,None,None)
        with self.assertRaises(ValueError):capacity.validate_capacity(None,10)
        evidence={'workspace':'amulyasp','environment':'main','observed_at':100,'evidence_reference':'fixture',
                  'resources':capacity.RESOURCES,'volume_version':2,'gpu_limit':10,'gpu_used':0,
                  'cpu_limit':161,'cpu_used':0,'memory_mib_limit':1968128,'memory_mib_used':0,
                  'container_limit':11,'containers_used':0,'volume_free_gib':3072,'worker_scratch_free_gib':100}
        capacity.validate_capacity(evidence,10,now=100)
        for change in ({'gpu_used':1},{'cpu_used':1},{'memory_mib_used':1},{'containers_used':1},
                       {'volume_free_gib':3071},{'worker_scratch_free_gib':99},{'observed_at':-801}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                capacity.validate_capacity(dict(evidence,**change),10,now=100)

    def test_holdout_rejects_id_question_and_template_contamination(self):
        rows=[{'case_id':f'new-{i}','question':f'Compute {i} apples'} for i in range(256)]
        with self.assertRaises(ValueError):audit_rows(rows[:1],[],rows)
        with self.assertRaises(ValueError):audit_rows([{'case_id':'old','question':' COMPUTE  0 Apples '}],[],rows)
        report=audit_rows([],[],rows)
        self.assertTrue(report['exact_disjoint']);self.assertTrue(report['near_duplicate_pairs'])

    def test_probe_bound_is_independent_and_not_capacity_proof(self):
        p=probe.make_plan('probe-a01',campaign()['source'])
        self.assertEqual(p['workers'],10);self.assertEqual(p['resources'],capacity.RESOURCES)
        self.assertAlmostEqual(p['budget_upper_bound_usd'],12.833333333333334)
        with self.assertRaises(ValueError):probe.make_plan('probe-a01',campaign()['source'],10,1)
        source=(ROOT/'admission_app.py').read_text()
        self.assertNotIn('modal_app',source);self.assertNotIn('Secret.from_name',source)
        self.assertNotIn('Volume.from_name',source);self.assertIn('nvidia-smi',source)

class RuntimeControlTests(unittest.TestCase):
    def test_lost_supervisor_lease_terminates_trainer_subprocess(self):
        import os
        import sys
        import tempfile
        from experiment.matched_container import run_command
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,'lease lost'):
                run_command([sys.executable,'-c','import time; time.sleep(60)'],Path(tmp)/'log.txt',
                            dict(os.environ),10,cancelled=lambda:True)

    def test_thirty_completion_proofs_reconcile_and_plan_drift_fails(self):
        index,_,_=simulation()
        receipts={r['call_id']:r['result'] for j in index['jobs'].values() for r in j['stages']}
        report=cli.reconcile(index,receipts.get,lambda _:None)
        self.assertTrue(report['evidence_complete']);self.assertEqual(len(report['reconciliation']),30)
        next(iter(receipts.values()))['completion']['plan_sha256']='0'*64
        self.assertFalse(cli.reconcile(index,receipts.get,lambda _:None)['evidence_complete'])
