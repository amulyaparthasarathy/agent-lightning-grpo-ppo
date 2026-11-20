"""Frozen exploratory protocol, stage deadlines and unchanged formal campaign controls."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from experiment import matched_contract as c, matched_exploratory as e, matched_supervisor as s
from experiment import matched_launch as launch
from scripts import phase2_study as cli

ROOT = Path(__file__).resolve().parents[1]
# Public baseline; private pre-exploratory history is not a reader dependency.
PARENT = '5c069f7e9cc23d03050d13360ce43e27e47e54aa'


def plan(phase='exploratory'):
    b=e.bounds(phase)
    return c.make_plan('explore-fixture',phase,1,b['stage_seconds'],b['startup_seconds'],b['budget_usd'],10,
        json.loads((ROOT/'evidence/phase2-prepared-data-manifest.json').read_text()),
        {'commit':'b'*40,'dirty':False,'files':{'docs/PHASE2_MATCHED_STUDY_CONTRACT.md':'c'*64}})


def simulate(p, mode='complete'):
    clock=[0.];calls={};active={};records=[];cancelled=[];peak=[0]
    def dispatch(job,stage):
        fid=f'fc-{len(calls)}';calls[fid]=(job,stage,clock[0]);active[fid]=True
        peak[0]=max(peak[0],len(active));return fid
    def get(fid):
        job,stage,start=calls[fid]
        if mode=='training_timeout' and stage['name']=='training' or mode=='baseline_timeout' and stage['name']=='baseline':
            raise TimeoutError()
        if clock[0]-start<10:raise TimeoutError()
        active.pop(fid,None)
        return {'status':'completed','call_id':fid,'output':stage['output'],
                'completion':{'plan_sha256':c.digest(p),'stage':stage['name']}}
    def cancel(fid):cancelled.append(fid);active.pop(fid,None)
    index=s.run(p,dispatch,get,cancel,lambda fid:{'call_id':fid,'time':clock[0]},lambda:False,records.append,
                clock=lambda:clock[0],sleep=lambda n:clock.__setitem__(0,clock[0]+n))
    return index,{'max_active':peak[0],'calls':len(calls),'cancelled':len(cancelled),'elapsed_seconds':clock[0]}


class ExploratoryContractTests(unittest.TestCase):
    def test_six_fixed_fresh_jobs_and_eighteen_stage_identities(self):
        p=plan();c.validate(p)
        self.assertEqual(len(p['jobs']),6);self.assertEqual(p['bounds']['concurrency'],2)
        self.assertEqual({j['run_seed'] for j in p['jobs']},{2001,2002,2003})
        self.assertEqual([j['algorithm'] for j in p['jobs']],list(c.ALGORITHMS)*3)
        self.assertEqual(p['study_protocol'],e.PROTOCOL)
        self.assertEqual(p['evaluation'],'exploratory-legacy-64')
        for i in range(0,6,2):
            a,b=p['jobs'][i:i+2]
            self.assertEqual(a['seed_streams']['paired_question_order'],b['seed_streams']['paired_question_order'])
            self.assertNotEqual(a['seed_streams']['torch_cpu'],b['seed_streams']['torch_cpu'])
        stages=[x for j in p['jobs'] for x in j['stages']]
        self.assertEqual(len({x['wandb']['id'] for x in stages}),18)
        self.assertEqual(len({x['output'] for x in stages}),18)
        for job in p['jobs']:
            self.assertEqual(job['steps'],20)
            for stage in job['stages']:
                cfg=launch.configuration(c.stage_request(p,job,stage),Path('/base'),Path('/data')/stage['output'])
                self.assertEqual(cfg['trainer.save_freq'],20)
                self.assertEqual(cfg['trainer.total_training_steps'],20)
                self.assertIsNone(cfg['trainer.max_actor_ckpt_to_keep'])  # no deletion path
                self.assertEqual(cfg['trainer.resume_mode'],'resume_path' if stage['name']=='checkpoint-eval' else 'disable')
                self.assertEqual(stage['wandb']['config']['evaluation'],'exploratory-legacy-64')
                self.assertEqual(stage['wandb']['config']['stage_timeout_seconds'],c.stage_limit(p,stage))

    def test_funding_and_stage_limits_are_frozen_no_reserve_access(self):
        p=plan()
        self.assertEqual(p['bounds']['stage_seconds_by_name'],{'baseline':1800,'training':5400,'checkpoint-eval':1800})
        self.assertEqual(p['bounds']['max_gpu_seconds'],54000)
        self.assertEqual(p['bounds']['supervisor_seconds'],27600)
        self.assertEqual(e.bounds('exploratory-qualification')['supervisor_seconds'],5100)
        self.assertLess(27600+5100,12*3600)
        self.assertLess(e.costs('exploratory')['compute_ceiling_usd']+12.60+0.80,175)
        self.assertEqual(e.costs('exploratory')['workers_ceiling_usd'],150)
        self.assertEqual(sum(p['funding'][k] for k in ('qualification_usd','study_workers_usd','other_overhead_usd','reserve_usd')),300)
        edits=[lambda x:x['bounds']['stage_seconds_by_name'].update(training=5401),
               lambda x:x['bounds'].update(startup_seconds=301),lambda x:x['bounds'].update(concurrency=5),
               lambda x:x['funding'].update(reserve_released_usd=1),lambda x:x['funding'].update(qualification_usd=51),
               lambda x:x.update(attempt=2),lambda x:x.update(evaluation='heldout-256'),
               lambda x:x['jobs'][0].update(run_seed=2004),lambda x:x['jobs'][0].update(steps=21),
               lambda x:x.update(automatic_retries=1)]
        for edit in edits:
            bad=copy.deepcopy(p);edit(bad)
            with self.assertRaises(ValueError):c.validate(bad)
        with self.assertRaises(ValueError):
            c.make_plan('fixture','exploratory',1,5400,300,175,10,p['data_manifest'],p['source'],6)

    def test_one_short_same_source_qualification_suffices_without_pilot(self):
        q=plan('exploratory-qualification');p=plan()
        self.assertEqual({j['run_seed'] for j in q['jobs']},{9101})
        self.assertEqual({j['steps'] for j in q['jobs']},{2})
        result,_=simulate(q)
        with patch.object(c,'source_identity',return_value=p['source']):
            cli.validate_launch(p,s.key(p)+':'+c.digest(p),result)
            with self.assertRaises(ValueError):cli.validate_launch(p,s.key(p)+':'+c.digest(p))
            bad=copy.deepcopy(result);bad['plan']['source']['commit']='a'*40
            with self.assertRaises(ValueError):cli.validate_launch(p,s.key(p)+':'+c.digest(p),bad)
            bad=copy.deepcopy(result);next(iter(bad['jobs'].values()))['stages'][2]['result']['completion']['plan_sha256']='a'*64
            with self.assertRaises(ValueError):cli.validate_launch(p,s.key(p)+':'+c.digest(p),bad)

    def test_public_baseline_formal_plans_are_identical(self):
        namespace={'__file__':str(ROOT/'experiment/matched_contract.py'),'__name__':'original_contract', '__package__':'experiment'}
        original=subprocess.check_output(['git','show',PARENT+':experiment/matched_contract.py'],cwd=ROOT,text=True)
        exec(compile(original,'original_contract','exec'),namespace)
        p=plan()
        for phase in ('qualification','pilot','campaign'):
            args=('formal',phase,1,3600,600,1000,10,p['data_manifest'],p['source'])
            self.assertEqual(c.make_plan(*args),namespace['make_plan'](*args))
        # Formal gate is byte-for-byte identical, including holdout/pilot/admission requirements.
        original=subprocess.check_output(['git','show',PARENT+':experiment/matched_capacity.py'],cwd=ROOT)
        self.assertEqual((ROOT/'experiment/matched_capacity.py').read_bytes(),original)

    def test_two_lane_rolling_simulation_and_exact_stage_deadlines(self):
        index,metrics=simulate(plan())
        self.assertEqual(index['status'],'completed');self.assertEqual(metrics['max_active'],2)
        self.assertEqual(metrics['calls'],18)
        index,metrics=simulate(plan(),'baseline_timeout')
        self.assertEqual(metrics['elapsed_seconds'],5400);self.assertEqual(metrics['calls'],6)
        self.assertEqual(metrics['cancelled'],6);self.assertEqual(index['status'],'failed')
        index,metrics=simulate(plan(),'training_timeout')
        self.assertEqual(metrics['elapsed_seconds'],16230);self.assertEqual(metrics['calls'],12)
        self.assertEqual(metrics['cancelled'],6);self.assertEqual(index['status'],'failed')
        self.assertTrue(all(j['stages'][2]['status']=='skipped' for j in index['jobs'].values()))

    def test_a02_read_only_protection_never_cancels(self):
        registry=Mock();registry.get.return_value={'call_id':'fc-active-a02'}
        fetch=Mock(side_effect=TimeoutError())
        with self.assertRaisesRegex(ValueError,'leave it untouched'):cli.protect_a02(registry,fetch)
        registry.put.assert_not_called()
        fetch.assert_called_once_with('fc-active-a02')
        registry.get.return_value=None
        cli.protect_a02(registry,Mock(side_effect=AssertionError('no call expected')))

    def test_input_hashing_consumes_existing_stage_budget(self):
        p=plan();job=p['jobs'][0];stage=job['stages'][0]
        with tempfile.TemporaryDirectory() as tmp:
            request=Path(tmp)/'request.json';request.write_text(json.dumps(c.stage_request(p,job,stage)))
            with (patch.object(sys,'argv',['matched_launch.py','--request',str(request)]),
                  patch.dict(os.environ,AGL_MATCHED_DEADLINE='1800'),patch('time.time',return_value=400),
                  patch.object(launch,'validate_inputs'),patch.object(launch,'complete_stage',return_value={}),
                  patch.object(launch.launch,'run_services',return_value=0) as run):
                launch.main()
                self.assertAlmostEqual(run.call_args.args[-1],(1800-400-120)/60)

    def test_extra_checkpoint_fails_without_deleting_anything(self):
        p=plan();job=p['jobs'][0];request=c.stage_request(p,job,job['stages'][1])
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);(run/'audit').mkdir()
            for step in (10,20):(run/'checkpoints'/f'global_step_{step}').mkdir(parents=True)
            events=[{'event':'wandb_finished'},{'event':'trainer_returned','global_step':20}]
            events += [{'event':'weight_refresh','step':i} for i in range(21)]
            events += [{'event':'optimizer_step','role':'actor','count':i} for i in range(1,21)]
            (run/'audit/events-fixture.jsonl').write_text('\n'.join(json.dumps(x) for x in events))
            with self.assertRaisesRegex(ValueError,'exactly the final checkpoint'):launch.complete_stage(run,request,run)
            self.assertEqual(len(list((run/'checkpoints').iterdir())),2)

class ExploratoryLaunchTests(unittest.TestCase):
    def test_single_detached_launch_and_reservation_no_extra_pilot_or_probe(self):
        from contextlib import nullcontext
        from types import SimpleNamespace as NS
        import time
        p=plan();q,_=simulate(plan('exploratory-qualification'))
        store={'index:'+s.key(q['plan']):q}
        def put(key,value,skip_if_exists=False):
            if skip_if_exists and key in store:return False
            store[key]=value;return True
        registry=Mock();registry.get.side_effect=store.get;registry.put.side_effect=put
        cloud=Mock();cloud.Dict.from_name.return_value=registry;cloud.enable_output.side_effect=lambda:nullcontext()
        app=Mock();app.app.run.side_effect=lambda **kw:nullcontext()
        app.supervise.with_options.return_value.spawn.return_value=NS(object_id='fc-exploratory')
        capacity={'workspace':'amulyasp','environment':'main','observed_at':time.time(),'evidence_reference':'TEST ONLY',
                  'resources':{'gpu':'H100!','cpu':16,'memory_mib':196608},'volume_version':2,
                  'gpu_limit':6,'gpu_used':0,'cpu_limit':97,'cpu_used':0,
                  'memory_mib_limit':1181696,'memory_mib_used':0,'container_limit':7,'containers_used':0,
                  'volume_free_gib':600,'worker_scratch_free_gib':100,'cuda_layers_cached':True,'funding':e.funding()}
        capacity.pop('volume_free_gib')
        capacity.update(gpu_limit_source='TEST ONLY provider quota',container_limit_source='TEST ONLY provider quota',
            volume_metadata={'volume_name':c.VOLUME,'volume_id':'vo-test','source':'TEST ONLY VolumeList',
                             'backend_version_raw':2,'backend_version_enum':'VOLUME_FS_VERSION_V2'},
            cpu_ram_quotas={'scope':'aggregate','source':'TEST ONLY provider quota','cpu_limit':97,'cpu_used':0,
                           'memory_mib_limit':1181696,'memory_mib_used':0},
            storage={'capacity_kind':'elastic-no-fixed-byte-capacity','inventory_method':'modal-volume-iterdir',
                     'source':'TEST ONLY listing','listed_bytes':1,'listed_entries':1,'planned_incremental_gib':600},
            scratch={'minimum_free_gib':100,'enforcement':'existing-worker-startup-check'},
            billing={'source':'TEST ONLY billing','remaining_gross_usage_usd':225,'remaining_funded_spending_usd':225},
            cuda_cache_source='TEST ONLY historical image')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name,value in (('plan',p),('qualification',q),('capacity',capacity)):
                (root/(name+'.json')).write_text(json.dumps(value))
            args=['launch','--plan',str(root/'plan.json'),'--qualification',str(root/'qualification.json'),
                  '--capacity',str(root/'capacity.json'),'--authorize',s.key(p)+':'+c.digest(p)]
            with patch.dict(sys.modules,{'modal':cloud,'matched_app':app}),patch.dict(os.environ,MODAL_PROFILE='amulyasp'),patch.object(c,'source_identity',return_value=p['source']):
                self.assertEqual(cli.main(args+['--out',str(root/'receipt.json')]),0)
                with self.assertRaisesRegex(ValueError,'already reserved'):
                    cli.main(args+['--out',str(root/'second-receipt.json')])
        app.app.run.assert_called_once_with(detach=True,environment_name='main')
        app.supervise.with_options.assert_called_once_with(timeout=27600,cpu=(1,1),memory=(2048,2048))
        app.supervise.with_options.return_value.spawn.assert_called_once_with(p)
        self.assertEqual({call.args[0] for call in cloud.Dict.from_name.call_args_list},{'cs329z-matched-study-v2'})

    def test_budget_ceiling_override_cannot_weaken_formal_campaign(self):
        p=plan()
        formal=c.make_plan('formal','campaign',1,3600,600,1000,10,p['data_manifest'],p['source'])
        formal['bounds'].update(supervisor_hourly_ceiling_usd=0,stage_seconds_by_name={'baseline':999999})
        self.assertEqual(c.stage_limit(formal,'baseline'),3600)
        formal['bounds']['budget_usd']=300
        with self.assertRaises(ValueError):c.validate(formal)
