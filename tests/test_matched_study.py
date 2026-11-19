import ast
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from experiment import matched_contract as c, matched_launch as launch, matched_supervisor as supervisor
from experiment import matched_runtime as runtime
from scripts import phase2_study as cli

ROOT = Path(__file__).resolve().parents[1]


def plan(phase='qualification'):
    files = json.loads((ROOT/'experiment/matched_data_lock.json').read_text())['files']
    return c.make_plan('fixture',phase,1,3600,600,200,10,{'files':files},
                      {'commit':'b'*40,'dirty':False,'files':{'docs/PHASE2_MATCHED_STUDY_CONTRACT.md':'c'*64}})


class ManifestTests(unittest.TestCase):
    def test_disjoint_paths_remote_identities_and_paired_schedule(self):
        p=plan();c.validate(p)
        self.assertEqual(len({s['output'] for j in p['jobs'] for s in j['stages']}),6)
        a,b=p['jobs']
        self.assertEqual(a['seed_streams']['paired_question_order'],b['seed_streams']['paired_question_order'])
        self.assertNotEqual(a['seed_streams']['torch_cpu'],b['seed_streams']['torch_cpu'])
        for j in p['jobs']:
            for s in j['stages']:
                request=c.stage_request(p,j,s)
                cfg=launch.configuration(request,Path('/model'),Path('/data')/s['output'])
                self.assertEqual(cfg['trainer.total_training_steps'],2)
                self.assertEqual(cfg['trainer.val_only'],s['name']!='training')
                self.assertEqual(cfg['data.seed'],j['seed_streams']['paired_question_order'])
                self.assertNotIn('actor_rollout_ref.rollout.seed',cfg)
                self.assertEqual(cfg['actor_rollout_ref.rollout.engine_kwargs.vllm.seed'],j['seed_streams']['actor_sampling'])
                self.assertEqual(cfg['critic.enable'],j['algorithm']==c.ALGORITHMS[1])
                self.assertEqual(cfg['ray_kwargs.ray_init.runtime_env.worker_process_setup_hook'], 'matched_runtime.register_in_worker')
                self.assertEqual(json.loads(cfg['ray_kwargs.ray_init.runtime_env.env_vars']['AGL_MATCHED_REQUEST']),request)
                self.assertEqual(cfg['trainer.resume_mode'],'resume_path' if s['name']=='checkpoint-eval' else 'disable')

    def test_rejects_tampering_budget_collision_leakage_and_campaign(self):
        edits = [lambda p:p['bounds'].update(budget_usd=1), lambda p:p['bounds'].update(stage_seconds=999999),
                 lambda p:p.update(phase='campaign'), lambda p:p.update(volume='cs329z-agent-lightning'),
                 lambda p:p.update(automatic_retries=1),lambda p:p['jobs'].append(p['jobs'][0]),
                 lambda p:p['jobs'][0].update(output_root='../runs/train-01'),
                 lambda p:p['jobs'][0]['stages'][1].update(output=p['jobs'][0]['stages'][0]['output']),
                 lambda p:p['jobs'][0]['seed_streams'].update(critic_init=9),
                 lambda p:p['data_manifest']['files']['eval_main.parquet']['case_ids'].__setitem__(0,'0')]
        for edit in edits:
            p=plan();edit(p)
            with self.assertRaises(ValueError):c.validate(p)

    def test_launch_is_explicit_clean_and_qualified(self):
        p=plan()
        with self.assertRaises(ValueError):cli.validate_launch(p,'')
        token=f'{supervisor.key(p)}:{c.digest(p)}'
        with patch.object(c,'source_identity',return_value=p['source']):
            cli.validate_launch(p,token)
        with patch.object(c,'source_identity',return_value={}),self.assertRaises(ValueError):
            cli.validate_launch(p,token)
        p=plan('pilot')
        with patch.object(c,'source_identity',return_value=p['source']),self.assertRaisesRegex(ValueError,'qualification'):
            cli.validate_launch(p,f'{supervisor.key(p)}:{c.digest(p)}')

    def test_offline_plan_and_validate_do_not_import_modal(self):
        import builtins
        real=builtins.__import__
        def guarded(name,*a,**kw):
            if name=='modal' or name.startswith('modal.') or name=='matched_app':raise AssertionError('cloud access')
            return real(name,*a,**kw)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'plan.json';path.write_text(json.dumps(plan()))
            with patch('builtins.__import__',side_effect=guarded):
                self.assertEqual(cli.main(['validate','--plan',str(path)]),0)

    def test_worker_program_and_remote_resource_dispatch(self):
        ast.parse(launch.WORKER)
        source=ast.parse((ROOT/'matched_app.py').read_text())
        fn=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='stage_worker')
        resources={k.arg:ast.literal_eval(k.value) for k in fn.decorator_list[0].keywords
                   if k.arg in ('gpu','cpu','memory','retries','max_containers','single_use_containers')}
        self.assertEqual(resources,dict(gpu='H100!',cpu=16,memory=196608,retries=0,max_containers=10,single_use_containers=True))
        p=plan('pilot')
        for job in p['jobs']:
            run=Path('/data')/job['stages'][1]['output']
            runtime_plan,cmd=launch.make_plan(c.stage_request(p,job,job['stages'][1]),Path('/bundle'),Path('/repo'),
                            Path('/model'),Path('/data/restored/data-small'),Path('/python'),run,8000)
            self.assertEqual(runtime_plan['algorithm'],job['algorithm'])
            self.assertEqual(runtime_plan['protocol'],c.PROTOCOL)
            cfg=dict(x.split('=',1) for x in runtime_plan['overrides'])
            self.assertEqual(json.loads(cfg['trainer.total_training_steps']),20)
            self.assertEqual(cmd['trainer'][2],str(run/'trainer_worker.py'))

    def test_tracking_remote_init_and_terminal_flush(self):
        p=plan();stage=p['jobs'][0]['stages'][0];wb=Mock()
        with tempfile.TemporaryDirectory() as tmp,patch.object(runtime.audit,'emit') as emit:
            tracker=runtime.tracking_class(wb,stage,Path(tmp))('ignored','ignored',['wandb'],{'trainer':{}})
            for key in ('entity','project','group','id','name','job_type','resume'):
                self.assertEqual(wb.init.call_args.kwargs[key],stage['wandb'][key])
            tracker.log({'loss':1.},20);tracker.finish()
            wb.init.return_value.finish.assert_called_once()
            self.assertEqual(json.loads((Path(tmp)/'metrics.jsonl').read_text())['step'],20)
            emit.assert_called_with('wandb_finished',id=stage['wandb']['id'])


class SupervisorTests(unittest.TestCase):
    def simulate(self, behavior='complete'):
        p=plan();now=[0];records=[];calls={};cancelled=[]
        def dispatch(job,stage):
            if behavior=='spawn_disconnect':raise ConnectionError('response lost')
            fid=f'fc-{len(calls)}';calls[fid]=(job,stage);return fid
        def result(fid):
            job,stage=calls[fid]
            if behavior in ('hydration','stalled','stop'):raise TimeoutError()
            if behavior=='failure':raise RuntimeError('trainer failed')
            if behavior=='no_receipt':return {'status':'completed'}
            return {'status':'completed','call_id':fid,'output':stage['output'],'completion':{'plan_sha256':c.digest(p),'stage':stage['name']}}
        def heartbeat(fid):
            return None if behavior=='hydration' else {'call_id':fid,'time':0 if behavior=='stalled' else now[0]}
        index=supervisor.run(p,dispatch,result,cancelled.append,heartbeat,
                  lambda:behavior=='stop',records.append,clock=lambda:now[0],sleep=lambda secs:now.__setitem__(0,now[0]+secs))
        return index,records,calls,cancelled,now[0]

    def test_durable_stage_dispatch_and_no_campaign_chaining(self):
        index,records,calls,cancelled,_=self.simulate()
        self.assertEqual(index['status'],'completed');self.assertEqual(len(calls),6);self.assertEqual(cancelled,[])
        for state in index['jobs'].values():
            self.assertEqual([s['stage'] for s in state['stages']],list(c.STAGES))
        for call in calls:
            self.assertTrue(any(any(s.get('call_id')==call and s['status']=='submitted' for j in r['jobs'].values()
                                   for s in j['stages']) for r in records))

    def test_hydration_loop_bounded_and_only_owned_calls_cancelled(self):
        index,_,calls,cancelled,elapsed=self.simulate('hydration')
        self.assertEqual(index['status'],'failed');self.assertEqual(set(cancelled),set(calls))
        self.assertLessEqual(elapsed,605);self.assertEqual(len(calls),2)
        self.assertTrue(all(s['error']=='startup_no_heartbeat' for j in index['jobs'].values() for s in j['stages'] if s.get('call_id')))

    def test_stalled_failure_owner_stop_and_uncertain_submission_never_retry(self):
        for behavior,expected in (('stalled','failed'),('failure','failed'),('stop','cancelled'),
                                  ('spawn_disconnect','unknown'),('no_receipt','unknown')):
            with self.subTest(behavior=behavior):
                index,_,calls,cancelled,_=self.simulate(behavior)
                self.assertEqual(index['status'],expected)
                self.assertLessEqual(len(calls),2)
                self.assertTrue(set(cancelled)<=set(calls))

    def test_status_does_not_convert_local_or_remote_exit_into_completion(self):
        index,*_=self.simulate()
        report=cli.reconcile(index,lambda call:None,lambda call:None,now=1)
        self.assertFalse(report['evidence_complete'])
        self.assertTrue(all(r['state']=='unknown' for r in report['reconciliation']))

    def test_targeted_stop_filters_completed_calls(self):
        index,*_=self.simulate('stop')
        self.assertEqual(supervisor.owned_calls(index),[])
        for i,state in enumerate(index['jobs'].values()):
            state['stages'][0].update(status='failed',call_id=f'fc-{i}')
        self.assertEqual(len(supervisor.owned_calls(index)),2)


class CheckpointTests(unittest.TestCase):
    def test_exact_final_identity_detects_byte_config_and_algorithm_changes(self):
        p=plan();job=p['jobs'][1]
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/'training';checkpoint=run/'checkpoints/global_step_2';checkpoint.mkdir(parents=True)
            (run/'resolved_config.json').write_text('{}')
            for role in ('actor','critic'):
                (checkpoint/role).mkdir()
                for name in ('model_world_size_1_rank_0.pt','optim_world_size_1_rank_0.pt','extra_state_world_size_1_rank_0.pt'):
                    (checkpoint/role/name).write_bytes(b'fixture')
            (checkpoint.parent/'latest_checkpointed_iteration.txt').write_text('2')
            req=c.stage_request(p,job,job['stages'][1])
            identity=launch.checkpoint_identity(checkpoint,req,run)
            (checkpoint/'matched_identity.json').write_text(json.dumps(identity))
            reload=c.stage_request(p,job,job['stages'][2])
            self.assertEqual(launch.verify_checkpoint(checkpoint,reload),c.digest(identity))
            (checkpoint/'critic/model_world_size_1_rank_0.pt').write_bytes(b'corrupted')
            with self.assertRaises(ValueError):launch.verify_checkpoint(checkpoint,reload)

    def test_exit_zero_without_terminal_audit_is_not_completion(self):
        p=plan();job=p['jobs'][0]
        with tempfile.TemporaryDirectory() as tmp,self.assertRaisesRegex(ValueError,'W&B'):
            launch.complete_stage(Path(tmp),c.stage_request(p,job,job['stages'][1]),Path(tmp))


class CLIAndEvidenceTests(unittest.TestCase):
    def test_launch_dispatches_one_detached_bounded_supervisor(self):
        from contextlib import nullcontext
        p=plan();registry=Mock();registry.get.return_value=None;registry.put.return_value=True
        cloud=Mock();cloud.Dict.from_name.return_value=registry;cloud.enable_output.side_effect=lambda:nullcontext()
        app=Mock();app.app.run.side_effect=lambda **kw:nullcontext()
        app.supervise.with_options.return_value.spawn.return_value=NS(object_id='fc-supervisor')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'plan.json';path.write_text(json.dumps(p));out=Path(tmp)/'receipt.json'
            with (patch.dict('sys.modules',{'modal':cloud,'matched_app':app}),
                  patch.dict('os.environ',MODAL_PROFILE='amulyasp'),patch.object(c,'source_identity',return_value=p['source'])):
                result=cli.main(['launch','--plan',str(path),'--authorize',f'{supervisor.key(p)}:{c.digest(p)}','--out',str(out)])
            self.assertEqual(result,0)
            app.app.run.assert_called_once_with(detach=True,environment_name='main')
            app.supervise.with_options.assert_called_once_with(timeout=p['bounds']['supervisor_seconds'])
            app.supervise.with_options.return_value.spawn.assert_called_once_with(p)
            self.assertFalse(json.loads(out.read_text())['completion_proven'])

    def test_volume_missing_or_mismatched_receipts_keep_evidence_incomplete(self):
        p=plan();index={'jobs':{j['job_id']:{'stages':[{'stage':s['name'],'call_id':f'fc-{i}'}
                    for i,s in enumerate(j['stages'])]} for j in p['jobs']}}
        def unavailable(name):raise FileNotFoundError(name)
        self.assertTrue(all(not x['available'] for x in cli.reconcile_volume(p,index,unavailable)))
        def read(path):
            stage=next(s for j in p['jobs'] for s in j['stages'] if path.startswith(s['output']+'/'))
            i=list(c.STAGES).index(stage['name'])
            completion={'plan_sha256':c.digest(p),'stage':stage['name']}
            values={'receipt.json':{'status':'completed','call_id':f'fc-{i}','completion':completion},
                    'completion.json':completion,'observability.json':{'wandb':stage['wandb']}}
            return [json.dumps(values[Path(path).name]).encode()]
        self.assertTrue(all(x['available'] for x in cli.reconcile_volume(p,index,read)))
