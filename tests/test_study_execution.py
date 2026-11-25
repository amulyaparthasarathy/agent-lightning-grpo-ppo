"""Synthetic study stages through the production worker, supervisor, release and offline review."""
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiment import matched_launch, matched_remote, matched_supervisor, reader_contract as c, reader_evidence
from experiment import reader_preflight, recovery_learning, recovery_task as task
from recovery_harness_support import run_episode, TEXT
from recovery_learning_support import resolved_fixture
from recovery_fixtures import assets_manifest
from study_support import profile, fixture_source
from test_reader_training import Registry
from test_recovery_learning_evidence import scripted_services, write


def trace_writer(template):
    """Instantiate a single scripted malformed-answer trace with each case's own request and identities."""
    def emit(run, script, *, row, request, route, rollout, simulate_live_labels):
        plan, job, stage = (request[k] for k in ('plan','job','stage'))
        records = copy.deepcopy(template)
        trace = c.digest([stage['wandb']['id'], rollout])[:32]
        for record in records:
            record.update(trace_id=trace, rollout_id=rollout, case_id=row['case_id'], eval_split=row['split'],
                          route_mode=route, phase=stage['name'], experiment_id=job['job_id'],
                          wandb_run_id=stage['wandb']['id'], source_sha256=c.digest(plan['source']),
                          task_sha256=c.digest(plan['task_identity']), configuration_sha256=plan['configuration_sha256'])
            if record['kind'] == 'llm':
                record['inputs']['messages'][-1]['content'] = row['question']+'\n'+task.PROMPT
            if record['kind'] == 'episode':
                record['inputs']['question'] = row['question']
        directory = run/'traces'
        directory.mkdir(exist_ok=True)
        (directory/f'trajectory-{trace}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        return {'error':None}
    return emit


class StudyTests(unittest.TestCase):
    def test_profile_and_failed_dispatch(self):
        p = profile()
        plan = c.make_plan(p, fixture_source())
        c.validate(plan)
        self.assertFalse(plan['training_admissible'])
        self.assertEqual(sum(s['expected_episodes'] for s in plan['study']['stages'].values()), 1888)
        for field, value in [('training_budget_usd',225), ('training_stage_seconds',14401), ('stage_seconds',3600)]:
            bad = copy.deepcopy(p)
            bad['bounds'][field] = value
            with self.assertRaises(ValueError):
                c.validate_profile(bad)
        with self.assertRaises(ValueError):
            c.study.final_release(plan,{})
        with self.assertRaises(ValueError):
            c.validate(json.loads((c.ROOT/c.study.DECLARATION).read_text()))
        journal = []
        with self.assertRaises(KeyboardInterrupt):
            matched_supervisor.run(plan,Mock(side_effect=KeyboardInterrupt),Mock(),Mock(),Mock(),lambda:False,journal.append)
        self.assertEqual(journal[-1]['gpu_seconds_upper_bound_used'],1800)
        self.assertEqual(journal[-1]['status'],'unknown')

    def test_gross_accounting(self):
        from scripts.study_billing import attribution
        submissions = {'prepare':{'app_id':'ap-prepare'},'launch':{'app_id':'ap-study'}}
        rows = [SimpleNamespace(object_id=obj,interval_start='hour',cost='0.24') for obj in
                ('ap-study','ap-prepare','vo-study','di-study','ap-unrelated')]
        report = attribution(rows,submissions,'vo-study','di-study')
        self.assertEqual(report['posted_before_credits_usd'],'0.96')
        with self.assertRaises(ValueError):
            attribution(rows+rows[:1],submissions,'vo-study','di-study')

    def test_eleven_stages_and_collected_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            physical = directory/'physical'
            physical.mkdir()
            mount = directory/'mount'
            mount.symlink_to(physical)
            p = dict(profile(), fixture=False, study_authority='all-eleven-stages-including-final')
            assets = assets_manifest()
            data = physical/p['paths']['data']
            assets['data'] = task.generate(data)
            self.assertEqual(c.study.verify_final_data(data, 'study-final'),assets['data'])
            plan = c.make_plan(p, fixture_source(), assets)
            sha = c.digest(plan)
            volume, registry = Mock(object_id='vo-synthetic'), Registry()
            with patch.object(matched_launch, 'resolve_training_config', resolved_fixture):
                bindings = reader_preflight.qualification_paths(plan,mount,volume.object_id)
                self.assertEqual(bindings,reader_preflight.qualification_paths(plan,physical,volume.object_id))
            out = physical/plan['root']
            self.assertFalse(out.exists())
            (out/'preflight').mkdir(parents=True)
            preflight = dict(status='completed',plan_sha256=sha,result=dict(status='passed',plan_sha256=sha,stage_paths=bindings))
            registry['result:preflight:'+plan['root']] = preflight
            write(out/'preflight/receipt.json',preflight)
            results = {}
            def command(argv, log, env, timeout, **kwargs):
                if Path(argv[1]).name != 'check_gpu.py':
                    with patch.object(sys,'argv',list(map(str,argv[1:]))),patch.dict(os.environ,env):
                        matched_launch.main(mount)
            async def spawn(req, deadline_epoch, image_id):
                stage = req['stage']
                call = 'fc-'+stage['wandb']['id']
                if stage['key'] == 'final-base':
                    # Exercise the real worker entry with a separate synthetic registry.
                    missing = Registry(registry)
                    missing.pop('release:'+sha)
                    with patch.object(matched_remote.runtime,'run_command') as services:
                        with self.assertRaisesRegex(ValueError,'durable final release'):
                            matched_remote.stage_worker(req,deadline_epoch,c=c,registry=missing,
                                volume=volume,call_id=call,mount=mount)
                        services.assert_not_called()
                with patch.dict(os.environ,MODAL_TASK_ID='ta-'+stage['wandb']['id']):
                    results[call] = matched_remote.stage_worker(req,deadline_epoch,c=c,registry=registry,
                        volume=volume,call_id=call,mount=mount)
                return SimpleNamespace(object_id=call)
            def options(timeout, **kwargs):
                self.assertIn(timeout,(1800,14400))
                return SimpleNamespace(spawn=SimpleNamespace(aio=spawn))
            modal = Mock()
            modal.current_function_call_id.return_value = 'fc-supervisor'
            modal.FunctionCall.from_id.side_effect = lambda call:Mock(get=Mock(return_value=results[call]))
            req = c.stage_request(plan,plan['jobs'][0],plan['jobs'][0]['stages'][0])
            template = run_episode(directory/'script',[TEXT('wrong')],request=req,simulate_live_labels=True)['records']
            with patch.object(matched_remote,'capture_build'),patch.object(matched_remote,'verify_package'), \
                 patch.object(matched_remote.shutil,'disk_usage',return_value=Mock(free=200*1024**3)), \
                 patch.object(matched_remote.runtime,'run_command',command),patch.object(matched_launch,'validate_inputs'), \
                 patch.object(matched_launch.signal,'signal'), patch.object(matched_launch.launch,'run_services',scripted_services), \
                 patch('test_recovery_learning_evidence.run_episode',trace_writer(template)):
                index = matched_remote.supervise(plan,c=c,registry=registry,volume=volume,modal=modal,
                                                 stage_worker=SimpleNamespace(with_options=options),mount=mount)
            self.assertEqual(index['status'],'completed',[(j,r.get('error')) for j,v in index['jobs'].items() for r in v['stages'] if r.get('error')])
            self.assertEqual(len(results),11)
            job = plan['jobs'][-1]
            req = c.stage_request(plan,job,job['stages'][0])
            binding = bindings[req['stage']['wandb']['id']]
            from experiment import qualification_paths
            checkpoint = qualification_paths.locations(req,binding)['training_checkpoint']
            wrong = physical/plan['jobs'][1]['stages'][0]['output']/'checkpoints/global_step_16'
            with self.assertRaisesRegex(ValueError,'Wrong bound checkpoint'):
                matched_launch.verify_checkpoint(wrong,req,binding=binding)
            (checkpoint.parent/'latest_checkpointed_iteration.txt').write_text('15')
            with self.assertRaisesRegex(ValueError,'declared final'):
                matched_launch.verify_checkpoint(checkpoint,req,binding=binding)
            (checkpoint.parent/'latest_checkpointed_iteration.txt').write_text('16')
            collected = directory/'elsewhere'
            shutil.copytree(out,collected)
            mount.unlink()
            shutil.rmtree(physical)
            read = reader_evidence.local_reader(plan,collected)
            index = read(plan['root']+'/supervisor/index.json')
            report = reader_evidence.validate_collection(plan,index,read,local=collected)
            self.assertTrue(report['evidence_complete'],report['problems'])
            self.assertEqual(report['paired_final']['effects'],{s:0 for s in ('2001','2002','2003')})
            self.assertEqual(sum(v['performance']['episodes'] for v in report['stages'].values()),1888)
            self.assertEqual(report['stages']['final-base']['secondary']['temporal_pairs'],
                             dict(both_correct=0,assessable=0,total=4))
            self.assertEqual(report['stages']['development-base']['secondary']['temporal_pairs']['total'],1)
            tables = {job['stages'][0]['policy']:index['jobs'][job['job_id']]['stages'][0]['result']['completion']['recovery_qualification']['episodes']
                      for job in plan['jobs'][7:]}
            for mutate in (lambda rows:rows.pop(),lambda rows:rows.__setitem__(0,rows[1]),
                           lambda rows:rows[0].update(reward_posted=False)):
                bad = copy.deepcopy(tables)
                mutate(bad['base'])
                with self.assertRaises(ValueError):
                    c.study.analyze_final(plan,bad)
            training = index['jobs'][plan['jobs'][1]['job_id']]['stages'][0]['result']['completion']
            training['checkpoint_sha256'] = 'wrong'
            with self.assertRaises(ValueError):
                c.study.final_release(plan,index)
