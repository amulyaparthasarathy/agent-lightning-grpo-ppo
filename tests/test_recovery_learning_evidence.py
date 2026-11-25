"""Review all four qualification stages from a simulated collected directory.

Every trace comes from the production recovery harness with scripted inference.
Controller/optimizer events, checkpoint bytes, provider identities and W&B are fixtures.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiment import matched_launch, matched_supervisor, reader_contract as c, reader_evidence as reader
from experiment import recovery_calibration as cal, recovery_learning as learning, recovery_task as task
from recovery_harness_support import TEXT, run_episode
from recovery_learning_support import profile
from test_reader_training import source
import recovery_fixtures as fx


def write(path, value):
    path.write_text(json.dumps(value))


def events(request, rows):
    spec = learning.stage_spec(request)
    name = request['stage']['name']
    stream = []
    def add(event, **fields):
        stream.append(dict(event=event, pid=4242, time=len(stream)+1, **fields))
    add('weight_refresh', step=spec['initial_step'])
    roles = ['actor','critic']
    for i, batch in enumerate(spec['batches'], 1):
        sampled, original = [], []
        for e in rows:
            if e['case_id'] in batch:
                entry = dict(rollout=e['rollout_id'], case_id=e['case_id'], reward=e['reward'], group=e['case_id'])
                sampled.append(dict(entry,slot=batch.index(e['case_id'])))
                original.append(dict(entry, fragments=1, original_generated_tokens=sum(e['completion_tokens'])))
        tokens = sum(r['original_generated_tokens'] for r in original)
        lengths = [sum(e['prompt_tokens'])+sum(e['completion_tokens']) for e in rows if e['case_id'] in batch]
        rep = dict(fragment_rows=32, fragments_per_rollout={r['rollout']:1 for r in sampled}, target_tokens=tokens,
                   generated_tokens=tokens, active_lengths=lengths, max_active_position=max(lengths)-1)
        add('recovery_collection',iteration=i,policy_step=i-1,episodes=sampled,representation=rep)
        add('collection_manifest',iteration=i,trajectories=original,adapter_metrics={'training/n_truncated_sample':0})
        if name == 'collection-probe':
            add('development_collection_admitted',iteration=i,real_rollouts=32,groups=8,generated_tokens=tokens,
                fragment_rows=32,constant_groups=8)
        else:
            add('recovery_old_probabilities',iteration=i,target_tokens=tokens,mean_delta=0,max_abs_delta=0)
            add('critic_preupdate',iteration=i,prediction={})
            add('matched_advantages',iteration=i,real_rollouts=32,generated_tokens=tokens,constant_groups=8)
            for role in reversed(roles):
                add('optimizer_step',role=role,count=i)
                add('gradient_step',role=role,optimizer_called=True,preclip_finite=True,preclip_norm=0)
            add('weight_refresh',step=i)
    if spec['updates']:
        add('native_checkpoint_saved',step=2)
    add('trainer_returned',global_step=1 if name=='collection-probe' else spec['initial_step']+spec['updates'],
        validation_only=spec['route']=='val',collection_only=name=='collection-probe',optimizer_updates=spec['updates'])
    add('wandb_finished',id=request['stage']['wandb']['id'])
    return stream


class RecoveryCollectionTests(unittest.TestCase):
    def test_complete_qualification_then_tampered_or_missing_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = fx.assets_manifest()
            assets['data'] = task.generate(root/'data')
            plan = c.make_plan(dict(profile('ppo-mc'), fixture=False), source(), assets)
            job = plan['jobs'][0]
            receipts = []
            index = {'plan':plan,'status':'completed','study_key':matched_supervisor.key(plan),
                     'owner':{'call_id':'fc-supervisor','plan_sha256':c.digest(plan)},
                     'jobs':{job['job_id']:{'status':'completed','stages':receipts}}}
            for stage in job['stages']:
                req = c.stage_request(plan,job,stage)
                spec = learning.stage_spec(req)
                run = root/job['algorithm']/stage['name']
                run.mkdir(parents=True)
                learning.materialize(req,root/'data',run,{'gateway':list(cal.GATEWAY_TEMPERATURES)})
                by_id = {r['case_id']:r for r in task.rows(spec['split'])}
                for cid in spec['case_ids']:
                    for sample in range(spec['rollouts_per_question']):
                        rid = f"{stage['name']}-{cid.split('/')[-1]}-{sample}"
                        r = run_episode(run,[TEXT('wrong')],row=by_id[cid],request=req,route=spec['route'],rollout=rid,
                                        simulate_live_labels=True)
                        self.assertIsNone(r['error'])
                rows = cal.episode_table(run,req)
                audit = events(req,rows)
                (run/'audit').mkdir()
                (run/'audit/events-4242.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in audit))
                if stage['name'] == 'training':
                    path = run/'checkpoints/global_step_2'
                    for role in ('actor','critic'):
                        (path/role).mkdir(parents=True)
                        for kind in ('model','optim','extra_state'):
                            (path/role/(kind+'_world_size_1_rank_0.pt')).write_bytes(b'synthetic')
                    (path/'data.pt').write_bytes(b'loader fixture')
                    (path.parent/'latest_checkpointed_iteration.txt').write_text('2')
                    write(run/'resolved_config.json',{})
                    (run/'metrics.jsonl').write_text(''.join(json.dumps({'step':i,'metrics':{'actor/pg_loss':0,
                                                      'critic/vf_loss':0}})+'\n' for i in (1,2)))
                call = 'fc-'+stage['wandb']['id']
                container = 'ta-'+stage['wandb']['id']
                with patch.dict(os.environ,AGL_READER_CALL_ID=call,MODAL_TASK_ID=container):
                    before = None
                    if stage['name'] == 'checkpoint-eval':
                        before = matched_launch.verify_checkpoint(run.parent/'training/checkpoints/global_step_2',req)
                    completion = matched_launch.complete_stage(run,req,root/'data')
                    if before is not None:
                        completion['checkpoint_before_load_sha256'] = before
                    proof = reader.complete(run,req,completion)
                receipt = {'status':'completed','output':stage['output'],'call_id':call,'container_id':container,
                           'completion':proof}
                self.assertTrue(matched_supervisor.recovery_stage_admitted(plan,job,len(receipts),receipt,c.digest(plan)))
                receipts.append({'stage':stage['name'],'status':'completed','result':receipt,
                                 'call_id':call,'wandb':stage['wandb'],'output':stage['output']})
                write(run/'receipt.json',receipt)
                write(run/'completion.json',proof)
                write(run/'request.json',req)
                write(run/'observability.json',{'experiment_id':job['job_id'],'wandb':stage['wandb'],
                    'wandb_run_id':stage['wandb']['id'],'request_sha256':hashlib.sha256(json.dumps(req,sort_keys=True).encode()).hexdigest()})
            (root/'supervisor').mkdir()
            write(root/'supervisor/plan.json',plan)
            write(root/'supervisor/index.json',index)
            read = reader.local_reader(plan,root)
            def review():
                return reader.validate_collection(plan,index,read,root)
            admitted = review()
            self.assertTrue(admitted['evidence_complete'],admitted['problems'])
            stages = admitted['stages']
            self.assertEqual(stages[job['algorithm']+'/training']['optimization']['informative_groups'],[0,0])
            probe = root/job['algorithm']/'collection-probe'
            audit = probe/'audit/events-4242.jsonl'
            raw = audit.read_text()
            for event in ({'event':'gradient_step','role':'actor'},{'event':'recovery_old_probabilities','iteration':1}):
                audit.write_text(raw+json.dumps(dict(event,time=999))+'\n')
                self.assertFalse(review()['evidence_complete'])
            audit.write_text(raw)
            trace = next((probe/'traces').glob('trajectory-*.jsonl'))
            content = trace.read_text()
            trace.write_text(content.replace('collection-probe','other-stage'))
            self.assertFalse(review()['evidence_complete'])
            trace.unlink()
            bad = review()
            self.assertEqual(bad['stages'][job['algorithm']+'/collection-probe']['confirmed_slots'],31)
            self.assertFalse(bad['evidence_complete'])
