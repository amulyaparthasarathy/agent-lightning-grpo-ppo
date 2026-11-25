"""Review all four qualification stages from a simulated collected directory.

Every trace comes from the production recovery harness with scripted inference.
Controller/optimizer events, checkpoint bytes, provider identities and W&B are fixtures.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import tempfile
import unittest
from unittest.mock import Mock, patch

from experiment import matched_launch, matched_remote, matched_supervisor, reader_preflight, request_transport, reader_contract as c, reader_evidence as reader
from experiment import recovery_calibration as cal, recovery_learning as learning, recovery_task as task
from recovery_harness_support import TEXT, run_episode
from recovery_learning_support import profile
from test_reader_training import Registry, source
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
    roles = ['actor','critic'] if request['job']['algorithm'] == c.ALGORITHMS[1] else ['actor']
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
        layout = {'input_ids':5120,'attention_mask':5120,'position_ids':5120,'prompts':3072,
                  'responses':2048,'response_mask':2048,'token_level_scores':2048,'rollout_log_probs':2048}
        rep['masked_response_context_tokens'] = 0
        rep['tensors'] = {}
        for key,width in layout.items():
            fp = key in ('token_level_scores','rollout_log_probs')
            rep['tensors'][key] = dict(shape=[32,width],dtype='torch.float32' if fp else 'torch.int64',
                                       bytes=32*width*(4 if fp else 8))
        rep['tensors']['is_drop_mask'] = dict(shape=[32],dtype='torch.bool',bytes=32)
        add('recovery_collection',iteration=i,policy_step=i-1,episodes=sampled,representation=rep)
        add('collection_manifest',iteration=i,trajectories=original,adapter_metrics={'training/n_truncated_sample':0})
        if name == 'collection-probe':
            add('development_collection_admitted',iteration=i,real_rollouts=32,groups=8,generated_tokens=tokens,
                fragment_rows=32,constant_groups=8)
        else:
            add('recovery_old_probabilities',iteration=i,target_tokens=tokens,mean_delta=0,max_abs_delta=0)
            from experiment.ppo_math import prediction_metrics
            if 'critic' in roles:
                add('critic_preupdate',iteration=i,prediction=prediction_metrics([0.]*tokens,[0.]*tokens))
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


def scripted_services(commands, rp, run, repo, bundle, remaining):
    req = rp['request']
    stage, job = req['stage'], req['job']
    spec = learning.stage_spec(req)
    cfg = {'actor_rollout_ref': {'model': {'path': rp['model']}}, 'agentlightning': {'async_rollout': {'enabled': False}}}
    for override in rp['overrides']:
        key, value = override.split('=', 1)
        node = cfg
        parts = key.split('.')
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = json.loads(value)
    env = cfg['ray_kwargs']['ray_init']['runtime_env']['env_vars']
    assert request_transport.load(env) == req
    matched_launch.validate_resolved(cfg, rp)
    write(run/'resolved_config.json', cfg)
    by_id = {r['case_id']:r for r in task.rows(spec['split'])}
    for cid in spec['case_ids']:
        for sample in range(spec['rollouts_per_question']):
            rid = f"{stage['name']}-{cid.split('/')[-1]}-{sample}"
            r = run_episode(run,[TEXT('wrong')],row=by_id[cid],request=req,route=spec['route'],rollout=rid,
                            simulate_live_labels=True)
            assert r['error'] is None, r['error']
    rows = cal.episode_table(run,req)
    audit = events(req,rows)
    (run/'audit').mkdir()
    (run/'audit/events-4242.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in audit))
    if stage['name'] == 'training':
        path = Path(cfg['trainer']['default_local_dir'])/'global_step_2'
        roles = ['actor','critic'] if job['algorithm'] == c.ALGORITHMS[1] else ['actor']
        for role in roles:
            (path/role).mkdir(parents=True)
            for kind in ('model','optim','extra_state'):
                (path/role/(kind+'_world_size_1_rank_0.pt')).write_bytes(b'synthetic')
        (path/'data.pt').write_bytes(b'loader fixture')
        (path.parent/'latest_checkpointed_iteration.txt').write_text('2')
        metrics = {'actor/pg_loss':0}
        if 'critic' in roles:
            metrics['critic/vf_loss'] = 0
        (run/'metrics.jsonl').write_text(''.join(json.dumps({'step':i,'metrics':metrics})+'\n' for i in (1,2)))
    write(run/'observability.json',{'experiment_id':job['job_id'],'wandb':stage['wandb'],
          'wandb_run_id':stage['wandb']['id'],'request_sha256':hashlib.sha256(json.dumps(req,sort_keys=True).encode()).hexdigest()})
    return 0


class RecoveryCollectionTests(unittest.TestCase):
    def exercise_mount(self, selection, alias):
        with tempfile.TemporaryDirectory() as tmp:
            physical = Path(tmp)/'provider'
            physical.mkdir()
            mount = physical
            if alias:
                mount = Path(tmp)/'logical'
                mount.symlink_to(physical, target_is_directory=True)
            p = dict(profile(selection), fixture=False)
            assets = fx.assets_manifest()
            assets['data'] = task.generate(physical/p['paths']['data'])
            plan = c.make_plan(p, source(), assets)
            job = plan['jobs'][0]
            volume = Mock(object_id='vo-synthetic')
            registry = Registry()
            preflight = {'status':'completed','plan_sha256':c.digest(plan),'result':{
                'status':'passed','plan_sha256':c.digest(plan),
                'stage_paths':reader_preflight.qualification_paths(plan,mount,volume.object_id)}}
            registry['result:preflight:'+plan['root']] = preflight
            output = physical/plan['root']
            self.assertFalse(output.exists())
            (output/'preflight').mkdir(parents=True)
            write(output/'preflight/receipt.json',preflight)
            receipts = []
            index = {'plan':plan,'status':'completed','study_key':matched_supervisor.key(plan),
                     'worker_seconds_reserved':14400, 'supervisor_reservation_usd':15000*10/3600,
                     'owner':{'call_id':'fc-supervisor','plan_sha256':c.digest(plan),
                              'preflight_sha256':c.digest(preflight),'volume_id':volume.object_id},
                     'jobs':{job['job_id']:{'status':'completed','stages':receipts}}}

            def command(argv, log, env, timeout, **kwargs):
                if Path(argv[1]).name == 'check_gpu.py':
                    return
                self.assertEqual(Path(argv[1]).name, 'matched_launch.py')
                with patch.object(sys, 'argv', list(map(str,argv[1:]))), patch.dict(os.environ,env):
                    matched_launch.main(mount)
            for stage in job['stages']:
                call = 'fc-'+stage['wandb']['id']
                container = 'ta-'+stage['wandb']['id']
                req = c.stage_request(plan,job,stage)
                with patch.dict(os.environ,MODAL_TASK_ID=container), patch.object(matched_remote,'capture_build'), \
                     patch.object(matched_remote.shutil,'disk_usage',return_value=Mock(free=200*1024**3)), \
                     patch.object(matched_remote,'verify_package'), patch.object(matched_remote.runtime,'run_command',command), \
                     patch.object(matched_launch,'validate_inputs'), patch.object(matched_launch.signal,'signal'), \
                     patch.object(matched_launch.launch,'run_services',scripted_services):
                    receipt = matched_remote.stage_worker(req,time.time()+3500,c=c,registry=registry,
                                     volume=volume,call_id=call,mount=mount)
                run = physical/stage['output']
                self.assertEqual(json.loads((run/'receipt.json').read_text()),receipt)
                self.assertTrue(matched_supervisor.recovery_stage_admitted(plan,job,len(receipts),receipt,c.digest(plan)))
                forged = copy.deepcopy(receipt)
                forged_report = forged['completion']['recovery_qualification']
                forged_report['episodes'][1] = forged_report['episodes'][0]
                self.assertFalse(matched_supervisor.recovery_stage_admitted(plan,job,len(receipts),forged,c.digest(plan)))
                receipts.append({'stage':stage['name'],'status':'completed','result':receipt,'reservation':{'seconds':3600,'usd':10},
                                 'call_id':call,'wandb':stage['wandb'],'output':stage['output']})
            (output/'supervisor').mkdir()
            write(output/'supervisor/plan.json',plan)
            write(output/'supervisor/index.json',index)
            root = Path(tmp)/'independent-review'
            # Collection transports evidence only. Native checkpoint payloads stay on the volume.
            shutil.copytree(output,root,ignore=shutil.ignore_patterns('*.pt','*.parquet'))
            shutil.rmtree(physical)
            read = reader.local_reader(plan,root)
            def review():
                return reader.validate_collection(plan,index,read,root)
            admitted = review()
            self.assertTrue(admitted['evidence_complete'],admitted['problems'])
            stages = admitted['stages']
            self.assertEqual(stages[job['algorithm']+'/training']['optimization']['informative_groups'],[0,0])
            # Corrupt one collected artifact at a time, keeping the independent anchors intact.
            training = root/job['algorithm']/'training'
            for path, mutate in (
                (training/'path-binding.json', lambda v: v.update(volume_id='vo-other')),
                (training/'resolved_config.json', lambda v: v['trainer'].update(test_freq=99)),
                (training/'checkpoints/global_step_2/matched_identity.json',
                 lambda v: v['roles']['actor']['optim_world_size_1_rank_0.pt'].update(sha256='0'*64)),
                (root/'preflight/receipt.json', lambda v: v.update(plan_sha256='0'*64)),
            ):
                with self.subTest(artifact=path.relative_to(root)):
                    original = path.read_bytes()
                    value = json.loads(original)
                    mutate(value)
                    write(path, value)
                    self.assertFalse(review()['evidence_complete'])
                    path.write_bytes(original)
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

    def test_grpo_aliased_volume(self):
        self.exercise_mount('grpo', True)

    def test_ppo_plain_volume(self):
        self.exercise_mount('ppo-mc', False)

    def test_grpo_plain_volume(self):
        self.exercise_mount('grpo', False)

    def test_ppo_aliased_volume(self):
        self.exercise_mount('ppo-mc', True)
