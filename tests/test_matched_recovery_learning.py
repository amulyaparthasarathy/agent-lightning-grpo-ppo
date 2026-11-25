"""Hash-guarded CPU loader and synthetic qualification collection fixtures."""
import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest

from experiment import reader_modes, recovery_admission, recovery_learning as learning, recovery_task as task
from recovery_learning_support import request


def fixture(req, iteration):
    ids = learning.stage_spec(req)['batches'][iteration-1]
    samples, episodes = [], []
    trusted = {r['case_id']: r for r in task.rows('train')}
    for i, cid in enumerate(ids):
        for replica in range(4):
            rid = f'{iteration}-{i}-{replica}'
            reward = 0 if i % 2 else replica % 2
            turns, events = [], []
            episode = {'case_id': cid, 'trace_id': 'trace-'+rid, 'attempt_id': 'attempt-'+rid, 'rollout_id': rid,
                'reward_posted': True, 'reward': reward, 'llm_calls': 6, 'completion_tokens': [], 'prompt_tokens': []}
            prompt = [1,2]
            for turn in range(6):
                if replica % 2:
                    prompt = [turn+10, 11]
                action = [turn+99, 17]
                probability = [-.5,-.75]
                turns.append(NS(prompt={'token_ids':prompt}, response={'token_ids':action, 'log_probs':probability}, image_urls=None))
                episode['completion_tokens'].append(len(action))
                episode['prompt_tokens'].append(len(prompt))
                events.append({'event_type':'model_request', 'data':{'prompt_token_ids':prompt, 'response_token_ids':action,
                    'response_log_probs':probability}, 'attempt_id':episode['attempt_id'], 'rollout_id':rid})
                prompt = prompt + action + [200, 201]  # masked observation/reminder
            events.append({'event_type':'reward','data':{'value':reward},'attempt_id':episode['attempt_id'],'rollout_id':rid})
            samples.append(NS(rollout_id=rid, data_id='g'+str(i), step=iteration, sample_idx_in_step=i,
                final_reward=reward, triplets=turns, input=trusted[cid].copy(), triplet_events=events,
                rollout_state='succeeded', error_message=None))
            episodes.append(episode)
    return samples,episodes


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires the pinned CPU environment')
class RecoveryLoaderTests(unittest.TestCase):
    def test_exact_order_from_actual_synchronous_loader_and_fit(self):
        from pinned_training_support import ROOT
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)/'data'
            task.generate(data)
            for p in data.glob('final_eval.*'):
                p.chmod(0)
            for seed in (9001,2001,2002,2003):
                subprocess.run([sys.executable, str(ROOT/'tests/recovery_loader.py'),str(data/learning.TRAIN),
                    str(data/learning.DEV), str(seed)], check=True, timeout=90,
                    env={**os.environ, 'PYTHONPATH':str(ROOT)+':'+str(ROOT/'tests'), 'OMP_NUM_THREADS':'1',
                         'HF_DATASETS_OFFLINE':'1', 'HF_DATASETS_CACHE':tmp+'/cache'})

    def test_eager_pinned_entrypoint_opens_only_frozen_inputs(self):
        from recovery_loader import eager_inputs
        from experiment import matched_launch
        import recovery_fixtures as fx
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = fx.assets_manifest()
            data = root/'data'
            assets['data'] = task.generate(data)
            for p in data.glob('final_eval.*'):
                p.chmod(0)
            for name in reader_modes.STAGES[reader_modes.RECOVERY_LEARNING]:
                req = request(name, assets=assets)
                run = root/name
                run.mkdir()
                rp, commands = matched_launch.make_plan(req,root,root,root/'model',data,Path(sys.executable),run,8000)
                learning.materialize(req,data,run,commands)
                a,b = eager_inputs(rp['train_file'],rp['eval_file'])
                for rows, binding in zip((a,b),learning.stage_spec(req)['inputs'],strict=True):
                    self.assertEqual(learning.matched.digest([r['case_id'] for r in rows]),binding['ordered_case_ids_sha256'])
                    self.assertFalse(any(r['split']=='final' for r in rows))

    def test_admission_rejects_changed_or_lost_inputs(self):
        req = request()
        rows, eps = fixture(req,1)
        recovery_admission.collection_inputs(req,1,rows,eps)
        for edit in (lambda r:setattr(r,'final_reward',None),lambda r:setattr(r,'rollout_state','failed'),
                     lambda r:r.input.update(split='development'),lambda r:r.input.update(case_id='other'),
                     lambda r:setattr(r,'sample_idx_in_step',1),lambda r:setattr(r,'step',2),
                     lambda r:r.triplet_events.pop(),lambda r:r.triplet_events[0].update(attempt_id='other'),
                     lambda r:r.triplets[0].response.update(log_probs=None),lambda r:r.triplets[0].response.update(log_probs=[]),
                     lambda r:r.triplets[0].response['log_probs'].__setitem__(0,float('nan'))):
            bad = copy.deepcopy(rows)
            edit(bad[0])
            with self.assertRaises(ValueError): recovery_admission.collection_inputs(req,1,bad,eps)
        for bad in (rows[:-1], [*rows[:-1],rows[0]]):
            with self.assertRaises(ValueError): recovery_admission.collection_inputs(req,1,bad,eps)
