"""CPU tests of pinned training bodies, including the real optimizer path.

Run explicitly with the isolated requirements in scripts/test_matched_training.sh.
No model, vLLM server, Ray cluster or Modal API is constructed.
"""
from contextlib import nullcontext
import copy
import importlib.util
import json
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS, ModuleType
import unittest
from unittest.mock import patch

from experiment import matched_runtime as runtime, matched_contract as c, matched_launch
try:
    from pinned_training_support import load, Proto, SOURCES
except ImportError:
    from tests.pinned_training_support import load, Proto, SOURCES

HAS_TORCH = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(HAS_TORCH and (SOURCES/'verl/verl').is_dir(), 'requires explicit pinned CPU test environment')
class PinnedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import numpy as np
        import tensordict
        from tensordict import TensorDict, NonTensorData, NonTensorStack
        from packaging.version import parse
        cls.torch, cls.np, cls.TD = torch, np, TensorDict
        cls.tu = load('verl', 'utils/tensordict_utils.py', dict(torch=torch, np=np, tensordict=tensordict,
            TensorDict=TensorDict, NonTensorData=NonTensorData, NonTensorStack=NonTensorStack,
            parse_version=parse, logging=logging))
        ns = dict(torch=torch)
        cls.functional = load('verl', 'utils/torch_functional.py', ns,
                               ['masked_sum', 'masked_mean', 'masked_var', 'masked_whiten'])
        cls.policy = staticmethod(load('agentlightning', 'verl/per_rollout_loss.py',
            dict(torch=torch, verl_F=cls.functional), ['compute_policy_loss_per_rollout_mean']).compute_policy_loss_per_rollout_mean)
        cls.padding = load('verl', 'workers/utils/padding.py',
                           dict(torch=torch, F=torch.nn.functional, tu=cls.tu), ['no_padding_2_padding'])
        from enum import Enum
        cls.metrics = load('verl', 'utils/metric/utils.py', dict(torch=torch, np=np, Enum=Enum))
        cls.native_ppo = load('verl','workers/utils/losses.py', dict(
            torch=torch, tu=cls.tu, Metric=cls.metrics.Metric, AggregationType=cls.metrics.AggregationType,
            no_padding_2_padding=cls.padding.no_padding_2_padding,
            get_policy_loss_fn=lambda name:cls.policy), ['ppo_loss'])
        runtime._NATIVE_PPO_LOSS = cls.native_ppo.ppo_loss
        from torch.distributed.fsdp import FullyShardedDataParallel, FSDPModule
        from torch.distributed.tensor import DTensor
        def append(target, values):
            for k, v in values.items():
                target.setdefault(k, []).append(v)
        cls.engine_utils = load('verl', 'workers/engine/utils.py',
            dict(torch=torch, tu=cls.tu, DatasetPadMode=NS(NO_PADDING='no_padding'), append_to_dict=append),
            ['prepare_micro_batches', 'postprocess_batch_func'])
        cls.engine = load('verl', 'workers/engine/fsdp/transformer_impl.py',
            dict(torch=torch, tu=cls.tu, nullcontext=nullcontext, get_device_id=lambda:'cpu',
                 prepare_micro_batches=cls.engine_utils.prepare_micro_batches,
                 postprocess_batch_func=cls.engine_utils.postprocess_batch_func,
                 FSDP=FullyShardedDataParallel, FSDPModule=FSDPModule, DTensor=DTensor),
            ['forward_backward_batch', 'optimizer_zero_grad', 'optimizer_step'], 'FSDPEngine')
        cls.base = load('verl', 'workers/engine/base.py', dict(maybe_fix_3d_position_ids=cls.tu.maybe_fix_3d_position_ids),
                        ['train_batch'], 'BaseEngine')
        # Real distributed collectives, CPU world size one: the supported per-job topology.
        cls.dist_tmp = tempfile.TemporaryDirectory()
        torch.distributed.init_process_group('gloo', init_method='file://'+cls.dist_tmp.name+'/dist', rank=0, world_size=1)

    @classmethod
    def tearDownClass(cls):
        cls.torch.distributed.destroy_process_group()
        cls.dist_tmp.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict('os.environ', AGL_PPO_AUDIT_DIR=self.tmp.name)
        self.env.start()
        runtime.audit._STATE.clear()
        runtime.audit._STATE['step'] = 1

    def tearDown(self):
        self.env.stop(); self.tmp.cleanup()

    def adapter(self, collection_only=False, recovery_request=None):
        import io, zipfile, base64
        module = load('agentlightning', 'verl/rollout_adapter.py', dict(torch=self.torch, np=self.np,
            TensorDict=self.TD, DataProto=Proto, json=json, io=io, zipfile=zipfile, base64=base64))
        runtime.instrument_adapter(module, collection_only=collection_only, recovery_request=recovery_request)
        return module.RolloutAdapter(max_prompt_length=3072 if recovery_request else 24, max_response_length=2048 if recovery_request else 32,
                                     device=self.torch.device('cpu'), pad_token_id=0, trace_aggregator_level='trajectory')

    def rollouts(self, fragmented=False, constant=False):
        rows = []
        # Two turns of different lengths, separated by tool-output observations.
        for i, lengths in enumerate(((1,3), (1,1), (3,2), (2,1))):
            first = list(range(10,10+lengths[0])); second = list(range(30,30+lengths[1]))
            prompt2 = [1]+first+[99,98] if not fragmented else [2,3,4]
            turns = [NS(prompt={'token_ids':p}, response={'token_ids':r,'log_probs':[-.4]*len(r)}, image_urls=None)
                     for p,r in (([1],first), (prompt2,second))]
            rows.append(NS(rollout_id=f'r{i}', data_id='q', step=1, sample_idx_in_step=i,
                           final_reward=0 if constant else i%2, triplets=turns, input={'case_id':'case-q'}))
        return rows

    def batch(self, fragmented=False, constant=False, estimator='gae'):
        batch, _ = self.adapter().get_train_data_batch(self.rollouts(fragmented, constant), global_steps=1)
        batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
        batch.batch['values'] = self.torch.ones_like(batch.batch['response_mask'])*.2
        batch.batch['old_log_probs'] = self.torch.zeros_like(batch.batch['values'])
        runtime.advantages(batch, adv_estimator=estimator)
        return batch

    def test_real_adapter_constant_groups_and_fragment_counts(self):
        for fragmented in (False,True):
            batch = self.batch(fragmented, True, 'grpo')
            self.assertEqual(len(batch), 8 if fragmented else 4)
            self.assertEqual(batch.batch['response_mask'].sum().item(),14)
            self.assertEqual(batch.batch['advantages'].abs().sum().item(),0)
            self.assertAlmostEqual(batch.batch['matched_weights'].sum().item(),1)
            self.assertEqual(set(batch.non_tensor_batch['ppo_case_id']), {'case-q'})
            self.assertFalse(batch.batch['is_drop_mask'].any())

    def test_collection_only_preserves_tokens_masks_groups_and_stops_before_updates(self):
        rows = []
        for group in range(8):
            for row in self.rollouts(fragmented=True, constant=True):
                row.rollout_id += f'-{group}'
                row.data_id = f'q-{group}'
                rows.append(row)
        with self.assertRaises(runtime.CollectionOnlyComplete):
            self.adapter(collection_only=True).get_train_data_batch(rows, global_steps=1)
        events = [json.loads(line) for p in Path(self.tmp.name).glob('*.jsonl') for line in p.read_text().splitlines()]
        admitted = [e for e in events if e['event'] == 'development_collection_admitted']
        self.assertEqual(len(admitted), 1)
        self.assertEqual(admitted[0]['generated_tokens'], 14*8)
        self.assertEqual(admitted[0]['fragment_rows'], 64)
        self.assertEqual(admitted[0]['constant_groups'], 8)
        self.assertFalse(any(e['event'] in ('matched_advantages', 'optimizer_step', 'gradient_step') for e in events))
        for kind in ('missing-reward', 'incomplete-group', 'truncated'):
            bad = copy.deepcopy(rows); adapter = self.adapter(collection_only=True)
            if kind == 'missing-reward': bad[0].final_reward = None
            if kind == 'incomplete-group': bad.pop()
            if kind == 'truncated': adapter.max_response_length = 1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                adapter.get_train_data_batch(bad, global_steps=1)

    def test_incomplete_truncated_duplicate_and_missing_reward_fail_closed(self):
        for kind in ('incomplete','truncated','duplicate','reward','no-trace'):
            rows = self.rollouts()
            adapter = self.adapter()
            if kind=='incomplete': rows.pop()
            if kind=='truncated': adapter.max_response_length=2
            if kind=='duplicate': rows[1].rollout_id=rows[0].rollout_id
            if kind=='reward': rows[0].final_reward=None
            if kind=='no-trace': rows[0].triplets=[]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                adapter.get_train_data_batch(rows, global_steps=1)

    def test_actual_pinned_gae_mc_whitening_matches_and_targets_unchanged(self):
        core = load('verl', 'trainer/ppo/core_algos.py', dict(torch=self.torch, verl_F=self.functional),
                    ['compute_gae_advantage_return'])
        for fragmented in (False,True):
            batch = self.batch(fragmented)
            b = batch.batch; mask = b['response_mask'].bool()
            adv, ret = core.compute_gae_advantage_return(b['token_level_rewards'], b['values'], mask.int(), 1., 1.)
            self.torch.testing.assert_close(adv[mask], b['advantages'][mask])
            self.torch.testing.assert_close(ret[mask], b['returns'][mask])
            saved = b['returns'].clone()
            runtime.normalize(b['advantages'], mask, batch.non_tensor_batch['rollout_id_list'], num_trained_rows=len(batch))
            self.torch.testing.assert_close(saved,b['returns'])

    def update(self, fragmented, role, micro=1, dp_scale=1, admitted_batch=None, pack=None):
        torch, tu = self.torch, self.tu
        batch = self.batch(fragmented, estimator='gae') if admitted_batch is None else admitted_batch
        td = batch.batch
        td['advantages'] = runtime.normalize(td['advantages'],td['response_mask'],
            batch.non_tensor_batch['rollout_id_list'], num_trained_rows=len(batch))
        td['loss_mask'] = td['response_mask']
        # Pinned outer minibatch worker reads nested input IDs for token accounting.
        fullmask = td['attention_mask'].bool()
        if pack is None:
            td['input_ids'] = torch.nested.as_nested_tensor([r[m] for r,m in zip(td['input_ids'],fullmask)], layout=torch.jagged)
        else:
            td = pack(td.clone())
        tu.assign_non_tensor(td, mini_batch_size=len(td), epochs=1, seed=43,
                            dataloader_kwargs={'shuffle':False}, use_dynamic_bsz=False,
                            micro_batch_size_per_gpu=micro, global_batch_size=len(td), pad_mode='no_padding')
        module = torch.nn.Linear(1,1,bias=False)
        with torch.no_grad(): module.weight.fill_(.4)
        optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3, weight_decay=0.)
        calls = []
        optimizer.register_step_post_hook(lambda *args: calls.append(1))
        cfg = NS(cliprange_value=.5)
        from omegaconf import OmegaConf
        acfg = OmegaConf.create({'clip_ratio':.2,'clip_ratio_low':.2,'clip_ratio_high':.3,'clip_ratio_c':3.,
                                 'global_batch_info':{'dp_size':dp_scale}, 'loss_scale_factor':None, 'loss_agg_mode':'token-mean',
                                 'policy_loss':{'loss_mode':'per_rollout_mean'}, 'entropy_coeff':0., 'use_kl_loss':False})
        native_policy = self.policy
        padding = ModuleType('verl.workers.utils.padding'); padding.no_padding_2_padding=self.padding.no_padding_2_padding
        outer = self
        class Engine:
            ulysses_sequence_parallel_size=1
            _qat_enabled=False
            optimizer_config=NS(clip_grad=.1)
            def get_data_parallel_group(self): return None
            def get_data_parallel_rank(self): return 0
            def get_data_parallel_size(self): return 1
            def is_mp_src_rank_with_outputs(self): return True
            def train_mode(self,**kwargs): return nullcontext()
            def forward_step(self, data, loss_function, forward_only):
                # Small differentiable CPU model; exact packed-output slicing goes through pinned padding.
                n=int(data['attention_mask'].sum())
                logits=module(torch.ones(n,1)).squeeze(-1)
                if role=='actor':
                    pred=outer.padding.no_padding_2_padding(logits,data)
                    with patch.dict('sys.modules',{'verl.workers.utils.padding':padding, 'verl.utils.metric':outer.metrics}):
                        loss,metrics=runtime.policy_loss(acfg,{'log_probs':logits},data)
                else:
                    tu.assign_non_tensor(data,dp_size=dp_scale)
                    with patch.dict('sys.modules',{'verl.workers.utils.padding':padding, 'verl.utils.metric':outer.metrics}):
                        loss,metrics=runtime.value_loss(cfg,{'values':logits},data)
                return loss,{'loss':loss.detach().item(),'metrics':metrics}
        engine=Engine();engine.module=module;engine.optimizer=optimizer
        for name in ('optimizer_step','optimizer_zero_grad','forward_backward_batch'):
            setattr(engine,name,getattr(self.engine,name).__get__(engine))
        engine.train_batch=self.base.train_batch.__get__(engine)
        # Execute the actual outer worker loop; only telemetry transport is a seam.
        from itertools import chain
        def append(target, values):
            for k,v in values.items(): target.setdefault(k,[]).append(v)
        Metric = self.metrics.Metric
        worker_methods=load('verl','workers/engine_workers.py',dict(
            torch=torch,tu=tu,maybe_fix_3d_position_ids=tu.maybe_fix_3d_position_ids,
            NonTensorData=__import__('tensordict').NonTensorData, Timer=lambda **kw:nullcontext(),
            chain=chain, Metric=Metric, append_to_dict=append),['train_mini_batch'],'TrainingWorker')
        def sum_metrics(values):
            metric=Metric(aggregation=values[0].aggregation)
            for value in values: metric.append(value)
            return metric
        worker=NS(engine=engine)
        worker.train_batch=lambda data:tu.get_tensordict({}, {'metrics':{
            k:[sum_metrics(v)] if isinstance(v,list) and isinstance(v[0],Metric) else [v] if isinstance(v,list) else v for k,v in engine.train_batch(data,None)['metrics'].items()}})
        output=worker_methods.train_mini_batch(worker,td)
        return module.weight.detach().clone(),module.weight.grad.clone(),calls,output

    def test_qualification_adapter_packing_recomputed_probabilities_and_updates(self):
        import hashlib
        from experiment import recovery_admission as admit
        from recovery_learning_support import request
        from test_matched_recovery_learning import fixture
        from pinned_training_support import ROOT
        from omegaconf import OmegaConf
        torch = self.torch
        torch.set_num_threads(1)
        padding_source = ROOT/'.local/bert_padding.py'
        self.assertEqual(hashlib.sha256(padding_source.read_bytes()).hexdigest(),
                         '805d449ac749f87a50f3a3114385710f0f926ff45521275000b74d9e80721e5c')
        import runpy
        flash = runpy.run_path(str(padding_source))
        pack = load('verl', 'workers/utils/padding.py', dict(torch=torch, tu=self.tu,
            unpad_input=flash['unpad_input']), ['left_right_2_no_padding']).left_right_2_no_padding
        old = load('verl', 'trainer/ppo/ray_trainer.py', dict(tu=self.tu, left_right_2_no_padding=pack,
            no_padding_2_padding=self.padding.no_padding_2_padding, DataProto=NS(from_tensordict=lambda x: Proto(x))),
            ['_compute_old_log_prob'], 'RayPPOTrainer')._compute_old_log_prob
        cfg = OmegaConf.create({'actor_rollout_ref':{'actor':{}}})
        def compute(td):
            lengths = td['input_ids'].offsets().diff().tolist()
            self.assertEqual(sorted(lengths), [4]*96+[24]*16)
            flat = td['input_ids'].values().float()*0-.25
            return self.tu.get_tensordict({'log_probs':flat,'entropy':flat*0}, {'metrics':{'mfu':0}})
        for selection in ('grpo','ppo-mc'):
            previous = []
            for i in (1,2):
                req = request(selection=selection)
                rows, eps = fixture(req,i)
                previous += eps
                with patch.dict('os.environ', AGL_PPO_RUN_DIR=self.tmp.name), patch.object(admit, 'reviewed_episodes', return_value=previous):
                    batch, metrics = self.adapter(recovery_request=req).get_train_data_batch(rows, global_steps=i)
                rep = admit.representation(batch,rows,metrics)
                self.assertEqual((rep['fragment_rows'],rep['target_tokens'],rep['masked_response_context_tokens']), (112,384,160))
                if i == 1 and selection == 'grpo':
                    print('Recovery CPU representation:', json.dumps({k:v for k,v in rep.items() if k not in
                          ('active_lengths','fragments_per_rollout')}, sort_keys=True))
                    for key in ('response_mask','rollout_log_probs','input_ids','position_ids'):
                        changed = batch.batch[key].clone()
                        batch.batch[key][0,-1 if key=='input_ids' else 0] += 1
                        if key == 'position_ids':
                            batch.batch[key][0,3071] += 1
                        with self.assertRaises(ValueError): admit.representation(batch,rows,metrics)
                        batch.batch[key] = changed
                result, _ = old(NS(config=cfg,actor_rollout_wg=NS(compute_log_prob=compute)),batch)
                b = batch.batch
                b['old_log_probs'] = result.batch['old_log_probs']
                b['values'] = torch.zeros_like(b['old_log_probs'])+.2
                b['token_level_rewards'] = b['token_level_scores']
                diff = admit.old_probabilities(batch,selection=='ppo-mc')
                self.assertEqual(diff['max_abs_delta'],.5)
                b['old_log_probs'][0,0] = float('inf')
                with self.assertRaises(ValueError): admit.old_probabilities(batch,False)
                b['old_log_probs'][0,0] = -.25
                runtime.advantages(batch, adv_estimator='grpo' if selection=='grpo' else 'gae')
                for role in (('actor',) if selection=='grpo' else ('actor','critic')):
                    value, grad, calls, _ = self.update(False,role,admitted_batch=Proto(b.clone(),batch.non_tensor_batch),pack=pack)
                    self.assertEqual(calls,[1])
                    self.assertTrue(torch.isfinite(grad).all() and torch.isfinite(value).all())
            probe = request('collection-probe', selection)
            rows,eps = fixture(probe,1)
            with patch.dict('os.environ', AGL_PPO_RUN_DIR=self.tmp.name), patch.object(admit, 'reviewed_episodes', return_value=eps):
                with self.assertRaises(runtime.CollectionOnlyComplete):
                    self.adapter(collection_only=True,recovery_request=probe).get_train_data_batch(rows, global_steps=1)

    def test_recovery_context_boundary_rejects_without_truncating(self):
        from experiment import recovery_admission as admit
        from recovery_learning_support import request
        from test_matched_recovery_learning import fixture
        req = request()
        rows, eps = fixture(req,1)
        r = rows[0]
        t = r.triplets[0]
        t.prompt['token_ids'] = [1]*3072
        t.response.update(token_ids=[7]*1024, log_probs=[-.5]*1024)
        r.triplets = [t]
        r.triplet_events = [r.triplet_events[0], r.triplet_events[-1]]
        r.triplet_events[0]['data'].update(prompt_token_ids=t.prompt['token_ids'],response_token_ids=t.response['token_ids'],
                                          response_log_probs=t.response['log_probs'])
        eps[0].update(llm_calls=1, prompt_tokens=[3072],completion_tokens=[1024])
        with patch.dict('os.environ',AGL_PPO_RUN_DIR=self.tmp.name), patch.object(admit,'reviewed_episodes',return_value=eps):
            adapter = self.adapter(recovery_request=req)
            batch, metrics = adapter.get_train_data_batch(rows, global_steps=1)
            self.assertEqual(admit.representation(batch,rows,metrics)['max_active_position'],4095)
            t.prompt['token_ids'].append(1)
            eps[0]['prompt_tokens'][0] += 1
            with self.assertRaises(ValueError): adapter.get_train_data_batch(rows,global_steps=1)

    def test_full_pinned_microbatch_backward_clip_adamw_fragment_invariance(self):
        for role in ('actor','critic'):
            merged=self.update(False,role)
            for fragmented,micro in ((True,1),(False,2),(True,2)):
                with self.subTest(role=role,fragmented=fragmented,micro=micro):
                    other=self.update(fragmented,role,micro)
                    self.torch.testing.assert_close(merged[0],other[0])
                    self.torch.testing.assert_close(merged[1],other[1],atol=1e-6,rtol=1e-5)
                    self.assertEqual(other[2],[1])
                    for metric in ('actor/ppo_kl', 'actor/pg_clipfrac', 'critic/vf_loss'):
                        first=self.tu.get(merged[3], 'metrics')
                        second=self.tu.get(other[3], 'metrics')
                        if metric in first:
                            self.assertAlmostEqual(first[metric][0],second[metric][0],places=6)
            self.assertEqual(merged[2],[1])

    def test_dp_scaling_is_applied_once_before_average(self):
        # The contract launches world-size one. Simulated loss scaling audits the
        # pinned DP multiplier separately; no multi-GPU qualification is claimed.
        cfg=__import__('omegaconf').OmegaConf.create({'clip_ratio':.2,'clip_ratio_low':.2,'clip_ratio_high':.3,
                'clip_ratio_c':3.,'global_batch_info':{'dp_size':2}})
        torch=self.torch
        old=torch.zeros(1,3);new=torch.zeros(1,3,requires_grad=True)
        loss,_=self.policy(old,new,torch.tensor([[.2,.3,99.]]),torch.tensor([[1,1,0]]),config=cfg)
        loss.backward()
        torch.testing.assert_close(new.grad/2,torch.tensor([[-.2,-.3,0.]]))

    def test_runtime_update_dispatch_uses_actual_row_count(self):
        from omegaconf import OmegaConf
        cfg=OmegaConf.create({'actor_rollout_ref':{'rollout':{'n':4,'temperature':1,'multi_turn':{'enable':True}},
             'actor':{'ppo_mini_batch_size':8,'ppo_epochs':1,'data_loader_seed':12,'shuffle':False,
                      'calculate_entropy':False,'entropy_coeff':0}},
             'critic':{'ppo_mini_batch_size':8,'ppo_epochs':1,'data_loader_seed':12,'shuffle':False}})
        ns=dict(tu=self.tu,DataProto=Proto,left_right_2_no_padding=lambda x:x,
                is_distillation_enabled=lambda x:False,rename_dict=lambda d,p:{p+k:v for k,v in d.items()})
        methods=load('verl','trainer/ppo/ray_trainer.py',ns,['_update_actor','_update_critic'],'RayPPOTrainer')
        batch=self.batch(True)
        seen=[]
        def actor(td):
            seen.append(td)
            return self.tu.get_tensordict({}, {'metrics':{'mfu':0}})
        worker=NS(config=cfg,actor_rollout_wg=NS(update_actor=actor),
                  critic_wg=NS(train_mini_batch=lambda td:NS(get=lambda:actor(td))))
        for name in ('_update_actor','_update_critic'):
            runtime.patch_update(getattr(methods,name))(worker,batch)
            td=seen[-1]
            self.assertEqual(td['mini_batch_size'],len(batch))
            self.assertEqual(td['epochs'],1)
            self.assertIn('matched_weights',td)

    def test_critic_rng_does_not_change_paired_order(self):
        torch=self.torch
        a=c.streams(9001,c.ALGORITHMS[0]);b=c.streams(9001,c.ALGORITHMS[1])
        def schedule(s):
            generator=torch.Generator().manual_seed(s['paired_question_order'])
            return torch.randperm(256,generator=generator)
        self.torch.testing.assert_close(schedule(a),schedule(b))
        runtime.seed_process(b); before=torch.get_rng_state()
        with torch.random.fork_rng():
            torch.manual_seed(b['critic_init']);torch.randn(40)
        self.torch.testing.assert_close(before,torch.get_rng_state())

    def test_pinned_trainer_step_retains_nonmultiple_and_constant_real_rollouts(self):
        from omegaconf import OmegaConf
        from collections import defaultdict
        import time, random
        batch=self.batch(True, True, 'grpo')
        del batch.batch['rollout_log_probs']
        # Deliberately fewer than the legacy 32-row floor, including constant groups.
        calls=[]
        cfg=OmegaConf.create({'actor_rollout_ref':{'rollout':{'n':4,'temperature':1},
                 'actor':{'ppo_mini_batch_size':8,'policy_loss':{'loss_mode':'per_rollout_mean'}}},
                 'algorithm':{'adv_estimator':'grpo','gamma':1.,'lam':1.,'use_kl_in_reward':False,
                              'enable_rollout_level_advantage':False,'rollout_correction':None},
                 'trainer':{'balance_batch':False,'critic_warmup':0},'agentlightning':{}})
        def union(self,other):
            for k,v in other.batch.items():self.batch[k]=v
            return self
        with patch.object(Proto,'union',union,create=True):
            namespace=dict(time=time,torch=self.torch,np=self.np,random=random,DataProto=Proto,
                marked_timer=lambda *a,**kw:nullcontext(),AdvantageEstimator=NS(REMAX='remax'),
                compute_advantage=runtime.advantages, _grpo_group_metrics=lambda batch:{},
                compute_data_metrics=lambda **kw:{}, PER_ROLLOUT_MEAN_LOSS_MODE='per_rollout_mean',
                CISPO_PER_ROLLOUT_MEAN_LOSS_MODE='cispo_per_rollout_mean',
                normalize_advantages_by_rollout=runtime.normalize,reduce_metrics=lambda x:x)
            pinned=load('agentlightning','verl/trainer.py',namespace,['_train_step'],'AgentLightningRayPPOTrainer')
            worker=NS(config=cfg,global_steps=1,use_reference_policy=False,use_critic=False,
                _next_train_batch_dict_for_rollout=lambda:{},_get_gen_batch=lambda b:b,
                _rollout=lambda *a,**kw:(batch,{}),
                _compute_old_log_prob=lambda b:(Proto({'old_log_probs':b.batch['old_log_probs']}),0),
                _update_actor=lambda b:(calls.append(len(b)) or Proto(meta_info={'metrics':{}})),
                checkpoint_manager=NS(sleep_replicas=lambda:None,update_weights=lambda step:calls.append('refresh')))
            metrics,retained=runtime.patch_train_step(pinned._train_step)(worker,{},False)
            self.assertEqual(len(retained),8)
            self.assertEqual(calls,[8,'refresh'])
            self.assertEqual(metrics['training/n_sample_dropped/same_reward'],0)
            self.assertEqual(metrics['training/n_sample_dropped/random'],0)
            self.assertEqual(retained.batch['advantages'].sum().item(),0)

    def test_two_rank_gloo_reduction_matches_single_rank_with_unequal_shards(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(distributed_reduction, args=(tmp,), nprocs=2, join=True)
            observed=json.loads((Path(tmp)/'distributed.json').read_text())
            expected=distributed_reduction(None,tmp)
            for role in ('actor','critic'):
                self.assertAlmostEqual(observed[role],expected[role],places=6)


def distributed_reduction(rank, directory):
    """Real two-rank CPU all-reduce; unequal fragment and generated-token counts."""
    import torch
    from enum import Enum
    from omegaconf import OmegaConf
    import numpy as np
    from tensordict import TensorDict, NonTensorData
    from unittest.mock import patch
    dp=1 if rank is None else 2
    if rank is not None:
        torch.distributed.init_process_group('gloo',init_method='file://'+directory+'/ranks',rank=rank,world_size=2)
    functional=load('verl','utils/torch_functional.py',dict(torch=torch),['masked_sum','masked_mean'])
    policy=load('agentlightning','verl/per_rollout_loss.py',dict(torch=torch,verl_F=functional),
                ['compute_policy_loss_per_rollout_mean']).compute_policy_loss_per_rollout_mean
    metric=load('verl','utils/metric/utils.py',dict(torch=torch,np=np,Enum=Enum))
    tu=NS(get_non_tensor_data=lambda data,key,default:data.get(key,default))
    padding=load('verl','workers/utils/padding.py',dict(torch=torch,F=torch.nn.functional,tu=tu),['no_padding_2_padding'])
    cfg=OmegaConf.create({'clip_ratio':.2,'clip_ratio_low':.2,'clip_ratio_high':.3,'clip_ratio_c':3.,
                          'global_batch_info':{'dp_size':dp}})
    lengths=[1,3,2,4,1];ids=['a','a','b','c','c']
    masks=torch.tensor([[1]*n+[0]*(4-n) for n in lengths],dtype=torch.float32)
    weights=runtime.rollout_weights(masks,ids)
    results={}
    for role in ('actor','critic'):
        parameter=torch.tensor(.11,requires_grad=True)
        for i,n in enumerate(lengths):
            if rank is not None and i%2!=rank:continue
            if role=='actor':
                advantages=torch.tensor([[.2+i*.1,-.4,.7,-.8]])*weights[i:i+1]
                loss,_=policy(torch.zeros(1,4),parameter.expand(1,4),advantages,masks[i:i+1],config=cfg)
            else:
                data=TensorDict({'values':torch.zeros(1,4),'returns':torch.ones(1,4)*(i%2),
                     'response_mask':masks[i:i+1],'matched_weights':weights[i:i+1],
                     'prompts':torch.ones(1,1),'responses':torch.ones(1,4),
                     'attention_mask':torch.tensor([[1]+[1]*n+[0]*(4-n)]),'dp_size':NonTensorData(dp)},batch_size=[1])
                with patch.dict('sys.modules',{'verl.utils.metric':metric,'verl.workers.utils.padding':padding}):
                    loss,_=runtime.value_loss(NS(cliprange_value=.5),{'values':parameter.expand(n+1)},data)
            loss.backward()
        gradient=parameter.grad.detach()
        if rank is not None:
            torch.distributed.all_reduce(gradient)
            gradient/=dp
        results[role]=float(gradient)
    if rank==0:(Path(directory)/'distributed.json').write_text(json.dumps(results))
    if rank is not None:torch.distributed.destroy_process_group()
    return results
