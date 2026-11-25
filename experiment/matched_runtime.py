# VERL-derived value-loss, clipping-diagnostic and update-overlay portions:
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
# Agent Lightning-derived clipping/normalization and trainer-overlay portions:
# Copyright (c) Microsoft. All rights reserved.
# Microsoft MIT terms: experiment/LICENSE-AgentLightning.txt.
# Modified for Agent Lightning Lab: matched rollout-mean value-loss and
# diagnostic reductions, plus explicit trainer/update overlays. Native actor
# loss/backward remains delegated; collection checks/instrumentation are lab additions.
# VERL 7aed6b230776f963fa09509c10d9c3a767d1102c:
# verl/workers/utils/losses.py (value_loss), verl/trainer/ppo/core_algos.py
# (compute_value_loss, with TRL ancestry; compute_policy_loss_vanilla),
# verl/trainer/ppo/ray_trainer.py (_update_actor, _update_critic).
# Agent Lightning cbc56e41e0eab643aa71d195e92e5bba652fa683:
# agentlightning/verl/per_rollout_loss.py and verl/trainer.py (_train_step).
# Licensed under the Apache License, Version 2.0 (the "License") for the
# identified VERL-derived portions; you may not use those portions except
# in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 and LICENSES/Apache-2.0.txt.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Original lab contributions: Copyright (c) 2026 Agent Lightning Lab
# contributors, MIT (root LICENSE). See LICENSING.md and NOTICE for scope.
# These comments were added for release preparation; the historical executed
# source remains 5fb6adfd48ee245509f7dff3ab1d849fcf707d86.

"""Hash-guarded matched-v2 overlay. Installed only in matched-study Ray workers.

One complete collection is one optimizer batch, regardless of fragment count.
The original pinned trainer and engine execute; narrow source edits remove its
legacy floor and supply the runtime minibatch size (no installed file edits).
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import textwrap

try:
    from . import ppo_runtime as audit, ppo_math, ppo_contract
except ImportError:
    import ppo_runtime as audit
    import ppo_math
    import ppo_contract

PROTOCOL = 'matched-rollout-mean-v2'
_INSTALLED = False
_NATIVE_PPO_LOSS = None


def seed_process(streams):
    import numpy as np
    import torch
    random.seed(streams['python'])
    np.random.seed(streams['numpy'])
    torch.manual_seed(streams['torch_cpu'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(streams['torch_cuda'])


def replace_function(fn, replacements, extra=None):
    """Fail closed on source drift; caller verifies complete package file hashes."""
    source = textwrap.dedent(inspect.getsource(fn))
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError(f'Pinned overlay anchor drift: {fn.__name__}: {old[:80]}')
        source = source.replace(old, new)
    namespace = {**fn.__globals__, **(extra or {})}
    exec(compile(source, inspect.getsourcefile(fn), 'exec'), namespace)
    return namespace[fn.__name__]


def patch_train_step(fn):
    source = textwrap.dedent(inspect.getsource(fn))
    start = source.index('    mini_bs = ')
    end = source.index('    print("AgentLightningRayPPOTrainer: sleeping', start)
    return replace_function(fn, [(source[start:end], '''    if len(batch) == 0:
        raise ValueError("No valid complete trajectories")
    metrics["training/n_sample_trained"] = len(batch)
    metrics["training/n_sample_dropped/same_reward"] = 0
    metrics["training/n_sample_dropped/random"] = 0

''')])


def patch_update(fn):
    return replace_function(fn, [(
        'ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n',
        'ppo_mini_batch_size = len(batch)  # one complete collection, one optimizer call')])


def rollout_weights(mask, ids):
    import torch
    totals = defaultdict(float)
    for rid, n in zip(ids, mask.sum(-1).tolist(), strict=True):
        totals[str(rid)] += n
    if not totals or min(totals.values()) <= 0:
        raise ValueError('Empty real trajectory')
    return torch.tensor([1 / (len(totals) * totals[str(r)]) for r in ids],
                        dtype=torch.float32, device=mask.device).unsqueeze(-1) * mask


def normalize(advantages, response_mask, rollout_ids, *, num_trained_rows):
    return advantages * rollout_weights(response_mask, rollout_ids)


def advantages(batch, **kwargs):
    """GRPO over real trajectories; PPO terminal MC before token whitening."""
    import torch
    b, nt = batch.batch, batch.non_tensor_batch
    mask = b['response_mask'].bool()
    ids, groups = nt['rollout_id_list'], nt['data_id_list']
    rewards, by_group = {}, defaultdict(dict)
    for i, (rid, group) in enumerate(zip(ids, groups, strict=True)):
        reward = float(b['token_level_rewards'][i].sum())
        if reward not in (0., 1.) or (rid in rewards and rewards[rid] != reward):
            raise ValueError('Nonbinary or inconsistent trajectory reward')
        rewards[rid] = reward
        by_group[group][rid] = reward
    if any(len(group) != 4 for group in by_group.values()):
        raise ValueError('GRPO/PPO require four distinct complete sampled trajectories per group')
    targets = torch.tensor([rewards[r] for r in ids], device=mask.device).unsqueeze(-1).expand_as(mask).float()
    weights = rollout_weights(mask, ids)
    if kwargs['adv_estimator'] == 'gae':
        raw = (targets - b['values']) * mask
        # Common token population, independent of fragmentation; unbiased token variance.
        valid = raw[mask]
        if valid.numel() < 2:
            raise ValueError('Insufficient generated tokens for whitening')
        policy = (raw - valid.mean()) * torch.rsqrt(valid.var(unbiased=True) + 1e-8) * mask
        earlier = audit._STATE.get('earlier_rewards', [])
        report = ppo_math.prediction_metrics(b['values'][mask].tolist(), targets[mask].tolist(),
                                            sum(earlier)/len(earlier) if earlier else None)
        audit.emit('critic_preupdate', iteration=audit._STATE['step'], prediction=report)
        audit._STATE['earlier_rewards'] = earlier + list(rewards.values())
    elif kwargs['adv_estimator'] == 'grpo':
        group_adv = {}
        for group in by_group.values():
            rs = torch.tensor(list(group.values()), dtype=torch.float32)
            for rid, value in group.items():
                group_adv[rid] = (value - rs.mean()) / (rs.std(unbiased=True) + 1e-6)
        raw = torch.stack([group_adv[r] for r in ids]).to(mask.device).unsqueeze(-1).expand_as(mask) * mask
        policy = raw
    else:
        raise ValueError('Unsupported matched estimator')
    b['returns'], b['advantages'], b['matched_weights'] = targets * mask, policy, weights
    audit.emit('matched_advantages', iteration=audit._STATE['step'], real_rollouts=len(rewards),
               rows=len(ids), generated_tokens=int(mask.sum()),
               constant_groups=sum(len(set(g.values())) == 1 for g in by_group.values()),
               bound='first four fragment rows', raw=raw[:4].tolist(), policy=policy[:4].tolist(),
               tensors={k: b[k][:4].detach().cpu().tolist() for k in
                        ('response_mask', 'old_log_probs', 'rollout_log_probs', 'values', 'returns',
                         'token_level_rewards', 'matched_weights') if k in b})
    return batch


def value_loss(config, model_output, data, dp_group=None):
    """Clipped native value objective with independent rollout-mean weights."""
    import torch
    from verl.utils.metric import Metric, AggregationType
    from verl.workers.utils.padding import no_padding_2_padding
    pred = no_padding_2_padding(model_output['values'], data)
    dp_size = data['dp_size']
    d = data.select('values', 'returns', 'response_mask', 'matched_weights').to_padded_tensor()
    clipped = d['values'] + (pred - d['values']).clamp(-config.cliprange_value, config.cliprange_value)
    raw = (pred - d['returns']).square()
    clipped_error = (clipped - d['returns']).square()
    weights = d['matched_weights'] * d['response_mask']
    loss = .5 * (torch.maximum(raw, clipped_error) * weights).sum() * dp_size
    return loss, Metric.from_dict({'critic/vf_loss': loss.detach().item(),
                  'critic/vf_clipfrac': (((clipped_error > raw)*weights).sum()*dp_size).item()},
                  aggregation=AggregationType.SUM)


def policy_loss(config, model_output, data, dp_group=None):
    """Keep native clipping/backward; aggregate diagnostics over real trajectories."""
    import torch
    from verl.utils.metric import Metric, AggregationType
    from verl.workers.utils.padding import no_padding_2_padding
    loss, metrics = _NATIVE_PPO_LOSS(config, model_output, data, dp_group)
    pred = no_padding_2_padding(model_output['log_probs'], data)
    d = data.select('old_log_probs', 'advantages', 'response_mask', 'matched_weights').to_padded_tensor()
    delta = (pred-d['old_log_probs']).clamp(-20., 20.)
    ratio = delta.exp()
    low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio
    direct = -d['advantages']*ratio
    clipped = -d['advantages']*ratio.clamp(1-low, 1+high)
    lower = (torch.maximum(direct, clipped) > -d['advantages']*config.clip_ratio_c) & (d['advantages'] < 0)
    weights = d['matched_weights']*d['response_mask']*data['dp_size']
    metrics.update(Metric.from_dict({
        'actor/ppo_kl': (-delta*weights).sum(),
        'actor/pg_clipfrac': ((clipped > direct)*weights).sum(),
        'actor/pg_clipfrac_lower': (lower*weights).sum(),
    }, aggregation=AggregationType.SUM))
    return loss, metrics


class CollectionOnlyComplete(Exception):
    """Stop after admitted real training collection, before advantage/optimizer work."""


def instrument_adapter(module, *, collection_only=False, calibration=False):
    # Reuse the original fragment/context mapper, not the legacy selection wrapper.
    audit._instrument_adapter(module)
    original = module.RolloutAdapter.get_train_data_batch
    def collect(self, completed_rollouts, *, global_steps=0):
        ids = [r.rollout_id for r in completed_rollouts]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate sampled rollout IDs')
        out, metrics = original(self, completed_rollouts, global_steps=global_steps)
        observed = Counter(out.non_tensor_batch['rollout_id_list'])
        audit.emit('collection_manifest', iteration=global_steps, trajectories=[
            {'rollout': r.rollout_id, 'group': r.data_id, 'case_id': (r.input or {}).get('case_id'),
             'reward': r.final_reward, 'fragments': observed[r.rollout_id],
             'original_generated_tokens': sum(len(t.response['token_ids']) for t in (r.triplets or []))}
            for r in completed_rollouts], adapter_metrics=metrics)
        # Fail the entire iteration on missing/truncated context. No reward-dependent exclusion,
        # no shortened partial trajectory silently treated as a complete group member.
        actual_tokens = defaultdict(int)
        for rid, n in zip(out.non_tensor_batch['rollout_id_list'], out.batch['response_mask'].sum(-1).tolist()):
            actual_tokens[rid] += int(n)
        for r in completed_rollouts:
            expected = sum(len(t.response['token_ids']) for t in (r.triplets or []))
            if r.final_reward not in (0, 1) or not expected or actual_tokens[r.rollout_id] != expected:
                raise ValueError('Missing reward/trace or truncated trajectory; collection rejected')
        if out.batch['is_drop_mask'].any():
            raise ValueError('Invalid prompt context; collection rejected')
        groups = defaultdict(set)
        for r in completed_rollouts:
            groups[r.data_id].add(r.rollout_id)
        if any(len(g) != 4 for g in groups.values()):
            raise ValueError('Incomplete group; collection rejected')
        if collection_only:
            if len(completed_rollouts) != 32 or len(groups) != 8:
                raise ValueError('Development collection must contain eight full groups')
            rewards = defaultdict(list)
            for r in completed_rollouts:
                rewards[r.data_id].append(r.final_reward)
            audit.emit('development_collection_admitted', iteration=global_steps,
                       real_rollouts=32, groups=8, generated_tokens=sum(actual_tokens.values()),
                       fragment_rows=len(out), constant_groups=sum(len(set(v)) == 1 for v in rewards.values()))
            if calibration:
                # Evidence only: read the admitted batch; nothing is changed or filtered.
                width = out.batch['prompts'].shape[1]
                attention = out.batch['attention_mask']
                measured = defaultdict(lambda: {'mask_tokens': 0, 'context_tokens': 0, 'prompt_tokens': []})
                for i, rid in enumerate(out.non_tensor_batch['rollout_id_list']):
                    mask = int(out.batch['response_mask'][i].sum())
                    measured[rid]['mask_tokens'] += mask
                    measured[rid]['context_tokens'] += int(attention[i, width:].sum()) - mask
                    measured[rid]['prompt_tokens'].append(int(attention[i, :width].sum()))
                audit.emit('calibration_collection_tokens', iteration=global_steps,
                           limits={'max_prompt_length': self.max_prompt_length, 'max_response_length': self.max_response_length},
                           truncated_rows=int(metrics.get('training/n_truncated_sample', 0)),
                           dropped_prompt_rows=int(out.batch['is_drop_mask'].sum()),
                           rollouts=[{'rollout': r.rollout_id, **measured[r.rollout_id]} for r in completed_rollouts])
            raise CollectionOnlyComplete()
        return out, metrics
    module.RolloutAdapter.get_train_data_batch = collect


def tracking_class(wb, stage, run):
    """Initialize W&B exactly where the trainer creates Tracking, in its Ray process."""
    class Tracking:
        def __init__(self, project_name, experiment_name, default_backend, config):
            self.run = wb.init(**{k: stage['wandb'][k] for k in
                                 ('entity', 'project', 'group', 'id', 'name', 'job_type', 'resume')},
                               config={**config, 'matched_study': stage['wandb']['config']})
            audit._STATE['tracking'] = self
        def log(self, data, step, **kwargs):
            # Local unsampled rows survive a hosted-history final-row omission.
            with (run/'metrics.jsonl').open('a') as f:
                f.write(json.dumps({'step': step, 'metrics': data}, default=float) + '\n')
            self.run.log(data, step=step)
        def finish(self):
            self.run.finish()
            audit.emit('wandb_finished', id=stage['wandb']['id'])
    return Tracking


def register_in_worker():
    global _INSTALLED, _NATIVE_PPO_LOSS
    if _INSTALLED:
        return
    audit.source_identity()
    lock = json.loads(Path(__file__).with_name('matched_source_lock.json').read_text())
    for package, files in lock.items():
        root = Path(importlib.util.find_spec(package).origin).parent
        for relative, expected in files.items():
            if hashlib.sha256((root/relative).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Unreviewed matched runtime source: {package}/{relative}')
    try:
        from .request_transport import load
    except ImportError:
        from request_transport import load
    request = load()
    run = Path(os.environ['AGL_PPO_RUN_DIR'])
    streams = request['job']['seed_streams']
    seed_process(streams)
    from agentlightning.verl.per_rollout_loss import register_in_worker as upstream
    upstream()
    from agentlightning.verl import trainer, rollout_adapter
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.workers import engine_workers
    from verl.workers.utils import losses
    import wandb
    try:
        from . import reader_modes as modes
    except ImportError:
        import reader_modes as modes
    # Mode selection only. Historical plans dispatch by stage name as before.
    collection_only = modes.collection_only(request['plan'], request['stage'])
    instrument_adapter(rollout_adapter, collection_only=collection_only,
                       calibration=modes.phase(request['plan']) == modes.CALIBRATION)
    trainer.compute_advantage = advantages
    trainer.normalize_advantages_by_rollout = normalize
    cls = trainer.AgentLightningRayPPOTrainer
    cls._train_step = patch_train_step(cls._train_step)
    cls._update_actor = patch_update(RayPPOTrainer._update_actor)
    cls._update_critic = patch_update(RayPPOTrainer._update_critic)
    losses.value_loss = value_loss
    _NATIVE_PPO_LOSS = losses.ppo_loss
    losses.ppo_loss = engine_workers.ppo_loss = policy_loss
    # Reuse audited model construction and real optimizer instrumentation. Its retention
    # callback is never used: matched runs retain only step-20/final saves, no deletion.
    audit.contract.CRITIC_SEED = streams['critic_init'] or 0
    audit._instrument_engine()
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
    make_optimizer = FSDPEngine._build_optimizer
    def optimizer(self, module):
        import torch
        # Optimizer initialization owns its stream and cannot advance actor sampling RNG.
        with torch.random.fork_rng():
            selected = streams['critic_init'] if self.model_config.model_type == 'value_model' else streams['actor_optimizer']
            torch.manual_seed(selected)
            torch.cuda.manual_seed_all(selected)
            return make_optimizer(self, module)
    FSDPEngine._build_optimizer = optimizer
    trainer.Tracking = tracking_class(wandb, request['stage'], run)
    original_fit = cls.fit
    def fit(self):
        try:
            try:
                result = original_fit(self)
            except CollectionOnlyComplete:
                if not collection_only:
                    raise
                audit.emit('trainer_returned', global_step=self.global_steps, validation_only=False,
                           collection_only=True, optimizer_updates=0, resources=audit._resource_snapshot())
                return
            audit.emit('trainer_returned', global_step=self.global_steps,
                       validation_only=bool(self.config.trainer.val_only), resources=audit._resource_snapshot())
            return result
        finally:
            if audit._STATE.get('tracking'):
                audit._STATE['tracking'].finish()
    cls.fit = fit
    original_init = cls.init_workers
    def init_workers(self):
        result = original_init(self)
        original_refresh = self.checkpoint_manager.update_weights
        def refresh(step, *args, **kwargs):
            result = original_refresh(step, *args, **kwargs)
            audit.emit('weight_refresh', step=step)
            return result
        self.checkpoint_manager.update_weights = refresh
        return result
    cls.init_workers = init_workers
    original_save = cls._save_checkpoint
    def save(self):
        original_save(self)
        audit.emit('native_checkpoint_saved', step=self.global_steps)
    cls._save_checkpoint = save
    _INSTALLED = True
    audit.emit('matched_registered', protocol=PROTOCOL, streams=streams,
               limitations='vLLM scheduling/Ray execution order and GPU kernels are not bitwise deterministic')
