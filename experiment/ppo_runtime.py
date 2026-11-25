# VERL-derived _instrument_engine.valuehead loading-route adaptation:
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Modified for Agent Lightning Lab: strict local Llama token classification,
# fallback prohibited, with loading and head-initialization diagnostics.
# Origin: VERL 7aed6b230776f963fa09509c10d9c3a767d1102c,
# verl/utils/model.py (load_valuehead_model).
# Remaining wrappers and optimizer instrumentation are lab additions calling
# upstream interfaces; this notice does not label the whole module copied VERL.
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

"""Explicit PPO-only, hash-guarded in-process instrumentation overlay.

No installed file is modified. This module is installed through Ray's documented
worker_process_setup_hook only for ppo-mc-v1. Runtime execution is a future gate.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
from collections import Counter, defaultdict
from contextlib import nullcontext

try:
    from . import ppo_contract as contract, ppo_math as math_port
except ImportError:
    import ppo_contract as contract
    import ppo_math as math_port

_INSTALLED = False
_STATE = {}


def source_identity(roots=None):
    lock = json.loads(Path(__file__).with_name('ppo_source_lock.json').read_text())
    for package, files in lock['files'].items():
        root = Path(roots[package]) if roots else Path(importlib.util.find_spec(package).origin).parent
        for relative, expected in files.items():
            actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f'Unreviewed runtime source: {package}/{relative}: expected={expected}, actual={actual}')
        if package == 'agentlightning':
            example = root.parent / 'examples/calc_x/train_calc_agent.py'
            if hashlib.sha256(example.read_bytes()).hexdigest() != lock['calc_example_sha256']:
                raise ValueError('Unreviewed Calc-X configuration/entrypoint')
    return {'lock_sha256': contract.digest_json(lock), 'agentlightning_commit': lock['agentlightning_commit'],
            'verl_source_commit': lock['verl_source_commit'], 'files_verified': 1+sum(map(len, lock['files'].values()))}


def emit(event, **fields):
    root = Path(os.environ['AGL_PPO_AUDIT_DIR'])
    root.mkdir(parents=True, exist_ok=True)
    with (root / f'events-{os.getpid()}.jsonl').open('a') as handle:
        handle.write(json.dumps({'event': event, 'pid': os.getpid(), 'time': time.time(), **fields},
                                default=str, allow_nan=False) + '\n')


def retention_path(path, run):
    path, run = Path(path).resolve(), Path(run).resolve()
    if (not run.name.startswith('ppo-mc-v1-') or run.parent.name != 'runs'
            or run.parent.parent.name != 'phase2' or not path.is_relative_to(run / 'checkpoints')):
        raise ValueError('Retention outside fresh PPO checkpoint namespace refused')
    if path.name not in ('actor', 'critic') or not path.parent.name.startswith('global_step_'):
        raise ValueError('Retention must target a native role checkpoint')
    return path


def runtime_assertions():
    """Real pinned Torch/VERL arithmetic, before model loading; never invoked by client tests."""
    import torch
    from verl.trainer.ppo.core_algos import compute_gae_advantage_return, compute_value_loss
    from agentlightning.verl.per_rollout_loss import normalize_advantages_by_rollout, compute_policy_loss_per_rollout_mean
    from omegaconf import OmegaConf
    from verl.workers.config import FSDPCriticConfig
    from verl.utils import omega_conf_to_dataclass
    # Gaps and padding are excluded; terminal positive and zero targets.
    values = torch.tensor([[.2, 99., .4, 0.], [.7, .3, 91., 0.]])
    mask = torch.tensor([[1, 0, 1, 0], [1, 1, 0, 0]])
    rewards = torch.tensor([[0., 0., 1., 0.], [0., 0., 0., 0.]])
    raw, targets, expected = math_port.gae(rewards.tolist(), values.tolist(), mask.tolist())
    adv, ret = compute_gae_advantage_return(rewards, values, mask, 1., 1.)
    torch.testing.assert_close(adv[mask.bool()], torch.tensor(expected)[mask.bool()], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(ret[mask.bool()], torch.tensor(targets)[mask.bool()], atol=2e-6, rtol=2e-6)
    saved = ret.clone()
    scaled = normalize_advantages_by_rollout(adv, mask, ['a', 'b'], num_trained_rows=2)
    torch.testing.assert_close(ret, saved)
    torch.testing.assert_close(scaled[mask.bool()], (adv/4)[mask.bool()])
    cfg = OmegaConf.create({'clip_ratio': .2, 'clip_ratio_low': .2, 'clip_ratio_high': .3,
                            'clip_ratio_c': 3., 'global_batch_info': {'dp_size': 1}})
    logp = torch.zeros_like(adv, requires_grad=True)
    loss, _ = compute_policy_loss_per_rollout_mean(torch.zeros_like(logp), logp, scaled, mask, config=cfg)
    loss.backward()
    torch.testing.assert_close(logp.grad, -scaled*mask)
    # Real pinned critic loss excludes observations/padding; engine sums microbatch losses.
    predicted = values.clone().requires_grad_()
    value_loss = sum(compute_value_loss(predicted[i:i+1], ret[i:i+1], values[i:i+1],
                                      mask[i:i+1], .5)[0] for i in range(2))
    value_loss.backward()
    expected_gradient = (values-ret)*mask/mask.sum(-1, keepdim=True)
    torch.testing.assert_close(predicted.grad, expected_gradient)
    # Real native merged/fragment equality under the same generated-token values.
    merged_v = torch.tensor([[.1, .2, 99., .4, .7, 0.]])
    merged_m = torch.tensor([[1, 1, 0, 1, 1, 0]])
    frag_v = torch.tensor([[.1, .2, 0.], [.4, .7, 0.]])
    frag_m = torch.tensor([[1, 1, 0], [1, 1, 0]])
    for reward in (0., 1.):
        ma, mr = compute_gae_advantage_return(torch.tensor([[0., 0., 0., 0., reward, 0.]]), merged_v, merged_m, 1., 1.)
        fa, fr = compute_gae_advantage_return(torch.tensor([[0., reward, 0.]]*2), frag_v, frag_m, 1., 1.)
        torch.testing.assert_close(ma[merged_m.bool()], fa[frag_m.bool()], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(mr[merged_m.bool()], fr[frag_m.bool()], atol=2e-6, rtol=2e-6)
    # Exact effective microbatch fields, not just accepted YAML spellings.
    config = json.loads(Path(os.environ['AGL_PPO_RUN_DIR'], 'resolved_config.json').read_text())
    cc = omega_conf_to_dataclass(OmegaConf.create(config['critic']))
    assert isinstance(cc, FSDPCriticConfig)
    assert cc.engine.micro_batch_size_per_gpu == cc.engine.infer_micro_batch_size_per_gpu == 1
    assert cc.engine.use_dynamic_bsz is False
    emit('runtime_arithmetic_assertions', status='passed', synthetic=True, model_loaded=False)


def _cgroup_memory(proc=Path('/proc/self')):
    """Read counters for this process's memory controller, including subtree mounts."""
    import re
    from pathlib import PurePosixPath

    def read(path):
        try:
            return path.read_text().strip()
        except (OSError, UnicodeError):
            return None

    membership = read(proc/'cgroup')
    result = {'scope': 'memory controller of the emitting process', 'process_cgroup': membership,
              'cgroup_version': None, 'memory.current': None, 'memory.peak': None,
              'memory.events': None, 'memory.failcnt': None, 'cgroup_process_count': None}
    groups = {}
    for line in (membership or '').splitlines():
        fields = line.split(':', 2)
        if len(fields) != 3:
            continue
        hierarchy, controllers, path = fields
        group = PurePosixPath(path)
        if not group.is_absolute() or '..' in group.parts:
            continue
        if 'memory' in controllers.split(','):
            groups[1] = group
        elif hierarchy == '0' and controllers == '':
            groups[2] = group
    # In a hybrid hierarchy, use the controller that actually owns memory.
    version = 1 if 1 in groups else 2
    def unescape(value):
        return re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), value)

    candidates = []
    result['cgroup_version'] = version if version in groups else None
    for line in (read(proc/'mountinfo') or '').splitlines():
        left, sep, right = line.partition(' - ')
        fields, fs = left.split(), right.split()
        if version not in groups or not sep or len(fields) < 6 or len(fs) < 3:
            continue
        if version == 1 and (fs[0] != 'cgroup' or 'memory' not in fs[2].split(',')):
            continue
        if version == 2 and fs[0] != 'cgroup2':
            continue
        root, mount = (Path(unescape(v)) for v in fields[3:5])
        if not root.is_absolute() or not mount.is_absolute() or not groups[version].is_relative_to(root):
            continue
        candidates.append((len(root.parts), mount/groups[version].relative_to(root)))
    if candidates:
        _, directory = max(candidates)
        files = ({'memory.current': 'memory.usage_in_bytes', 'memory.peak': 'memory.max_usage_in_bytes',
                  'memory.failcnt': 'memory.failcnt'} if version == 1 else
                 {'memory.current': 'memory.current', 'memory.peak': 'memory.peak'})
        for key, name in files.items():
            value = read(directory/name)
            if value is not None and re.fullmatch('[0-9]+', value):
                result[key] = value
        if version == 2:
            events = read(directory/'memory.events')
            if events and all(re.fullmatch(r'[a-z_]+ [0-9]+', line) for line in events.splitlines()):
                result['memory.events'] = events
        members = read(directory/'cgroup.procs')
        if members is not None and all(v.isdecimal() for v in members.splitlines()):
            result['cgroup_process_count'] = len(members.splitlines())
    result['cgroup_available'] = result['memory.current'] is not None or result['memory.peak'] is not None
    return result


def _resource_snapshot():
    import torch
    result = _cgroup_memory()
    result['cuda_scope'] = 'this process allocator cumulative peak; excludes other processes and driver allocations'
    if torch.cuda.is_available():
        result.update(cuda_allocated_peak=torch.cuda.max_memory_allocated(),
                      cuda_reserved_peak=torch.cuda.max_memory_reserved())
    return result


def _rows(batch, step):
    b, nt = batch.batch, batch.non_tensor_batch
    masks = b['response_mask'].cpu().tolist()
    return [{'iteration': step, 'row_id': str(nt['ppo_row_id'][i]), 'group': str(nt['data_id_list'][i]),
             'case_id': str(nt['ppo_case_id'][i]),
             'rollout': str(nt['rollout_id_list'][i]), 'fragment_index': int(nt['ppo_fragment'][i]),
             'generated_turns': json.loads(str(nt['ppo_turns'][i])),
             'generated_tokens': sum(mask), 'reward': float(b['token_level_scores'][i].sum()),
             'marked_drop': bool(b['is_drop_mask'][i]), 'padded': False}
            for i, mask in enumerate(masks)]


def _instrument_adapter(adapter):
    original = adapter.RolloutAdapter.get_train_data_batch

    def collect(self, completed_rollouts, *, global_steps=0):
        import numpy as np
        out, metrics = original(self, completed_rollouts, global_steps=global_steps)
        by_id = {r.rollout_id: r for r in completed_rollouts}
        counts, turns, fragments = Counter(), [], []
        for i, rid in enumerate(out.non_tensor_batch['rollout_id_list']):
            b = out.batch
            prompt = b['prompts'][i][b['attention_mask'][i, :b['prompts'].shape[1]].bool()].tolist()
            response = b['responses'][i].tolist()
            mask = b['response_mask'][i].tolist()
            full = prompt + response
            mapping = {}
            for turn, triplet in enumerate(by_id[rid].triplets or []):
                tp, tr = triplet.prompt['token_ids'], triplet.response['token_ids']
                start = len(tp)-len(prompt)
                if start >= 0 and full[:len(tp)] == tp:
                    for j, token in enumerate(tr):
                        pos = start+j
                        if pos < len(response) and mask[pos] and response[pos] == token:
                            mapping[pos] = turn
                        else:
                            break
            included = [mapping.get(t) for t, m in enumerate(mask) if m]
            if not bool(b['is_drop_mask'][i]) and any(t is None for t in included):
                raise ValueError('Retained generated token lacks its original model-call context')
            turns.append(json.dumps(included))
            fragments.append(counts[rid]); counts[rid] += 1
        out.non_tensor_batch['ppo_row_id'] = np.array([f'{global_steps}:{i}' for i in range(len(out))])
        out.non_tensor_batch['ppo_fragment'] = np.array(fragments)
        out.non_tensor_batch['ppo_turns'] = np.array(turns)
        out.non_tensor_batch['ppo_case_id'] = np.array([
            str((by_id[rid].input or {}).get('case_id', 'unavailable'))
            for rid in out.non_tensor_batch['rollout_id_list']])
        _STATE['collected'] = _rows(out, global_steps)
        _STATE['step'] = global_steps
        _STATE['rollout_rewards'] = [float(r.final_reward if r.final_reward is not None else self.reward_fillna_value)
                                     for r in completed_rollouts]
        emit('collected_rows', scope='pre-selection', rows=_STATE['collected'],
             grpo=math_port.grpo_diagnostic(_STATE['collected']), adapter_metrics=metrics)
        return out, metrics
    adapter.RolloutAdapter.get_train_data_batch = collect


def _instrument_trainer(trainer):
    original_adv = trainer.compute_advantage
    original_norm = trainer.normalize_advantages_by_rollout
    original_candidates = trainer._same_reward_uid_indices
    cls = trainer.AgentLightningRayPPOTrainer
    original_step, original_save, original_load = cls._train_step, cls._save_checkpoint, cls._load_checkpoint

    def candidates(batch):
        indices = original_candidates(batch)
        _STATE['constant_candidates'] = {str(batch.non_tensor_batch['ppo_row_id'][i]) for i in indices}
        return indices

    def advantage(batch, **kwargs):
        import torch
        rows = _rows(batch, _STATE['step'])
        mask = batch.batch['response_mask'].bool()
        scores = batch.batch['token_level_rewards']
        # No shaping, observation rewards, or terminal bootstrap in this variant.
        assert torch.all(scores[~mask] == 0)
        for i in range(len(batch)):
            positions = mask[i].nonzero().flatten()
            assert len(positions) and torch.all(scores[i, :positions[-1]] == 0)
        before_returns = scores.sum(-1, keepdim=True).expand_as(scores)
        expected_raw = before_returns - batch.batch['values']
        import verl.utils.torch_functional as functional
        native_whiten = functional.masked_whiten
        captured = []
        def capture(values, mask, *args, **kw):
            captured.append(values.detach().clone())
            return native_whiten(values, mask, *args, **kw)
        functional.masked_whiten = capture
        try:
            batch = original_adv(batch, **kwargs)
        finally:
            functional.masked_whiten = native_whiten
        assert len(captured) == 1, 'Expected exactly one GAE whitening operation'
        raw = captured[0]
        residual = float((raw[mask]-expected_raw[mask]).abs().max())
        emit('mc_finite_precision_check', max_abs_residual=residual if math_isfinite(residual) else None,
             finite=math_isfinite(residual), dtype=str(raw.dtype), atol=2e-4, rtol=2e-5)
        torch.testing.assert_close(raw[mask], expected_raw[mask], atol=2e-4, rtol=2e-5)
        torch.testing.assert_close(batch.batch['returns'][mask], before_returns[mask], atol=2e-4, rtol=2e-5)
        values, targets = batch.batch['values'][mask].cpu().tolist(), batch.batch['returns'][mask].cpu().tolist()
        n = _STATE.get('previous_reward_n', 0)
        baseline = _STATE.get('previous_reward_sum', 0)/n if n else None
        report = math_port.prediction_metrics(values, targets, baseline)
        _STATE['prediction'] = report
        _STATE['retained'] = rows
        _STATE['raw'] = raw.detach().cpu().tolist()
        _STATE['whitened'] = batch.batch['advantages'].detach().cpu().tolist()
        _STATE['mask'] = mask.cpu().tolist()
        emit('advantages', stage='raw_pre_whitening', prediction=report,
             mc_residual_max_abs=float((raw[mask]-expected_raw[mask]).abs().max()),
             summaries=math_port.stage_summaries(_STATE['raw'], _STATE['mask'], rows))
        emit('advantages', stage='post_whitening', scope='post-selection',
             summaries=math_port.stage_summaries(_STATE['whitened'], _STATE['mask'], rows),
             grpo=math_port.grpo_diagnostic(rows))
        mismatch = None
        if 'rollout_log_probs' in batch.batch:
            difference = (batch.batch['old_log_probs']-batch.batch['rollout_log_probs'])[mask]
            mismatch = math_port.summary(difference.detach().cpu().tolist())
        emit('rollout_training_logprob_mismatch', available=mismatch is not None,
             old_training_minus_sampling=mismatch, actor_ppo_kl='separate approximate sampled update diagnostic')
        # Bounded audit: four complete rows per iteration, original input IDs and positions included.
        keys = ('input_ids', 'position_ids', 'attention_mask', 'responses', 'response_mask',
                'values', 'token_level_rewards', 'returns', 'advantages', 'old_log_probs', 'rollout_log_probs')
        emit('selected_token_audit', prediction_time='pre-update', rows=rows[:4],
             tensors={k: batch.batch[k][:4].detach().cpu().tolist() for k in keys if k in batch.batch},
             raw_advantages=_STATE['raw'][:4], bound='first four rows; configured sequence limits')
        # Position and turn strata avoid end-of-answer predictions dominating the claim.
        strata = defaultdict(lambda: ([], []))
        vrows, rrows = batch.batch['values'].cpu().tolist(), batch.batch['returns'].cpu().tolist()
        for row, vs, rs, ms in zip(rows, vrows, rrows, _STATE['mask'], strict=True):
            positions = [j for j, m in enumerate(ms) if m]
            for ordinal, (j, turn) in enumerate(zip(positions, row['generated_turns'], strict=True)):
                for key in (f'turn:{turn}', f'position_quartile:{min(3, ordinal*4//len(positions))}'):
                    strata[key][0].append(vs[j]); strata[key][1].append(rs[j])
        emit('prediction_strata', strata={k: math_port.prediction_metrics(v, r, baseline)
                                         for k, (v, r) in strata.items() if len(v) > 1})
        return batch

    def normalize(advantages, response_mask, rollout_ids, *, num_trained_rows):
        result = original_norm(advantages, response_mask, rollout_ids, num_trained_rows=num_trained_rows)
        _, coefficients = math_port.normalize_actor(advantages.cpu().tolist(), response_mask.cpu().tolist(), rollout_ids)
        for i, (row, coefficient) in enumerate(zip(_STATE['retained'], coefficients, strict=True)):
            row.update(status='retained', reason='legacy retention', optimizer_minibatch=i//32,
                       actor_normalization_coefficient=coefficient,
                       critic_weighting='sum of per-row generated-token means; microbatch=1')
        emit('advantages', stage='post_actor_normalization',
             summaries=math_port.stage_summaries(result.cpu().tolist(), response_mask.cpu().tolist(), _STATE['retained']),
             selected_effective_advantages=result[:4].cpu().tolist())
        return result

    def step(self, *args, **kwargs):
        _STATE.update(retained=[], constant_candidates=set())
        start = time.monotonic()
        try:
            result = original_step(self, *args, **kwargs)
            retained = {r['row_id']: r for r in _STATE['retained']}
            audit = []
            for row in _STATE.get('collected', []):
                if row['row_id'] in retained:
                    audit.append(retained[row['row_id']])
                else:
                    reason = ('marked_prompt_or_adapter_invalid' if row['marked_drop'] else
                              'legacy_constant_reward_floor' if row['row_id'] in _STATE['constant_candidates'] else
                              'legacy_random_floor')
                    audit.append(dict(row, status='dropped', reason=reason, optimizer_minibatch=None,
                                      actor_normalization_coefficient=0.0))
            emit('row_selection', rows=audit, elapsed_seconds=time.monotonic()-start, resources=_resource_snapshot())
            if result:
                report = _STATE['prediction']
                result[0].update({'ppo_audit/target_variance': report['target_variance'],
                                  'ppo_audit/ev_valid': int(report['ev_valid']),
                                  'ppo_audit/preupdate_mse': report['mse'],
                                  'ppo_audit/preupdate_mae': report['mae'],
                                  'ppo_audit/preupdate_bias': report['bias']})
            # Update the baseline only AFTER scoring this batch. Each real rollout once.
            rewards = _STATE.get('rollout_rewards', [])
            _STATE['previous_reward_sum'] = _STATE.get('previous_reward_sum', 0) + sum(rewards)
            _STATE['previous_reward_n'] = _STATE.get('previous_reward_n', 0) + len(rewards)
            return result
        except BaseException:
            emit('step_failed', iteration=self.global_steps, resources=_resource_snapshot())
            raise

    def save(self):
        run = Path(os.environ['AGL_PPO_RUN_DIR'])
        if Path(self.config.trainer.default_local_dir).resolve() != (run/'checkpoints').resolve():
            raise ValueError('Checkpoint directory differs from fresh run')
        emit('checkpoint_save_begin', step=self.global_steps, resources=_resource_snapshot())
        try:
            original_save(self)
        except BaseException:
            emit('checkpoint_save_failed', step=self.global_steps, resources=_resource_snapshot())
            raise
        checkpoint = run/'checkpoints'/f'global_step_{self.global_steps}'
        roles = {}
        for role in ('actor', 'critic'):
            roles[role] = {}
            for path in sorted((checkpoint/role).rglob('*')):
                if path.is_file():
                    with path.open('rb') as handle:
                        sha = hashlib.file_digest(handle, 'sha256').hexdigest()
                    roles[role][str(path.relative_to(checkpoint/role))] = {'bytes': path.stat().st_size, 'sha256': sha}
        record = dict(contract.identity(contract.PPO), step=self.global_steps, complete=True,
                      base_revision=contract.BASE_REVISION, roles=roles, source_identity=source_identity(),
                      resolved_config_sha256=hashlib.sha256((run/'resolved_config.json').read_bytes()).hexdigest(),
                      bundle_source_sha256=json.loads((run/'manifest.json').read_text())['bundle_source_sha256'])
        if self.global_steps == 2:
            contract.validate_checkpoint(record, contract.PPO)
        with (checkpoint/'algorithm_manifest.json').open('x') as handle:
            json.dump(record, handle, indent=2)
        emit('checkpoint_saved', checkpoint=str(checkpoint), total_bytes=sum(f['bytes'] for fs in roles.values() for f in fs.values()),
             resources=_resource_snapshot())

    def load(self):
        result = original_load(self)
        if self.config.trainer.resume_mode != 'disable':
            emit('native_reload_returned', actor=str(Path(self.config.trainer.resume_from_path)/'actor'),
                 critic=str(Path(self.config.trainer.resume_from_path)/'critic'),
                 optimizer_continuation_validated=False, inference_refresh='subsequent upstream call; inspect logs')
        return result

    trainer._same_reward_uid_indices = candidates
    trainer.compute_advantage = advantage
    trainer.normalize_advantages_by_rollout = normalize
    cls._train_step, cls._save_checkpoint, cls._load_checkpoint = step, save, load


def _instrument_engine():
    import torch
    import verl.utils.model as model_utils
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
    from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager
    build = FSDPEngine._build_module
    optimizer = FSDPEngine._build_optimizer
    opt_step = FSDPEngine.optimizer_step
    delete = BaseCheckpointManager.remove_previous_save_local_path

    def valuehead(local_path, torch_dtype, model_config, trust_remote_code):
        from transformers import AutoModelForTokenClassification
        assert Path(local_path).is_dir() and model_config.model_type == 'llama' and not trust_remote_code
        model, info = AutoModelForTokenClassification.from_pretrained(
            local_path, torch_dtype=torch_dtype, config=model_config, attn_implementation='flash_attention_2',
            trust_remote_code=False, local_files_only=True, output_loading_info=True)
        emit('critic_loading_info', class_name=type(model).__name__, info=info,
             seed=contract.CRITIC_SEED, fresh_parameters=['score.weight', 'score.bias'])
        contract.validate_loading_info(info)
        if type(model).__name__ != 'LlamaForTokenClassification':
            raise ValueError('Unexpected value-head model class; fallback prohibited')
        emit('critic_head_initialization', parameters={
            name: {'shape': list(p.shape), 'dtype': str(p.dtype),
                   'sha256': hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
            for name, p in model.named_parameters() if name.startswith('score.')},
            weight_initializer='Transformers normal(0, config.initializer_range); bias=zeros',
            seed=contract.CRITIC_SEED)
        return model

    def module(self):
        critic = self.model_config.model_type == 'value_model'
        emit('model_construction_begin', role='critic' if critic else 'actor', resources=_resource_snapshot())
        tokenizer = self.model_config.tokenizer
        for key in ('bos_token_id', 'eos_token_id', 'pad_token_id'):
            if getattr(self.model_config.hf_config, key) != getattr(tokenizer, key):
                raise ValueError(f'Model/tokenizer {key} mismatch')
        if tokenizer.chat_template != (Path(self.model_config.local_path)/'chat_template.jinja').read_text():
            raise ValueError('Model tokenizer template differs from the prepared serving template')
        # fork_rng restores actor/data-order RNG state after constructing the critic.
        with torch.random.fork_rng() if critic else nullcontext():
            if critic:
                torch.manual_seed(contract.CRITIC_SEED)
                torch.cuda.manual_seed_all(contract.CRITIC_SEED)
            result = build(self)
        counts = Counter()
        for p in result.parameters():
            if p.requires_grad:
                counts[str(p.dtype)] += p.numel()
        emit('model_constructed', role='critic' if critic else 'actor', class_name=type(result).__name__,
             trainable_parameters_by_storage_dtype=dict(counts), configured_compute_dtype='bfloat16',
             token_ids={key: getattr(tokenizer, key) for key in ('bos_token_id', 'eos_token_id', 'pad_token_id')},
             parameter_bytes=sum(p.numel()*p.element_size() for p in result.parameters()), resources=_resource_snapshot())
        return result

    def make_optimizer(self, module):
        opt = optimizer(self, module)
        self._ppo_actual_steps = 0
        def counted(opt, args, kwargs):
            self._ppo_actual_steps += 1
            state_bytes = sum(value.numel()*value.element_size() for state in opt.state.values()
                              for value in state.values() if torch.is_tensor(value))
            emit('optimizer_step', role='critic' if self.model_config.model_type == 'value_model' else 'actor', count=self._ppo_actual_steps,
                 optimizer_state_tensor_bytes=state_bytes, scope='local optimizer state tensors; excludes allocator overhead')
        opt.register_step_post_hook(counted)
        return opt

    def update(self):
        before = getattr(self, '_ppo_actual_steps', 0)
        norm = opt_step(self)
        # single GPU smoke only: local gradient norm after clipping, before zero_grad.
        squares = [p.grad.detach().float().square().sum() for p in self.module.parameters() if p.grad is not None]
        post = float(torch.stack(squares).sum().sqrt()) if squares else None
        emit('gradient_step', role='critic' if self.model_config.model_type == 'value_model' else 'actor', preclip_norm=norm if math_isfinite(norm) else None,
             preclip_finite=math_isfinite(norm), postclip_local_norm=post if post is None or math_isfinite(post) else None,
             optimizer_called=self._ppo_actual_steps > before, resources=_resource_snapshot())
        return norm

    def remove(self, paths):
        for path in [paths] if isinstance(paths, str) else paths:
            retention_path(path, os.environ['AGL_PPO_RUN_DIR'])
        return delete(self, paths)

    model_utils.load_valuehead_model = valuehead
    FSDPEngine._build_module, FSDPEngine._build_optimizer, FSDPEngine.optimizer_step = module, make_optimizer, update
    BaseCheckpointManager.remove_previous_save_local_path = remove


def math_isfinite(value):
    import math
    return math.isfinite(value)


def register_in_worker():
    global _INSTALLED
    if _INSTALLED:
        return
    source = source_identity()
    from agentlightning.verl.per_rollout_loss import register_in_worker as upstream_hook
    upstream_hook()
    from agentlightning.verl import trainer, rollout_adapter
    _instrument_adapter(rollout_adapter)
    _instrument_trainer(trainer)
    _instrument_engine()
    _INSTALLED = True
    emit('overlay_registered', source=source, algorithm=contract.PPO, protocol=contract.PROTOCOL)
