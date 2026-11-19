"""Versioned, dependency-free contracts for the bounded PPO plumbing smoke."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

GRPO = 'grpo'
PPO = 'ppo-mc-v1'
ALGORITHMS = (GRPO, PPO)
PROTOCOL = 'ppo-mc-smoke-legacy-retention-v1'
LEGACY_PROTOCOL = 'historical-grpo-v1'
BASE_REVISION = '0cb88a4f764b7a12671c53f0838cd831a0843b95'
CRITIC_SEED = 739391
DESCRIPTION = ('PPO clipped actor objective with Monte Carlo terminal returns and a learned '
               'token-level value baseline; implemented through VERL GAE with gamma=lambda=1.')


def identity(algorithm=GRPO):
    if algorithm not in ALGORITHMS:
        raise ValueError(f'Unknown algorithm: {algorithm}')
    return {'algorithm': algorithm, 'protocol': PROTOCOL if algorithm == PPO else LEGACY_PROTOCOL}


def validate_selection(algorithm, phase, profile, name, protocol=None):
    expected = identity(algorithm)
    if protocol and protocol != expected['protocol']:
        raise ValueError('Algorithm/protocol mismatch')
    if algorithm == PPO:
        if phase not in ('smoke', 'smoke-reload-eval') or profile != 'llama3b':
            raise ValueError('ppo-mc-v1 is restricted to llama3b smoke and smoke-reload-eval')
        if not name or not name.startswith('ppo-mc-v1-'):
            raise ValueError('PPO requires a fresh ppo-mc-v1- run name')
    return expected


def verify_identity(manifest, algorithm):
    expected = identity(algorithm)
    if 'algorithm' not in manifest:
        if algorithm == GRPO and 'protocol' not in manifest:
            return 'historical-GRPO-only compatibility; never PPO'
        raise ValueError('Missing algorithm identity; cannot load as PPO')
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError('Checkpoint algorithm/protocol mismatch')
    return 'explicit'


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def runtime_env_overrides(config_overrides, defaults):
    """AGL shallow-merges runtime_env, so preserve the pinned VERL environment."""
    key = 'ray_kwargs.ray_init.runtime_env.env_vars'
    requested = next(json.loads(item.split('=', 1)[1]) for item in config_overrides if item.startswith(key+'='))
    return list(config_overrides) + [key+'='+json.dumps({**defaults.get('env_vars', {}), **requested})]


def validate_loading_info(info):
    """The prepared causal LM has a backbone, no token classification head."""
    if set(info.get('missing_keys', [])) != {'score.weight', 'score.bias'}:
        raise ValueError(f'Unexpected critic missing keys: {info.get("missing_keys")}')
    if set(info.get('unexpected_keys', [])) - {'lm_head.weight'}:
        raise ValueError(f'Unexpected critic extra keys: {info.get("unexpected_keys")}')
    if info.get('mismatched_keys') or info.get('error_msgs'):
        raise ValueError('Critic loading reported mismatched weights or errors')


def overrides(model: Path):
    """No user-supplied dotlist; all PPO settings are reviewed here."""
    return {
        'algorithm.adv_estimator': 'gae', 'algorithm.gamma': 1.0, 'algorithm.lam': 1.0,
        'algorithm.enable_rollout_level_advantage': False, 'algorithm.use_kl_in_reward': False,
        'algorithm.rollout_correction.rollout_is': None, 'algorithm.rollout_correction.rollout_rs': None,
        'algorithm.rollout_correction.bypass_mode': False,
        'critic.enable': True, 'critic.strategy': 'fsdp',
        'critic.model.path': str(model), 'critic.model.tokenizer_path': str(model),
        'critic.model.trust_remote_code': False, 'critic.model.use_shm': False,
        'critic.model.enable_gradient_checkpointing': True, 'critic.model.enable_activation_offload': False,
        'critic.model.use_remove_padding': True,
        'critic.ppo_mini_batch_size': 8, 'critic.ppo_epochs': 1,
        'critic.ppo_micro_batch_size_per_gpu': 1, 'critic.forward_micro_batch_size_per_gpu': 1,
        # The pinned FSDPCriticConfig does NOT propagate the above fields to its engine.
        'critic.fsdp.micro_batch_size_per_gpu': 1, 'critic.fsdp.infer_micro_batch_size_per_gpu': 1,
        'critic.fsdp.use_dynamic_bsz': False, 'critic.use_dynamic_bsz': False,
        'critic.shuffle': False, 'critic.data_loader_seed': 42,
        'critic.optim.optimizer': 'AdamW', 'critic.optim.optimizer_impl': 'torch.optim',
        'critic.optim.lr': 1e-5, 'critic.optim.weight_decay': 0.01,
        'critic.optim.betas': [0.9, 0.999], 'critic.optim.clip_grad': 1.0,
        'critic.optim.lr_scheduler_type': 'constant', 'critic.optim.lr_warmup_steps': 0,
        'critic.optim.lr_warmup_steps_ratio': 0.0, 'critic.grad_clip': 1.0,
        'critic.cliprange_value': 0.5, 'critic.loss_agg_mode': 'token-mean',
        'critic.fsdp.param_offload': True, 'critic.fsdp.optimizer_offload': True,
        'critic.fsdp.model_dtype': 'fp32', 'critic.fsdp.dtype': 'bfloat16',
        'critic.fsdp.mixed_precision': {'param_dtype': 'bf16', 'reduce_dtype': 'fp32', 'buffer_dtype': 'fp32'},
        'critic.fsdp.seed': CRITIC_SEED,
        'critic.checkpoint.save_contents': ['model', 'optimizer', 'extra'],
        'critic.checkpoint.load_contents': ['model', 'optimizer', 'extra'],
        'trainer.critic_warmup': 0, 'trainer.max_actor_ckpt_to_keep': 2,
        'trainer.max_critic_ckpt_to_keep': 2, 'trainer.del_local_ckpt_after_load': False,
        'trainer.remove_previous_ckpt_in_save': False,
        'actor_rollout_ref.actor.optim.lr': 1e-6,
        'actor_rollout_ref.actor.use_kl_loss': False, 'actor_rollout_ref.actor.kl_loss_coef': 0.0,
        'actor_rollout_ref.actor.entropy_coeff': 0.0,
        'actor_rollout_ref.actor.clip_ratio_low': 0.2, 'actor_rollout_ref.actor.clip_ratio_high': 0.3,
        'actor_rollout_ref.actor.clip_ratio_c': 3.0,
        'actor_rollout_ref.actor.policy_loss.loss_mode': 'per_rollout_mean',
        'actor_rollout_ref.actor.ppo_epochs': 1, 'actor_rollout_ref.actor.shuffle': False,
        'actor_rollout_ref.model.trust_remote_code': False, 'data.trust_remote_code': False,
    }


def validate_resolved(config, model):
    for key, expected in overrides(Path(model)).items():
        current = config
        for part in key.split('.'):
            current = current[part]
        if current != expected:
            raise ValueError(f'PPO resolved config mismatch at {key}: {current!r}')
    if config['actor_rollout_ref']['model']['path'] != str(model):
        raise ValueError('Actor and critic must load the same prepared base')
    if config['trainer']['total_training_steps'] != 2:
        raise ValueError('PR A supports exactly two outer smoke iterations')


def validate_checkpoint(manifest, algorithm, checkpoint_name='global_step_2'):
    verify_identity(manifest, algorithm)
    if algorithm == PPO:
        if manifest.get('step') != 2 or checkpoint_name != 'global_step_2':
            raise ValueError('PPO reload requires completed step 2')
        if manifest.get('base_revision') != BASE_REVISION or manifest.get('complete') is not True:
            raise ValueError('PPO checkpoint identity or completion invalid')
        if set(manifest.get('roles', {})) != {'actor', 'critic'}:
            raise ValueError('PPO checkpoint needs actor and critic inventories')
        for role, files in manifest['roles'].items():
            if not files or not all(any(Path(p).name.startswith(prefix) for p in files)
                                    for prefix in ('model_', 'optim_', 'extra_state_')):
                raise ValueError(f'Incomplete {role} native checkpoint')
        if not manifest.get('resolved_config_sha256') or not manifest.get('source_identity'):
            raise ValueError('Checkpoint lacks immutable config/source identity')
        sources = manifest.get('bundle_source_sha256', {})
        required = {'launch.py', 'observed_calc_agent.py', 'ppo_contract.py', 'ppo_runtime.py', 'ppo_source_lock.json'}
        if not required <= set(sources):
            raise ValueError('Checkpoint lacks lab source identities')
        for filename in sources:
            if Path(filename).name != filename or filename in ('.', '..'):
                raise ValueError('Unsafe checkpoint source filename')
        lock = json.loads(Path(__file__).with_name('ppo_source_lock.json').read_text())
        if manifest['source_identity'].get('lock_sha256') != digest_json(lock):
            raise ValueError('Checkpoint upstream source identity differs from reviewed sources')


def verify_inventory(checkpoint, manifest):
    """Verify recorded byte sizes/hashes; no weight loading or deletion."""
    for role, files in manifest['roles'].items():
        actual = {str(p.relative_to(checkpoint / role)) for p in (checkpoint / role).rglob('*') if p.is_file()}
        if actual != set(files):
            raise ValueError(f'Checkpoint inventory differs: {role}')
        for relative, record in files.items():
            path = Path(relative)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Unsafe checkpoint inventory')
            target = checkpoint / role / path
            if target.is_symlink() or not target.resolve().is_relative_to((checkpoint / role).resolve()):
                raise ValueError('Checkpoint inventory escapes role directory')
            with target.open('rb') as handle:
                sha = hashlib.file_digest(handle, 'sha256').hexdigest()
            if target.stat().st_size != record['bytes'] or sha != record['sha256']:
                raise ValueError(f'Checkpoint content mismatch: {role}/{relative}')
