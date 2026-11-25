"""Validate the frozen non-executable study and analyze a complete paired reward table."""
import json
import math
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment import matched_contract as c, reader_assets, recovery_learning as learning, recovery_task as task

SEEDS = (2001, 2002, 2003)
POLICIES = ('base', *(str(s) for s in SEEDS))
DECLARATION = ROOT/'docs/recovery-learning-study-v1.json'


def proposal():
    model = reader_assets.model_lock()
    return {'schema': 'recovery-learning-study-proposal-v1', 'executable': False,
        'task': task.TASK, 'environment_sha256': task.digest(task.environment()),
        'prompt_sha256': c.digest(task.PROMPT), 'episode_policy': task.EPISODE_POLICY,
        'model_revision': c.BASE_REVISION, 'tokenizer_sha256': model['tokenizer.json']['sha256'],
        'chat_template_sha256': model['chat_template.jinja']['sha256'], 'parser': 'llama3_json',
        'data_manifest_sha256': learning.DATA_SHA256,
        'algorithm': c.ALGORITHMS[0], 'protocol': c.PROTOCOL, 'base_initialization': 'fresh for every run',
        'runs': [{'seed': s, 'streams': c.streams(s, c.ALGORITHMS[0]), 'schedule_sha256': c.digest(learning.schedule(s))}
                 for s in SEEDS], 'schedule_file': 'experiment/recovery_schedules.json',
        'updates': 16, 'questions': 8, 'samples': 4, 'training_temperature': 1,
        'terminal_checkpoint_only': 16, 'checkpoint_selection': 'all three terminal checkpoints',
        'optimization': {'estimator': 'grpo', 'group_std': 'unbiased', 'group_epsilon': 1e-6,
            'loss': 'per_rollout_mean', 'optimizer': 'torch.optim.AdamW', 'lr': 1e-6, 'betas': [0.9,0.999],
            'weight_decay': 0.01, 'scheduler': 'constant', 'warmup_ratio': 0, 'grad_clip': 1.0,
            'clip_ratio_low': 0.2, 'clip_ratio_high': 0.3, 'dual_clip': 3, 'entropy_coeff': 0,
            'reward_kl': False, 'actor_kl': False, 'ppo_epochs': 1, 'critic': False, 'minibatches_per_update': 1,
            'actor_microbatch_per_gpu': 1, 'log_prob_microbatch_per_gpu': 1, 'dynamic_batching': False,
            'rollout_correction_is': None, 'rollout_correction_rs': None, 'recompute_old_log_probs': True,
            'param_offload': True, 'optimizer_offload': True, 'gradient_checkpointing': True,
            'serving_gpu_utilization': 0.5, 'max_num_seqs': 32, 'context': task.CONTEXT_LIMITS},
        'evaluation': {'policies': list(POLICIES), 'samples': 1, 'temperature': 0, 'route': 'val',
            'streams': c.streams(9001, c.ALGORITHMS[0]), 'seed': 9001, 'development': 'dev_panel.parquet',
            'final': 'final_eval.parquet', 'sampled_final_episodes': 0, 'final_lock': 'all checkpoints and analysis choices frozen'},
        'episodes': {'training': 1536, 'development': 96, 'final': 256, 'total': 1888},
        'analysis': {'estimand': 'mean final case strict reward difference versus fresh base, per seed',
            'unit': 'four-case policy block', 'paired_policies': list(POLICIES), 'bootstrap_replicates': 10000,
            'analysis_seed': 329, 'quantile': 'linear percentile 2.5 and 97.5', 'missing': 'reject entire incomplete table',
            'interval_label': 'conditional synthetic case-mix uncertainty'}}


def validate(value):
    if c.digest(value) != c.digest(proposal()):
        raise ValueError('Study declaration differs from frozen proposal')


def blocks(split):
    grouped = {}
    for row in task.rows(split):
        asset, _ = task.target(row['question'])
        policy = task.call('get_asset', {'asset_id': asset})['policy_id']
        grouped.setdefault(policy, []).append(row['case_id'])
    result = [sorted(v) for _, v in sorted(grouped.items())]
    if len(result) != {'train': 32, 'development': 6, 'final': 16}[split] or any(len(v) != 4 for v in result):
        raise ValueError('Generator policy block membership differs')
    return result


def analyze(rewards, samples=1):
    """Each policy maps case IDs to ordered sample vectors. All vectors stay inside their cases."""
    clusters = blocks('final')
    cases = {cid for cluster in clusters for cid in cluster}
    if type(samples) != int or samples < 1 or set(rewards) != set(POLICIES):
        raise ValueError('Incomplete paired policies or invalid multiplicity')
    means = {}
    for policy, table in rewards.items():
        if set(table) != cases:
            raise ValueError('Incomplete or unexpected cases')
        if any(not isinstance(v, list) or len(v) != samples or
               any(type(x) not in (int, float) or x not in (0, 1) for x in v) for v in table.values()):
            raise ValueError('Missing or invalid strict reward')
        means[policy] = {cid: statistics.fmean(v) for cid, v in table.items()}
    differences = [[statistics.fmean(means[str(s)][cid]-means['base'][cid] for cid in cluster)
                    for s in SEEDS] for cluster in clusters]
    rng = random.Random(329)
    draws = [[] for _ in range(4)]
    for _ in range(10000):
        sample = rng.choices(differences, k=len(clusters))
        effects = [statistics.fmean(values) for values in zip(*sample)]
        for vector, effect in zip(draws, [*effects, statistics.fmean(effects)], strict=True):
            vector.append(effect)
    estimates = [statistics.fmean(col) for col in zip(*differences)]
    def interval(values):
        ordered = sorted(values)
        def quantile(p):
            i = (len(values)-1)*p
            return ordered[math.floor(i)]*(1-i%1) + ordered[math.ceil(i)]*(i%1)
        return [quantile(.025), quantile(.975)]
    return {'effects': dict(zip(POLICIES[1:], estimates)), 'seed_mean': statistics.fmean(estimates),
            'descriptive_seed_sd': statistics.stdev(estimates), 'replicates': 10000,
            'conditional synthetic case-mix uncertainty': dict(zip((*POLICIES[1:], 'mean'), map(interval, draws)))}


if __name__ == '__main__':
    validate(json.loads(DECLARATION.read_text()))
    print('Validated recovery-learning-study-proposal-v1 (1888 episodes, proposal only).')
