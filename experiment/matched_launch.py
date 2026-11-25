"""Remote stage entrypoint; requires frozen local assets and a validated study request."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

try:
    from . import launch, matched_contract as c, ppo_contract, summarize, reader_modes as modes
    from .request_transport import environment as request_environment
except ImportError:
    import launch
    import matched_contract as c
    import ppo_contract
    import summarize
    import reader_modes as modes
    from request_transport import environment as request_environment


def calibration(request):
    """The recovery calibration module for a calibration request, else None."""
    if modes.phase(request['plan']) != modes.CALIBRATION:
        return None
    try:
        from . import recovery_calibration
    except ImportError:
        import recovery_calibration
    return recovery_calibration

WORKER = '''import json, os, sys
from pathlib import Path
import ray
from omegaconf import OmegaConf
from ppo_runtime import source_identity
from ppo_contract import runtime_env_overrides
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from matched_runtime import seed_process
from matched_launch import validate_resolved
plan = json.loads(Path(sys.argv[1]).read_text())
source_identity()
seed_process(plan['request']['job']['seed_streams'])
sys.path.insert(0, plan['calc_dir'])
from train_calc_agent import build_config, train
kwargs = dict(model=plan['model'], agl_base_url=plan['gateway'], agl_key=plan['local_key'],
              run_name=plan['run_name'], config_overrides=runtime_env_overrides(plan['overrides'], get_ppo_ray_runtime_env()))
config = build_config(**kwargs)
resolved = OmegaConf.to_container(config, resolve=True)
validate_resolved(resolved, plan)
Path(plan['resolved_config']).write_text(json.dumps(resolved, indent=2)+'\\n')
try:
    train(train_file=plan['train_file'], val_file=plan['eval_file'], **kwargs)
finally:
    ray.shutdown()
'''


def configuration(request, model, run):
    job, stage, plan = request['job'], request['stage'], request['plan']
    s = job['seed_streams']; ppo = job['algorithm'] == c.ALGORITHMS[1]
    cfg = ppo_contract.overrides(model)
    cfg.update({
        'algorithm.adv_estimator': 'gae' if ppo else 'grpo', 'critic.enable': ppo,
        'data.seed': s['paired_question_order'], 'data.train_batch_size': 8,
        # VERL 0.8.0 RolloutConfig has no seed field. Its vLLM server merges
        # engine_kwargs after default engine arguments, preserving this stream.
        'data.shuffle': True, 'actor_rollout_ref.rollout.engine_kwargs.vllm.seed': s['actor_sampling'],
        'actor_rollout_ref.actor.fsdp_config.seed': s['actor_optimizer'],
        'actor_rollout_ref.actor.data_loader_seed': s['row_batching'],
        'critic.data_loader_seed': s['row_batching'], 'critic.fsdp.seed': s['critic_init'] or 0,
        'actor_rollout_ref.actor.fsdp_config.micro_batch_size_per_gpu': 1,
        'actor_rollout_ref.actor.use_dynamic_bsz': False,
        'actor_rollout_ref.rollout.log_prob_use_dynamic_bsz': False,
        'actor_rollout_ref.actor.fsdp_config.full_determinism': False,
        'critic.fsdp.full_determinism': False,
        'actor_rollout_ref.actor.fsdp_config.use_dynamic_bsz': False,
        'actor_rollout_ref.actor.calculate_entropy': False,
        'trainer.balance_batch': False,
        'agentlightning.trace_aggregator.level': 'trajectory',
        'trainer.total_training_steps': job['steps'],
        'trainer.val_only': modes.validation_only(plan, stage),
        'trainer.val_before_train': modes.validation_only(plan, stage),
        'trainer.test_freq': -1,
        'trainer.save_freq': job['steps'] if c.exploratory.is_exploratory(plan) else 20,  # upstream always saves the final step as well
        'trainer.max_actor_ckpt_to_keep': None, 'trainer.max_critic_ckpt_to_keep': None,
        'trainer.logger': ['wandb'], 'trainer.experiment_name': stage['wandb']['name'],
        'trainer.default_local_dir': str(run/'checkpoints'),
        'ray_kwargs.ray_init.runtime_env.worker_process_setup_hook': 'matched_runtime.register_in_worker',
        'ray_kwargs.ray_init.runtime_env.env_vars': {
            **request_environment(request, run), 'AGL_PPO_RUN_DIR': str(run),
            'PYTHONHASHSEED': str(s['python']),
            'AGL_PPO_AUDIT_DIR': str(run/'audit'), 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'},
        'lab_protocol': {'protocol': c.PROTOCOL, 'algorithm': job['algorithm'],
                         'seed_streams': s, 'comparison_qualified': False},
    })
    if request['plan'].get('schema') in ('reader-reference-v1', 'reader-task-v2'):
        cfg['trainer.project_name'] = stage['wandb']['project']
        task = plan['task_identity']
        cfg['agentlightning.local.agent_class'] = task['agent_class']
        cfg['agentlightning.local.env_map'] = task['data_mapping']
        if 'context_limits' in task:
            # Versioned, task-bound limits. Tasks that declare none keep the shared values.
            limits = task['context_limits']
            cfg['agentlightning.trace_aggregator.trajectory_max_prompt_length'] = limits['trajectory_max_prompt_length']
            cfg['agentlightning.trace_aggregator.trajectory_max_response_length'] = limits['trajectory_max_response_length']
            cfg['actor_rollout_ref.rollout.max_model_len'] = limits['max_model_len']
    if stage['name'] == 'checkpoint-eval':
        checkpoint = run.parent/'training/checkpoints'/f"global_step_{job['steps']}"
        cfg.update({'trainer.resume_mode': 'resume_path', 'trainer.resume_from_path': str(checkpoint)})
    else:
        cfg['trainer.resume_mode'] = 'disable'
    return cfg


def make_plan(request, bundle, repo, model, data, python, run, port):
    # Reuse only common launcher settings; never dispatch through the smoke selector.
    args = SimpleNamespace(algorithm='grpo', phase='main', profile='llama3b', hooks=None, protocol=None)
    plan, commands = launch.make_plan(args, bundle, repo, model, data, python, run, None, port)
    cfg = {key: json.loads(value) for key, value in (v.split('=', 1) for v in plan['overrides'])}
    cfg.update(configuration(request, model, run))
    plan.update(protocol=c.PROTOCOL, algorithm=request['job']['algorithm'], request=request,
                phase=request['stage']['name'], policy_version=c.PROTOCOL,
                overrides=[launch.dot(k, v) for k, v in cfg.items()])
    if calibration(request):
        # Both upstream dataset arguments are routed explicitly to run-local copies of the
        # declared development file. No launcher default (train_main/eval_main) remains.
        plan['train_file'], plan['eval_file'] = calibration(request).routes(run)
    elif request['plan'].get('phase') == 'collection-only':
        plan['train_file'] = str(run/'development-panel.parquet')
    return plan, commands


def validate_resolved(config, plan):
    expected = configuration(plan['request'], Path(plan['model']), Path(plan['resolved_config']).parent)
    for key, value in expected.items():
        node = config
        for part in key.split('.'):
            node = node[part]
        if key.endswith('runtime_env.env_vars'):
            if not all(node.get(k) == v for k, v in value.items()):
                raise ValueError('Ray environment drift')
        elif node != value:
            raise ValueError(f'Matched resolved configuration drift: {key}')
    if calibration(plan['request']):
        run = Path(plan['resolved_config']).parent
        if [plan['train_file'], plan['eval_file']] != list(calibration(plan['request']).routes(run)):
            raise ValueError('Calibration dataset routing changed')
        limits = plan['request']['plan']['task_identity']['context_limits']
        # launch.run_services sets AGL_MAX_TOKENS=1024 for every harness.
        if limits['max_tokens'] != 1024 or config['actor_rollout_ref']['rollout']['max_model_len'] != limits['max_model_len']:
            raise ValueError('Declared context limits differ from the launcher')
    elif plan['request']['plan'].get('schema') == 'reader-task-v2':
        paths = plan['request']['plan']['profile']['paths']
        for key, name in (('train_file', 'train_main.parquet'), ('eval_file', 'eval_main.parquet')):
            if key == 'train_file' and plan['request']['plan'].get('phase') == 'collection-only':
                if Path(plan[key]) != Path(plan['resolved_config']).parent/'development-panel.parquet':
                    raise ValueError('Development panel routing changed')
                continue
            if Path(plan[key]) != Path('/data')/paths['data']/name:
                raise ValueError('Task dataset routing changed')
    if config['actor_rollout_ref']['model']['path'] != plan['model']:
        raise ValueError('Actor base changed')


def validate_inputs(request, model, data, repo, bundle, contract=c):
    contract.validate(request['plan'])
    if request != contract.stage_request(request['plan'], request['job'], request['stage']):
        raise ValueError('Request identity mismatch')
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    if head != launch.AGL_COMMIT or dirty:
        raise ValueError('Unreviewed training checkout')
    if contract is not c:
        try:
            from .reader_assets import inspect, owned_path
        except ImportError:
            from reader_assets import inspect, owned_path
        # A calibration stage is model-facing: verify development files only.
        scope = {'scope': 'development'} if calibration(request) else {}
        if inspect(model, data, owned_path('/data', request['plan']['profile']['paths']['staging']), task=request['plan']['profile']['task'], **scope) != request['plan']['assets']:
            raise ValueError('Actual reader assets differ from exact plan manifests')
        return
    lock = json.loads((bundle/'ppo_source_lock.json').read_text())
    for name, identity in lock['prepared_assets'].items():
        if (model/name).stat().st_size != identity['bytes'] or c.file_hash(model/name) != identity['sha256']:
            raise ValueError(f'Prepared base changed: {name}')
    if json.loads((model/'model_manifest.json').read_text())['revision'] != c.BASE_REVISION:
        raise ValueError('Wrong prepared model revision')
    manifest = json.loads((data/'data_manifest.json').read_text())
    if manifest != request['plan']['data_manifest']:
        raise ValueError('Data manifest differs from plan')
    for name in ('train_main.parquet', 'eval_main.parquet', 'eval_main.jsonl'):
        if c.file_hash(data/name) != manifest['files'][name]['sha256']:
            raise ValueError('Dataset bytes differ from frozen manifest')
    # Verify IDs and normalized questions against actual parquet records, not only claimed IDs.
    import pandas as pd
    train, test = (pd.read_parquet(data/f'{part}_main.parquet').to_dict('records') for part in ('train', 'eval'))
    for records, part in ((train, 'train'), (test, 'eval')):
        if [r['case_id'] for r in records] != manifest['files'][f'{part}_main.parquet']['case_ids']:
            raise ValueError('Parquet case ID order differs from manifest')
    norm = lambda row: ' '.join(row['question'].casefold().split())
    if {norm(r) for r in train} & {norm(r) for r in test}:
        raise ValueError('Normalized training/evaluation question leakage')
    if manifest.get('holdout'):
        import hashlib
        actual = [hashlib.sha256(norm(r).encode()).hexdigest() for r in test]
        if actual != manifest['holdout']['normalized_question_sha256']:
            raise ValueError('Holdout question commitment differs')


def checkpoint_identity(path, request, run):
    roles = ('actor', 'critic') if request['job']['algorithm'] == c.ALGORITHMS[1] else ('actor',)
    inventory = {}
    for role in roles:
        files = {str(p.relative_to(path/role)): {'bytes': p.stat().st_size, 'sha256': c.file_hash(p)}
                 for p in sorted((path/role).rglob('*')) if p.is_file()}
        if not all(any(Path(n).name.startswith(prefix) for n in files) for prefix in ('model_', 'optim_', 'extra_state_')):
            raise ValueError(f'Incomplete native {role} checkpoint')
        inventory[role] = files
    return {'protocol': c.PROTOCOL, 'algorithm': request['job']['algorithm'], 'step': request['job']['steps'],
            'plan_sha256': request['plan_sha256'], 'roles': inventory,
            'resolved_config_sha256': c.file_hash(run/'resolved_config.json')}


def verify_checkpoint(path, request):
    recorded = json.loads((path/'matched_identity.json').read_text())
    actual = checkpoint_identity(path, request, path.parent.parent)
    if recorded != actual:
        raise ValueError('Checkpoint/config/algorithm identity changed')
    latest = path.parent/'latest_checkpointed_iteration.txt'
    if int(latest.read_text()) != request['job']['steps'] or path.name != f"global_step_{request['job']['steps']}":
        raise ValueError('Not the declared final native checkpoint')
    return c.digest(recorded)


def complete_stage(run, request, data):
    events = [json.loads(line) for p in (run/'audit').glob('events-*.jsonl') for line in p.read_text().splitlines()]
    if not any(e['event']=='wandb_finished' for e in events):
        raise ValueError('No remote W&B finish receipt')
    stage, job = request['stage']['name'], request['job']
    result = {'protocol': c.PROTOCOL, 'plan_sha256': request['plan_sha256'], 'stage': stage}
    terminal = [e for e in events if e['event']=='trainer_returned']
    if calibration(request):
        # Schema-specific evidence rules; never the qualification or asset-development rules.
        result.update(calibration(request).complete(run, request, events))
        mode = result['calibration']['mode']
        result['trajectories'] = summarize.summarize(run, stage, 'val' if mode == 'greedy' else 'train')[0]
        expected = result['calibration']['expected_episodes']
        if any(result['trajectories'][k] != expected for k in ('attempts_selected', 'completed', 'reward_posts_confirmed')):
            raise ValueError('Incomplete calibration trajectory evidence')
        return result
    if request['plan'].get('phase') == 'collection-only':
        from collections import Counter
        admitted = [e for e in events if e['event'] == 'development_collection_admitted']
        manifests = [e for e in events if e['event'] == 'collection_manifest']
        forbidden = {'optimizer_step', 'gradient_step', 'matched_advantages', 'native_checkpoint_saved'}
        if (len(terminal) != 1 or terminal[0].get('collection_only') is not True
                or terminal[0]['global_step'] != 1 or terminal[0].get('optimizer_updates') != 0
                or any(e['event'] in forbidden for e in events)
                or [e['step'] for e in events if e['event'] == 'weight_refresh'] != [0]
                or len(admitted) != 1 or admitted[0]['real_rollouts'] != 32
                or len(manifests) != 1 or Counter(r['case_id'] for r in manifests[0]['trajectories']) !=
                    Counter({cid: 4 for cid in request['plan']['development']['case_ids']})
                or list((run/'checkpoints').glob('global_step_*'))):
            raise ValueError('Invalid collection-only audit or unexpected optimization/checkpoint')
        result['development'] = admitted[0]
        result['optimizer_updates'] = {'actor': 0, 'critic': 0}
        result['trajectories'] = summarize.summarize(run, 'training', 'train')[0]
        if any(result['trajectories'][k] != 32 for k in ('attempts_selected', 'completed', 'reward_posts_confirmed')):
            raise ValueError('Incomplete development trajectory evidence')
        return result
    expected_step = job['steps'] if stage=='training' else job['steps'] if stage=='checkpoint-eval' else 0
    if len(terminal) != 1 or terminal[0]['global_step'] != expected_step:
        raise ValueError('Missing or incorrect terminal global step')
    if stage == 'training':
        if sorted(e['step'] for e in events if e['event']=='weight_refresh') != list(range(job['steps']+1)):
            raise ValueError('Missing or extra actor weight refresh events')
        for role in ('actor', 'critic') if job['algorithm']==c.ALGORITHMS[1] else ('actor',):
            calls = [e for e in events if e['event']=='optimizer_step' and e['role']==role]
            if len(calls) != job['steps'] or max(e['count'] for e in calls) != job['steps']:
                raise ValueError(f'Wrong actual {role} optimizer count')
        path = run/'checkpoints'/f"global_step_{job['steps']}"
        if c.exploratory.is_exploratory(request['plan']) and sorted((run/'checkpoints').glob('global_step_*')) != [path]:
            raise ValueError('Exploratory checkpoint policy requires exactly the final checkpoint; no automatic deletion')
        identity = checkpoint_identity(path, request, run)
        with (path/'matched_identity.json').open('x') as f:
            json.dump(identity, f, indent=2)
        result['checkpoint_sha256'] = verify_checkpoint(path, request)
        result['checkpoint_bytes'] = sum(f['bytes'] for role in identity['roles'].values() for f in role.values())
    else:
        if any(e['event']=='optimizer_step' for e in events):
            raise ValueError('No-update evaluation performed an optimizer step')
        if [e['step'] for e in events if e['event']=='weight_refresh'] != [expected_step]:
            raise ValueError('Evaluation did not refresh the declared checkpoint/base')
        report = summarize.summarize(run, stage, 'val')
        checked = summarize.compare(report, report, data/'eval_main.jsonl')
        if not checked['available']:
            raise ValueError(f'Incomplete evaluation: {checked["reasons"]}')
        result['evaluation'] = report[0]
        if stage == 'checkpoint-eval':
            result['checkpoint_sha256'] = verify_checkpoint(run.parent/'training/checkpoints'/f"global_step_{job['steps']}", request)
        if list((run/'checkpoints').glob('global_step_*')):
            raise ValueError('Evaluation unexpectedly saved a checkpoint')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    run = args.request.parent
    contract = c
    repo, model, data = Path('/opt/agent-lightning'), Path('/data/restored/models/llama3b'), Path('/data/restored/data-small')
    if request['plan'].get('schema') in ('reader-reference-v1', 'reader-task-v2'):
        try:
            from . import reader_contract as contract
        except ImportError:
            import reader_contract as contract
        model = Path('/data')/request['plan']['profile']['paths']['model']
        data = Path('/data')/request['plan']['profile']['paths']['data']
    elif request['plan']['data_manifest'].get('holdout'):
        data = Path('/data')/request['plan']['data_manifest']['holdout']['volume_path']
    bundle = Path(__file__).parent
    validate_inputs(request, model, data, repo, bundle, contract)
    if request['stage']['name']=='checkpoint-eval':
        verify_checkpoint(run.parent/'training/checkpoints'/f"global_step_{request['job']['steps']}", request)
    (run/'logs').mkdir()
    plan, commands = make_plan(request, bundle, repo, model, data, repo/'.venv/bin/python', run, launch.free_port())
    if calibration(request):
        if calibration(request).materialize(request, data, run, commands) != [plan['train_file'], plan['eval_file']]:
            raise ValueError('Calibration dataset routing changed')
    elif request['plan'].get('phase') == 'collection-only':
        import pyarrow.parquet as pq
        panel = pq.read_table(data/'train_smoke.parquet').slice(0, 8)
        if panel.column('case_id').to_pylist() != request['plan']['development']['case_ids']:
            raise ValueError('Development panel identity differs')
        pq.write_table(panel, plan['train_file'])
    launch.dump_json(run/'run_plan.json', plan)
    (run/'trainer_worker.py').write_text(WORKER)
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    remaining = c.stage_limit(request['plan'],request['stage'])-120
    if c.exploratory.is_exploratory(request['plan']) or contract is not c:
        remaining = float(os.environ['AGL_MATCHED_DEADLINE'])-time.time()-120
        if remaining <= 0:
            raise TimeoutError('Stage budget exhausted during prepared-asset hashing')
    code = launch.run_services(commands, plan, run, repo, bundle,
                               remaining/60)
    if code:
        raise SystemExit(code)
    completion = complete_stage(run, request, data)
    if contract is not c:
        try:
            from .reader_evidence import complete
        except ImportError:
            from reader_evidence import complete
        completion = complete(run, request, completion)
    launch.dump_json(run/'completion.json', completion)


if __name__ == '__main__':
    main()
