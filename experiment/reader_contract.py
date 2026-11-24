"""Reader execution identities. Pure stdlib; never imports a cloud or ML client.

The historical matched contract remains closed. Only the scientific constants
and seed schedule are shared; reader paths and run identities are reconstructed.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess

try:
    from . import matched_contract as historical, reader_modes as modes
except ImportError:
    import matched_contract as historical
    import reader_modes as modes

SCHEMA = 'reader-task-v2'
LEGACY_SCHEMA = 'reader-reference-v1'

def is_reader(plan):
    return plan.get('schema') in (SCHEMA, LEGACY_SCHEMA)

PROTOCOL = historical.PROTOCOL
ALGORITHMS = historical.ALGORITHMS
BASE_REVISION = historical.BASE_REVISION
MODEL = 'meta-llama/Llama-3.2-3B-Instruct'
ROOT = Path(__file__).resolve().parents[1]
digest, file_hash = historical.digest, historical.file_hash
RESOURCES = {'gpu': 'H100!', 'cpu': 16, 'memory_mib': 196608}
PROTECTED = ('restored', 'phase2', 'models', 'data-small', 'runs', 'exports')


# Dispatch deadlines are shortened, never extended, to allow for cross-host clock uncertainty.
# Provider timeouts and the existing worker admission checks remain independent hard bounds.
DISPATCH_CLOCK_RESERVE_SECONDS = 30


def dispatch_deadline(limit, issued_at):
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (limit, issued_at)) or limit <= DISPATCH_CLOCK_RESERVE_SECONDS:
        raise ValueError('Invalid reader dispatch deadline bounds')
    return issued_at + limit - DISPATCH_CLOCK_RESERVE_SECONDS


def exact(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys.split()):
        raise ValueError(f'Incomplete or unknown {label} fields')


def slug(value):
    if not isinstance(value, str) or not re.fullmatch('[a-z0-9][a-z0-9-]{0,47}', value):
        raise ValueError('Expected an explicit lowercase resource/namespace slug')
    if value.startswith(('cs329z', 'agl-explore', 'agl-matched')) or value == 'agent-lightning-credentials':
        raise ValueError('Protected historical resource/identity')
    return value


def path(value):
    if (not isinstance(value, str) or not value or len(value) > 180
            or not re.fullmatch('[a-zA-Z0-9_./-]+', value)
            or PurePosixPath(value).is_absolute() or any(p in ('', '.', '..') for p in value.split('/'))
            or value.split('/')[0] in PROTECTED):
        raise ValueError('Unsafe or protected Volume-relative path')
    return PurePosixPath(value)


def validate_profile(p):
    exact(p, 'schema fixture modal paths experiment attempt wandb model task algorithms resources bounds'
          + (' purpose' if 'purpose' in p else ''), 'profile')
    if 'purpose' in p and (p['purpose'], p['task']) not in (
            (modes.COLLECTION_ONLY, 'asset-status-v1'), (modes.CALIBRATION, 'recovery-readiness-v1')):
        raise ValueError('Only the bounded asset collection-only development purpose is supported, '
                         'or development-calibration for recovery-readiness-v1')
    if p.get('task') == 'recovery-readiness-v1' and p.get('purpose') != modes.CALIBRATION:
        # No training length, checkpoint or final evaluation is qualified for this task.
        raise ValueError('recovery-readiness-v1 supports only the development-calibration purpose')
    if p['schema'] != SCHEMA or type(p['fixture']) is not bool:
        raise ValueError('Unsupported reader schema/fixture label; migrate profile to reader-task-v2 and regenerate plans')
    exact(p['modal'], 'profile workspace environment app registry volume secret', 'Modal')
    for value in p['modal'].values():
        slug(value)
    if len({p['modal'][k] for k in ('app', 'registry', 'volume', 'secret')}) != 4:
        raise ValueError('Use distinct explicit resource names')
    exact(p['paths'], 'model data staging output', 'paths')
    paths = [path(v) for v in p['paths'].values()]
    if any(a == b or a in b.parents or b in a.parents for i, a in enumerate(paths) for b in paths[i+1:]):
        raise ValueError('Prepared/staging/output paths overlap')
    slug(p['experiment'])
    if type(p['attempt']) is not int or not 1 <= p['attempt'] <= 99:
        raise ValueError('Invalid attempt')
    exact(p['wandb'], 'entity project', 'W&B')
    for v in p['wandb'].values():
        slug(v)
    if p['model'] != {'id': MODEL, 'revision': BASE_REVISION, 'profile': 'llama3b', 'parser': 'llama3_json'}:
        raise ValueError('Unsupported immutable model/tokenizer/parser/critic combination')
    if p['task'] not in modes.TASKS or p['algorithms'] != list(ALGORITHMS) or p['resources'] != RESOURCES:
        raise ValueError('Only registered tasks, both matched algorithms and the continuity worker shape are supported')
    b = p['bounds']
    exact(b, 'stage_seconds startup_seconds preparation_seconds preflight_seconds concurrency '
          'worker_hourly_usd supervisor_hourly_usd preparation_budget_usd training_budget_usd '
          'build_budget_usd storage_gib retention_days storage_budget_usd collection_gib total_budget_usd', 'bounds')
    for k, v in b.items():
        if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
            raise ValueError(f'Invalid positive finite bound: {k}')
    for k, lo, hi in (('stage_seconds', 300, 7200), ('startup_seconds', 60, 1800),
                      ('preparation_seconds', 300, 3600), ('preflight_seconds', 60, 900),
                      ('storage_gib', 100, 300), ('retention_days', 1, 7), ('collection_gib', 1, 5)):
        if type(b[k]) is not int or not lo <= b[k] <= hi:
            raise ValueError(f'Out of bounds: {k}')
    if b['concurrency'] != 1 or type(b['concurrency']) is not int or b['startup_seconds'] >= b['stage_seconds']:
        raise ValueError('Exactly one GPU lane; startup must fit stage deadline')
    if b['worker_hourly_usd'] < 7 or b['supervisor_hourly_usd'] < .07:
        raise ValueError('Rate ceilings below the reviewed conservative minimum; re-review pricing')
    # Supervisor uses the worker ceiling in the existing common supervisor. Six stages
    # per qualification; development calibration schedules eight.
    stages = 8 if p.get('purpose') == modes.CALIBRATION else 6
    compute = (2*stages*b['stage_seconds']+600)*b['worker_hourly_usd']/3600
    if (b['training_budget_usd'] < compute or b['preparation_budget_usd'] < 3
            or b['build_budget_usd'] < 5 or b['storage_budget_usd'] < b['storage_gib']*.09*b['retention_days']/30
            or b['total_budget_usd'] < sum(b[k] for k in ('training_budget_usd', 'preparation_budget_usd', 'build_budget_usd', 'storage_budget_usd')) + b['collection_gib']*.04):
        raise ValueError('Budget does not cover bounded compute/build/preparation/storage/collection')


def source_identity(root=ROOT, clean=False):
    try:
        from .reader_source import identity
    except ImportError:
        from reader_source import identity
    return identity(root, file_hash, clean)


def packaged(files):
    return {p: h for p, h in files.items() if (p.startswith('experiment/') and Path(p).suffix in ('.py', '.json', '.jinja', '.txt'))
            or p in ('reader_app.py', 'modal_image.py', 'modal_build.py', 'modal_reader_build.py', 'validation/compatibility.py',
                     'scripts/install_modal_training.sh', 'scripts/finalize_modal_training.sh')}


def task_module(task):
    """The synthetic task implementation; calculator has none. Unknown tasks fail closed."""
    from importlib import import_module
    if task not in modes.TASKS[1:]:
        raise ValueError('Unknown task or task without a synthetic implementation')
    name = {'asset-status-v1': 'asset_task', 'recovery-readiness-v1': 'recovery_task'}[task]
    return import_module('.'+name, __package__) if __package__ else import_module(name)


def task_identity(source, task="calculator-v1"):
    if task != 'calculator-v1':
        return task_module(task).identity(source)
    return {'name': 'calculator-v1', 'visible_fields': ['question'], 'label_fields': ['result'],
            'data_mapping': {'QUESTION': 'input.question', 'RESULT': 'input.result',
                             'AGL_CASE_ID': 'input.case_id', 'AGL_DATA_SPLIT': 'input.split', 'EVAL_SPLIT': 'input.split'},
            'agent_class': 'observed_calc_agent.Agent', 'cwd': 'examples/calc_x',
            'mcp': {'command': 'training-python', 'args': ['-m', 'mcp_server_calculator'], 'environment': 'inherited by existing harness'},
            'output': 'existing ### ANSWER extractor', 'verifier': 'eval_utils.scalar_are_results_same; tolerance 0.01',
            'verifier_source': {'commit': 'cbc56e41e0eab643aa71d195e92e5bba652fa683',
                                'path': 'examples/calc_x/eval_utils.py'},
            'terminal_policy': 'binary reward only after complete response; errors propagate without reward',
            'tracing_reward_cleanup': 'existing observed_calc_agent.Agent and launch.run_services',
            'harness_sha256': source['files']['experiment/observed_calc_agent.py'],
            'tracing_sha256': source['files']['experiment/observed_runtime.py'],
            'entrypoint_lock_sha256': source['files']['experiment/ppo_source_lock.json'],
            'launcher_sha256': source['files']['experiment/launch.py']}


def make_plan(profile, source, assets=None):
    validate_profile(profile)
    p = copy.deepcopy(profile)
    development = p.get('purpose') == modes.COLLECTION_ONLY
    calibration = p.get('purpose') == modes.CALIBRATION
    if assets is not None:
        try:
            from .reader_assets import validate_manifest
        except ImportError:
            from reader_assets import validate_manifest
        validate_manifest(assets, p['task'])
    lock = json.loads((Path(__file__).parent/'ppo_source_lock.json').read_text())
    upstream = {k: lock[k] for k in ('agentlightning_commit', 'verl_source_commit', 'verl_wheel', 'calc_example_sha256')}
    root = f"{p['paths']['output']}/{p['experiment']}/a{p['attempt']:02d}"
    plan = {'schema': SCHEMA, 'protocol': PROTOCOL, 'profile': p, 'source': source, 'upstream': upstream,
            'configuration_sha256': digest(p), 'packaged_sha256': digest(source['runtime_files']),
            'assets': assets, 'data_manifest': assets['data'] if assets else None,
            'task_identity': task_identity(source, p['task']), 'study_id': p['experiment'],
            'phase': modes.COLLECTION_ONLY if development else modes.CALIBRATION if calibration else modes.QUALIFICATION,
            'attempt': p['attempt'], 'root': root, 'jobs': [], 'automatic_retries': 0,
            'remote_ownership': 'UNVERIFIED offline', 'availability': 'UNVERIFIED offline',
            'training_admissible': bool(assets and not p['fixture'] and not source['dirty']),
            'bounds': {'stage_seconds': p['bounds']['stage_seconds'], 'startup_seconds': p['bounds']['startup_seconds'],
                       'supervisor_seconds': 6*p['bounds']['stage_seconds']+600, 'concurrency': 1,
                       'max_gpu_seconds': 6*p['bounds']['stage_seconds'],
                       'budget_usd': p['bounds']['training_budget_usd'],
                       'all_resource_hourly_ceiling_usd': p['bounds']['worker_hourly_usd']}}
    if development:
        from importlib import import_module
        task = import_module('.asset_task', __package__) if __package__ else import_module('asset_task')
        plan['development'] = {'schema': 'asset-collection-only-v1', 'questions': 8, 'rollouts_per_question': 4,
            'case_ids': [r['case_id'] for r in task.cases()['train_smoke.parquet'][:8]],
            'optimizer_updates': 0, 'collections_per_algorithm': 1, 'temperature': 1,
            'qualification_evidence': False}
        plan['bounds'].update(supervisor_seconds=2*p['bounds']['stage_seconds']+600,
                              max_gpu_seconds=2*p['bounds']['stage_seconds'])
    if calibration:
        from importlib import import_module
        declared = (import_module('.recovery_calibration', __package__) if __package__
                    else import_module('recovery_calibration')).declaration()
        plan['calibration'] = declared
        plan['bounds'].update(supervisor_seconds=8*p['bounds']['stage_seconds']+600,
                              max_gpu_seconds=8*p['bounds']['stage_seconds'])
    for algorithm in ALGORITHMS:
        name = f"{p['experiment']}-a{p['attempt']:02d}-{algorithm}"
        job = {'job_id': name, 'algorithm': algorithm, 'run_seed': 9001, 'steps': 2,
               'pair_id': f"{p['experiment']}-a{p['attempt']:02d}",
               'seed_streams': historical.streams(9001, algorithm), 'output_root': f'{root}/{algorithm}', 'stages': []}
        for kind in modes.STAGES[plan['phase']]:
            identity = {'schema': SCHEMA, 'source': digest(source), 'profile': digest(p),
                        'assets': digest(assets), 'algorithm': algorithm, 'stage': kind}
            job['stages'].append({'name': kind, 'output': f'{job["output_root"]}/{kind}', 'wandb': {
                **p['wandb'], 'group': job['pair_id'], 'id': digest(identity)[:24], 'name': f'{name}-{kind}',
                'job_type': kind, 'resume': 'never', 'config': {**identity, 'protocol': PROTOCOL,
                    'seed_streams': job['seed_streams'], 'resources': RESOURCES, 'steps': 0 if development or calibration else 2,
                    'task': plan['task_identity'], 'model': p['model'], 'evaluation': p['task']+'-64'}}})
            if calibration:
                # Each stage carries the exact panel it may use; no final-evaluation identity.
                job['stages'][-1]['wandb']['config'].update(
                    evaluation='development-calibration', calibration=plan['calibration']['stages'][kind])
        plan['jobs'].append(job)
    return plan


def validate(plan):
    if plan.get('schema') != SCHEMA:
        raise ValueError('Old/incompatible reader plan: migrate profile to reader-task-v2 and regenerate; historical reports remain archival')
    if plan != make_plan(plan['profile'], plan['source'], plan['assets']):
        raise ValueError('Plan identities/configuration/dependencies changed; regenerate')
    source = plan['source']
    exact(source, 'commit dirty repository baseline files runtime_files', 'source')
    try:
        from .reader_source import repository, safe_path, BASELINE
    except ImportError:
        from reader_source import repository, safe_path, BASELINE
    if repository('https://github.com/'+source['repository']) != source['repository'] or source['baseline'] != BASELINE:
        raise ValueError('Source repository or canonical baseline changed')
    if not source['runtime_files'] or any(source['files'].get(n) != h for n, h in source['runtime_files'].items()):
        raise ValueError('Runtime inventory differs from admitted source')
    if not re.fullmatch('[a-f0-9]{40}', source['commit']) or type(source['dirty']) is not bool:
        raise ValueError('Invalid source identity')
    for name, sha in source['files'].items():
        safe_path(name)
        if Path(name).is_absolute() or '..' in Path(name).parts or not re.fullmatch('[a-f0-9]{64}', sha):
            raise ValueError('Unsafe source inventory')


def validate_current(plan, profile, training=False):
    validate(plan)
    if profile != plan['profile'] or source_identity(clean=True) != plan['source']:
        raise ValueError('Stale source or configuration; regenerate plan')
    if profile['fixture'] or (training and not plan['training_admissible']):
        raise ValueError('Fixture or missing actual asset manifests cannot execute')


def stage_request(plan, job, stage):
    validate(plan)
    if job not in plan['jobs'] or stage not in job['stages']:
        raise ValueError('Unowned stage')
    return {'plan': plan, 'job': job, 'stage': stage, 'plan_sha256': digest(plan)}


def stage_limit(plan, stage):
    return plan['bounds']['stage_seconds']
