"""Offline manifest, bounds and identities for the separately authorized study."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

try:
    from . import matched_exploratory as exploratory
except ImportError:
    import matched_exploratory as exploratory

PROTOCOL = 'matched-rollout-mean-v2'
ALGORITHMS = ('grpo-matched-v2', 'ppo-mc-matched-v2')
VOLUME = 'cs329z-agent-lightning-working-stanford-v2'
WORKSPACE = 'amulyasp'
STAGES = ('baseline', 'training', 'checkpoint-eval')
BASE_REVISION = '0cb88a4f764b7a12671c53f0838cd831a0843b95'
ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def seed(*parts):
    return int(digest(parts)[:8], 16) % (2**31-1)


def streams(run_seed, algorithm):
    names = ('python', 'numpy', 'torch_cpu', 'torch_cuda', 'actor_sampling', 'actor_optimizer', 'row_batching', 'critic_init')
    result = {n: seed(PROTOCOL, run_seed, algorithm, n) for n in names}
    result['paired_question_order'] = seed(PROTOCOL, run_seed, 'paired_question_order')
    if algorithm == ALGORITHMS[0]:
        result['critic_init'] = None
    return result


def source_identity(root=ROOT, clean=False):
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True).strip()
    if clean and dirty:
        raise ValueError('Launch requires committed clean source; regenerate the plan after committing')
    paths = ['matched_app.py', 'modal_app.py', 'modal_image.py', 'modal_build.py', 'scripts/phase2_study.py',
             'docs/PHASE2_MATCHED_STUDY_CONTRACT.md', 'scripts/install_modal_training.sh',
             'scripts/finalize_modal_training.sh', 'scripts/phase2_probe.py', 'scripts/phase2_evaluation.py', 'admission_app.py',
             'docs/PHASE2_TENWAY.md', 'docs/PHASE2_EXPLORATORY.md']
    paths += [str(p.relative_to(root)) for p in (root/'experiment').rglob('*')
              if p.is_file() and p.suffix in ('.py', '.json', '.jinja', '.txt') and '__pycache__' not in p.parts]
    return {'commit': head, 'dirty': bool(dirty), 'files': {p: file_hash(root/p) for p in sorted(paths)}}


def make_plan(study, phase, attempt, stage_seconds, startup_seconds, budget_usd, hourly_ceiling, data_manifest, source, concurrency=None):
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,23}', study) or not 1 <= attempt <= 99:
        raise ValueError('Invalid study/attempt')
    if phase not in ('qualification', 'pilot', 'campaign', *exploratory.PHASES):
        raise ValueError('Unknown study phase')
    steps = {**{'qualification': 2, 'pilot': 20, 'campaign': 60}, **exploratory.STEPS}[phase]
    seeds = exploratory.SEEDS[phase] if phase in exploratory.PHASES else list(range(1001, 1006)) if phase == 'campaign' else [9001]
    requested_concurrency = concurrency
    concurrency = concurrency if concurrency is not None else (10 if phase == 'campaign' else 2)
    if phase in exploratory.PHASES:
        expected = exploratory.bounds(phase)
        if requested_concurrency is not None and requested_concurrency != expected['concurrency']:
            raise ValueError('Exploratory concurrency is frozen')
        if (stage_seconds, startup_seconds, budget_usd, hourly_ceiling) != (expected['stage_seconds'], expected['startup_seconds'], expected['budget_usd'],10):
            raise ValueError('Use frozen exploratory stage limits and allocations')
        concurrency = expected['concurrency']
    jobs = []
    for run_seed, algorithm in ((s, a) for s in seeds for a in ALGORITHMS):
        jid = f'{study}-{phase}-{algorithm}-s{run_seed}-a{attempt:02d}'
        root = f'phase2/studies/{study}/{phase}/{jid}'
        job = {'job_id': jid, 'algorithm': algorithm, 'run_seed': run_seed, 'steps': steps,
               'pair_id': f'{study}-{phase}-s{run_seed}', 'seed_streams': streams(run_seed, algorithm), 'output_root': root}
        job['stages'] = []
        for kind in STAGES:
            name = f'{jid}-{kind}'
            job['stages'].append({'name': kind, 'output': f'{root}/{kind}', 'wandb': {
                'entity': 'amulyap-stanford-university', 'project': 'cs329z-agent-lightning',
                'group': f'{study}-{phase}', 'id': digest(name)[:24], 'name': name,
                'job_type': kind, 'resume': 'never', 'config': {
                    'pair_id': job['pair_id'], 'algorithm': algorithm, 'stage': kind, 'run_seed': run_seed, 'seed_streams': job['seed_streams'],
                    'protocol': PROTOCOL, 'protocol_sha256': source['files']['docs/PHASE2_MATCHED_STUDY_CONTRACT.md'],
                    'commit': source['commit'], 'source_sha256': digest(source),
                    'data_manifest_sha256': digest(data_manifest), 'steps': steps,
                    'resources': {'gpu': 'H100!', 'cpu': 16, 'memory_mib': 196608},
                    'evaluation': 'heldout-256' if data_manifest.get('holdout') else 'exploratory-legacy-64', 'dataset_selection_seed': 329}}})
        jobs.append(job)
    plan = {'schema': PROTOCOL, 'study_id': study, 'phase': phase, 'attempt': attempt,
            'workspace': WORKSPACE, 'environment': 'main', 'volume': VOLUME,
            'jobs': jobs, 'source': source, 'data_manifest': data_manifest, 'model_revision': BASE_REVISION,
            'bounds': {'stage_seconds': stage_seconds, 'startup_seconds': startup_seconds,
                       'supervisor_seconds': 3 * math.ceil(len(jobs)/max(1, concurrency)) * stage_seconds + 600, 'concurrency': concurrency,
                       'max_gpu_seconds': 3 * len(jobs) * stage_seconds, 'budget_usd': budget_usd,
                       'all_resource_hourly_ceiling_usd': hourly_ceiling},
            'automatic_retries': 0, 'campaign_chaining': False,
            'qualification_receipt': None,
            'evaluation': 'heldout-256' if data_manifest.get('holdout') else 'exploratory-legacy-64'}
    if phase in exploratory.PHASES:
        plan.update(study_protocol=exploratory.PROTOCOL, funding=exploratory.funding(), bounds=exploratory.bounds(phase))
        for job in jobs:
            for stage in job['stages']:
                stage['wandb']['config'].update(study_protocol=exploratory.PROTOCOL,
                    stage_timeout_seconds=plan['bounds']['stage_seconds_by_name'][stage['name']],
                    checkpoint_policy=plan['funding']['checkpoint_policy'])
    validate(plan)
    return plan


def validate(plan):
    b = plan['bounds']
    special = exploratory.is_exploratory(plan)
    if special:
        exploratory.validate(plan)
    if not special and (not isinstance(b['stage_seconds'], int) or not 300 <= b['stage_seconds'] <= 14400
            or not isinstance(b['startup_seconds'], int) or not 60 <= b['startup_seconds'] <= min(1800, b['stage_seconds'])
            or type(b['concurrency']) is not int or not 1 <= b['concurrency'] <= 10
            or b['max_gpu_seconds'] != 3*len(plan['jobs'])*b['stage_seconds']
            or b['supervisor_seconds'] != 3*math.ceil(len(plan['jobs'])/b['concurrency'])*b['stage_seconds']+600
            or b['supervisor_seconds'] > 86400):
        raise ValueError('Invalid absolute resource/lifetime bounds')
    cost, rate = b['budget_usd'], b['all_resource_hourly_ceiling_usd']
    if (not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0 for v in (cost, rate))
            or (rate * b['max_gpu_seconds'] + b['supervisor_seconds']*(b['supervisor_hourly_ceiling_usd'] if special else rate))/3600 > cost):
        raise ValueError('Budget must cover the conservative all-resource hourly ceiling, including supervisor')
    # Reconstruct every identity/path/seed/resource field rather than accepting arbitrary JSON overrides.
    if plan['schema'] != PROTOCOL or plan['phase'] not in ('qualification', 'pilot', 'campaign', *exploratory.PHASES):
        raise ValueError('Unknown executable study protocol')
    if (plan['workspace'], plan['environment'], plan['volume']) != (WORKSPACE, 'main', VOLUME):
        raise ValueError('Wrong Stanford binding')
    if plan['model_revision'] != BASE_REVISION:
        raise ValueError('Wrong base revision')
    for relative, sha in plan['source']['files'].items():
        if Path(relative).is_absolute() or '..' in Path(relative).parts or not re.fullmatch('[a-f0-9]{64}', sha):
            raise ValueError('Unsafe or unhashed source identity')
    if plan['automatic_retries'] != 0 or plan['campaign_chaining'] is not False:
        raise ValueError('Retries/chaining forbidden')
    seeds = exploratory.SEEDS[plan['phase']] if special else list(range(1001, 1006)) if plan['phase'] == 'campaign' else [9001]
    if [(j['run_seed'], j['algorithm']) for j in plan['jobs']] != [(s,a) for s in seeds for a in ALGORITHMS]:
        raise ValueError('Requires exact matched seed pairs')
    paths, wbids = set(), set()
    for job in plan['jobs']:
        run_seed = job['run_seed']
        jid = f"{plan['study_id']}-{plan['phase']}-{job['algorithm']}-s{run_seed}-a{plan['attempt']:02d}"
        if (not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,23}', plan['study_id']) or not 1 <= plan['attempt'] <= 99
                or job['job_id'] != jid or job['pair_id'] != f"{plan['study_id']}-{plan['phase']}-s{run_seed}" or job['run_seed'] not in seeds or job['steps'] != {**{'qualification':2, 'pilot':20, 'campaign':60}, **exploratory.STEPS}[plan['phase']]
                or job['seed_streams'] != streams(run_seed, job['algorithm'])
                or job['output_root'] != f"phase2/studies/{plan['study_id']}/{plan['phase']}/{jid}"):
            raise ValueError('Invalid job identity/seed/path/steps')
        if [s['name'] for s in job['stages']] != list(STAGES):
            raise ValueError('Stage dependencies changed')
        for stage in job['stages']:
            w = stage['wandb']; name = f"{jid}-{stage['name']}"
            if (stage['output'] != f"{job['output_root']}/{stage['name']}" or stage['output'] in paths
                    or w['id'] != digest(name)[:24] or w['id'] in wbids or w['name'] != name
                    or w['group'] != f"{plan['study_id']}-{plan['phase']}" or w['resume'] != 'never'
                    or w['job_type'] != stage['name'] or w['entity'] != 'amulyap-stanford-university'
                    or w['project'] != 'cs329z-agent-lightning'):
                raise ValueError('Stage/W&B collision or identity drift')
            expected_config = {
                'pair_id': job['pair_id'], 'algorithm': job['algorithm'], 'stage': stage['name'],
                'run_seed': run_seed, 'seed_streams': job['seed_streams'], 'protocol': PROTOCOL,
                'protocol_sha256': plan['source']['files']['docs/PHASE2_MATCHED_STUDY_CONTRACT.md'],
                'commit': plan['source']['commit'], 'source_sha256': digest(plan['source']),
                'data_manifest_sha256': digest(plan['data_manifest']), 'steps': job['steps'],
                'resources': {'gpu': 'H100!', 'cpu': 16, 'memory_mib': 196608}, 'evaluation': plan['evaluation'], 'dataset_selection_seed': 329}
            if special:
                expected_config.update(study_protocol=exploratory.PROTOCOL,
                    stage_timeout_seconds=b['stage_seconds_by_name'][stage['name']],
                    checkpoint_policy=plan['funding']['checkpoint_policy'])
            if w['config'] != expected_config:
                raise ValueError('Remote W&B provenance/config drift')
            paths.add(stage['output']); wbids.add(w['id'])
    files = plan['data_manifest']['files']
    frozen = json.loads(Path(__file__).with_name('matched_data_lock.json').read_text())['files']
    if files.get('train_main.parquet') != frozen['train_main.parquet']:
        raise ValueError('Training data changed')
    train, test = files['train_main.parquet']['case_ids'], files['eval_main.parquet']['case_ids']
    heldout = plan['data_manifest'].get('holdout')
    if heldout:
        if plan['phase'] != 'campaign':
            raise ValueError('Final holdout is forbidden during qualification/pilot')
        try:
            from .matched_holdout import validate_manifest
        except ImportError:
            from matched_holdout import validate_manifest
        validate_manifest(plan['data_manifest'], frozen)
        expected_label = 'heldout-256'
    else:
        if any(files.get(name) != expected for name, expected in frozen.items()):
            raise ValueError('Legacy data changed')
        expected_label = 'exploratory-legacy-64'
    if plan['evaluation'] != expected_label:
        raise ValueError('Evaluation label drift')
    if len(train) != 256 or len(set(train)) != 256 or len(set(test)) != len(test) or set(train)&set(test):
        raise ValueError('Training/evaluation ID leakage')
    if files['eval_main.jsonl']['case_ids'] != test:
        raise ValueError('Evaluation parquet/JSON panel mismatch')
    for filename in ('train_main.parquet', 'eval_main.parquet', 'eval_main.jsonl'):
        if not re.fullmatch('[a-f0-9]{64}', files[filename]['sha256']):
            raise ValueError('Missing frozen data hash')


def stage_request(plan, job, stage):
    validate(plan)
    if job not in plan['jobs'] or stage not in job['stages']:
        raise ValueError('Unowned stage')
    return {'plan': plan, 'job': job, 'stage': stage, 'plan_sha256': digest(plan)}


def stage_limit(plan, stage):
    name = stage['name'] if isinstance(stage, dict) else stage
    if exploratory.is_exploratory(plan):
        return plan['bounds']['stage_seconds_by_name'][name]
    return plan['bounds']['stage_seconds']
