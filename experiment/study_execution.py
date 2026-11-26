"""Execution binding for the immutable recovery study proposal."""
import json
from pathlib import Path

try:
    from . import matched_contract as matched, recovery_learning as learning, recovery_task as task
except ImportError:
    import matched_contract as matched
    import recovery_learning as learning
    import recovery_task as task

DECLARATION = 'docs/recovery-learning-study-v1.json'
SHA256 = 'a49003098176e66c896a14879eb691209a16bbe30d6f65bc46bf228e9c61be5e'
ANALYSIS_SOURCES = {
    'docs/PHASE4C_LEARNING_STUDY_PROPOSAL.md':'c748c7cff8b6aca8a46ab55b9dfebdbcb1edb1206b81be658a30afa96c64476f',
    'scripts/recovery_study.py':'9b9732f7063ad4949ba388a96a361207b6934f90abf3246961e57e22130e3210',
}
TRAIN_SECONDS = 14400
EVAL_SECONDS = 1800
FULL = 'three-seeds-including-final'
PILOT = 'seed-2001-development-only'
AUTHORITIES = {FULL: 'all-eleven-stages-including-final', PILOT: PILOT}


def stage_order(scope):
    if scope not in AUTHORITIES:
        raise ValueError('Unknown recovery study scope')
    seeds = (2001,) if scope == PILOT else (2001, 2002, 2003)
    stages = [('development', 'base')]
    for seed in seeds:
        stages.extend([('training', str(seed)), ('development', str(seed))])
    if scope == FULL:
        stages.extend(('final', policy) for policy in ('base', '2001', '2002', '2003'))
    return stages


def deadlines(scope):
    worker = sum(TRAIN_SECONDS if panel == 'training' else EVAL_SECONDS for panel, _ in stage_order(scope))
    return worker, worker + 600


def frozen(source):
    path = Path(__file__).resolve().parents[1]/DECLARATION
    if matched.file_hash(path) != SHA256 or source['runtime_files'].get(DECLARATION) != SHA256:
        raise ValueError('Study declaration bytes or package identity changed')
    value = json.loads(path.read_text())
    for name, sha in ANALYSIS_SOURCES.items():
        if source['files'].get(name) != sha:
            raise ValueError('Frozen analysis source changed: '+name)
    for name, sha in value['frozen_sources'].items():
        if source['runtime_files'].get('experiment/'+name) != sha:
            raise ValueError('Frozen scientific source changed: '+name)
    return value


def populate(plan):
    p = plan['profile']
    declaration = frozen(plan['source'])
    if plan['data_manifest'] and matched.digest(plan['data_manifest']) != declaration['data_manifest_sha256']:
        raise ValueError('Study asset commitment changed')
    algorithm = declaration['algorithm']
    specs = {}
    plan['study'] = {'proposal_sha256': SHA256, 'analysis': declaration['analysis'], 'stages': specs}
    scope = p['study_scope']
    worker, supervisor = deadlines(scope)
    plan['bounds'].update(supervisor_seconds=supervisor, max_gpu_seconds=worker)
    for panel, policy in stage_order(scope):
        training = panel == 'training'
        name = 'training' if training else 'baseline' if policy == 'base' else 'checkpoint-eval'
        key = panel+'-'+policy
        run_seed = int(policy) if training else declaration['evaluation']['seed']
        streams = matched.streams(run_seed, algorithm)
        batches = learning.schedule(run_seed) if training else []
        if training:
            row = next(r for r in declaration['runs'] if r['seed'] == run_seed)
            if matched.digest(batches) != row['schedule_sha256'] or streams != row['streams']:
                raise ValueError('Study loader or seed streams changed')
        elif streams != declaration['evaluation']['streams']:
            raise ValueError('Study evaluation streams changed')
        split = 'train' if training else panel
        ids = [cid for batch in batches for cid in batch] if training else [r['case_id'] for r in task.rows(panel)]
        inputs = list(declaration['eager_inputs'][panel][k] for k in ('train_file', 'val_file'))
        specs[key] = dict(train_file=inputs[0]['file'], val_file=inputs[1]['file'], inputs=inputs,
            file_hashes={v['file']:v['sha256'] for v in inputs}, case_ids=ids, batches=batches,
            split=split, route='train' if training else 'val', temperature=int(training),
            rollouts_per_question=4 if training else 1, expected_episodes=len(ids)*(4 if training else 1),
            updates=16 if training else 0, initial_step=16 if name == 'checkpoint-eval' else 0)
        identity = dict(study=SHA256, source=matched.digest(plan['source']), profile=plan['configuration_sha256'],
                        assets=matched.digest(plan['assets']), algorithm=algorithm, stage=key)
        jid = p['experiment']+'-'+key
        root = plan['root']+'/'+key
        stage = dict(name=name, key=key, policy=policy, panel=panel, output=root+'/'+name,
                     wandb={**p['wandb'], 'group':p['experiment'], 'id':matched.digest(identity)[:24],
                            'name':jid, 'job_type':name, 'resume':'never', 'config':dict(identity,
                             protocol=plan['protocol'], seed_streams=streams, resources=p['resources'],
                             steps=specs[key]['updates'], task=plan['task_identity'], model=p['model'], evaluation=split)})
        plan['jobs'].append(dict(job_id=jid, algorithm=algorithm, pair_id=p['experiment'], run_seed=run_seed,
                                 steps=16, seed_streams=streams, output_root=root, stages=[stage]))
    return plan


def training_job(request):
    policy = request['stage']['policy']
    return next(job for job in request['plan']['jobs'] if job['stages'][0]['key'] == 'training-'+policy)


def verify_final_data(directory, scope):
    """Final model stages open only world, manifest and the single declared panel."""
    directory = Path(directory)
    if scope != 'study-final' or {p.name for p in directory.iterdir()} != set(task.FILES) | {'world.json','data_manifest.json'}:
        raise ValueError('Unexpected final data inventory')
    if any(p.is_symlink() or not p.is_file() for p in directory.iterdir()):
        raise ValueError('Invalid final data entry')
    manifest = json.loads((directory/'data_manifest.json').read_text(), object_pairs_hook=task.unique_object)
    task.validate_manifest(manifest)
    name = 'final_eval.parquet'
    if (matched.digest(manifest) != learning.DATA_SHA256 or task.hash_file(directory/'world.json') != manifest['world_file_sha256']
            or task.hash_file(directory/name) != manifest['files'][name]['sha256'] or task.read_rows(directory/name) != task.rows('final')):
        raise ValueError('Final panel identity differs')
    return manifest


def gate(plan, index, count):
    """Join preceding scientific stages in dispatch order without selecting on reward."""
    try:
        from . import matched_supervisor, reader_contract as contract
    except ImportError:
        import matched_supervisor
        import reader_contract as contract
    if count not in (3, 7) or index['plan'] != plan:
        raise ValueError('Invalid study gate')
    sha = contract.digest(plan)
    previous = index['started_at']
    checkpoints, calls, containers, identities = {}, set(), set(), []
    for job in plan['jobs'][:count]:
        stage, = job['stages']
        state = index['jobs'][job['job_id']]
        receipt, = state['stages']
        result = receipt['result']
        seconds = contract.stage_limit(plan, stage)
        if (state['status'] != 'completed' or receipt['status'] != 'completed'
                or not previous <= receipt['submitted_at'] <= receipt['finished_at'] <= receipt['submitted_at']+seconds
                or receipt['finished_at']-index['started_at'] >= plan['bounds']['supervisor_seconds']
                or receipt['reservation'] != dict(seconds=seconds, usd=seconds*plan['bounds']['all_resource_hourly_ceiling_usd']/3600)
                or receipt['call_id'] != result['call_id'] or result['call_id'] in calls or result['container_id'] in containers
                or not matched_supervisor.recovery_stage_admitted(plan, job, 0, result, sha)):
            raise ValueError('Study gate lacks a bounded, admitted stage')
        previous = receipt['finished_at']
        calls.add(result['call_id'])
        containers.add(result['container_id'])
        identities.append(contract.digest(result))
        proof = result['completion']
        policy = stage['policy']
        if stage['name'] == 'training':
            checkpoints[policy] = dict(step=16, sha256=proof['checkpoint_sha256'], worker=proof['worker'])
        if stage['name'] == 'checkpoint-eval':
            trained = checkpoints[policy]
            if (proof['checkpoint_before_load_sha256'] != trained['sha256'] or proof['checkpoint_sha256'] != trained['sha256']
                    or proof['training_worker'] != trained['worker']):
                raise ValueError('Study gate lacks fresh-worker terminal checkpoint verification')
    return dict(plan_sha256=sha, preceding_receipts=identities, checkpoints=checkpoints)


def final_release(plan, index):
    if plan['profile']['study_scope'] != FULL or plan['profile']['study_authority'] != AUTHORITIES[FULL] or not plan['training_admissible']:
        raise ValueError('Missing explicit final evaluation authority')
    return {**gate(plan, index, 7), 'schema':'recovery-study-final-release-v1',
            'authority':plan['profile']['study_authority'], 'declaration_sha256':SHA256,
            'analysis':plan['study']['analysis'], 'analysis_sources':dict(ANALYSIS_SOURCES), 'base':plan['profile']['model'],
            'assets_sha256':matched.digest(plan['assets']), 'source_sha256':matched.digest(plan['source']),
            'task_sha256':matched.digest(plan['task_identity']), 'dataset_sha256':learning.DATA_SHA256,
            'final_runs':[j['stages'][0]['wandb']['id'] for j in plan['jobs'][7:]]}


def read_release(plan, control, index=None):
    if index is None:
        index = json.loads((control/'index.json').read_text())
    expected = final_release(plan, index)
    if index.get('final_release') != expected or json.loads((control/'final-release.json').read_text()) != expected:
        raise ValueError('Missing or inconsistent durable final release')
    return expected


def analyze_final(plan, tables):
    path = 'scripts/recovery_study.py'
    if matched.file_hash(Path(__file__).resolve().parents[1]/path) != ANALYSIS_SOURCES[path]:
        raise ValueError('Local analysis differs from frozen source')
    from scripts import recovery_study
    try:
        from . import recovery_calibration as cal
    except ImportError:
        import recovery_calibration as cal
    if set(tables) != set(recovery_study.POLICIES):
        raise ValueError('Missing or unexpected final policy')
    rewards = {}
    for policy, rows in tables.items():
        cal.check_table(plan['study']['stages']['final-'+policy], rows)
        if any(type(e['reward']) not in (int, float) for e in rows):
            raise ValueError('Invalid final outcome')
        rewards[policy] = {e['case_id']:[e['reward']] for e in rows}
    return {**recovery_study.analyze(rewards), 'confirmed_rewards':rewards}


def secondary(report, entry, events, metrics, integration):
    from collections import Counter, defaultdict
    try:
        from . import recovery_calibration as cal
    except ImportError:
        import recovery_calibration as cal
    rows = report['episodes']
    pairs = defaultdict(list)
    for row in rows:
        pairs[row['case_id']].append(row)
    temporal = defaultdict(list)
    if report['declared']['route'] == 'val':
        for case in task.rows(report['declared']['split']):
            if task.family(case['case_id'])[1] != 'time_sibling':
                continue
            asset, _ = task.target(case['question'])
            temporal[asset].append(pairs[case['case_id']][0]['decision_correct'])
        if any(len(values) != 2 for values in temporal.values()):
            raise ValueError('Unexpected temporal pair membership')
    assessable_pairs = [v for v in temporal.values() if None not in v]
    field_counts = {key:dict(correct=sum(e[key] is True for e in rows),
                             assessable=sum(e[key] is not None for e in rows), all=len(rows))
                    for key in ('decision_correct','evidence_correct')}
    calls = Counter(c['name'] for e in rows for c in e['sampled_calls'])
    observed = Counter()
    names = {(row['rollout_id'],call['id']):call['name'] for row in rows for call in row['sampled_calls']}
    for trace in integration['coverage']:
        for chain in trace['chains']:
            observed[names[chain['rollout_id'],chain['tool_call_id']]] += 1
    update_timing = [{ 'step': row['step'], 'seconds':{k:v for k,v in row['metrics'].items() if k.startswith('timing_s/')} }
                     for row in metrics if any(k.startswith('timing_s/') for k in row['metrics'])]
    return dict(summary=cal._summary(rows), family=cal._by_family(rows), correctness=field_counts,
                temporal_pairs=dict(both_correct=sum(all(v) for v in assessable_pairs),
                                    assessable=len(assessable_pairs), total=len(temporal)),
                typed_failure_episodes=sum(bool(e['policy_errors']) for e in rows),
                malformed_final=sum(e['format_compliant'] is False for e in rows),
                calls=dict(calls), confirmed_observations=dict(observed), update_timings=update_timing,
                informative_groups=sum(report['informative_groups']), sampled_groups=8*len(report['collections']),
                targeted_tokens=sum(e['representation']['target_tokens'] for e in report['collections']),
                stage_seconds=entry['finished_at']-entry['submitted_at'], reservation_usd=entry['reservation']['usd'],
                usage_before_credits_usd=None, timing_measurements=[
                    {k:e[k] for k in ('event','step','seconds') if k in e} for e in events if 'seconds' in e])
