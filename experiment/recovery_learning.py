"""Closed recovery-learning qualification. The later study has no stage dispatcher."""
import json
from pathlib import Path

try:
    from . import matched_contract as matched, reader_modes as modes, recovery_task as task
except ImportError:
    import matched_contract as matched
    import reader_modes as modes
    import recovery_task as task

ROUTES = {'grpo': [matched.ALGORITHMS[0]], 'ppo-mc': [matched.ALGORITHMS[1]], 'paired': list(matched.ALGORITHMS)}
STAGE_SECONDS = 3600
DATA_SHA256 = 'c7d07942dda1ee81b8797ce9a59255e7f26bca712e668c18cd1a936d83d6e8ab'
TRAIN = 'train_main.parquet'
DEV = 'dev_panel.parquet'
PROBE_SHA256 = '1e8f9127fd30d6159bf64bf4095c5577c9ac13a14d6d1e9fc9f4e20a7ed7cbd4'


def schedule(seed):
    record = json.loads(Path(__file__).with_name('recovery_schedules.json').read_text())[str(seed)]
    batches = record['batches']
    expected = [r['case_id'] for r in task.rows('train')]
    ids = [cid for batch in batches for cid in batch]
    if (len(batches) != (2 if seed == 9001 else 16) or any(len(batch) != 8 for batch in batches)
            or len(set(ids)) != len(ids) or not set(ids) <= set(expected) or matched.digest(batches) != record['sha256']):
        raise ValueError('Invalid frozen loader schedule')
    return batches


def declaration(selection, manifest):
    if manifest is not None and matched.digest(manifest) != DATA_SHA256:
        raise ValueError('Recovery dataset differs from the immutable Phase 4B manifest')
    batches = schedule(9001)
    dev = task.panels()['development']['case_ids']
    stages = {}
    for name in modes.STAGES[modes.RECOVERY_LEARNING]:
        probe, training = name == 'collection-probe', name == 'training'
        sampled = probe or training
        train_file, val_file = (TRAIN if sampled else DEV), (TRAIN if probe else DEV)
        collections = batches if training else batches[:1] if probe else []
        ids = [cid for batch in collections for cid in batch] if sampled else dev
        inputs = []
        for file in (train_file, val_file):
            ordered = ids if probe else [r['case_id'] for r in task.rows('train' if file == TRAIN else 'development')]
            inputs.append({'file': 'training-probe.parquet' if probe else file, 'ordered_case_ids_sha256': matched.digest(ordered),
                           'rows': len(ordered), 'sha256': PROBE_SHA256 if probe else manifest['files'][file]['sha256'] if manifest else None})
        stages[name] = {'train_file': train_file, 'val_file': val_file,
            'file_hashes': {f: manifest['files'][f]['sha256'] if manifest else None for f in (train_file, val_file)},
            'inputs': inputs, 'case_ids': ids, 'batches': collections, 'split': 'train' if sampled else 'development',
            'route': 'train' if sampled else 'val', 'temperature': int(sampled),
            'rollouts_per_question': 4 if sampled else 1, 'expected_episodes': len(ids) * (4 if sampled else 1),
            'updates': 2 if training else 0, 'initial_step': 2 if name == 'checkpoint-eval' else 0}
    return {'schema': modes.RECOVERY_LEARNING, 'selection': selection, 'stages': stages,
            'run_seed': 9001, 'stage_seconds': STAGE_SECONDS, 'total_episodes':
                len(ROUTES[selection]) * sum(s['expected_episodes'] for s in stages.values())}


def stage_spec(request):
    if modes.phase(request['plan']) != modes.RECOVERY_LEARNING:
        raise ValueError('Expected a recovery qualification')
    return request['plan']['qualification']['stages'][request['stage']['name']]


def scope(request):
    return {'baseline': 'qual-dev', 'checkpoint-eval': 'qual-dev', 'training': 'qual-train-dev',
            'collection-probe': 'qual-train'}[request['stage']['name']]


def routes(request, data, run):
    spec = stage_spec(request)
    if request['stage']['name'] == 'collection-probe':
        return [str(run/'training-probe.parquet')] * 2
    return [str(data/spec[key]) for key in ('train_file', 'val_file')]


def materialize(request, data, run, commands):
    import pyarrow.parquet as pq
    try:
        from .recovery_calibration import GATEWAY_TEMPERATURES
    except ImportError:
        from recovery_calibration import GATEWAY_TEMPERATURES
    spec = stage_spec(request)
    if not all(commands['gateway'].count(v) == 1 for v in GATEWAY_TEMPERATURES):
        raise ValueError('Unexpected qualification gateway temperatures')
    paths = routes(request, data, run)
    for filename in dict.fromkeys((spec['train_file'], spec['val_file'])):
        path = data/filename
        if matched.file_hash(path) != spec['file_hashes'][filename]:
            raise ValueError('Qualification input bytes differ')
        table = pq.read_table(path)
        if table.to_pylist() != task.cases()[filename]:
            raise ValueError('Qualification source membership/order differs')
        if request['stage']['name'] == 'collection-probe':
            indices = {cid: i for i, cid in enumerate(table.column('case_id').to_pylist())}
            pq.write_table(table.take([indices[cid] for cid in spec['case_ids']]), paths[0])
    actual = []
    for path in paths:
        ids = pq.read_table(path, columns=['case_id']).column('case_id').to_pylist()
        actual.append({'file': Path(path).name, 'ordered_case_ids_sha256': matched.digest(ids), 'rows': len(ids),
                       'sha256': matched.file_hash(path)})
    if actual != spec['inputs']:
        raise ValueError('Materialized qualification input order or bytes differ')
    with (run/'qualification-inputs.json').open('x') as handle:
        json.dump(actual, handle)
    return paths
