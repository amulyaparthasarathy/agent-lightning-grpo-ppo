"""asset-status-v1: deterministic synthetic inputs and an independent terminal verifier.

QUESTION/RESULT are a deliberately shared transport schema, not calculator logic.
Only QUESTION (a request string) enters the agent prompt. RESULT is verifier-only.
"""
from pathlib import Path
import hashlib
import json
import re

TASK = 'asset-status-v1'
SEED = 'asset-status-v1/2026-11-20/329'
SERVICE = ('operational', 'degraded', 'offline')
BACKUP = ('healthy', 'stale', 'failed')
TOOL_SCHEMA = {'type': 'object', 'properties': {'asset_id': {'type': 'string', 'pattern': '^asset-[0-9]{3}$'}},
               'required': ['asset_id'], 'additionalProperties': False}
PROMPT = ('Use lookup_asset to look up the requested asset. Return exactly one JSON object with only '
          'asset_id, service_state, backup_health, all strings. No markdown or other text. '
          'service_state must be operational, degraded, or offline; backup_health must be healthy, stale, or failed.')
FILES = {'train_main.parquet': 256, 'eval_main.parquet': 64, 'eval_main.jsonl': 64,
         'train_smoke.parquet': 16, 'eval_smoke.parquet': 8, 'eval_smoke.jsonl': 8}


def dumps(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def registry():
    # Hash sampling avoids status information in visible numeric IDs; no PRNG/version dependence.
    result = {}
    for i in range(328):
        aid = f'asset-{i:03d}'
        result[aid] = {'asset_id': aid,
            'service_state': SERVICE[int(digest([SEED, aid, 'service']), 16) % len(SERVICE)],
            'backup_health': BACKUP[int(digest([SEED, aid, 'backup']), 16) % len(BACKUP)]}
    return result


def lookup(arguments):
    if not isinstance(arguments, dict) or set(arguments) != {'asset_id'} or not isinstance(arguments['asset_id'], str):
        raise ValueError('invalid_arguments')
    aid = arguments['asset_id']
    if not re.fullmatch(r'asset-[0-9]{3}', aid):
        raise ValueError('invalid_asset_id')
    if aid not in registry():
        raise ValueError('unknown_asset')
    return registry()[aid]


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def parse(text):
    value = json.loads(text, object_pairs_hook=unique_object,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    if (not isinstance(value, dict) or set(value) != {'asset_id', 'service_state', 'backup_health'}
            or any(type(v) is not str for v in value.values())
            or not re.fullmatch(r'asset-[0-9]{3}', value['asset_id'])
            or value['service_state'] not in SERVICE or value['backup_health'] not in BACKUP):
        raise ValueError('invalid final object')
    return value


def target(question):
    match = re.fullmatch(r'Look up asset (asset-[0-9]{3}) and report its service state and backup health\.', question)
    if not match:
        raise ValueError('Unknown versioned request format')
    return match[1]


def expected(question, label):
    # Verify the privileged record against the independently regenerated environment.
    truth = lookup({'asset_id': target(question)})
    if parse(label) != truth:
        raise ValueError('Untrusted or mismatched verifier label')
    return truth


def reward(text, truth):
    try:
        return float(parse(text) == truth)
    except (ValueError, TypeError):
        return 0.0


def cases():
    assets = registry()
    order = sorted(assets, key=lambda aid: digest([SEED, 'split', aid]))
    panels = {}
    for split, ids in (('train', order[:256]), ('heldout', order[256:320]), ('smoke_eval', order[320:])):
        panels[split] = [{'question': f'Look up asset {aid} and report its service state and backup health.',
                         'result': dumps(assets[aid]), 'case_id': f'{TASK}/{split}/{aid}', 'split': split} for aid in ids]
    return {'train_main.parquet': panels['train'], 'eval_main.parquet': panels['heldout'],
            'eval_main.jsonl': panels['heldout'], 'train_smoke.parquet': panels['train'][:16],
            'eval_smoke.parquet': panels['smoke_eval'], 'eval_smoke.jsonl': panels['smoke_eval']}


def environment():
    rows = cases()
    return {'task': TASK, 'recipe': SEED, 'registry_sha256': digest(registry()),
            'ordered_cases_sha256': {n: digest(v) for n, v in rows.items()},
            'tool': {'name': 'lookup_asset', 'input_schema': TOOL_SCHEMA}, 'prompt': PROMPT,
            'output': {'fields': ['asset_id', 'service_state', 'backup_health'],
                       'service_state': list(SERVICE), 'backup_health': list(BACKUP),
                       'parser': 'whole-response-json-unique-keys-no-extra-fields-v1'},
            'reward': 'independent exact terminal correctness, binary; incomplete has no reward'}


def identity(source):
    return {'name': TASK, 'visible_fields': ['question'], 'label_fields': ['result'],
            'data_mapping': {'QUESTION': 'input.question', 'RESULT': 'input.result',
                'AGL_CASE_ID': 'input.case_id', 'AGL_DATA_SPLIT': 'input.split', 'EVAL_SPLIT': 'input.split'},
            'agent_class': 'asset_agent.Agent', 'cwd': 'examples/calc_x',
            'mcp': {'command': 'training-python', 'args': ['-I', 'asset_stdio.py'],
                    'environment': {'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1'},
                    'ownership_receipt': 'optional absolute directory argument; never a model input'},
            'local_probe': 'asset_probe.check',
            'integration_rule': 'reader-integration-v2; live chain required in all three stages per algorithm',
            'environment': environment(), 'environment_sha256': digest(environment()),
            'source_sha256': {n: source['files'][f'experiment/{n}'] for n in
                ('asset_task.py', 'asset_server.py', 'asset_stdio.py', 'asset_agent.py', 'asset_probe.py',
                 'owned_processes.py', 'observed_runtime.py')},
            'entrypoint_lock_sha256': source['files']['experiment/ppo_source_lock.json'],
            'launcher_sha256': source['files']['experiment/launch.py']}


def generate(directory):
    import pyarrow as pa
    import pyarrow.parquet as pq
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'registry.json').write_text(dumps(registry())+'\n')
    manifest = {'schema': 'asset-data-v1', 'task': TASK, 'environment': environment(),
                'environment_sha256': digest(environment()), 'registry_file_sha256': hash_file(directory/'registry.json'), 'files': {}}
    for name, rows in cases().items():
        p = directory/name
        if p.suffix == '.parquet':
            pq.write_table(pa.Table.from_pylist(rows), p, compression='NONE', version='2.6')
        else:
            p.write_text(''.join(dumps(r)+'\n' for r in rows))
        manifest['files'][name] = {'rows': len(rows), 'case_ids': [r['case_id'] for r in rows], 'sha256': hash_file(p)}
    (directory/'data_manifest.json').write_text(dumps(manifest)+'\n')
    return verify(directory)


def validate_manifest(m):
    if (set(m) != {'schema', 'task', 'environment', 'environment_sha256', 'registry_file_sha256', 'files'}
            or m['schema'] != 'asset-data-v1' or m['task'] != TASK or m['environment'] != environment()
            or m['environment_sha256'] != digest(environment()) or set(m['files']) != set(FILES)
            or m['registry_file_sha256'] != hashlib.sha256((dumps(registry())+'\n').encode()).hexdigest()):
        raise ValueError('Incompatible synthetic environment manifest')
    for name, rows in cases().items():
        item = m['files'][name]
        if (set(item) != {'rows', 'case_ids', 'sha256'} or item['rows'] != len(rows)
                or item['case_ids'] != [r['case_id'] for r in rows]
                or not re.fullmatch('[a-f0-9]{64}', item['sha256'])):
            raise ValueError('Changed ordered dataset membership or hash')


def verify(directory):
    import pyarrow.parquet as pq
    directory = Path(directory)
    m = json.loads((directory/'data_manifest.json').read_text(), object_pairs_hook=unique_object)
    validate_manifest(m)
    if {p.name for p in directory.iterdir()} != set(FILES) | {'registry.json', 'data_manifest.json'}:
        raise ValueError('Undeclared dataset inputs')
    for p in directory.iterdir():
        if p.is_symlink() or not p.is_file():
            raise ValueError('Dataset must contain regular files')
    if hash_file(directory/'registry.json') != m['registry_file_sha256']:
        raise ValueError('Changed registry bytes')
    for name, expected_rows in cases().items():
        p = directory/name
        if hash_file(p) != m['files'][name]['sha256']:
            raise ValueError('Changed dataset bytes')
        rows = pq.read_table(p).to_pylist() if p.suffix == '.parquet' else [json.loads(s, object_pairs_hook=unique_object) for s in p.read_text().splitlines()]
        if rows != expected_rows:
            raise ValueError('Changed cases/labels/order, duplicate records, or asset split leakage')
    return m
