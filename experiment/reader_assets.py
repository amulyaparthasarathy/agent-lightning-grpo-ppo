"""Acquire and verify fresh reader assets in the separate training-image venv.

Manifest validation is stdlib-only. Reading actual Parquet rows requires PyArrow;
acquisition additionally requires the image's HF/gdown dependencies. No operation
here runs on import. Historical locks and preparation restrictions are unchanged.
"""
from __future__ import annotations
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

try:
    from . import reader_contract as c, prepare_model
except ImportError:
    import reader_contract as c
    import prepare_model

BUNDLE = Path(__file__).parent
DATA_ID = '1FQMyKLLd6hP9dw9rfZn1EZOWNvKaDsqw'
DATA_FILES = {'train_main.parquet': 256, 'eval_main.parquet': 64, 'eval_main.jsonl': 64,
              'train_smoke.parquet': 16, 'eval_smoke.parquet': 8, 'eval_smoke.jsonl': 8}


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def model_lock():
    return read(BUNDLE/'ppo_source_lock.json')['prepared_assets']


def verify_model(directory):
    directory = Path(directory)
    lock = model_lock()
    for name, item in lock.items():
        p = directory/name
        if p.is_symlink() or not p.is_file() or p.stat().st_size != item['bytes'] or c.file_hash(p) != item['sha256']:
            raise ValueError(f'Model content differs from immutable reference: {name}')
    weights = sorted(n for n in lock if n.endswith('.safetensors'))
    if sorted(str(p.relative_to(directory)) for p in directory.rglob('*.safetensors')) != weights:
        raise ValueError('Missing or unexpected model weight files')
    index = read(directory/'model.safetensors.index.json')
    if set(index['weight_map'].values()) != set(weights):
        raise ValueError('Incomplete model shard inventory')
    cfg, tokenizer = read(directory/'config.json'), read(directory/'tokenizer_config.json')
    if (cfg['architectures'] != ['LlamaForCausalLM'] or cfg['model_type'] != 'llama'
            or cfg['bos_token_id'] != 128000
            or tokenizer['chat_template'] != prepare_model.effective_template()
            or (directory/'chat_template.jinja').read_text() != tokenizer['chat_template']):
        raise ValueError('Architecture/token IDs/tokenizer/template mismatch')
    manifest = read(directory/'model_manifest.json')
    if (manifest['repo_id'] != c.MODEL or manifest['revision'] != c.BASE_REVISION
            or manifest['profile'] != 'llama3b' or manifest['weights_files'] != weights
            or manifest['prepared_tokenizer_config_sha256'] != lock['tokenizer_config.json']['sha256']
            or manifest['template_sha256'] != lock['chat_template.jinja']['sha256']):
        raise ValueError('Prepared model identity mismatch')
    return {'id': c.MODEL, 'revision': c.BASE_REVISION, 'parser': 'llama3_json',
            'files': lock, 'manifest': manifest, 'manifest_sha256': c.file_hash(directory/'model_manifest.json')}


def validate_data_manifest(m):
    if m.get('schema_version') != 1 or m.get('seed') != 329 or set(m['files']) != set(DATA_FILES):
        raise ValueError('Unsupported prepared data schema/selection/files')
    for name, count in DATA_FILES.items():
        f = m['files'][name]
        if (f['rows'] != count or len(f['case_ids']) != count or len(set(f['case_ids'])) != count
                or not all(isinstance(x, str) and x for x in f['case_ids'])
                or not re.fullmatch('[a-f0-9]{64}', f['sha256'])):
            raise ValueError('Dataset count/ID/hash mismatch')
    train, test, smoke = (m['files'][n]['case_ids'] for n in ('train_main.parquet', 'eval_main.parquet', 'eval_smoke.parquet'))
    if set(train)&set(test+smoke) or set(test)&set(smoke):
        raise ValueError('Dataset split leakage')
    if (m['files']['train_smoke.parquet']['case_ids'] != train[:16]
            or m['files']['eval_main.jsonl']['case_ids'] != test
            or m['files']['eval_smoke.jsonl']['case_ids'] != smoke):
        raise ValueError('Dataset ordered panel mismatch')
    for split in ('train', 'eval'):
        if not re.fullmatch('[a-f0-9]{64}', m['sources'][split]['sha256']):
            raise ValueError('Missing acquired source hash')


def verify_data(directory, staging=None):
    import pyarrow.parquet as pq
    directory = Path(directory)
    m = read(directory/'data_manifest.json')
    validate_data_manifest(m)
    panels = {}
    for name, item in m['files'].items():
        p = directory/name
        if p.is_symlink() or c.file_hash(p) != item['sha256']:
            raise ValueError(f'Dataset bytes changed: {name}')
        rows = pq.read_table(p).to_pylist() if p.suffix == '.parquet' else [json.loads(x) for x in p.read_text().splitlines()]
        if [r['case_id'] for r in rows] != item['case_ids']:
            raise ValueError('Actual ordered case IDs differ')
        for row in rows:
            if (set(row) != {'question', 'result', 'case_id', 'split'}
                    or not all(isinstance(v, str) and v.strip() for v in row.values())
                    or row['split'] != ('train' if name.startswith('train') else 'smoke_eval' if 'smoke' in name else 'heldout')):
                raise ValueError('Invalid data schema/split; privileged solution columns forbidden')
        keys = [' '.join(r['question'].casefold().split()) for r in rows]
        if len(set(keys)) != len(keys):
            raise ValueError('Duplicate/conflicting normalized questions')
        panels[name] = rows
    train, test, smoke = (panels[n] for n in ('train_main.parquet', 'eval_main.parquet', 'eval_smoke.parquet'))
    norm = lambda rows: {' '.join(r['question'].casefold().split()) for r in rows}
    if norm(train)&norm(test+smoke) or norm(test)&norm(smoke):
        raise ValueError('Normalized question split leakage')
    if (panels['train_smoke.parquet'] != train[:16] or panels['eval_main.jsonl'] != test
            or panels['eval_smoke.jsonl'] != smoke):
        raise ValueError('Prepared labels/content differ between panels')
    if staging is not None:
        try:
            from . import prepare_data
        except ImportError:
            import prepare_data
        sources = {}
        for split, name, label in (('train', 'train.parquet', 'train'), ('eval', 'test.parquet', 'heldout')):
            path = Path(staging)/name
            if (m['sources'][split]['path'] != str(path) or path.is_symlink()
                    or c.file_hash(path) != m['sources'][split]['sha256']):
                raise ValueError('Acquired source path/content differs')
            sources[split], _ = prepare_data.read_rows(path, label)
        selected, _ = prepare_data.select_subsets(sources['train'], sources['eval'], 329)
        if any(panels[name] != rows for name, rows in selected.items()):
            raise ValueError('Prepared content differs from deterministic acquired-source selection')
    return m


def historical_match(data):
    old = read(BUNDLE/'matched_data_lock.json')
    return {'output_files': {n: data['files'].get(n) == f for n, f in old['files'].items()},
            'source_hashes': {k: data['sources'][k]['sha256'] == v['sha256'] for k, v in old.get('sources', {}).items()},
            'historical_acquisition_hashes': 'not present in committed matched data lock',
            'basis': 'Stable source/output hashes and ordered case IDs; paths/timestamps excluded',
            'claim': 'methodology rerun; historical identity never reassigned'}


def validate_manifest(m, task="calculator-v1"):
    c.exact(m, 'schema fixture model data historical_match', 'asset manifest')
    if m['schema'] != 'reader-assets-v1' or m['fixture'] is not False:
        raise ValueError('Fixture/unverified model assets cannot be launch-admissible')
    model = m['model']
    if (model['id'] != c.MODEL or model['revision'] != c.BASE_REVISION
            or model['parser'] != 'llama3_json' or model['files'] != model_lock()
            or not re.fullmatch('[a-f0-9]{64}', model['manifest_sha256'])):
        raise ValueError('Model identity does not match supported reference')
    if task != 'calculator-v1':
        # Explicit synthetic-task dispatch; an unknown task fails closed.
        c.task_module(task).validate_manifest(m['data'])
        if m['historical_match'] != synthetic_claim(task):
            raise ValueError('Incorrect historical data claim')
        return
    validate_data_manifest(m['data'])
    if m['historical_match'] != historical_match(m['data']):
        raise ValueError('Historical comparison differs')


def synthetic_claim(task):
    return {'asset-status-v1': {'claim': 'synthetic asset task; no calculator-data equivalence'},
            'recovery-readiness-v1': {'claim': 'synthetic recovery-readiness task; no calculator or asset-data equivalence'}}[task]


def inspect(model, data, staging=None, task="calculator-v1", scope='all'):
    """scope='development' is accepted only by the recovery task: a model-facing
    calibration stage verifies the development files and never opens the others."""
    if task != 'calculator-v1':
        verify = c.task_module(task).verify
        if scope != 'all' and task != 'recovery-readiness-v1':
            raise ValueError('Scoped dataset verification is not defined for this task')
        m = {'schema': 'reader-assets-v1', 'fixture': False, 'model': verify_model(model),
             'data': verify(data) if scope == 'all' else verify(data, scope=scope),
             'historical_match': synthetic_claim(task)}
        validate_manifest(m, task)
        return m
    if scope != 'all':
        raise ValueError('Scoped dataset verification is not defined for this task')
    m = {'schema': 'reader-assets-v1', 'fixture': False, 'model': verify_model(model), 'data': verify_data(data, staging)}
    m['historical_match'] = historical_match(m['data'])
    validate_manifest(m)
    return m


def owned_path(root, relative):
    c.path(relative)
    root = Path(root).resolve()
    p = root/relative
    if p.resolve() != p or not p.is_relative_to(root):
        raise ValueError('Symlink or escaped owned path')
    return p


def storage_bytes(profile, root=Path('/data')):
    """Owned file sizes, not misleading Volume filesystem free-space statistics."""
    return sum(p.stat().st_size for relative in profile['paths'].values()
               for p in owned_path(root, relative).rglob('*') if p.is_file())


def storage_exceeded(profile):
    return storage_bytes(profile) > profile['bounds']['storage_gib']*1024**3


def prepare(plan, root, run_command, log, env):
    """One new staging reservation, never overwrite or clean up partial outputs."""
    c.validate(plan)
    paths = {k: owned_path(root, v) for k, v in plan['profile']['paths'].items()}
    if any(paths[k].exists() for k in ('model', 'data', 'staging')):
        raise ValueError('Prepared/staging collision; inspect partial evidence and choose fresh paths')
    staging = paths['staging']
    staging.mkdir(parents=True, exist_ok=False)
    def run(args, name):
        run_command([sys.executable, *args], log/f'{name}.log', env, 2700)
    run([BUNDLE/'prepare_model.py', '--profile', 'llama3b', '--output-dir', staging/'model'], 'model')
    verify_model(staging/'model')
    task = plan['profile']['task']
    if task != 'calculator-v1':
        c.task_module(task).generate(staging/'data')
    else:
        archive = staging/'calc-x.zip'
        run(['-m', 'gdown', DATA_ID, '-O', archive], 'acquire-data')
        with zipfile.ZipFile(archive) as z:
            for name in ('train.parquet', 'test.parquet'):
                entries = [m for m in z.infolist() if Path(m.filename).name == name]
                if len(entries) != 1 or entries[0].file_size > 2*1024**3:
                    raise ValueError('Ambiguous/oversized calculator source archive')
                with z.open(entries[0]) as source, (staging/name).open('xb') as dest:
                    shutil.copyfileobj(source, dest)
        run([BUNDLE/'prepare_data.py', '--train-parquet', staging/'train.parquet',
             '--eval-parquet', staging/'test.parquet', '--out', staging/'data', '--seed', '329'], 'data')
    assets = inspect(staging/'model', staging/'data', staging, task)
    write_new(staging/'assets.json', assets)
    # Parent attempts are serialized by the durable remote preparation reservation.
    for k in ('model', 'data'):
        paths[k].parent.mkdir(parents=True, exist_ok=True)
        if paths[k].exists():
            raise ValueError('Destination appeared during preparation')
        (staging/k).rename(paths[k])
    return assets


if __name__ == '__main__':
    # Only the explicit CPU worker invokes this; ordinary client never needs Arrow.
    plan = read(sys.argv[1])
    from matched_container import run_command
    import os
    import signal
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    assets = prepare(plan, Path('/data'), run_command, Path(sys.argv[2]), dict(os.environ))
    write_new(Path(sys.argv[2])/'assets.json', assets)
