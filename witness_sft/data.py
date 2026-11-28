"""Reproduce training-only wire conversations from the committed synthetic split manifest."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from experiment import recovery_learning, recovery_prompt, recovery_solver, recovery_task as task

DIRECTORY = Path(__file__).resolve().parents[1] / 'evidence/witness-sft'
SEED = 2001
QUOTAS = dict(zip(task.FAMILIES, (8, 8, 4, 4, 4, 4), strict=True))
# Pin the training bytes independently of the generated manifest.
DATASET_SHA256 = '15bcd21eefc2e8f7a8bfdc77209f6be3f333f5b67285ced35161531be9db903b'


def read_source(directory):
    source = json.loads((directory / 'source-manifest.json').read_text())
    if task.digest(source) != recovery_learning.DATA_SHA256:
        raise ValueError('Wrong recovery split manifest')
    return source


def partition(source, split):
    # rows(split) also labels the other partitions internally. Read only the
    # requested partition of the shared generator and bind its transport rows.
    name = {'train': 'train_main.parquet', 'development': 'dev_panel.parquet'}[split]
    rows = [{k: case[k] for k in ('case_id', 'split', 'question')} | {'result': task.label(case)}
            for case in task._world()['cases'][split]]
    ids = [r['case_id'] for r in rows]
    file = source['files'][name]
    excluded = {case_id for f in source['files'].values() if f['split'] != split for case_id in f['case_ids']}
    if (ids != file['case_ids'] or set(ids) & excluded or len(set(ids)) != len(ids)
            or task.digest(rows) != source['environment']['ordered_cases_sha256'][name]):
        raise ValueError('Split membership or ordered source rows differ')
    return rows


def rank(stream, row):
    return task.digest([SEED, stream, row['case_id']])


def selected(rows):
    groups = defaultdict(list)
    for row in rows:
        family, variant = task.family(row['case_id'])
        groups[family, variant].append(row)
    result = []
    for family, quota in QUOTAS.items():
        # Cycle through represented variants so boundaries and distractors
        # survive subsampling. Rank within variants without trajectory evidence.
        pools = [sorted(group, key=lambda r: rank('selection', r), reverse=True)
                 for (f, _), group in sorted(groups.items()) if f == family]
        for _ in range(quota):
            pool = pools.pop(0)
            result.append(pool.pop())
            if pool:
                pools.append(pool)
    return sorted(result, key=lambda r: rank('order', r))


def build(directory):
    source = read_source(directory)
    demos, index = [], []
    for row in selected(partition(source, 'train')):
        question = row['question']
        raw, ledger = recovery_solver.solve(question, task.call)
        answer = json.loads(raw)
        eligible = set(answer['supporting_run_ids'])
        if eligible:
            runs = json.loads(ledger[-1]['content'])['runs']
            witness = max((r for r in runs if r['run_id'] in eligible),
                          key=lambda r: (r['recovery_point_at'], r['run_id']))
            answer['supporting_run_ids'] = [witness['run_id']]
        target = task.dumps(answer)
        case = task.expected(question, row['result'])
        messages = recovery_prompt.conversation(question, [(r['tool'], r['arguments']) for r in ledger])
        observed = [m['content'] for m in messages if m['role'] == 'tool']
        if observed != [r['content'] for r in ledger] or task.reward(target, case, ledger) != 1:
            raise ValueError('Demonstration failed strict replay')
        demos.append({'messages': messages, 'target': target})
        index.append({'case_id': row['case_id'], 'family': case['family'], 'variant': case['variant'],
                      'actual': answer['decision'], 'strict_reward': 1})
    payload = ''.join(task.dumps(d) + '\n' for d in demos).encode()
    manifest = {'schema': 'witness-demonstrations-v1', 'seed': SEED, 'streams': ['selection', 'order'],
                'source_manifest_sha256': task.digest(source),
                'train_file': source['files']['train_main.parquet']['sha256'],
                'dataset_sha256': hashlib.sha256(payload).hexdigest(),
                'balance': dict(Counter(r['actual'] for r in index)), 'cases_in_training_order': index}
    return payload, manifest


def load(directory=DIRECTORY):
    read_source(directory)
    data = (directory / 'demonstrations.jsonl').read_bytes()
    manifest = json.loads((directory / 'manifest.json').read_text())
    if (hashlib.sha256(data).hexdigest() != DATASET_SHA256 or manifest['dataset_sha256'] != DATASET_SHA256
            or task.digest(manifest) != MANIFEST_SHA256):
        raise ValueError('Demonstration bytes or manifest differ from frozen commitments')
    return [json.loads(line) for line in data.splitlines()]


MANIFEST_SHA256 = '02dc7c7607201a75613a78d14e940a4b94713fc1131d6d80948575bc1bd404f2'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reproduce', type=Path, help='write reproduced files to a new directory')
    args = parser.parse_args()
    payload, manifest = build(DIRECTORY)
    if args.reproduce:
        args.reproduce.mkdir(parents=True, exist_ok=False)
        (args.reproduce / 'demonstrations.jsonl').write_bytes(payload)
        (args.reproduce / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if load() != [json.loads(line) for line in payload.splitlines()] or task.digest(manifest) != MANIFEST_SHA256:
        raise ValueError('Reproduction differs')
    print(manifest['dataset_sha256'], manifest['balance'], '32 strict rewards of one')


if __name__ == '__main__':
    main()
