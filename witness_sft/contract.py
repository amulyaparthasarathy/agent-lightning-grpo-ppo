"""Closed, offline SFT plan. No account, resource lookup or model construction."""
import base64
import subprocess
import tempfile
import time
from pathlib import Path
import re

from experiment import reader_assets, reader_contract
from . import artifact, data, evaluation, training

ROOT = Path(__file__).resolve().parents[1]
STAGES = evaluation.STAGES
LIMITS = dict(zip(STAGES, [1800, 1800, 1800], strict=True))
STARTUP = 420
# A separately reviewed, one-use authorization must bind an owner profile and
# a tested source/plan. No command-line switch, profile field or pilot balance grants it.
OWNER_ED25519_PUBLIC_PEM = None


def authorize(plan, profile, permit):
    if OWNER_ED25519_PUBLIC_PEM is None:
        raise PermissionError('No paid witness-SFT reservation')
    # A later reviewed owner key can verify a signature made AFTER the exact
    # source is committed. There is no self-reference in the plan commitment.
    body = permit['body']
    expiry = body['expires_epoch']
    if (set(body) != {'plan_sha256', 'owner_sha256', 'expires_epoch', 'ceiling_usd'}
            or body['plan_sha256'] != data.task.digest(plan) or body['owner_sha256'] != data.task.digest(profile)
            or type(expiry) != int or expiry <= time.time() or body['ceiling_usd'] != plan['ceiling_usd']):
        raise PermissionError('Expired or mismatched reservation')
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / 'owner.pem').write_text(OWNER_ED25519_PUBLIC_PEM)
        (root / 'message').write_text(data.task.dumps(body))
        (root / 'sig').write_bytes(base64.b64decode(permit['signature'], validate=True))
        subprocess.run(['openssl', 'pkeyutl', '-verify', '-pubin', '-inkey', str(root / 'owner.pem'), '-rawin',
                        '-in', str(root / 'message'), '-sigfile', str(root / 'sig')], check=True,
                       capture_output=True, timeout=10)


def make_plan(source, files):
    return {'schema': 'recovery-witness-sft-v1', 'source': source, 'files': files, 'stages': list(STAGES),
            'seed': data.SEED, 'stage_seconds': LIMITS, 'startup_seconds': STARTUP,
            'examples': 32, 'updates': training.UPDATES, 'live_episodes': [24, 0, 24],
            'automatic_retries': 0, 'replacement': False, 'ceiling_usd': 25, 'retention_days': 7,
            'allocation_seconds': 7200, 'output_gib': 1, 'max_concurrent_workers': 1,
            'recipe': training.RECIPE, 'manifest_sha256': data.MANIFEST_SHA256,
            'dataset_sha256': data.DATASET_SHA256, 'base_revision': reader_contract.BASE_REVISION,
            'model_files_sha256': data.task.digest(reader_assets.model_lock()),
            'gate': {'ready': 8, 'not_ready': 8, 'assessable': 24}}


def current_plan():
    source = reader_contract.source_identity(clean=True)
    names = sorted(set(source['runtime_files']) |
                   {str(p.relative_to(ROOT)) for pattern in ('witness_sft/*.py', 'evidence/witness-sft/*')
                    for p in ROOT.glob(pattern) if p.is_file()})
    return make_plan(source['commit'], {n: reader_contract.file_hash(ROOT / n) for n in names})


def check_stage(plan, stage, previous):
    if (not re.fullmatch('[0-9a-f]{40}', plan['source']) or plan != make_plan(plan['source'], plan['files'])
            or stage not in STAGES or len(previous) != STAGES.index(stage)):
        raise ValueError('Unrecognized plan or stage order')
    seen_calls, seen_containers = set(), set()
    for name, receipt in zip(STAGES, previous):
        w = artifact.worker(**receipt['worker'])
        if (receipt['stage'] != name or receipt['plan_sha256'] != data.task.digest(plan)
                or w['call'] in seen_calls or w['container'] in seen_containers):
            raise ValueError('Wrong predecessor stage, plan or worker')
        seen_calls.add(w['call'])
        seen_containers.add(w['container'])
        result = receipt['result']
        if name == 'supervised-pass':
            expected = artifact.identity(artifact.LIVE, plan['source'], w, reader_assets.model_lock()['config.json']['sha256'])
            if result.get('identity') != expected:
                raise ValueError('Incomplete production SFT identity')
            artifact.validate_stats(result['stats'])
            if not re.fullmatch('[0-9a-f]{64}', result['artifact_sha256']):
                raise ValueError('Missing terminal SFT artifact')
        elif result['summary'] != evaluation.summarize(result['episodes']):
            raise ValueError('Incomplete development evaluation')
