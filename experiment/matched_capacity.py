"""Explicit fresh evidence gates; local proposals never imply cloud capacity."""
import math
import time
from . import matched_contract as c

RESOURCES = {'gpu': 'H100!', 'cpu': 16, 'memory_mib': 196608}
# 5 arms each * 3 checkpoints (20/40/60); no historical deletion assumed.
CHECKPOINT_GIB = 5 * 3 * (73.31 + 37.39)
REQUIRED_FREE_GIB = 3072


def validate_capacity(evidence, concurrency, now=None, storage=True):
    now = time.time() if now is None else now
    if not evidence or evidence.get('workspace') != c.WORKSPACE or evidence.get('environment') != 'main':
        raise ValueError('Fresh Stanford capacity evidence required')
    if not 0 <= now-evidence['observed_at'] <= 900 or not evidence.get('evidence_reference'):
        raise ValueError('Capacity evidence must be at most 15 minutes old and attributable')
    if evidence['resources'] != RESOURCES or evidence['volume_version'] != 2:
        raise ValueError('Resource or Volume version mismatch')
    for limit, used, required in (('gpu_limit','gpu_used',concurrency), ('cpu_limit','cpu_used',16*concurrency+1),
                                  ('memory_mib_limit','memory_mib_used',196608*concurrency+2048),
                                  ('container_limit','containers_used',concurrency+1)):
        if any(type(evidence[k]) not in (int,float) or not math.isfinite(evidence[k]) or evidence[k]<0 for k in (limit,used)):
            raise ValueError('Invalid resource evidence')
        if evidence[limit]-evidence[used] < required:
            raise ValueError('Insufficient workspace headroom: '+limit)
    if storage and any(type(evidence[k]) not in (int,float) or not math.isfinite(evidence[k]) or evidence[k]<0
                       for k in ('volume_free_gib','worker_scratch_free_gib')):
        raise ValueError('Invalid storage evidence')
    if storage and (evidence['volume_free_gib'] < REQUIRED_FREE_GIB or evidence['worker_scratch_free_gib'] < 100):
        raise ValueError('Insufficient measured storage headroom')


def validate_campaign_gate(plan, capacity, pilot, admission, now=None):
    if plan['evaluation'] != 'heldout-256':
        raise ValueError('Campaign blocked: legacy 64-case panel is exploratory; untouched 256-case holdout unavailable')
    validate_capacity(capacity, plan['bounds']['concurrency'], now)
    if (not pilot or pilot.get('status') != 'completed' or pilot['plan']['phase'] != 'pilot'
            or pilot['plan']['source'] != plan['source']):
        raise ValueError('Completed same-source pilot required')
    c.validate(pilot['plan'])
    if len(pilot['jobs']) != len(pilot['plan']['jobs']):
        raise ValueError('Missing pilot jobs')
    if any(len(j['stages']) != 3 or any(s['status'] != 'completed' or not s.get('result', {}).get('completion') for s in j['stages']) for j in pilot['jobs'].values()):
        raise ValueError('Incomplete pilot proof')
    if (not admission or admission.get('status') != 'completed' or admission.get('max_simultaneous',0) < plan['bounds']['concurrency']
            or admission.get('source_commit') != plan['source']['commit'] or admission.get('resources') != RESOURCES
            or admission.get('billing_reconciled') is not True or not admission.get('billing_evidence_reference')
            or not admission.get('billing_reviewer')
            or not 0 <= (time.time() if now is None else now)-admission.get('finished_at',0) <= 86400):
        raise ValueError('Measured same-source admission and reconciled billing required')
