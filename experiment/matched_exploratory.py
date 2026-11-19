"""Frozen exploratory report protocol; numerical backend remains matched-rollout-mean-v2."""
import copy
import math
import time

PROTOCOL = 'exploratory-matched-20-v1'
PHASES = ('exploratory-qualification', 'exploratory')
SEEDS = {'exploratory-qualification': [9101], 'exploratory': [2001, 2002, 2003]}
STEPS = {'exploratory-qualification': 2, 'exploratory': 20}
LIMITS = {
    'exploratory-qualification': {'baseline':1800, 'training':900, 'checkpoint-eval':1800},
    'exploratory': {'baseline':1800, 'training':5400, 'checkpoint-eval':1800},
}
FUNDING = {'maximum_additional_usd':300, 'qualification_usd':50, 'study_workers_usd':150,
           'other_overhead_usd':25, 'reserve_usd':75, 'reserve_released_usd':0,
           'operational_limit_usd':225, 'worker_hourly_ceiling_usd':10,
           'supervisor_hourly_ceiling_usd':1, 'new_storage_ceiling_gib':600,
           'storage_cost_horizon_days':7, 'collection_ceiling_gib_per_phase':10,
           'automatic_retries':0, 'automatic_timeout_extensions':False,
           'seed_replacement':False, 'outcome_selection':False,
           'checkpoint_policy':'one-final-per-run-no-deletion', 'qualification_required_once':True,
           'standalone_pilot_required':False, 'admission_probe_required':False}


def is_exploratory(plan):
    return plan['phase'] in PHASES


def bounds(phase):
    limits = LIMITS[phase]
    jobs = 2 if phase == PHASES[0] else 6
    return {'stage_seconds':max(limits.values()), 'stage_seconds_by_name':dict(limits),
            'startup_seconds':300, 'supervisor_seconds':(jobs//2)*sum(limits.values())+600,
            'concurrency':2, 'max_gpu_seconds':jobs*sum(limits.values()),
            'budget_usd':50 if phase == PHASES[0] else 175,
            'worker_budget_usd':50 if phase == PHASES[0] else 150,
            'all_resource_hourly_ceiling_usd':10, 'supervisor_hourly_ceiling_usd':1}


def funding():
    return copy.deepcopy(FUNDING)


def validate(plan):
    if (plan.get('study_protocol') != PROTOCOL or plan.get('funding') != FUNDING
            or plan['bounds'] != bounds(plan['phase'])
            or plan.get('evaluation') != 'exploratory-legacy-64'
            or plan['data_manifest'].get('holdout') or plan['attempt'] != 1):
        raise ValueError('Exploratory protocol/budget/limits drift; reserve, retries and extensions are unavailable')
    if plan['bounds']['max_gpu_seconds']/3600*10 > plan['bounds']['worker_budget_usd']:
        raise ValueError('Exploratory worker allocation exceeded')


def costs(phase):
    b = bounds(phase)
    workers = b['max_gpu_seconds']/3600*10
    supervisor = b['supervisor_seconds']/3600
    return {'worker_hours':b['max_gpu_seconds']/3600, 'workers_ceiling_usd':workers,
            'supervisor_ceiling_usd':supervisor, 'compute_ceiling_usd':workers+supervisor}


def validate_capacity(evidence, phase, now=None, *, bounded_admission=False):
    """Exploratory-only evidence checks; formal campaign policy is unchanged.

    Volume versions are server facts, not name suffixes. Elastic Volume storage
    has no meaningful statfs free-byte value. Unknown quota/billing is not zero
    usage, an unlimited quota, or permission to spend the contingency reserve.
    """
    from .matched_capacity import RESOURCES
    from .matched_contract import WORKSPACE, VOLUME
    now = time.time() if now is None else now
    if bounded_admission and phase not in PHASES:
        raise ValueError('Bounded admission is exploratory-only')
    delegated = bounded_admission
    def number(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    def headroom(limit, used, required):
        # Unknown availability is resolved by real admission for explicitly bounded exploratory admission.
        # A known insufficient limit or malformed value still fails locally.
        if delegated and (limit is None or used is None):
            return ((limit is None or number(limit) and limit >= required)
                    and (used is None or number(used)))
        return number(limit) and number(used) and limit-used >= required
    if not evidence or (evidence.get('workspace'), evidence.get('environment')) != (WORKSPACE,'main'):
        raise ValueError('Fresh Stanford capacity evidence required')
    observed = evidence.get('observed_at')
    if not number(observed) or not 0 <= now-observed <= 900 or not evidence.get('evidence_reference'):
        raise ValueError('Fresh attributable capacity evidence required (15 minute maximum age)')
    if evidence.get('resources') != RESOURCES:
        raise ValueError('Exploratory resource request differs')
    metadata = evidence.get('volume_metadata') or {}
    if (metadata.get('volume_name') != VOLUME or not metadata.get('volume_id','').startswith('vo-')
            or not metadata.get('source') or type(metadata.get('backend_version_raw')) is not int
            or metadata.get('backend_version_raw') not in (1,2)
            or evidence.get('volume_version') != metadata.get('backend_version_raw')
            or metadata.get('backend_version_enum') != 'VOLUME_FS_VERSION_V'+str(evidence['volume_version'])):
        raise ValueError('Missing or inconsistent server Volume backend metadata')
    concurrency = bounds(phase)['concurrency']
    for limit, used, required in (('gpu_limit','gpu_used',concurrency),
                                  ('container_limit','containers_used',concurrency+1)):
        if not delegated and not evidence.get(limit+'_source'):
            raise ValueError('Current provider quota evidence required: '+limit)
        if not headroom(evidence.get(limit), evidence.get(used), required):
            raise ValueError('Insufficient or invalid current headroom: '+limit)
    quotas = evidence.get('cpu_ram_quotas') or {}
    if not delegated and not quotas.get('source'):
        raise ValueError('CPU/RAM quota applicability requires provider evidence')
    if quotas.get('scope') == 'aggregate':
        for limit, used, required in (('cpu_limit','cpu_used',16*concurrency+1),
                                      ('memory_mib_limit','memory_mib_used',196608*concurrency+2048)):
            if not headroom(quotas.get(limit), quotas.get(used), required):
                raise ValueError('Insufficient or unknown aggregate CPU/RAM quota: '+limit)
    elif quotas.get('scope') == 'per-container-only-confirmed':
        if (not headroom(quotas.get('max_cpu'), 0, 16)
                or not headroom(quotas.get('max_memory_mib'), 0, 196608)):
            raise ValueError('Unsupported per-container CPU/RAM request')
    elif delegated and quotas.get('scope') in (None, 'unknown'):
        pass  # Do not invent an aggregate quota; provider admission is authoritative.
    else:
        raise ValueError('CPU/RAM quota applicability is unknown; do not invent aggregate quotas')
    storage = evidence.get('storage') or {}
    if ('volume_free_gib' in evidence or storage.get('capacity_kind') != 'elastic-no-fixed-byte-capacity'
            or storage.get('inventory_method') not in ('modal-volume-iterdir','modal-dashboard')
            or not storage.get('source') or not number(storage.get('listed_bytes'))
            or type(storage.get('listed_entries')) is not int or storage['listed_entries'] < 0
            or storage.get('planned_incremental_gib') != FUNDING['new_storage_ceiling_gib']):
        raise ValueError('Use provider Volume inventory and storage budget, never df/statfs free capacity')
    if evidence['volume_version']==1 and storage['listed_entries'] >= 500000:
        raise ValueError('Volume v1 inode limit reached')
    if evidence['volume_version']==1 and concurrency+1 > 5:
        if storage.get('v1_parallel_writes_reviewed') is not True or not storage.get('v1_parallel_writes_review_source'):
            raise ValueError('Six workers plus supervisor exceed v1 five-writer recommendation; review required')
    scratch = evidence.get('scratch') or {}
    if scratch != {'minimum_free_gib':100,'enforcement':'existing-worker-startup-check'}:
        raise ValueError('Keep the existing real worker scratch check; no invented prelaunch free space')
    billing = evidence.get('billing') or {}
    billing_fields = ('remaining_gross_usage_usd',) if delegated else ('remaining_gross_usage_usd','remaining_funded_spending_usd')
    for name in billing_fields:
        if not billing.get('source') or not number(billing.get(name)) or billing[name] < bounds(phase)['budget_usd']:
            raise ValueError('Verified remaining usage/spending allowance required: '+name)
    if not delegated and (evidence.get('cuda_layers_cached') is not True or not evidence.get('cuda_cache_source')):
        raise ValueError('Confirm existing CUDA/VERL cache reuse before authorization')
    if evidence.get('funding') != FUNDING:
        raise ValueError('Capacity/spending review must accept the exact capped funding envelope')
