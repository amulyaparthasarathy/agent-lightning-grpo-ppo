"""Read-only checks for explicitly authorized bounded exploratory admission.

No GPU submission, image resolution, Secret access, or resource mutation here.
Unexposed quota/cache availability stays unknown; actual admission resolves it.
"""
import math
import time
from . import matched_contract as c, matched_exploratory as e
from .matched_capacity import RESOURCES

VOLUME_ID = 'vo-SfQGusN1IqkAw5A3o0fiE7'


def validate_request(plan, owner_headroom, usage_limit):
    c.validate(plan)
    if plan['phase'] not in e.PHASES or plan.get('study_protocol') != e.PROTOCOL:
        raise ValueError('Bounded admission is only for the frozen exploratory phases')
    if (usage_limit != 600 or type(owner_headroom) not in (float,int)
            or not math.isfinite(owner_headroom)
            or not plan['bounds']['budget_usd'] <= owner_headroom <= usage_limit):
        raise ValueError('Owner must confirm sufficient gross headroom and reviewed $600 usage limit')


def inspect(modal, plan, owner_headroom, usage_limit):
    validate_request(plan, owner_headroom, usage_limit)
    workspace = modal.Workspace.from_context().hydrate()
    if workspace.name != c.WORKSPACE:
        raise ValueError('Authenticated workspace differs from amulyasp')
    volume = modal.Volume.from_name(c.VOLUME, environment_name='main', create_if_missing=False).hydrate()
    metadata = volume._get_metadata()  # Raw server metadata from the pinned SDK.
    if volume.object_id != VOLUME_ID or metadata.version != 1:
        raise ValueError('Prepared Stanford V1 Volume identity differs')
    entries = list(volume.iterdir('/', recursive=True))
    billing = {'source':'Owner-confirmed gross headroom and reviewed usage limit; current Workspace.billing.summary when available',
               'owner_confirmed_gross_headroom_usd':owner_headroom, 'reviewed_usage_limit_usd':usage_limit,
               'remaining_gross_usage_usd':owner_headroom, 'provider_per_study_cap':False,
               'credits_are_extra_budget':False}
    try:
        summary = workspace.billing.summary()
    except Exception as exc:
        billing['live_summary'] = 'unavailable: '+type(exc).__name__+': '+str(exc)
    else:
        metered = float(summary.metered_cost)
        if not math.isfinite(metered) or metered < 0:
            raise ValueError('Invalid provider billing usage')
        billing.update(live_metered_usd=metered, live_billed_usd=str(summary.billed_cost),
                       remaining_gross_usage_usd=min(owner_headroom,usage_limit-metered))
    evidence = {'mode':'bounded-admission', 'capacity_status':'UNVERIFIED; actual provider admission and cache resolution required',
        'observed_at':time.time(), 'evidence_reference':'Read-only authenticated Modal SDK identity/inventory/billing + explicit owner headroom',
        'workspace':workspace.name, 'environment':'main', 'resources':dict(RESOURCES),
        'volume_version':1,'volume_metadata':{'source':'Modal Volume server metadata', 'volume_name':c.VOLUME,
            'volume_id':volume.object_id, 'backend_version_raw':metadata.version,'backend_version_enum':'VOLUME_FS_VERSION_V1'},
        'gpu_limit':None,'gpu_used':None,'container_limit':None,'containers_used':None,
        'cpu_ram_quotas':{'scope':'unknown'},'cuda_layers_cached':None,
        'storage':{'capacity_kind':'elastic-no-fixed-byte-capacity','inventory_method':'modal-volume-iterdir',
            'source':'read-only Volume.iterdir','listed_bytes':sum(x.size for x in entries),
            'listed_entries':len(entries),'planned_incremental_gib':e.FUNDING['new_storage_ceiling_gib']},
        'scratch':{'minimum_free_gib':100,'enforcement':'existing-worker-startup-check'},
        'billing':billing,'funding':e.funding(),'source_commit':plan['source']['commit'],'plan_sha256':c.digest(plan)}
    e.validate_capacity(evidence,plan['phase'],bounded_admission=True)
    return evidence
