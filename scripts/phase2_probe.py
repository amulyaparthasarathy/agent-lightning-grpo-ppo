#!/usr/bin/env python3
"""Plan or explicitly authorize a separate short ten-H100 admission measurement."""
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment import matched_contract as c
from experiment.matched_capacity import validate_capacity, RESOURCES
from scripts.phase2_study import write_new


def make_plan(probe_id, source, rate=10., budget=15.):
    if not re.fullmatch('[a-z0-9-]{1,60}',probe_id) or not 0 < rate <= budget:
        raise ValueError('Invalid probe identity/rate/budget')
    bound = rate*(10*420+420)/3600
    if bound>budget:
        raise ValueError('Probe budget below conservative bound')
    return {'schema':'tenway-admission-v1','probe_id':probe_id,'source_commit':source['commit'],
            'source':source,'resources':RESOURCES,'workers':10,'startup_seconds':180,
            'worker_seconds':300,'supervisor_seconds':420,'overlap_seconds':30,'automatic_retries':0,
            'all_resource_hourly_ceiling_usd':rate,'budget_usd':budget,'budget_upper_bound_usd':bound}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('plan','validate','launch','status','collect','stop-owned'))
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--out',type=Path)
    p.add_argument('--probe-id',default='agl-tenway-admission-a01')
    p.add_argument('--authorize',default='');p.add_argument('--capacity',type=Path)
    p.add_argument('--rate',type=float,default=10);p.add_argument('--budget',type=float,default=15)
    a=p.parse_args(argv)
    plan=make_plan(a.probe_id,c.source_identity(),a.rate,a.budget) if a.command=='plan' else json.loads(a.plan.read_text())
    if plan != make_plan(plan['probe_id'],plan['source'],plan['all_resource_hourly_ceiling_usd'],plan['budget_usd']):
        raise ValueError('Probe plan drift')
    if a.command=='plan':write_new(a.plan,plan)
    token=plan['probe_id']+':'+c.digest(plan)
    if a.command in ('plan','validate'):
        print(json.dumps({'offline':True,'authorization_token':token,'plan':plan},indent=2));return 0
    if os.environ.get('MODAL_PROFILE')!='amulyasp':raise ValueError('Explicit Stanford profile required')
    if a.command=='launch':
        if a.authorize!=token or c.source_identity(clean=True)!=plan['source']:
            raise ValueError('Exact source-bound authorization required')
        validate_capacity(json.loads(a.capacity.read_text()) if a.capacity else None,10,storage=False)
        if not a.out or a.out.exists():raise ValueError('New --out submission receipt required')
    import modal
    registry=modal.Dict.from_name('cs329z-tenway-admission',environment_name='main',create_if_missing=a.command=='launch')
    key=plan['probe_id']
    if a.command=='launch':
        if not registry.put(key+':submission',{'plan_sha256':c.digest(plan),'time':time.time()},skip_if_exists=True):
            raise ValueError('Probe already submitted or ambiguous; never retry')
        import admission_app
        with modal.enable_output(),admission_app.app.run(detach=True,environment_name='main'):
            call=admission_app.supervise.spawn(plan)
            receipt={'call_id':call.object_id,'plan_sha256':c.digest(plan),'completion_proven':False}
            registry.put(key+':submission',receipt);write_new(a.out,receipt)
        return 0
    submission=registry.get(key+':submission') or {}
    if submission.get('plan_sha256')!=c.digest(plan):raise ValueError('Probe owner digest differs')
    report=registry.get(key+':report') or {'status':'unknown','reason':'No supervisor report','submission':submission}
    if a.command=='stop-owned':
        registry.put(key+':stop',True)
        for slot in range(10):
            record=registry.get(f'{key}:entered:{slot}') or report.get('calls',{}).get(str(slot),{})
            if record.get('call_id'):modal.FunctionCall.from_id(record['call_id']).cancel(terminate_containers=True)
        if submission.get('call_id'):modal.FunctionCall.from_id(submission['call_id']).cancel(terminate_containers=True)
        return 0
    if a.command=='collect' and not a.out:raise ValueError('collect requires --out')
    if a.out:write_new(a.out,report)
    print(json.dumps(report,indent=2))
    return 0 if report['status']=='completed' else 2


if __name__=='__main__':
    raise SystemExit(main())
