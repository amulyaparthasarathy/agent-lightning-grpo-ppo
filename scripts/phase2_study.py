#!/usr/bin/env python3
"""Stable plan/validate/launch/status/collect/stop-owned interface. Plan and validate are offline."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment import matched_contract as c, matched_supervisor as s


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2); f.write('\n')


def validate_launch(plan, approval, qualification=None):
    c.validate(plan)
    if approval != f"{s.key(plan)}:{c.digest(plan)}":
        raise ValueError('Billable launch requires --authorize with this exact study/attempt and plan digest')
    if c.source_identity(clean=True) != plan['source']:
        raise ValueError('Source differs from plan; regenerate it from the committed clean tree')
    if plan['phase'] == 'exploratory':
        validate_exploratory_qualification(plan,qualification)
    if plan['phase'] in ('pilot', 'campaign'):
        if not qualification or qualification.get('status') != 'completed':
            raise ValueError('Pilot requires completed live matched qualification evidence')
        previous = qualification['plan']
        c.validate(previous)
        if (previous['phase'] != 'qualification' or previous['source'] != plan['source']
                or (plan['phase']=='pilot' and previous['data_manifest'] != plan['data_manifest'])):
            raise ValueError('Qualification source/data differs from pilot')
        for job in previous['jobs']:
            stages = qualification['jobs'][job['job_id']]['stages']
            if len(stages) != 3 or any(x['status'] != 'completed' or not x.get('result', {}).get('completion') for x in stages):
                raise ValueError('Qualification stages incomplete')


def validate_exploratory_qualification(plan, qualification):
    if not qualification or qualification.get('status') != 'completed':
        raise ValueError('Exploratory report requires its completed short same-source qualification')
    previous = qualification['plan']
    c.validate(previous)
    if (previous['phase'] != 'exploratory-qualification' or previous['source'] != plan['source']
            or previous['data_manifest'] != plan['data_manifest'] or previous['study_id'] != plan['study_id']):
        raise ValueError('Exploratory qualification source/data/report identity differs')
    if set(qualification['jobs']) != {j['job_id'] for j in previous['jobs']}:
        raise ValueError('Qualification jobs incomplete')
    for job in previous['jobs']:
        receipts = qualification['jobs'][job['job_id']]['stages']
        if len(receipts) != 3:
            raise ValueError('Qualification stages incomplete')
        for stage, receipt in zip(job['stages'],receipts):
            result = receipt.get('result',{})
            proof = result.get('completion',{})
            if (receipt['status'] != 'completed' or result.get('status') != 'completed'
                    or result.get('call_id') != receipt.get('call_id')
                    or result.get('output') != stage['output'] or proof.get('stage') != stage['name']
                    or proof.get('plan_sha256') != c.digest(previous)):
                raise ValueError('Qualification terminal proof differs')


def protect_a02(registry, fetch):
    """Read-only guard; never cancels or duplicates the previous qualification."""
    key = 'agl-matched-v2/qualification/a02'
    row = registry.get('owner:'+key) or registry.get('submission:'+key)
    if not row:
        return
    if not row.get('call_id'):
        raise ValueError('Existing a02 submission is ambiguous; inspect it before new work')
    try:
        result = fetch(row['call_id'])
    except Exception as exc:
        raise ValueError('Existing a02 may be active or unresolved; leave it untouched') from exc
    if not isinstance(result,dict) or result.get('status') not in ('completed','failed','cancelled'):
        raise ValueError('Existing a02 is unresolved; leave it untouched')


def reconcile(index, fetch, heartbeat, now=None):
    """Read-only reconciliation. A dead supervisor or unavailable receipt never implies success."""
    now = time.time() if now is None else now
    result = json.loads(json.dumps(index))
    result['reconciliation'] = []
    for state in result['jobs'].values():
        for stage in state['stages']:
            call_id = stage.get('call_id')
            if not call_id:
                continue
            row = {'call_id': call_id, 'recorded_status': stage['status'], 'heartbeat': heartbeat(call_id)}
            try:
                receipt = fetch(call_id)
                row['actual'] = receipt
                row['state'] = 'completed' if (isinstance(receipt, dict) and receipt.get('status')=='completed'
                    and receipt.get('call_id')==call_id and receipt.get('output')==stage['output']
                    and receipt.get('completion',{}).get('plan_sha256')==c.digest(index['plan'])
                    and receipt['completion'].get('stage')==stage['stage']) else 'unknown'
            except TimeoutError:
                row['state'] = 'pending' if now-stage['submitted_at'] < c.stage_limit(index['plan'],stage['stage']) else 'unknown'
            except Exception as exc:
                row.update(state='unavailable_or_failed', error=f'{type(exc).__name__}: {exc}')
            result['reconciliation'].append(row)
    result['evidence_complete'] = result['status']=='completed' and len(result['reconciliation'])==sum(len(j['stages']) for j in index['plan']['jobs']) and all(
        r['state']=='completed' for r in result['reconciliation'])
    return result


def reconcile_volume(plan, index, read_file):
    proofs = []
    for job in plan['jobs']:
        recorded = {s['stage']: s for s in index['jobs'][job['job_id']]['stages']}
        for stage in job['stages']:
            proof = {'output': stage['output'], 'available': False,
                     'wandb': stage['wandb'], 'hosted_history': 'not independently queried'}
            try:
                def read(name):
                    return json.loads(b''.join(read_file(stage['output']+'/'+name)))
                receipt, completion, observability = read('receipt.json'), read('completion.json'), read('observability.json')
                if (receipt['status'] != 'completed' or receipt['call_id'] != recorded[stage['name']]['call_id']
                        or receipt['completion'] != completion or completion['plan_sha256'] != c.digest(plan)
                        or observability['wandb'] != stage['wandb']):
                    raise ValueError('Durable receipt/completion/W&B identity mismatch')
                proof['available'] = True
            except Exception as exc:
                proof['error'] = f'{type(exc).__name__}: {exc}'
            proofs.append(proof)
    return proofs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'validate', 'launch', 'status', 'collect', 'stop-owned'))
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--study-id', default='agl-matched-v2')
    parser.add_argument('--phase', choices=('qualification', 'pilot', 'campaign', *c.exploratory.PHASES), default='qualification')
    parser.add_argument('--concurrency', type=int)
    parser.add_argument('--capacity', type=Path)
    parser.add_argument('--bounded-admission', action='store_true', help='Exploratory only: resolve unknown quotas/cache through bounded real admission')
    parser.add_argument('--owner-gross-headroom-usd', type=float)
    parser.add_argument('--reviewed-usage-limit-usd', type=float)
    parser.add_argument('--pilot', type=Path)
    parser.add_argument('--admission', type=Path)
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--stage-seconds', type=int)
    parser.add_argument('--startup-seconds', type=int)
    parser.add_argument('--budget-usd', type=float)
    parser.add_argument('--all-resource-hourly-ceiling-usd', type=float)
    parser.add_argument('--data-manifest', type=Path)
    parser.add_argument('--authorize', default='')
    parser.add_argument('--qualification', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    if args.command=='plan':
        special = args.phase in c.exploratory.PHASES
        defaults = c.exploratory.bounds(args.phase) if special else {'stage_seconds':3600,'startup_seconds':600}
        for arg,key in (('stage_seconds','stage_seconds'),('startup_seconds','startup_seconds'),
                        ('budget_usd','budget_usd'),('all_resource_hourly_ceiling_usd','all_resource_hourly_ceiling_usd')):
            if getattr(args,arg) is None and key in defaults:
                setattr(args,arg,defaults[key])
        if not args.data_manifest or args.budget_usd is None or args.all_resource_hourly_ceiling_usd is None:
            parser.error('plan requires frozen --data-manifest and owner-selected budget/rate ceiling')
        plan = c.make_plan(args.study_id, args.phase, args.attempt, args.stage_seconds, args.startup_seconds,
                           args.budget_usd, args.all_resource_hourly_ceiling_usd,
                           json.loads(args.data_manifest.read_text()), c.source_identity(), args.concurrency)
        write_new(args.plan, plan)
    else:
        plan = json.loads(args.plan.read_text())
    c.validate(plan)
    if args.bounded_admission:
        if args.command != 'launch' or args.capacity or plan['phase'] not in c.exploratory.PHASES:
            raise ValueError('Bounded admission requires exploratory launch, without --capacity')
        from experiment import matched_admission
        matched_admission.validate_request(plan,args.owner_gross_headroom_usd,args.reviewed_usage_limit_usd)
    elif args.owner_gross_headroom_usd is not None or args.reviewed_usage_limit_usd is not None:
        raise ValueError('Owner admission evidence requires --bounded-admission')
    if args.command in ('plan', 'validate'):
        print(json.dumps({'valid': True, 'offline': True, 'plan_sha256': c.digest(plan),
                          'authorization_token': f'{s.key(plan)}:{c.digest(plan)}',
                          'source_clean': not plan['source']['dirty'], 'bounds': plan['bounds']}, indent=2))
        return 0
    # No Modal import/client/account access occurs before this explicit boundary.
    import modal
    if os.environ.get('MODAL_PROFILE') != c.WORKSPACE:
        raise ValueError('Select Stanford explicitly: MODAL_PROFILE=amulyasp')
    if args.command=='launch':
        qualification = json.loads(args.qualification.read_text()) if args.qualification else None
        validate_launch(plan, args.authorize, qualification)
        if args.out is None or args.out.exists():
            raise ValueError('Launch requires a new --out submission receipt path')
        if c.exploratory.is_exploratory(plan):
            if args.pilot or args.admission:
                raise ValueError('Exploratory protocol has no separate pilot/admission probe')
            if args.bounded_admission:
                try:
                    evidence = matched_admission.inspect(modal,plan,args.owner_gross_headroom_usd,args.reviewed_usage_limit_usd)
                except Exception as exc:
                    write_new(args.out.with_suffix('.admission.json'), {'status':'rejected','plan_sha256':c.digest(plan),
                        'error':type(exc).__name__+': '+str(exc),'submission_attempted':False})
                    raise
                write_new(args.out.with_suffix('.admission.json'),evidence)
                print('Bounded admission: quota/cache availability UNVERIFIED; provider will resolve it.',file=sys.stderr)
            else:
                c.exploratory.validate_capacity(json.loads(args.capacity.read_text()) if args.capacity else None,plan['phase'])
        if plan['phase'] == 'campaign':
            from experiment.matched_capacity import validate_campaign_gate
            validate_campaign_gate(plan, *(json.loads(p.read_text()) if p else None for p in (args.capacity, args.pilot, args.admission)))
        registry = modal.Dict.from_name('cs329z-matched-study-v2', environment_name='main', create_if_missing=True)
        if plan['phase']=='campaign':
            pilot = json.loads(args.pilot.read_text())
            admission = json.loads(args.admission.read_text())
            if registry.get('index:'+s.key(pilot['plan'])) != pilot:
                raise ValueError('Pilot differs from durable index')
            probes = modal.Dict.from_name('cs329z-tenway-admission', environment_name='main')
            remote_probe = probes.get(admission['probe_id']+':report')
            # Billing review is an explicit signed-off local supplement; measurements must match.
            if not remote_probe or any(admission.get(k) != v for k,v in remote_probe.items() if k!='billing_reconciled'):
                raise ValueError('Admission differs from remote measurements')
        if qualification is not None and registry.get('index:'+s.key(qualification['plan'])) != qualification:
            raise ValueError('Qualification receipt differs from the durable remote supervisor index')
        if c.exploratory.is_exploratory(plan):
            protect_a02(registry,lambda call:modal.FunctionCall.from_id(call).get(timeout=0))
            allocation_key = 'exploratory-allocation:'+plan['study_id']+':'+plan['phase']
            if not registry.put(allocation_key, {'plan_sha256':c.digest(plan),'source':plan['source']['commit'],
                                                'funding':plan['funding']},skip_if_exists=True):
                raise ValueError('This report allocation was already reserved; no automatic repeat or reserve use')
        if registry.get('owner:'+s.key(plan)) or registry.get('submission:'+s.key(plan)):
            raise ValueError('Attempt already submitted/owned; investigate and use a new attempt ID')
        # Atomic reservation includes uncertain local submission outcomes; never transparently resubmit.
        if not registry.put('submission:'+s.key(plan), {'plan_sha256': c.digest(plan), 'time': time.time()}, skip_if_exists=True):
            raise ValueError('Concurrent submission refused')
        os.environ['AGL_WORKING_VOLUME'] = c.VOLUME
        receipt = {'status':'unknown','study_key':s.key(plan),'plan_sha256':c.digest(plan),'completion_proven':False}
        try:
            import matched_app
            with modal.enable_output(), matched_app.app.run(detach=True, environment_name='main'):
                options = {'timeout':plan['bounds']['supervisor_seconds']}
                if c.exploratory.is_exploratory(plan):
                    options.update(cpu=(1,1),memory=(2048,2048))
                call = matched_app.supervise.with_options(**options).spawn(plan)
                receipt.update(status='submitted',call_id=call.object_id)
                registry.put('submission:'+s.key(plan), receipt)
                write_new(args.out, receipt)
                print(json.dumps(receipt, indent=2))
        except BaseException as exc:
            if args.bounded_admission:
                receipt.update(status='unknown',error=type(exc).__name__+': '+str(exc),automatic_retry=False)
                write_new(args.out.with_suffix('.failure.json'),receipt)
                # Preserve the reservation even if the provider/control plane is unavailable.
                try:
                    registry.put('submission:'+s.key(plan),receipt)
                except Exception:
                    pass
                if not args.out.exists():
                    write_new(args.out,receipt)
            raise
        return 0
    registry = modal.Dict.from_name('cs329z-matched-study-v2', environment_name='main')
    index = registry.get('index:'+s.key(plan))
    if not index:
        print(json.dumps({'status': 'unknown', 'reason': 'No supervisor index; investigate startup/disconnect',
                          'submission': registry.get('submission:'+s.key(plan))}, indent=2))
        if args.command=='stop-owned':
            submission = registry.get('submission:'+s.key(plan)) or {}
            if submission.get('plan_sha256') != c.digest(plan) or not submission.get('call_id', '').startswith('fc-'):
                raise ValueError('No verified owned supervisor call to cancel')
            modal.FunctionCall.from_id(submission['call_id']).cancel(terminate_containers=True)
        return 2
    if c.digest(index['plan']) != c.digest(plan):
        raise ValueError('Remote owner plan mismatch')
    if args.command=='stop-owned':
        registry.put('stop:'+s.key(plan), {'time': time.time(), 'plan_sha256': c.digest(plan)})
        calls = set(s.owned_calls(index))
        for job in plan['jobs']:
            for stage in job['stages']:
                entered = registry.get('stage:'+c.digest(plan)+':'+stage['wandb']['id']) or {}
                if entered.get('call_id', '').startswith('fc-'):
                    calls.add(entered['call_id'])
        for call_id in calls:
            modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)
        # Stop is durable even if the supervisor died. Workers check it before entry.
        # Completed calls may also be addressed safely; IDs are exact attempt-owned records.
        print('Cancellation requested for owned calls only; use status to reconcile.')
        return 0
    report = reconcile(index, lambda call: modal.FunctionCall.from_id(call).get(timeout=0),
                       lambda call: registry.get('heartbeat:'+call))
    try:
        report['supervisor_actual'] = modal.FunctionCall.from_id(index['owner']['call_id']).get(timeout=0)['status']
    except TimeoutError:
        report['supervisor_actual'] = 'pending'
    except Exception as exc:
        report['supervisor_actual'] = f'unavailable_or_failed: {type(exc).__name__}'
    volume = modal.Volume.from_name(c.VOLUME, environment_name='main')
    report['volume_evidence'] = reconcile_volume(plan, index, volume.read_file)
    report['evidence_complete'] = (report['evidence_complete'] and
                                  all(p['available'] for p in report['volume_evidence']))
    if args.command=='collect':
        if args.out is None:
            raise ValueError('collect requires a new --out directory')
        args.out.mkdir(parents=True, exist_ok=False)
        report['collection_errors'] = []
        collected_bytes = 0
        collection_limit = plan.get('funding',{}).get('collection_ceiling_gib_per_phase',float('inf'))*1024**3
        for job in plan['jobs']:
            for stage in job['stages']:
                try:
                    for entry in volume.iterdir(stage['output'], recursive=True):
                        relative = Path(entry.path.lstrip('/'))
                        if '..' in relative.parts or not relative.is_relative_to(stage['output']):
                            raise ValueError('Volume returned unowned path')
                        # Evidence only; native weights stay in the immutable remote checkpoint.
                        if relative.suffix not in ('.json', '.jsonl', '.log', '.txt') or 'temporary' in relative.parts:
                            continue
                        # Reject before download when the provider reports a file size.
                        if collected_bytes + getattr(entry,'size',0) > collection_limit:
                            raise ValueError('Exploratory collection byte ceiling reached; no automatic extension')
                        dest = args.out/relative
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with dest.open('xb') as f:
                            for chunk in volume.read_file(entry.path):
                                collected_bytes += len(chunk)
                                if collected_bytes > collection_limit:
                                    raise ValueError('Exploratory collection byte ceiling reached; no automatic extension')
                                f.write(chunk)
                except Exception as exc:
                    report['collection_errors'].append({'stage': stage['output'], 'error': str(exc)})
        report['collected_bytes'] = collected_bytes
        report['evidence_complete'] = report['evidence_complete'] and not report['collection_errors']
        write_new(args.out/'index.json', index)
        write_new(args.out/'collection.json', report)
    elif args.out:
        write_new(args.out, report)
    print(json.dumps(report, indent=2))
    return 0 if report['evidence_complete'] else 2


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2)
