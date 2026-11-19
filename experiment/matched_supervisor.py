"""Dependency-injected durable supervisor; no Modal imports or hidden retries."""
from __future__ import annotations
import copy
import hashlib
import json
import time

TERMINAL = {'completed', 'failed', 'cancelled', 'unknown'}


def key(plan):
    return f"{plan['study_id']}/{plan['phase']}/a{plan['attempt']:02d}"


def run(plan, dispatch, get_result, cancel, read_heartbeat, stop_requested, persist,
        clock=time.time, sleep=time.sleep):
    """Sole aggregate-index writer. Every spawn is recorded before polling another call.

    A lost response to spawn is unknown, never retried. A hydration/import loop
    without a heartbeat is cancelled by absolute startup deadline, even retries=0.
    """
    plan_sha256 = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    exploratory = plan['phase'] in ('exploratory-qualification','exploratory')
    start = clock()
    index = {'study_key': key(plan), 'status': 'started', 'started_at': start, 'plan': plan,
             'jobs': {j['job_id']: {'status': 'planned', 'stages': []} for j in plan['jobs']}}
    active = {}
    pending = [(job, 0) for job in plan['jobs']]
    uncertain = False
    def save():
        index['updated_at'] = clock()
        index['gpu_seconds_upper_bound_used'] = sum(
            (r.get('finished_at', clock())-r['submitted_at']) for job in index['jobs'].values()
            for r in job['stages'] if r.get('call_id'))
        worker_rate = plan['bounds']['all_resource_hourly_ceiling_usd']
        index['worker_cost_usd_upper_bound_used'] = index['gpu_seconds_upper_bound_used']/3600*worker_rate
        index['cost_usd_upper_bound_used'] = index['worker_cost_usd_upper_bound_used'] + (clock()-start)/3600*(plan['bounds']['supervisor_hourly_ceiling_usd'] if exploratory else worker_rate)
        persist(copy.deepcopy(index))
    def submit(job, position):
        state = index['jobs'][job['job_id']]
        stage = job['stages'][position]
        receipt = {'stage': stage['name'], 'output': stage['output'], 'status': 'planned',
                   'wandb': stage['wandb'], 'submitted_at': clock(), 'call_id': None}
        state['stages'].append(receipt)
        save()
        try:
            call_id = dispatch(job, stage)
        except Exception as exc:
            # Spawn may have reached the server. No ID means no safe targeted cancel.
            receipt.update(status='unknown', error=f'spawn outcome uncertain: {type(exc).__name__}: {exc}')
            state['status'] = 'unknown'
            save()
            return
        receipt.update(status='submitted', call_id=call_id)
        state['status'] = 'submitted'
        active[job['job_id']] = (job, position, receipt)
        save()
    try:
        while active or pending:
            # Fill free slots immediately, never wait for a whole sequential batch.
            while pending and len(active) < plan['bounds']['concurrency'] and not uncertain and not stop_requested() and clock()-start < plan['bounds']['supervisor_seconds']-60:
                job, position = pending.pop(0)
                submit(job, position)
                uncertain = index['jobs'][job['job_id']]['status'] == 'unknown'
            if uncertain:
                pending.clear()
            stopped = stop_requested() or uncertain
            if stopped:
                for job, _ in pending:
                    index['jobs'][job['job_id']]['status'] = 'cancelled'
                pending.clear()
            save()
            deadline = (clock()-start >= plan['bounds']['supervisor_seconds']-60
                        or index['gpu_seconds_upper_bound_used'] >= plan['bounds']['max_gpu_seconds']
                        or index['cost_usd_upper_bound_used'] >= plan['bounds']['budget_usd']
                        or index['worker_cost_usd_upper_bound_used'] >= (plan['bounds']['worker_budget_usd'] if exploratory else plan['bounds']['budget_usd']))
            if deadline:
                for job, _ in pending:
                    index['jobs'][job['job_id']]['status'] = 'failed'
                pending.clear()
            for jid, (job, position, receipt) in list(active.items()):
                now = clock()
                heartbeat = read_heartbeat(receipt['call_id'])
                if heartbeat and (heartbeat.get('call_id') != receipt['call_id'] or not receipt['submitted_at'] <= heartbeat.get('time', -1) <= now+5):
                    heartbeat = None
                heartbeat = heartbeat or receipt.get('heartbeat')
                if heartbeat:
                    receipt['heartbeat'] = heartbeat
                    if receipt['status']=='submitted':
                        receipt['status'] = index['jobs'][jid]['status'] = 'started'
                        save()
                try:
                    result = get_result(receipt['call_id'])
                except TimeoutError:
                    age = now-receipt['submitted_at']
                    missing_start = not heartbeat and age >= plan['bounds']['startup_seconds']
                    stalled = heartbeat and now-heartbeat['time'] > 180
                    expired = age >= (plan['bounds']['stage_seconds_by_name'][receipt['stage']] if exploratory else plan['bounds']['stage_seconds'])
                    if stopped or deadline or missing_start or stalled or expired:
                        reason = ('owner_stop' if stopped else 'supervisor_deadline' if deadline else
                                  'startup_no_heartbeat' if missing_start else 'heartbeat_lost' if stalled else 'stage_deadline')
                        try:
                            cancel(receipt['call_id'])
                            receipt.update(status='cancelled' if stopped else 'failed', error=reason, cancellation_requested=True)
                        except Exception as exc:
                            receipt.update(status='unknown', error=f'{reason}; cancellation unconfirmed: {exc}')
                        receipt['finished_at'] = clock()
                        index['jobs'][jid]['status'] = receipt['status']
                        del active[jid]
                        save()
                        continue
                    continue  # Only pending work is subject to liveness cancellation.
                except Exception as exc:
                    receipt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                    try:
                        cancel(receipt['call_id'])
                        receipt['cancellation_requested'] = True
                    except Exception as cancel_error:
                        receipt.update(status='unknown', cancellation_error=str(cancel_error))
                else:
                    if (not isinstance(result, dict) or result.get('status') != 'completed'
                            or result.get('call_id') != receipt['call_id'] or result.get('output') != receipt['output']
                            or result.get('completion', {}).get('plan_sha256') != plan_sha256
                            or result['completion'].get('stage') != receipt['stage']):
                        receipt.update(status='unknown', error='Missing or mismatched durable completion proof')
                    else:
                        receipt.update(status='completed', result=result)
                receipt['finished_at'] = clock()
                del active[jid]
                index['jobs'][jid]['status'] = receipt['status']
                save()
                if receipt['status']=='completed' and position+1 < len(job['stages']):
                    pending.insert(0, (job, position+1))
            if active:
                sleep(5)
    except BaseException:
        for jid, (_, _, receipt) in active.items():
            try:
                cancel(receipt['call_id'])
                receipt.update(status='cancelled', error='supervisor_interrupted')
            except Exception as exc:
                receipt.update(status='unknown', error=str(exc))
            receipt['finished_at'] = clock()
            index['jobs'][jid]['status'] = receipt['status']
        for job, _ in pending:
            index['jobs'][job['job_id']]['status'] = 'cancelled'
        raise
    finally:
        # Every logical stage has an explicit terminal receipt, including unsubmitted dependents.
        for job in plan['jobs']:
            state = index['jobs'][job['job_id']]
            if state['status'] == 'planned':
                state['status'] = 'unknown' if uncertain else 'cancelled'
            seen = {r['stage'] for r in state['stages']}
            for stage in job['stages']:
                if stage['name'] not in seen:
                    state['stages'].append({'stage':stage['name'], 'output':stage['output'], 'wandb':stage['wandb'],
                                            'status':'skipped', 'call_id':None, 'reason':'dependency_or_supervisor_failed'})
        states = {v['status'] for v in index['jobs'].values()}
        index['status'] = ('completed' if states == {'completed'} else
                           'unknown' if 'unknown' in states else 'cancelled' if 'cancelled' in states else 'failed')
        index['finished_at'] = clock()
        save()
    return index


def owned_calls(index):
    """Only exact recorded active IDs; never shared app/display-name cancellation."""
    ids = []
    for state in index['jobs'].values():
        for stage in state['stages']:
            if stage['status'] in ('submitted', 'started', 'unknown', 'failed') and stage.get('call_id'):
                if not stage['call_id'].startswith('fc-'):
                    raise ValueError('Invalid owned FunctionCall ID')
                ids.append(stage['call_id'])
    return ids
