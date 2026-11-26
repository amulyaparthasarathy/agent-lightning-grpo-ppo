"""Dependency-injected durable supervisor; no Modal imports or hidden retries."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import time

TERMINAL = {'completed', 'failed', 'cancelled', 'unknown'}


def key(plan):
    return f"{plan['study_id']}/{plan['phase']}/a{plan['attempt']:02d}"


def baseline_integration_admitted(plan, job, completion):
    """V2 qualification dependency, independent of task correctness/learning.

    The worker owns the trace joins; the supervisor checks its declared coverage
    before spending on dependent stages. Collection still rechecks all stages.
    Historical and archived-v1 scheduling is unchanged.
    """
    if plan.get('schema') != 'reader-task-v2' or plan['phase'] != 'qualification':
        return True
    report = completion.get('model_tool_integration')
    try:
        coverage = report['coverage']
        chains = [chain for entry in coverage for chain in entry['chains']]
        environment = plan['task_identity'].get('environment_sha256') or hashlib.sha256(
            json.dumps(plan['data_manifest'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        return (report['schema'] == 'reader-integration-v2' and report['algorithm'] == job['algorithm']
                and report['stage'] == 'baseline' and report['qualified'] is True
                and type(report['episodes_with_valid_chain']) is int
                and report['episodes_with_valid_chain'] == sum(bool(e['chains']) for e in coverage) > 0
                and report['trace_files'] == len(coverage)
                and all(chain['evidence_class'] == 'live-model'
                        and chain['task_id'] == plan['task_identity']['name']
                        and chain['environment_sha256'] == environment
                        and type(chain['terminal_reward']) in (int, float)
                        and chain['terminal_reward'] in (0, 1) for chain in chains))
    except (KeyError, TypeError, AttributeError):
        return False


def recovery_stage_admitted(plan, job, position, result, sha):
    """Require the stage-specific worker admission before dispatching a dependent stage."""
    stage = job['stages'][position]
    try:
        try:
            from . import recovery_calibration, reader_contract
        except ImportError:
            import recovery_calibration
            import reader_contract
        proof = result['completion']
        audit = proof['recovery_qualification']
        spec = reader_contract.learning.stage_spec(dict(plan=plan, stage=stage))
        recovery_calibration.check_table(spec, audit['episodes'])
        if plan['phase'] == reader_contract.modes.STUDY:
            steps = spec['updates']
            start = spec['initial_step']
            if (audit['actor_calls'] != list(range(1,steps+1))
                    or audit['refresh_steps'] != list(range(start,start+steps+1))
                    or audit['checkpoint_steps'] != ([steps] if steps else [])
                    or len(audit['gradient_norms']) != steps or not all(map(math.isfinite,audit['gradient_norms']))
                    or [loss['step'] for loss in audit['losses']] != list(range(1,steps+1))
                    or any(loss['metric'] != 'actor/pg_loss' or not math.isfinite(loss['value']) for loss in audit['losses'])
                    or proof['assets_sha256'] != reader_contract.digest(plan['assets'])
                    or proof['task_sha256'] != reader_contract.digest(plan['task_identity'])):
                return False
        roles = ['actor','critic'] if job['algorithm'] == reader_contract.ALGORITHMS[1] else ['actor']
        return (proof['source_sha256'] == reader_contract.digest(plan['source'])
                and proof['configuration_sha256'] == plan['configuration_sha256']
                and proof['plan_sha256'] == sha and result['output'] == stage['output']
                and audit['optimizer_calls'] == {role:spec['updates'] for role in roles}
                and all(e['evidence_class'] == 'live-model' for e in audit['episodes'])
                and proof['worker']['call_id'] == result['call_id']
                and proof['worker']['container_id'] == result['container_id']
                and proof['wandb_id'] == stage['wandb']['id'] and audit['plan_sha256'] == sha
                and audit['schema'] == 'recovery-qualification-evidence-v1' and audit['stage'] == stage['name']
                and audit['declared'] == spec and audit['episodes_admitted'] == spec['expected_episodes']
                and len(audit['episodes']) == spec['expected_episodes']
                and len(audit['collections']) == len(spec['batches']))
    except (KeyError, TypeError, ValueError):
        return False


def run(plan, dispatch, get_result, cancel, read_heartbeat, stop_requested, persist,
        clock=time.time, sleep=time.sleep):
    """Sole aggregate-index writer. Every spawn is recorded before polling another call.

    A lost response to spawn is unknown, never retried. A hydration/import loop
    without a heartbeat is cancelled by absolute startup deadline, even retries=0.
    """
    plan_sha256 = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    exploratory = plan['phase'] in ('exploratory-qualification','exploratory')
    study = recovery = False
    if plan.get('schema') == 'reader-task-v2':
        try:
            from . import reader_contract as reader
        except ImportError:
            import reader_contract as reader
        study = plan['phase'] == reader.modes.STUDY
        recovery = plan['phase'] in (reader.modes.STUDY, reader.modes.RECOVERY_LEARNING)
    # Recovery must journal an interrupted provider exchange, including during cleanup.
    provider_errors = BaseException if recovery else Exception
    start = clock()
    index = {'study_key': key(plan), 'status': 'started', 'started_at': start, 'plan': plan,
             'jobs': {j['job_id']: {'status': 'planned', 'stages': []} for j in plan['jobs']}}
    if recovery:
        rate = plan['bounds']['all_resource_hourly_ceiling_usd']
        index['supervisor_reservation_usd'] = plan['bounds']['supervisor_seconds']*rate/3600
    active = {}
    pending = [(job, 0) for job in plan['jobs']]
    uncertain = False
    def save():
        index['updated_at'] = clock()
        index['gpu_seconds_upper_bound_used'] = sum(
            (r.get('finished_at', clock())-r['submitted_at']) for job in index['jobs'].values()
            for r in job['stages'] if r.get('call_id'))
        if recovery:
            stages = [s for job in index['jobs'].values() for s in job['stages'] if 'reservation' in s]
            index['worker_seconds_reserved'] = sum(s['reservation']['seconds'] for s in stages)
            # An absent acknowledgement or merely requested cancellation is not an end-of-billing receipt.
            index['gpu_seconds_upper_bound_used'] = sum(
                s['reservation']['seconds'] if s['status'] == 'unknown' or s.get('cancellation_requested')
                else s.get('finished_at',clock())-s['submitted_at'] for s in stages)
        worker_rate = plan['bounds']['all_resource_hourly_ceiling_usd']
        index['worker_cost_usd_upper_bound_used'] = index['gpu_seconds_upper_bound_used']/3600*worker_rate
        index['cost_usd_upper_bound_used'] = index['worker_cost_usd_upper_bound_used'] + (clock()-start)/3600*(plan['bounds']['supervisor_hourly_ceiling_usd'] if exploratory else worker_rate)
        persist(copy.deepcopy(index))
    def submit(job, position):
        state = index['jobs'][job['job_id']]
        stage = job['stages'][position]
        receipt = {'stage': stage['name'], 'output': stage['output'], 'status': 'planned',
                   'wandb': stage['wandb'], 'submitted_at': clock(), 'call_id': None}
        if recovery:
            seconds = reader.stage_limit(plan, stage)
            receipt['reservation'] = {'seconds':seconds, 'usd':seconds*rate/3600}
        if study:
            position_in_study = plan['jobs'].index(job)
            if position_in_study == 3:
                index['first_seed_gate'] = reader.study.gate(plan, index, 3)
            if position_in_study == 7:
                index['final_release'] = reader.study.final_release(plan, index)
            if position_in_study >= 7 and index.get('final_release') != reader.study.final_release(plan, index):
                raise ValueError('Final release differs before dispatch')
            projected = index.get('worker_seconds_reserved', 0) + seconds
            if (projected > plan['bounds']['max_gpu_seconds']
                    or projected*rate/3600 + index['supervisor_reservation_usd'] > plan['bounds']['budget_usd']
                    or clock()-start+seconds > plan['bounds']['supervisor_seconds']):
                raise ValueError('Insufficient remaining study reservation or deadline')
        state['stages'].append(receipt)
        save()
        try:
            call_id = dispatch(job, stage)
        except provider_errors as exc:
            # Spawn may have reached the server. No ID means no safe targeted cancel.
            receipt.update(status='unknown', error=f'spawn outcome uncertain: {type(exc).__name__}: {exc}')
            state['status'] = 'unknown'
            save()
            if not isinstance(exc, Exception):
                raise
            return
        receipt.update(status='submitted', call_id=call_id)
        state['status'] = 'submitted'
        active[job['job_id']] = (job, position, receipt)
        save()
    try:
        while active or pending:
            if recovery and any(s['status'] in ('unknown','failed','cancelled') for s in index['jobs'].values()):
                break  # one lane; preserve the failed attempt and all undispatched work
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
                    limit = plan['bounds']['stage_seconds_by_name'][receipt['stage']] if exploratory else plan['bounds']['stage_seconds']
                    if study:
                        limit = receipt['reservation']['seconds']
                    expired = age >= limit
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
                    elif study and (clock()-receipt['submitted_at'] > receipt['reservation']['seconds']
                                    or clock()-start > plan['bounds']['supervisor_seconds']):
                        receipt.update(status='failed', result=result, error='study_deadline_exceeded')
                    elif recovery and not recovery_stage_admitted(plan, job, position, result, plan_sha256):
                        receipt.update(status='failed', result=result, error='recovery_stage_evidence_incomplete')
                    else:
                        receipt.update(status='completed', result=result)
                receipt['finished_at'] = clock()
                del active[jid]
                index['jobs'][jid]['status'] = receipt['status']
                if (study and plan['profile']['study_scope'] == reader.study.PILOT
                        and job == plan['jobs'][-1] and receipt['status'] == 'completed'):
                    try:
                        index['first_seed_gate'] = reader.study.gate(plan, index, 3)
                    except (KeyError, TypeError, ValueError) as exc:
                        index['jobs'][jid].update(status='failed', gate_error=str(exc))
                save()
                if receipt['status']=='completed' and position+1 < len(job['stages']):
                    gated = (receipt['stage'] == 'baseline' and plan.get('schema') == 'reader-task-v2'
                             and plan['phase'] == 'qualification')
                    admitted = not gated or baseline_integration_admitted(plan, job, receipt['result']['completion'])
                    if gated:
                        index['jobs'][jid]['dependency_gate'] = {
                            'rule': 'baseline-live-integration-v2', 'stage': 'baseline', 'admitted': admitted,
                            'reason': 'baseline_integration_qualified' if admitted else 'baseline_integration_unqualified'}
                    if not admitted:
                        # Preserve the durable completed baseline and diagnostics. Only
                        # its dependent stages are denied; the other algorithm is independent.
                        index['jobs'][jid]['status'] = 'failed'
                    save()
                    if admitted:
                        pending.insert(0, (job, position+1))
            if active:
                sleep(5)
    except BaseException:
        for jid, (_, _, receipt) in active.items():
            if recovery:
                receipt['cancellation_requested'] = True
            try:
                cancel(receipt['call_id'])
                receipt.update(status='cancelled', error='supervisor_interrupted')
            except provider_errors as exc:
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
                                            'status':'skipped', 'call_id':None, 'reason':('baseline_integration_unqualified' if state.get('dependency_gate', {}).get('admitted') is False
                                                      else 'dependency_or_supervisor_failed')})
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
