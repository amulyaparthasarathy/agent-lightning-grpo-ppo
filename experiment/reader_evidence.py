"""Additional reader evidence checks, after the shared native completion checks."""
import json
import math
import os
import re
from pathlib import Path
import socket

try:
    from . import reader_contract as c, reader_integration, reader_modes as modes
except ImportError:
    import reader_contract as c
    import reader_integration
    import reader_modes as modes


def container_identity(value):
    if not isinstance(value, str) or not re.fullmatch(r'ta-[A-Za-z0-9]+', value):
        raise ValueError('Missing/invalid provider container identity')
    return value


def loss_evidence(metrics, roles):
    # Pinned VERL reduces minibatch Metric values before Tracking.log(data, global_steps).
    # A logger may split role keys across records; join by (global step, exact metric).
    required = {'actor': 'actor/pg_loss', 'critic': 'critic/vf_loss'}
    found = {}
    for row in metrics:
        for key, value in row['metrics'].items():
            if 'loss' not in key:
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('Invalid/nonfinite loss evidence')
            if key not in [required[r] for r in roles]:
                continue
            step = row.get('step')
            if type(step) is not int or step not in (1, 2):
                raise ValueError('Loss evidence outside declared updates')
            pair = (step, key)
            if pair in found and found[pair] != value:
                raise ValueError('Conflicting duplicate loss evidence')
            found[pair] = value
    if set(found) != {(step, required[role]) for role in roles for step in (1, 2)}:
        raise ValueError('Missing update/role loss evidence')
    return [{'step': step, 'metric': key, 'value': value} for (step, key), value in sorted(found.items())]


def complete(run, request, completion):
    events = [json.loads(line) for p in (run/'audit').glob('events-*.jsonl') for line in p.read_text().splitlines()]
    stage, job = request['stage'], request['job']
    finished = [e for e in events if e['event'] == 'wandb_finished']
    if len(finished) != 1 or finished[0]['id'] != stage['wandb']['id']:
        raise ValueError('Missing/duplicate/wrong W&B completion identity')
    terminal = [e for e in events if e['event'] == 'trainer_returned']
    if len(terminal) != 1 or terminal[0]['validation_only'] != modes.validation_only(request['plan'], stage):
        raise ValueError('Wrong trainer evaluation mode')
    result = {**completion, 'execution_schema': request['plan']['schema'],
              'source_sha256': c.digest(request['plan']['source']),
              'configuration_sha256': request['plan']['configuration_sha256'],
              'assets_sha256': c.digest(request['plan']['assets']),
              'task_sha256': c.digest(request['plan']['task_identity']),
              'worker': {'call_id': os.environ['AGL_READER_CALL_ID'], 'hostname': socket.gethostname(),
                         'container_id': container_identity(os.environ.get('MODAL_TASK_ID')),
                         'launcher_pid': os.getpid(), 'trainer_pid': terminal[0]['pid']},
              'wandb_id': stage['wandb']['id']}
    if modes.kind(request['plan'], stage) == 'training':
        roles = ('actor', 'critic') if job['algorithm'] == c.ALGORITHMS[1] else ('actor',)
        checkpoint = json.loads((run/'checkpoints/global_step_2/matched_identity.json').read_text())
        if set(checkpoint['roles']) != set(roles):
            raise ValueError('Unexpected native checkpoint roles')
        for role in roles:
            files = checkpoint['roles'][role]
            required = {f'{kind}_world_size_1_rank_0.pt' for kind in ('model', 'optim', 'extra_state')}
            if not required <= files.keys() or any(files[n]['bytes'] <= 0 for n in required):
                raise ValueError('Incomplete single-rank native checkpoint role inventory')
        if any(e['role'] not in roles for e in events if e['event'] == 'optimizer_step'):
            raise ValueError('Unexpected optimizer role')
        for role in roles:
            counts = sorted(e['count'] for e in events if e['event'] == 'optimizer_step' and e['role'] == role)
            gradients = [e for e in events if e['event'] == 'gradient_step' and e['role'] == role]
            if counts != [1, 2] or len(gradients) != 2 or any(not e['preclip_finite'] or not e['optimizer_called'] for e in gradients):
                raise ValueError('Wrong optimizer sequence or missing/nonfinite gradient evidence')
        collections = [e for e in events if e['event'] == 'matched_advantages']
        if (len(collections) != 2 or len({e['iteration'] for e in collections}) != 2
                or any(e['real_rollouts'] != 32 or e['generated_tokens'] <= 0
                       or not 0 <= e['constant_groups'] <= 8 for e in collections)):
            raise ValueError('Missing/duplicate/incomplete rollout accounting')
        if [e['step'] for e in events if e['event'] == 'native_checkpoint_saved'] != [2]:
            raise ValueError('Missing/extra final native checkpoint save')
        metrics = [json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()]
        losses = loss_evidence(metrics, roles)
        result['learning_diagnostics'] = {'collections': [{k: e[k] for k in ('iteration', 'generated_tokens', 'real_rollouts', 'constant_groups')} for e in collections],
            'informative_reward_groups': sum(8-e['constant_groups'] for e in collections),
            'informative_group_relative_task_advantage': ('observed' if any(e['constant_groups'] < 8 for e in collections) else 'not observed') if job['algorithm'] == c.ALGORITHMS[0] else 'not applicable',
            'losses_by_update': losses, 'finite_loss_values': len(losses), 'accuracy_improvement_required': False}
    if stage['name'] == 'checkpoint-eval':
        training_file = run.parent/'training/completion.json'
        if modes.phase(request['plan']) == modes.RECOVERY_LEARNING:
            try:
                from . import qualification_paths
            except ImportError:
                import qualification_paths
            binding = json.loads((run/'path-binding.json').read_text())
            training_file = qualification_paths.locations(request, binding)['training_completion']
        training = json.loads(training_file.read_text())
        if (training['plan_sha256'] != request['plan_sha256']
                or training['checkpoint_sha256'] != result['checkpoint_sha256']
                or training['worker']['call_id'] == result['worker']['call_id']
                or container_identity(training['worker'].get('container_id')) == result['worker']['container_id']):
            raise ValueError('Reload requires the exact training checkpoint in a distinct worker/process')
        result['training_worker'] = training['worker']
    result['model_tool_integration'] = reader_integration.report(run, request)
    return result


def durable_stage(plan, job, stage, entry, read, calls):
    """Join one stage's supervisor entry with its durable receipt, completion and observability.

    The single definition of durable stage identity, shared by qualification,
    asset development and calibration. Raises ValueError on any mismatch.
    """
    receipt, proof, obs = (read(stage['output']+'/'+n) for n in ('receipt.json', 'completion.json', 'observability.json'))
    if (entry['status'] != 'completed' or receipt['status'] != 'completed'
            or entry['output'] != stage['output'] or receipt['output'] != stage['output']
            or receipt != entry['result'] or proof['execution_schema'] != plan['schema']
            or receipt['call_id'] != entry['call_id'] or receipt['call_id'] in calls
            or receipt['completion'] != proof or proof != entry['result']['completion']
            or proof['plan_sha256'] != c.digest(plan) or proof['stage'] != stage['name']
            or proof['worker']['call_id'] != receipt['call_id']
            or container_identity(proof['worker'].get('container_id')) != receipt.get('container_id')
            or proof['source_sha256'] != c.digest(plan['source'])
            or proof['configuration_sha256'] != plan['configuration_sha256']
            or proof['assets_sha256'] != c.digest(plan['assets'])
            or proof['task_sha256'] != c.digest(plan['task_identity'])
            or proof['wandb_id'] != stage['wandb']['id'] or obs['wandb'] != stage['wandb']):
        raise ValueError('Durable receipt/completion/observability identity mismatch')
    calls.add(receipt['call_id'])
    integration = proof.get('model_tool_integration')
    if integration is None:
        integration = {'qualified': False, 'gaps': ['Missing integration report']}
    elif (integration['schema'] != ('reader-integration-v2' if plan['schema'] == c.SCHEMA else 'reader-integration-v1') or integration['algorithm'] != job['algorithm']
            or integration['stage'] != stage['name']
            or integration['qualified'] != (integration['episodes_with_valid_chain'] > 0)
            or integration['episodes_with_valid_chain'] != sum(bool(e['chains']) for e in integration['coverage'])
            or integration['trace_files'] != len(integration['coverage'])):
        raise ValueError('Invalid integration coverage report')
    return proof, integration


def local_reader(plan, directory):
    """Read durable files from a directory written by `collect` (paths relative to the plan root)."""
    from pathlib import PurePosixPath
    root = Path(directory)
    def read(name):
        return json.loads((root/PurePosixPath(name).relative_to(plan['root'])).read_text())
    return read


def calibration_collection(plan, index, read, local=None):
    """Review every calibration stage and build the report. Tolerant of missing evidence.

    The plan itself must be a valid current plan: that is checked on every path and
    raises. Everything else is reported. `index` is the supervisor index or None;
    `local` is a collected directory holding traces and audit events, or None.
    A live calibration decision needs all of them (see recovery_calibration.review_stage).
    """
    try:
        from . import recovery_calibration as calibration
    except ImportError:
        import recovery_calibration as calibration
    from pathlib import PurePosixPath
    c.validate(plan)
    if modes.phase(plan) != modes.CALIBRATION:
        raise ValueError('Not a calibration plan')
    plan_problems = []
    if not plan['training_admissible']:
        plan_problems.append('the plan is not executable (fixture profile, unbound assets or uncommitted source), '
                             'so it cannot have produced live evidence')
    index_problem = None
    if index is None:
        index_problem = 'no supervisor index was supplied'
    elif (not isinstance(index, dict) or index.get('plan') != plan
          or not isinstance(index.get('jobs'), dict) or set(index['jobs']) != {j['job_id'] for j in plan['jobs']}):
        index_problem = 'the supervisor index is malformed or belongs to a different plan'
    def tolerant(name):
        try:
            return read(name)
        except Exception:  # an unavailable durable file is evidence of nothing
            return None
    reviews, calls, timings, route_problems = {}, set(), {}, {}
    if not index_problem:
        states = [v.get('status') if isinstance(v, dict) else None for v in index['jobs'].values()]
        # The supervisor derives the index status from its jobs; anything else was not written by it.
        if index.get('status') != ('completed' if set(states) == {'completed'} else 'unknown' if 'unknown' in states
                                   else 'cancelled' if 'cancelled' in states else 'failed'):
            index_problem = 'the supervisor index status is inconsistent with its jobs'
    for job in plan['jobs']:
        state = {} if index_problem else index['jobs'][job['job_id']]
        state = state if isinstance(state, dict) else {}
        entries = state.get('stages') if isinstance(state.get('stages'), list) else []
        entries = [e if isinstance(e, dict) else {} for e in entries]
        names = [e.get('stage') for e in entries]
        route = reviews[job['algorithm']] = []
        if not index_problem:
            # The route's job must have completed with exactly the planned stages, in order.
            problems = route_problems[job['algorithm']] = []
            if state.get('status') != 'completed':
                problems.append(f"the supervisor index records this job as {state.get('status')!r}, not completed")
            if names != [s['name'] for s in job['stages']]:
                problems.append('the supervisor index does not hold exactly the planned stages in order')
        for stage in job['stages']:
            request = c.stage_request(plan, job, stage)
            entry = entries[names.index(stage['name'])] if names.count(stage['name']) == 1 else None
            proof = None
            if index_problem:
                durable = None if index is None else {'verified': False, 'problems': [index_problem]}
            elif entry is None:
                durable = {'verified': False, 'problems': ['the supervisor index has no single entry for this stage']}
            else:
                try:
                    proof, _ = durable_stage(plan, job, stage, entry, read, calls)
                    durable = {'verified': True, 'problems': []}
                except Exception as exc:
                    durable = {'verified': False, 'problems': [f'{type(exc).__name__}: {exc}'[:200]]}
                result = entry.get('result') if isinstance(entry.get('result'), dict) else {}
                if all(isinstance(result.get(k), (int, float)) for k in ('started_at', 'finished_at')):
                    timings[f"{job['algorithm']}/{stage['name']}"] = round(result['finished_at'] - result['started_at'], 1)
            run = Path(local)/PurePosixPath(stage['output']).relative_to(plan['root']) if local is not None else None
            route.append(calibration.review_stage(
                request, supervisor=entry, receipt=tolerant(stage['output']+'/receipt.json'),
                proof=proof if proof is not None else tolerant(stage['output']+'/completion.json'), durable=durable, run=run))
    provenance = ('not verified: ' + index_problem if index_problem else
                  'supervisor index, receipts, completions and observability joined per stage'
                  + ('; traces and audit events recomputed from the collected directory' if local is not None
                     else '; traces and audit events not available to this reader'))
    report = calibration.report(plan, reviews, plan_problems, timings, provenance, route_problems)
    complete = report['live_policy_measurement']
    admitted = all(s['packet_status'] == 'admitted' for r in report['routes'].values() for s in r['stages'].values())
    out = {'evidence_complete': complete, 'operational_completion': 'completed' if admitted else 'incomplete or unverified',
           'calibration': report, 'qualification_evidence': False,
           'live_qualification': 'development calibration only; zero optimization; not qualification or learning evidence',
           'hosted_wandb_history': 'not independently queried'}
    if not complete:
        out['incomplete_reason'] = ('Calibration evidence is not a complete, verified live collection for every route; '
                                    'see calibration.routes[*].decision.reasons and failure_diagnostics'
                                    + ('' if local is not None else '. Traces and audit events are required: run collect, '
                                       'then scripts/task_local.py report on the collected directory'))
    return out


def validate_collection(plan, index, read, local=None):
    """Join durable proof with exact supervisor calls; do not trust directory existence.

    Calibration plans return their reviewed report whether or not the evidence is
    complete (evidence_complete says which); other phases raise on any gap, as before.
    """
    if modes.phase(plan) == modes.RECOVERY_LEARNING:
        try:
            from .recovery_learning_evidence import review
        except ImportError:
            from recovery_learning_evidence import review
        return review(plan, index, read, local)
    if modes.phase(plan) == modes.CALIBRATION:
        return calibration_collection(plan, index, read, local)
    if index['plan'] != plan or index['status'] != 'completed' or set(index['jobs']) != {j['job_id'] for j in plan['jobs']}:
        raise ValueError('Incomplete or mismatched supervisor index')
    calls = set()
    checkpoints = {}
    outcomes = {}
    for job in plan['jobs']:
        outcome = outcomes[job['algorithm']] = {'operational_stage_completion': {}, 'model_tool_integration': {}}
        receipts = index['jobs'][job['job_id']]['stages']
        development = plan['phase'] == 'collection-only'
        expected_stages = ['training'] if development else ['baseline', 'training', 'checkpoint-eval']
        if len(receipts) != len(expected_stages) or [r['stage'] for r in receipts] != expected_stages:
            raise ValueError('Missing/duplicate stage receipts')
        for stage, entry in zip(job['stages'], receipts, strict=True):
            proof, integration = durable_stage(plan, job, stage, entry, read, calls)
            outcome['operational_stage_completion'][stage['name']] = 'completed'
            outcome['model_tool_integration'][stage['name']] = integration
            if development:
                if proof.get('optimizer_updates') != {'actor': 0, 'critic': 0} or proof.get('development', {}).get('real_rollouts') != 32:
                    raise ValueError('Missing collection-only no-update evidence')
                outcome['development'] = proof['development']
            elif stage['name'] == 'training':
                checkpoints[job['job_id']] = proof
                diagnostics = proof.get('learning_diagnostics', {})
                outcome['informative_grpo_reward_groups'] = (diagnostics.get('informative_reward_groups')
                    if job['algorithm'] == c.ALGORITHMS[0] else 'not applicable')
            elif stage['name'] == 'checkpoint-eval':
                previous = checkpoints[job['job_id']]
                if (proof['checkpoint_sha256'] != previous['checkpoint_sha256']
                        or proof['training_worker'] != previous['worker']
                        or proof['worker']['container_id'] == previous['worker']['container_id']):
                    raise ValueError('Training/reload evidence differs')
        outcome['integration_rule'] = 'all-three-stages-v2' if plan['schema'] == c.SCHEMA else 'any-stage-v1'
        outcome['harness_integration_qualified'] = (all if plan['schema'] == c.SCHEMA else any)(r['qualified'] for r in outcome['model_tool_integration'].values())
        outcome['qualification_gaps'] = [] if outcome['harness_integration_qualified'] else ['Model/tool integration unqualified: required stage coverage missing in this algorithm workflow']
        if not development and job['algorithm'] == c.ALGORITHMS[0] and not outcome['informative_grpo_reward_groups']:
            outcome['qualification_gaps'].append('Informative group-relative task-advantage evidence was not observed')
    return {'evidence_complete': True, 'operational_completion': 'completed', 'algorithms': outcomes,
            'qualification_evidence': plan['phase'] != 'collection-only',
            'live_qualification': ('development only; not six-stage qualification' if plan['phase'] == 'collection-only'
                else 'gaps remain' if any(o['qualification_gaps'] for o in outcomes.values())
                else 'operational and harness evidence observed; learning success not established'),
            'hosted_wandb_history': 'not independently queried'}
