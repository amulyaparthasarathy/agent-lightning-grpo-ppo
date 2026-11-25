"""recovery-calibration-v1: development calibration of recovery-readiness-v1.

Zero optimization. Per algorithm route: one greedy evaluation of the frozen 24-case
development panel (gateway `val` route, temperature 0) and three sampled
collections of eight cases x four rollouts (gateway `train` route, temperature 1)
through the production collection adapter, which stops before advantages.
Only development files are ever routed to a stage; the final-evaluation panel is
identified in the plan by hash and never opened by a model-facing path.

Pure stdlib except materialize (PyArrow, in the worker). Decisions are computed
per algorithm route; routes are never pooled. Diagnostics never become rewards.
"""
from collections import Counter
import hashlib
import json
import math
import re
from pathlib import Path

try:
    from . import recovery_task as task, reader_integration, reader_modes as modes
except ImportError:
    import recovery_task as task
    import reader_integration
    import reader_modes as modes

SCHEMA = 'recovery-calibration-v1'
STAGE_SCHEMA = 'recovery-calibration-stage-v1'
REPORT_SCHEMA = 'recovery-calibration-report-v1'
# Fixed before any live sampling. A project-specific engineering screen, not a
# research-established threshold: six of 24 sampled groups is about two informative
# groups per eight-question batch on this panel. Passing it is not proof of learning.
CRITERIA = {'version': 'recovery-screen-v1', 'applies_to': 'each algorithm route separately; never pooled',
            'groups': 24, 'min_semantic_evidence_informative_groups': 6, 'min_case_families': 2,
            'max_interface_share_of_zero_rewards': 0.5,
            'interface_failure_classes': ['format', 'action'], 'substantive_failure_classes': ['semantic', 'evidence'],
            'operational_requirement': 'all 120 episodes complete with confirmed rewards and zero optimization events',
            'status': 'engineering heuristic; no universal accuracy band is imposed'}
FORBIDDEN_EVENTS = ('optimizer_step', 'gradient_step', 'matched_advantages', 'native_checkpoint_saved')
GATEWAY_TEMPERATURES = ('default_proxy.train.temperature=1.0', 'default_proxy.val.temperature=0.0')


def declaration():
    """Frozen panel, packets, modes, counts and criteria bound into every calibration plan."""
    panels = task.panels()
    dev = panels['development']
    stages = {'greedy-development': {
        'mode': 'greedy', 'route': 'val', 'temperature': 0, 'rollouts_per_question': 1, 'split': 'development',
        'panel_id': dev['panel_id'], 'packet_id': None, 'file': dev['file'], 'case_ids': dev['case_ids'],
        'expected_episodes': len(dev['case_ids'])}}
    for number, packet in enumerate(dev['packets'], 1):
        stages[f'sampled-packet-{number}'] = {
            'mode': 'sampled', 'route': 'train', 'temperature': 1, 'rollouts_per_question': 4, 'split': 'development',
            'panel_id': dev['panel_id'], 'packet_id': packet['packet_id'], 'file': packet['file'],
            'case_ids': packet['case_ids'], 'expected_episodes': 4 * len(packet['case_ids'])}
    if list(stages) != list(modes.STAGES[modes.CALIBRATION]):
        raise ValueError('Calibration stages differ from the declared reader phase')
    return {'schema': SCHEMA, 'task': task.TASK, 'split': 'development', 'panel_id': dev['panel_id'],
            'panel_cases': 24, 'packets': [p['packet_id'] for p in dev['packets']], 'stages': stages,
            'questions_per_collection': 8, 'rollouts_per_question': 4, 'collections_per_algorithm': 3,
            'greedy_episodes_per_algorithm': 24, 'sampled_episodes_per_algorithm': 96,
            'episodes_per_algorithm': 120, 'total_episodes': 240, 'optimizer_updates': 0,
            'seed_derivation': 'unchanged matched-rollout-mean-v2 streams for run seed 9001; the stages of a job share them',
            'final_evaluation': {**panels['final'], 'model_use': 'forbidden during calibration'},
            'criteria': CRITERIA, 'qualification_evidence': False}


def routes(run):
    """Both upstream dataset arguments. The pinned entry point loads both eagerly."""
    run = Path(run)
    return str(run/'calibration-train.parquet'), str(run/'calibration-val.parquet')


def stage_declaration(request):
    plan = request['plan']
    if modes.phase(plan) != modes.CALIBRATION or plan['calibration'] != declaration():
        raise ValueError('Not a current recovery calibration plan')
    return plan['calibration']['stages'][request['stage']['name']]


def materialize(request, data, run, commands):
    """Copy exactly the declared development file to both run-local dataset paths."""
    import pyarrow.parquet as pq
    stage = stage_declaration(request)
    if not all(any(part == value for part in commands['gateway']) for value in GATEWAY_TEMPERATURES):
        raise ValueError('Gateway sampling temperatures differ from the declared calibration modes')
    table = pq.read_table(Path(data)/stage['file'])
    rows = table.to_pylist()
    if (rows != task.cases()[stage['file']] or [r['case_id'] for r in rows] != stage['case_ids']
            or any(r['split'] != 'development' for r in rows)):
        raise ValueError('Calibration panel identity, membership, order or split differs')
    for path in routes(run):
        pq.write_table(table, path)
    return list(routes(run))


def wilson(successes, total, z=1.96):
    """95% Wilson score interval; None when the denominator is empty."""
    if not total:
        return None
    p, d = successes / total, 1 + z * z / total
    centre, half = (p + z * z / (2 * total)) / d, z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def served_evidence(records, llms):
    """Check each generation against the response the server returned for it.

    The served record is written below the pinned client, before any name, finish
    reason, content or usage is rewritten. The recorded generation must carry the
    same call identity, and the served evidence must itself establish completion.
    """
    served = [r for r in records if r['kind'] == 'served_response']
    if len(served) != len(llms):
        raise ValueError('Served response evidence is missing for a generation')
    for raw, llm in zip(served, llms):
        s, content = raw['outputs'], llm['outputs']['content']
        calls = content if isinstance(content, list) else []
        text = content if isinstance(content, str) else None
        if not llm['started_at'] <= raw['started_at'] <= raw['ended_at'] <= llm['ended_at']:
            raise ValueError('Served response evidence belongs to a different generation')
        rebuilt = {'finish_reason': s['finish_reason'], 'content': text if s['content_present'] else None,
                   'tool_calls': s['tool_calls'], 'usage': s['usage']}
        state, reason = task.generation_status(rebuilt, llm['inputs']['client_generation_defaults']['max_tokens'])
        usage = llm['outputs'].get('usage')
        if (calls != s['tool_calls'] or s['completion'] != {'state': state, 'reason': reason} or state != 'complete'
                or (text is not None and s['content_present']
                    and hashlib.sha256(text.encode()).hexdigest() != s['content_sha256'])
                # The recorded usage is the served usage, or absent where the client could not tell zeros from none.
                or (usage is not None and usage != s['usage'])):
            raise ValueError('Served response evidence differs from the recorded generation or does not establish completion')
    return [r['outputs'] for r in served]


def episode_row(records, request, evidence_class='live-model'):
    """One reviewed episode from its own trace, re-verified from recorded evidence.

    Raises for an incomplete, cross-stage, cross-source or internally inconsistent
    trace. Carries tool names as sampled and hashes of arguments, no answer text,
    and no label beyond the trusted family name.
    """
    stage = stage_declaration(request)
    outcome = reader_integration.episode_outcome(records)
    if outcome not in ('complete_final', 'complete_recovered', 'terminal_policy_failure'):
        raise ValueError('Incomplete or untrusted episode; no reward is admitted')
    chains = reader_integration.chains(records, request, evidence_class=evidence_class)
    episode = next(r for r in records if r['kind'] == 'episode')
    out = episode['outputs']
    if episode['case_id'] not in stage['case_ids'] or episode['eval_split'] != 'development':
        raise ValueError('Episode is outside the declared development panel')
    case = dict(next(r for r in task.rows('development') if r['case_id'] == episode['case_id']))
    if episode['inputs'] != {'question': case['question']} or episode.get('episode_policy') != task.EPISODE_POLICY['version']:
        raise ValueError('Episode request or episode policy differs')
    trusted = task.expected(case['question'], case['result'])
    actions = [r for r in records if r['kind'] == 'action_outcome']
    llms = [r for r in records if r['kind'] == 'llm']
    served = served_evidence(records, llms)
    ledger = [{'tool': a['inputs']['call']['name'], 'arguments': json.loads(a['inputs']['call']['arguments']),
               'content': a['inputs']['observation']['content']} for a in actions if a['outputs']['successful_tool_call']]
    terminal = outcome == 'terminal_policy_failure'
    # Re-derive each recorded classification from the retained call and observation.
    for a in actions:
        call, observation = a['inputs']['call'], a['inputs']['observation']
        derived = task.classify_error(call['name'], call['arguments'], observation['content']) if observation['is_error'] else None
        if (derived != a['outputs']['policy_error'] or a['outputs']['successful_tool_call'] is not (not observation['is_error'])
                or (observation['is_error'] and derived is None) or observation['name'] != call['name']):
            raise ValueError('Recorded action outcome differs from the typed evidence')
    # In a completed episode every successful call must be joined to the model input that saw it.
    if not terminal and len(chains) != len(ledger):
        raise ValueError('A successful call is not joined to its observation in a later model input')
    reward, diagnostics = task.assess('' if terminal else out['raw_answer'], trusted, ledger)
    if terminal:
        reward = 0.0
        diagnostics.update(json_syntax=None, format_compliant=None, failure_class='action')
    rounds = sum(bool(s['tool_calls']) for s in served)
    if (reward != out['reward'] or diagnostics != out['diagnostics'] or out['successful_tool_calls'] != len(ledger)
            or out['tool_rounds'] != rounds):
        raise ValueError('Recorded reward or diagnostics differ from independent re-verification')
    usage = [(r['outputs'] or {}).get('usage') or {} for r in llms]
    family, variant = task.family(case['case_id'])
    answer = out['raw_answer']
    return {'case_id': episode['case_id'], 'family': family, 'variant': variant,
            'rollout_id': episode['rollout_id'], 'attempt_id': episode['attempt_id'], 'trace_id': episode['trace_id'],
            'route': episode['route_mode'], 'evidence_class': episode.get('evidence_class', 'live-model'),
            'outcome': outcome, 'reward': reward, 'reward_posted': True,
            'failure_class': diagnostics['failure_class'],
            **{k: diagnostics[k] for k in ('json_syntax', 'format_compliant', 'target_correct', 'time_correct',
                                           'decision_correct', 'evidence_correct', 'asset_record_correct',
                                           'policy_record_correct', 'inventory_correct', 'supporting_runs_correct',
                                           'required_evidence_observed', 'observed')},
            'answer_content': None if terminal else 'empty' if answer == '' else 'whitespace' if not answer.strip() else 'text',
            'policy_errors': out['policy_errors'], 'policy_error_codes': [a['outputs']['policy_error'] for a in actions
                                                                           if a['outputs']['policy_error']],
            # Call identity exactly as the server returned it; nothing here is framework-normalized.
            'sampled_calls': [{'id': a['inputs']['call']['id'], 'name': a['inputs']['call']['name'],
                               'arguments_sha256': hashlib.sha256(a['inputs']['call']['arguments'].encode()).hexdigest(),
                               'policy_error': a['outputs']['policy_error'], 'successful': a['outputs']['successful_tool_call']}
                              for a in actions],
            # Re-derived from the served names: how many the pinned client would have rewritten.
            'tool_names_rewritten_by_framework': sum(re.sub(r'[^a-zA-Z0-9_-]', '_', c['name'])[:64] != c['name']
                                                     for s in served for c in s['tool_calls']),
            'recovered': out['recovered'], 'tool_rounds': rounds, 'llm_calls': len(llms),
            'successful_calls_by_tool': dict(Counter(row['tool'] for row in ledger)),
            'joined_observation_chains': len(chains),
            'prompt_tokens': [u.get('prompt_tokens') for u in usage],
            'completion_tokens': [u.get('completion_tokens') for u in usage],
            'served_finish_reasons': [s['finish_reason'] for s in served],
            # The pinned client's relabelled finish reasons, kept only for comparison.
            'framework_finish_reasons': [(r['outputs'] or {}).get('finish_reason') for r in llms],
            'served_stop_reasons': [s['stop_reason'] for s in served]}


def read_traces(run):
    """Every trace file of a stage directory as (name, records or None)."""
    out = []
    for path in sorted((Path(run)/'traces').glob('trajectory-*.jsonl')):
        try:
            records = [json.loads(line) for line in path.read_text().splitlines()]
            if not records or any(r['trace_id'] != path.stem.removeprefix('trajectory-') for r in records):
                records = None
        except (ValueError, KeyError, TypeError, OSError):
            records = None
        out.append((path.name, records))
    return out


def read_events(run):
    return [json.loads(line) for p in sorted((Path(run)/'audit').glob('events-*.jsonl')) for line in p.read_text().splitlines()]


def episode_table(run, request, evidence_class='live-model'):
    rows = []
    for name, records in read_traces(run):
        if records is None:
            raise ValueError('Unreadable trace or trace filename identity differs')
        rows.append(episode_row(records, request, evidence_class))
    return sorted(rows, key=lambda r: (r['case_id'], r['rollout_id']))


def audit_counts(events):
    counts = Counter(e['event'] for e in events)
    return {**{name: counts[name] for name in FORBIDDEN_EVENTS},
            'weight_refresh_steps': [e['step'] for e in events if e['event'] == 'weight_refresh'],
            'trainer_returned': counts['trainer_returned'], 'wandb_finished': counts['wandb_finished'],
            'collection_manifest': counts['collection_manifest'],
            'development_collection_admitted': counts['development_collection_admitted']}


def check_table(stage, episodes):
    """Membership, multiplicity and identity checks shared by the worker and the reviewer."""
    expected = Counter({cid: stage['rollouts_per_question'] for cid in stage['case_ids']})
    if (len(episodes) != stage['expected_episodes'] or Counter(e['case_id'] for e in episodes) != expected
            or len({e['rollout_id'] for e in episodes}) != len(episodes)
            or len({e['trace_id'] for e in episodes}) != len(episodes)
            or any(e['route'] != stage['route'] or e['reward'] not in (0, 1) or e['reward_posted'] is not True
                   or task.family(e['case_id']) != (e['family'], e['variant']) for e in episodes)):
        raise ValueError('Calibration episodes differ from the declared panel, counts or route')


def complete(run, request, events, evidence_class='live-model'):
    """Worker-side stage completion from the run directory's traces and audit events."""
    return assemble(request, events, episode_table(run, request, evidence_class), evidence_class,
                    checkpoints=bool(list((Path(run)/'checkpoints').glob('global_step_*'))))


def assemble(request, events, episodes, evidence_class='live-model', checkpoints=False):
    """Join reviewed episodes with production audit events. No optimization may have occurred."""
    stage = stage_declaration(request)
    audit = audit_counts(events)
    terminal = [e for e in events if e['event'] == 'trainer_returned']
    if (any(audit[name] for name in FORBIDDEN_EVENTS) or audit['weight_refresh_steps'] != [0]
            or len(terminal) != 1 or audit['wandb_finished'] != 1 or checkpoints):
        raise ValueError('Calibration stage performed optimization, saved a checkpoint, refreshed more than the '
                         'initial policy, or lacks exactly one trainer return and one W&B finish')
    episodes = [dict(e) for e in episodes]
    check_table(stage, episodes)
    if any(e['evidence_class'] != evidence_class for e in episodes):
        raise ValueError('Episode evidence class differs from the stage')
    report = {'schema': STAGE_SCHEMA, 'stage': request['stage']['name'], 'algorithm': request['job']['algorithm'],
              'mode': stage['mode'], 'route': stage['route'], 'temperature': stage['temperature'],
              'panel_id': stage['panel_id'], 'packet_id': stage['packet_id'], 'case_ids': stage['case_ids'],
              'expected_episodes': stage['expected_episodes'], 'evidence_class': evidence_class,
              'seed_streams': request['job']['seed_streams'], 'audit': audit, 'episodes': episodes}
    if stage['mode'] == 'greedy':
        if (terminal[0].get('validation_only') is not True or terminal[0]['global_step'] != 0
                or audit['collection_manifest'] or audit['development_collection_admitted']):
            raise ValueError('Greedy development stage was not a validation-only pass of the initial policy')
        return {'calibration': report, 'optimizer_updates': {'actor': 0, 'critic': 0}}
    admitted = [e for e in events if e['event'] == 'development_collection_admitted']
    manifests = [e for e in events if e['event'] == 'collection_manifest']
    tokens = [e for e in events if e['event'] == 'calibration_collection_tokens']
    if (terminal[0].get('collection_only') is not True or terminal[0]['global_step'] != 1
            or terminal[0].get('optimizer_updates') != 0 or len(admitted) != 1 or len(manifests) != 1 or len(tokens) != 1
            or admitted[0]['real_rollouts'] != 32 or admitted[0]['groups'] != 8):
        raise ValueError('Invalid collection-only audit for a sampled calibration packet')
    sampled = {t['rollout']: t for t in manifests[0]['trajectories']}
    by_rollout = {e['rollout_id']: e for e in episodes}
    groups = {}
    for rollout, row in sampled.items():
        groups.setdefault(row['group'], set()).add(row['case_id'])
    measured = {t['rollout']: t for t in tokens[0]['rollouts']}
    if (len(sampled) != len(manifests[0]['trajectories']) or set(sampled) != set(by_rollout) or set(measured) != set(sampled)
            or any(sampled[r]['case_id'] != e['case_id'] or sampled[r]['reward'] != e['reward'] for r, e in by_rollout.items())
            or len(groups) != 8 or any(len(cases) != 1 for cases in groups.values())
            or any(measured[r]['mask_tokens'] != sampled[r]['original_generated_tokens'] for r in sampled)
            or sum(t['original_generated_tokens'] for t in sampled.values()) != admitted[0]['generated_tokens']
            or tokens[0]['truncated_rows'] or tokens[0]['dropped_prompt_rows']
            # Server-reported completion tokens of the recorded calls equal the adapter's
            # sampled tokens (they did for all 192 Phase 3 training rollouts). A lost or
            # duplicated generation upstream of the adapter therefore fails the join.
            or any(None not in e['completion_tokens'] and sum(e['completion_tokens']) != sampled[r]['original_generated_tokens']
                   for r, e in by_rollout.items())):
        raise ValueError('Sampled rollouts, rewards, groups or tokens differ between the adapter and the traces')
    for rollout, e in by_rollout.items():
        e.update(sampled_tokens=sampled[rollout]['original_generated_tokens'], fragment_rows=sampled[rollout]['fragments'],
                 mask_tokens=measured[rollout]['mask_tokens'], masked_context_tokens=measured[rollout]['context_tokens'],
                 fragment_prompt_tokens=measured[rollout]['prompt_tokens'])
    # Measured values from the adapter's own event, never constants.
    report['collection'] = {'real_rollouts': admitted[0]['real_rollouts'], 'groups': admitted[0]['groups'],
                            'generated_tokens': admitted[0]['generated_tokens'],
                            'fragment_rows': admitted[0]['fragment_rows'],
                            'adapter_constant_groups': admitted[0]['constant_groups'],
                            'limits': tokens[0]['limits'], 'truncated_rows': tokens[0]['truncated_rows'],
                            'dropped_prompt_rows': tokens[0]['dropped_prompt_rows'],
                            'rollouts_with_token_evidence': len(measured),
                            'stopped_before': 'advantages, backward pass, optimizer step and checkpoint save'}
    return {'calibration': report, 'optimizer_updates': {'actor': 0, 'critic': 0}}


def validate_stage(plan, job, stage, proof):
    """Validation of a stage completion summary against the plan.

    Necessary, never sufficient: a summary that passes is internally consistent with
    the plan. Whether it is admitted is decided by review_stage, which also requires
    durable identity, the supporting traces and the audit events.
    """
    declared = plan['calibration']['stages'][stage['name']]
    report = proof.get('calibration')
    if (proof.get('optimizer_updates') != {'actor': 0, 'critic': 0} or not isinstance(report, dict)
            or report.get('schema') != STAGE_SCHEMA or report['stage'] != stage['name']
            or report['algorithm'] != job['algorithm']
            or any(report[k] != declared[k] for k in ('mode', 'route', 'temperature', 'panel_id', 'packet_id',
                                                       'case_ids', 'expected_episodes'))
            or any(report['audit'][name] for name in FORBIDDEN_EVENTS) or report['audit']['weight_refresh_steps'] != [0]
            or report['audit']['trainer_returned'] != 1 or report['audit']['wandb_finished'] != 1
            or report['evidence_class'] not in ('live-model', 'scripted-inference')
            or any(e.get('evidence_class') != report['evidence_class'] for e in report['episodes'])
            or report.get('seed_streams') != job['seed_streams']):
        raise ValueError('Calibration stage evidence differs from the plan, shows optimization or lacks completion events')
    check_table(declared, report['episodes'])
    if declared['mode'] == 'sampled':
        c = report.get('collection') or {}
        if (c.get('real_rollouts') != 32 or c.get('groups') != 8 or c.get('truncated_rows') != 0
                or c.get('dropped_prompt_rows') != 0 or c.get('rollouts_with_token_evidence') != 32
                or c.get('generated_tokens') != sum(e['sampled_tokens'] for e in report['episodes'])
                or any(e['sampled_tokens'] != e['mask_tokens'] or e['sampled_tokens'] <= 0 for e in report['episodes'])):
            raise ValueError('Sampled calibration token or group accounting differs')
    return report


RECOVERY_OUTCOMES = ('complete_final', 'complete_recovered', 'terminal_policy_failure')


def inspect_traces(run, request, declared_class):
    """Diagnostic reading of whatever traces exist. Never raises for bad evidence.

    Returns facts, the rows that pass review, and one entry per trace file with the
    slot it claims (case and rollout), whether a reward delivery was confirmed and
    whether the episode completed.
    """
    facts = {'trace_files': 0, 'unreadable': 0, 'rejected_untrusted': 0, 'rejection_reasons': Counter(),
             'incomplete_exceptions': Counter(), 'evidence_classes': Counter(), 'model_calls': 0,
             'model_calls_with_reported_usage': 0}
    rows, attempts = [], []
    for name, records in read_traces(run):
        facts['trace_files'] += 1
        try:
            if records is None or not all(isinstance(r, dict) for r in records):
                raise ValueError('unreadable trace')
            episodes = [r for r in records if r.get('kind') == 'episode']
            episode = episodes[0] if len(episodes) == 1 else {}
            out = episode.get('outputs') if isinstance(episode.get('outputs'), dict) else {}
            calls = [r for r in records if r.get('kind') == 'llm']
            facts['model_calls'] += len(calls)
            facts['model_calls_with_reported_usage'] += sum(
                isinstance(r.get('outputs'), dict) and bool(r['outputs'].get('usage')) for r in calls)
            facts['evidence_classes'][str(episode.get('evidence_class', 'unrecorded'))] += 1
            confirmed = episode.get('execution_status') == 'ok' and out.get('reward_posted') is True
            try:
                completed = reader_integration.episode_outcome(records) in RECOVERY_OUTCOMES
            except Exception:
                completed = False
            exception = episode.get('exception') if isinstance(episode.get('exception'), dict) else {}
            attempt = {'case_id': records[0].get('case_id'), 'rollout_id': records[0].get('rollout_id'),
                       'confirmed': confirmed, 'completed': completed}
        except Exception:
            # Unreadable or malformed: a record that exists and cannot be trusted or placed.
            facts['unreadable'] += 1
            facts['rejected_untrusted'] += 1
            facts['rejection_reasons']['unreadable or malformed trace'] += 1
            continue
        attempts.append(attempt)
        if not confirmed:
            # A real attempt without a confirmed reward: missing, never zero.
            facts['incomplete_exceptions'][str(exception.get('type', 'no terminal episode record'))] += 1
            continue
        try:
            rows.append(episode_row(records, request, declared_class))
        except Exception as exc:
            facts['rejected_untrusted'] += 1
            facts['rejection_reasons'][f'{type(exc).__name__}: {exc}'[:160]] += 1
    for key in ('rejection_reasons', 'incomplete_exceptions', 'evidence_classes'):
        facts[key] = dict(facts[key])
    return facts, sorted(rows, key=lambda r: (r['case_id'], r['rollout_id'])), attempts


def slot_accounting(declared, attempts):
    """Count expected (case, rollout) slots, not files.

    A slot is one of the declared rollouts of a declared case. Several traces of one
    rollout fill one slot; a slot's reward is confirmed if any of them confirmed it.
    Traces beyond the declared multiplicity, for other cases, or unreadable ones fill
    no slot and are reported as extra.
    """
    per = declared['rollouts_per_question']
    by_case = {}
    for a in attempts:
        if a['case_id'] in declared['case_ids'] and isinstance(a['rollout_id'], str) and a['rollout_id']:
            slot = by_case.setdefault(a['case_id'], {}).setdefault(a['rollout_id'], {'confirmed': False, 'completed': False})
            slot['confirmed'] |= a['confirmed']
            slot['completed'] |= a['completed']
    observed = confirmed = completed = 0
    for case_id in declared['case_ids']:
        slots = sorted(by_case.get(case_id, {}).values(), key=lambda s: (not s['confirmed'], not s['completed']))[:per]
        observed += len(slots)
        confirmed += sum(s['confirmed'] for s in slots)
        completed += sum(s['completed'] for s in slots)
    return observed, completed, confirmed


def token_coverage(declared, events, facts):
    """What token, fragment, truncation and dropped-row evidence exists, with its coverage.

    A value is None when no evidence covers it. It is never reported as zero by
    default, and absent local files are not read as a statement about the remote run.
    """
    out = {'model_calls_observed': facts['model_calls'] if facts else None,
           'model_calls_with_reported_usage': facts['model_calls_with_reported_usage'] if facts else None}
    if declared['mode'] != 'sampled':
        return {**out, 'adapter_evidence': 'not applicable to a greedy stage'}
    empty = {**out, 'rollouts_expected': declared['expected_episodes'], 'rollouts_in_adapter_manifest': None,
             'rollouts_with_token_evidence': None, 'sampled_tokens': None, 'mask_tokens': None, 'masked_context_tokens': None,
             'fragment_rows': None, 'largest_fragment_prompt_tokens': None, 'truncated_rows': None, 'dropped_prompt_rows': None}
    if events is None:
        return {**empty, 'adapter_evidence': 'unknown: no readable audit events are available locally'}
    try:
        measured = [e for e in events if e['event'] == 'calibration_collection_tokens']
        manifests = [e for e in events if e['event'] == 'collection_manifest']
        if len(measured) > 1 or len(manifests) > 1:
            return {**empty, 'adapter_evidence': 'ambiguous: more than one collection is recorded in the audit events'}
        covered = measured[0]['rollouts'] if measured else []
        return {**empty,
                'rollouts_in_adapter_manifest': len(manifests[0]['trajectories']) if manifests else None,
                'rollouts_with_token_evidence': len(covered),
                'sampled_tokens': sum(t['original_generated_tokens'] for t in manifests[0]['trajectories']) if manifests else None,
                'mask_tokens': sum(t['mask_tokens'] for t in covered) if covered else None,
                'masked_context_tokens': sum(t['context_tokens'] for t in covered) if covered else None,
                'fragment_rows': sum(len(t['prompt_tokens']) for t in covered) if covered else None,
                'largest_fragment_prompt_tokens': max((p for t in covered for p in t['prompt_tokens']), default=None),
                'truncated_rows': measured[0]['truncated_rows'] if measured else None,
                'dropped_prompt_rows': measured[0]['dropped_prompt_rows'] if measured else None,
                'adapter_evidence': ('measured by the collection adapter' if measured else
                                     'the collected audit events hold no adapter token measurement, so truncation and '
                                     'dropped rows are unknown')}
    except (KeyError, TypeError, AttributeError):
        return {**empty, 'adapter_evidence': 'unknown: the audit events are malformed'}


def review_stage(request, *, supervisor=None, receipt=None, proof=None, durable=None, run=None):
    """One stage's evidence: its diagnostics, and whether the packet is admitted.

    Admission is strict and has one definition. A packet is admitted only when
    (1) its completion summary is valid for the plan, (2) the durable identity join
    of supervisor index, receipt, completion and observability passed, (3) every
    expected episode is recomputed from the collected traces, and (4) the stage
    report recomputed from those traces and the collected audit events equals the
    completion summary. A summary, or a field that says "live-model", is never enough.

    Diagnostics are separate and tolerant; malformed evidence is reported, not
    raised. Missing local files never establish that remote work did not happen:
    unattempted work is claimed only when the supervisor index shows the stage was
    never dispatched and nothing local contradicts it. Every expected slot of a
    dispatched stage that has no local trace is unknown.
    """
    plan, job, stage = request['plan'], request['job'], request['stage']
    declared = plan['calibration']['stages'][stage['name']]
    expected = declared['expected_episodes']
    problems, checks = [], {}
    supervisor = supervisor if isinstance(supervisor, dict) else {}
    receipt = receipt if isinstance(receipt, dict) else {}
    proof = proof if isinstance(proof, dict) else None
    status = supervisor.get('status', 'unavailable')
    summary = declared_class = None
    if proof is None:
        checks['completion_summary'] = 'absent'
    else:
        try:
            summary = validate_stage(plan, job, stage, proof)
            checks['completion_summary'] = 'valid'
            declared_class = summary['evidence_class']
        except Exception as exc:
            checks['completion_summary'] = 'invalid'
            problems.append(f'completion summary: {type(exc).__name__}: {exc}'[:240])
    if durable is None:
        checks['durable_identity'] = 'not checked: no supervisor index or receipts were available'
    else:
        checks['durable_identity'] = 'verified' if durable['verified'] else 'failed'
        problems += [f'durable identity: {p}' for p in durable['problems']]

    facts = rows = attempts = events = None
    traces_present = run is not None and (Path(run)/'traces').is_dir() and any((Path(run)/'traces').glob('trajectory-*.jsonl'))
    audit_present = run is not None and (Path(run)/'audit').is_dir() and any((Path(run)/'audit').glob('events-*.jsonl'))
    checks['traces'] = 'present' if traces_present else 'absent locally'
    checks['audit_events'] = 'present' if audit_present else 'absent locally'
    if traces_present:
        # Without a valid summary there is no declared class; review as live, the strictest reading.
        facts, rows, attempts = inspect_traces(run, request, declared_class or 'live-model')
    if audit_present:
        try:
            events = read_events(run)
            if not all(isinstance(e, dict) and isinstance(e.get('event'), str) for e in events):
                raise ValueError('malformed audit event')
            audit = audit_counts(events)
        except Exception as exc:
            events = None
            problems.append(f'audit events unreadable or malformed: {type(exc).__name__}')
    recomputed = False
    if summary is not None and traces_present and events is not None:
        try:
            if any(e.get('id') != stage['wandb']['id'] for e in events if e['event'] == 'wandb_finished'):
                raise ValueError('W&B finish ID differs from the planned stage run ID')
            again = assemble(request, events, rows, summary['evidence_class'],
                             checkpoints=bool(list((Path(run)/'checkpoints').glob('global_step_*'))))
            recomputed = (facts['trace_files'] == expected and len(rows) == expected
                          and json.loads(json.dumps(again)) == {k: proof[k] for k in ('calibration', 'optimizer_updates')})
            if not recomputed:
                problems.append('the stage report recomputed from traces and audit events differs from the completion summary')
        except Exception as exc:
            problems.append(f'recomputation from traces and audit events: {type(exc).__name__}: {exc}'[:240])
    elif summary is not None:
        problems.append('supporting traces or audit events are not available, so the summary cannot be rechecked')
    checks['recomputed_from_traces_and_audit'] = recomputed
    admitted = summary is not None and durable is not None and durable['verified'] and recomputed

    # Execution status, from the supervisor first and the stage's own receipt second.
    local_evidence = traces_present or audit_present or proof is not None or bool(receipt)
    never_dispatched = bool(supervisor) and status == 'skipped' and not supervisor.get('call_id')
    if admitted:
        packet = 'admitted'
    elif never_dispatched and not local_evidence:
        packet = 'unattempted'
    elif never_dispatched:
        packet = 'rejected'
        problems.append('the supervisor index records the stage as never dispatched, but local evidence exists for it')
    elif status in ('failed', 'cancelled') or receipt.get('status') == 'failed':
        packet = 'failed'  # known to have failed, whether or not it left any episode evidence
    elif status in ('submitted', 'started', 'planned', 'unknown'):
        packet = 'unknown'  # not known to have ended; partial evidence is diagnostic only
    elif traces_present or proof is not None:
        packet = 'rejected'
    else:
        packet = 'unknown'  # nothing local and no evidence about what happened

    episodes = {'expected': expected, 'admitted': expected if admitted else 0}
    if packet == 'unattempted':
        episodes.update(trace_files=0, observed_attempts=0, completed=0, rewards_confirmed=0, rewards_missing=0,
                        rewards_confirmed_not_admitted=0, rejected_untrusted=0, reviewable=0, unexpected_extra_attempts=0,
                        unattempted=expected, unknown=0)
    elif facts is not None:
        observed, completed, confirmed = slot_accounting(declared, attempts)
        episodes.update(trace_files=facts['trace_files'], observed_attempts=observed, completed=completed,
                        rewards_confirmed=confirmed, rewards_missing=observed - confirmed,
                        rewards_confirmed_not_admitted=confirmed - episodes['admitted'],
                        rejected_untrusted=facts['rejected_untrusted'], reviewable=len(rows),
                        unexpected_extra_attempts=facts['trace_files'] - observed,
                        # A slot with no local trace is not known to have run or not run.
                        unattempted=0, unknown=expected - observed)
    else:
        # A summary alone, or nothing at all: what was attempted is not established here.
        episodes.update(trace_files=None, observed_attempts=None, completed=None, rewards_confirmed=None, rewards_missing=None,
                        rewards_confirmed_not_admitted=None, rejected_untrusted=None, reviewable=None,
                        unexpected_extra_attempts=None, unattempted=0, unknown=expected)
    return {'stage': stage['name'], 'mode': declared['mode'], 'packet_id': declared['packet_id'], 'packet_status': packet,
            'evidence_class': summary['evidence_class'] if admitted else None,
            'execution': {'supervisor_status': status, 'dispatched': (True if supervisor.get('call_id') else
                                                                      False if never_dispatched else None),
                          'receipt_status': receipt.get('status', 'unavailable'), 'receipt_error': receipt.get('error')},
            'episodes': episodes, 'checks': checks, 'problems': problems,
            'trace_facts': facts, 'token_evidence': token_coverage(declared, events, facts),
            'audit': audit if events is not None else None,
            'rows': rows if admitted else [],
            # Reviewable episodes of a packet that is not admitted: for diagnosis only.
            'diagnostic_rows': [] if admitted or rows is None else rows,
            'summary_report': summary if admitted else None}


def _summary(episodes):
    """Counts with explicit denominators. Unassessable values are counted separately."""
    n = len(episodes)
    def count(key, value=True):
        return sum(e[key] is value for e in episodes)
    assessable = [e for e in episodes if e['decision_correct'] is not None]
    rewards = sum(e['reward'] for e in episodes)
    return {'episodes': n, 'reward_sum': rewards, 'reward_mean': rewards / n if n else None,
            'reward_wilson95': wilson(rewards, n),
            'json_syntax_valid': count('json_syntax'), 'contract_format_valid': count('format_compliant'),
            'target_correct': count('target_correct'), 'time_correct': count('time_correct'),
            'decision_assessable': len(assessable), 'decision_correct': sum(e['decision_correct'] for e in assessable),
            'evidence_correct': sum(e['evidence_correct'] is True for e in assessable),
            'decision_correct_evidence_invalid': sum(e['decision_correct'] and not e['evidence_correct'] for e in assessable),
            'decision_not_assessable': n - len(assessable),
            'failure_classes': dict(Counter(e['failure_class'] for e in episodes)),
            'outcomes': dict(Counter(e['outcome'] for e in episodes)),
            'answer_content': dict(Counter(str(e['answer_content']) for e in episodes)),
            'typed_action_errors': sum(e['policy_errors'] for e in episodes),
            'typed_action_error_codes': dict(Counter(code for e in episodes for code in e['policy_error_codes'])),
            'unknown_tool_names_as_sampled': dict(Counter(c['name'] for e in episodes for c in e['sampled_calls']
                                                          if c['policy_error'] == 'unknown_tool')),
            'tool_names_rewritten_by_framework': sum(e['tool_names_rewritten_by_framework'] for e in episodes),
            'recovered_episodes': sum(e['recovered'] for e in episodes),
            'terminal_policy_failures': sum(e['outcome'] == 'terminal_policy_failure' for e in episodes),
            'episodes_with_successful_call': {name: sum(e['successful_calls_by_tool'].get(name, 0) > 0 for e in episodes)
                                              for name in task.TOOLS},
            'episodes_with_required_evidence_observed': count('required_evidence_observed'),
            'correctly_joined_observation_chains': sum(e['joined_observation_chains'] for e in episodes),
            'tool_rounds': dict(sorted(Counter(e['tool_rounds'] for e in episodes).items())),
            'model_calls': sum(e['llm_calls'] for e in episodes),
            'episodes_without_reported_usage': sum(None in e['completion_tokens'] for e in episodes),
            'prompt_tokens_max': max((t for e in episodes for t in e['prompt_tokens'] if t is not None), default=None),
            'completion_tokens_reported': sum(t for e in episodes for t in e['completion_tokens'] if t is not None),
            'served_finish_reasons': dict(Counter(f for e in episodes for f in e['served_finish_reasons']))}


def _groups(episodes):
    by_case = {}
    for e in sorted(episodes, key=lambda e: e['rollout_id']):
        by_case.setdefault(e['case_id'], []).append(e)
    rows = []
    interface, substantive = CRITERIA['interface_failure_classes'], CRITERIA['substantive_failure_classes']
    for case_id, members in by_case.items():
        zeros = [e['failure_class'] for e in members if e['reward'] == 0]
        if not zeros:
            kind = 'constant_one'
        elif len(zeros) == len(members):
            kind = 'constant_zero'
        elif any(c in substantive for c in zeros):
            kind = 'mixed_semantic_evidence'
        else:
            kind = 'mixed_format_action_only'
        rows.append({'case_id': case_id, 'family': members[0]['family'], 'variant': members[0]['variant'],
                     'rewards': [e['reward'] for e in members], 'failure_classes': [e['failure_class'] for e in members],
                     'group_type': kind})
    zeros = [e['failure_class'] for e in episodes if e['reward'] == 0]
    return rows, {'groups': len(rows), **{k: sum(r['group_type'] == k for r in rows) for k in
                  ('constant_zero', 'constant_one', 'mixed_semantic_evidence', 'mixed_format_action_only')},
                  'mixed': sum(r['group_type'].startswith('mixed') for r in rows),
                  'semantic_evidence_informative_families': sorted({r['family'] for r in rows if r['group_type'] == 'mixed_semantic_evidence'}),
                  'zero_rewards': len(zeros), 'interface_zero_rewards': sum(c in interface for c in zeros),
                  'substantive_zero_rewards': sum(c in substantive for c in zeros)}


def _by_family(episodes):
    return {family: {'episodes': len(rows), 'reward_sum': sum(e['reward'] for e in rows),
                     'failure_classes': dict(Counter(e['failure_class'] for e in rows))}
            for family in task.FAMILIES for rows in [[e for e in episodes if e['family'] == family]]}


def screen(greedy, sampled, groups):
    """The predeclared screen applied to one route's complete rows. Pure arithmetic."""
    c = CRITERIA
    informative = groups['mixed_semantic_evidence']
    families = groups['semantic_evidence_informative_families']
    share = groups['interface_zero_rewards'] / groups['zero_rewards'] if groups['zero_rewards'] else None
    passed = (informative >= c['min_semantic_evidence_informative_groups'] and len(families) >= c['min_case_families']
              and share is not None and share <= c['max_interface_share_of_zero_rewards'])
    if passed:
        verdict = 'screen_passed'
    elif groups['constant_one'] > c['groups'] - c['min_semantic_evidence_informative_groups']:
        # Too many all-correct groups for the screen to pass, whatever the few zeros are.
        verdict = 'saturated'
    elif share is not None and share > c['max_interface_share_of_zero_rewards']:
        verdict = 'interface_dominated'
    elif groups['constant_zero'] > c['groups'] - c['min_semantic_evidence_informative_groups']:
        verdict = 'too_difficult'
    else:
        verdict = 'insufficient_semantic_variation'
    return {'verdict': verdict, 'semantic_evidence_informative_groups': informative, 'of_groups': groups['groups'],
            'informative_case_families': families, 'interface_share_of_zero_rewards': share,
            'greedy_successes_and_failures_both_observed': 0 < greedy['reward_sum'] < greedy['episodes'],
            'sampled_successes_and_failures_both_observed': 0 < sampled['reward_sum'] < sampled['episodes']}


def decide(reviews, results, plan_problems, route_problems=()):
    """Operational validity and scientific suitability, as separate decisions.

    A live scientific decision is given only for a route whose four packets are all
    admitted from live evidence under an executable plan. Anything else is explicitly
    invalid, unverified or mechanics-only, with its reasons, and stays unassessed.
    """
    reasons = list(plan_problems) + list(route_problems)
    statuses = {r['stage']: r['packet_status'] for r in reviews}
    reasons += [f"{name}: packet {status}" for name, status in statuses.items() if status != 'admitted']
    complete = all(status == 'admitted' for status in statuses.values())
    classes = {r['evidence_class'] for r in reviews if r['packet_status'] == 'admitted'}
    live = complete and classes == {'live-model'}
    if complete and not live:
        reasons.append('evidence is scripted or simulated, not a live model' if classes == {'scripted-inference'}
                       else 'evidence classes are mixed within the route')
    if live:
        rows = [e for r in reviews for e in r['rows']]
        limit = task.CONTEXT_LIMITS['trajectory_max_prompt_length']
        # Every generation of every episode must carry its own served usage.
        if any(None in e['completion_tokens'] or None in e['prompt_tokens'] or not all(p > 0 for p in e['prompt_tokens'])
               for e in rows):
            reasons.append('some generations have no reported usage, so completion, prompt size and token accounting '
                           'are not established')
        elif any(p > limit for e in rows for p in e['prompt_tokens']):
            reasons.append('a served prompt exceeds the declared trajectory prompt limit')
    if not complete:
        validity = 'invalid'
    elif not live:
        validity = 'mechanics_only_not_live'
    elif reasons:
        validity = 'invalid'
    else:
        validity = 'valid'
    out = {'operational_validity': validity, 'reasons': reasons,
           'scientific_suitability': 'not_assessed', 'packet_status': statuses}
    if results is not None:
        applied = screen(results['greedy'], results['sampled'], results['sampled']['groups'])
        verdict = applied.pop('verdict')
        out.update(applied)
        if validity == 'valid':
            out['scientific_suitability'] = verdict
        else:
            out['screen_on_unadmitted_or_non_live_evidence'] = verdict
    if validity == 'valid':
        passed = out['scientific_suitability'] == 'screen_passed'
        out['next_step'] = ('eligible for a separately proposed controlled learning study; not evidence of learning' if passed
                            else 'revise the task, prompt or harness using development evidence under new identities; '
                                 'do not repeat an unchanged collection')
        out['interpretation'] = ('an engineering screen was passed; this is not evidence of learning' if passed
                                 else 'a negative screen is a valid calibration result')
    elif validity == 'mechanics_only_not_live':
        out['next_step'] = 'none: scripted or simulated evidence supports no calibration decision'
        out['interpretation'] = 'scripted or simulated evidence: harness mechanics only, never a policy measurement'
    else:
        out['next_step'] = ('diagnose the failure from the stage diagnostics, reconcile any unknown execution with the '
                            'provider, and use a fresh attempt; nothing here is scientifically assessed')
        out['interpretation'] = 'incomplete, unverified or untrusted evidence is diagnosed, never admitted or interpreted'
    return out


def route_results(reviews, declared):
    """Scientific results of a route whose four packets are all admitted, else None."""
    if any(r['packet_status'] != 'admitted' for r in reviews):
        return None
    greedy = [e for r in reviews if r['mode'] == 'greedy' for e in r['rows']]
    sampled = [e for r in reviews if r['mode'] == 'sampled' for e in r['rows']]
    per_question, groups = _groups(sampled)
    packets = {r['stage']: {'packet_id': r['packet_id'], **_groups(r['rows'])[1], 'collection': r['summary_report']['collection']}
               for r in reviews if r['mode'] == 'sampled'}
    return {'greedy': {**_summary(greedy), 'by_family': _by_family(greedy),
                       'per_question_reward': {e['case_id']: e['reward'] for e in greedy}},
            'sampled': {**_summary(sampled), 'by_family': _by_family(sampled), 'groups': groups, 'by_packet': packets,
                        'per_question': per_question}}


def report(plan, reviews, plan_problems=(), timings=None, provenance=None, route_problems=None):
    """Machine-readable calibration report from reviewed stages (see review_stage).

    reviews[algorithm] is the ordered list of stage reviews of that route. This
    function does arithmetic on reviews; it does not establish them. The entry
    points that may produce a live decision build reviews only through
    reader_evidence.calibration_collection. Admission and diagnostics are kept
    apart: `accounting` and `stages` describe everything that is known, including
    failures and unknowns; `results` and a scientific decision exist only for a
    route whose every packet is admitted. Nothing is zero-filled, padded, dropped or
    pooled across routes.
    """
    declared = plan['calibration']
    if modes.phase(plan) != modes.CALIBRATION or declared != declaration():
        raise ValueError('Not a current recovery calibration plan')
    out_routes, admitted_classes = {}, set()
    for job in plan['jobs']:
        route = reviews[job['algorithm']]
        if [r['stage'] for r in route] != [s['name'] for s in job['stages']]:
            raise ValueError('Stage reviews differ from the planned stages')
        results = route_results(route, declared)
        totals = {'expected': declared['episodes_per_algorithm']}
        for key in ('admitted', 'unattempted', 'unknown'):
            totals[key] = sum(r['episodes'][key] for r in route)
        for key in ('trace_files', 'observed_attempts', 'completed', 'rewards_confirmed', 'rewards_missing',
                    'rewards_confirmed_not_admitted', 'rejected_untrusted', 'reviewable', 'unexpected_extra_attempts'):
            values = [r['episodes'][key] for r in route if r['episodes'][key] is not None]
            # Unknown when no stage has the evidence; otherwise a sum over the stages that do.
            totals[key] = sum(values) if values else None
        totals['stages_with_trace_evidence'] = sum(r['trace_facts'] is not None for r in route)
        totals['stages'] = len(route)
        totals['note'] = ('expected = observed attempts + unattempted + unknown, counted in expected (case, rollout) slots; '
                          'observed = rewards confirmed + missing; confirmed = admitted + confirmed but not admitted. '
                          'Slot counts cover stages with collected traces or shown never to have been dispatched; every '
                          'other slot is unknown, never missing and never zero')
        classes = {r['evidence_class'] for r in route if r['packet_status'] == 'admitted'}
        admitted_classes |= classes
        admitted_reports = {r['stage']: r['summary_report'] for r in route if r['packet_status'] == 'admitted'}
        diagnostics = {}
        for r in route:
            if r['packet_status'] == 'admitted':
                continue
            rows = r['diagnostic_rows']
            diagnostics[r['stage']] = {
                'packet_status': r['packet_status'], 'problems': r['problems'], 'checks': r['checks'],
                'execution': r['execution'], 'episodes': r['episodes'], 'trace_facts': r['trace_facts'],
                'token_evidence': r['token_evidence'], 'audit': r['audit'],
                'reviewable_episode_summary': _summary(rows) if rows else None,
                'reviewable_reward_vectors': _groups(rows)[0] if rows and r['mode'] == 'sampled' else None,
                'label': 'diagnostic only: not admitted, not scientifically assessed, no group padded or zero-filled'}
        out_routes[job['algorithm']] = {
            'evidence_class': ('none admitted' if not classes else 'live-model' if classes == {'live-model'}
                               else 'scripted-inference' if classes == {'scripted-inference'} else 'mixed'),
            'accounting': totals,
            'stages': {r['stage']: {k: r[k] for k in ('mode', 'packet_id', 'packet_status', 'evidence_class', 'execution',
                                                      'episodes', 'checks', 'problems', 'token_evidence', 'audit')}
                       for r in route},
            'results': results,
            'failure_diagnostics': diagnostics,
            'optimization_events': {name: sum(s['audit'][name] for s in admitted_reports.values()) for name in FORBIDDEN_EVENTS}
                                   if admitted_reports else None,
            'optimization_events_coverage': f'{len(admitted_reports)} of {len(route)} stages (admitted stages only)',
            'weight_refresh_steps_by_stage': {name: s['audit']['weight_refresh_steps'] for name, s in admitted_reports.items()},
            'decision': decide(route, results, plan_problems, (route_problems or {}).get(job['algorithm'], ()))}
    valid = [r['decision']['operational_validity'] == 'valid' for r in out_routes.values()]
    return {'schema': REPORT_SCHEMA, 'task': task.TASK, 'episode_policy': task.EPISODE_POLICY['version'],
            # The class of the evidence that is admitted, over all routes. Not a statement of completeness.
            'evidence_class': ('none admitted' if not admitted_classes else 'live-model' if admitted_classes == {'live-model'}
                               else 'scripted-inference' if admitted_classes == {'scripted-inference'} else 'mixed'),
            # True only when every route is a valid live calibration under the single admission definition.
            'live_policy_measurement': bool(valid) and all(valid),
            'provenance': provenance or 'not verified: no supervisor index was supplied',
            'plan_problems': list(plan_problems),
            'panel_id': declared['panel_id'], 'packets': declared['packets'], 'split': 'development',
            'final_evaluation': declared['final_evaluation'], 'criteria': CRITERIA,
            'expected_total_episodes': declared['total_episodes'], 'optimizer_updates_declared': 0,
            'routes': out_routes, 'stage_timings_seconds': timings or 'not supplied',
            'resource_bounds': {'gpu_stages': sum(len(j['stages']) for j in plan['jobs']),
                                'stage_seconds_limit': plan['bounds']['stage_seconds'],
                                'max_gpu_seconds': plan['bounds']['max_gpu_seconds'],
                                'reserved_budget_usd': plan['bounds']['budget_usd'],
                                'hourly_ceiling_usd': plan['bounds']['all_resource_hourly_ceiling_usd'],
                                'note': 'reservation bounds, not spending; observed stage times are above when supplied'},
            'unverified_assumptions': [
                'Sampling temperatures are set by the gateway routes named in each stage; they are checked in the '
                'worker launch command, not observed per request.',
                'Token counts for greedy episodes are server-reported usage; sampled-token counts come from the adapter.',
                'Scripted or simulated evidence says nothing about how the model behaves on this task.'],
            'claims': {'local_operation': 'established only by executed offline tests',
                       'harness_effects': 'not measured by this report',
                       'useful_signal': 'assessed per route by the fixed screen, only on admitted live evidence',
                       'learning_improvement': 'not assessed; calibration performs zero optimization'}}


def summary(result):
    """Readable summary of a report; states every denominator and what is unknown."""
    def show(value):
        return 'unknown' if value is None else f'{value:g}' if isinstance(value, float) else str(value)
    lines = [f"# Recovery readiness development calibration ({result['evidence_class']})", '',
             f"Panel `{result['panel_id']}`, split `{result['split']}`, {result['expected_total_episodes']} expected episodes, "
             'zero optimizer updates declared.', '', f"Provenance: {result['provenance']}.", '']
    if not result['live_policy_measurement']:
        lines += ['**Not a live policy measurement.** The evidence is scripted, simulated, incomplete or unverified; '
                  'nothing below is a statement about how a model performs.', '']
    if result['plan_problems']:
        lines += ['Plan problems: ' + '; '.join(result['plan_problems']), '']
    for algorithm, r in result['routes'].items():
        a, d = r['accounting'], r['decision']
        lines += [f"## {algorithm} ({r['evidence_class']})", '',
                  f"- Operational validity: **{d['operational_validity']}**. Scientific suitability: **{d['scientific_suitability']}**.",
                  f"- Episodes: {a['expected']} expected; {a['admitted']} admitted; {show(a['observed_attempts'])} attempts observed, "
                  f"{show(a['completed'])} completed, {show(a['rewards_confirmed'])} rewards confirmed, "
                  f"{show(a['rewards_missing'])} missing; {a['unattempted']} unattempted; {a['unknown']} unknown.",
                  f"- Records: {show(a['trace_files'])} trace files, {show(a['rejected_untrusted'])} rejected as untrusted, "
                  f"{show(a['unexpected_extra_attempts'])} beyond the expected slots "
                  f"(trace evidence for {a['stages_with_trace_evidence']} of {a['stages']} stages).",
                  '- Packets: ' + ', '.join(f"{name} {s['packet_status']}" for name, s in r['stages'].items()) + '.']
        if d['reasons']:
            lines += ['- Reasons: ' + '; '.join(d['reasons']) + '.']
        if r['results']:
            g, s, groups = r['results']['greedy'], r['results']['sampled'], r['results']['sampled']['groups']
            lines += [f"- Greedy: {g['reward_sum']:g}/{g['episodes']} rewarded (95% Wilson {g['reward_wilson95']}); "
                      f"contract-valid {g['contract_format_valid']}/{g['episodes']}; decision correct "
                      f"{g['decision_correct']}/{g['decision_assessable']} assessable; evidence correct {g['evidence_correct']}/{g['decision_assessable']}.",
                      f"- Sampled: {s['reward_sum']:g}/{s['episodes']} rewarded (95% Wilson {s['reward_wilson95']}); "
                      f"contract-valid {s['contract_format_valid']}/{s['episodes']}; decision correct "
                      f"{s['decision_correct']}/{s['decision_assessable']} assessable; evidence correct {s['evidence_correct']}/{s['decision_assessable']}.",
                      f"- Groups ({groups['groups']}): {groups['constant_zero']} constant-zero, {groups['constant_one']} constant-one, "
                      f"{groups['mixed_semantic_evidence']} mixed with semantic/evidence failures, "
                      f"{groups['mixed_format_action_only']} mixed by format/action only.",
                      f"- Zero rewards: {groups['zero_rewards']} ({groups['interface_zero_rewards']} interface, "
                      f"{groups['substantive_zero_rewards']} substantive).",
                      f"- Typed action errors {s['typed_action_errors']}, recovered episodes {s['recovered_episodes']}, "
                      f"terminal policy failures {s['terminal_policy_failures']} (sampled).",
                      f"- Optimization events: {r['optimization_events']}; refresh steps {r['weight_refresh_steps_by_stage']}."]
            if 'screen_on_unadmitted_or_non_live_evidence' in d:
                lines += [f"- Screen applied to non-live evidence, for mechanics only: {d['screen_on_unadmitted_or_non_live_evidence']} "
                          f"({d['semantic_evidence_informative_groups']}/{d['of_groups']} informative groups)."]
        for name, f in r['failure_diagnostics'].items():
            e = f['episodes']
            lines += [f"- Diagnostic, {name} ({f['packet_status']}): {show(e['observed_attempts'])} attempts, "
                      f"{show(e['rewards_confirmed'])} rewards confirmed, {show(e['rewards_missing'])} missing, "
                      f"{e['unattempted']} unattempted, {e['unknown']} unknown. "
                      + ('; '.join(f['problems'][:3]) if f['problems'] else 'No further detail.')]
        lines += ['']
    c = result['criteria']
    lines += ['## Screen', '',
              f"At least {c['min_semantic_evidence_informative_groups']} of {c['groups']} sampled groups with "
              f"semantic/evidence variation, across at least {c['min_case_families']} case families, with interface "
              f"failures at most {c['max_interface_share_of_zero_rewards']:.0%} of zero rewards, per algorithm route. "
              'This is an engineering heuristic, not a research threshold or proof of learning. It is applied as a '
              'decision only to admitted live evidence.', '']
    return '\n'.join(lines)
