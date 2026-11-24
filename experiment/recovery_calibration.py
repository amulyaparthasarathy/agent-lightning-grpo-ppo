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
import json
import math
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


def episode_row(records, request, evidence_class='live-model'):
    """One reviewed episode from its own trace, re-verified from recorded evidence.

    Raises for an incomplete, cross-stage, cross-source or internally inconsistent
    trace. Contains no raw model text and no labels beyond the trusted family name.
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
    ledger = [{'tool': a['inputs']['call']['name'], 'arguments': json.loads(a['inputs']['call']['arguments']),
               'content': a['inputs']['observation']['content']} for a in actions if a['outputs']['successful_tool_call']]
    terminal = outcome == 'terminal_policy_failure'
    reward, diagnostics = task.assess('' if terminal else out['raw_answer'], trusted, ledger)
    if terminal:
        reward = 0.0
        diagnostics.update(json_syntax=None, format_compliant=None, failure_class='action')
    if reward != out['reward'] or diagnostics != out['diagnostics'] or out['successful_tool_calls'] != len(ledger):
        raise ValueError('Recorded reward or diagnostics differ from independent re-verification')
    usage = [(r['outputs'] or {}).get('usage') or {} for r in llms]
    family, variant = task.family(case['case_id'])
    return {'case_id': episode['case_id'], 'family': family, 'variant': variant,
            'rollout_id': episode['rollout_id'], 'attempt_id': episode['attempt_id'], 'trace_id': episode['trace_id'],
            'route': episode['route_mode'], 'outcome': outcome, 'reward': reward, 'reward_posted': True,
            'failure_class': diagnostics['failure_class'],
            **{k: diagnostics[k] for k in ('json_syntax', 'format_compliant', 'target_correct', 'time_correct',
                                           'decision_correct', 'evidence_correct', 'asset_record_correct',
                                           'policy_record_correct', 'inventory_correct', 'supporting_runs_correct',
                                           'required_evidence_observed', 'observed')},
            'policy_errors': out['policy_errors'], 'policy_error_codes': [a['outputs']['policy_error'] for a in actions
                                                                           if a['outputs']['policy_error']],
            'recovered': out['recovered'], 'tool_rounds': out['tool_rounds'], 'llm_calls': len(llms),
            'successful_calls_by_tool': dict(Counter(row['tool'] for row in ledger)),
            'joined_observation_chains': len(chains),
            'prompt_tokens': [u.get('prompt_tokens') for u in usage],
            'completion_tokens': [u.get('completion_tokens') for u in usage],
            'finish_reasons': [(r['outputs'] or {}).get('finish_reason') for r in llms]}


def episode_table(run, request, evidence_class='live-model'):
    rows = []
    for path in sorted((Path(run)/'traces').glob('trajectory-*.jsonl')):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if not records or any(r['trace_id'] != path.stem.removeprefix('trajectory-') for r in records):
            raise ValueError('Trace filename identity differs')
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
            or len(terminal) != 1 or checkpoints):
        raise ValueError('Calibration stage performed optimization, saved a checkpoint or refreshed more than the initial policy')
    episodes = [dict(e) for e in episodes]
    check_table(stage, episodes)
    report = {'schema': STAGE_SCHEMA, 'stage': request['stage']['name'], 'algorithm': request['job']['algorithm'],
              'mode': stage['mode'], 'route': stage['route'], 'temperature': stage['temperature'],
              'panel_id': stage['panel_id'], 'packet_id': stage['packet_id'], 'case_ids': stage['case_ids'],
              'expected_episodes': stage['expected_episodes'], 'evidence_class': evidence_class,
              'audit': audit, 'episodes': episodes}
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
            or tokens[0]['truncated_rows'] or tokens[0]['dropped_prompt_rows']):
        raise ValueError('Sampled rollouts, rewards, groups or tokens differ between the adapter and the traces')
    for rollout, e in by_rollout.items():
        e.update(sampled_tokens=sampled[rollout]['original_generated_tokens'], fragment_rows=sampled[rollout]['fragments'],
                 mask_tokens=measured[rollout]['mask_tokens'], masked_context_tokens=measured[rollout]['context_tokens'],
                 fragment_prompt_tokens=measured[rollout]['prompt_tokens'])
    report['collection'] = {'real_rollouts': 32, 'groups': 8, 'generated_tokens': admitted[0]['generated_tokens'],
                            'fragment_rows': admitted[0]['fragment_rows'],
                            'adapter_constant_groups': admitted[0]['constant_groups'],
                            'limits': tokens[0]['limits'], 'truncated_rows': 0, 'dropped_prompt_rows': 0,
                            'stopped_before': 'advantages, backward pass, optimizer step and checkpoint save'}
    return {'calibration': report, 'optimizer_updates': {'actor': 0, 'critic': 0}}


def validate_stage(plan, job, stage, proof):
    """Reviewer-side validation of a durable stage completion against the plan."""
    declared = plan['calibration']['stages'][stage['name']]
    report = proof.get('calibration')
    if (proof.get('optimizer_updates') != {'actor': 0, 'critic': 0} or not isinstance(report, dict)
            or report.get('schema') != STAGE_SCHEMA or report['stage'] != stage['name']
            or report['algorithm'] != job['algorithm']
            or any(report[k] != declared[k] for k in ('mode', 'route', 'temperature', 'panel_id', 'packet_id',
                                                       'case_ids', 'expected_episodes'))
            or any(report['audit'][name] for name in FORBIDDEN_EVENTS) or report['audit']['weight_refresh_steps'] != [0]
            or report['evidence_class'] not in ('live-model', 'scripted-inference')):
        raise ValueError('Calibration stage evidence differs from the plan or shows optimization')
    check_table(declared, report['episodes'])
    if declared['mode'] == 'sampled':
        c = report.get('collection') or {}
        if (c.get('real_rollouts') != 32 or c.get('groups') != 8 or c.get('truncated_rows') or c.get('dropped_prompt_rows')
                or c.get('generated_tokens') != sum(e['sampled_tokens'] for e in report['episodes'])
                or any(e['sampled_tokens'] != e['mask_tokens'] or e['sampled_tokens'] <= 0 for e in report['episodes'])):
            raise ValueError('Sampled calibration token or group accounting differs')
    return report


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
            'semantics_not_assessable': n - len(assessable),
            'failure_classes': dict(Counter(e['failure_class'] for e in episodes)),
            'outcomes': dict(Counter(e['outcome'] for e in episodes)),
            'typed_action_errors': sum(e['policy_errors'] for e in episodes),
            'typed_action_error_codes': dict(Counter(code for e in episodes for code in e['policy_error_codes'])),
            'recovered_episodes': sum(e['recovered'] for e in episodes),
            'terminal_policy_failures': sum(e['outcome'] == 'terminal_policy_failure' for e in episodes),
            'episodes_with_successful_call': {name: sum(e['successful_calls_by_tool'].get(name, 0) > 0 for e in episodes)
                                              for name in task.TOOLS},
            'episodes_with_required_evidence_observed': count('required_evidence_observed'),
            'correctly_joined_observation_chains': sum(e['joined_observation_chains'] for e in episodes),
            'tool_rounds': dict(sorted(Counter(e['tool_rounds'] for e in episodes).items())),
            'model_calls': sum(e['llm_calls'] for e in episodes),
            'prompt_tokens_max': max((t for e in episodes for t in e['prompt_tokens'] if t is not None), default=None),
            'completion_tokens_reported': sum(t for e in episodes for t in e['completion_tokens'] if t is not None),
            'finish_reasons': dict(Counter(f for e in episodes for f in e['finish_reasons']))}


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


def decide(greedy, sampled, groups, complete, live):
    """Operational validity and scientific suitability are separate decisions."""
    c = CRITERIA
    if not complete:
        return {'operational_validity': 'invalid', 'scientific_suitability': 'not_assessed',
                'reason': 'incomplete or unadmitted evidence; no partial collection is interpreted'}
    informative = groups['mixed_semantic_evidence']
    families = groups['semantic_evidence_informative_families']
    share = groups['interface_zero_rewards'] / groups['zero_rewards'] if groups['zero_rewards'] else None
    passed = (informative >= c['min_semantic_evidence_informative_groups'] and len(families) >= c['min_case_families']
              and share is not None and share <= c['max_interface_share_of_zero_rewards'])
    if passed:
        verdict = 'screen_passed'
    elif share is not None and share > c['max_interface_share_of_zero_rewards']:
        verdict = 'interface_dominated'
    elif groups['constant_one'] > c['groups'] - c['min_semantic_evidence_informative_groups']:
        verdict = 'saturated'
    elif groups['constant_zero'] > c['groups'] - c['min_semantic_evidence_informative_groups']:
        verdict = 'too_difficult'
    else:
        verdict = 'insufficient_semantic_variation'
    return {'operational_validity': 'valid' if live else 'mechanics_only_not_live',
            'scientific_suitability': verdict if live else 'not_assessed',
            'screen_on_supplied_evidence': verdict,
            'semantic_evidence_informative_groups': informative, 'of_groups': groups['groups'],
            'informative_case_families': families, 'interface_share_of_zero_rewards': share,
            'greedy_successes_and_failures_both_observed': 0 < greedy['reward_sum'] < greedy['episodes'],
            'sampled_successes_and_failures_both_observed': 0 < sampled['reward_sum'] < sampled['episodes'],
            'next_step': ('none: scripted or simulated evidence supports no calibration decision' if not live
                          else 'revise the task, prompt or harness using development evidence under new identities; '
                          'do not repeat an unchanged collection' if verdict != 'screen_passed'
                          else 'eligible for a separately proposed controlled learning study; not evidence of learning'),
            'interpretation': 'a negative screen is a valid calibration result' if live
                              else 'scripted or simulated evidence: harness mechanics only, never a policy measurement'}


def report(plan, completions, timings=None):
    """Machine-readable calibration report. completions[algorithm][stage] are stage proofs.

    A route with any missing or invalid stage is reported as operationally invalid with
    its counts; nothing is zero-filled, dropped, padded or pooled across routes.
    """
    declared = plan['calibration']
    if modes.phase(plan) != modes.CALIBRATION or declared != declaration():
        raise ValueError('Not a current recovery calibration plan')
    out_routes, classes = {}, set()
    for job in plan['jobs']:
        stages, problems, greedy, sampled = {}, [], [], []
        for stage in job['stages']:
            proof = completions.get(job['algorithm'], {}).get(stage['name'])
            if proof is None:
                problems.append(f"{stage['name']}: no admitted completion")
                continue
            try:
                stage_report = validate_stage(plan, job, stage, proof)
            except (ValueError, KeyError, TypeError) as exc:
                problems.append(f"{stage['name']}: {type(exc).__name__}: {exc}")
                continue
            classes.add(stage_report['evidence_class'])
            stages[stage['name']] = stage_report
            (greedy if stage_report['mode'] == 'greedy' else sampled).extend(stage_report['episodes'])
        attempted = len(greedy) + len(sampled)
        per_question, groups = _groups(sampled)
        packets = {}
        for name, s in stages.items():
            if s['mode'] == 'sampled':
                ids = set(s['case_ids'])
                packets[name] = {'packet_id': s['packet_id'], **_groups([e for e in sampled if e['case_id'] in ids])[1],
                                 'collection': s['collection']}
        complete = not problems and attempted == declared['episodes_per_algorithm']
        live = classes == {'live-model'}
        g, s = _summary(greedy), _summary(sampled)
        out_routes[job['algorithm']] = {
            'episodes': {'expected': declared['episodes_per_algorithm'], 'attempted_and_admitted': attempted,
                         'complete': attempted, 'not_admitted': declared['episodes_per_algorithm'] - attempted,
                         'rewards_confirmed': attempted, 'rewards_missing': declared['episodes_per_algorithm'] - attempted,
                         'note': 'episodes of a stage that failed completion are not admitted and are never zero-filled'},
            'stage_problems': problems,
            'greedy': {**g, 'by_family': _by_family(greedy),
                       'per_question_reward': {e['case_id']: e['reward'] for e in greedy}},
            'sampled': {**s, 'by_family': _by_family(sampled), 'groups': groups, 'by_packet': packets,
                        'per_question': per_question,
                        'tokens': {'source': ('production adapter events' if live else
                                              'scripted or simulated events; not measurements'),
                                   'sampled_tokens': sum(e.get('sampled_tokens', 0) for e in sampled),
                                   'mask_tokens': sum(e.get('mask_tokens', 0) for e in sampled),
                                   'masked_context_tokens': sum(e.get('masked_context_tokens', 0) for e in sampled),
                                   'fragment_rows': sum(e.get('fragment_rows', 0) for e in sampled),
                                   'largest_fragment_prompt_tokens': max((t for e in sampled for t in e.get('fragment_prompt_tokens', [])), default=None),
                                   'conservation': 'mask tokens equal sampled tokens for every admitted rollout',
                                   'truncated_or_dropped_rows': 0}},
            'optimization_events': {name: sum(st['audit'][name] for st in stages.values()) for name in FORBIDDEN_EVENTS},
            'weight_refresh_steps_by_stage': {name: st['audit']['weight_refresh_steps'] for name, st in stages.items()},
            'decision': decide(g, s, groups, complete, live)}
    evidence = ('live-model' if classes == {'live-model'} else 'none' if not classes else 'scripted-inference')
    return {'schema': REPORT_SCHEMA, 'task': task.TASK, 'episode_policy': task.EPISODE_POLICY['version'],
            'evidence_class': evidence, 'live_policy_measurement': evidence == 'live-model',
            'panel_id': declared['panel_id'], 'packets': declared['packets'], 'split': 'development',
            'final_evaluation': declared['final_evaluation'], 'criteria': CRITERIA,
            'expected_total_episodes': declared['total_episodes'], 'optimizer_updates_declared': 0,
            'routes': out_routes, 'stage_timings_seconds': timings or 'not supplied',
            'unverified_assumptions': [
                'Sampling temperatures are set by the gateway routes named in each stage; they are checked in the '
                'worker launch command, not observed per request.',
                'Token counts for greedy episodes are server-reported usage; sampled-token counts come from the adapter.',
                'Scripted or simulated evidence says nothing about how the model behaves on this task.'],
            'claims': {'local_operation': 'established only by executed offline tests',
                       'harness_effects': 'not measured by this report',
                       'useful_signal': 'assessed per route by the fixed screen, only on live evidence',
                       'learning_improvement': 'not assessed; calibration performs zero optimization'}}


def summary(result):
    """Readable summary of a report; states every denominator."""
    lines = [f"# Recovery readiness development calibration ({result['evidence_class']})", '',
             f"Panel `{result['panel_id']}`, split `{result['split']}`, {result['expected_total_episodes']} expected episodes, "
             'zero optimizer updates declared.', '']
    if not result['live_policy_measurement']:
        lines += ['**Not a live policy measurement.** Inference was scripted or simulated; the numbers below show '
                  'that the reporting path works, not how any model performs.', '']
    for algorithm, r in result['routes'].items():
        g, s, d, groups = r['greedy'], r['sampled'], r['decision'], r['sampled']['groups']
        e = r['episodes']
        lines += [f'## {algorithm}', '',
                  f"- Episodes: {e['attempted_and_admitted']}/{e['expected']} admitted with confirmed rewards; "
                  f"{e['not_admitted']} not admitted.",
                  f"- Greedy: {g['reward_sum']:g}/{g['episodes']} rewarded (95% Wilson {g['reward_wilson95']}); "
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
                  f"- Optimization events: {r['optimization_events']}; refresh steps {r['weight_refresh_steps_by_stage']}.",
                  f"- Operational validity: **{d['operational_validity']}**. Scientific suitability: **{d['scientific_suitability']}**."]
        if r['stage_problems']:
            lines += ['- Stage problems: ' + '; '.join(r['stage_problems'])]
        if 'screen_on_supplied_evidence' in d:
            lines += [f"- Screen applied to the supplied evidence: {d['screen_on_supplied_evidence']} "
                      f"({d['semantic_evidence_informative_groups']}/{d['of_groups']} informative groups, families "
                      f"{d['informative_case_families']}, interface share of zeros {d['interface_share_of_zero_rewards']})."]
        lines += ['']
    c = result['criteria']
    lines += ['## Screen', '',
              f"At least {c['min_semantic_evidence_informative_groups']} of {c['groups']} sampled groups with "
              f"semantic/evidence variation, across at least {c['min_case_families']} case families, with interface "
              f"failures at most {c['max_interface_share_of_zero_rewards']:.0%} of zero rewards, per algorithm route. "
              'This is an engineering heuristic, not a research threshold or proof of learning.', '']
    return '\n'.join(lines)
