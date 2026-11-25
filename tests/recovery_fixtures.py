"""TEST ONLY: synthetic reviewed-episode rows and simulated adapter audit events.

Pure Python. These build the same structures the production harness and adapter
produce (the schemas are compared with real output in test_recovery_harness and
test_matched_recovery_calibration). They are simulated evidence: useful for
exercising completion, aggregation and reporting, never a policy measurement.
"""
import copy
from pathlib import Path
from experiment import reader_assets as assets, reader_contract as c, recovery_calibration as calibration
from experiment import recovery_task as task

ROOT = Path(__file__).resolve().parents[1]
BEHAVIORS = ('correct', 'recovered', 'semantic', 'evidence', 'format', 'action')


def calibration_plan(source, fixture=True, assets_manifest=None):
    profile = assets.read(ROOT/'examples/reader-profile.recovery-calibration.synthetic.json')
    profile['fixture'] = fixture
    return c.make_plan(profile, source, assets_manifest)


def assets_manifest():
    """SIMULATED asset manifest with the real model lock and task manifest shape.

    It lets a test plan be `training_admissible`, as a plan must be before it could
    have produced live evidence. No model or dataset bytes exist behind it.
    """
    import hashlib
    data = {'schema': 'recovery-data-v1', 'task': task.TASK, 'environment': task.environment(),
            'environment_sha256': task.digest(task.environment()),
            'world_file_sha256': hashlib.sha256(task._world_bytes()).hexdigest(),
            'files': {name: {'rows': len(rows), 'split': task.SPLIT_OF_FILE[name], 'case_ids': [r['case_id'] for r in rows],
                             'sha256': task.digest([name, rows])} for name, rows in task.cases().items()}}
    return {'schema': 'reader-assets-v1', 'fixture': False,
            'model': {'id': c.MODEL, 'revision': c.BASE_REVISION, 'parser': 'llama3_json', 'files': assets.model_lock(),
                      'manifest': {'simulated': True}, 'manifest_sha256': 'a'*64},
            'data': data, 'historical_match': assets.synthetic_claim(task.TASK)}


def executable_plan(source):
    """A non-fixture, asset-bound plan from a clean source identity (all simulated)."""
    return calibration_plan(source, fixture=False, assets_manifest=assets_manifest())


def row(case_id, behavior, route, rollout, evidence_class='live-model'):
    """One reviewed episode in the stage-table schema for a named scripted behavior."""
    if behavior not in BEHAVIORS:
        raise ValueError('unknown fixture behavior')
    family, variant = task.family(case_id)
    good = behavior in ('correct', 'recovered')
    assessable = behavior in ('correct', 'recovered', 'semantic', 'evidence')
    terminal = behavior == 'action'
    flags = {'json_syntax': None if terminal else behavior != 'format', 'format_compliant': None if terminal else behavior != 'format',
             'target_correct': True if assessable else None, 'time_correct': True if assessable else None,
             'decision_correct': (behavior != 'semantic') if assessable else None,
             'evidence_correct': (good if behavior != 'semantic' else True) if assessable else None,
             'asset_record_correct': True if assessable else None, 'policy_record_correct': True if assessable else None,
             'inventory_correct': (behavior != 'evidence') if assessable else None,
             'supporting_runs_correct': True if assessable else None,
             'required_evidence_observed': not terminal, 'observed': {k: not terminal for k in ('asset', 'policy', 'inventory')}}
    errors = {'recovered': ['invalid_arguments'], 'action': ['invalid_arguments', 'unknown_tool']}.get(behavior, [])
    calls = 0 if terminal else 3
    generations = 2 if terminal else calls + len(errors) + 1
    sampled = ([{'id': f'{rollout}-e{i}', 'name': 'get asset' if code == 'unknown_tool' else 'get_asset', 'arguments_sha256': 'e'*64,
                 'policy_error': code, 'successful': False} for i, code in enumerate(errors)]
               + [{'id': f'{rollout}-c{i}', 'name': name, 'arguments_sha256': 'c'*64, 'policy_error': None, 'successful': True}
                  for i, name in enumerate(task.TOOLS if calls else ())])
    return {'case_id': case_id, 'family': family, 'variant': variant, 'rollout_id': rollout, 'attempt_id': 'at-'+rollout,
            'trace_id': 'tr-'+rollout, 'route': route, 'evidence_class': evidence_class,
            'answer_content': None if terminal else 'text', 'sampled_calls': sampled,
            'tool_names_rewritten_by_framework': sum(c['name'] == 'get asset' for c in sampled),
            'served_finish_reasons': ['tool_calls'] * (calls + len(errors)) + ([] if terminal else ['stop']),
            'served_stop_reasons': [None] * generations,
            'outcome': 'terminal_policy_failure' if terminal else 'complete_recovered' if behavior == 'recovered' else 'complete_final',
            'reward': 1.0 if good else 0.0, 'reward_posted': True,
            'failure_class': 'correct' if good else behavior, **flags,
            'policy_errors': len(errors), 'policy_error_codes': errors, 'recovered': behavior == 'recovered',
            'tool_rounds': calls + len(errors), 'llm_calls': 2 if terminal else calls + len(errors) + 1,
            'successful_calls_by_tool': {} if terminal else {n: 1 for n in task.TOOLS},
            # Simulated server-reported usage, as a live run reports it.
            'joined_observation_chains': calls, 'prompt_tokens': [1000 + 70 * i for i in range(generations)],
            'completion_tokens': [20] * generations,
            'framework_finish_reasons': ['function_calls'] * (calls + len(errors)) + ([] if terminal else ['stop'])}


def rows(stage, behavior, tag='', evidence_class='live-model'):
    """Rows for one declared stage. behavior(case_id, sample_index) names a fixture behavior."""
    out = []
    for case_id in stage['case_ids']:
        for sample in range(stage['rollouts_per_question']):
            out.append(row(case_id, behavior(case_id, sample), stage['route'], f"{tag}{case_id.rsplit('/', 1)[1]}-{sample}",
                           evidence_class))
    return out


def events(stage, episodes, wandb_id='simulated', pid=0):
    """SIMULATED audit events with the production event schema; no optimization events.

    Sampled-token counts follow the episodes' reported completion tokens when they
    have them, as the real adapter's counts did in Phase 3.
    """
    common = [{'event': 'matched_registered', 'pid': pid, 'simulated': True},
              {'event': 'weight_refresh', 'step': 0, 'pid': pid, 'simulated': True},
              {'event': 'wandb_finished', 'id': wandb_id, 'pid': pid, 'simulated': True}]
    if stage['mode'] == 'greedy':
        return common + [{'event': 'trainer_returned', 'global_step': 0, 'validation_only': True, 'pid': pid, 'simulated': True}]
    groups = {case_id: f'group-{i}' for i, case_id in enumerate(stage['case_ids'])}
    def sampled_tokens(e):
        return sum(e['completion_tokens']) if None not in e['completion_tokens'] else 30 * e['llm_calls']
    trajectories = [{'rollout': e['rollout_id'], 'group': groups[e['case_id']], 'case_id': e['case_id'], 'reward': e['reward'],
                     'fragments': e['llm_calls'], 'original_generated_tokens': sampled_tokens(e)} for e in episodes]
    rewards = {}
    for e in episodes:
        rewards.setdefault(e['case_id'], set()).add(e['reward'])
    return common + [
        {'event': 'collection_manifest', 'iteration': 1, 'trajectories': trajectories, 'adapter_metrics': {'training/n_truncated_sample': 0}},
        {'event': 'development_collection_admitted', 'iteration': 1, 'real_rollouts': 32, 'groups': 8,
         'generated_tokens': sum(t['original_generated_tokens'] for t in trajectories),
         'fragment_rows': sum(t['fragments'] for t in trajectories),
         'constant_groups': sum(len(v) == 1 for v in rewards.values())},
        {'event': 'calibration_collection_tokens', 'iteration': 1,
         'limits': {'max_prompt_length': task.CONTEXT_LIMITS['trajectory_max_prompt_length'],
                    'max_response_length': task.CONTEXT_LIMITS['trajectory_max_response_length']},
         'truncated_rows': 0, 'dropped_prompt_rows': 0,
         'rollouts': [{'rollout': t['rollout'], 'mask_tokens': t['original_generated_tokens'], 'context_tokens': 0,
                       'prompt_tokens': [1000 + 80 * i for i in range(t['fragments'])]} for t in trajectories]},
        {'event': 'trainer_returned', 'global_step': 1, 'validation_only': False, 'collection_only': True, 'optimizer_updates': 0,
         'pid': pid, 'simulated': True}]


def proofs(plan, behavior, evidence_class='live-model', skip=()):
    """Stage completions for every algorithm and stage. behavior(algorithm, stage, case_id, sample)."""
    out = {}
    for job in plan['jobs']:
        for stage in job['stages']:
            if (job['algorithm'], stage['name']) in skip:
                continue
            request = c.stage_request(plan, job, stage)
            declared = plan['calibration']['stages'][stage['name']]
            episodes = rows(declared, lambda case_id, sample: behavior(job['algorithm'], stage['name'], case_id, sample),
                            tag=job['algorithm'][:4]+'-', evidence_class=evidence_class)
            out.setdefault(job['algorithm'], {})[stage['name']] = calibration.assemble(
                request, events(declared, episodes), episodes, evidence_class)
    return out


def durable(plan, completions):
    """A completed supervisor index and its durable files for reader_evidence.validate_collection."""
    files, index = {}, {'plan': plan, 'status': 'completed', 'jobs': {}}
    for job in plan['jobs']:
        receipts = []
        for stage in job['stages']:
            call = 'fc-' + stage['wandb']['id']
            proof = {'execution_schema': c.SCHEMA, 'plan_sha256': c.digest(plan), 'stage': stage['name'],
                     'worker': {'call_id': call, 'hostname': 'modal', 'container_id': 'ta-' + c.digest(call)[:16]},
                     'source_sha256': c.digest(plan['source']), 'configuration_sha256': plan['configuration_sha256'],
                     'assets_sha256': c.digest(plan['assets']), 'task_sha256': c.digest(plan['task_identity']),
                     'wandb_id': stage['wandb']['id'], **copy.deepcopy(completions[job['algorithm']][stage['name']]),
                     'model_tool_integration': {'schema': 'reader-integration-v2', 'algorithm': job['algorithm'],
                                                'stage': stage['name'], 'qualified': True, 'episodes_with_valid_chain': 1,
                                                'trace_files': 1, 'coverage': [{'chains': [{'simulated_test_only': True}]}]}}
            receipt = {'call_id': call, 'container_id': proof['worker']['container_id'], 'status': 'completed',
                       'completion': proof, 'output': stage['output'], 'started_at': 100.0, 'finished_at': 460.0}
            files[stage['output'] + '/receipt.json'] = receipt
            files[stage['output'] + '/completion.json'] = proof
            files[stage['output'] + '/observability.json'] = {'wandb': stage['wandb']}
            receipts.append({'stage': stage['name'], 'status': 'completed', 'call_id': call, 'result': receipt, 'output': stage['output']})
        index['jobs'][job['job_id']] = {'status': 'completed', 'stages': receipts}
    return index, files


def informative(algorithm, stage, case_id, sample):
    """A mixed scripted pattern with semantic and evidence failures across several families."""
    family = task.family(case_id)[0]
    if stage == 'greedy-development':
        return 'semantic' if family in ('stale_success', 'recent_integrity_failed') else 'correct'
    if family == 'stale_success':
        return ('correct', 'semantic', 'semantic', 'correct')[sample]
    if family == 'failed_newer_usable_older':
        return ('correct', 'evidence', 'correct', 'recovered')[sample]
    if family == 'recent_integrity_failed':
        return ('semantic', 'correct', 'format', 'correct')[sample]
    if family == 'recent_incomplete':
        return 'semantic'
    return 'correct'


def reviews(plan, behavior, evidence_class='live-model', skip=()):
    """SIMULATED admitted stage reviews, for arithmetic tests of the report and screen only.

    They are built from fixture rows and simulated events and are marked admitted
    without the durable, trace and audit checks that real admission requires. A
    skipped stage is reviewed by the real review_stage with no evidence at all.
    """
    out = {}
    for job in plan['jobs']:
        route = out[job['algorithm']] = []
        for stage in job['stages']:
            request = c.stage_request(plan, job, stage)
            declared = plan['calibration']['stages'][stage['name']]
            if (job['algorithm'], stage['name']) in skip:
                route.append(calibration.review_stage(request))
                continue
            episodes = rows(declared, lambda case_id, sample: behavior(job['algorithm'], stage['name'], case_id, sample),
                            tag=job['algorithm'][:4]+'-', evidence_class=evidence_class)
            simulated = events(declared, episodes)
            done = calibration.assemble(request, simulated, episodes, evidence_class)['calibration']
            n = len(episodes)
            facts = {'trace_files': n, 'unreadable': 0, 'rejected_untrusted': 0,
                     'rejection_reasons': {}, 'incomplete_exceptions': {}, 'evidence_classes': {evidence_class: n},
                     'model_calls': sum(e['llm_calls'] for e in episodes),
                     'model_calls_with_reported_usage': sum(e['llm_calls'] for e in episodes)}
            route.append({'stage': stage['name'], 'mode': declared['mode'], 'packet_id': declared['packet_id'],
                          'packet_status': 'admitted', 'evidence_class': evidence_class,
                          'execution': {'supervisor_status': 'completed', 'dispatched': True, 'receipt_status': 'completed', 'receipt_error': None},
                          'episodes': {'expected': n, 'admitted': n, 'trace_files': n, 'observed_attempts': n, 'completed': n,
                                       'rewards_confirmed': n, 'rewards_missing': 0, 'rewards_confirmed_not_admitted': 0,
                                       'rejected_untrusted': 0, 'reviewable': n, 'unexpected_extra_attempts': 0,
                                       'unattempted': 0, 'unknown': 0},
                          'checks': {'SIMULATED': 'admission bypassed by a test fixture'}, 'problems': [], 'trace_facts': facts,
                          'token_evidence': calibration.token_coverage(declared, simulated, facts),
                          'audit': calibration.audit_counts(simulated), 'rows': done['episodes'], 'diagnostic_rows': [],
                          'summary_report': done})
    return out
