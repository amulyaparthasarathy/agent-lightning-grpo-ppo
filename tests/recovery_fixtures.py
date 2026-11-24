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


def row(case_id, behavior, route, rollout):
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
    return {'case_id': case_id, 'family': family, 'variant': variant, 'rollout_id': rollout, 'attempt_id': 'at-'+rollout,
            'trace_id': 'tr-'+rollout, 'route': route,
            'outcome': 'terminal_policy_failure' if terminal else 'complete_recovered' if behavior == 'recovered' else 'complete_final',
            'reward': 1.0 if good else 0.0, 'reward_posted': True,
            'failure_class': 'correct' if good else behavior, **flags,
            'policy_errors': len(errors), 'policy_error_codes': errors, 'recovered': behavior == 'recovered',
            'tool_rounds': calls + len(errors), 'llm_calls': 2 if terminal else calls + len(errors) + 1,
            'successful_calls_by_tool': {} if terminal else {n: 1 for n in task.TOOLS},
            'joined_observation_chains': calls, 'prompt_tokens': [None] * (2 if terminal else calls + len(errors) + 1),
            'completion_tokens': [None] * (2 if terminal else calls + len(errors) + 1),
            'finish_reasons': ['function_calls'] * (calls + len(errors)) + ([] if terminal else ['stop'])}


def rows(stage, behavior, tag=''):
    """Rows for one declared stage. behavior(case_id, sample_index) names a fixture behavior."""
    out = []
    for case_id in stage['case_ids']:
        for sample in range(stage['rollouts_per_question']):
            out.append(row(case_id, behavior(case_id, sample), stage['route'], f"{tag}{case_id.rsplit('/', 1)[1]}-{sample}"))
    return out


def events(stage, episodes):
    """Simulated audit events with the production event schema; no optimization events."""
    common = [{'event': 'matched_registered'}, {'event': 'weight_refresh', 'step': 0}, {'event': 'wandb_finished', 'id': 'simulated'}]
    if stage['mode'] == 'greedy':
        return common + [{'event': 'trainer_returned', 'global_step': 0, 'validation_only': True}]
    groups = {case_id: f'group-{i}' for i, case_id in enumerate(stage['case_ids'])}
    trajectories = [{'rollout': e['rollout_id'], 'group': groups[e['case_id']], 'case_id': e['case_id'], 'reward': e['reward'],
                     'fragments': e['llm_calls'], 'original_generated_tokens': 30 * e['llm_calls']} for e in episodes]
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
        {'event': 'trainer_returned', 'global_step': 1, 'validation_only': False, 'collection_only': True, 'optimizer_updates': 0}]


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
                            tag=job['algorithm'][:4]+'-')
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
        index['jobs'][job['job_id']] = {'stages': receipts}
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
