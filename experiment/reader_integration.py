"""Read task traces; qualify integration independently of terminal correctness."""
from datetime import datetime
import hashlib
import json


# Prospective episode policies with explicit outcome rules; any other trace keeps the
# historical treatment. Limits are those declared by each task's EPISODE_POLICY.
EPISODE_POLICIES = {'asset-episode-v2': {'policy_error_limit': 2}, 'recovery-episode-v1': {'policy_error_limit': 2}}
TOOLS = {'calculator-v1': ('calculate',), 'asset-status-v1': ('lookup_asset',),
         'recovery-readiness-v1': ('get_asset', 'get_policy', 'list_backup_runs')}


def stage_route(plan, stage):
    try:
        from .reader_modes import route
    except ImportError:
        from reader_modes import route
    return route(plan, stage)


IDENTITY = ('trace_id', 'experiment_id', 'wandb_run_id', 'case_id', 'phase', 'policy_version',
            'eval_split', 'model_profile', 'rollout_id', 'attempt_id', 'route_mode')


def episode_outcome(records):
    """Execution outcome is separate from a successful live tool chain."""
    episodes = [r for r in records if r.get('kind') == 'episode']
    if len(episodes) != 1:
        return 'incomplete'
    e = episodes[0]; out = e.get('outputs') or {}
    if e.get('execution_status') != 'ok' or out.get('reward_posted') is not True:
        return 'incomplete'
    if e.get('episode_policy') not in EPISODE_POLICIES:
        return 'historical_complete'
    models = [r for r in records if r['kind'] == 'llm']
    actions = [r for r in records if r['kind'] == 'action_outcome']
    terminations = [r for r in records if r['kind'] == 'termination']
    if (not models or any(r['execution_status'] != 'ok' or not r.get('outputs')
                         or r['outputs'].get('finish_reason') not in ('stop', 'function_calls') for r in models)
            or type(out.get('reward')) not in (int, float) or out['reward'] not in (0, 1)):
        return 'incomplete'
    calls = [call for r in models if isinstance(r['outputs']['content'], list) for call in r['outputs']['content']]
    observed = [r['inputs']['call'] for r in actions]
    if calls != observed or len({c['id'] for c in calls}) != len(calls):
        return 'incomplete'
    errors = sum(bool(r['outputs']['policy_error']) for r in actions)
    if errors != out.get('policy_errors'):
        return 'incomplete'
    outcome = out.get('outcome')
    if outcome == 'terminal_policy_failure':
        if (errors < EPISODE_POLICIES[e['episode_policy']]['policy_error_limit'] or out['reward'] != 0 or out.get('raw_answer') is not None
                or len(terminations) != 1 or terminations[0]['name'] != 'policy_error_limit'):
            return 'incomplete'
        return outcome
    final = models[-1]['outputs']
    if (terminations or final['finish_reason'] != 'stop' or final['content'] != out.get('raw_answer')
            or not isinstance(final['content'], str) or not final['content'].strip()):
        return 'incomplete'
    recovered = False; seen_error = False
    for action in actions:
        seen_error |= bool(action['outputs']['policy_error'])
        recovered |= seen_error and action['outputs']['successful_tool_call']
    expected = 'complete_recovered' if recovered else 'complete_final'
    return expected if outcome == expected else 'incomplete'


def chains(records, request, *, evidence_class="live-model"):
    """Join AutoGen 0.7.5 FunctionCall/ToolResult/FunctionExecutionResult records."""
    stage, job = request['stage'], request['job']
    plan = request.get('plan', {})
    modern = plan.get('schema') == 'reader-task-v2'
    task = plan.get('task_identity', {}).get('name', 'calculator-v1')
    if task not in TOOLS:
        raise ValueError('Unknown task; no integration rule')
    tool_names = TOOLS[task]
    identities = IDENTITY + (('task_id', 'task_sha256', 'environment_sha256', 'source_sha256', 'configuration_sha256') if modern else ())
    episodes = [r for r in records if r['kind'] == 'episode']
    if len(episodes) != 1:
        raise ValueError('Expected one terminal episode')
    episode = episodes[0]
    if (episode['experiment_id'] != job['job_id'] or episode['wandb_run_id'] != stage['wandb']['id']
            or episode['phase'] != stage['name'] or episode['policy_version'] != 'matched-rollout-mean-v2'
            or episode['model_profile'] != 'llama3b'
            or episode['route_mode'] != stage_route(plan, stage)):
        raise ValueError('Trace belongs to a different stage or protocol')
    if modern:
        try:
            from .reader_contract import digest
        except ImportError:
            from reader_contract import digest
        expected = {'task_id': task, 'task_sha256': digest(plan['task_identity']),
                    'environment_sha256': plan['task_identity'].get('environment_sha256') or digest(plan['data_manifest']),
                    'source_sha256': digest(plan['source']), 'configuration_sha256': plan['configuration_sha256']}
        if any(episode.get(k) != v for k, v in expected.items()):
            raise ValueError('Cross-task/environment/source/configuration trace')
    if any(not episode.get(k) for k in identities):
        raise ValueError('Missing trace/rollout/attempt identity')
    if evidence_class not in ('live-model', 'scripted-inference', 'simulated-trace'):
        raise ValueError('Unknown evidence class')
    spans = set()
    for record in records:
        if (record['schema_version'] != 1 or record['synthetic_fixture'] is not (evidence_class != 'live-model')
                or record.get('evidence_class', 'live-model') != evidence_class
                or any(record[k] != episode[k] for k in identities)):
            raise ValueError('Synthetic or inconsistent trace identity')
        if record['kind'] == 'trace_status':
            continue
        if record['span_id'] in spans:
            raise ValueError('Duplicate span')
        spans.add(record['span_id'])
        if record is not episode and record['parent_span_id'] != episode['span_id']:
            raise ValueError('Different parent episode')
        if not (datetime.fromisoformat(episode['started_at']) <= datetime.fromisoformat(record['started_at'])
                <= datetime.fromisoformat(record['ended_at']) <= datetime.fromisoformat(episode['ended_at'])):
            raise ValueError('Invalid span timing')
    def ok(record):
        return record['execution_status'] == 'ok' and record['exception'] is None
    output = episode['outputs'] or {}
    if episode.get('episode_policy') in EPISODE_POLICIES and episode_outcome(records) in ('incomplete', 'terminal_policy_failure'):
        return []
    if not ok(episode) or output.get('reward_posted') is not True or type(output.get('reward')) not in (int, float) or output['reward'] not in (0, 1):
        return []
    def before(left, right):
        return datetime.fromisoformat(left['ended_at']) <= datetime.fromisoformat(right['started_at'])
    if modern and episode.get('evidence_class', 'live-model') != evidence_class:
        raise ValueError('Scripted inference is not live evidence')
    llms = [r for r in records if r['kind'] == 'llm' and ok(r)]
    tools = [r for r in records if r['kind'] == 'tool']
    if modern:
        terminal = sorted(llms, key=lambda r: r['ended_at'])[-1:]
        if (not terminal or terminal[0]['outputs'].get('finish_reason') != 'stop'
                or terminal[0]['outputs'].get('content') != output.get('raw_answer')):
            return []
    found = []
    for model in llms:
        content = model['outputs']['content']
        if not isinstance(content, list):
            continue
        for call in content:
            if call.get('name') not in tool_names or not call.get('id'):
                continue
            same_calls = [call_record for llm in llms if isinstance(llm['outputs']['content'], list)
                          for call_record in llm['outputs']['content'] if call_record.get('id') == call['id']]
            if len(same_calls) != 1:
                continue
            matches = [t for t in tools if t['inputs']['tool_call_id'] == call['id']]
            if len(matches) != 1:
                continue
            tool = matches[0]
            result = tool['outputs'] or {}
            if (not ok(tool) or tool['name'] != call['name'] or result.get('name') != call['name']
                    or result.get('is_error') is not False or not before(model, tool)
                    or json.loads(call['arguments']) != tool['inputs']['arguments']):
                continue
            parts = result.get('result', [])
            if not parts or any(p['type'] != 'TextResultContent' for p in parts):
                continue
            observation = '\n'.join(p['content'] for p in parts)  # ToolResult.to_text(), no images for calculator
            for continuation in llms:
                if not before(tool, continuation) or datetime.fromisoformat(continuation['ended_at']) > datetime.fromisoformat(episode['ended_at']):
                    continue
                observations = [o for message in continuation['inputs']['messages']
                    if message.get('type') == 'FunctionExecutionResultMessage'
                    for o in message['content'] if o.get('call_id') == call['id']]
                if (len(observations) == 1 and observations[0].get('name') == call['name']
                        and observations[0].get('is_error') is False and observations[0].get('content') == observation):
                    found.append({'trace_id': episode['trace_id'], 'case_id': episode['case_id'],
                        'rollout_id': episode['rollout_id'], 'attempt_id': episode['attempt_id'],
                        'model_span': model['span_id'], 'tool_span': tool['span_id'], 'tool_call_id': call['id'],
                        'continuation_span': continuation['span_id'], 'episode_span': episode['span_id'],
                        'observation_sha256': hashlib.sha256(observation.encode()).hexdigest(),
                        'terminal_reward': output['reward'],
                        **({'task_id': task, 'environment_sha256': expected['environment_sha256'],
                            'evidence_class': evidence_class} if modern else {})})
                    break
    return found


def report(run, request):
    coverage, valid = [], 0
    for path in sorted((run/'traces').glob('trajectory-*.jsonl')):
        raw = path.read_bytes()
        entry = {'file': path.name, 'sha256': hashlib.sha256(raw).hexdigest()}
        try:
            records = [json.loads(line) for line in raw.splitlines()]
            if not records or any(r['trace_id'] != path.stem.removeprefix('trajectory-') for r in records):
                raise ValueError('Trace filename identity differs')
            entry['chains'] = chains(records, request)
            entry['episode_outcome'] = episode_outcome(records)
            episode = next(r for r in records if r['kind'] == 'episode')
            entry['diagnostics'] = (episode.get('outputs') or {}).get('diagnostics')
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            entry.update(chains=[], invalid_trace=type(exc).__name__)
        valid += bool(entry['chains'])
        coverage.append(entry)
    return {'schema': 'reader-integration-v2' if request.get('plan', {}).get('schema') == 'reader-task-v2' else 'reader-integration-v1', 'algorithm': request['job']['algorithm'],
        'stage': request['stage']['name'], 'qualified': valid > 0,
        'episodes_with_valid_chain': valid, 'trace_files': len(coverage), 'coverage': coverage,
        'gaps': [] if valid else ['No valid model → task tool → model observation → delivered terminal reward chain observed'],
        'task_correctness_required': False}
