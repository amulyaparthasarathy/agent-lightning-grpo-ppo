"""Read existing calculator traces; qualify integration independently of task reward."""
from datetime import datetime
import hashlib
import json


IDENTITY = ('trace_id', 'experiment_id', 'wandb_run_id', 'case_id', 'phase', 'policy_version',
            'eval_split', 'model_profile', 'rollout_id', 'attempt_id', 'route_mode')


def chains(records, request):
    """Join AutoGen 0.7.5 FunctionCall/ToolResult/FunctionExecutionResult records."""
    stage, job = request['stage'], request['job']
    episodes = [r for r in records if r['kind'] == 'episode']
    if len(episodes) != 1:
        raise ValueError('Expected one terminal episode')
    episode = episodes[0]
    if (episode['experiment_id'] != job['job_id'] or episode['wandb_run_id'] != stage['wandb']['id']
            or episode['phase'] != stage['name'] or episode['policy_version'] != 'matched-rollout-mean-v2'
            or episode['model_profile'] != 'llama3b'
            or episode['route_mode'] != ('train' if stage['name'] == 'training' else 'val')):
        raise ValueError('Trace belongs to a different stage or protocol')
    if any(not episode.get(k) for k in IDENTITY):
        raise ValueError('Missing trace/rollout/attempt identity')
    spans = set()
    for record in records:
        if (record['schema_version'] != 1 or record['synthetic_fixture'] is not False
                or any(record[k] != episode[k] for k in IDENTITY)):
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
    if not ok(episode) or output.get('reward_posted') is not True or type(output.get('reward')) not in (int, float) or output['reward'] not in (0, 1):
        return []
    def before(left, right):
        return datetime.fromisoformat(left['ended_at']) <= datetime.fromisoformat(right['started_at'])
    llms = [r for r in records if r['kind'] == 'llm' and ok(r)]
    tools = [r for r in records if r['kind'] == 'tool']
    found = []
    for model in llms:
        content = model['outputs']['content']
        if not isinstance(content, list):
            continue
        for call in content:
            if call.get('name') != 'calculate' or not call.get('id'):
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
            if (not ok(tool) or tool['name'] != 'calculate' or result.get('name') != 'calculate'
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
                if (len(observations) == 1 and observations[0].get('name') == 'calculate'
                        and observations[0].get('is_error') is False and observations[0].get('content') == observation):
                    found.append({'trace_id': episode['trace_id'], 'case_id': episode['case_id'],
                        'rollout_id': episode['rollout_id'], 'attempt_id': episode['attempt_id'],
                        'model_span': model['span_id'], 'tool_span': tool['span_id'], 'tool_call_id': call['id'],
                        'continuation_span': continuation['span_id'], 'episode_span': episode['span_id'],
                        'observation_sha256': hashlib.sha256(observation.encode()).hexdigest(),
                        'terminal_reward': output['reward']})
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
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            entry.update(chains=[], invalid_trace=type(exc).__name__)
        valid += bool(entry['chains'])
        coverage.append(entry)
    return {'schema': 'reader-integration-v1', 'algorithm': request['job']['algorithm'],
        'stage': request['stage']['name'], 'qualified': valid > 0,
        'episodes_with_valid_chain': valid, 'trace_files': len(coverage), 'coverage': coverage,
        'gaps': [] if valid else ['No valid model → calculator → model observation → delivered terminal reward chain observed'],
        'task_correctness_required': False}
