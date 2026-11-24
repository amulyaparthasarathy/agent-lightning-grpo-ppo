"""Effective served prompt for the reader harnesses: render, hash and measure.

Source prompt strings are not what the model receives. The served chat template
rewrites the first user turn while tools are advertised, wraps tool results, adds a
fixed date header and drops the tool preamble at the tool-less reflection call.
This module renders that template for exact OpenAI wire requests (or recorded
trace inputs) and counts tokens with the pinned tokenizer when one is supplied.

Rendering follows the served stack: the prepared template (fixed date string),
Jinja2 with trim_blocks/lstrip_blocks and a JSON `tojson` filter as in
Transformers, tool-call arguments passed to the template as objects as in vLLM,
and tools omitted when the request carries none. Jinja2 and tokenizers are needed
only when these functions are called; nothing here runs inside an episode.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

try:
    from . import prepare_model, recovery_task as task
except ImportError:
    import prepare_model
    import recovery_task as task

BOS = '<|begin_of_text|>'
LIMITS = {k: v for k, v in task.CONTEXT_LIMITS.items() if k != 'version'}
# Declared allowance for the sampled final answer when bounding a fully merged
# trajectory. It is an accounting assumption, not a decoding constraint: a longer
# generation is never cut or repaired, and the production adapter rejects any
# collection containing a truncated trajectory.
ANSWER_TOKENS = 192


@lru_cache(maxsize=1)
def compiled():
    return environment().from_string(prepare_model.effective_template())


def environment():
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise ValueError(message)

    def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters['tojson'] = tojson
    env.globals['raise_exception'] = raise_exception
    return env


def render(messages, tools=None):
    """Prompt text for one wire request. `tools` is None when the request has none."""
    prepared = []
    for message in messages:
        message = {k: v for k, v in message.items() if k != 'name'}
        if message.get('tool_calls'):
            # The server parses each arguments string before templating.
            message['tool_calls'] = [{**c, 'function': {**c['function'], 'arguments': json.loads(c['function']['arguments'])}}
                                     for c in message['tool_calls']]
        prepared.append(message)
    return compiled().render(messages=prepared, tools=tools or None, bos_token=BOS, add_generation_prompt=True)


def from_trace(inputs):
    """Wire messages/tools for a recorded `llm` trace input (AutoGen message dumps)."""
    messages = []
    for m in inputs['messages']:
        kind = m['type']
        if kind == 'SystemMessage':
            messages.append({'role': 'system', 'content': m['content']})
        elif kind == 'UserMessage':
            messages.append({'role': 'user', 'content': m['content']})
        elif kind == 'AssistantMessage' and isinstance(m['content'], list):
            messages.append({'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': c['id'], 'type': 'function', 'function': {'name': c['name'], 'arguments': c['arguments']}}
                for c in m['content']]})
        elif kind == 'AssistantMessage':
            messages.append({'role': 'assistant', 'content': m['content']})
        elif kind == 'FunctionExecutionResultMessage':
            messages += [{'role': 'tool', 'content': r['content'], 'tool_call_id': r['call_id']} for r in m['content']]
        else:
            raise ValueError('Unsupported recorded message type')
    tools = [{'type': 'function', 'function': {'name': t['name'], 'description': t.get('description', ''),
                                               'parameters': t['parameters'], 'strict': False}}
             for t in inputs.get('tools') or []]
    return messages, tools


def tokenizer(path):
    """The pinned tokenizer file, verified against the committed model lock."""
    from tokenizers import Tokenizer
    path = Path(path)
    lock = json.loads(Path(__file__).with_name('ppo_source_lock.json').read_text())['prepared_assets']['tokenizer.json']
    if path.stat().st_size != lock['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != lock['sha256']:
        raise ValueError('Tokenizer differs from the pinned model lock')
    loaded = Tokenizer.from_file(str(path))
    return lambda text: len(loaded.encode(text, add_special_tokens=False).ids)


def tool_definitions():
    return [{'type': 'function', 'function': {'name': n, 'description': task.TOOL_DESCRIPTIONS[n],
                                              'parameters': task.TOOL_SCHEMAS[n], 'strict': False}} for n in task.TOOLS]


def conversation(question, actions, final=False):
    """Wire messages the production harness sends after the given (tool, arguments) actions.

    `arguments` may be a dict or a raw string. Observations come from the public tool
    surface; typed errors are rendered exactly as the server reports them.
    """
    try:
        from .recovery_agent import SYSTEM_MESSAGE, FINAL_INSTRUCTION
    except ImportError:
        from recovery_agent import SYSTEM_MESSAGE, FINAL_INSTRUCTION
    messages = [{'role': 'system', 'content': SYSTEM_MESSAGE},
                {'role': 'user', 'name': 'user', 'content': question + '\n' + task.PROMPT}]
    for number, (name, arguments) in enumerate(actions, 1):
        raw = arguments if isinstance(arguments, str) else task.dumps(arguments)
        try:
            content = task.dumps(task.call(name, json.loads(raw)))
        except task.PolicyActionError as exc:
            content = task.dumps(task.policy_error(str(exc)))
        messages += [{'role': 'assistant', 'content': None, 'tool_calls': [
                         {'id': f'call-{number}', 'type': 'function', 'function': {'name': name, 'arguments': raw}}]},
                     {'role': 'tool', 'content': content, 'tool_call_id': f'call-{number}'}]
    if final:
        messages.append({'role': 'user', 'name': 'user', 'content': FINAL_INSTRUCTION})
    return messages


def episode(question, actions, reflect=False):
    """Every prompt of one episode: the initial call, each continuation and, if the
    tool rounds are exhausted, the tool-less reflection with its reminder."""
    views = []
    for done in range(len(actions) + 1):
        last = done == len(actions)
        if last and reflect:
            text, kind, tools = render(conversation(question, actions, final=True)), 'reflection', False
        else:
            text, kind, tools = render(conversation(question, actions[:done]), tool_definitions()), \
                'initial' if done == 0 else 'continuation', True
        views.append({'call': done + 1, 'kind': kind, 'tools_advertised': tools,
                      'tool_choice': 'auto' if tools else 'none (no tools sent)',
                      'generation': {'max_tokens': LIMITS['max_tokens'], 'parallel_tool_calls': False,
                                     'temperature': 'set by the gateway route: 0 for val, 1 for train'},
                      'prompt_sha256': hashlib.sha256(text.encode()).hexdigest(), 'prompt': text})
    return views


def worst_case_actions():
    """Five rounds of the largest inventory the world can return (all runs visible)."""
    registry = task.registry()
    asset_id = max(sorted(registry['runs']), key=lambda a: len(task.dumps(
        task.call('list_backup_runs', {'asset_id': a, 'as_of': '2027-12-31T23:59Z'}))))
    return [('list_backup_runs', {'asset_id': asset_id, 'as_of': '2027-12-31T23:59Z'})] * task.EPISODE_POLICY['max_tool_iterations']


def scenarios():
    """The measured prompt sets: development reference and five-round paths, and the
    bounded worst case with and without an initial typed error."""
    rounds = task.EPISODE_POLICY['max_tool_iterations']
    reference, extended = [], []
    for row in task.rows('development'):
        asset_id, as_of = task.target(row['question'])
        policy_id = task.call('get_asset', {'asset_id': asset_id})['policy_id']
        actions = [('get_asset', {'asset_id': asset_id}), ('get_policy', {'policy_id': policy_id}),
                   ('list_backup_runs', {'asset_id': asset_id, 'as_of': as_of})]
        reference.append([v['prompt'] for v in episode(row['question'], actions)])
        extended.append([v['prompt'] for v in episode(row['question'], actions + actions[2:] * (rounds - 3), reflect=True)])
    longest = max(task.rows('development'), key=lambda r: len(r['question']))['question']
    worst = [v['prompt'] for v in episode(longest, worst_case_actions(), reflect=True)]
    error = [v['prompt'] for v in episode(longest, [('get_asset', {'asset_id': 'unknown'})] + worst_case_actions()[1:], reflect=True)]
    return {'reference': reference, 'extended': extended, 'worst': worst, 'error': error}


def prompt_set_sha256(prompts=None):
    """Identity of every measured prompt; a committed measurement is current only if it matches."""
    prompts = prompts or scenarios()
    return task.digest({name: [[hashlib.sha256(p.encode()).hexdigest() for p in row] for row in
                               (rows if name in ('reference', 'extended') else [rows])] for name, rows in prompts.items()})


def budget(count):
    """Measured prompt sizes against the actual prompt, model and trajectory limits.

    `count` maps prompt text to its token count under the pinned tokenizer. Every
    development request is measured on its reference path; the worst case is
    bounded by the episode policy and the largest observation in the whole world.
    """
    rounds = task.EPISODE_POLICY['max_tool_iterations']
    prompts = scenarios()
    reference = [[count(p) for p in row] for row in prompts['reference']]
    extended = [[count(p) for p in row] for row in prompts['extended']]
    worst, error = [count(p) for p in prompts['worst']], [count(p) for p in prompts['error']]
    largest = max(max(worst), max(error))
    # Two extremes of the trajectory aggregator. Live Phase 3 collections put every call
    # in its own fragment, so each whole prompt must fit the prompt limit. If instead
    # every turn merged into one row, everything after the first prompt would be
    # "response": observations and reminder (masked) plus sampled tokens (unmasked).
    merged = max(worst[-1], error[-1]) - min(worst[0], error[0]) + ANSWER_TOKENS
    checks = {
        'every_prompt_within_trajectory_prompt_limit': largest <= LIMITS['trajectory_max_prompt_length'],
        'prompt_plus_generation_within_model_context': largest + LIMITS['max_tokens'] <= LIMITS['max_model_len'],
        'merged_trajectory_within_response_limit': merged <= LIMITS['trajectory_max_response_length']}
    return {'schema': 'recovery-prompt-budget-v1', 'task': task.TASK, 'episode_policy': task.EPISODE_POLICY['version'],
            'measured_prompts_sha256': prompt_set_sha256(prompts),
            'template_sha256': hashlib.sha256(prepare_model.effective_template().encode()).hexdigest(),
            'context_limits': task.CONTEXT_LIMITS, 'assumed_answer_tokens': ANSWER_TOKENS,
            'development_reference_path': {'cases': len(reference), 'calls': 4,
                                           'prompt_tokens_by_call_max': [max(c) for c in zip(*reference)],
                                           'prompt_tokens_by_call_min': [min(c) for c in zip(*reference)]},
            'development_five_round_reflection': {'cases': len(extended), 'calls': rounds + 1,
                                                  'prompt_tokens_by_call_max': [max(c) for c in zip(*extended)]},
            'worst_case_bounded': {'basis': f'{rounds} rounds of the largest inventory in the world, then reflection',
                                   'prompt_tokens_by_call': worst, 'with_initial_typed_error': error,
                                   'largest_prompt_tokens': largest, 'merged_response_tokens_upper_bound': merged,
                                   'generation_headroom_tokens': LIMITS['trajectory_max_response_length'] - merged + ANSWER_TOKENS},
            'checks': checks, 'within_limits': all(checks.values()),
            'truncation_policy': 'never hidden: the adapter rejects a collection whose fragment tokens differ from sampled tokens'}


def replay(directory, count):
    """Compare rendered prompts with server-reported prompt tokens in collected traces.

    Works on any collected stage directory. Agreement shows that this renderer and
    tokenizer reproduce what the served stack tokenized for those recorded calls.
    """
    calls = exact = 0
    differences = {}
    for path in sorted(Path(directory).rglob('trajectory-*.jsonl')):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            usage = (record.get('outputs') or {}).get('usage') if record.get('kind') == 'llm' else None
            if not usage or not usage.get('prompt_tokens'):
                continue
            messages, tools = from_trace(record['inputs'])
            difference = count(render(messages, tools)) - usage['prompt_tokens']
            calls += 1
            exact += difference == 0
            if difference:
                differences[str(difference)] = differences.get(str(difference), 0) + 1
    return {'schema': 'effective-prompt-replay-v1', 'model_calls_with_reported_usage': calls,
            'rendered_token_count_equals_reported': exact, 'differences': differences}
