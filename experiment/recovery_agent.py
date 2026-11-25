"""Production recovery-readiness harness (recovery-episode-v1).

No scripted-model mode, answer repair or simulated reward bypass. The episode
contract is experiment/recovery_task.py EPISODE_POLICY; asset-episode-v2 and its
harness (asset_agent.py) are unchanged.

Terminal outcomes, from the pinned AutoGen 0.7.5 AssistantAgent loop:
- complete_final / complete_recovered: a non-empty text answer with finish reason
  `stop`, from a tool-enabled turn or from the tool-less reflection that follows
  the fifth tool round. Scored by the strict evidence verifier, reward 0 or 1.
  An early answer without the required observations is a complete zero.
- terminal_policy_failure: the second typed, attributable invalid action. One
  confirmed zero, no further generation, no invented answer.
- incomplete (exception, no reward): anything else, including an unclassified tool
  error, a lost or altered observation, more than one call in a generation,
  confirmed truncation, an ambiguous exact-cap tool call, absent answer content,
  completion that cannot be established, cancellation, a backend failure or an
  unconfirmed reward post.

Completion and tool-call identity are read from the served response, not from the
pinned client's rewritten view of it (see ServedIdentity). A sampled tool name is
never rewritten: an invalid name is an ordinary typed unknown-tool error. An
answer that finished with `stop` and has string content is complete, so a
whitespace-only answer is scored as a malformed answer. An empty string is scored
that way only when completion tokens were reported; absent or null content is
never an answer.
"""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import httpx
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core.models import ModelFamily, UserMessage
from autogen_core.tools import TextResultContent, ToolResult
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
try:
    from . import recovery_task as task
    from .observed_runtime import Recorder, ObservedClient, ObservedWorkbench
except ImportError:
    import recovery_task as task
    from observed_runtime import Recorder, ObservedClient, ObservedWorkbench


SYSTEM_MESSAGE = (
    'You assess backup recovery readiness using the provided tools. '
    'Call one tool at a time: get_asset with asset_id, get_policy with policy_id, and '
    'list_backup_runs with asset_id and as_of. '
    'Use only identifiers and values that appear in the request or in tool results. '
    'The evaluation time is the as_of in the request. It is the current time for this task, '
    'even if another date appears elsewhere. '
    'When you have the asset record, its policy record and the backup inventory, give the final '
    'answer in the format requested by the user. Your entire final response must be a JSON object '
    'with exactly the keys "asset_id", "as_of", "decision", "asset_record_id", "policy_record_id", '
    '"inventory_id", "supporting_run_ids". '
    'Do not explain, summarize, use markdown, or add text outside the object. '
    'If a call fails because its arguments are invalid, correct the arguments and try again. '
    'You have at most five tool-call rounds. Two invalid tool calls end the task as a failure.'
)
FINAL_INSTRUCTION = (
    'Tool use has ended. Answer the original request now. '
    'Return only the JSON object with exactly "asset_id", "as_of", "decision", "asset_record_id", '
    '"policy_record_id", "inventory_id" and "supporting_run_ids", using identifiers from the tool '
    'results above. No introduction, explanation, markdown, or other text.'
)


class PolicyLimitReached(Exception):
    """Declared environment termination after the second observed invalid action."""


class ObjectArguments(McpWorkbench):
    """MCP tools/call can only carry an object (or nothing) as arguments.

    Return the typed policy error for a sampled non-object without contacting the
    server and without altering the action (see asset_agent.ObjectArguments).
    """
    async def call_tool(self, name, arguments=None, cancellation_token=None, call_id=None):
        if name in task.TOOLS and not isinstance(arguments, dict):
            return ToolResult(name=name, is_error=True, result=[TextResultContent(
                content=task.dumps(task.policy_error('non_object_arguments')))])
        if name in task.TOOLS:
            try:
                json.dumps(arguments, ensure_ascii=False).encode('utf-8')
            except UnicodeEncodeError:
                # A lone surrogate cannot cross the stdio transport: the MCP client would
                # hang until its read timeout. Such a value never satisfies the declared
                # schema, so return exactly the typed error the server's validation gives.
                try:
                    task.call(name, arguments)
                except task.PolicyActionError as exc:
                    return ToolResult(name=name, is_error=True, result=[TextResultContent(
                        content=task.dumps(task.policy_error(str(exc))))])
        return await super().call_tool(name, arguments, cancellation_token, call_id)


class RecoveryWorkbench(ObservedWorkbench, ObjectArguments):
    """ObservedWorkbench.call_tool records, then delegates to ObjectArguments."""


def classify(call, observation):
    """Typed policy-error code for a sampled call's error observation, or None."""
    return task.classify_error(call.name, call.arguments, observation.content)


class ServedIdentity(OpenAIChatCompletionClient):
    """Keep what the server returned, below the observed client and above the pinned one.

    The pinned client rewrites a sampled tool name (every character outside
    [A-Za-z0-9_-] becomes "_", and the name is cut at 64), replaces a finish reason
    whenever tool calls are present, turns absent content into "" and absent usage
    into zeros. Scoring any of those rewritten values would score something the
    model did not sample. This layer restores the original tool-call identity on
    the parsed result, so the trace, the tool lookup, the error observation, the
    classification and the next prompt all carry the name as sampled, and records
    the served response for review. Nothing is repaired: an invalid name stays
    invalid and is answered by the framework's own unknown-tool error.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.served = None
        completions = self._client.chat.completions
        send = completions.create

        async def create(*args, **options):
            response = await send(*args, **options)
            choice, usage = response.choices[0], response.usage
            self.served = {
                'finish_reason': choice.finish_reason, 'content': choice.message.content,
                # vLLM reports which stop string or token ended generation, when any did.
                'stop_reason': (getattr(choice, 'model_extra', None) or {}).get('stop_reason'),
                'tool_calls': [{'id': c.id, 'name': c.function.name, 'arguments': c.function.arguments}
                               for c in choice.message.tool_calls or []],
                'usage': None if usage is None else {'prompt_tokens': getattr(usage, 'prompt_tokens', None),
                                                     'completion_tokens': getattr(usage, 'completion_tokens', None)}}
            return response
        completions.create = create

    async def create(self, messages, **kwargs):
        self.served = None
        result = await super().create(messages, **kwargs)
        served = self.served
        if served is None:
            raise RuntimeError('Served response was not captured; terminal reward forbidden')
        parsed = result.content if isinstance(result.content, list) else []
        client_names = [call.name for call in parsed]
        if ([call.id for call in parsed] != [c['id'] for c in served['tool_calls']]
                or any(call.arguments != c['arguments'] for call, c in zip(parsed, served['tool_calls']))
                or (not parsed and result.content != (served['content'] or ''))):
            raise RuntimeError('Parsed response differs from the served response; terminal reward forbidden')
        for call, original in zip(parsed, served['tool_calls']):
            call.name = original['name']
        self.completion = task.generation_status(served, self._create_args['max_tokens'])
        with self.recorder.stage('served_response', 'model_generation', {}, lambda: None) as (record, _):
            record['outputs'] = {
                'finish_reason': served['finish_reason'], 'content_present': type(served['content']) is str,
                'content_sha256': (hashlib.sha256(served['content'].encode()).hexdigest()
                                   if type(served['content']) is str else None),
                'stop_reason': served['stop_reason'], 'tool_calls': served['tool_calls'], 'usage': served['usage'],
                'completion': {'state': self.completion[0], 'reason': self.completion[1]},
                # What the pinned client made of the same response; never used for scoring.
                'framework_normalized': {'finish_reason': result.finish_reason, 'tool_names': client_names,
                                         'tool_names_differ': client_names != [c['name'] for c in served['tool_calls']]}}
        return result


class RecoveryClient(ObservedClient, ServedIdentity):
    """Observed inference plus exact call/observation joins and a final-format reminder.

    ObservedClient records each generation and delegates to ServedIdentity, so every
    record carries the tool call as sampled. Every sampled call is matched to its
    actual observation in the next model input before another generation is admitted.
    Successful observations enter the evidence ledger only then, so the ledger holds
    what demonstrably reached the model. No decoding constraint, repaired response or
    additional model is involved.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pending = []
        self.ledger = []
        self.policy_errors = 0
        self.successful_calls = 0
        self.tool_rounds = 0
        self.recovered = False
        self.completion = None
        self.call_ids = set()

    async def create(self, messages, **kwargs):
        for call in self.pending:
            observations = [o for m in messages if m.type == 'FunctionExecutionResultMessage'
                            for o in m.content if o.call_id == call.id and o.name == call.name]
            if len(observations) != 1:
                raise RuntimeError('Missing or ambiguous action observation')
            observation = observations[0]
            code = None
            if observation.is_error:
                code = classify(call, observation)
                if code is None:
                    raise RuntimeError('Unclassified tool failure; terminal reward forbidden')
                self.policy_errors += 1
            else:
                arguments = json.loads(call.arguments)
                try:
                    trusted = task.dumps(task.call(call.name, arguments))
                except task.PolicyActionError:
                    trusted = None
                if observation.content != trusted:
                    raise RuntimeError('Observation differs from the trusted environment; terminal reward forbidden')
                self.ledger.append({'tool': call.name, 'arguments': arguments, 'content': observation.content})
                self.successful_calls += 1
                self.recovered |= self.policy_errors > 0
            with self.recorder.stage('action_outcome', call.name,
                    {'call': call.model_dump() if hasattr(call, 'model_dump') else vars(call),
                     'observation': observation.model_dump()}, lambda: None) as (record, _):
                record['outputs'] = {'policy_error': code, 'successful_tool_call': not observation.is_error}
        self.pending = []
        if self.policy_errors >= task.EPISODE_POLICY['policy_error_limit']:
            with self.recorder.stage('termination', 'policy_error_limit',
                    {'messages': [m.model_dump() for m in messages]}, lambda: None) as (record, _):
                record['outputs'] = {'episode_policy': task.EPISODE_POLICY['version'],
                    'policy_errors': self.policy_errors, 'declared_terminal_task_failure': True}
            raise PolicyLimitReached()
        if kwargs.get('tool_choice') == 'none':
            messages = [*messages, UserMessage(content=FINAL_INSTRUCTION, source='user')]
        result = await super().create(messages, **kwargs)
        # Completion is established only by explicit served evidence (task.generation_status).
        state, reason = self.completion
        if state != 'complete':
            raise RuntimeError(f'Incomplete model generation ({state}): {reason}')
        if isinstance(result.content, list):
            # A call is joined to its observation by id. A repeated id would make the
            # join, and every later reading of the trace, ambiguous.
            if result.content[0].id in self.call_ids:
                raise RuntimeError('Tool call id repeats an earlier call; the observation join would be ambiguous')
            self.call_ids.add(result.content[0].id)
            self.pending = list(result.content)
            self.tool_rounds += 1
        return result


def server_params():
    # MCP's stdio client adds a small OS default environment. The server wrapper
    # clears that too, before importing task code (see recovery_stdio.py).
    return StdioServerParams(command=sys.executable,
        args=['-I', str(Path(__file__).with_name('recovery_stdio.py')), *([os.environ['AGL_MCP_OWNERS']] if os.getenv('AGL_MCP_OWNERS') else [])],
        env={'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1'}, read_timeout_seconds=120)


class Agent:
    async def run(self):
        question = os.environ['QUESTION']
        case = task.expected(question, os.environ['RESULT'])
        key, event_url = os.environ['AGL_KEY'], os.environ['AGL_EVENT_URL']
        policy = task.EPISODE_POLICY
        r = Recorder(question)
        r.common['episode_policy'] = policy['version']
        r.init_weave()
        client = None
        try:
            with r.stage('episode', task.TASK, {'question': question}, lambda: None) as (record, _):
                async with RecoveryWorkbench(server_params(), recorder=r) as workbench:
                    client = RecoveryClient(recorder=r, model='auto', base_url=os.environ['AGL_OPENAI_BASE_URL'], api_key=key,
                        model_info={'vision': False, 'function_calling': True, 'json_output': False,
                                    'family': ModelFamily.UNKNOWN, 'structured_output': False},
                        max_retries=6, parallel_tool_calls=policy['parallel_tool_calls'],
                        max_tokens=int(os.getenv('AGL_MAX_TOKENS', '1024')))
                    agent = AssistantAgent(name='recovery', model_client=client, workbench=workbench,
                                           system_message=SYSTEM_MESSAGE, reflect_on_tool_use=True,
                                           max_tool_iterations=policy['max_tool_iterations'])
                    terminal_failure = False
                    try:
                        response = await asyncio.wait_for(agent.run(task=question+'\n'+task.PROMPT), 300)
                    except PolicyLimitReached:
                        terminal_failure = True
                        response = None
                # A tool summary, truncation, cancellation or transport error is not terminal zero.
                last = response.messages[-1] if response and response.messages else None
                # An empty string is a completed answer here: the client admitted it only
                # after the served response showed string content and a `stop` finish.
                if not terminal_failure and (not isinstance(last, TextMessage) or last.source != 'recovery'
                        or not isinstance(last.content, str)
                        or r.last_finish_reason != 'stop' or client.pending):
                    raise RuntimeError('Incomplete recovery trajectory; terminal reward forbidden')
                if r.warnings or r.log is None:
                    raise RuntimeError('Missing trajectory evidence; terminal reward forbidden')
                answer = None if terminal_failure else last.content
                # Privileged verification after the trajectory is closed; never a model input.
                value, diagnostics = task.assess('' if terminal_failure else answer, case, client.ledger)
                if terminal_failure:
                    value = 0.0
                    diagnostics.update(json_syntax=None, format_compliant=None, failure_class='action')
                outcome = ('terminal_policy_failure' if terminal_failure else
                           'complete_recovered' if client.recovered else 'complete_final')
                record['outputs'] = {'raw_answer': answer, 'answer': answer, 'reward': value,
                    'outcome': outcome, 'termination': 'policy_error_limit' if terminal_failure else 'final_response',
                    'policy_errors': client.policy_errors, 'successful_tool_calls': client.successful_calls,
                    'tool_rounds': client.tool_rounds, 'recovered': client.recovered, 'diagnostics': diagnostics,
                    'answer_correct': bool(value), 'reward_posted': False, **r.counts}
                httpx.post(event_url, json={'event_type': 'reward', 'data': {'value': value}},
                           headers={'Authorization': f'Bearer {key}'}, timeout=10).raise_for_status()
                record['outputs']['reward_posted'] = True
        finally:
            try:
                if client is not None:
                    await client.close()
            finally:
                r.close()


if __name__ == '__main__':
    asyncio.run(Agent().run())
