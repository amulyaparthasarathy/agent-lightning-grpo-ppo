"""Production asset harness; no scripted-model mode or simulated reward bypass."""
import asyncio
import json
import os
from pathlib import Path
import sys
import httpx
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core.models import ModelFamily, UserMessage
from autogen_ext.tools.mcp import StdioServerParams
try:
    from . import asset_task as task
    from .observed_runtime import Recorder, ObservedClient, ObservedWorkbench
except ImportError:
    import asset_task as task
    from observed_runtime import Recorder, ObservedClient, ObservedWorkbench


SYSTEM_MESSAGE = (
    'You are an assistant that answers asset questions using the provided tool. '
    'Call lookup_asset with exactly one argument: asset_id. '
    'Do not supply service_state or backup_health as tool arguments; those fields '
    'are observations returned by the tool. '
    'After receiving the tool result, provide the final answer in the format '
    'requested by the user. Your entire final response must be a JSON object '
    'with exactly the keys "asset_id", "service_state", "backup_health". '
    'Use the observed values. Do not explain, summarize, use markdown, or add text outside the object. '
    'If a call fails because its arguments are invalid, correct the arguments and try again. '
    'You have at most two tool-call rounds. Two invalid actions end the task as a failure.'
)
FINAL_INSTRUCTION = (
    'Tool use has ended. Answer the original asset question now. '
    'Return only the JSON object with exactly "asset_id", "service_state", '
    'and "backup_health", using the observed values as strings. '
    'No introduction, explanation, markdown, or other text.'
)


class PolicyLimitReached(Exception):
    """Declared environment termination after two observed invalid actions."""


class AssetClient(ObservedClient):
    """Keep AutoGen's context and observed inference; add a final-format reminder.

    Match every sampled call to its actual observation before admitting another
    generation. Unrecognized MCP errors and missing observations fail closed.
    No decoding constraint, repaired response, or additional model is involved.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pending = []
        self.policy_errors = 0
        self.successful_calls = 0
        self.recovered = False

    async def create(self, messages, **kwargs):
        for call in self.pending:
            observations = [o for m in messages if m.type == 'FunctionExecutionResultMessage'
                            for o in m.content if o.call_id == call.id and o.name == call.name]
            if len(observations) != 1:
                raise RuntimeError('Missing or ambiguous action observation')
            observation = observations[0]
            code = None
            if observation.is_error:
                # These exact AutoGen 0.7.5 branches do not invoke MCP. A parser
                # exception elsewhere is not a policy mistake.
                try:
                    json.loads(call.arguments)
                except json.JSONDecodeError as exc:
                    if observation.content == f'Error: {exc}':
                        code = 'invalid_argument_json'
                if call.name != 'lookup_asset' and observation.content == f"Error: tool '{call.name}' not found in any workbench":
                    code = 'unknown_tool'
                if code is None:
                    try:
                        error = json.loads(observation.content)
                    except (ValueError, TypeError):
                        error = None
                    if (isinstance(error, dict) and error.get('error_type') == 'asset_policy_action_v2'
                            and error.get('code') in ('invalid_arguments', 'invalid_asset_id', 'unknown_asset')):
                        # Verify that the action itself fails the declared input
                        # validation with the same code, not just an error string.
                        try:
                            task.lookup(json.loads(call.arguments))
                        except task.PolicyActionError as exc:
                            if str(exc) == error['code']:
                                code = str(exc)
                if code is None:
                    raise RuntimeError('Unclassified tool failure; terminal reward forbidden')
                self.policy_errors += 1
            else:
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
        if result.finish_reason not in ('stop', 'function_calls'):
            raise RuntimeError('Incomplete model generation')
        if isinstance(result.content, list):
            if not result.content or result.finish_reason != 'function_calls':
                raise RuntimeError('Missing or inconsistent model actions')
            self.pending = list(result.content)
        elif not isinstance(result.content, str) or not result.content.strip() or result.finish_reason != 'stop':
            raise RuntimeError('Missing or inconsistent model answer')
        return result


def diagnostics(answer, truth):
    """Privileged diagnostics, never model inputs or additional reward terms."""
    result = {'json_syntax': False, 'format_compliant': False, 'target_correct': None,
              'content_correct': None}
    try:
        value = json.loads(answer, object_pairs_hook=task.unique_object)
        result['json_syntax'] = True
        parsed = task.parse(answer)
    except (ValueError, TypeError):
        return result
    result.update(format_compliant=True, target_correct=parsed['asset_id'] == truth['asset_id'],
                  content_correct=parsed == truth)
    return result


def server_params():
    # MCP's stdio client adds a small OS default environment. The server wrapper
    # clears that too, before importing task code (see asset_stdio.py).
    return StdioServerParams(command=sys.executable,
        args=['-I', str(Path(__file__).with_name('asset_stdio.py')), *([os.environ['AGL_MCP_OWNERS']] if os.getenv('AGL_MCP_OWNERS') else [])],
        env={'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1'}, read_timeout_seconds=120)


class Agent:
    async def run(self):
        question = os.environ['QUESTION']
        truth = task.expected(question, os.environ['RESULT'])
        key, event_url = os.environ['AGL_KEY'], os.environ['AGL_EVENT_URL']
        r = Recorder(question)
        r.common['episode_policy'] = task.EPISODE_POLICY['version']
        r.init_weave()
        client = None
        try:
            with r.stage('episode', task.TASK, {'question': question}, lambda: None) as (record, _):
                async with ObservedWorkbench(server_params(), recorder=r) as workbench:
                    client = AssetClient(recorder=r, model='auto', base_url=os.environ['AGL_OPENAI_BASE_URL'], api_key=key,
                        model_info={'vision': False, 'function_calling': True, 'json_output': False,
                                    'family': ModelFamily.UNKNOWN, 'structured_output': False},
                        max_retries=6, parallel_tool_calls=False, max_tokens=int(os.getenv('AGL_MAX_TOKENS', '1024')))
                    agent = AssistantAgent(name='asset', model_client=client, workbench=workbench,
                                           system_message=SYSTEM_MESSAGE,
                                           reflect_on_tool_use=True,
                                           max_tool_iterations=task.EPISODE_POLICY['max_tool_iterations'])
                    terminal_failure = False
                    try:
                        response = await asyncio.wait_for(agent.run(task=question+'\n'+task.PROMPT), 300)
                    except PolicyLimitReached:
                        terminal_failure = True
                        response = None
                # A tool summary, truncation, cancellation or transport error is not terminal zero.
                last = response.messages[-1] if response and response.messages else None
                if not terminal_failure and (not isinstance(last, TextMessage) or last.source != 'asset'
                        or not isinstance(last.content, str) or not last.content.strip()
                        or r.last_finish_reason != 'stop' or client.pending):
                    raise RuntimeError('Incomplete asset trajectory; terminal reward forbidden')
                if r.warnings or r.log is None:
                    raise RuntimeError('Missing trajectory evidence; terminal reward forbidden')
                answer = None if terminal_failure else last.content
                value = 0.0 if terminal_failure else task.reward(answer, truth)
                outcome = ('terminal_policy_failure' if terminal_failure else
                           'complete_recovered' if client.recovered else 'complete_final')
                record['outputs'] = {'raw_answer': answer, 'answer': answer, 'reward': value,
                    'outcome': outcome, 'termination': 'policy_error_limit' if terminal_failure else 'final_response',
                    'policy_errors': client.policy_errors, 'successful_tool_calls': client.successful_calls,
                    'recovered': client.recovered, 'diagnostics': diagnostics(answer, truth),
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
