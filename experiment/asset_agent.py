"""Production asset harness; no scripted-model mode or simulated reward bypass."""
import asyncio
import os
from pathlib import Path
import sys
import httpx
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core.models import ModelFamily
from autogen_ext.tools.mcp import StdioServerParams
try:
    from . import asset_task as task
    from .observed_runtime import Recorder, ObservedClient, ObservedWorkbench
except ImportError:
    import asset_task as task
    from observed_runtime import Recorder, ObservedClient, ObservedWorkbench


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
        r.init_weave()
        client = None
        try:
            with r.stage('episode', task.TASK, {'question': question}, lambda: None) as (record, _):
                async with ObservedWorkbench(server_params(), recorder=r) as workbench:
                    client = ObservedClient(recorder=r, model='auto', base_url=os.environ['AGL_OPENAI_BASE_URL'], api_key=key,
                        model_info={'vision': False, 'function_calling': True, 'json_output': False,
                                    'family': ModelFamily.UNKNOWN, 'structured_output': False},
                        max_retries=6, parallel_tool_calls=False, max_tokens=int(os.getenv('AGL_MAX_TOKENS', '1024')))
                    agent = AssistantAgent(name='asset', model_client=client, workbench=workbench,
                                           reflect_on_tool_use=True, max_tool_iterations=2)
                    response = await asyncio.wait_for(agent.run(task=question+'\n'+task.PROMPT), 300)
                # A tool summary, truncation, cancellation or transport error is not terminal zero.
                last = response.messages[-1] if response.messages else None
                if (not isinstance(last, TextMessage) or last.source != 'asset'
                        or not isinstance(last.content, str) or not last.content.strip()
                        or r.last_finish_reason != 'stop' or r.counts['tool_errors']):
                    raise RuntimeError('Incomplete asset trajectory; terminal reward forbidden')
                value = task.reward(last.content, truth)
                record['outputs'] = {'raw_answer': last.content, 'answer': last.content, 'reward': value,
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
