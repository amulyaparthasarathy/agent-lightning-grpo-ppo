"""TEST ONLY: scripted OpenAI response and local HTTP reward boundaries.

Production Agent.run, ObservedClient, AssistantAgent and MCP transport execute.
This module is excluded from the image. Traces are explicitly scripted-inference.
"""
import asyncio
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == '__main__' and sys.argv[1:] and sys.argv[1] == '--hanging-server':
    from experiment.owned_processes import register
    import time
    register(sys.argv[2])
    time.sleep(60)
    raise SystemExit(0)
import tempfile
from threading import Thread
from unittest.mock import patch
from autogen_core import FunctionCall
from autogen_core.models import CreateResult, RequestUsage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from experiment import asset_agent, asset_task as task, owned_processes, reader_integration
from experiment.observed_runtime import Recorder


def run_case(root, final=None, mode='normal', request=None):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    row = task.cases()['eval_main.parquet'][0]
    aid = task.target(row['question'])
    truth = task.lookup({'asset_id': aid})
    if final is None:
        final = task.dumps(truth)
    posts, inputs, options = [], [], []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            posts.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(500 if mode == 'reward-failure' else 200)
            self.end_headers()
        def log_message(self, *_):
            pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    class ScriptedRecorder(Recorder):
        def __init__(self, question):
            super().__init__(question)
            self.common.update(synthetic_fixture=True, evidence_class='scripted-inference')
    async def create(_self, messages, **kwargs):
        inputs.append([m.model_dump() for m in messages])
        options.append({k: v for k, v in kwargs.items() if k in ('tools', 'tool_choice', 'json_output', 'extra_create_args')})
        if mode == 'transport-failure':
            raise ConnectionError('test-only model transport failure')
        if mode == 'timeout':
            raise asyncio.TimeoutError('test-only model timeout')
        if mode == 'cancelled':
            raise asyncio.CancelledError()
        step = len(inputs)
        if mode.endswith('-second') and step == 2 or mode.endswith('-reflection') and step == 3:
            if mode.startswith('cancelled'):
                raise asyncio.CancelledError()
            if mode.startswith('transport-failure'):
                raise ConnectionError('test-only model transport failure after tool')
            if mode.startswith('missing'):
                return CreateResult(content='', finish_reason='stop', usage=RequestUsage(prompt_tokens=0, completion_tokens=0), cached=False)
        two_calls = mode in ('two-tools', 'recover', 'policy-limit', 'malformed-recover', 'unknown-recover', 'infrastructure-after-tool') or mode.endswith('-reflection')
        if step == 1 or (two_calls and step == 2):
            arguments = task.dumps({'asset_id': aid})
            name = 'lookup_asset'
            if (mode in ('recover', 'policy-limit') and step == 1) or (mode == 'policy-limit' and step == 2):
                arguments = task.dumps({'asset_id': aid, 'extra': 'keep this invalid action'})
            if mode == 'malformed-recover' and step == 1:
                arguments = '{'
            if mode == 'unknown-recover' and step == 1:
                name = 'missing_tool'
            return CreateResult(content=[FunctionCall(id=f'scripted-call-{step}', name=name, arguments=arguments)],
                                finish_reason='length' if mode == 'truncated-action' else 'function_calls', usage=RequestUsage(prompt_tokens=0, completion_tokens=0), cached=False)
        answer = task.dumps({**truth, 'service_state': next(x for x in task.SERVICE if x != truth['service_state'])}) if mode == 'corrupt-observation' else final
        return CreateResult(content=answer, finish_reason='length' if mode in ('truncated', 'truncated-reflection') else 'stop',
                            usage=RequestUsage(prompt_tokens=0, completion_tokens=0), cached=False)
    env = {'QUESTION': row['question'], 'RESULT': row['result'], 'AGL_CASE_ID': row['case_id'],
           'AGL_DATA_SPLIT': 'heldout', 'EVAL_SPLIT': 'heldout', 'AGL_KEY': 'local-test-only',
           'HF_TOKEN': 'test-secret-canary', 'WANDB_API_KEY': 'test-secret-canary',
           'AGL_EVENT_URL': f'http://127.0.0.1:{server.server_port}/reward',
           'AGL_OPENAI_BASE_URL': 'http://127.0.0.1:1/proxy/rollout/scripted/attempt/one/mode/val/openai/v1',
           'AGL_TRACE_DIR': str(root/'traces'), 'AGL_MCP_OWNERS': str(root/'owners')}
    if request:
        from experiment.reader_contract import digest
        p, j, s = request['plan'], request['job'], request['stage']
        env['AGL_OPENAI_BASE_URL'] = env['AGL_OPENAI_BASE_URL'].replace('/mode/val/', '/mode/train/' if s['name'] == 'training' else '/mode/val/')
        env.update(AGL_TASK_ID=task.TASK, AGL_TASK_SHA256=digest(p['task_identity']),
            AGL_ENVIRONMENT_SHA256=task.digest(task.environment()), AGL_SOURCE_SHA256=digest(p['source']),
            AGL_CONFIGURATION_SHA256=p['configuration_sha256'], AGL_EXPERIMENT_ID=j['job_id'],
            AGL_WANDB_RUN_ID=s['wandb']['id'], AGL_PHASE=s['name'], AGL_POLICY_VERSION=p['protocol'], AGL_MODEL_PROFILE='llama3b')
    original_call = asset_agent.ObservedWorkbench.call_tool
    async def corrupt_call(workbench, *args, **kwargs):
        result = await original_call(workbench, *args, **kwargs)  # actual MCP execution still occurs
        if mode == 'infrastructure-after-tool' and len(inputs) == 2:
            result.is_error = True
            result.result[0].content = 'backend unavailable'
            return result
        if not result.is_error:
            result.result[0].content = task.dumps({**truth, 'service_state': next(x for x in task.SERVICE if x != truth['service_state'])})
        return result
    error = None
    try:
        with ExitStack() as stack, patch.dict(os.environ, env), patch.object(asset_agent, 'Recorder', ScriptedRecorder), patch.object(OpenAIChatCompletionClient, 'create', create):
            if mode in ('corrupt-observation', 'infrastructure-after-tool'):
                stack.enter_context(patch.object(asset_agent.ObservedWorkbench, 'call_tool', corrupt_call))
            try:
                asyncio.run(asset_agent.Agent().run())
            except BaseException as exc:
                error = type(exc).__name__
    finally:
        server.shutdown(); server.server_close(); thread.join(5)
        cleanup = owned_processes.cleanup(root/'owners')
    records = [json.loads(line) for p in (root/'traces').glob('*.jsonl') for line in p.read_text().splitlines()]
    children = [json.loads(p.read_text()) for p in (root/'owners').glob('*.json')]
    return {'posts': posts, 'inputs': inputs, 'options': options, 'records': records, 'children': children, 'error': error, 'cleanup': cleanup}


def demonstration(output):
    from test_reader_training import source, profile
    from experiment import reader_contract as c
    p = profile(); p['task'] = task.TASK
    plan = c.make_plan(p, source()); job = plan['jobs'][0]
    request = c.stage_request(plan, job, job['stages'][0])
    with Path(output).open('x') as receipt, tempfile.TemporaryDirectory(prefix='asset-scripted-') as root:
        result = run_case(root, request=request)
        chains = reader_integration.chains(result['records'], request, evidence_class='scripted-inference')
        passed = result['cleanup']['complete'] and result['error'] is None and len(chains) == 1 and result['posts'] == [{'event_type': 'reward', 'data': {'value': 1.0}}]
        report = {'schema': 'asset-scripted-harness-v1', 'task': task.TASK,
            'environment_sha256': task.digest(task.environment()), 'status': 'passed' if passed else 'failed',
            'evidence_class': 'scripted-inference', 'transport': 'real local stdio MCP',
            'reward_boundary': 'real loopback HTTP, test-only receiver', 'live_qualification': False,
            'chains': chains, 'records': result['records'], 'owned_process_cleanup': result['cleanup']}
        json.dump(report, receipt, indent=2)
        receipt.write('\n')
    if not passed:
        raise ValueError('Scripted harness validation failed')
    return {k: v for k, v in report.items() if k != 'records'}
