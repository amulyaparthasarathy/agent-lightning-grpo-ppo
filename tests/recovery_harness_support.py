"""TEST ONLY: scripted OpenAI SDK responses and a local HTTP reward receiver.

The boundary is the OpenAI SDK call (AsyncCompletions.create). Production
Agent.run, RecoveryClient, AutoGen request construction and response parsing,
AssistantAgent and the real stdio MCP server all execute. This module is excluded
from the image. Traces are explicitly `scripted-inference`; they describe harness
mechanics, never model behavior.
"""
import asyncio
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tempfile
from threading import Thread
from unittest.mock import patch
import httpx
import openai
from openai.resources.chat.completions import AsyncCompletions
from openai.types.chat import ChatCompletion
from autogen_agentchat.agents import AssistantAgent
from experiment import recovery_agent, recovery_task as task, recovery_solver as solver
from experiment import owned_processes, reader_integration
from experiment.observed_runtime import Recorder


class State:
    """What a scripted policy may read: the request it was sent, nothing privileged."""
    def __init__(self, request, step):
        self.request, self.step, self.messages = request, step, request['messages']
        self.question = next(m['content'] for m in self.messages if m['role'] == 'user').split('\n')[0]
        self.asset_id, self.as_of = solver.request(self.question)
        calls = {c['id']: c['function'] for m in self.messages if m['role'] == 'assistant'
                 for c in m.get('tool_calls') or []}
        self.observations = []
        for m in self.messages:
            if m['role'] == 'tool':
                try:
                    parsed = json.loads(m['content'])
                except ValueError:
                    parsed = None
                self.observations.append({'name': calls[m['tool_call_id']]['name'], 'content': m['content'],
                                          'parsed': parsed if isinstance(parsed, dict) and 'error_type' not in parsed else None})

    def seen(self, name):
        return next((o['parsed'] for o in self.observations if o['name'] == name and o['parsed']), None)

    def answer(self, **changes):
        asset, policy, inventory = (self.seen(n) for n in task.TOOLS)
        usable = solver.qualifying(inventory, policy['rpo'], self.as_of)
        value = {'asset_id': self.asset_id, 'as_of': self.as_of, 'decision': 'ready' if usable else 'not_ready',
                 'asset_record_id': asset['record_id'], 'policy_record_id': policy['record_id'],
                 'inventory_id': inventory['inventory_id'], 'supporting_run_ids': usable}
        value.update(changes)
        return task.dumps(value)


def completion(step, *, content=None, calls=(), finish=None, completion_tokens=0):
    message = {'role': 'assistant', 'content': content}
    if calls:
        message['tool_calls'] = [{'id': f'scripted-call-{step}-{i}', 'type': 'function',
                                  'function': {'name': name, 'arguments': arguments}}
                                 for i, (name, arguments) in enumerate(calls)]
    return ChatCompletion.model_validate({
        'id': f'scripted-{step}', 'object': 'chat.completion', 'created': 0, 'model': 'auto',
        'choices': [{'index': 0, 'message': message, 'finish_reason': finish or ('tool_calls' if calls else 'stop')}],
        'usage': {'prompt_tokens': 0, 'completion_tokens': completion_tokens, 'total_tokens': completion_tokens}})


def CALL(name, **arguments):
    return lambda st: completion(st.step, calls=[(name, task.dumps(arguments))])


def RAW(name, arguments, **options):
    return lambda st: completion(st.step, calls=[(name, arguments)], **options)


def TEXT(text, **options):
    return lambda st: completion(st.step, content=text, **options)


def FAIL(error):
    def step(st):
        raise error() if isinstance(error, type) else error
    return step


def ASSET(st):
    return CALL('get_asset', asset_id=st.asset_id)(st)


def POLICY(st):
    return CALL('get_policy', policy_id=st.seen('get_asset')['policy_id'])(st)


def INVENTORY(st):
    return CALL('list_backup_runs', asset_id=st.asset_id, as_of=st.as_of)(st)


def FINAL(st):
    return TEXT(st.answer())(st)


def ANSWER(**changes):
    return lambda st: TEXT(st.answer(**changes))(st)


def SOLVE(st):
    """Next reference-solver action given only what the request shows."""
    for name, action in zip(task.TOOLS, (ASSET, POLICY, INVENTORY)):
        if st.seen(name) is None:
            return action(st)
    return FINAL(st)


def connection_error():
    return openai.APIConnectionError(request=httpx.Request('POST', 'http://127.0.0.1:1/v1/chat/completions'))


SOLVED = [SOLVE, SOLVE, SOLVE, SOLVE]


def identity_environment(request):
    """The identity variables the production launcher sets for one reader stage."""
    from experiment.reader_contract import digest
    p, j, s = request['plan'], request['job'], request['stage']
    return dict(AGL_TASK_ID=task.TASK, AGL_TASK_SHA256=digest(p['task_identity']),
                AGL_ENVIRONMENT_SHA256=task.digest(task.environment()), AGL_SOURCE_SHA256=digest(p['source']),
                AGL_CONFIGURATION_SHA256=p['configuration_sha256'], AGL_EXPERIMENT_ID=j['job_id'],
                AGL_WANDB_RUN_ID=s['wandb']['id'], AGL_PHASE=s['name'], AGL_POLICY_VERSION=p['protocol'],
                AGL_MODEL_PROFILE='llama3b')


def run_episode(root, script=None, *, row=None, mode='normal', request=None, route='val', rollout='scripted',
                attempt='one', fault_step=None):
    """One production episode with scripted SDK responses. Returns everything observable."""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    row = row or task.rows('development')[0]
    script = list(SOLVED if script is None else script)
    posts, requests = [], []

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

    async def create(_self, **kwargs):
        # Exactly the JSON body the SDK would send: omitted values are absent.
        requests.append(json.loads(json.dumps({k: v for k, v in kwargs.items()
                                               if type(v).__name__ not in ('NotGiven', 'Omit')})))
        if len(requests) > len(script):
            raise AssertionError('Scripted policy exhausted')
        return script[len(requests)-1](State(requests[-1], len(requests)))

    env = {'QUESTION': row['question'], 'RESULT': row['result'], 'AGL_CASE_ID': row['case_id'],
           'AGL_DATA_SPLIT': row['split'], 'EVAL_SPLIT': row['split'], 'AGL_KEY': 'local-test-only',
           'HF_TOKEN': 'test-secret-canary', 'WANDB_API_KEY': 'test-secret-canary',
           'AGL_EVENT_URL': f'http://127.0.0.1:{server.server_port}/reward',
           'AGL_OPENAI_BASE_URL': f'http://127.0.0.1:1/proxy/rollout/{rollout}/attempt/{attempt}/mode/{route}/openai/v1',
           'AGL_TRACE_DIR': str(root/'traces'), 'AGL_MCP_OWNERS': str(root/'owners')}
    if request:
        env.update(identity_environment(request))
    original_call = recovery_agent.ObservedWorkbench.call_tool
    original_execute = AssistantAgent._execute_tool_call
    executed = []

    async def faulty_call(workbench, *args, **kwargs):
        result = await original_call(workbench, *args, **kwargs)  # the real MCP call still happens
        executed.append(1)
        if fault_step in (None, len(executed)):
            if mode == 'tool-backend-error':
                result.is_error = True
                result.result[0].content = 'backend unavailable'
            elif mode == 'corrupt-observation' and not result.is_error:
                result.result[0].content = result.result[0].content.replace('"succeeded"', '"failed"').replace('rec-', 'rec-x')
        return result

    async def lost_observation(*args, **kwargs):
        call, result = await original_execute(*args, **kwargs)
        executed.append(1)
        if fault_step in (None, len(executed)):
            result.call_id = 'lost-'+result.call_id
        return call, result
    error = message = None
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, env))
            stack.enter_context(patch.object(recovery_agent, 'Recorder', ScriptedRecorder))
            stack.enter_context(patch.object(AsyncCompletions, 'create', create))
            if mode in ('tool-backend-error', 'corrupt-observation'):
                stack.enter_context(patch.object(recovery_agent.ObservedWorkbench, 'call_tool', faulty_call))
            if mode == 'lost-observation':
                stack.enter_context(patch.object(AssistantAgent, '_execute_tool_call', staticmethod(lost_observation)))
            try:
                asyncio.run(recovery_agent.Agent().run())
            except BaseException as exc:
                error, message = type(exc).__name__, str(exc)
    finally:
        server.shutdown(); server.server_close(); thread.join(5)
        cleanup = owned_processes.cleanup(root/'owners')
    records = [json.loads(line) for p in sorted((root/'traces').glob('*.jsonl')) for line in p.read_text().splitlines()]
    children = [json.loads(p.read_text()) for p in (root/'owners').glob('*.json')]
    return {'posts': posts, 'requests': requests, 'records': records, 'children': children,
            'error': error, 'message': message, 'cleanup': cleanup, 'root': str(root)}


def episode(result):
    return next(r for r in result['records'] if r['kind'] == 'episode')


def calibration_request(stage=0, algorithm=0):
    from test_reader_training import source
    from experiment import reader_assets as assets, reader_contract as c
    profile = assets.read(Path(__file__).resolve().parents[1]/'examples/reader-profile.recovery-calibration.synthetic.json')
    plan = c.make_plan(profile, source()); job = plan['jobs'][algorithm]
    return c.stage_request(plan, job, job['stages'][stage])


def demonstration(output):
    request = calibration_request()
    with Path(output).open('x') as receipt, tempfile.TemporaryDirectory(prefix='recovery-scripted-') as root:
        result = run_episode(root, request=request)
        chains = reader_integration.chains(result['records'], request, evidence_class='scripted-inference')
        passed = (result['cleanup']['complete'] and result['error'] is None and len(chains) == 3
                  and result['posts'] == [{'event_type': 'reward', 'data': {'value': 1.0}}])
        report = {'schema': 'recovery-scripted-harness-v1', 'task': task.TASK,
                  'environment_sha256': task.digest(task.environment()), 'status': 'passed' if passed else 'failed',
                  'evidence_class': 'scripted-inference', 'transport': 'real local stdio MCP',
                  'inference_boundary': 'scripted OpenAI SDK responses; no model',
                  'reward_boundary': 'real loopback HTTP, test-only receiver', 'live_qualification': False,
                  'chains': chains, 'records': result['records'], 'owned_process_cleanup': result['cleanup']}
        json.dump(report, receipt, indent=2)
        receipt.write('\n')
    if not passed:
        raise ValueError('Scripted harness validation failed')
    return {k: v for k, v in report.items() if k != 'records'}


def _flip(st):
    value = json.loads(st.answer())
    ready = value['decision'] == 'ready'
    inventory = st.seen('list_backup_runs')
    cited = [] if ready else [r['run_id'] for r in inventory['runs'][:1]] or ['run-00000000']
    return TEXT(st.answer(decision='not_ready' if ready else 'ready', supporting_run_ids=cited))(st)


def _ungrounded(st):
    # Skips the policy lookup and guesses: identifiers it never observed cannot be evidence.
    asset, inventory = st.seen('get_asset'), st.seen('list_backup_runs')
    good = [r['run_id'] for r in inventory['runs'] if r['status'] == 'succeeded' and r['integrity'] == 'passed']
    return TEXT(task.dumps({'asset_id': st.asset_id, 'as_of': st.as_of, 'decision': 'ready' if good else 'not_ready',
                            'asset_record_id': asset['record_id'], 'policy_record_id': 'rec-0000000000',
                            'inventory_id': inventory['inventory_id'], 'supporting_run_ids': good[:1]}))(st)


def _extra_argument(st):
    return CALL('get_asset', asset_id=st.asset_id, as_of=st.as_of)(st)


SCRIPTED_BEHAVIORS = {
    'correct': SOLVED,
    'recovered': [_extra_argument, SOLVE, SOLVE, SOLVE, SOLVE],
    'wrong_decision': [SOLVE, SOLVE, SOLVE, _flip],
    'ungrounded': [ASSET, INVENTORY, _ungrounded],
    'prose': [SOLVE, SOLVE, SOLVE, TEXT('The asset looks recovery-ready based on its latest backup.')],
    'two_invalid_actions': [_extra_argument, RAW('get_policy', '{')],
}


def scripted_behavior(algorithm, stage, case_id, sample):
    """A fixed, arbitrary mix of scripted behaviors. It models no policy whatsoever."""
    pick = int(task.digest(['scripted-demonstration', algorithm, stage, case_id, sample]), 16)
    if stage == 'greedy-development':
        return 'wrong_decision' if pick % 4 == 0 else 'correct'
    if pick % 37 == 0:
        return 'two_invalid_actions'
    return ('correct', 'correct', 'correct', 'correct', 'recovered', 'wrong_decision', 'wrong_decision',
            'ungrounded', 'prose', 'correct')[pick % 10]


def calibration_demonstration(output):
    """All 240 calibration episodes through the production harness with scripted inference.

    Traces and episode rows are real harness output over real stdio MCP. Adapter and
    trainer audit events are simulated (the real adapter is exercised in the pinned
    CPU gate). The result exercises completion, aggregation and reporting only.
    """
    from experiment import recovery_calibration as calibration, reader_assets as assets, reader_contract as c
    import recovery_fixtures
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    profile = assets.read(Path(__file__).resolve().parents[1]/'examples/reader-profile.recovery-calibration.synthetic.json')
    plan = c.make_plan(profile, c.source_identity())
    assets.write_new(output/'plan.json', plan)
    rows = {r['case_id']: r for r in task.rows('development')}
    completions = {}
    for job in plan['jobs']:
        for stage in job['stages']:
            request = c.stage_request(plan, job, stage)
            declared = plan['calibration']['stages'][stage['name']]
            run = output/job['algorithm']/stage['name']
            for case_id in declared['case_ids']:
                for sample in range(declared['rollouts_per_question']):
                    behavior = scripted_behavior(job['algorithm'], stage['name'], case_id, sample)
                    result = run_episode(run, SCRIPTED_BEHAVIORS[behavior], row=rows[case_id], request=request,
                                         route=declared['route'], rollout=f"ro-{case_id.rsplit('/', 1)[1]}-{sample}",
                                         attempt=f'at-{sample}')
                    if result['error'] or not result['cleanup']['complete']:
                        raise ValueError('Scripted calibration episode failed; inspect '+str(run))
            episodes = calibration.episode_table(run, request, 'scripted-inference')
            events = recovery_fixtures.events(declared, episodes)
            (run/'audit').mkdir()
            (run/'audit/events-simulated.jsonl').write_text(''.join(json.dumps(dict(e, simulated=True))+'\n' for e in events))
            done = {'plan_sha256': c.digest(plan), 'stage': stage['name'],
                    'adapter_and_trainer_events': 'simulated; not produced by a trainer process',
                    **calibration.assemble(request, events, episodes, 'scripted-inference')}
            assets.write_new(run/'completion.json', done)
            completions.setdefault(job['algorithm'], {})[stage['name']] = done
    report = calibration.report(plan, completions)
    assets.write_new(output/'report.json', report)
    (output/'summary.md').write_text(calibration.summary(report))
    return {'schema': 'recovery-scripted-calibration-v1', 'evidence_class': report['evidence_class'],
            'live_policy_measurement': False, 'episodes': sum(r['episodes']['attempted_and_admitted'] for r in report['routes'].values()),
            'decisions': {a: r['decision'] for a, r in report['routes'].items()}, 'output': str(output)}
