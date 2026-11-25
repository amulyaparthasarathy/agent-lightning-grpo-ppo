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
from autogen_ext.models.openai import OpenAIChatCompletionClient
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


ABSENT = object()


def completion(step, *, content=None, calls=(), finish=None, completion_tokens=ABSENT, usage=True, call_ids=None, stop_reason=None):
    """One scripted chat completion in the server's response format.

    Completion tokens default to a small positive count, as a server would report.
    usage=False omits usage; content=None is null content, as for a tool call.
    """
    message = {'role': 'assistant', 'content': content}
    if calls:
        message['tool_calls'] = [{'id': (call_ids or [f'scripted-call-{step}-{i}' for i in range(len(calls))])[i], 'type': 'function',
                                  'function': {'name': name, 'arguments': arguments}}
                                 for i, (name, arguments) in enumerate(calls)]
    if completion_tokens is ABSENT:
        completion_tokens = 12 if calls else max(1, len(content or '') // 4)
    choice = {'index': 0, 'message': message, 'finish_reason': finish or ('tool_calls' if calls else 'stop')}
    if stop_reason is not None:
        choice['stop_reason'] = stop_reason
    body = {'id': f'scripted-{step}', 'object': 'chat.completion', 'created': 0, 'model': 'auto', 'choices': [choice]}
    if usage:
        body['usage'] = {'prompt_tokens': 0, 'completion_tokens': completion_tokens, 'total_tokens': completion_tokens}
    return ChatCompletion.model_validate(body)


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
                attempt='one', fault_step=None, simulate_live_labels=False):
    """One production episode with scripted SDK responses. Returns everything observable.

    Traces are labelled scripted-inference unless simulate_live_labels is set, which
    only tests of the live-evidence admission path do (see `collected`).
    """
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
    pinned_create = OpenAIChatCompletionClient.create

    async def tampered_create(client, messages, **kwargs):
        # Below the served-identity layer: what the pinned client hands back.
        result = await pinned_create(client, messages, **kwargs)
        if mode == 'served-not-captured':
            client.served = None
        elif isinstance(result.content, list) and mode == 'parsed-arguments-differ':
            result.content[0].arguments = result.content[0].arguments + ' '
        elif isinstance(result.content, list) and mode == 'parsed-id-differs':
            result.content[0].id = 'another-call'
        return result
    error = message = None
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, env))
            if not simulate_live_labels:
                stack.enter_context(patch.object(recovery_agent, 'Recorder', ScriptedRecorder))
            stack.enter_context(patch.object(AsyncCompletions, 'create', create))
            if mode in ('tool-backend-error', 'corrupt-observation'):
                stack.enter_context(patch.object(recovery_agent.ObservedWorkbench, 'call_tool', faulty_call))
            if mode in ('served-not-captured', 'parsed-arguments-differ', 'parsed-id-differs'):
                stack.enter_context(patch.object(OpenAIChatCompletionClient, 'create', tampered_create))
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

    The result is a directory in the format `reader_training.py collect` writes, so
    the report command reviews it exactly as it would review collected evidence.
    Traces and episode rows are real harness output over real stdio MCP. Audit
    events, receipts and the supervisor index are simulated. Everything is labelled
    scripted-inference and can support no calibration decision.
    """
    from experiment import reader_assets as assets, reader_contract as c, reader_evidence
    profile = assets.read(Path(__file__).resolve().parents[1]/'examples/reader-profile.recovery-calibration.synthetic.json')
    plan = c.make_plan(profile, c.source_identity())
    collected(output, plan)
    reviewed = reader_evidence.calibration_collection(plan, assets.read(Path(output)/'collection.json')['index'],
                                                      reader_evidence.local_reader(plan, output), local=output)
    report = reviewed['calibration']
    assets.write_new(Path(output)/'report.json', report)
    from experiment.recovery_calibration import summary
    (Path(output)/'summary.md').write_text(summary(report))
    return {'schema': 'recovery-scripted-calibration-v1', 'evidence_class': report['evidence_class'],
            'live_policy_measurement': report['live_policy_measurement'],
            'episodes_run': sum(r['accounting']['observed_attempts'] for r in report['routes'].values()),
            'decisions': {a: {k: r['decision'][k] for k in ('operational_validity', 'scientific_suitability')}
                          for a, r in report['routes'].items()}, 'output': str(output)}


def collected(output, plan, *, simulate_live_labels=False, behavior=scripted_behavior, only=None):
    """TEST ONLY: a collected-attempt directory in production format.

    Every episode runs through the production harness over real stdio MCP with
    scripted model responses. The stage completion is built by the production
    completion functions. Audit events, worker identities, receipts and the
    supervisor index are simulated in their production formats.

    simulate_live_labels=True leaves the traces with the labels a live run would
    carry, so that tests can exercise the path that admits live evidence. Such a
    directory is still simulated: it is written only under a test's temporary
    directory, its collection.json says so, and no model produced it.

    `only` restricts execution to some (algorithm, stage) pairs; the others are
    recorded in the index as never dispatched.
    """
    from unittest.mock import patch as mock_patch
    from experiment import matched_launch, matched_supervisor, reader_assets as assets, reader_contract as c, reader_evidence
    from experiment import recovery_calibration as calibration, summarize
    import recovery_fixtures
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    assets.write_new(output/'plan.json', plan)
    evidence_class = 'live-model' if simulate_live_labels else 'scripted-inference'
    rows = {r['case_id']: r for r in task.rows('development')}
    index = {'study_key': matched_supervisor.key(plan), 'status': 'completed', 'plan': plan, 'jobs': {}}
    clock = 1000.0
    for job in plan['jobs']:
        receipts = []
        for stage in job['stages']:
            request = c.stage_request(plan, job, stage)
            declared = plan['calibration']['stages'][stage['name']]
            run = output/job['algorithm']/stage['name']
            if only is not None and (job['algorithm'], stage['name']) not in only:
                receipts.append({'stage': stage['name'], 'output': stage['output'], 'wandb': stage['wandb'], 'status': 'skipped',
                                 'call_id': None, 'reason': 'dependency_or_supervisor_failed'})
                continue
            for case_id in declared['case_ids']:
                for sample in range(declared['rollouts_per_question']):
                    result = run_episode(run, SCRIPTED_BEHAVIORS[behavior(job['algorithm'], stage['name'], case_id, sample)],
                                         row=rows[case_id], request=request, route=declared['route'],
                                         rollout=f"ro-{case_id.rsplit('/', 1)[1]}-{sample}", attempt=f'at-{sample}',
                                         simulate_live_labels=simulate_live_labels)
                    if result['error'] or not result['cleanup']['complete']:
                        raise ValueError('Scripted calibration episode failed; inspect '+str(run))
            episodes = calibration.episode_table(run, request, evidence_class)
            call = 'fc-' + stage['wandb']['id']
            events = recovery_fixtures.events(declared, episodes, wandb_id=stage['wandb']['id'], pid=4242)
            (run/'audit').mkdir()
            (run/'audit/events-4242.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
            worker = {'AGL_READER_CALL_ID': call, 'MODAL_TASK_ID': 'ta-' + c.digest(call)[:16]}
            if simulate_live_labels:
                # The two production completion functions, exactly as the stage worker calls them.
                with mock_patch.dict(os.environ, worker):
                    proof = reader_evidence.complete(run, request, matched_launch.complete_stage(run, request, None))
            else:
                # The production path admits only live evidence; mirror its fields for scripted traces.
                proof = {'protocol': c.PROTOCOL, 'plan_sha256': c.digest(plan), 'stage': stage['name'],
                         **calibration.complete(run, request, events, evidence_class),
                         'trajectories': summarize.summarize(run, stage['name'], declared['route'])[0],
                         'execution_schema': plan['schema'], 'source_sha256': c.digest(plan['source']),
                         'configuration_sha256': plan['configuration_sha256'], 'assets_sha256': c.digest(plan['assets']),
                         'task_sha256': c.digest(plan['task_identity']),
                         'worker': {'call_id': call, 'hostname': 'simulated', 'container_id': worker['MODAL_TASK_ID'],
                                    'launcher_pid': 0, 'trainer_pid': 4242},
                         'wandb_id': stage['wandb']['id'], 'model_tool_integration': reader_integration.report(run, request)}
            receipt = {'status': 'completed', 'call_id': call, 'output': stage['output'], 'started_at': clock,
                       'container_id': worker['MODAL_TASK_ID'], 'completion': proof, 'finished_at': clock + 360.0}
            clock += 400.0
            for name, value in (('completion.json', proof), ('receipt.json', receipt), ('request.json', request),
                                ('observability.json', {'experiment_id': job['job_id'], 'wandb_run_id': stage['wandb']['id'],
                                                        'wandb_project': stage['wandb']['project'], 'wandb': stage['wandb']})):
                (run/name).write_text(json.dumps(value, indent=2)+'\n')
            receipts.append({'stage': stage['name'], 'output': stage['output'], 'status': 'completed', 'wandb': stage['wandb'],
                             'submitted_at': receipt['started_at'] - 5, 'call_id': call, 'result': receipt,
                             'finished_at': receipt['finished_at']})
        index['jobs'][job['job_id']] = {'status': 'completed' if all(r['status'] == 'completed' for r in receipts) else 'failed',
                                         'stages': receipts}
    if any(j['status'] != 'completed' for j in index['jobs'].values()):
        index['status'] = 'failed'
    assets.write_new(output/'collection.json', {
        'SIMULATED': 'test fixture or scripted demonstration; no model, trainer, provider or supervisor produced this directory',
        'inference': 'scripted OpenAI SDK responses', 'trace_labels': evidence_class,
        'adapter_trainer_receipt_and_index_evidence': 'simulated in production formats',
        'plan_sha256': c.digest(plan), 'index': index})
    return index
