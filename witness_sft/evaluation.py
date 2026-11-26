"""Greedy loopback serving and independent recovery-agent trace replay."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import secrets
import signal
import subprocess
import sys
from threading import Thread
import time
from urllib.error import URLError
from urllib.request import urlopen

from experiment import reader_integration, recovery_calibration, recovery_learning
from . import data

ENDPOINT = 'http://127.0.0.1:8017'
STAGES = ('fresh-base', 'supervised-pass', 'terminal-reload')


def serve_command(model, base):
    return [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--host', '127.0.0.1',
            '--port', '8017', '--model', str(model), '--tokenizer', str(base), '--served-model-name', 'auto',
            '--load-format', 'safetensors', '--dtype', 'bfloat16', '--max-model-len', '4096', '--max-num-seqs', '1',
            '--enforce-eager', '--gpu-memory-utilization', '0.5', '--enable-auto-tool-choice',
            '--tool-call-parser', 'llama3_json', '--chat-template', str(base / 'chat_template.jinja'),
            '--seed', '2001', '--generation-config', str(base), '--override-generation-config',
            '{"temperature":0,"top_p":1,"top_k":-1}']


@contextmanager
def serving(model, base, log):
    with log.open('x') as output:
        proc = subprocess.Popen(serve_command(model, base), stdout=output, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            end = time.monotonic() + 420
            while time.monotonic() < end:
                if proc.poll() is not None:
                    raise RuntimeError('Local vLLM exited during startup')
                try:
                    with urlopen(ENDPOINT + '/v1/models', timeout=2) as response:
                        if any(m['id'] == 'auto' for m in json.load(response)['data']):
                            break
                except (URLError, TimeoutError):
                    pass
                time.sleep(1)
            else:
                raise TimeoutError('Local vLLM startup exceeded 420 seconds')
            yield
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


@contextmanager
def reward_sink():
    token, rewards = secrets.token_hex(24), []
    class Receiver(BaseHTTPRequestHandler):
        def do_POST(self):
            ok = False
            if self.path == '/reward' and self.headers.get('Authorization') == 'Bearer ' + token and not rewards:
                n = int(self.headers.get('Content-Length', 0))
                if 0 < n <= 512:
                    event = json.loads(self.rfile.read(n))
                    value = event.get('data', {}).get('value')
                    ok = (type(value) in (int, float) and value in (0, 1)
                          and event == {'event_type': 'reward', 'data': {'value': value}})
                    if ok:
                        rewards.append(value)
            self.send_response(200 if ok else 400)
            self.end_headers()

        def log_message(self, *args):
            pass

    with HTTPServer(('127.0.0.1', 0), Receiver) as http:
        thread = Thread(target=http.serve_forever, daemon=True)
        thread.start()
        try:
            yield token, f'http://127.0.0.1:{http.server_port}/reward', rewards
        finally:
            http.shutdown()
            thread.join()


def replay(row, records, rewards):
    outcome = reader_integration.episode_outcome(records)
    if outcome not in ('complete_final', 'complete_recovered', 'terminal_policy_failure'):
        raise ValueError('Incomplete agent trace')
    llms = [r for r in records if r['kind'] == 'llm']
    actions = [r for r in records if r['kind'] == 'action_outcome']
    ep = next(r for r in records if r['kind'] == 'episode')
    out = ep['outputs']
    if (any(r['case_id'] != row['case_id'] or r['trace_id'] != ep['trace_id'] for r in records)
            or ep['inputs'] != {'question': row['question']} or ep['episode_policy'] != data.task.EPISODE_POLICY['version']):
        raise ValueError('Trace case or policy mismatch')
    recovery_learning.verify_history(row['question'], llms, actions)
    recovery_calibration.served_evidence(records, llms)
    ledger = [{'tool': a['inputs']['call']['name'], 'arguments': json.loads(a['inputs']['call']['arguments']),
               'content': a['inputs']['observation']['content']} for a in actions if a['outputs']['successful_tool_call']]
    case = data.task.expected(row['question'], row['result'])
    value, diagnostics = data.task.assess(out['raw_answer'] or '', case, ledger)
    if outcome == 'terminal_policy_failure':
        diagnostics.update(json_syntax=None, format_compliant=None, failure_class='action')
    if rewards != [value] or out['reward'] != value or diagnostics != out['diagnostics']:
        raise ValueError('Reward event and strict replay disagree')
    return {'case_id': row['case_id'], 'actual': json.loads(row['result'])['decision'],
            'answer': out['raw_answer'], 'strict': value, 'diagnostics': diagnostics, 'outcome': outcome}


def summarize(rows):
    panel = data.partition(data.read_source(data.DIRECTORY), 'development')
    expected = [(r['case_id'], json.loads(r['result'])['decision']) for r in panel]
    if [(r['case_id'], r['actual']) for r in rows] != expected:
        raise ValueError('Wrong development cases, truth or order')
    if any(type(r['strict']) not in (int, float) or r['strict'] not in (0, 1)
           or r['diagnostics']['decision_correct'] not in (True, False, None) for r in rows):
        raise ValueError('Invalid episode reward or assessment')
    strict = {cls: sum(r['strict'] for r in rows if r['actual'] == cls) for cls in data.task.DECISIONS}
    assessable = sum(r['diagnostics']['decision_correct'] is not None for r in rows)
    return {'class_strict': strict, 'assessable': assessable,
            'gate_passed': min(strict.values()) >= 8 and assessable == 24}


def admit_live_trace(row, records, rewards, stage, plan_digest):
    if (stage not in (STAGES[0], STAGES[2]) or row['split'] != 'development' or not records
            or any(r.get('synthetic_fixture') is not False or r.get('evidence_class') != 'live-model'
                   or r.get('eval_split') != 'development' or r.get('phase') != stage
                   or r.get('configuration_sha256') != plan_digest for r in records)):
        raise ValueError('Wrong live evaluation trace identity')
    return replay(row, records, rewards)


def evaluate(model, base, run, plan_digest):
    result = []
    with serving(model, base, run / 'vllm.log'):
        for row in data.partition(data.read_source(data.DIRECTORY), 'development'):
            traces = run / f'case-{len(result):02d}'
            with reward_sink() as (key, event_url, rewards):
                env = dict(os.environ, QUESTION=row['question'], RESULT=row['result'], AGL_CASE_ID=row['case_id'],
                           EVAL_SPLIT='development', AGL_PHASE=run.name, AGL_CONFIGURATION_SHA256=plan_digest,
                           AGL_OPENAI_BASE_URL=ENDPOINT+'/v1', AGL_KEY=key,
                           AGL_EVENT_URL=event_url, AGL_TRACE_DIR=str(traces), AGL_MAX_TOKENS='1024',
                           AGL_WEAVE_PROJECT='')
                subprocess.run([sys.executable, '-m', 'experiment.recovery_agent'], env=env, check=True, timeout=330)
            trace, = traces.glob('trajectory-*.jsonl')
            records = [json.loads(line) for line in trace.read_text().splitlines()]
            result.append(admit_live_trace(row, records, rewards, run.name, plan_digest))
    return {'episodes': result, 'summary': summarize(result)}
