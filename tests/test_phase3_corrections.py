"""Real owned subprocesses; explicitly mocked cloud completion/health boundaries."""
from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiment import owned_processes as owned, asset_probe, launch, matched_supervisor as supervisor
from experiment import reader_contract as c
from test_reader_training import plan, ROOT


class CleanupTests(unittest.TestCase):
    def child(self, directory):
        script = 'from experiment.owned_processes import register; import time; register('+repr(str(directory))+'); time.sleep(60)'
        proc = subprocess.Popen([sys.executable, '-c', script], cwd=ROOT, start_new_session=True)
        self.addCleanup(self.reap, proc)
        deadline = time.monotonic()+5
        while not list(directory.glob('*.json')) and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertEqual(len(list(directory.glob('*.json'))), 1)
        return proc

    @staticmethod
    def reap(proc):
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)

    def test_malformed_before_valid_group_and_partial_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            proc = self.child(directory)
            valid = next(directory.glob('*.json')); valid.rename(directory/'z-valid.json')
            (directory/'a-invalid.json').write_text('{"pid":')
            (directory/'b-interrupted.pending').write_text('{')
            report = owned.cleanup(directory)
            self.assertEqual(report['signalled'], 1)
            self.assertFalse(report['complete'])
            self.assertEqual(len(report['anomalies']), 2)
            self.assertEqual(report['receipts'][0]['status'], 'group_terminated')
            proc.wait(timeout=5)  # The owning test reaps; cleanup does not claim to reap another parent's child.
            self.assertEqual(proc.returncode, -signal.SIGKILL)
            self.assertTrue((directory/'a-invalid.json').exists())

    def test_interrupted_register_never_publishes_partial_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = '''from unittest.mock import patch
from experiment import owned_processes as o
import sys
def interrupted(row, f):
    f.write('{'); f.flush()
    raise RuntimeError('deterministic interrupted publication')
with patch.object(o.json, 'dump', side_effect=interrupted):
    o.register(sys.argv[1])
'''
            proc = subprocess.run([sys.executable, '-c', script, tmp], cwd=ROOT, start_new_session=True, capture_output=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertFalse(list(Path(tmp).glob('*.json')))
            self.assertEqual(len(list(Path(tmp).glob('*.pending'))), 1)
            with patch.object(owned.os, 'killpg') as kill:
                report = owned.cleanup(tmp)
            kill.assert_not_called()
            self.assertFalse(report['complete'])

    def test_stale_ambiguous_and_invalid_receipts_never_signal(self):
        for mode in ('stale', 'duplicate', 'wrong_type', 'duplicate_key', 'symlink'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp); proc = self.child(directory)
                p = next(directory.glob('*.json')); row = json.loads(p.read_text())
                if mode == 'stale': row['start_ticks'] = str(int(row['start_ticks'])+1); p.write_text(json.dumps(row))
                if mode == 'duplicate': (directory/'duplicate.json').write_text(p.read_text())
                if mode == 'wrong_type': row['pid'] = str(row['pid']); p.write_text(json.dumps(row))
                if mode == 'duplicate_key': p.write_text(p.read_text()[:-1]+',"pid": 2}')
                if mode == 'symlink':
                    target = directory/'original.pending'; p.rename(target); p.symlink_to(target)
                with patch.object(owned.os, 'killpg') as kill:
                    report = owned.cleanup(directory)
                kill.assert_not_called()
                self.assertFalse(report['complete'])
                self.assertIsNone(proc.poll())
                self.reap(proc)

    def test_launcher_retains_original_failure_and_closes_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp); (run/'logs').mkdir(); owners = run/'mcp-processes'; owners.mkdir()
            proc = self.child(owners)
            next(owners.glob('*.json')).rename(owners/'z-valid.json')
            (owners/'a-invalid.json').write_text('{')
            commands = {name: [sys.executable, '-c', 'import time; time.sleep(30)'] for name in ('gateway', 'controller')}
            commands['trainer'] = [sys.executable, '-c', 'raise RuntimeError("original trainer failure")']
            configured = dict(phase='baseline', policy_version='test', profile='llama3b', calc_dir=tmp, gateway='http://localhost:1', overrides=[])
            ready = Mock(); ready.__enter__ = Mock(return_value=SimpleNamespace(status=200)); ready.__exit__ = Mock(return_value=False)
            original = launch.subprocess.Popen; handles = []
            def start(*args, **kwargs):
                handles.append(kwargs['stdout'])
                return original(*args, **kwargs)
            with patch.object(launch.urllib.request, 'urlopen', return_value=ready), patch.object(launch.subprocess, 'Popen', side_effect=start), redirect_stderr(io.StringIO()):
                code = launch.run_services(commands, configured, run, run, run, .5)
            self.assertEqual(code, 1)
            result = json.loads((run/'result.json').read_text())
            self.assertEqual(result['status'], 'trainer_failed')
            self.assertFalse(result['owned_process_cleanup_completed'])
            self.assertEqual(result['cleanup']['mcp']['signalled'], 1)
            self.assertTrue(all(h.closed for h in handles))
            self.assertIn('original trainer failure', (run/'logs/trainer.log').read_text())
            self.assertIn('Trainer exited 1', (run/'error.txt').read_text())
            proc.wait(timeout=5)
            self.assertEqual(proc.returncode, -signal.SIGKILL)

    def test_probe_partial_receipt_does_not_suppress_timeout_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = asset_probe.subprocess.Popen
            def spawn(*args, **kwargs):
                (Path(kwargs['env']['AGL_MCP_OWNERS'])/'a-partial.json').write_text('{')
                return original(*args, **kwargs)
            output = Path(tmp)/'timeout.json'
            with patch.object(asset_probe.subprocess, 'Popen', side_effect=spawn):
                with self.assertRaises(ValueError): asset_probe.check(output, timeout=.001)
            result = json.loads(output.read_text())
            self.assertEqual(result['error_type'], 'TimeoutExpired')
            self.assertFalse(result['owned_process_cleanup_completed'])
            self.assertEqual(result['owned_process_cleanup']['anomalies'][0]['file'], 'a-partial.json')
            self.assertEqual((Path(result['ownership_receipts_directory'])/'a-partial.json').read_text(), '{')

    def test_probe_retains_failure_with_cleanup_exception_and_startup_failure(self):
        for startup in (False, True):
            with self.subTest(startup=startup), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)/'failure.json'
                # Real worker timeout; only the exceptional cleanup boundary is injected.
                with patch.object(asset_probe, 'cleanup', side_effect=OSError('synthetic cleanup failure')):
                    if startup:
                        with patch.object(asset_probe.subprocess, 'Popen', side_effect=OSError('synthetic startup failure')):
                            with self.assertRaises(ValueError): asset_probe.check(output)
                    else:
                        with self.assertRaises(ValueError): asset_probe.check(output, timeout=.001)
                result = json.loads(output.read_text())
                self.assertEqual(result['status'], 'failed')
                self.assertTrue(Path(result['ownership_receipts_directory']).is_dir())
                self.assertFalse(result['owned_process_cleanup_completed'])
                self.assertEqual(result['error_type'], 'OSError' if startup else 'TimeoutExpired')
                self.assertEqual(result['owned_process_cleanup']['anomalies'][0]['error_type'], 'OSError')


class BaselineGateTests(unittest.TestCase):
    def simulate(self, schema='reader-task-v2', mode='zero_integration'):
        p = plan(); p['schema'] = schema
        calls = {}; records = []
        def dispatch(job, stage):
            fid = 'fc-mocked-'+str(len(calls)); calls[fid] = (job, stage); return fid
        def result(fid):
            job, stage = calls[fid]
            # Mocked worker completion, not a generated trace relabeled as live.
            chain = {'evidence_class': 'live-model', 'terminal_reward': 0.0,
                     'task_id': p['task_identity']['name'], 'environment_sha256': c.digest(p['data_manifest'])}
            qualified = mode != 'zero_integration' or job is p['jobs'][1]
            coverage = [{'chains': [chain]}] if qualified else [{'chains': []}]
            report = dict(schema='reader-integration-v2', algorithm=job['algorithm'], stage=stage['name'],
                          qualified=qualified, episodes_with_valid_chain=int(qualified), trace_files=1, coverage=coverage)
            if mode == 'scripted': chain['evidence_class'] = 'scripted-inference'
            if mode == 'wrong_environment': chain['environment_sha256'] = 'f'*64
            if mode == 'missing': report = None
            if mode == 'inconsistent': report['episodes_with_valid_chain'] = 0
            return dict(status='completed', call_id=fid, output=stage['output'], completion={
                'plan_sha256': c.digest(p), 'stage': stage['name'], 'model_tool_integration': report,
                'diagnostic': 'retained zero-accuracy baseline'})
        index = supervisor.run(p, dispatch, result, Mock(), lambda _: None, lambda: False,
                               records.append, clock=lambda: 0, sleep=lambda _: None)
        return p, index, calls, records

    def test_failed_baseline_blocks_only_its_dependents_and_retains_proof(self):
        p, index, calls, records = self.simulate()
        failed, passed = [index['jobs'][j['job_id']] for j in p['jobs']]
        self.assertEqual(len(calls), 4)  # Failed algorithm baseline only; other algorithm all three stages.
        self.assertEqual(failed['status'], 'failed')
        self.assertEqual([r['status'] for r in failed['stages']], ['completed', 'skipped', 'skipped'])
        self.assertEqual(failed['stages'][0]['result']['completion']['diagnostic'], 'retained zero-accuracy baseline')
        self.assertEqual(failed['stages'][1]['reason'], 'baseline_integration_unqualified')
        self.assertEqual(passed['status'], 'completed')
        self.assertTrue(any(r['jobs'][p['jobs'][0]['job_id']].get('dependency_gate') for r in records))

    def test_valid_zero_reward_continues_both_algorithms(self):
        _, index, calls, _ = self.simulate(mode='zero_reward')
        self.assertEqual(index['status'], 'completed')
        self.assertEqual(len(calls), 6)
        self.assertTrue(all(j['dependency_gate']['admitted'] for j in index['jobs'].values()))

    def test_missing_inconsistent_scripted_and_wrong_environment_block(self):
        for mode in ('missing', 'inconsistent', 'scripted', 'wrong_environment'):
            with self.subTest(mode=mode):
                _, index, calls, _ = self.simulate(mode=mode)
                self.assertEqual(index['status'], 'failed')
                self.assertEqual(len(calls), 2)

    def test_archived_v1_and_historical_scheduling_unchanged(self):
        for schema in ('reader-reference-v1', 'historical'):
            _, index, calls, _ = self.simulate(schema=schema)
            self.assertEqual(index['status'], 'completed')
            self.assertEqual(len(calls), 6)
