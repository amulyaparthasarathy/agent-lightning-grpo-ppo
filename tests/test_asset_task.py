"""Deterministic dataset, independent parser/reward and real MCP/harness checks."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from experiment import asset_task as task, asset_probe, owned_processes, reader_contract as c, reader_integration
from asset_harness_support import run_case
from test_reader_training import source, profile


class DatasetTests(unittest.TestCase):
    def test_generation_verification_order_and_identity_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'a', Path(tmp)/'b'
            self.assertEqual(task.generate(a), task.generate(b))
            self.assertEqual(task.verify(a), task.verify(b))
            panels = task.cases()
            sets = [set(task.target(r['question']) for r in panels[n]) for n in
                    ('train_main.parquet', 'eval_main.parquet', 'eval_smoke.parquet')]
            self.assertEqual([len(x) for x in sets], [256, 64, 8])
            self.assertEqual(len(set.union(*sets)), 328)
            for name in task.FILES:
                self.assertEqual((a/name).read_bytes(), (b/name).read_bytes())
            p = a/'registry.json'; p.write_text(p.read_text()+' ')
            with self.assertRaisesRegex(ValueError, 'registry bytes'): task.verify(a)

    def test_rehashed_duplicate_conflict_label_order_and_split_corruption(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        for mutation in ('duplicate', 'label', 'order', 'leakage'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)/'data'; m = task.generate(root)
                rows = task.cases()['eval_main.parquet']
                if mutation == 'duplicate': rows[1] = rows[0]
                if mutation == 'label': rows[0]['result'] = '{}'
                if mutation == 'order': rows.reverse()
                if mutation == 'leakage': rows[0] = task.cases()['train_main.parquet'][0]
                pq.write_table(pa.Table.from_pylist(rows), root/'eval_main.parquet')
                # Even rehashing bytes and retaining claimed IDs cannot hide content drift.
                m['files']['eval_main.parquet']['sha256'] = task.hash_file(root/'eval_main.parquet')
                (root/'data_manifest.json').write_text(task.dumps(m))
                with self.assertRaises(ValueError): task.verify(root)

    def test_asset_preparation_never_acquires_calculator_archive(self):
        from experiment import reader_assets as assets
        from test_reader_training import AssetsTests
        import shutil
        p = profile(); p['task'] = task.TASK
        plan = c.make_plan(p, source())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); seed = root/'model-fixture'
            lock = AssetsTests().model_fixture(seed)
            calls = []
            def acquire(command, *_):
                calls.append(command)
                if not str(command[1]).endswith('prepare_model.py'):
                    raise AssertionError('Unexpected calculator acquisition')
                shutil.copytree(seed, command[-1])
            with patch.object(assets, 'model_lock', return_value=lock):
                result = assets.prepare(plan, root, acquire, root, {})
                assets.validate_manifest(result, task.TASK)
                self.assertEqual(len(calls), 1)
                self.assertEqual(result['data'], task.verify(root/p['paths']['data']))
                self.assertEqual(result['historical_match'], {'claim': 'synthetic asset task; no calculator-data equivalence'})
                with self.assertRaises(ValueError): assets.validate_manifest(result, 'calculator-v1')
                # Fake weights are confined to the patched test boundary, never admitted by production.
            with self.assertRaises(ValueError): assets.verify_model(root/p['paths']['model'])

    def test_parser_and_independent_reward(self):
        truth = task.lookup({'asset_id': 'asset-047'}); valid = task.dumps(truth)
        self.assertEqual(task.reward(valid, truth), 1)
        for text in ('{}', '[]', 'null', valid+valid, '```json\n'+valid+'\n```', 'prefix '+valid,
                     valid[:-1]+',"asset_id":"asset-047"}', valid[:-1]+',"extra":0}',
                     valid.replace('asset-047', 'asset-048'), valid.replace('"asset-047"', '47'),
                     valid.replace(truth['service_state'], 'unknown'), valid[:-1]+',"x":NaN}'):
            self.assertEqual(task.reward(text, truth), 0, text)
        row = task.cases()['eval_main.parquet'][0]
        with self.assertRaises(ValueError): task.expected(row['question'], valid)


class RealHarnessTests(unittest.TestCase):
    def request(self):
        p = profile(); p['task'] = task.TASK
        plan = c.make_plan(p, source()); j = plan['jobs'][0]
        return c.stage_request(plan, j, j['stages'][0])

    def test_real_mcp_schema_invalid_calls_and_calculator_regression(self):
        from experiment.reader_calculator import check
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(asset_probe.check(Path(tmp)/'asset.json')['status'], 'passed')
            self.assertEqual(check(Path(tmp)/'calc.json')['status'], 'passed')
            with self.assertRaises(FileExistsError): asset_probe.check(Path(tmp)/'asset.json')

    def test_actual_harness_observation_reward_child_environment_and_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = self.request(); result = run_case(tmp, request=request)
            self.assertIsNone(result['error'])
            self.assertEqual(result['posts'], [{'event_type': 'reward', 'data': {'value': 1.0}}])
            self.assertEqual(len(result['inputs']), 2)
            self.assertEqual(len(result['children']), 1)
            for child in result['children']:
                self.assertEqual(child['environment_keys'], ['PYTHONDONTWRITEBYTECODE', 'PYTHONUNBUFFERED'])
                self.assertFalse(Path(f"/proc/{child['pid']}").exists())
            first = json.dumps(result['inputs'][0]); self.assertNotIn('test-secret-canary', first)
            self.assertNotIn(task.cases()['eval_main.parquet'][0]['result'], first)
            users = [m['content'] for m in result['inputs'][0] if m['type'] == 'UserMessage']
            self.assertEqual(users, [task.cases()['eval_main.parquet'][0]['question']+'\n'+task.PROMPT])
            self.assertEqual(len(reader_integration.chains(result['records'], request, evidence_class='scripted-inference')), 1)
            with self.assertRaises(ValueError): reader_integration.chains(result['records'], request)
            for field in ('task_id', 'environment_sha256', 'configuration_sha256', 'source_sha256'):
                bad = copy.deepcopy(result['records']); bad[0][field] = 'different'
                with self.assertRaises(ValueError): reader_integration.chains(bad, request, evidence_class='scripted-inference')

    def test_completed_wrong_malformed_and_wrong_target_receive_zero_despite_tool_success(self):
        for final in ('not JSON', '{}', task.dumps(task.lookup({'asset_id': 'asset-047'}))):
            with self.subTest(final=final), tempfile.TemporaryDirectory() as tmp:
                result = run_case(tmp, final=final)
                self.assertIsNone(result['error']); self.assertEqual(result['posts'][0]['data']['value'], 0)
                self.assertTrue(any(r['kind'] == 'tool' and r['execution_status'] == 'ok' for r in result['records']))

    def test_incomplete_no_reward_and_failed_delivery_is_retained(self):
        for mode in ('transport-failure', 'timeout', 'cancelled', 'truncated', 'reward-failure'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                result = run_case(tmp, mode=mode)
                self.assertIsNotNone(result['error'])
                episode = next(r for r in result['records'] if r['kind'] == 'episode')
                self.assertEqual(episode['execution_status'], 'exception')
                self.assertFalse((episode['outputs'] or {}).get('reward_posted'))
                self.assertEqual(len(result['posts']), 1 if mode == 'reward-failure' else 0)

    def test_corrupted_observation_cannot_override_independent_verifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_case(tmp, mode='corrupt-observation')
            self.assertIsNone(result['error'])
            self.assertEqual(result['posts'][0]['data']['value'], 0)

    def test_scripted_join_matrix_all_algorithms_and_stages(self):
        p = profile(); p['task'] = task.TASK
        plan = c.make_plan(p, source())
        for job in plan['jobs']:
            for stage in job['stages']:
                req = c.stage_request(plan, job, stage)
                with self.subTest(algorithm=job['algorithm'], stage=stage['name']), tempfile.TemporaryDirectory() as tmp:
                    result = run_case(tmp, request=req)
                    self.assertEqual(len(reader_integration.chains(result['records'], req, evidence_class='scripted-inference')), 1)
                    self.assertFalse(reader_integration.report(Path(tmp), req)['qualified'])

    def test_actual_mcp_hanging_server_separate_session_cleanup(self):
        import signal
        with tempfile.TemporaryDirectory() as tmp:
            script = """import asyncio, sys
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
async def main():
    async with McpWorkbench(StdioServerParams(command=sys.executable, args=[sys.argv[1], '--hanging-server', sys.argv[2]], read_timeout_seconds=.1)) as w:
        await w.list_tools()
asyncio.run(main())
"""
            proc = subprocess.Popen([sys.executable, '-c', script, str(Path(__file__).with_name('asset_harness_support.py').resolve()), tmp],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            try:
                deadline = time.monotonic()+5
                while not list(Path(tmp).glob('*.json')) and proc.poll() is None and time.monotonic()<deadline:
                    time.sleep(.02)
                receipts = [json.loads(p.read_text()) for p in Path(tmp).glob('*.json')]
                self.assertEqual(len(receipts), 1)
                self.assertNotEqual(receipts[0]['pid'], proc.pid)
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=5)
                owned_processes.cleanup(tmp)
                # Repeated cleanup is idempotent and PID identity is checked.
                self.assertNotEqual(proc.returncode, 0)
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=5)
                owned_processes.cleanup(tmp)

    def test_real_probe_startup_failure_and_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): asset_probe.check(Path(tmp)/'timeout.json', timeout=.001)
            # Real interpreter failure, not a mocked process object.
            with patch.object(asset_probe.sys, 'executable', '/bin/false'):
                with self.assertRaises(ValueError): asset_probe.check(Path(tmp)/'startup.json')

    def test_cleanup_separately_created_owned_process_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = str(Path(__file__).resolve().parents[1])
            script = 'from experiment.owned_processes import register; import time; register('+repr(tmp)+'); time.sleep(60)'
            proc = subprocess.Popen([sys.executable, '-c', script], cwd=root, start_new_session=True)
            try:
                deadline = time.monotonic()+10
                while not list(Path(tmp).glob('*.json')) and proc.poll() is None and time.monotonic()<deadline:
                    time.sleep(.02)
                self.assertEqual(owned_processes.cleanup(tmp), 1)
                proc.wait(timeout=5)
                self.assertLess(proc.returncode, 0)
            finally:
                if proc.poll() is None: proc.kill(); proc.wait()
