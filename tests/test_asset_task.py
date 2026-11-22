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
    def test_episode_revision_preserves_registry_and_dataset_identity(self):
        self.assertEqual(task.digest(task.registry()), '241b5672d43b7dff7db7d203f2805d91e0a541e7a20e18d4edb608419e670e44')
        self.assertEqual(task.digest(task.environment()), 'd683b4ca993f4d014de260fe8ba04afa901b1488be036a633c37b4fec7ef4089')
        self.assertEqual(task.identity(source())['episode_policy']['version'], 'asset-episode-v2')
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
    def test_two_calls_reflection_context_and_strict_final(self):
        from experiment.asset_agent import FINAL_INSTRUCTION
        for final, reward in ((None, 1), ('The asset is operational.', 0)):
            with self.subTest(final=final), tempfile.TemporaryDirectory() as tmp:
                result = run_case(tmp, mode='two-tools', final=final, request=self.request())
                self.assertIsNone(result['error'])
                self.assertEqual(result['posts'], [{'event_type': 'reward', 'data': {'value': reward}}])
                self.assertEqual([len(x) for x in result['inputs']], [2, 4, 7])
                self.assertEqual(result['inputs'][2][:4], result['inputs'][1])
                self.assertEqual(result['inputs'][2][-1]['content'], FINAL_INSTRUCTION)
                self.assertEqual(result['options'][2]['tool_choice'], 'none')
                self.assertIsNone(result['options'][2]['json_output'])
                self.assertFalse(result['options'][2].get('tools'))
                self.assertEqual(len(reader_integration.chains(result['records'], self.request(), evidence_class='scripted-inference')), 2)

    def test_recovery_and_explicit_terminal_failure(self):
        for mode in ('recover', 'malformed-recover', 'unknown-recover', 'policy-limit'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                req = self.request(); result = run_case(tmp, mode=mode, request=req)
                self.assertIsNone(result['error'])
                terminal = mode == 'policy-limit'
                self.assertEqual(result['posts'], [{'event_type': 'reward', 'data': {'value': 0 if terminal else 1}}])
                episode = next(r for r in result['records'] if r['kind'] == 'episode')['outputs']
                self.assertEqual(episode['outcome'], 'terminal_policy_failure' if terminal else 'complete_recovered')
                self.assertEqual(episode['policy_errors'], 2 if terminal else 1)
                self.assertEqual(reader_integration.episode_outcome(result['records']), episode['outcome'])
                actions = [r for r in result['records'] if r['kind'] == 'action_outcome']
                self.assertEqual(len(actions), 2)
                self.assertTrue(actions[0]['inputs']['observation']['is_error'])
                self.assertEqual(len(result['inputs']), 2 if terminal else 3)
                if terminal:
                    self.assertIsNone(episode['raw_answer'])
                    self.assertEqual(len([r for r in result['records'] if r['kind'] == 'termination']), 1)
                self.assertEqual(len(reader_integration.chains(result['records'], req, evidence_class='scripted-inference')), 0 if terminal else 1)

    def test_incomplete_at_each_generation_boundary(self):
        for mode in ('truncated-action', 'truncated-reflection', 'missing-second', 'missing-reflection',
                     'transport-failure-second', 'transport-failure-reflection', 'cancelled-second',
                     'cancelled-reflection', 'infrastructure-after-tool'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                result = run_case(tmp, mode=mode)
                self.assertIsNotNone(result['error'])
                self.assertEqual(result['posts'], [])
                self.assertTrue(result['cleanup']['complete'])

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
            for messages in result['inputs']:
                systems = [m['content'] for m in messages if m['type'] == 'SystemMessage']
                self.assertEqual(len(systems), 1)
                self.assertIn('Call lookup_asset with exactly one argument: asset_id.', systems[0])
                self.assertIn('Do not supply service_state or backup_health as tool arguments', systems[0])
                self.assertIn('final answer in the format requested by the user', systems[0])
                self.assertNotIn('TERMINATE', systems[0])
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
                child_stat = Path(f"/proc/{receipts[0]['pid']}/stat")
                deadline = time.monotonic()+2
                while child_stat.exists() and child_stat.read_text().rsplit(')', 1)[1].split()[0] != 'Z' and time.monotonic()<deadline:
                    time.sleep(.01)
                self.assertTrue(not child_stat.exists() or child_stat.read_text().rsplit(')', 1)[1].split()[0] == 'Z')
                # No live owned server remains; the OS may briefly retain a reaped-parent zombie.
                self.assertNotEqual(proc.returncode, 0)
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait(timeout=5)
                owned_processes.cleanup(tmp)

    def test_real_probe_startup_failure_and_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): asset_probe.check(Path(tmp)/'timeout.json', timeout=.001)
            # Actual MCP client starts a server that exits during initialization.
            import asyncio
            from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
            async def failed_server():
                async with McpWorkbench(StdioServerParams(command='/bin/false', read_timeout_seconds=.5)) as workbench:
                    await workbench.list_tools()
            with self.assertRaises(Exception):
                asyncio.run(failed_server())

    def test_cleanup_separately_created_owned_process_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = str(Path(__file__).resolve().parents[1])
            script = 'from experiment.owned_processes import register; import time; register('+repr(tmp)+'); time.sleep(60)'
            proc = subprocess.Popen([sys.executable, '-c', script], cwd=root, start_new_session=True)
            try:
                deadline = time.monotonic()+10
                while not list(Path(tmp).glob('*.json')) and proc.poll() is None and time.monotonic()<deadline:
                    time.sleep(.02)
                self.assertEqual(owned_processes.cleanup(tmp)['signalled'], 1)
                proc.wait(timeout=5)
                self.assertLess(proc.returncode, 0)
            finally:
                if proc.poll() is None: proc.kill(); proc.wait()
