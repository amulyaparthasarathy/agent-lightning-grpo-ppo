"""Offline contract, strict replay, and production wire format tests (reader-harness group)."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from experiment import recovery_prompt
from witness_sft import contract, data, dispatch, evaluation
from recovery_harness_support import ASSET, POLICY, INVENTORY, TEXT, run_episode


class WitnessContractTests(unittest.TestCase):
    def test_frozen_reproduction(self):
        payload, manifest = data.build(data.DIRECTORY)
        self.assertEqual(payload, (data.DIRECTORY / 'demonstrations.jsonl').read_bytes())
        self.assertEqual(manifest, json.loads((data.DIRECTORY / 'manifest.json').read_text()))
        self.assertEqual(manifest['balance'], {'ready': 16, 'not_ready': 16})
        rows = manifest['cases_in_training_order']
        self.assertEqual(len({r['case_id'] for r in rows}), 32)
        self.assertTrue(all(r['strict_reward'] == 1 for r in rows))
        self.assertTrue({'boundary', 'just_stale', 'multi_witness', 'integrity_pending'} <= {r['variant'] for r in rows})
        source = data.read_source(data.DIRECTORY)
        source['files']['dev_panel.parquet']['case_ids'][0] = rows[0]['case_id']
        with self.assertRaises(ValueError):
            data.partition(source, 'train')

    def test_rejects_contamination_and_invalid_witness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, mutate in [
                ('demonstrations.jsonl', lambda text: text.replace('You assess', 'Truth is ready. You assess', 1)),
                ('manifest.json', lambda text: text.replace('c-', 'fake-', 1)),
                ('source-manifest.json', lambda text: text.replace('c-', 'fake-', 1)),
            ]:
                for p in data.DIRECTORY.iterdir():
                    shutil.copyfile(p, root / p.name)
                p = root / name
                p.write_text(mutate(p.read_text()))
                with self.assertRaises(ValueError):
                    data.load(root)
        demo = next(x for x in data.load() if json.loads(x['target'])['decision'] == 'ready')
        question = demo['messages'][1]['content'].split('\n')[0]
        answer = json.loads(demo['target'])
        answer['supporting_run_ids'] = ['run-00000000']
        _, ledger = data.recovery_solver.solve(question, data.task.call)
        self.assertEqual(data.task.reward(data.task.dumps(answer), data.task._world()['by_question'][question], ledger), 0)

    def test_real_harness_scripted_on_training_row(self):
        demo = data.load()[0]
        row = data.selected(data.partition(data.read_source(data.DIRECTORY), 'train'))[0]
        with tempfile.TemporaryDirectory() as tmp:
            result = run_episode(tmp, [ASSET, POLICY, INVENTORY, TEXT(demo['target'])], row=row)
        self.assertIsNone(result['error'], result['message'])
        request = result['requests'][-1]
        self.assertEqual(recovery_prompt.render(request['messages'], request['tools']),
                         recovery_prompt.render(demo['messages'], recovery_prompt.tool_definitions()))
        rewards = [p['data']['value'] for p in result['posts']]
        with self.assertRaises(ValueError):
            evaluation.admit_live_trace(row, result['records'], rewards, 'fresh-base', '0'*64)
        self.assertEqual(evaluation.replay(row, result['records'], rewards)['strict'], 1)
        with self.assertRaises(ValueError):
            evaluation.replay(row, result['records'], [])

    def test_gate_requires_both_classes_and_assessability(self):
        rows = data.partition(data.read_source(data.DIRECTORY), 'development')
        rows = [{'case_id': r['case_id'], 'actual': json.loads(r['result'])['decision'],
                 'diagnostics': {'decision_correct': True}, 'strict': 1} for r in rows]
        for actual in data.task.DECISIONS:
            for r in [x for x in rows if x['actual'] == actual][:4]:
                r['strict'] = 0
        self.assertTrue(evaluation.summarize(rows)['gate_passed'])
        row = next(r for r in rows if r['actual'] == 'ready' and r['strict'])
        row['strict'] = 0
        self.assertFalse(evaluation.summarize(rows)['gate_passed'])
        row['strict'] = 1
        row['diagnostics']['decision_correct'] = None
        self.assertFalse(evaluation.summarize(rows)['gate_passed'])
        with self.assertRaises(ValueError):
            evaluation.summarize(rows[::-1])

    def test_no_paid_route_or_extra_stage(self):
        plan = contract.make_plan('0' * 40, {})
        with patch.dict('sys.modules', modal=None), self.assertRaises(PermissionError):
            dispatch.dispatch(plan, {}, {}, None)
        contract.check_stage(plan, 'fresh-base', [])
        for stage in ('training', 'grpo', 'ppo', 'final', 'terminal-reload', 'supervised-pass'):
            with self.assertRaises(ValueError):
                contract.check_stage(plan, stage, [])
        for k, value in [('seed', 2002), ('stages', ['training']), ('schema', 'reader-task-v2')]:
            with self.assertRaises(ValueError):
                contract.check_stage({**plan, k: value}, 'fresh-base', [])
        cmd = evaluation.serve_command(Path('model'), Path('base'))
        self.assertEqual(json.loads(cmd[-1]), {'temperature': 0, 'top_k': -1, 'top_p': 1})


if __name__ == '__main__':
    unittest.main()
