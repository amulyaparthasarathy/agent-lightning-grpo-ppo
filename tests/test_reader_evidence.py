"""Explicitly synthetic traces/losses and mocked calculator submissions; no model execution."""
import asyncio
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from experiment import reader_calculator as calculator, reader_evidence as evidence, reader_integration as integration
from experiment import reader_contract as c
from test_reader_training import plan


class CalculatorTests(unittest.TestCase):
    def workbench(self):
        schema = {'type': 'object', 'properties': {'expression': {'title': 'Expression', 'type': 'string'}},
                  'required': ['expression'], 'additionalProperties': False}
        workbench = Mock()
        workbench.list_tools = AsyncMock(return_value=[{'name': 'calculate', 'parameters': schema}])
        workbench.call_tool = AsyncMock(side_effect=[Mock(model_dump=Mock(return_value={
            'name': 'calculate', 'result': [{'type': 'TextResultContent', 'content': text}], 'is_error': error}))
            for text, error in [('20', False), ('Error executing tool calculate: Unknown identifier: unknown_reader_symbol', True)]])
        return workbench

    def test_mocked_valid_and_invalid_requests_and_exact_schema(self):
        workbench = self.workbench()
        result = asyncio.run(calculator.probe(workbench))
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(workbench.call_tool.await_args_list[0].args, ('calculate', {'expression': '(2 + 3) * 4'}))
        for edit in (lambda w: w.list_tools.return_value.clear(),
                     lambda w: w.list_tools.return_value[0]['parameters'].update(required=['guessed_argument'])):
            workbench = self.workbench(); edit(workbench)
            self.assertEqual(asyncio.run(calculator.probe(workbench))['status'], 'failed')
            workbench.call_tool.assert_not_awaited()

    def test_wrong_result_or_error_behavior_fails(self):
        for text, error in [('21', False), ('20', True), ('unknown error', True)]:
            workbench = self.workbench()
            workbench.call_tool.side_effect = None
            workbench.call_tool.return_value = Mock(model_dump=Mock(return_value={
                'name': 'calculate', 'result': [{'type': 'TextResultContent', 'content': text}], 'is_error': error}))
            self.assertEqual(asyncio.run(calculator.probe(workbench))['status'], 'failed')

    def test_timeout_has_failure_receipt_and_owned_group_cleanup(self):
        process = Mock(pid=12345)
        process.communicate.side_effect = [subprocess.TimeoutExpired('synthetic', 90), ('', '')]
        with tempfile.TemporaryDirectory() as tmp, patch.object(calculator.subprocess, 'Popen', return_value=process), patch.object(calculator.os, 'killpg') as kill:
            output = Path(tmp)/'receipt.json'
            with self.assertRaisesRegex(ValueError, 'probe failed'): calculator.check(output)
            self.assertEqual(json.loads(output.read_text())['error'], 'calculator timeout')
            kill.assert_called_once_with(12345, calculator.signal.SIGKILL)
            with self.assertRaises(FileExistsError): calculator.check(output)

    def test_timeout_cleans_separate_mcp_session(self):
        process = Mock(pid=12345)
        process.communicate.side_effect = [subprocess.TimeoutExpired('synthetic', 90), ('', '')]
        def spawn(command, **kwargs):
            Path(command[-1]).write_text('23456')  # Synthetic PID file from owned server wrapper.
            return process
        with tempfile.TemporaryDirectory() as tmp, patch.object(calculator.subprocess, 'Popen', side_effect=spawn), patch.object(calculator.os, 'killpg') as kill:
            with self.assertRaisesRegex(ValueError, 'probe failed'): calculator.check(Path(tmp)/'receipt.json')
            self.assertEqual([call.args[0] for call in kill.call_args_list], [12345, 23456])


class LossTests(unittest.TestCase):
    def rows(self):
        return [{'step': step, 'metrics': {metric: 0.0}} for step in (1, 2)
                for metric in ('actor/pg_loss', 'critic/vf_loss')]

    def test_split_role_records_zero_and_matching_duplicates(self):
        rows = self.rows()
        result = evidence.loss_evidence(rows + rows[:1], ('actor', 'critic'))
        self.assertEqual(len(result), 4)
        self.assertEqual([r['value'] for r in result], [0.0]*4)

    def test_missing_update_role_invalid_and_conflicting_duplicates(self):
        for bad in (self.rows()[:-1], self.rows()[2:], [r for r in self.rows() if 'critic/vf_loss' not in r['metrics']]):
            with self.assertRaisesRegex(ValueError, 'Missing'): evidence.loss_evidence(bad, ('actor', 'critic'))
        for value in (None, True, '0', float('nan'), float('inf'), -float('inf')):
            rows = self.rows(); rows[0]['metrics']['actor/pg_loss'] = value
            with self.assertRaisesRegex(ValueError, 'Invalid'): evidence.loss_evidence(rows, ('actor', 'critic'))
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            evidence.loss_evidence(self.rows()+[{'step': 1, 'metrics': {'actor/pg_loss': 1.0}}], ('actor', 'critic'))
        for step in (0, 3, True, 1.0, None):
            rows = self.rows(); rows[0]['step'] = step
            with self.assertRaisesRegex(ValueError, 'outside'): evidence.loss_evidence(rows, ('actor', 'critic'))


class IntegrationTests(unittest.TestCase):
    def fixture(self, algorithm=0, stage=0):
        # Simulated real-record fields for parser tests, never a qualification receipt.
        p = plan(); job = p['jobs'][algorithm]; stage = job['stages'][stage]
        request = c.stage_request(p, job, stage)
        common = {'schema_version': 1, 'trace_id': 'synthetic-trace', 'experiment_id': job['job_id'],
            'wandb_run_id': stage['wandb']['id'], 'case_id': 'synthetic-case', 'phase': stage['name'],
            'policy_version': 'matched-rollout-mean-v2', 'model_profile': 'llama3b', 'eval_split': 'final',
            'rollout_id': 'synthetic-rollout', 'attempt_id': 'synthetic-attempt',
            'route_mode': 'train' if stage['name'] == 'training' else 'val', 'synthetic_fixture': True, 'evidence_class': 'simulated-trace',
            'task_id': 'calculator-v1', 'task_sha256': c.digest(p['task_identity']),
            'environment_sha256': c.digest(p['data_manifest']), 'source_sha256': c.digest(p['source']),
            'configuration_sha256': p['configuration_sha256'],
            'execution_status': 'ok', 'exception': None, 'parent_span_id': 'episode'}
        def record(kind, span, start, end, inputs, outputs):
            return {**common, 'kind': kind, 'name': 'calculate' if kind == 'tool' else kind, 'span_id': span,
                'started_at': f'2026-01-01T00:00:0{start}+00:00', 'ended_at': f'2026-01-01T00:00:0{end}+00:00',
                'inputs': inputs, 'outputs': outputs}
        records = [record('llm', 'request', 1, 2, {'messages': []}, {'content': [
            {'id': 'call-1', 'name': 'calculate', 'arguments': '{"expression":"2+3"}'}]}),
            record('tool', 'tool', 3, 4, {'tool_call_id': 'call-1', 'arguments': {'expression': '2+3'}},
                {'name': 'calculate', 'is_error': False, 'result': [{'type': 'TextResultContent', 'content': '5'}]}),
            record('llm', 'continuation', 5, 6, {'messages': [{'type': 'FunctionExecutionResultMessage',
                'content': [{'call_id': 'call-1', 'name': 'calculate', 'content': '5', 'is_error': False}]}]}, {'content': 'answer', 'finish_reason': 'stop'}),
            record('episode', 'episode', 0, 7, {}, {'reward_posted': True, 'reward': 0.0, 'raw_answer': 'answer'})]
        records[-1]['parent_span_id'] = None
        return request, records

    def test_valid_chain_independent_of_correctness_for_both_algorithms_all_stages(self):
        for algorithm in (0, 1):
            for stage in (0, 1, 2):
                request, records = self.fixture(algorithm, stage)
                result = integration.chains(records, request, evidence_class="simulated-trace")
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0]['terminal_reward'], 0.0)
                # JSONL is written on span completion, not a required input ordering.
                self.assertEqual(integration.chains(list(reversed(records)), request, evidence_class="simulated-trace"), result)

    def test_missing_continuation_observation_tool_success_or_reward_is_unqualified(self):
        edits = [lambda r: r.pop(2), lambda r: r[2]['inputs'].update(messages=[]),
            lambda r: r[2]['inputs']['messages'][0]['content'][0].update(content='wrong'),
            lambda r: r[2]['inputs']['messages'][0]['content'][0].update(call_id='different'),
            lambda r: r[1]['outputs'].update(is_error=True),
            lambda r: r[3]['outputs'].update(reward_posted=False),
            lambda r: r[1]['inputs'].update(arguments={'expression': 'different'}),
            lambda r: r[2].update(started_at=r[0]['started_at']),
            lambda r: r.append(copy.deepcopy(r[1]) | {'span_id': 'duplicate-call'})]
        for edit in edits:
            request, records = self.fixture(); edit(records)
            self.assertEqual(integration.chains(records, request, evidence_class="simulated-trace"), [])

    def test_wrong_stage_synthetic_and_duplicate_trace_rejected(self):
        for edit in (lambda r: r[1].update(attempt_id='other'), lambda r: r[0].update(synthetic_fixture=False),
                     lambda r: r[1].update(span_id='request'), lambda r: r[1].update(parent_span_id='other')):
            request, records = self.fixture(); edit(records)
            with self.assertRaises(ValueError): integration.chains(records, request, evidence_class="simulated-trace")
        request, records = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'traces').mkdir()
            path = root/'traces/trajectory-synthetic-trace.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in records))
            self.assertFalse(integration.report(root, request)['qualified'])  # simulated traces never qualify live work
            records[2]['inputs']['messages'] = []
            path.write_text(''.join(json.dumps(r)+'\n' for r in records))
            result = integration.report(root, request)
            self.assertFalse(result['qualified'])
            self.assertEqual(result['episodes_with_valid_chain'], 0)
            self.assertTrue(result['gaps'])
