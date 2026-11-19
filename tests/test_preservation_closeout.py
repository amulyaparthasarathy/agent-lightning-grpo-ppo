"""Budget reconciliation regressions with synthetic terminal metadata only."""
import copy
import unittest

from test_preservation_workers import plan
from scripts.backup_qualification import digest
from scripts.preservation_closeout import reconcile_cpu
from scripts.preservation_workflow import new_ledger, ORDER


class CloseoutTests(unittest.TestCase):
    def setUp(self):
        p = plan()
        for phase in p['phases']:
            phase['startup_reserve_seconds'] = 600
            phase['reservation']['gpu_seconds' if phase['kind'] == 'gpu' else 'cpu_seconds'] = 610
        self.ledger = new_ledger(p)
        self.ledger['accounting'] = copy.deepcopy(self.ledger['accounting'])
        self.ledger['accounting']['cpu_seconds'] = 34000
        for name in ORDER[:-2]:
            self.ledger['phases'][name] = {'status': 'passed', 'function_call_id': 'call-' + name,
                                          'usage': {'cpu_seconds': 620}, 'result': {'revision': 'b' * 40}}
        self.observation = {'function_call_id': 'call-compare', 'total_inputs': 1,
                            'succeeded_inputs': {'count': 1, 'latest': [{
                                'started_at': 100, 'finished_at': 125.2,
                                'execution_seconds': 25.2, 'startup_seconds': 3.1}]}}
        for key in ('failed_inputs', 'pending_inputs', 'timeout_inputs', 'cancelled_inputs'):
            self.observation[key] = {'count': 0, 'latest': []}

    def run_reconcile(self, evidence=None):
        return reconcile_cpu(self.ledger, {'compare': self.observation} if evidence is None else evidence,
                             revision='a' * 40, reviewed_at='fixture-time')

    def test_retains_measured_overhead_and_margin_without_mutating_history(self):
        before = copy.deepcopy(self.ledger)
        revised, audit = self.run_reconcile()
        self.assertEqual(self.ledger, before)
        self.assertEqual(revised['plan'], before['plan'])
        self.assertEqual(revised['phases'], before['phases'])
        self.assertEqual(audit['adjustment_cpu_seconds'], -471)
        self.assertEqual(revised['accounting']['cpu_seconds'], 33529)
        self.assertEqual(audit['prior_ledger_sha256'], digest(before))
        for key in before['accounting']:
            if key != 'cpu_seconds':
                self.assertEqual(revised['accounting'][key], before['accounting'][key])
        self.ledger = revised
        with self.assertRaisesRegex(ValueError, 'already reconciled'):
            self.run_reconcile()

    def test_does_not_release_failed_retried_or_gpu_consumption(self):
        for mutation in ({'status': 'failed'}, {'attempt': 1}, {'recoveries': [{}]}):
            with self.subTest(mutation=mutation):
                old = copy.deepcopy(self.ledger['phases']['compare'])
                self.ledger['phases']['compare'].update(mutation)
                with self.assertRaises(ValueError): self.run_reconcile()
                self.ledger['phases']['compare'] = old
        self.observation['function_call_id'] = 'call-replication_train'
        with self.assertRaisesRegex(ValueError, 'CPU phases'):
            self.run_reconcile({'replication_train': self.observation})

    def test_rejects_wrong_call_unknown_nonfinite_or_ambiguous_measurement(self):
        for mutation in ('wrong_call', 'unknown', 'nan', 'ambiguous', 'bad_duration'):
            with self.subTest(mutation=mutation):
                obs = copy.deepcopy(self.observation)
                row = obs['succeeded_inputs']['latest'][0]
                if mutation == 'wrong_call': obs['function_call_id'] = 'other'
                if mutation == 'unknown': row['startup_seconds'] = None
                if mutation == 'nan': row['execution_seconds'] = float('nan')
                if mutation == 'ambiguous': obs['pending_inputs']['count'] = 1
                if mutation == 'bad_duration': row['finished_at'] = 200
                with self.assertRaises(ValueError): self.run_reconcile({'compare': obs})

    def test_both_reservations_must_fit_all_limits(self):
        # After release, the first 610-second reservation fits; the second fails.
        self.ledger['accounting']['cpu_seconds'] = 35500
        with self.assertRaisesRegex(ValueError, 'cpu_seconds'): self.run_reconcile()
        self.ledger['accounting']['cpu_seconds'] = 34000
        self.ledger['accounting']['cost_usd'] = 101
        with self.assertRaisesRegex(ValueError, 'invalid resource accounting'): self.run_reconcile()

    def test_rejects_missing_negative_nonfinite_and_boolean_accounting(self):
        for value in (-1, float('inf'), float('nan'), None, True):
            with self.subTest(value=value):
                self.ledger['accounting']['cost_usd'] = value
                with self.assertRaisesRegex(ValueError, 'invalid resource accounting'):
                    self.run_reconcile()
        del self.ledger['accounting']['cost_usd']
        with self.assertRaisesRegex(ValueError, 'invalid resource accounting'):
            self.run_reconcile()

    def test_rejects_invalid_revision_empty_evidence_and_completed_workflow(self):
        for revision in ('z' * 40, 'a' * 39, None):
            with self.subTest(revision=revision), self.assertRaisesRegex(ValueError, 'immutable ledger'):
                reconcile_cpu(self.ledger, {'compare': self.observation},
                              revision=revision, reviewed_at='fixture-time')
        with self.assertRaisesRegex(ValueError, 'evidence missing'): self.run_reconcile({})
        self.ledger['status'] = 'complete'
        with self.assertRaisesRegex(ValueError, 'not ready'): self.run_reconcile()

    def test_never_releases_recorded_worker_elapsed_when_modal_duration_is_shorter(self):
        self.ledger['phases']['compare']['usage']['cpu_seconds'] = 700
        _, audit = self.run_reconcile()
        self.assertEqual(audit['rows'][0]['reconciled_seconds'], 220)
        before = copy.deepcopy(self.ledger)
        self.observation['succeeded_inputs']['latest'][0]['startup_seconds'] = 700
        with self.assertRaisesRegex(ValueError, 'no releasable allowance'): self.run_reconcile()
        self.assertEqual(self.ledger, before)

    def test_refuses_claimed_remaining_phase_or_changed_plan(self):
        self.ledger['phases']['replication_publish'] = {'status': 'launching'}
        with self.assertRaisesRegex(ValueError, 'already claimed'): self.run_reconcile()
        del self.ledger['phases']['replication_publish']
        self.ledger['plan']['execution_commit'] = 'f' * 40
        with self.assertRaisesRegex(ValueError, 'plan identity'): self.run_reconcile()
