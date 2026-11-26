"""Offline regression tests, including public evidence joins without private fixtures."""
from collections import Counter, defaultdict
import csv
import json
import unittest

from scripts import diagnose_functional_pilot as diag


class PilotDiagnosisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = diag.PUBLIC / 'diagnosis'
        cls.paired = diag.read(cls.directory / 'development.json')
        cls.summary = diag.read(cls.directory / 'summary.json')
        cls.published = diag.read(diag.PUBLIC / 'P02.json')

    def test_public_commitments(self):
        self.assertEqual(diag.contract.file_hash(diag.__file__), self.summary['analysis_script_sha256'])
        for name, digest in self.summary['tables_sha256'].items():
            self.assertEqual(diag.contract.file_hash(self.directory / name), digest)
        for allocation, source in self.summary['sources'].items():
            self.assertEqual(diag.contract.file_hash(diag.PUBLIC / (allocation + '.json')),
                             source['public_summary_sha256'])

    def test_missing_duplicate_and_reordered_cases(self):
        def table(stage):
            return {r['case_id']: {**r[stage], **{k: r[k] for k in ('case_id', 'actual', 'family', 'variant')}}
                    for r in reversed(self.paired)}
        stages = [table(k) for k in ('p01_base', 'base', 'reload')]
        published = self.published['paired_development']['cases']
        facts = {r['case_id']: {} for r in self.paired}
        pairs = diag.pairs(*stages, published[1:] + published[:1], facts)
        self.assertEqual([r['case_id'] for r in pairs], sorted(facts))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            diag.pairs(*stages, published + published[:1], facts)
        stages[0].pop(published[0]['case_id'])
        with self.assertRaisesRegex(ValueError, 'Unpaired'):
            diag.pairs(*stages, published, facts)

    def test_raw_decision_and_terminal_failure(self):
        case = diag.task.rows('development')[0]
        raw, _ = diag.solver.solve(case['question'], diag.task.call)
        self.assertEqual(diag.parse_answer(raw, False)['decision'], json.loads(case['result'])['decision'])
        self.assertIsNone(diag.parse_answer(raw, True))
        self.assertIsNone(diag.parse_answer('{"decision":"not_ready"}', False))
        self.assertIsNone(diag.parse_answer(raw + ' trailing text', False))

    def test_decision_conditional_evidence_and_hidden_regressions(self):
        self.assertEqual(len(self.paired), 24)
        wrong = [r for r in self.paired if r['actual'] == 'ready']
        self.assertEqual(len(wrong), 12)
        for row in wrong:
            after = row['reload']
            self.assertEqual((after['decision'], after['selected'], after['reward']), ('not_ready', [], 0))
            self.assertFalse(after['decision_correct'])
            self.assertTrue(after['evidence_correct'])
            self.assertTrue(all(after['observed'].values()))
        hidden = [r for r in wrong if r['base']['decision_correct'] and not r['base']['reward']]
        self.assertEqual(len(hidden), 3)
        self.assertTrue(all(r['base']['selected_issues'] == ['outside_rpo'] for r in hidden))

    def test_four_real_rewards(self):
        self.assertEqual(diag.advantages([1] * 4), [0] * 4)
        self.assertEqual(diag.advantages([0] * 4), [0] * 4)
        for n, expected in [(1, 1.4999970197677612), (2, 0.8660238981246948), (3, 0.4999989867210388)]:
            adv = diag.advantages([1] * n + [0] * (4 - n))
            self.assertEqual(adv[:n], [expected] * n)
            self.assertAlmostEqual(sum(adv), 0, places=6)
        for invalid in ([1] * 3, [1] * 5, [0, 0, 0, 2]):
            with self.assertRaises(ValueError):
                diag.advantages(invalid)

    def test_declared_rollout_slots_reject_replacement_and_omission(self):
        case = diag.task.rows('train')[0]['case_id']
        family, variant = diag.task.family(case)
        stage = dict(case_ids=[case], rollouts_per_question=4, expected_episodes=4, route='train')
        rows = [dict(case_id=case, rollout_id=str(i), trace_id=str(i), route='train', reward=0,
                     reward_posted=True, family=family, variant=variant) for i in range(4)]
        diag.calibration.check_table(stage, rows)
        for broken in (rows[:-1], rows[:-1] + [dict(rows[0], trace_id='another-trace')]):
            with self.assertRaises(ValueError):
                diag.calibration.check_table(stage, broken)

    def test_collection_membership_and_class_counts(self):
        with (self.directory / 'training.csv').open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(diag.unique(rows, 'trace_sha256')), 512)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            diag.unique(rows[:-1] + rows[:1], 'trace_sha256')
        groups = defaultdict(list)
        for r in rows:
            groups[(int(r['collection']), r['case_id'])].append(r)
        self.assertEqual(len(groups), 128)
        self.assertEqual(len({cid for _, cid in groups}), 128)
        mixed = successes = 0
        for n, event in enumerate(diag.read(self.directory / 'collections.json'), 1):
            self.assertEqual((event['collection'], event['policy_step'], event['actor_call']), (n, n-1, n))
            batch = [g for (i, _), g in groups.items() if i == n]
            self.assertEqual(len(batch), 8)
            for g in batch:
                rewards = [int(r['reward']) for r in g]
                self.assertEqual([float(r['advantage']) for r in g], diag.advantages(rewards))
                mixed += len(set(rewards)) == 2
                successes += sum(rewards)
            for label, counts in event['by_class'].items():
                samples = [r for g in batch for r in g if r['actual'] == label]
                decisions = Counter(r['decision'] or 'unassessable' for r in samples)
                self.assertEqual(counts['trajectories'], len(samples))
                self.assertEqual(counts['questions'] * 4, len(samples))
                for key in (*diag.CLASSES, 'unassessable'):
                    self.assertEqual(counts[key], decisions[key])
                for key in ('correct', 'semantic', 'evidence', 'format', 'action'):
                    self.assertEqual(counts[key], sum(r['failure'] == key for r in samples))
        self.assertEqual((mixed, successes), (76, 206))
        self.assertEqual(Counter(r['json_syntax'] for r in rows if r['failure'] == 'format'),
                         {'False': 28, 'True': 7})
        terminal = [r for r in rows if r['outcome'] == 'terminal_policy_failure']
        self.assertEqual(len(terminal), 6)
        self.assertTrue(all(r['decision'] == r['format_compliant'] == r['decision_correct'] == '' for r in terminal))
        self.assertEqual(sum(r['time_correct'] == 'False' and r['decision_correct'] == '' for r in rows), 2)
