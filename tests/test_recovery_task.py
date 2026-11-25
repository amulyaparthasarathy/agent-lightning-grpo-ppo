"""recovery-readiness-v1: deterministic data, split isolation, verifier and oracle checks."""
from collections import Counter
import copy
import itertools
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch
from experiment import recovery_task as task, recovery_solver as solver

ROOT = Path(__file__).resolve().parents[1]
READY, NOT_READY = task.DECISIONS


class OpenRecorder:
    """Record every file the interpreter opens while active (audit hooks cannot be removed)."""
    active = None

    def __enter__(self):
        OpenRecorder.active = []
        return OpenRecorder.active

    def __exit__(self, *_):
        OpenRecorder.active = None


def _record_open(event, args):
    if event == 'open' and OpenRecorder.active is not None:
        OpenRecorder.active.append(str(args[0]))


sys.addaudithook(_record_open)


def trusted_cases(split=None):
    world = task._world()
    return [c for s, rows in world['cases'].items() if split in (None, s) for c in rows]


def solve(case):
    answer, ledger = solver.solve(case['question'], task.call)
    return json.loads(answer), ledger


def first(family, variant=None, split='development'):
    return next(c for c in trusted_cases(split) if c['family'] == family and variant in (None, c['variant']))


class DataTests(unittest.TestCase):
    def test_frozen_identities_counts_and_balance(self):
        self.assertEqual({s: len(r) for s, r in task.rows().items()}, {'train': 128, 'development': 24, 'final': 64})
        for split, rows in task.rows().items():
            decisions = Counter(json.loads(r['result'])['decision'] for r in rows)
            self.assertEqual(decisions[READY], decisions[NOT_READY], split)
            self.assertEqual({c['family'] for c in trusted_cases(split)}, set(task.FAMILIES))
        panels = task.panels()
        # Frozen before any live sampling. A change here is a new dataset identity.
        self.assertEqual(task.digest(task.environment()), '0c384ed90027f7fa0fb0aeb097220e6e83876dfdd85094353f03c8dd035b1ca3')
        self.assertEqual(panels['development']['panel_id'], 'rr1-dev-ac7fd952343b0c7d')
        self.assertEqual([p['packet_id'] for p in panels['development']['packets']],
                         ['rr1-dev-p1-98576e24a30d5457', 'rr1-dev-p2-292a56c798a2b41f', 'rr1-dev-p3-ee1d7ab06b3bf9ec'])
        self.assertEqual(panels['final']['panel_id'], 'rr1-final-a76397cabf99d31a')

    def test_generation_verification_and_byte_determinism(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'a', Path(tmp)/'b'
            self.assertEqual(task.generate(a), task.generate(b))
            self.assertEqual(task.verify(a), task.verify(b, scope='development'))
            for name in (*task.FILES, 'world.json', 'data_manifest.json'):
                self.assertEqual((a/name).read_bytes(), (b/name).read_bytes())
            with self.assertRaises(FileExistsError): task.generate(a)
            manifest = json.loads((a/'data_manifest.json').read_text())
            self.assertEqual({n: f['rows'] for n, f in manifest['files'].items()}, task.FILES)
            self.assertEqual({f['split'] for n, f in manifest['files'].items() if n.startswith('dev_')}, {'development'})
            (a/'world.json').write_text((a/'world.json').read_text()+' ')
            with self.assertRaisesRegex(ValueError, 'world bytes'): task.verify(a)
            (b/'extra.parquet').write_text('x')
            with self.assertRaisesRegex(ValueError, 'Undeclared'): task.verify(b)

    def test_rehashed_duplicate_label_order_leakage_and_split_corruption(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        for name, mutation in itertools.product(('dev_panel.parquet', 'dev_packet_2.parquet', 'final_eval.parquet'),
                                                ('duplicate', 'label', 'order', 'leakage', 'split')):
            with self.subTest(name=name, mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)/'data'; m = task.generate(root)
                rows = copy.deepcopy(task.cases()[name])
                if mutation == 'duplicate': rows[1] = rows[0]
                if mutation == 'label': rows[0]['result'] = rows[0]['result'].replace('ready', 'not_ready') if 'not_ready' not in rows[0]['result'] else rows[0]['result'].replace('not_ready', 'ready')
                if mutation == 'order': rows.reverse()
                if mutation == 'leakage': rows[0] = task.cases()['train_main.parquet'][0]
                if mutation == 'split': rows[0]['split'] = 'train'
                pq.write_table(pa.Table.from_pylist(rows), root/name)
                # Rehashing the bytes and keeping the claimed IDs cannot hide content drift.
                m['files'][name]['sha256'] = task.hash_file(root/name)
                (root/'data_manifest.json').write_text(task.dumps(m))
                with self.assertRaises(ValueError): task.verify(root)
                if mutation in ('duplicate', 'order', 'leakage'):
                    # The stdlib manifest check alone rejects changed ordered membership.
                    bad = copy.deepcopy(m); bad['files'][name]['case_ids'] = [r['case_id'] for r in rows]
                    with self.assertRaises(ValueError): task.validate_manifest(bad)

    def test_development_scope_never_opens_final_or_training_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'data'; task.generate(root)
            closed = [root/n for n in task.FILES if task.SPLIT_OF_FILE[n] != 'development']
            for path in closed:
                path.chmod(0)  # an unreadable sentinel: any open() raises
            with OpenRecorder() as opened:
                manifest = task.verify(root, scope='development')
            seen = [p for p in opened if str(root) in p]
            self.assertEqual(manifest['files']['final_eval.parquet']['rows'], 64)  # metadata only
            self.assertTrue(seen)
            self.assertFalse([p for p in seen if 'final_eval' in p or 'train_main' in p])
            with self.assertRaises(PermissionError): task.verify(root, scope='all')
            with self.assertRaises(ValueError): task.verify(root, scope='final')
            for path in closed:
                path.chmod(0o600)

    def test_split_group_isolation_and_sibling_colocation(self):
        world = task._world()
        owners = {}
        for split, rows in world['cases'].items():
            for case in rows:
                policy = world['assets'][case['asset_id']]['policy_id']
                runs = {r['run_id'] for r in world['runs'][case['asset_id']]}
                for identity in (case['asset_id'], policy, *runs):
                    self.assertIn(owners.setdefault(identity, split), (split,), identity)
        # Counterfactual siblings share an asset and stay together.
        by_asset = {}
        for case in trusted_cases():
            by_asset.setdefault(case['asset_id'], []).append(case)
        pairs = [v for v in by_asset.values() if len(v) > 1]
        self.assertEqual(Counter(v[0]['split'] for v in pairs), {'train': 8, 'final': 4, 'development': 1})
        for pair in pairs:
            self.assertEqual(len({c['split'] for c in pair}), 1)
            self.assertEqual({task.truth(c['asset_id'], c['as_of'])['decision'] for c in pair}, {READY, NOT_READY})
        questions = [' '.join(r['question'].casefold().split()) for rows in task.rows().values() for r in rows]
        self.assertEqual(len(set(questions)), 216)
        # Every policy serves both decisions somewhere, so the policy alone decides nothing.
        packets = task.packets()
        self.assertEqual([len(p) for p in packets], [8, 8, 8])
        self.assertEqual(len(set(sum(packets, []))), 24)
        self.assertEqual(sum(packets, []), task.panels()['development']['case_ids'])
        for packet in packets:
            decisions = Counter(task.truth(world['by_id'][c]['asset_id'], world['by_id'][c]['as_of'])['decision'] for c in packet)
            self.assertEqual(decisions, {READY: 4, NOT_READY: 4})
        self.assertEqual({task.family(c)[0] for c in sum(packets, [])}, set(task.FAMILIES))

    def test_rows_and_tool_outputs_carry_no_label_family_or_split_information(self):
        hidden = re.compile('|'.join((*task.FAMILIES, 'time_sibling', 'boundary', 'multi_witness', 'just_stale',
                                      'completed_recent', 'integrity_pending', 'qualifying', 'usable', 'eligible',
                                      'reward', 'expected', 'decision', r'\bready\b', 'not_ready', 'train', 'development', 'final')))
        allowed = {'get_asset': {'record_id', 'asset_id', 'policy_id'}, 'get_policy': {'record_id', 'policy_id', 'rpo'},
                   'list_backup_runs': {'inventory_id', 'asset_id', 'as_of', 'complete', 'runs'}}
        for split, rows in task.rows().items():
            for row in rows:
                self.assertEqual(set(row), {'question', 'result', 'case_id', 'split'})
                self.assertRegex(row['case_id'], rf'^{task.TASK}/{split}/c-[0-9a-f]{{10}}$')
                self.assertIsNone(hidden.search(row['question'].replace('recovery-ready', '')), row['question'])
                asset_id, as_of = task.target(row['question'])
                asset = task.call('get_asset', {'asset_id': asset_id})
                views = {'get_asset': asset, 'get_policy': task.call('get_policy', {'policy_id': asset['policy_id']}),
                         'list_backup_runs': task.call('list_backup_runs', {'asset_id': asset_id, 'as_of': as_of})}
                for name, view in views.items():
                    self.assertEqual(set(view), allowed[name])
                    self.assertIsNone(hidden.search(task.dumps(view)), task.dumps(view))
                inventory = views['list_backup_runs']
                self.assertLessEqual(len(inventory['runs']), task.MAX_RUNS)
                for run in inventory['runs']:
                    self.assertEqual(set(run), {'run_id', 'recovery_point_at', 'completed_at', 'status', 'integrity'})
                    self.assertLessEqual(task.minutes(run['recovery_point_at']), task.minutes(as_of))
                    if run['completed_at'] is not None:
                        self.assertLessEqual(task.minutes(run['completed_at']), task.minutes(as_of))

    def test_declared_families_agree_with_an_independent_view_classifier(self):
        for case in trusted_cases():
            answer, _ = solve(case)
            inventory = task.call('list_backup_runs', {'asset_id': case['asset_id'], 'as_of': case['as_of']})
            runs = sorted(inventory['runs'], key=lambda r: r['recovery_point_at'], reverse=True)
            usable = set(answer['supporting_run_ids'])
            if not runs:
                family = 'empty_history'
            elif runs[0]['run_id'] in usable:
                family = 'recent_usable'
            elif usable:
                family = 'failed_newer_usable_older'
                self.assertEqual(runs[0]['status'], 'failed')
            elif runs[0]['status'] == 'running':
                family = 'recent_incomplete'
            elif runs[0]['status'] == 'succeeded' and runs[0]['integrity'] in ('failed', 'pending'):
                family = 'recent_integrity_failed'
            else:
                family = 'stale_success'
                self.assertEqual((runs[0]['status'], runs[0]['integrity']), ('succeeded', 'passed'))
            self.assertEqual(family, case['family'], case)
            if case['variant'] == 'multi_witness': self.assertGreaterEqual(len(usable), 2)
            if case['variant'] == 'integrity_pending': self.assertEqual(runs[0]['integrity'], 'pending')

    def test_superficial_shortcuts_do_not_solve_the_task(self):
        def newest(inv):
            return max(inv['runs'], key=lambda r: r['recovery_point_at']) if inv['runs'] else None
        def ok(run):
            return run['status'] == 'succeeded' and run['integrity'] == 'passed'
        def within(run, inv, policy, field):
            return ok(run) and run[field] and solver.moment(inv['as_of']) - solver.moment(run[field]) <= solver.window(policy['rpo'])
        heuristics = {
            'always_ready': lambda inv, pol, case: True,
            'always_not_ready': lambda inv, pol, case: False,
            'any_run': lambda inv, pol, case: bool(inv['runs']),
            'any_succeeded_passed': lambda inv, pol, case: any(ok(r) for r in inv['runs']),
            'newest_succeeded_passed': lambda inv, pol, case: bool(inv['runs']) and ok(newest(inv)),
            'first_listed_succeeded_passed': lambda inv, pol, case: bool(inv['runs']) and ok(inv['runs'][0]),
            'newest_only_within_rpo': lambda inv, pol, case: bool(inv['runs']) and within(newest(inv), inv, pol, 'recovery_point_at'),
            'completed_at_instead_of_recovery_point': lambda inv, pol, case: any(within(r, inv, pol, 'completed_at') for r in inv['runs']),
            'more_than_two_runs': lambda inv, pol, case: len(inv['runs']) > 2,
            'asset_id_parity': lambda inv, pol, case: int(case['asset_id'][-1], 16) % 2 == 0,
            'case_id_parity': lambda inv, pol, case: int(case['case_id'][-1], 16) % 2 == 0,
            'long_rpo': lambda inv, pol, case: solver.window(pol['rpo']).total_seconds() >= 24 * 3600,
        }
        for split in ('train', 'development', 'final'):
            cases = trusted_cases(split)
            for name, rule in heuristics.items():
                right = 0
                for case in cases:
                    asset = task.call('get_asset', {'asset_id': case['asset_id']})
                    policy = task.call('get_policy', {'policy_id': asset['policy_id']})
                    inventory = task.call('list_backup_runs', {'asset_id': case['asset_id'], 'as_of': case['as_of']})
                    right += rule(inventory, policy, case) == (task.truth(case['asset_id'], case['as_of'])['decision'] == READY)
                # The strongest shortcut (using completed_at) fails only on its trap cases.
                limit = 0.97 if name == 'completed_at_instead_of_recovery_point' else 0.80
                self.assertLess(right/len(cases), limit, (split, name, right, len(cases)))
        # Row order and file position carry no decision signal.
        for name, rows in task.cases().items():
            decisions = [json.loads(r['result'])['decision'] == READY for r in rows]
            self.assertNotEqual(decisions, sorted(decisions)); self.assertNotEqual(decisions, sorted(decisions, reverse=True))
            self.assertNotEqual([r['case_id'] for r in rows], sorted(r['case_id'] for r in rows))
        # The trap is present where it matters: training and final, and observable in development.
        trap = [c for c in trusted_cases() if c['variant'] == 'completed_recent']
        self.assertEqual(Counter(c['split'] for c in trap), {'train': 3, 'final': 2, 'development': 1})
        orders = Counter()
        for case in trusted_cases():
            inventory = task.call('list_backup_runs', {'asset_id': case['asset_id'], 'as_of': case['as_of']})
            points = [r['recovery_point_at'] for r in inventory['runs']]
            if len(points) > 2:
                orders['newest_first' if points == sorted(points, reverse=True) else 'oldest_first' if points == sorted(points) else 'mixed'] += 1
        self.assertGreater(orders['mixed'], orders['newest_first'] + orders['oldest_first'])

    def test_time_parsing_is_explicit_utc_and_never_wall_clock(self):
        self.assertEqual(task.stamp(task.minutes('2026-03-01T00:00Z') - 1), '2026-02-28T23:59Z')
        self.assertEqual(task.minutes('2026-11-01T08:30Z') - task.minutes('2026-10-31T08:30Z'), 1440)
        for bad in ('2026-02-30T00:00Z', '2026-13-01T00:00Z', '2026-01-01T24:00Z', '2026-01-01T00:00', '2026-01-01 00:00Z',
                    '2026-01-01T00:00:00Z', 20260101, None):
            with self.assertRaises(ValueError): task.minutes(bad)
        source = (ROOT/'experiment/recovery_task.py').read_text() + (ROOT/'experiment/recovery_server.py').read_text()
        for clock in ('time.time(', 'datetime.now', 'utcnow', 'localtime', 'perf_counter', 'monotonic', 'date.today'):
            self.assertNotIn(clock, source)
        # The evaluation times straddle the served template's fixed date header.
        header = task.minutes('2026-11-11T00:00Z')
        times = [task.minutes(c['as_of']) for c in trusted_cases('development')]
        self.assertTrue(min(times) < header < max(times))
        self.assertFalse([t for t in times if abs(t - header) < 1440])


class VerifierTests(unittest.TestCase):
    def test_reference_solver_uses_only_public_tools_and_agrees_on_every_case(self):
        text = (ROOT/'experiment/recovery_solver.py').read_text()
        self.assertNotRegex(text, r'^\s*(from|import) .*(recovery_task|recovery_calibration)', )
        self.assertNotIn('recovery_task', re.sub(r'""".*?"""', '', text, flags=re.S))
        for case in trusted_cases():
            answer, ledger = solver.solve(case['question'], task.call)
            reward, diagnostics = task.assess(answer, case, ledger)
            self.assertEqual((reward, diagnostics['failure_class']), (1.0, 'correct'), case)
            self.assertEqual([row['tool'] for row in ledger], list(task.TOOLS))
            truth = task.truth(case['asset_id'], case['as_of'])
            self.assertEqual(sorted(json.loads(answer)['supporting_run_ids']), truth['qualifying_run_ids'])
            self.assertEqual(json.loads(task.label(case))['decision'], truth['decision'])

    def test_eligibility_predicate_boundaries(self):
        now, rpo = task.minutes('2026-06-01T12:00Z'), 240
        base = {'asset_id': 'a', 'status': 'succeeded', 'integrity': 'passed', 'recovery_point_at': now - 100,
                'completed_at': now - 80, 'integrity_checked_at': now - 70}
        cases = [({}, True), ({'recovery_point_at': now - rpo}, True),  # inclusive boundary
                 ({'recovery_point_at': now - rpo - 1}, False),  # just stale
                 ({'recovery_point_at': now + 1, 'completed_at': now - 1, 'integrity_checked_at': now - 1}, False),  # future point
                 ({'recovery_point_at': now, 'completed_at': now, 'integrity_checked_at': now}, True),
                 ({'completed_at': now + 1, 'integrity_checked_at': now + 2}, False),  # completion after as_of
                 ({'integrity_checked_at': now + 1}, False),  # integrity known only after as_of
                 ({'integrity_checked_at': None}, False), ({'integrity': 'failed'}, False),
                 ({'integrity': 'pending'}, False), ({'status': 'failed'}, False), ({'status': 'running'}, False),
                 ({'asset_id': 'b'}, False)]
        for change, expected in cases:
            self.assertIs(task.usable({**base, **change}, now, rpo, 'a'), expected, change)

    def test_boundary_and_masked_knowledge_through_the_generated_world(self):
        world = task._world()
        for case in trusted_cases():
            truth = task.truth(case['asset_id'], case['as_of'])
            policy = world['policies'][world['assets'][case['asset_id']]['policy_id']]
            rpo = policy['rpo']['value'] * task.UNIT_MINUTES[policy['rpo']['unit']]
            now = task.minutes(case['as_of'])
            raw = {r['run_id']: r for r in world['runs'][case['asset_id']]}
            view = {r['run_id']: r for r in task.call('list_backup_runs', {'asset_id': case['asset_id'], 'as_of': case['as_of']})['runs']}
            ages = sorted(now - raw[r]['recovery_point_at'] for r in truth['qualifying_run_ids'])
            if case['variant'] == 'boundary':
                self.assertEqual(ages[-1], rpo)
            if case['variant'] == 'just_stale':
                self.assertEqual(min(now - r['recovery_point_at'] for r in raw.values() if r['status'] == 'succeeded'), rpo + 1)
            if case['variant'] == 'completed_recent':
                trap = min(raw.values(), key=lambda r: now - r['recovery_point_at'])
                self.assertGreater(now - trap['recovery_point_at'], rpo)
                self.assertLessEqual(now - trap['completed_at'], rpo)
                self.assertEqual((view[trap['run_id']]['status'], view[trap['run_id']]['integrity']), ('succeeded', 'passed'))
            if case['family'] == 'recent_incomplete':
                hidden = [r for r in raw.values() if r['completed_at'] > now]
                self.assertTrue(hidden)
                for run in hidden:  # the future success is never exposed
                    self.assertEqual((raw[run['run_id']]['status'], view[run['run_id']]['status']), ('succeeded', 'running'))
                    self.assertIsNone(view[run['run_id']]['completed_at'])
                    self.assertEqual(view[run['run_id']]['integrity'], 'not_applicable')
            if case['variant'] == 'integrity_pending':
                later = [r for r in raw.values() if r['integrity_checked_at'] and r['integrity_checked_at'] > now >= r['completed_at']]
                self.assertTrue(later)
                for run in later:
                    self.assertEqual((raw[run['run_id']]['integrity'], view[run['run_id']]['integrity']), ('passed', 'pending'))
                    # Evaluated after the check finished, the same run would qualify.
                    self.assertIn(run['run_id'], task.truth(case['asset_id'], task.stamp(run['integrity_checked_at']))['qualifying_run_ids'])
        # A later evaluation time shows more; an earlier one hides the run entirely.
        case = first('recent_usable')
        newest = max(world['runs'][case['asset_id']], key=lambda r: r['recovery_point_at'])
        earlier = task.call('list_backup_runs', {'asset_id': case['asset_id'], 'as_of': task.stamp(newest['recovery_point_at'] - 1)})
        self.assertNotIn(newest['run_id'], [r['run_id'] for r in earlier['runs']])
        self.assertNotEqual(earlier['inventory_id'], task.inventory_id(case['asset_id'], case['as_of']))

    def test_ready_witness_sets_order_duplicates_and_foreign_runs(self):
        # A multi-witness case whose inventory also lists a run that does not qualify.
        case = next(c for c in trusted_cases() if c['variant'] == 'multi_witness'
                    and len(task._world()['runs'][c['asset_id']]) > len(task.truth(c['asset_id'], c['as_of'])['qualifying_run_ids']))
        answer, ledger = solve(case)
        witnesses = answer['supporting_run_ids']
        self.assertGreaterEqual(len(witnesses), 2)
        for size in range(1, len(witnesses) + 1):
            for subset in itertools.permutations(witnesses, size):
                self.assertEqual(task.reward(task.dumps({**answer, 'supporting_run_ids': list(subset)}), case, ledger), 1.0)
        inventory = json.loads(ledger[2]['content'])
        other = next(r['run_id'] for r in inventory['runs'] if r['run_id'] not in witnesses)
        foreign_case = first('failed_newer_usable_older')
        foreign = solve(foreign_case)[0]['supporting_run_ids'][0]
        for runs in ([], [witnesses[0], witnesses[0]], [*witnesses, other], [other], ['run-00000000'],
                     [witnesses[0], 'run-00000000'], [foreign], [*witnesses, foreign]):
            reward, d = task.assess(task.dumps({**answer, 'supporting_run_ids': runs}), case, ledger)
            self.assertEqual((reward, d['decision_correct'], d['supporting_runs_correct'], d['failure_class']),
                             (0.0, True, False, 'evidence'), runs)

    def test_newer_failed_with_older_usable_and_not_ready_proofs(self):
        case = first('failed_newer_usable_older')
        answer, ledger = solve(case)
        self.assertEqual(answer['decision'], READY)
        inventory = json.loads(ledger[2]['content'])
        newest = max(inventory['runs'], key=lambda r: r['recovery_point_at'])
        self.assertEqual(newest['status'], 'failed'); self.assertNotIn(newest['run_id'], answer['supporting_run_ids'])
        self.assertEqual(task.reward(task.dumps({**answer, 'supporting_run_ids': [newest['run_id']]}), case, ledger), 0)
        self.assertEqual(task.reward(task.dumps({**answer, 'decision': NOT_READY, 'supporting_run_ids': []}), case, ledger), 0)
        for family in task.FAMILIES[2:]:
            case = first(family)
            answer, ledger = solve(case)
            self.assertEqual((answer['decision'], answer['supporting_run_ids']), (NOT_READY, []))
            self.assertEqual(task.reward(task.dumps(answer), case, ledger), 1.0)
            inventory = json.loads(ledger[2]['content'])
            self.assertEqual(bool(inventory['runs']), family != 'empty_history')
            self.assertIs(inventory['complete'], True)
            # Citing one unusable run is not a proof that no usable run exists.
            for run in inventory['runs'][:1]:
                reward, d = task.assess(task.dumps({**answer, 'supporting_run_ids': [run['run_id']]}), case, ledger)
                self.assertEqual((reward, d['failure_class']), (0.0, 'evidence'))
            reward, d = task.assess(task.dumps({**answer, 'decision': READY, 'supporting_run_ids': [r['run_id'] for r in inventory['runs'][:1]] or ['run-00000000']}), case, ledger)
            self.assertEqual((reward, d['decision_correct'], d['failure_class']), (0.0, False, 'semantic'))
            # Without the observed complete inventory there is no not-ready proof.
            reward, d = task.assess(task.dumps(answer), case, ledger[:2])
            self.assertEqual((reward, d['decision_correct'], d['inventory_correct'], d['failure_class']), (0.0, True, False, 'evidence'))

    def test_evidence_must_be_correct_observed_and_correctly_joined(self):
        case, other = first('recent_usable'), first('stale_success')
        answer, ledger = solve(case)
        other_answer, other_ledger = solve(other)
        wrong_time = task.stamp(task.minutes(case['as_of']) + 60)
        cross_time = {'tool': 'list_backup_runs', 'arguments': {'asset_id': case['asset_id'], 'as_of': wrong_time},
                      'content': task.dumps(task.call('list_backup_runs', {'asset_id': case['asset_id'], 'as_of': wrong_time}))}
        full = lambda **changes: task.dumps({**answer, **changes})
        zero = [  # (answer, ledger, diagnostic that must be False)
            (full(asset_record_id='rec-0000000000'), ledger, 'asset_record_correct'),
            (full(asset_record_id=other_answer['asset_record_id']), ledger + other_ledger, 'asset_record_correct'),
            (full(policy_record_id='rec-0000000000'), ledger, 'policy_record_correct'),
            (full(inventory_id='inv-000000000000'), ledger, 'inventory_correct'),
            (full(inventory_id=json.loads(cross_time['content'])['inventory_id']), ledger[:2] + [cross_time], 'inventory_correct'),
            (full(), ledger[:2] + [cross_time], 'inventory_correct'),  # right ID, only a cross-time observation
            (full(), ledger[1:], 'asset_record_correct'),  # the asset record exists but was never observed
            (full(), [ledger[0], ledger[2]], 'policy_record_correct'),
            (full(), ledger[:2], 'inventory_correct'),
            (full(), [], 'asset_record_correct'),
            (full(), other_ledger, 'asset_record_correct'),  # another episode's observations
        ]
        for text, rows, field in zero:
            reward, d = task.assess(text, case, rows)
            self.assertEqual((reward, d['decision_correct'], d[field], d['evidence_correct'], d['failure_class']),
                             (0.0, True, False, False, 'evidence'), (field, text))
        # Observing a different policy does not establish the applicable RPO.
        stranger = next(p for p in task._world()['policies'] if p != task.truth(case['asset_id'], case['as_of'])['policy_id'])
        wrong_policy = {'tool': 'get_policy', 'arguments': {'policy_id': stranger},
                        'content': task.dumps(task.call('get_policy', {'policy_id': stranger}))}
        reward, d = task.assess(full(policy_record_id=json.loads(wrong_policy['content'])['record_id']), case, [ledger[0], wrong_policy, ledger[2]])
        self.assertEqual((reward, d['policy_record_correct'], d['required_evidence_observed']), (0.0, False, False))
        # Wrong target or time is not assessable against this case.
        for change in ({'asset_id': other['asset_id']}, {'as_of': wrong_time}, {'as_of': '2026-11-11T00:00Z'}):
            reward, d = task.assess(full(**change), case, ledger)
            self.assertEqual((reward, d['format_compliant'], d['decision_correct'], d['failure_class']), (0.0, True, None, 'semantic'))
        self.assertEqual(task.reward(full(), case, ledger + [cross_time] + other_ledger), 1.0)  # extra observations are harmless
        # A ledger that disagrees with the trusted environment is an untrusted execution, never a zero.
        for bad in ([{**ledger[2], 'content': ledger[2]['content'].replace('succeeded', 'failed')}],
                    [{**ledger[0], 'arguments': {'asset_id': 'ast-00000000'}}], [{'tool': 'get_asset', 'arguments': {}}],
                    [{**ledger[0], 'tool': 'lookup_asset'}]):
            with self.assertRaises(task.EvidenceIntegrityError): task.assess(full(), case, bad)

    def test_strict_json_contract(self):
        case = first('recent_usable')
        answer, ledger = solve(case)
        valid = task.dumps(answer)
        self.assertEqual(task.reward(' \n'+valid+'\n', case, ledger), 1.0)
        self.assertEqual(task.reward(json.dumps(answer, indent=2), case, ledger), 1.0)
        syntax_ok = ('{}', '[]', 'null', '"ready"', valid[:-1]+',"extra":0}', task.dumps({k: v for k, v in answer.items() if k != 'as_of'}),
                     valid.replace('"ready"', '"READY"'), valid.replace('"ready"', 'true'),
                     task.dumps({**answer, 'supporting_run_ids': answer['supporting_run_ids'][0]}),
                     task.dumps({**answer, 'supporting_run_ids': [1]}), task.dumps({**answer, 'supporting_run_ids': None}),
                     task.dumps({**answer, 'as_of': case['as_of'].replace('Z', ':00Z')}),
                     task.dumps({**answer, 'asset_record_id': None}), task.dumps({'name': 'final', 'parameters': answer}))
        syntax_bad = ('', 'ready', valid+valid, '```json\n'+valid+'\n```', 'Answer: '+valid, valid+' done',
                      valid[:-1]+',"decision":"ready"}', valid[:-1]+',"x":NaN}', valid[:-1], '{"asset_id":')
        for text in syntax_ok + syntax_bad:
            reward, d = task.assess(text, case, ledger)
            self.assertEqual((reward, d['format_compliant'], d['failure_class']), (0.0, False, 'format'), text)
            self.assertEqual(d['json_syntax'], text in syntax_ok, text)
            # A malformed answer has no known semantic content.
            self.assertEqual([d[k] for k in ('target_correct', 'decision_correct', 'evidence_correct')], [None, None, None])
        with self.assertRaises(ValueError): task.expected(case['question'], valid)
        with self.assertRaises(ValueError): task.expected(case['question']+' ', task.label(case))
        with self.assertRaises(ValueError): task.expected('Determine whether asset ast-00000000 is recovery-ready as of 2026-01-01T00:00Z (UTC).', task.label(case))

    def test_world_mutations_move_solver_and_verifier_together(self):
        base = first('recent_usable', 'plain')
        answer, ledger = solve(base)
        witness = answer['supporting_run_ids'][0]
        now = task.minutes(base['as_of'])
        policy_id = task.truth(base['asset_id'], base['as_of'])['policy_id']
        rpo = task.call('get_policy', {'policy_id': policy_id})['rpo']
        limit = rpo['value'] * task.UNIT_MINUTES[rpo['unit']]
        mutations = {
            'just_stale': lambda r: r.update(recovery_point_at=now - limit - 1),
            'completed_after_as_of': lambda r: r.update(completed_at=now + 1, integrity_checked_at=now + 2),
            'integrity_known_after_as_of': lambda r: r.update(integrity_checked_at=now + 1),
            'integrity_failed': lambda r: r.update(integrity='failed'),
            'run_failed': lambda r: r.update(status='failed', integrity='not_applicable', integrity_checked_at=None),
        }
        original = task._world()
        for name, mutate in mutations.items():
            world = copy.deepcopy(original)
            target = next(r for r in world['runs'][base['asset_id']] if r['run_id'] == witness)
            mutate(target)
            with self.subTest(mutation=name), patch.object(task, '_world', return_value=world):
                self.assertEqual(task.truth(base['asset_id'], base['as_of'])['decision'], NOT_READY)
                changed, changed_ledger = solver.solve(base['question'], task.call)
                self.assertEqual(json.loads(changed)['decision'], NOT_READY)
                self.assertEqual(task.reward(changed, base, changed_ledger), 1.0)
                # The stale answer, replayed with evidence from the changed world, earns nothing.
                self.assertEqual(task.reward(task.dumps(answer), base, changed_ledger), 0.0)
                # Observations recorded in the original world no longer match the environment.
                with self.assertRaises(task.EvidenceIntegrityError): task.reward(task.dumps(answer), base, ledger)
        # Boundary counterexample in the other direction: exactly one RPO old is usable.
        stale = first('stale_success', 'just_stale')
        world = copy.deepcopy(original)
        newest = max((r for r in world['runs'][stale['asset_id']] if r['status'] == 'succeeded'), key=lambda r: r['recovery_point_at'])
        newest['recovery_point_at'] += 1
        with patch.object(task, '_world', return_value=world):
            self.assertEqual(task.truth(stale['asset_id'], stale['as_of'])['qualifying_run_ids'], [newest['run_id']])
            changed, changed_ledger = solver.solve(stale['question'], task.call)
            self.assertEqual(json.loads(changed)['supporting_run_ids'], [newest['run_id']])
            self.assertEqual(task.reward(changed, stale, changed_ledger), 1.0)
        # A run that belongs to another asset never supports this one, even if listed.
        world = copy.deepcopy(original)
        next(r for r in world['runs'][base['asset_id']] if r['run_id'] == witness)['asset_id'] = 'ast-ffffffff'
        with patch.object(task, '_world', return_value=world):
            self.assertNotIn(witness, task.truth(base['asset_id'], base['as_of'])['qualifying_run_ids'])

    def test_tool_argument_validation_is_typed_and_never_coerced(self):
        case = first('recent_usable')
        policy_id = task.truth(case['asset_id'], case['as_of'])['policy_id']
        errors = {('get_asset', '{}'): 'invalid_arguments', ('get_asset', '[]'): 'invalid_arguments',
                  ('get_asset', json.dumps({'asset_id': case['asset_id'], 'as_of': case['as_of']})): 'invalid_arguments',
                  ('get_asset', json.dumps({'asset_id': 7})): 'invalid_arguments',
                  ('get_asset', json.dumps({'asset_id': case['asset_id'].upper()})): 'invalid_asset_id',
                  ('get_asset', json.dumps({'asset_id': 'ast-00000000'})): 'unknown_asset',
                  ('get_policy', json.dumps({'policy_id': case['asset_id']})): 'invalid_policy_id',
                  ('get_policy', json.dumps({'policy_id': 'pol-00000000'})): 'unknown_policy',
                  ('get_policy', json.dumps({'asset_id': case['asset_id']})): 'invalid_arguments',
                  ('list_backup_runs', json.dumps({'asset_id': case['asset_id']})): 'invalid_arguments',
                  ('list_backup_runs', json.dumps({'asset_id': case['asset_id'], 'as_of': '2026-02-30T00:00Z'})): 'invalid_as_of',
                  ('list_backup_runs', json.dumps({'asset_id': case['asset_id'], 'as_of': 'now'})): 'invalid_as_of',
                  ('list_backup_runs', json.dumps({'asset_id': 'ast-00000000', 'as_of': case['as_of']})): 'unknown_asset',
                  ('lookup_asset', json.dumps({'asset_id': case['asset_id']})): 'unknown_tool'}
        for (name, raw), code in errors.items():
            with self.assertRaises(task.PolicyActionError) as raised: task.call(name, json.loads(raw))
            self.assertEqual(str(raised.exception), code, (name, raw))
            self.assertIn(code, task.EPISODE_POLICY['policy_error_codes'])
        self.assertEqual(task.call('get_policy', {'policy_id': policy_id})['policy_id'], policy_id)
        self.assertEqual(set(task.policy_error('invalid_arguments')), {'error_type', 'code', 'instruction'})


if __name__ == '__main__':
    unittest.main()
