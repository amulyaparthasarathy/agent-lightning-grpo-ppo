"""Calibration evidence review on production-format collected directories.

Every directory here is SIMULATED test evidence. The base fixture is produced by
running all 240 calibration episodes through the production harness over real
stdio MCP with scripted model responses, and by the production stage-completion
functions; audit events, receipts and the supervisor index are simulated in their
production formats. Its traces carry the labels a live run would carry, so that
the path which admits live evidence can be exercised. Nothing here is a model
result, and nothing is written outside a temporary directory.

The report is always produced through the actual entry point,
scripts/task_local.py report (in process, and as a subprocess for the main cases).
"""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from experiment import reader_assets as assets, reader_contract as c, recovery_task as task
from scripts import task_local
import recovery_fixtures as fx
from recovery_harness_support import SCRIPTED_BEHAVIORS, collected, run_episode
from test_reader_training import source

ROOT = Path(__file__).resolve().parents[1]
GRPO, PPO = c.ALGORITHMS
STAGES = ['greedy-development', 'sampled-packet-1', 'sampled-packet-2', 'sampled-packet-3']


def load(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='recovery-evidence-')
        cls.root = Path(cls.tmp.name)
        cls.plan = fx.executable_plan(source())
        cls.base = cls.root/'complete'
        cached = os.environ.get('RECOVERY_EVIDENCE_FIXTURE')  # development convenience; a stale copy fails review
        if cached:
            shutil.copytree(cached, cls.base)
        else:
            collected(cls.base, cls.plan, simulate_live_labels=True)
        cls.count = 0

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def variant(self):
        type(self).count += 1
        path = self.root/f'variant-{self.count}'
        shutil.copytree(self.base, path)
        return path

    def report(self, directory):
        """The actual reporting entry point, in process."""
        type(self).count += 1
        out = self.root/f'report-{self.count}'
        printed = task_local.calibration_report(SimpleNamespace(evidence=Path(directory), plan=Path(directory)/'plan.json', out=out))
        full = load(out/'report.json')
        self.assertEqual(printed['live_policy_measurement'], full['live_policy_measurement'])
        self.assertTrue((out/'summary.md').read_text().startswith('# Recovery readiness development calibration'))
        return printed, full

    def cli(self, directory, *extra):
        type(self).count += 1
        out = self.root/f'cli-{self.count}'
        done = subprocess.run([sys.executable, str(ROOT/'scripts/task_local.py'), 'report', '--task', 'recovery-readiness-v1',
                               '--evidence', str(directory), '--plan', str(Path(directory)/'plan.json'), '--out', str(out), *extra],
                              capture_output=True, text=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        return done, (json.loads(done.stdout) if done.returncode == 0 else None)

    def stage_dir(self, directory, algorithm, stage):
        return Path(directory)/algorithm/stage

    def rewrite_proof(self, directory, algorithm, stage, edit):
        """Edit a completion consistently in completion.json, receipt.json and the index."""
        run = self.stage_dir(directory, algorithm, stage)
        proof = load(run/'completion.json'); edit(proof); dump(run/'completion.json', proof)
        receipt = load(run/'receipt.json'); receipt['completion'] = proof; dump(run/'receipt.json', receipt)
        collection = load(Path(directory)/'collection.json')
        job = next(j for j in self.plan['jobs'] if j['algorithm'] == algorithm)
        entry = next(e for e in collection['index']['jobs'][job['job_id']]['stages'] if e['stage'] == stage)
        entry['result'] = receipt
        dump(Path(directory)/'collection.json', collection)

    def set_index(self, directory, algorithm, stage, **fields):
        collection = load(Path(directory)/'collection.json')
        job = next(j for j in self.plan['jobs'] if j['algorithm'] == algorithm)
        state = collection['index']['jobs'][job['job_id']]
        entry = next(e for e in state['stages'] if e['stage'] == stage)
        for key in ('result', 'finished_at'):
            entry.pop(key, None)
        entry.update(fields)
        self.derive_status(collection['index'], state)
        dump(Path(directory)/'collection.json', collection)

    @staticmethod
    def derive_status(index, state=None):
        """Job and index status as matched_supervisor derives them from stage and job states."""
        if state is not None:
            stages = {e['status'] for e in state['stages']}
            state['status'] = ('completed' if stages == {'completed'} else 'unknown' if 'unknown' in stages
                               else 'cancelled' if 'cancelled' in stages else 'failed')
        states = {j['status'] for j in index['jobs'].values()}
        index['status'] = ('completed' if states == {'completed'} else 'unknown' if 'unknown' in states
                           else 'cancelled' if 'cancelled' in states else 'failed')

    def never_dispatched(self, directory, algorithm, stage):
        shutil.rmtree(self.stage_dir(directory, algorithm, stage))
        self.set_index(directory, algorithm, stage, status='skipped', call_id=None, reason='dependency_or_supervisor_failed')

    # ----- Fix 1: valid live calibration requires validated evidence -----

    def test_complete_collection_is_valid_through_the_report_command(self):
        done, printed = self.cli(self.base)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual((printed['live_policy_measurement'], printed['evidence_class'], printed['episodes_admitted_after_recomputation_from_traces'],
                          printed['plan_problems']), (True, 'live-model', 240, []))
        self.assertIn('traces and audit events recomputed from the collected directory', printed['provenance'])
        for algorithm in (GRPO, PPO):
            self.assertEqual(printed['accounting'][algorithm], {'expected': 120, 'admitted': 120, 'observed_attempts': 120, 'rewards_confirmed': 120,
                                                               'rewards_missing': 0, 'unattempted': 0, 'unknown': 0, 'rejected_untrusted': 0,
                                                               'unexpected_extra_attempts': 0})
            self.assertEqual((printed['decisions'][algorithm]['operational_validity'], printed['decisions'][algorithm]['reasons']), ('valid', []))
            self.assertIn(printed['decisions'][algorithm]['scientific_suitability'],
                          ('screen_passed', 'saturated', 'too_difficult', 'interface_dominated', 'insufficient_semantic_variation'))
        _, full = self.report(self.base)
        route = full['routes'][GRPO]
        self.assertEqual({s['packet_status'] for s in route['stages'].values()}, {'admitted'})
        self.assertEqual({(s['checks']['completion_summary'], s['checks']['durable_identity'], s['checks']['traces'], s['checks']['audit_events'],
                           s['checks']['recomputed_from_traces_and_audit']) for s in route['stages'].values()},
                         {('valid', 'verified', 'present', 'present', True)})
        self.assertEqual(route['results']['sampled']['episodes'], 96)
        self.assertEqual(route['results']['sampled']['episodes_without_reported_usage'], 0)
        self.assertGreater(route['results']['sampled']['prompt_tokens_max'], 0)  # SIMULATED positive served prompt counts
        self.assertEqual(route['failure_diagnostics'], {})
        self.assertEqual(full['stage_timings_seconds'][f'{GRPO}/greedy-development'], 360.0)
        # The fixture itself says what it is.
        self.assertIn('no model', load(self.base/'collection.json')['SIMULATED'])

    def test_summary_only_directories_are_unverified_and_never_live(self):
        # The reported failure: completion summaries alone, with nothing to support them.
        for zeroed in (False, True):
            with self.subTest(zeroed=zeroed):
                directory = self.root/f'summaries-{zeroed}'
                for job in self.plan['jobs']:
                    for stage in job['stages']:
                        proof = load(self.stage_dir(self.base, job['algorithm'], stage['name'])/'completion.json')
                        self.assertEqual(proof['calibration']['evidence_class'], 'live-model')  # the field says live
                        if zeroed:
                            proof.update(plan_sha256='0'*64, source_sha256='0'*64, configuration_sha256='0'*64, task_sha256='0'*64,
                                         assets_sha256='0'*64, execution_schema='wrong-schema')
                            proof['calibration']['audit'].update(trainer_returned=0, wandb_finished=0)
                        run = self.stage_dir(directory, job['algorithm'], stage['name']); run.mkdir(parents=True)
                        dump(run/'completion.json', proof)
                shutil.copyfile(self.base/'plan.json', directory/'plan.json')
                done, printed = self.cli(directory)
                self.assertEqual(done.returncode, 0, done.stderr)
                self.assertEqual((printed['live_policy_measurement'], printed['evidence_class'], printed['episodes_admitted_after_recomputation_from_traces']),
                                 (False, 'none admitted', 0))
                self.assertIn('not verified: no supervisor index was supplied', printed['provenance'])
                for algorithm in (GRPO, PPO):
                    self.assertEqual((printed['decisions'][algorithm]['operational_validity'], printed['decisions'][algorithm]['scientific_suitability']),
                                     ('invalid', 'not_assessed'))
                    # What was attempted is not established by summaries: unknown (None), not missing and not zero.
                    self.assertEqual(printed['accounting'][algorithm], {'expected': 120, 'admitted': 0, 'observed_attempts': None, 'rewards_confirmed': None,
                                                                       'rewards_missing': None, 'unattempted': 0, 'unknown': 120, 'rejected_untrusted': None,
                                                                       'unexpected_extra_attempts': None})
                _, full = self.report(directory)
                text = (self.root/f'report-{self.count}'/'summary.md').read_text()
                self.assertIn('**Not a live policy measurement.**', text)
                self.assertIn('unknown attempts observed, unknown completed, unknown rewards confirmed, unknown missing; 0 unattempted; 120 unknown', text)
                for stage in full['routes'][PPO]['stages'].values():
                    self.assertEqual((stage['packet_status'], stage['checks']['completion_summary'], stage['episodes']['observed_attempts']),
                                     ('rejected', 'invalid' if zeroed else 'valid', None))
                    self.assertTrue(stage['checks']['durable_identity'].startswith('not checked'))
                self.assertIsNone(full['routes'][PPO]['results'])
        # The plan is validated on every path, and there is no mode that skips it.
        done, _ = self.cli(self.base, '--unvalidated-plan')
        self.assertEqual(done.returncode, 2); self.assertIn('unrecognized arguments', done.stderr)
        bad = self.variant()
        plan = load(bad/'plan.json'); plan['configuration_sha256'] = '0'*64; dump(bad/'plan.json', plan)
        done, _ = self.cli(bad)
        self.assertNotEqual(done.returncode, 0); self.assertIn('regenerate', done.stderr)

    def test_missing_supporting_files_identities_schema_and_events_reject_the_packet(self):
        target = (PPO, 'sampled-packet-2')
        def remove(name):
            return lambda d: (shutil.rmtree if (self.stage_dir(d, *target)/name).is_dir() else os.remove)(self.stage_dir(d, *target)/name)
        def audit_without(event):
            def edit(d):
                path = next((self.stage_dir(d, *target)/'audit').glob('events-*.jsonl'))
                path.write_text(''.join(line + '\n' for line in path.read_text().splitlines() if json.loads(line)['event'] != event))
            return edit
        def audit_with(event):
            def edit(d):
                path = next((self.stage_dir(d, *target)/'audit').glob('events-*.jsonl'))
                path.write_text(path.read_text() + json.dumps(event) + '\n')
            return edit
        def proof_edit(edit):
            return lambda d: self.rewrite_proof(d, *target, edit)
        cases = {
            'no traces': (remove('traces'), 'supporting traces or audit events are not available'),
            'no audit events': (remove('audit'), 'supporting traces or audit events are not available'),
            'no receipt': (remove('receipt.json'), 'durable identity'),
            'no observability': (remove('observability.json'), 'durable identity'),
            'no completion': (remove('completion.json'), 'durable identity'),
            'one trace missing': (lambda d: os.remove(sorted((self.stage_dir(d, *target)/'traces').glob('*.jsonl'))[0]), 'recomput'),
            'zeroed source hash': (proof_edit(lambda p: p.update(source_sha256='0'*64)), 'identity mismatch'),
            'zeroed plan hash': (proof_edit(lambda p: p.update(plan_sha256='0'*64)), 'identity mismatch'),
            'zeroed task hash': (proof_edit(lambda p: p.update(task_sha256='0'*64)), 'identity mismatch'),
            'zeroed assets hash': (proof_edit(lambda p: p.update(assets_sha256='0'*64)), 'identity mismatch'),
            'zeroed configuration hash': (proof_edit(lambda p: p.update(configuration_sha256='0'*64)), 'identity mismatch'),
            'wrong execution schema': (proof_edit(lambda p: p.update(execution_schema='reader-reference-v1')), 'identity mismatch'),
            'wrong wandb identity': (proof_edit(lambda p: p.update(wandb_id='another-run')), 'identity mismatch'),
            'summary says no trainer return': (proof_edit(lambda p: p['calibration']['audit'].update(trainer_returned=0)), 'lacks completion events'),
            'summary says no wandb finish': (proof_edit(lambda p: p['calibration']['audit'].update(wandb_finished=0)), 'lacks completion events'),
            'audit file lacks the trainer return': (audit_without('trainer_returned'), 'recomputation from traces and audit events'),
            'audit file lacks the wandb finish': (audit_without('wandb_finished'), 'recomputation from traces and audit events'),
            'audit file shows an optimizer step': (audit_with({'event': 'optimizer_step', 'role': 'actor', 'count': 1}), 'recomputation from traces and audit events'),
            'audit file shows a checkpoint': (audit_with({'event': 'native_checkpoint_saved', 'step': 1}), 'recomputation from traces and audit events'),
            'summary reward differs from traces': (proof_edit(lambda p: p['calibration']['episodes'][0].update(reward=1.0 - p['calibration']['episodes'][0]['reward'])), 'differs from the completion summary'),
            'receipt disagrees with the index': (lambda d: dump(self.stage_dir(d, *target)/'receipt.json',
                                                               {**load(self.stage_dir(d, *target)/'receipt.json'), 'call_id': 'fc-other'}), 'identity mismatch'),
        }
        for label, (edit, expected) in cases.items():
            with self.subTest(case=label):
                directory = self.variant(); edit(directory)
                printed, full = self.report(directory)
                stage = full['routes'][PPO]['stages'][target[1]]
                self.assertNotEqual(stage['packet_status'], 'admitted')
                self.assertTrue(any(expected in problem for problem in stage['problems']), stage['problems'])
                # The other packets of the route are still individually admitted, but the route is not.
                self.assertEqual([s['packet_status'] for n, s in full['routes'][PPO]['stages'].items() if n != target[1]], ['admitted'] * 3)
                self.assertIsNone(full['routes'][PPO]['results'])
                self.assertEqual((printed['decisions'][PPO]['operational_validity'], printed['decisions'][PPO]['scientific_suitability']), ('invalid', 'not_assessed'))
                self.assertEqual(printed['accounting'][PPO]['admitted'], 88)
                # The other route is unaffected, and the collection as a whole is not a live measurement.
                self.assertEqual(printed['decisions'][GRPO]['operational_validity'], 'valid')
                self.assertFalse(printed['live_policy_measurement'])
        # Without the collected supervisor index nothing is identity-checked, so nothing is admitted.
        directory = self.variant(); os.remove(directory/'collection.json')
        printed, full = self.report(directory)
        self.assertEqual((printed['live_policy_measurement'], printed['episodes_admitted_after_recomputation_from_traces']), (False, 0))
        self.assertEqual({s['packet_status'] for r in full['routes'].values() for s in r['stages'].values()}, {'rejected'})
        self.assertEqual(printed['accounting'][GRPO]['observed_attempts'], 120)  # the traces are still counted as diagnostics
        # An index for another plan verifies nothing either.
        directory = self.variant()
        collection = load(directory/'collection.json'); collection['index']['plan'] = dict(collection['index']['plan'], attempt=2)
        dump(directory/'collection.json', collection)
        printed, _ = self.report(directory)
        self.assertEqual((printed['live_policy_measurement'], printed['episodes_admitted_after_recomputation_from_traces']), (False, 0))
        self.assertIn('belongs to a different plan', printed['provenance'])

    def test_scripted_evidence_is_never_live_even_when_its_fields_say_so(self):
        # One genuinely scripted-labelled packet, produced the same way, placed in an otherwise complete collection.
        scripted = self.root/'scripted-stage'
        collected(scripted, self.plan, only={(GRPO, 'greedy-development')})
        directory = self.variant()
        shutil.rmtree(self.stage_dir(directory, GRPO, 'greedy-development'))
        shutil.copytree(self.stage_dir(scripted, GRPO, 'greedy-development'), self.stage_dir(directory, GRPO, 'greedy-development'))
        collection = load(directory/'collection.json')
        job = self.plan['jobs'][0]
        collection['index']['jobs'][job['job_id']]['stages'][0] = load(scripted/'collection.json')['index']['jobs'][job['job_id']]['stages'][0]
        dump(directory/'collection.json', collection)
        printed, full = self.report(directory)
        # Honestly labelled: the packet is consistent, the route is mixed, and nothing is assessed.
        route = full['routes'][GRPO]
        self.assertEqual((route['stages']['greedy-development']['packet_status'], route['stages']['greedy-development']['evidence_class'], route['evidence_class']),
                         ('admitted', 'scripted-inference', 'mixed'))
        self.assertEqual((route['decision']['operational_validity'], route['decision']['scientific_suitability']), ('mechanics_only_not_live', 'not_assessed'))
        self.assertIn('evidence classes are mixed within the route', route['decision']['reasons'])
        self.assertEqual((printed['evidence_class'], printed['live_policy_measurement']), ('mixed', False))
        self.assertEqual(full['routes'][PPO]['decision']['operational_validity'], 'valid')  # routes are not pooled
        # Relabelled: every summary field now says live-model. The traces still say what they are.
        def relabel(proof):
            proof['calibration']['evidence_class'] = 'live-model'
            for episode in proof['calibration']['episodes']:
                episode['evidence_class'] = 'live-model'
        self.rewrite_proof(directory, GRPO, 'greedy-development', relabel)
        printed, full = self.report(directory)
        stage = full['routes'][GRPO]['stages']['greedy-development']
        self.assertEqual((stage['packet_status'], stage['checks']['completion_summary'], stage['checks']['durable_identity'],
                          stage['checks']['recomputed_from_traces_and_audit'], stage['episodes']['rejected_untrusted']),
                         ('rejected', 'valid', 'verified', False, 24))
        self.assertEqual((printed['decisions'][GRPO]['operational_validity'], printed['decisions'][GRPO]['scientific_suitability'], printed['live_policy_measurement']),
                         ('invalid', 'not_assessed', False))
        reasons = full['routes'][GRPO]['failure_diagnostics']['greedy-development']['trace_facts']['rejection_reasons']
        self.assertEqual(sum(reasons.values()), 24)
        self.assertTrue(all('Synthetic or inconsistent trace identity' in reason for reason in reasons))

    def test_scripted_demonstration_reports_mechanics_only(self):
        # The documented demonstration command, restricted to a few packets to keep the gate short.
        directory = self.root/'demonstration'
        plan = c.make_plan(assets.read(ROOT/'examples/reader-profile.recovery-calibration.synthetic.json'), source())
        collected(directory, plan, only={(GRPO, 'greedy-development'), (GRPO, 'sampled-packet-1')})
        printed, full = self.report(directory)
        self.assertEqual((printed['evidence_class'], printed['live_policy_measurement']), ('scripted-inference', False))
        self.assertEqual(len(printed['plan_problems']), 1)  # a fixture profile cannot have produced live evidence
        route = full['routes'][GRPO]
        self.assertEqual([s['packet_status'] for s in route['stages'].values()], ['admitted', 'admitted', 'unattempted', 'unattempted'])
        self.assertEqual({k: route['accounting'][k] for k in ('admitted', 'observed_attempts', 'unattempted', 'unknown')},
                         {'admitted': 56, 'observed_attempts': 56, 'unattempted': 64, 'unknown': 0})
        self.assertEqual((route['decision']['operational_validity'], route['decision']['scientific_suitability']), ('invalid', 'not_assessed'))
        self.assertEqual(full['routes'][PPO]['accounting']['unattempted'], 120)

    def test_collect_command_reviews_the_downloaded_traces_before_reporting_completeness(self):
        # The production status and collect commands, with the provider volume and registry mocked.
        from experiment import matched_supervisor
        from scripts import reader_training as cli
        root = self.plan['root']
        files = [p for p in self.base.rglob('*') if p.is_file() and p.name not in ('collection.json', 'plan.json')]

        class Volume:
            def iterdir(_, path, recursive):
                return [SimpleNamespace(path='/' + root + '/' + f.relative_to(self.base).as_posix(), size=f.stat().st_size) for f in files]

            def read_file(_, path):
                yield (self.base/Path(path.lstrip('/')).relative_to(root)).read_bytes()
        volume = Volume()
        registry = {'index:' + matched_supervisor.key(self.plan): load(self.base/'collection.json')['index']}
        read = lambda name: json.loads(b''.join(volume.read_file(name)))
        # status sees receipts but no traces: durable identity is verified, nothing is admitted.
        status = cli.status(self.plan, registry, read)
        self.assertFalse(status['evidence_complete'])
        self.assertIn('Traces and audit events are required: run collect', status['incomplete_reason'])
        self.assertEqual({s['checks']['durable_identity'] for r in status['calibration']['routes'].values() for s in r['stages'].values()}, {'verified'})
        self.assertEqual({s['packet_status'] for r in status['calibration']['routes'].values() for s in r['stages'].values()}, {'rejected'})
        # collect downloads, then reviews again from the local copy with the same definition.
        out = self.root/'collected-by-command'
        cli.collect(self.plan, volume, out, status)
        collection = load(out/'collection.json')
        self.assertEqual((collection['evidence_complete'], collection['collection_errors'], 'incomplete_reason' in collection), (True, [], False))
        self.assertTrue(collection['calibration']['live_policy_measurement'])
        self.assertEqual({r['decision']['operational_validity'] for r in collection['calibration']['routes'].values()}, {'valid'})
        # The report command accepts exactly what collect wrote.
        shutil.copyfile(self.base/'plan.json', out/'plan.json')
        printed, _ = self.report(out)
        self.assertEqual((printed['live_policy_measurement'], printed['episodes_admitted_after_recomputation_from_traces']), (True, 240))
        # A download that lost one trace is reported as incomplete by collect itself.
        lossy, victim = self.root/'collected-lossy', sorted(f for f in files if f.suffix == '.jsonl' and 'traces' in f.parts)[0]
        files.remove(victim)
        cli.collect(self.plan, volume, lossy, cli.status(self.plan, registry, read))
        collection = load(lossy/'collection.json')
        self.assertFalse(collection['evidence_complete'])
        lost = collection['calibration']['routes'][victim.parts[-4]]['stages'][victim.parts[-3]]
        # The supervisor says the stage completed: the lost trace is unavailable evidence, not unattempted work.
        self.assertEqual((lost['packet_status'], lost['episodes']['unattempted'], lost['episodes']['unknown']), ('rejected', 0, 1))
        # A collection that cannot be reviewed still keeps the download and the index.
        from unittest import mock
        broken = self.root/'collected-unreviewable'
        before = cli.status(self.plan, registry, read)
        with mock.patch('experiment.reader_evidence.validate_collection', side_effect=RuntimeError('review failed')):
            cli.collect(self.plan, volume, broken, before)
        collection = load(broken/'collection.json')
        self.assertEqual((collection['evidence_complete'], collection['index'] == registry['index:' + matched_supervisor.key(self.plan)]), (False, True))
        self.assertIn('could not be reviewed: RuntimeError', collection['incomplete_reason'])
        self.assertEqual(sum(r['accounting']['admitted'] for r in load(lossy/'collection.json')['calibration']['routes'].values()), 240 - len(
            self.plan['calibration']['stages'][victim.parts[-3]]['case_ids']) * self.plan['calibration']['stages'][victim.parts[-3]]['rollouts_per_question'])

    # ----- Fix 2: failure diagnostics, separate from scientific admission -----

    def failed_packet(self):
        """Greedy complete; packet 1 failed with 31 confirmed rewards and one missing; later packets never dispatched."""
        directory = self.variant()
        run = self.stage_dir(directory, GRPO, 'sampled-packet-1')
        os.remove(run/'completion.json')
        # One rollout really fails at reward delivery, through the production harness.
        victim = sorted((run/'traces').glob('*.jsonl'))[5]
        episode = next(r for r in map(json.loads, victim.read_text().splitlines()) if r['kind'] == 'episode')
        os.remove(victim)
        job = self.plan['jobs'][0]
        request = c.stage_request(self.plan, job, job['stages'][1])
        row = next(r for r in task.rows('development') if r['case_id'] == episode['case_id'])
        failed = run_episode(run, SCRIPTED_BEHAVIORS['correct'], row=row, request=request, route='train', rollout=episode['rollout_id'],
                             attempt='at-failed', mode='reward-failure', simulate_live_labels=True)
        self.assertEqual(failed['error'], 'HTTPStatusError')
        # The collection was never admitted: the stage's audit holds no trainer return and no adapter evidence.
        audit = next((run/'audit').glob('events-*.jsonl'))
        audit.write_text(''.join(json.dumps(e) + '\n' for e in ({'event': 'matched_registered', 'pid': 4242, 'simulated': True},
                                                                {'event': 'weight_refresh', 'step': 0, 'pid': 4242, 'simulated': True})))
        receipt = {k: v for k, v in load(run/'receipt.json').items() if k != 'completion'}
        receipt.update(status='failed', error='SystemExit: 1')  # the format matched_remote.stage_worker writes
        dump(run/'receipt.json', receipt)
        self.set_index(directory, GRPO, 'sampled-packet-1', status='failed', error='SystemExit: 1', finished_at=receipt['finished_at'])
        for later in STAGES[2:]:
            self.never_dispatched(directory, GRPO, later)
        return directory

    def test_failed_packet_with_one_missing_reward_and_unattempted_later_stages(self):
        directory = self.failed_packet()
        done, printed = self.cli(directory)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(printed['accounting'][GRPO], {'expected': 120, 'admitted': 24, 'observed_attempts': 56, 'rewards_confirmed': 55,
                                                      'rewards_missing': 1, 'unattempted': 64, 'unknown': 0, 'rejected_untrusted': 0,
                                                      'unexpected_extra_attempts': 0})
        self.assertEqual(load(printed['report'])['routes'][GRPO]['accounting']['rewards_confirmed_not_admitted'], 31)
        self.assertEqual((printed['decisions'][GRPO]['operational_validity'], printed['decisions'][GRPO]['scientific_suitability']), ('invalid', 'not_assessed'))
        self.assertEqual(printed['decisions'][GRPO]['reasons'], ["the supervisor index records this job as 'failed', not completed",
                                                                 'sampled-packet-1: packet failed', 'sampled-packet-2: packet unattempted',
                                                                 'sampled-packet-3: packet unattempted'])
        self.assertFalse(printed['live_policy_measurement'])
        _, full = self.report(directory)
        route = full['routes'][GRPO]
        self.assertEqual([s['packet_status'] for s in route['stages'].values()], ['admitted', 'failed', 'unattempted', 'unattempted'])
        self.assertEqual(route['accounting']['completed'], 55)
        self.assertIsNone(route['results'])  # the admitted greedy packet alone is not a result
        self.assertNotIn('semantic_evidence_informative_groups', route['decision'])
        failed = route['failure_diagnostics']['sampled-packet-1']
        self.assertEqual(failed['execution'], {'supervisor_status': 'failed', 'dispatched': True, 'receipt_status': 'failed', 'receipt_error': 'SystemExit: 1'})
        self.assertEqual({k: failed['episodes'][k] for k in ('expected', 'admitted', 'observed_attempts', 'completed', 'rewards_confirmed',
                                                             'rewards_missing', 'rejected_untrusted', 'reviewable', 'unattempted', 'unknown')},
                         {'expected': 32, 'admitted': 0, 'observed_attempts': 32, 'completed': 31, 'rewards_confirmed': 31, 'rewards_missing': 1,
                          'rejected_untrusted': 0, 'reviewable': 31, 'unattempted': 0, 'unknown': 0})
        self.assertEqual(failed['trace_facts']['incomplete_exceptions'], {'HTTPStatusError': 1})
        self.assertEqual(failed['checks']['completion_summary'], 'absent')
        # Reviewable episodes are diagnostic: one group has three rewards, and it is not padded or zero-filled.
        vectors = failed['reviewable_reward_vectors']
        self.assertEqual(sorted(len(v['rewards']) for v in vectors), [3] + [4] * 7)
        self.assertEqual(failed['reviewable_episode_summary']['episodes'], 31)
        self.assertIn('diagnostic only', failed['label'])
        # Token evidence is explicit about its coverage: no adapter evidence means unknown, not zero.
        tokens = failed['token_evidence']
        self.assertEqual((tokens['rollouts_expected'], tokens['rollouts_with_token_evidence'], tokens['rollouts_in_adapter_manifest'],
                          tokens['truncated_rows'], tokens['dropped_prompt_rows'], tokens['sampled_tokens'], tokens['fragment_rows']),
                         (32, 0, None, None, None, None, None))  # the collected audit events hold no adapter measurement
        self.assertIn('unknown', tokens['adapter_evidence'])
        self.assertEqual((tokens['model_calls_observed'] > 0, tokens['model_calls_with_reported_usage'] == tokens['model_calls_observed']), (True, True))
        self.assertEqual(failed['audit']['trainer_returned'], 0)
        for later in STAGES[2:]:
            unattempted = route['failure_diagnostics'][later]
            self.assertEqual((unattempted['packet_status'], unattempted['execution']['dispatched'], unattempted['episodes']['unattempted'],
                              unattempted['episodes']['unknown'], unattempted['token_evidence']['truncated_rows']), ('unattempted', False, 32, 0, None))
        self.assertEqual(route['optimization_events_coverage'], '1 of 4 stages (admitted stages only)')
        # The other route completed and is assessed on its own.
        self.assertEqual((full['routes'][PPO]['decision']['operational_validity'], full['routes'][PPO]['accounting']['admitted']), ('valid', 120))
        text = Path(printed['summary']).read_text()
        self.assertIn('55 rewards confirmed, 1 missing; 64 unattempted; 0 unknown', text)
        self.assertIn('Diagnostic, sampled-packet-1 (failed): 32 attempts, 31 rewards confirmed, 1 missing', text)

    def test_sampled_stages_that_were_never_attempted(self):
        directory = self.variant()
        for algorithm in (GRPO, PPO):
            for stage in STAGES[1:]:
                self.never_dispatched(directory, algorithm, stage)
        printed, full = self.report(directory)
        for algorithm in (GRPO, PPO):
            self.assertEqual(printed['accounting'][algorithm], {'expected': 120, 'admitted': 24, 'observed_attempts': 24, 'rewards_confirmed': 24,
                                                               'rewards_missing': 0, 'unattempted': 96, 'unknown': 0, 'rejected_untrusted': 0,
                                                               'unexpected_extra_attempts': 0})
            self.assertEqual([s['packet_status'] for s in full['routes'][algorithm]['stages'].values()], ['admitted'] + ['unattempted'] * 3)
            self.assertEqual((printed['decisions'][algorithm]['operational_validity'], printed['decisions'][algorithm]['scientific_suitability']),
                             ('invalid', 'not_assessed'))
        # Missing local files alone do not show that remote work never happened.
        directory = self.variant()
        shutil.rmtree(self.stage_dir(directory, GRPO, 'sampled-packet-3'))
        printed, full = self.report(directory)
        lost = full['routes'][GRPO]['stages']['sampled-packet-3']
        self.assertEqual((lost['packet_status'], lost['execution']['supervisor_status'], lost['episodes']['unknown'], lost['episodes']['unattempted'],
                          lost['episodes']['observed_attempts']), ('unknown', 'completed', 32, 0, None))
        self.assertEqual({k: printed['accounting'][GRPO][k] for k in ('admitted', 'unattempted', 'unknown')}, {'admitted': 88, 'unattempted': 0, 'unknown': 32})
        # A submission whose outcome is unknown stays unknown.
        directory = self.variant()
        shutil.rmtree(self.stage_dir(directory, GRPO, 'sampled-packet-3'))
        self.set_index(directory, GRPO, 'sampled-packet-3', status='unknown', call_id=None, error='spawn outcome uncertain: TimeoutError')
        _, full = self.report(directory)
        lost = full['routes'][GRPO]['stages']['sampled-packet-3']
        self.assertEqual((lost['packet_status'], lost['episodes']['unknown'], lost['episodes']['unattempted']), ('unknown', 32, 0))

    def test_present_but_invalid_or_untrusted_partial_evidence(self):
        directory = self.variant()
        run = self.stage_dir(directory, GRPO, 'sampled-packet-2')
        traces = sorted((run/'traces').glob('*.jsonl'))
        for path in traces[16:]:
            os.remove(path)  # only half of the packet's traces were collected
        # One remaining trace has its reward changed after the fact; another is not parseable.
        records = [json.loads(line) for line in traces[0].read_text().splitlines()]
        episode = next(r for r in records if r['kind'] == 'episode')
        episode['outputs']['reward'] = 1.0 - episode['outputs']['reward']
        traces[0].write_text(''.join(json.dumps(r) + '\n' for r in records))
        traces[1].write_text('{not json\n')
        # A trace from another stage is dropped into this one.
        foreign = sorted((self.stage_dir(directory, GRPO, 'sampled-packet-1')/'traces').glob('*.jsonl'))[0]
        shutil.copyfile(foreign, run/'traces'/foreign.name)
        expected = {'trace_files': 17, 'observed_attempts': 15, 'rewards_confirmed': 15, 'rewards_missing': 0, 'rewards_confirmed_not_admitted': 15,
                    # Files: 14 intact, one altered, one unreadable, one foreign. Slots: the 14 intact and the altered one.
                    # The supervisor says the stage completed, so the 17 slots without a local trace did run somewhere:
                    # their evidence is unavailable here, which is unknown and not unattempted.
                    'rejected_untrusted': 3, 'reviewable': 14, 'unexpected_extra_attempts': 2, 'unattempted': 0, 'unknown': 17, 'admitted': 0}
        printed, full = self.report(directory)
        stage = full['routes'][GRPO]['stages']['sampled-packet-2']
        self.assertEqual((stage['packet_status'], stage['checks']['completion_summary'], stage['checks']['durable_identity'],
                          stage['checks']['recomputed_from_traces_and_audit']), ('rejected', 'valid', 'verified', False))
        self.assertEqual({k: stage['episodes'][k] for k in expected}, expected)
        facts = full['routes'][GRPO]['failure_diagnostics']['sampled-packet-2']['trace_facts']
        self.assertEqual(facts['unreadable'], 1)
        self.assertEqual(sum(facts['rejection_reasons'].values()), 3)
        self.assertTrue(any('differ from independent re-verification' in reason for reason in facts['rejection_reasons']))
        self.assertTrue(any('different stage' in reason for reason in facts['rejection_reasons']))
        self.assertEqual((printed['decisions'][GRPO]['operational_validity'], printed['decisions'][GRPO]['scientific_suitability']), ('invalid', 'not_assessed'))
        self.assertEqual(printed['accounting'][GRPO], {'expected': 120, 'admitted': 88, 'observed_attempts': 103, 'rewards_confirmed': 103, 'rewards_missing': 0,
                                                      'unattempted': 0, 'unknown': 17, 'rejected_untrusted': 3, 'unexpected_extra_attempts': 2})
        self.assertIsNone(full['routes'][GRPO]['results'])
        # The same evidence under a stage that is not known to have ended: the packet is unknown, not rejected.
        self.set_index(directory, GRPO, 'sampled-packet-2', status='unknown', error='heartbeat_lost; cancellation unconfirmed')
        printed, full = self.report(directory)
        stage = full['routes'][GRPO]['stages']['sampled-packet-2']
        self.assertEqual((stage['packet_status'], stage['checks']['durable_identity']), ('unknown', 'failed'))
        self.assertEqual({k: stage['episodes'][k] for k in expected}, expected)
        self.assertTrue(any("records this job as 'unknown'" in reason for reason in printed['decisions'][GRPO]['reasons']))
        # A stage still in flight with partial traces is unknown as well.
        self.set_index(directory, GRPO, 'sampled-packet-2', status='started')
        _, full = self.report(directory)
        self.assertEqual(full['routes'][GRPO]['stages']['sampled-packet-2']['packet_status'], 'unknown')
        # Extra attempts beyond the declared multiplicity are reported, never silently dropped or counted as slots.
        directory = self.variant()
        run = self.stage_dir(directory, PPO, 'greedy-development')
        job = self.plan['jobs'][1]
        extra = run_episode(run, SCRIPTED_BEHAVIORS['correct'], row=task.rows('development')[0], request=c.stage_request(self.plan, job, job['stages'][0]),
                            route='val', rollout='ro-extra', attempt='at-extra', simulate_live_labels=True)
        self.assertIsNone(extra['error'])
        _, full = self.report(directory)
        stage = full['routes'][PPO]['stages']['greedy-development']
        self.assertEqual((stage['packet_status'], stage['episodes']['trace_files'], stage['episodes']['observed_attempts'],
                          stage['episodes']['unexpected_extra_attempts'], stage['episodes']['unknown'], stage['episodes']['admitted']),
                         ('rejected', 25, 24, 1, 0, 0))

    def test_accounting_counts_expected_slots_and_never_infers_unattempted_work_from_missing_files(self):
        # A completed, verified stage whose download lost twelve traces: the episodes ran, their evidence is unavailable.
        directory = self.variant()
        run = self.stage_dir(directory, GRPO, 'sampled-packet-3')
        for path in sorted((run/'traces').glob('*.jsonl'))[:12]:
            os.remove(path)
        printed, full = self.report(directory)
        stage = full['routes'][GRPO]['stages']['sampled-packet-3']
        self.assertEqual((stage['packet_status'], stage['checks']['durable_identity'], stage['episodes']['observed_attempts'],
                          stage['episodes']['unattempted'], stage['episodes']['unknown']), ('rejected', 'verified', 20, 0, 12))
        self.assertEqual((printed['accounting'][GRPO]['unattempted'], printed['accounting'][GRPO]['unknown']), (0, 12))
        self.assertIn('0 unattempted; 12 unknown', Path(printed['summary']).read_text())
        # The same without any supervisor index, where only the local receipt says the stage completed.
        os.remove(directory/'collection.json')
        _, full = self.report(directory)
        stage = full['routes'][GRPO]['stages']['sampled-packet-3']
        self.assertEqual((stage['packet_status'], stage['episodes']['unattempted'], stage['episodes']['unknown']), ('rejected', 0, 12))
        # A failed stage: a duplicate failed attempt of one rollout does not hide a rollout that left no trace.
        directory = self.failed_packet()
        run = self.stage_dir(directory, GRPO, 'sampled-packet-1')
        job = self.plan['jobs'][0]
        request = c.stage_request(self.plan, job, job['stages'][1])
        confirmed = []
        for path in sorted((run/'traces').glob('*.jsonl')):
            episode = next(r for r in map(json.loads, path.read_text().splitlines()) if r['kind'] == 'episode')
            if episode['execution_status'] == 'ok':
                confirmed.append((path, episode))
        os.remove(confirmed[0][0])  # this rollout now has no trace at all
        _, again = confirmed[1]
        row = next(r for r in task.rows('development') if r['case_id'] == again['case_id'])
        retry = run_episode(run, SCRIPTED_BEHAVIORS['correct'], row=row, request=request, route='train', rollout=again['rollout_id'],
                            attempt='at-retry', mode='reward-failure', simulate_live_labels=True)
        self.assertEqual(retry['error'], 'HTTPStatusError')
        _, full = self.report(directory)
        stage = full['routes'][GRPO]['stages']['sampled-packet-1']
        self.assertEqual({k: stage['episodes'][k] for k in ('expected', 'trace_files', 'observed_attempts', 'rewards_confirmed', 'rewards_missing',
                                                            'unexpected_extra_attempts', 'unattempted', 'unknown')},
                         # 32 files over 31 distinct rollouts: 30 confirmed, one failed delivery, one duplicate attempt, one slot with no trace.
                         {'expected': 32, 'trace_files': 32, 'observed_attempts': 31, 'rewards_confirmed': 30, 'rewards_missing': 1,
                          'unexpected_extra_attempts': 1, 'unattempted': 0, 'unknown': 1})
        self.assertEqual(stage['packet_status'], 'failed')
        # Every route total partitions the expected slots, and unknown totals are unknown, not zero.
        for report in (full, self.report(self.base)[1]):
            for route in report['routes'].values():
                a = route['accounting']
                self.assertEqual(a['expected'], (a['observed_attempts'] or 0) + a['unattempted'] + a['unknown'])
                self.assertEqual(a['observed_attempts'], a['rewards_confirmed'] + a['rewards_missing'])
                self.assertEqual(a['rewards_confirmed'], a['admitted'] + a['rewards_confirmed_not_admitted'])

    def test_stage_known_to_have_failed_without_any_episode_evidence_is_failed_not_unknown(self):
        # The normal shape of a stage that dies before its first episode: a failed receipt and nothing else.
        directory = self.variant()
        run = self.stage_dir(directory, GRPO, 'sampled-packet-2')
        receipt = {k: v for k, v in load(run/'receipt.json').items() if k != 'completion'}
        receipt.update(status='failed', error='startup_no_heartbeat')
        shutil.rmtree(run); run.mkdir(); dump(run/'receipt.json', receipt)
        self.set_index(directory, GRPO, 'sampled-packet-2', status='failed', error='startup_no_heartbeat')
        self.never_dispatched(directory, GRPO, 'sampled-packet-3')
        printed, full = self.report(directory)
        route = full['routes'][GRPO]
        self.assertEqual([s['packet_status'] for s in route['stages'].values()], ['admitted', 'admitted', 'failed', 'unattempted'])
        failed = route['stages']['sampled-packet-2']
        # It failed; whether any episode ran is not established by an absence of local traces.
        self.assertEqual((failed['execution']['receipt_error'], failed['episodes']['observed_attempts'], failed['episodes']['unattempted'],
                          failed['episodes']['unknown']), ('startup_no_heartbeat', None, 0, 32))
        self.assertIn('unknown', failed['token_evidence']['adapter_evidence'])
        self.assertIsNone(failed['token_evidence']['rollouts_with_token_evidence'])
        self.assertEqual({k: printed['accounting'][GRPO][k] for k in ('admitted', 'observed_attempts', 'unattempted', 'unknown')},
                         {'admitted': 56, 'observed_attempts': 56, 'unattempted': 32, 'unknown': 32})
        # A cancelled stage with no local directory is known to have ended too.
        directory = self.variant()
        shutil.rmtree(self.stage_dir(directory, GRPO, 'sampled-packet-2'))
        self.set_index(directory, GRPO, 'sampled-packet-2', status='cancelled', error='heartbeat_lost')
        _, full = self.report(directory)
        self.assertEqual(full['routes'][GRPO]['stages']['sampled-packet-2']['packet_status'], 'failed')
        # An index that says a stage was never dispatched is contradicted by local evidence for it: nothing is claimed.
        directory = self.variant()
        self.set_index(directory, GRPO, 'sampled-packet-2', status='skipped', call_id=None, reason='dependency_or_supervisor_failed')
        _, full = self.report(directory)
        stage = full['routes'][GRPO]['stages']['sampled-packet-2']
        self.assertEqual((stage['packet_status'], stage['episodes']['unattempted'], stage['episodes']['observed_attempts']), ('rejected', 0, 32))
        self.assertTrue(any('never dispatched, but local evidence exists' in p for p in stage['problems']))

    def test_supervisor_index_status_and_stage_list_are_required(self):
        job = self.plan['jobs'][0]
        def edit_index(edit):
            directory = self.variant()
            collection = load(directory/'collection.json'); edit(collection['index']); dump(directory/'collection.json', collection)
            return self.report(directory)
        def states(index, job_status, index_status):
            for state in index['jobs'].values():
                state['status'] = job_status
            index['status'] = index_status
        entry = lambda index: index['jobs'][job['job_id']]['stages']
        cases = {
            'index cancelled, jobs failed': (lambda i: states(i, 'failed', 'cancelled'), 'inconsistent with its jobs', {GRPO, PPO}),
            'jobs failed consistently': (lambda i: states(i, 'failed', 'failed'), "records this job as 'failed'", {GRPO, PPO}),
            'one job cancelled': (lambda i: (i['jobs'][job['job_id']].update(status='cancelled'), i.update(status='cancelled')),
                                  "records this job as 'cancelled'", {GRPO}),
            'index status missing': (lambda i: i.pop('status'), 'inconsistent with its jobs', {GRPO, PPO}),
            'an extra training stage entry': (lambda i: entry(i).append({'stage': 'training', 'status': 'completed', 'call_id': 'fc-extra',
                                                                         'output': job['stages'][0]['output']}),
                                              'exactly the planned stages in order', {GRPO}),
            'stage entries reversed': (lambda i: entry(i).reverse(), 'exactly the planned stages in order', {GRPO}),
            'a duplicated stage entry': (lambda i: entry(i).append(dict(entry(i)[0])), 'exactly the planned stages in order', {GRPO}),
            'a stage entry that is not an object': (lambda i: entry(i).__setitem__(2, 'completed'), 'exactly the planned stages in order', {GRPO}),
            'a job entry that is null': (lambda i: i['jobs'].__setitem__(job['job_id'], None), 'inconsistent with its jobs', {GRPO, PPO}),
        }
        for label, (edit, reason, invalid) in cases.items():
            with self.subTest(case=label):
                printed, full = edit_index(edit)
                self.assertFalse(printed['live_policy_measurement'])
                for algorithm in (GRPO, PPO):
                    decision = printed['decisions'][algorithm]
                    if algorithm in invalid:
                        self.assertEqual((decision['operational_validity'], decision['scientific_suitability']), ('invalid', 'not_assessed'))
                        self.assertTrue(any(reason in r for r in decision['reasons']) or reason in printed['provenance'], (decision, printed['provenance']))
                        self.assertNotIn('screen_on_unadmitted_or_non_live_evidence', decision)
                    else:
                        self.assertEqual(decision['operational_validity'], 'valid')

    def test_malformed_evidence_is_reported_and_never_raises_or_admits(self):
        target = (PPO, 'sampled-packet-1')
        def trace_edit(edit):
            def apply(d):
                path = sorted((self.stage_dir(d, *target)/'traces').glob('*.jsonl'))[0]
                records = [json.loads(line) for line in path.read_text().splitlines()]
                edit(records)
                path.write_text(''.join(json.dumps(r) + '\n' for r in records))
            return apply
        def audit_append(event):
            def apply(d):
                path = next((self.stage_dir(d, *target)/'audit').glob('events-*.jsonl'))
                path.write_text(path.read_text() + json.dumps(event) + '\n')
            return apply
        def replace(name, value):
            return lambda d: dump(self.stage_dir(d, *target)/name, value)
        cases = {
            'episode outputs are a string': trace_edit(lambda rs: next(r for r in rs if r['kind'] == 'episode').update(outputs='reward 1.0')),
            'episode exception is a string': trace_edit(lambda rs: next(r for r in rs if r['kind'] == 'episode').update(exception='boom', execution_status='error')),
            'generation outputs are a string': trace_edit(lambda rs: next(r for r in rs if r['kind'] == 'llm').update(outputs='text')),
            'a trace record is not an object': trace_edit(lambda rs: rs.append('not a record')),
            'an audit event has no event name': audit_append({'step': 0}),
            'an audit event is not an object': audit_append('weight_refresh'),
            'a weight refresh has no step': audit_append({'event': 'weight_refresh'}),
            'a token event has no rollouts': audit_append({'event': 'calibration_collection_tokens'}),
            'the completion is a list': replace('completion.json', []),
            'the completion has a null calibration': lambda d: self.rewrite_proof(d, *target, lambda p: p.update(calibration=None)),
            'the receipt is a list': replace('receipt.json', []),
            'the observability record is a string': replace('observability.json', 'wandb'),
        }
        for label, edit in cases.items():
            with self.subTest(case=label):
                directory = self.variant(); edit(directory)
                done, printed = self.cli(directory)
                self.assertEqual(done.returncode, 0, done.stderr[-600:])
                self.assertEqual((printed['live_policy_measurement'], printed['decisions'][PPO]['operational_validity'],
                                  printed['decisions'][PPO]['scientific_suitability'], printed['accounting'][PPO]['admitted']),
                                 (False, 'invalid', 'not_assessed', 88))
                self.assertEqual(printed['decisions'][GRPO]['operational_validity'], 'valid')
                stage = load(printed['report'])['routes'][PPO]['stages'][target[1]]
                self.assertNotEqual(stage['packet_status'], 'admitted')
                self.assertTrue(stage['problems'])


if __name__ == '__main__':
    unittest.main()
