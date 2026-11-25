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
from recovery_harness_support import SCRIPTED_BEHAVIORS, calibration_demonstration, collected, run_episode
from test_reader_training import source

ROOT = Path(__file__).resolve().parents[1]
GRPO, PPO = c.ALGORITHMS
STAGES = ['greedy-development', 'sampled-packet-1', 'sampled-packet-2', 'sampled-packet-3']
ZERO = {'observed_attempts': 0, 'completed': 0, 'rewards_confirmed': 0, 'rewards_missing': 0, 'rejected_untrusted': 0}


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
        state['status'] = collection['index']['status'] = 'failed'
        dump(Path(directory)/'collection.json', collection)

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
                                                               'rewards_missing': 0, 'unattempted': 0, 'unknown': 0, 'rejected_untrusted': 0})
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
                    # What was attempted is not established by summaries: unknown, not missing and not zero.
                    self.assertEqual(printed['accounting'][algorithm], {'expected': 120, 'admitted': 0, 'observed_attempts': 0, 'rewards_confirmed': 0,
                                                                       'rewards_missing': 0, 'unattempted': 0, 'unknown': 120, 'rejected_untrusted': 0})
                _, full = self.report(directory)
                self.assertIn('**Not a live policy measurement.**', (self.root/f'report-{self.count}'/'summary.md').read_text())
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
            'summary reward differs from traces': (proof_edit(lambda p: p['calibration']['episodes'][0].update(reward=1.0 - p['calibration']['episodes'][0]['reward'])), ''),
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
        self.assertTrue(callable(calibration_demonstration))

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
        self.assertEqual(sum(r['accounting']['admitted'] for r in collection['calibration']['routes'].values()), 240 - len(
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
                                                      'rewards_missing': 1, 'unattempted': 64, 'unknown': 0, 'rejected_untrusted': 0})
        self.assertEqual((printed['decisions'][GRPO]['operational_validity'], printed['decisions'][GRPO]['scientific_suitability']), ('invalid', 'not_assessed'))
        self.assertEqual(printed['decisions'][GRPO]['reasons'], ['sampled-packet-1: packet failed', 'sampled-packet-2: packet unattempted',
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
                         (32, 0, None, None, None, None, None))
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
                                                               'rewards_missing': 0, 'unattempted': 96, 'unknown': 0, 'rejected_untrusted': 0})
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
        self.set_index(directory, GRPO, 'sampled-packet-2', status='unknown', error='heartbeat_lost; cancellation unconfirmed')
        printed, full = self.report(directory)
        stage = full['routes'][GRPO]['stages']['sampled-packet-2']
        self.assertEqual((stage['packet_status'], stage['checks']['completion_summary'], stage['checks']['durable_identity'],
                          stage['checks']['recomputed_from_traces_and_audit']), ('rejected', 'valid', 'failed', False))
        self.assertEqual({k: stage['episodes'][k] for k in ('observed_attempts', 'rewards_confirmed', 'rewards_missing', 'rejected_untrusted',
                                                            'reviewable', 'unattempted', 'unknown', 'admitted')},
                         # 17 files: 14 intact, one altered, one unreadable, one foreign. The stage is not known to
                         # have ended, so the 15 slots without a trace are unknown rather than unattempted.
                         {'observed_attempts': 17, 'rewards_confirmed': 16, 'rewards_missing': 1, 'rejected_untrusted': 3,
                          'reviewable': 14, 'unattempted': 0, 'unknown': 15, 'admitted': 0})
        facts = full['routes'][GRPO]['failure_diagnostics']['sampled-packet-2']['trace_facts']
        self.assertEqual(facts['unreadable'], 1)
        self.assertEqual(sum(facts['rejection_reasons'].values()), 3)
        self.assertTrue(any('differ from independent re-verification' in reason for reason in facts['rejection_reasons']))
        self.assertTrue(any('different stage' in reason for reason in facts['rejection_reasons']))
        self.assertEqual((printed['decisions'][GRPO]['operational_validity'], printed['decisions'][GRPO]['scientific_suitability']), ('invalid', 'not_assessed'))
        self.assertEqual(printed['accounting'][GRPO]['admitted'], 88)
        self.assertIsNone(full['routes'][GRPO]['results'])
        # Extra attempts beyond the declared multiplicity are reported, never silently dropped.
        directory = self.variant()
        run = self.stage_dir(directory, PPO, 'greedy-development')
        job = self.plan['jobs'][1]
        extra = run_episode(run, SCRIPTED_BEHAVIORS['correct'], row=task.rows('development')[0], request=c.stage_request(self.plan, job, job['stages'][0]),
                            route='val', rollout='ro-extra', attempt='at-extra', simulate_live_labels=True)
        self.assertIsNone(extra['error'])
        _, full = self.report(directory)
        stage = full['routes'][PPO]['stages']['greedy-development']
        self.assertEqual((stage['packet_status'], stage['episodes']['observed_attempts'], stage['episodes']['unexpected_extra_attempts'],
                          stage['episodes']['admitted']), ('rejected', 25, 1, 0))


if __name__ == '__main__':
    unittest.main()
