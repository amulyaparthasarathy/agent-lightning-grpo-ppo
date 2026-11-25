"""Qualification contract, real files, independent verifier, statistics and offline boundaries."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from experiment import matched_launch, matched_supervisor, reader_contract as c, recovery_calibration as cal, recovery_task as task
from experiment import recovery_learning as learning, recovery_learning_evidence as evidence
from scripts import reader_training, recovery_study as study
from experiment import qualification_paths
from recovery_learning_support import profile, request
from test_reader_training import source
import recovery_fixtures as fx


class QualificationTests(unittest.TestCase):
    def test_closed_profiles_schedules_and_reservations(self):
        for selection, routes in learning.ROUTES.items():
            p = c.make_plan(profile(selection), source())
            c.validate(p)
            self.assertEqual([j['algorithm'] for j in p['jobs']], routes)
            self.assertEqual(p['qualification']['total_episodes'], 144*len(routes))
            self.assertEqual(p['bounds']['supervisor_seconds'], 14400*len(routes)+600)
            specs = p['qualification']['stages']
            self.assertEqual([s['expected_episodes'] for s in specs.values()], [24,32,64,24])
            self.assertEqual(specs['collection-probe']['case_ids'], specs['training']['batches'][0])
        for field, value in [('algorithms', []), ('selection','unknown'), ('algorithms', list(c.ALGORITHMS)),
                             ('attempt',2), ('purpose','study'), ('task','calculator-v1'), ('extra',1)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                c.validate_profile(dict(profile(), **{field:value}))
        p = request()['plan']
        for edit in [lambda p: p['jobs'][0].update(steps=16), lambda p: p['qualification']['stages']['training']['batches'].reverse(),
                     lambda p: p.update(phase='qualification'), lambda p: p['jobs'][0]['stages'].pop(),
                     lambda p: p['qualification']['stages']['checkpoint-eval'].update(split='final')]:
            bad = copy.deepcopy(p)
            edit(bad)
            with self.assertRaises(ValueError): c.validate(bad)
        with self.assertRaises(ValueError): c.validate_profile(dict(profile(), bounds=dict(profile()['bounds'], stage_seconds=7200)))

    def test_lost_acknowledgement_holds_exposure_and_stops_paired_dispatch(self):
        plan = c.make_plan(profile('paired'), source())
        persisted, calls = [], []
        def lost(job, stage):
            calls.append((job['algorithm'],stage['name']))
            self.assertEqual(persisted[-1]['worker_seconds_reserved'],3600)
            raise TimeoutError('lost acknowledgement')
        def unexpected(*args):
            self.fail('Unexpected provider poll, cancellation or retry')
        index = matched_supervisor.run(plan,lost,unexpected,unexpected,unexpected,lambda:False,persisted.append,
                                       clock=lambda:100, sleep=unexpected)
        self.assertEqual(calls,[(c.ALGORITHMS[0],'baseline')])
        self.assertEqual(index['status'],'unknown')
        self.assertEqual(index['worker_seconds_reserved'],3600)
        self.assertEqual(index['gpu_seconds_upper_bound_used'],3600)
        self.assertEqual(index['worker_cost_usd_upper_bound_used'],10)
        self.assertAlmostEqual(index['supervisor_reservation_usd'],(8+1/6)*10)
        stages = [stage for job in index['jobs'].values() for stage in job['stages']]
        self.assertEqual([s['status'] for s in stages],['unknown']+['skipped']*7)

    def test_immutable_parquet_and_explicit_eager_paths(self):
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root/profile()['paths']['data']
            assets = fx.assets_manifest()
            assets['data'] = task.generate(data)
            for path in data.glob('final_eval.*'):
                path.chmod(0)
            try:
                for name in c.modes.STAGES[c.modes.RECOVERY_LEARNING]:
                    req = request(name, assets=assets)
                    binding = qualification_paths.observe(req,root,'vo-test')
                    locations = qualification_paths.locations(req,binding)
                    run = locations['run']
                    run.mkdir(parents=True)
                    spec = learning.stage_spec(req)
                    task.verify(data, scope=learning.scope(req))
                    plan, commands = matched_launch.make_plan(req, root, root, locations['model'], data, root/'python', run, 8000, binding=binding)
                    paths = learning.materialize(req, data, run, commands)
                    self.assertEqual(paths, [plan['train_file'], plan['eval_file']])
                    for path, binding in zip(paths, spec['inputs'], strict=True):
                        ids = pq.read_table(path)['case_id'].to_pylist()
                        self.assertEqual(c.digest(ids), binding['ordered_case_ids_sha256'])
                        self.assertEqual(c.file_hash(path), binding['sha256'])
                    self.assertEqual(json.loads((run/'qualification-inputs.json').read_text()), spec['inputs'])
                with self.assertRaises(PermissionError): pq.read_table(data/'final_eval.parquet')
            finally:
                for path in data.glob('final_eval.*'):
                    path.chmod(0o600)

    def test_proposal_rejection_precedes_provider_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'study.json'
            p.write_text(study.DECLARATION.read_text())
            with patch.object(reader_training, 'online', side_effect=AssertionError('provider boundary')):
                with self.assertRaisesRegex(ValueError, 'non-executable'):
                    reader_training.main(['launch', '--plan', str(p), '--profile', str(c.ROOT/'examples/reader-profile.recovery-learning.synthetic.json')])
        validate = study.validate
        validate(json.loads(study.DECLARATION.read_text()))
        with self.assertRaises(ValueError): validate(dict(study.proposal(), executable=True))

    def test_unavailable_or_fabricated_summaries_are_diagnostic(self):
        req = request()
        p = req['plan']
        for index in (None, {'plan':p, 'jobs':[], 'status':'completed'}):
            result = evidence.review(p,index,lambda _: {},None)
            self.assertFalse(result['evidence_complete'])
            self.assertEqual(len(result['stages']),4)

    def test_checkpoint_identity_includes_controller_state(self):
        req = request(selection='ppo-mc')
        with tempfile.TemporaryDirectory() as tmp:
            binding = qualification_paths.observe(req,Path(tmp),'vo-test')
            run = qualification_paths.locations(req,binding)['run']
            run.mkdir(parents=True)
            (run/'path-binding.json').write_text(json.dumps(binding))
            path = run/'checkpoints/global_step_2'
            for role in ('actor','critic'):
                (path/role).mkdir(parents=True)
                for kind in ('model','optim','extra_state'):
                    (path/role/(kind+'_world_size_1_rank_0.pt')).write_bytes(b'synthetic state')
            (run/'resolved_config.json').write_text('{}')
            (path/'data.pt').write_bytes(b'data')
            (path.parent/'latest_checkpointed_iteration.txt').write_text('2')
            identity = matched_launch.checkpoint_identity(path,req,run)
            (path/'matched_identity.json').write_text(json.dumps(identity))
            self.assertEqual(matched_launch.verify_checkpoint(path,req,binding=binding),c.digest(identity))
            for wrong in (path.with_name('global_step_1'), Path(tmp)/'other'/path.relative_to(tmp)):
                with self.assertRaisesRegex(ValueError, 'Wrong bound checkpoint'):
                    matched_launch.verify_checkpoint(wrong,req,binding=binding)
            (path/'actor'/'unexpected').symlink_to(run/'resolved_config.json')
            with self.assertRaises(ValueError):
                matched_launch.verify_checkpoint(path,req,binding=binding)
            (path/'actor'/'unexpected').unlink()
            (path/'data.pt').write_bytes(b'other')
            with self.assertRaises(ValueError): matched_launch.verify_checkpoint(path,req,binding=binding)


class QualificationSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.plan = c.make_plan(profile('paired'), source())
        self.now = 100
        self.snapshots = []
        self.owned_id = 'fc-recovery-baseline'
        self.dispatch = Mock(return_value=self.owned_id)
        self.poll = Mock()
        self.cancel = Mock()

    def supervise(self):
        return matched_supervisor.run(
            self.plan, self.dispatch, self.poll, self.cancel, lambda _: None, lambda: False,
            self.snapshots.append, clock=lambda: self.now, sleep=self.fail)

    def check_interrupted_stage(self, status, call_id):
        index = self.snapshots[-1]
        first, second = self.plan['jobs']
        self.dispatch.assert_called_once_with(first, first['stages'][0])
        receipts = index['jobs'][first['job_id']]['stages']
        stage = receipts[0]
        self.assertEqual(index['jobs'][first['job_id']]['status'], status)
        self.assertEqual((stage['status'], stage['call_id']), (status, call_id))
        skipped_stages = receipts[1:] + index['jobs'][second['job_id']]['stages']
        self.assertEqual(len(skipped_stages), 7)
        for skipped in skipped_stages:
            self.assertEqual((skipped['status'], skipped['call_id']), ('skipped', None))
            self.assertNotIn('reservation', skipped)
        self.assertEqual(stage['reservation'], {'seconds': 3600, 'usd': 10})
        self.assertEqual(index['worker_seconds_reserved'], 3600)
        self.assertEqual(index['gpu_seconds_upper_bound_used'], 3600)
        self.assertEqual(index['worker_cost_usd_upper_bound_used'], 10)
        self.assertAlmostEqual(index['cost_usd_upper_bound_used'], 10 + (self.now - 100)*10/3600)
        return stage

    def test_interruption_during_dispatch(self):
        interrupt = KeyboardInterrupt('before acknowledgement')

        def interrupted_dispatch(*_):
            self.assertEqual(self.snapshots[-1]['worker_seconds_reserved'], 3600)
            self.now += 30
            raise interrupt

        self.dispatch.side_effect = interrupted_dispatch
        with self.assertRaises(KeyboardInterrupt) as raised:
            self.supervise()
        self.assertIs(raised.exception, interrupt)
        self.assertEqual(self.snapshots[-1]['status'], 'unknown')
        # The unknown submission was persisted even before final campaign cleanup.
        first_job = self.plan['jobs'][0]['job_id']
        self.assertTrue(any(i['jobs'][first_job]['status'] == 'unknown'
                            and i['gpu_seconds_upper_bound_used'] == 3600 for i in self.snapshots[:-1]))
        stage = self.check_interrupted_stage('unknown', None)
        self.assertIn('KeyboardInterrupt', stage['error'])
        self.poll.assert_not_called()
        self.cancel.assert_not_called()

    def check_active_interruption(self, cancel_error=None):
        interrupt = KeyboardInterrupt('during poll')

        def interrupted_poll(call_id):
            self.assertEqual(call_id, self.owned_id)
            self.now += 30
            raise interrupt

        def cancel(call_id):
            self.assertEqual(call_id, self.owned_id)
            self.now += 1
            if cancel_error is not None:
                raise cancel_error

        self.poll.side_effect = interrupted_poll
        self.cancel.side_effect = cancel
        with self.assertRaises(KeyboardInterrupt) as raised:
            self.supervise()
        self.assertIs(raised.exception, interrupt)
        self.poll.assert_called_once_with(self.owned_id)
        self.cancel.assert_called_once_with(self.owned_id)
        stage = self.check_interrupted_stage('unknown' if cancel_error else 'cancelled', self.owned_id)
        self.assertIs(stage['cancellation_requested'], True)
        if cancel_error is not None:
            self.assertIn(str(cancel_error), stage['error'])

    def test_interruption_after_acknowledgement(self):
        self.check_active_interruption()

    def test_cancellation_failure_preserves_original_interruption(self):
        self.check_active_interruption(ConnectionError('cancel acknowledgement lost'))

    def test_interrupted_cleanup_preserves_original_interruption(self):
        self.check_active_interruption(SystemExit('cleanup interrupted'))

    def test_confirmed_completion_accounts_only_elapsed_worker_time(self):
        self.plan = c.make_plan(profile('grpo'), source())
        job = self.plan['jobs'][0]
        proofs = {}
        for stage in job['stages']:
            name = stage['name']
            spec = self.plan['qualification']['stages'][name]
            call_id = 'fc-' + name
            proof = {'plan_sha256': c.digest(self.plan), 'stage': name, 'source_sha256': c.digest(self.plan['source']),
                     'configuration_sha256': self.plan['configuration_sha256'], 'wandb_id': stage['wandb']['id'],
                     'worker': {'call_id': call_id, 'container_id': 'ta-' + name}}
            proof['recovery_qualification'] = {
                'schema': 'recovery-qualification-evidence-v1', 'plan_sha256': proof['plan_sha256'], 'stage': name,
                'declared': spec, 'episodes_admitted': spec['expected_episodes'],
                'episodes': fx.rows(spec, lambda *_: 'correct'), 'collections': list(spec['batches']),
                'optimizer_calls': {'actor': spec['updates']}}
            proofs[call_id] = {'call_id': call_id, 'output': stage['output'], 'status': 'completed',
                               'container_id': proof['worker']['container_id'], 'completion': proof}

        self.dispatch.side_effect = lambda job,stage: 'fc-' + stage['name']
        def completed(call_id):
            self.now += 30
            return proofs[call_id]
        self.poll.side_effect = completed
        index = self.supervise()
        self.assertEqual(index['status'], 'completed')
        self.assertEqual(index['worker_seconds_reserved'], 4*3600)
        self.assertEqual(index['gpu_seconds_upper_bound_used'], 4*30)
        self.assertAlmostEqual(index['worker_cost_usd_upper_bound_used'], 4*30*10/3600)
        self.assertAlmostEqual(index['cost_usd_upper_bound_used'], 2*4*30*10/3600)
        self.cancel.assert_not_called()


class StudyTests(unittest.TestCase):
    def test_frozen_study_routes_match_metadata_without_evaluating_final_cases(self):
        proposed = json.loads(study.DECLARATION.read_text())
        with tempfile.TemporaryDirectory() as directory:
            manifest = task.generate(Path(directory)/'data')
        self.assertEqual(c.digest(manifest),proposed['data_manifest_sha256'])
        for stage, splits in {'training':('train','development'), 'development':('development','development'),
                              'final':('final','final')}.items():
            for arg,split in zip(('train_file','val_file'),splits,strict=True):
                binding = proposed['eager_inputs'][stage][arg]
                info = manifest['files'][binding['file']]
                self.assertEqual(info['split'],split)
                self.assertEqual((binding['rows'],binding['sha256']),(info['rows'],info['sha256']))
                self.assertEqual(binding['ordered_case_ids_sha256'],c.digest(info['case_ids']))
        proposed['eager_inputs']['final']['train_file'] = proposed['eager_inputs']['development']['train_file']
        with self.assertRaises(ValueError): study.validate(proposed)

    def test_paired_block_analysis_of_repeated_samples(self):
        blocks = study.blocks('final')
        self.assertEqual([len(study.blocks(s)) for s in ('train','development','final')], [32,6,16])
        # Every case in a block shares a signal. Seed 3 has its exact opposite.
        data = {p:{} for p in study.POLICIES}
        for i, block in enumerate(blocks):
            for cid in block:
                for p, samples in zip(study.POLICIES, ([0,1], [i%2]*2, [1,1], [1-i%2]*2), strict=True):
                    data[p][cid] = samples
        out = study.analyze(data, 2)
        self.assertEqual(out['effects'], {'2001':0, '2002':.5, '2003':0})
        intervals = out['conditional synthetic case-mix uncertainty']
        self.assertEqual(intervals['mean'], [1/6,1/6])
        self.assertEqual(intervals['2001'], [-.25,.25])
        data['base'][cid][1] = None
        with self.assertRaises(ValueError): study.analyze(data, 2)
        del data['base'][cid]
        with self.assertRaises(ValueError): study.analyze(data, 2)
