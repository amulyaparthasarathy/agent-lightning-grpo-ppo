"""Qualification contract, real files, independent verifier, statistics and offline boundaries."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiment import matched_launch, reader_contract as c, recovery_calibration as cal, recovery_task as task
from experiment import recovery_learning as learning, recovery_learning_evidence as evidence
from scripts import reader_training, recovery_study as study
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
        from experiment import matched_supervisor
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
            data = root/'data'
            assets = fx.assets_manifest()
            assets['data'] = task.generate(data)
            for path in data.glob('final_eval.*'):
                path.chmod(0)
            try:
                for name in c.modes.STAGES[c.modes.RECOVERY_LEARNING]:
                    req = request(name, assets=assets)
                    run = root/name
                    run.mkdir()
                    spec = learning.stage_spec(req)
                    task.verify(data, scope=learning.scope(req))
                    plan, commands = matched_launch.make_plan(req, root, root, root/'model', data, root/'python', run, 8000)
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
            run = Path(tmp)
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
            self.assertEqual(matched_launch.verify_checkpoint(path,req),c.digest(identity))
            (path/'data.pt').write_bytes(b'other')
            with self.assertRaises(ValueError): matched_launch.verify_checkpoint(path,req)


class StudyTests(unittest.TestCase):
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
