"""Offline qualification admission at the worker, launcher and CPU preflight boundaries."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import Mock, patch

from experiment import matched_launch, matched_remote, qualification_paths, reader_assets
from experiment import reader_contract as contract, reader_preflight, request_transport
from recovery_learning_support import request, resolved_fixture
from test_reader_training import Registry


class QualificationPathTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(matched_launch, 'resolve_training_config', resolved_fixture))
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)/'physical'
        self.root.mkdir()
        self.mount = self.root.parent/'logical'
        self.mount.symlink_to(self.root, target_is_directory=True)

    def launch_plan(self, req):
        binding = qualification_paths.observe(req, self.mount, 'vo-test')
        paths = qualification_paths.locations(req, binding)
        run = reader_assets.owned_path(self.mount, req['stage']['output'])
        return matched_launch.make_plan(req, self.root, self.root, paths['model'], paths['data'],
                                        self.root/'python', run, 1234, binding=binding)[0]

    def test_cpu_preflight_has_no_future_outputs_and_checks_all_paths(self):
        plan = request(selection='paired')['plan']
        records = reader_preflight.qualification_paths(plan, self.mount, 'vo-test')
        self.assertEqual(8, len(records))
        self.assertEqual([], list(self.root.iterdir()))
        for job in plan['jobs']:
            for stage in job['stages']:
                req = contract.stage_request(plan, job, stage)
                self.assertEqual(records[stage['wandb']['id']], self.launch_plan(req)['path_binding'])
        # A nested escape under one future reload location fails the CPU gate.
        training = next(s for s in job['stages'] if s['name'] == 'training')
        run = self.root/training['output']
        run.mkdir(parents=True)
        (run/'checkpoints').symlink_to(self.root.parent)
        with self.assertRaisesRegex(ValueError, 'escapes'):
            reader_preflight.qualification_paths(plan, self.mount, 'vo-test')
        self.assertFalse((run/'request.json').exists())

    def test_worker_rejects_before_gpu_or_service(self):
        req = request('baseline')
        plan = req['plan']
        volume = Mock(object_id='vo-test')
        for variant in ('volume', 'root', 'stage', 'request', 'plan'):
            with self.subTest(variant=variant):
                registry = Registry()
                records = reader_preflight.qualification_paths(plan, self.mount, volume.object_id)
                record = records[req['stage']['wandb']['id']]
                if variant == 'volume':
                    record['volume_id'] = 'vo-other'
                elif variant == 'root':
                    record['root'] = str(self.root.parent/'other')
                elif variant == 'stage':
                    record['paths']['run'] = plan['jobs'][0]['stages'][1]['output']
                else:
                    record[variant+'_sha256'] = '0'*64
                registry['result:preflight:'+plan['root']] = {
                    'status':'completed', 'plan_sha256':contract.digest(plan),
                    'result':{'status':'passed', 'plan_sha256':contract.digest(plan), 'stage_paths':records}}
                with patch.object(matched_remote.runtime,'run_command') as command, self.assertRaises(ValueError):
                    matched_remote.stage_worker(req,time.time()+3500,c=contract,registry=registry,
                                                volume=volume,call_id='fc-fixture',mount=self.mount)
                command.assert_not_called()
                self.assertEqual([], list(self.root.iterdir()))

    def test_unowned_relative_and_absolute_substitutions(self):
        req = request()
        record = qualification_paths.observe(req, self.mount, 'vo-test')
        for key, value in (('run','other/'+record['paths']['run']), ('request_file','../request.json'),
                           ('train_file','/data/train.parquet'), ('training_checkpoint','other/checkpoints/global_step_2')):
            bad = copy.deepcopy(record)
            bad['paths'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                qualification_paths.locations(req,bad)
        for root in (str(self.root)+'/../physical', '/', 'relative'):
            with self.assertRaises(ValueError):
                qualification_paths.describe(req,root,'vo-test')

    def test_probe_model_request_audit_and_reload_routes(self):
        for name in ('collection-probe', 'checkpoint-eval'):
            rp = self.launch_plan(request(name))
            overrides = dict((k, json.loads(v)) for k,v in (s.split('=',1) for s in rp['overrides']))
            for field, value in (('model','/other/model'),('train_file','/data/development.parquet'),
                                 ('eval_file','/data/final_eval.parquet')):
                with self.subTest(stage=name, field=field), self.assertRaises(ValueError):
                    matched_launch.check_launch_paths(dict(rp, **{field:value}), overrides)
            for field in ('AGL_PPO_RUN_DIR','AGL_PPO_AUDIT_DIR','AGL_MATCHED_REQUEST_FILE','AGL_MATCHED_REQUEST_SHA256'):
                bad = copy.deepcopy(overrides)
                bad['ray_kwargs.ray_init.runtime_env.env_vars'][field] = '/other'
                with self.subTest(stage=name, field=field), self.assertRaises(ValueError):
                    matched_launch.check_launch_paths(rp, bad)
            if name == 'checkpoint-eval':
                ckpt = Path(overrides['trainer.resume_from_path'])
                for wrong in (ckpt.with_name('global_step_1'), self.root/'other'/ckpt.relative_to(self.root)):
                    with self.assertRaises(ValueError):
                        matched_launch.check_launch_paths(rp,dict(overrides, **{'trainer.resume_from_path':str(wrong)}))

    def test_transport_requires_canonical_file_and_exact_content(self):
        req = request('baseline')
        run = reader_assets.owned_path(self.mount, req['stage']['output'])
        run.mkdir(parents=True)
        (run/'request.json').write_text(json.dumps(req))
        for directory in (run, self.mount/req['stage']['output']):
            env = dict(request_transport.environment(req,directory),AGL_PPO_RUN_DIR=str(directory))
            if directory == run:
                self.assertEqual(request_transport.load(env),req)
            else:
                with self.assertRaises(ValueError):
                    request_transport.load(env)
        (run/'request.json').write_text('{}')
        with self.assertRaises(ValueError):
            request_transport.load(dict(request_transport.environment(req,run),AGL_PPO_RUN_DIR=str(run)))
