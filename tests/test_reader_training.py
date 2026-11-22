"""Synthetic local fixtures and explicitly mocked submissions; never GPU qualification."""
import ast
import builtins
import copy
from contextlib import redirect_stdout, nullcontext
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiment import reader_contract as c, reader_assets as assets, reader_evidence as evidence
from experiment import matched_contract as historical, matched_launch, matched_supervisor as supervisor
from scripts import reader_training as cli

ROOT = Path(__file__).resolve().parents[1]


def profile():
    return assets.read(ROOT/'examples/reader-profile.synthetic.json')


def source():
    from experiment.reader_source import BASELINE
    files = {str(p.relative_to(ROOT)): 'c'*64 for p in (ROOT/'experiment').glob('*.py')}
    files.update({'experiment/ppo_source_lock.json': 'c'*64, 'reader_app.py': 'c'*64})
    return {'commit': 'b'*40, 'dirty': False, 'repository': 'amulyaparthasarathy/agent-lightning-grpo-ppo',
            'baseline': dict(BASELINE), 'files': files, 'runtime_files': c.packaged(files)}


def plan():
    return c.make_plan(profile(), source())


class Registry(dict):
    object_id = 'di-synthetic'
    def put(self, key, value, skip_if_exists=False):
        if skip_if_exists and key in self:
            return False
        self[key] = copy.deepcopy(value)
        return True


class ContractTests(unittest.TestCase):
    def test_collection_only_completion_rejects_optimizer_or_wrong_panel(self):
        p = profile(); p.update(task='asset-status-v1', purpose='collection-only')
        plan = c.make_plan(p, source()); j = plan['jobs'][0]
        req = c.stage_request(plan, j, j['stages'][0])
        events = [{'event': 'wandb_finished'}, {'event': 'weight_refresh', 'step': 0},
            {'event': 'trainer_returned', 'global_step': 1, 'collection_only': True, 'optimizer_updates': 0},
            {'event': 'development_collection_admitted', 'real_rollouts': 32},
            {'event': 'collection_manifest', 'trajectories': [{'case_id': cid}
                for cid in plan['development']['case_ids'] for _ in range(4)]}]
        for mutation in ('none', 'optimizer', 'gradient', 'checkpoint', 'panel', 'missing'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                run = Path(tmp); (run/'audit').mkdir(); rows = copy.deepcopy(events)
                if mutation in ('optimizer', 'gradient'):
                    rows.append({'event': mutation+'_step'})
                if mutation == 'checkpoint': rows.append({'event': 'native_checkpoint_saved'})
                if mutation == 'panel': rows[-1]['trajectories'][0]['case_id'] = 'wrong'
                if mutation == 'missing': rows.pop()
                (run/'audit/events-fixture.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in rows))
                with patch.object(matched_launch.summarize, 'summarize', return_value=({
                        'attempts_selected': 32, 'completed': 32, 'reward_posts_confirmed': 32}, [])):
                    if mutation == 'none':
                        result = matched_launch.complete_stage(run, req, Path('/data'))
                        self.assertEqual(result['optimizer_updates'], {'actor': 0, 'critic': 0})
                    else:
                        with self.assertRaises(ValueError): matched_launch.complete_stage(run, req, Path('/data'))

    def test_declared_collection_only_panel_and_real_training_configuration(self):
        from experiment import asset_task
        p = profile(); p.update(task=asset_task.TASK, purpose='collection-only')
        development = c.make_plan(p, source()); c.validate(development)
        self.assertEqual(development['phase'], 'collection-only')
        self.assertEqual(development['development']['case_ids'], [r['case_id'] for r in asset_task.cases()['train_smoke.parquet'][:8]])
        self.assertFalse(set(development['development']['case_ids']) & {r['case_id'] for r in asset_task.cases()['eval_main.parquet']})
        self.assertEqual(development['task_identity']['episode_policy'], asset_task.EPISODE_POLICY)
        for job in development['jobs']:
            self.assertEqual([s['name'] for s in job['stages']], ['training'])
            req = c.stage_request(development, job, job['stages'][0])
            with tempfile.TemporaryDirectory() as tmp:
                cfg = matched_launch.configuration(req, Path('/model'), Path(tmp))
                self.assertFalse(cfg['trainer.val_only'])
                self.assertFalse(cfg['trainer.val_before_train'])
                self.assertEqual(cfg['data.train_batch_size'], 8)
        changed = copy.deepcopy(development); changed['development']['case_ids'].reverse()
        with self.assertRaises(ValueError): c.validate(changed)

    def test_synthetic_profile_and_plan_are_offline_non_executable(self):
        p = plan()
        c.validate(p)
        self.assertFalse(p['training_admissible'])
        self.assertEqual(len(p['jobs']), 2)
        self.assertEqual(len({s['wandb']['id'] for j in p['jobs'] for s in j['stages']}), 6)
        self.assertEqual(p['bounds']['max_gpu_seconds'], 21600)
        self.assertEqual(p['bounds']['supervisor_seconds'], 22200)

    def test_reader_deadlines_shorten_exposure_and_preserve_strict_worker_bounds(self):
        issued = 1000.0; limit = 900
        deadline = c.dispatch_deadline(limit, issued)
        self.assertEqual(deadline, 1870.0)
        for worker_now in (issued-20, issued, issued+20):
            self.assertTrue(worker_now < deadline <= worker_now+limit)
        self.assertLess(deadline, issued+limit)
        for bad_limit, bad_time in ((30, issued), (float('inf'), issued), (limit, float('nan')), (True, issued)):
            with self.assertRaises(ValueError): c.dispatch_deadline(bad_limit, bad_time)

    def test_reject_missing_unsupported_unsafe_colliding_and_underfunded(self):
        edits = [lambda p: p['modal'].pop('workspace'), lambda p: p['modal'].update(volume=historical.VOLUME),
                 lambda p: p.update(task='asset-status'), lambda p: p['model'].update(parser='hermes'),
                 lambda p: p['model'].update(revision='main'), lambda p: p.update(algorithms=['ppo-mc-v1']),
                 lambda p: p['paths'].update(output='../escape'), lambda p: p['paths'].update(data='/data'),
                 lambda p: p['paths'].update(staging=p['paths']['model']+'/stage'),
                 lambda p: p['paths'].update(output='restored/run'), lambda p: p['bounds'].update(concurrency=2),
                 lambda p: p['bounds'].update(stage_seconds=float('nan')),
                 lambda p: p['bounds'].update(training_budget_usd=1), lambda p: p['bounds'].update(total_budget_usd=20),
                 lambda p: p['bounds'].update(startup_seconds=3600), lambda p: p['resources'].update(cpu=8)]
        for edit in edits:
            with self.subTest(edit=edit):
                p = profile(); edit(p)
                with self.assertRaises(ValueError):
                    c.validate_profile(p)

    def test_no_cloud_or_preparation_import_for_actual_cli_examples(self):
        original = builtins.__import__
        def guard(name, *args, **kwargs):
            if name.split('.')[0] in ('modal', 'modal_app', 'matched_app', 'reader_app', 'huggingface_hub', 'pyarrow', 'torch'):
                raise AssertionError('Cloud/ML import in offline command: '+name)
            return original(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp, patch('builtins.__import__', side_effect=guard), redirect_stdout(io.StringIO()):
            target = Path(tmp)/'plan.json'
            with patch.object(c, 'source_identity', return_value=source()), patch.object(assets, 'prepare', side_effect=AssertionError('prepare')):
                self.assertEqual(cli.main(['plan', '--profile', str(ROOT/'examples/reader-profile.synthetic.json'), '--out', str(target)]), 0)
                self.assertEqual(cli.main(['validate', '--profile', str(ROOT/'examples/reader-profile.synthetic.json'), '--plan', str(target)]), 0)
                with self.assertRaises(FileExistsError):
                    cli.main(['plan', '--profile', str(ROOT/'examples/reader-profile.synthetic.json'), '--out', str(target)])

    def test_changed_plan_source_or_config_refused(self):
        p = plan()
        with patch.object(c, 'source_identity', return_value={}), self.assertRaisesRegex(ValueError, 'Stale'):
            c.validate_current(p, p['profile'])
        with patch.object(c, 'source_identity', return_value=p['source']), self.assertRaisesRegex(ValueError, 'Fixture'):
            c.validate_current(p, p['profile'])
        p['jobs'][0]['steps'] = 3
        with self.assertRaises(ValueError):
            c.validate(p)

    def test_real_dirty_and_packaged_file_source_binding(self):
        actual = c.source_identity()
        if actual['dirty']:
            with self.assertRaisesRegex(ValueError, 'clean committed'):
                c.source_identity(clean=True)
        self.assertIn('experiment/reader_contract.py', c.packaged(source()['files']))
        p = plan()
        changed = profile(); changed['wandb']['project'] = 'another-reader-project'
        new = c.make_plan(changed, source())
        self.assertNotEqual(p['jobs'][0]['stages'][0]['wandb']['id'], new['jobs'][0]['stages'][0]['wandb']['id'])

    def test_reader_values_reach_numerical_launcher_without_smoke_dispatch(self):
        p = plan()
        for job in p['jobs']:
            for stage in job['stages']:
                request = c.stage_request(p, job, stage)
                run = Path('/data')/stage['output']
                configured, commands = matched_launch.make_plan(request, Path('/bundle'), Path('/repo'),
                    Path('/data')/p['profile']['paths']['model'], Path('/data')/p['profile']['paths']['data'], Path('/python'), run, 8000)
                cfg = {k: json.loads(v) for k, v in (x.split('=', 1) for x in configured['overrides'])}
                self.assertEqual(cfg['trainer.project_name'], p['profile']['wandb']['project'])
                self.assertEqual(cfg['data.train_batch_size'], 8)
                self.assertEqual(cfg['actor_rollout_ref.rollout.n'], 4)
                self.assertEqual(cfg['algorithm.gamma'], 1)
                self.assertEqual(cfg['algorithm.lam'], 1)
                self.assertEqual(cfg['trainer.total_training_steps'], 2)
                self.assertEqual(configured['request'], request)
                self.assertIn(p['profile']['paths']['data'], configured['train_file'])
                self.assertEqual(cfg['agentlightning.local.env_map']['QUESTION'], 'input.question')
                self.assertEqual(cfg['agentlightning.local.env_map']['RESULT'], 'input.result')

    def test_historical_contract_cannot_accept_reader_overrides(self):
        with self.assertRaises((ValueError, KeyError)):
            historical.validate(plan())
        from experiment.ppo_contract import validate_selection
        with self.assertRaises(ValueError):
            validate_selection('ppo-mc-v1', 'main', 'llama3b', 'reader')


class OwnershipTests(unittest.TestCase):
    def test_first_qualification_mocked_dispatch_requires_preflight_not_itself(self):
        p = plan(); p['profile']['fixture'] = False
        p = c.make_plan(p['profile'], source())
        registry = Registry()
        receipt = {'status': 'completed', 'plan_sha256': c.digest(p), 'result': {'status': 'passed'}}
        registry['result:preflight:'+p['root']] = receipt
        cloud, app, volume = Mock(), Mock(), Mock()
        cloud.Workspace.from_context.return_value.hydrate.return_value.name = p['profile']['modal']['workspace']
        cloud.enable_output.side_effect = lambda: nullcontext()
        volume.iterdir.return_value = []
        volume.object_id = 'vo-synthetic'; volume._get_metadata.return_value.version = 1
        volume.read_file.return_value = [json.dumps(receipt).encode()]
        app.app.run.side_effect = lambda **kw: nullcontext()
        app.image.object_id = 'im-synthetic'; app.app.app_id = 'ap-synthetic'
        app.supervise.with_options.return_value.spawn.return_value.object_id = 'fc-mocked'
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {'modal': cloud, 'reader_app': app}), \
             patch.dict(os.environ, {'MODAL_PROFILE': p['profile']['modal']['profile']}), \
             patch.object(cli, 'handles', return_value=(registry, volume)), patch.object(c, 'validate_current'):
            args = SimpleNamespace(command='launch', out=Path(tmp)/'receipt.json')
            result = cli.online(args, p, p['profile'])
            self.assertEqual(result['call_id'], 'fc-mocked')
            self.assertEqual(json.loads(os.environ['AGL_READER_PROFILE']), p['profile'])
            app.supervise.with_options.return_value.spawn.assert_called_once_with(p, image_id='im-synthetic')
            app.app.run.assert_called_once_with(detach=True, environment_name=p['profile']['modal']['environment'])

    def test_account_mismatch_precedes_resource_calls(self):
        modal = Mock()
        modal.Workspace.from_context.return_value.hydrate.return_value.name = 'wrong-account'
        with patch.dict(os.environ, {'MODAL_PROFILE': profile()['modal']['profile']}), self.assertRaises(ValueError):
            cli.account(modal, profile())
        modal.Volume.from_name.assert_not_called()
        modal.Secret.from_name.assert_not_called()

    def test_mocked_submission_unknown_is_durable_and_not_retried(self):
        registry, p = Registry(), plan()
        spawn = Mock(side_effect=TimeoutError('unknown outcome'))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'submission.json'
            with self.assertRaises(TimeoutError):
                cli.submit(p, 'launch', registry, spawn, out, {'workspace': 'synthetic'})
            self.assertEqual(assets.read(out)['status'], 'unknown')
            with self.assertRaises(ValueError):
                cli.submit(p, 'launch', registry, spawn, Path(tmp)/'retry.json', {})
            self.assertEqual(spawn.call_count, 1)

    def test_owned_cancel_does_not_touch_other_plan(self):
        p, r = plan(), Registry()
        r['submission:launch:'+p['root']] = {'plan_sha256': c.digest(p), 'call_id': 'fc-owned'}
        r['submission:prepare:'+p['root']] = {'plan_sha256': 'different', 'call_id': 'fc-unowned'}
        job = p['jobs'][0]; stage = job['stages'][0]
        r['stage:'+c.digest(p)+':'+stage['wandb']['id']] = {'call_id': 'fc-entered'}
        cancel = Mock()
        report = cli.cancel_owned(p, r, cancel)
        self.assertEqual(report['owned_calls'], ['fc-entered', 'fc-owned'])
        self.assertEqual(cancel.call_count, 2)

    def test_supervisor_timeout_and_uncertain_spawn_use_shared_logic(self):
        p = plan(); now = [0]; cancelled = []; saved = []
        def clock(): return now[0]
        def sleep(seconds): now[0] += seconds
        index = supervisor.run(p, Mock(side_effect=TimeoutError()), Mock(), cancelled.append,
            lambda _: None, lambda: False, saved.append, clock=clock, sleep=sleep)
        self.assertEqual(index['status'], 'unknown')
        self.assertFalse(cancelled)
        index = supervisor.run(p, lambda *a: 'fc-synthetic', Mock(side_effect=TimeoutError()), cancelled.append,
            lambda _: None, lambda: False, saved.append, clock=clock, sleep=sleep)
        self.assertEqual(index['status'], 'failed')
        self.assertTrue(cancelled)


class AssetsTests(unittest.TestCase):
    def model_fixture(self, root):
        root.mkdir()
        files = {'config.json': {'architectures': ['LlamaForCausalLM'], 'model_type': 'llama', 'bos_token_id': 128000, 'eos_token_id': [128001, 128008, 128009]},
                 'tokenizer_config.json': {'chat_template': assets.prepare_model.effective_template()},
                 'model.safetensors.index.json': {'weight_map': {'synthetic': 'model.safetensors'}}}
        for name, value in files.items(): (root/name).write_text(json.dumps(value))
        (root/'model.safetensors').write_bytes(b'EXPLICIT SYNTHETIC NON-MODEL BYTES')
        (root/'chat_template.jinja').write_text(assets.prepare_model.effective_template())
        lock = {p.name: {'bytes': p.stat().st_size, 'sha256': c.file_hash(p)} for p in root.iterdir()}
        assets.write_new(root/'model_manifest.json', {'profile': 'llama3b', 'repo_id': c.MODEL, 'revision': c.BASE_REVISION,
            'weights_files': ['model.safetensors'], 'prepared_tokenizer_config_sha256': lock['tokenizer_config.json']['sha256'],
            'template_sha256': lock['chat_template.jinja']['sha256'], 'fixture': True})
        return lock

    def test_model_inventory_template_and_exact_manifest_with_synthetic_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'model'; lock = self.model_fixture(root)
            with patch.object(assets, 'model_lock', return_value=lock):
                assets.verify_model(root)
                (root/'unexpected.safetensors').write_bytes(b'extra')
                with self.assertRaisesRegex(ValueError, 'unexpected'): assets.verify_model(root)
                (root/'unexpected.safetensors').unlink()
                (root/'chat_template.jinja').write_text('changed')
                with self.assertRaisesRegex(ValueError, 'content differs'): assets.verify_model(root)
                lock['chat_template.jinja'] = {'bytes': 7, 'sha256': c.file_hash(root/'chat_template.jinja')}
                with self.assertRaisesRegex(ValueError, 'template mismatch'): assets.verify_model(root)

    def test_symlink_and_existing_staging_are_rejected_before_acquisition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'linked').symlink_to('/tmp', target_is_directory=True)
            with self.assertRaises(ValueError): assets.owned_path(root, 'linked/escape')
            p = plan(); (root/p['profile']['paths']['staging']).mkdir(parents=True)
            run = Mock()
            with self.assertRaises(ValueError): assets.prepare(p, root, run, root, {})
            run.assert_not_called()

    def test_synthetic_assets_never_admissible(self):
        with self.assertRaises(ValueError):
            assets.validate_manifest({'schema': 'reader-assets-v1', 'fixture': True, 'model': {}, 'data': {}, 'historical_match': {}})

    def test_corruption_and_extra_weights_rejected_with_small_content_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'model.safetensors').write_bytes(b'synthetic weight')
            lock = {'model.safetensors': {'bytes': 16, 'sha256': c.file_hash(root/'model.safetensors')}}
            with patch.object(assets, 'model_lock', return_value=lock):
                (root/'model.safetensors').write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError, 'content differs'): assets.verify_model(root)

    def test_real_parquet_preparation_and_label_split_verification(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest('Optional locked reader-test group supplies PyArrow; no GPU dependencies')
        from experiment import prepare_data
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for part, count in [('train', 300), ('eval', 80)]:
                rows = [{'id': f'{part}-{i}', 'question': f'Synthetic {part} {i} + 1?', 'result': str(i+1), 'chain': 'PRIVILEGED'} for i in range(count)]
                pq.write_table(pa.Table.from_pylist(rows), root/f'{part}.parquet')
            argv = ['prepare_data', '--train-parquet', str(root/'train.parquet'), '--eval-parquet', str(root/'eval.parquet'), '--out', str(root/'prepared')]
            with patch('sys.argv', argv), redirect_stdout(io.StringIO()): prepare_data.main()
            m = assets.verify_data(root/'prepared')
            self.assertEqual(m['files']['train_main.parquet']['rows'], 256)
            self.assertNotIn('chain', pq.read_schema(root/'prepared/train_main.parquet').names)
            original = pq.read_table(root/'prepared/eval_main.parquet').to_pylist()
            for mutate in (lambda rows: rows[0].update(case_id='wrong-id'),
                           lambda rows: rows[0].update(split='train'),
                           lambda rows: rows[1].update(question=rows[0]['question'])):
                rows = copy.deepcopy(original); mutate(rows)
                file = root/'prepared/eval_main.parquet'
                pq.write_table(pa.Table.from_pylist(rows), file)
                m['files']['eval_main.parquet']['sha256'] = c.file_hash(file)
                (root/'prepared/data_manifest.json').write_text(json.dumps(m))
                with self.assertRaises(ValueError): assets.verify_data(root/'prepared')
            # A self-consistently rehashed manifest cannot hide privileged columns.
            file = root/'prepared/eval_main.parquet'
            rows = copy.deepcopy(original); rows[0]['chain'] = 'privileged'
            pq.write_table(pa.Table.from_pylist(rows), file)
            m['files']['eval_main.parquet']['sha256'] = c.file_hash(file)
            (root/'prepared/data_manifest.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, 'schema'): assets.verify_data(root/'prepared')

    def test_explicit_preparation_pipeline_uses_owned_paths_and_real_local_parquet(self):
        import zipfile
        import pyarrow as pa
        import pyarrow.parquet as pq
        from experiment import prepare_data
        p = plan()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); seed_model = root/'fixture-model'; lock = self.model_fixture(seed_model)
            calls = []
            def mocked_acquisition(command, log, env, timeout):
                calls.append([str(x) for x in command])
                if str(command[1]).endswith('prepare_model.py'):
                    shutil.copytree(seed_model, command[-1])
                elif command[1:3] == ['-m', 'gdown']:
                    with zipfile.ZipFile(command[-1], 'w') as z:
                        for name, count in [('train', 300), ('test', 80)]:
                            f = root/(name+'.parquet')
                            pq.write_table(pa.Table.from_pylist([{'id': f'{name}-{i}', 'question': f'Synthetic {name} {i}', 'result': str(i)} for i in range(count)]), f)
                            z.write(f, arcname=f.name)
                else:
                    with patch('sys.argv', [str(x) for x in command[1:]]), redirect_stdout(io.StringIO()): prepare_data.main()
            with patch.object(assets, 'model_lock', return_value=lock):
                actual = assets.prepare(p, root, mocked_acquisition, root, {'WANDB_ENTITY': 'fixture'})
                self.assertEqual(actual, assets.inspect(root/p['profile']['paths']['model'], root/p['profile']['paths']['data']))
                # Real Modal mounts alias a physical Volume directory. Preparation
                # records physical source paths; later workers must use owned_path
                # too, without rewriting manifests or accepting nested symlinks.
                mount = root/'mount-alias'; mount.symlink_to(root, target_is_directory=True)
                paths = p['profile']['paths']
                with self.assertRaisesRegex(ValueError, 'Acquired source path/content differs'):
                    assets.inspect(*(mount/paths[key] for key in ('model', 'data', 'staging')))
                self.assertEqual(actual, assets.inspect(*(assets.owned_path(mount, paths[key])
                                                         for key in ('model', 'data', 'staging'))))
                source_file = root/paths['staging']/'train.parquet'
                original = source_file.read_bytes(); source_file.write_bytes(b'corrupt source')
                with self.assertRaisesRegex(ValueError, 'Acquired source path/content differs'):
                    assets.verify_data(root/paths['data'], assets.owned_path(mount, paths['staging']))
                source_file.write_bytes(original)
                outside = root/'unexpected-source.parquet'; source_file.rename(outside); source_file.symlink_to(outside)
                with self.assertRaisesRegex(ValueError, 'Acquired source path/content differs'):
                    assets.verify_data(root/paths['data'], assets.owned_path(mount, paths['staging']))
                self.assertFalse(c.make_plan(profile(), source(), actual)['training_admissible'])
            self.assertEqual(len(calls), 3)
            self.assertTrue(all(p['profile']['paths']['staging'] in ' '.join(call) for call in calls))


class ReaderImportTests(unittest.TestCase):
    def test_actual_reader_image_recipe_has_no_resource_bindings(self):
        import modal
        from modal_image import matched_image
        with (patch.object(modal.Volume, 'from_name', side_effect=AssertionError('Volume')),
             patch.object(modal.Dict, 'from_name', side_effect=AssertionError('Dict')),
             patch.object(modal.Secret, 'from_name', side_effect=AssertionError('Secret'))):
            from experiment.reader_source import inventory
            copies = []
            original = modal.Image.add_local_file
            def capture(image, local, remote, **kwargs):
                copies.append((Path(local), remote))
                return original(image, local, remote, **kwargs)
            extra = ROOT/'experiment/__pycache__/phase3_ignored_extra.py'
            extra.parent.mkdir(exist_ok=True)
            with extra.open('x') as f:
                f.write('raise RuntimeError("must never be packaged")')
            try:
                with patch.object(modal.Image, 'add_local_file', autospec=True, side_effect=capture):
                    image = matched_image(reader=True)
                self.assertIsInstance(image, modal.Image)
                self.assertNotIn(extra, [local for local, _ in copies])
                self.assertEqual({str(local.relative_to(ROOT)) for local, remote in copies
                                  if remote.startswith('/opt/agent-lightning-lab/experiment/')},
                                 {n for n in inventory(ROOT) if n.startswith('experiment/')})
            finally:
                extra.unlink()

    def test_real_child_process_receives_reader_settings_and_observability(self):
        from experiment import launch
        p = plan(); job = p['jobs'][0]; stage = job['stages'][0]
        request = c.stage_request(p, job, stage)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'logs').mkdir()
            configured, commands = matched_launch.make_plan(request, root, root, root/'model', root/'data', Path(sys.executable), root, 8000)
            configured['calc_dir'] = str(root)
            commands['gateway'] = [sys.executable, '-c', 'import time; time.sleep(30)']
            commands['controller'] = commands['gateway'][:]
            commands['trainer'] = [sys.executable, '-c', "import json,os;from pathlib import Path;Path('child-env.json').write_text(json.dumps({k:os.environ[k] for k in ['WANDB_ENTITY','WANDB_PROJECT','WANDB_RUN_ID','AGL_MATCHED_REQUEST_FILE','AGL_MATCHED_REQUEST_SHA256','AGL_TASK_ID','AGL_TASK_SHA256','AGL_ENVIRONMENT_SHA256','AGL_SOURCE_SHA256','AGL_CONFIGURATION_SHA256']}))"]
            ready = Mock(); ready.__enter__ = Mock(return_value=SimpleNamespace(status=200)); ready.__exit__ = Mock(return_value=False)
            # Service health response is explicitly mocked; process creation, environment,
            # logs, observability and owned process-group cleanup are real CPU operations.
            with patch.object(launch.urllib.request, 'urlopen', return_value=ready):
                self.assertEqual(launch.run_services(commands, configured, root, root, root, .5), 0)
            env = assets.read(root/'child-env.json'); obs = assets.read(root/'observability.json')
            self.assertEqual(env['AGL_TASK_ID'], p['profile']['task'])
            self.assertEqual(env['AGL_TASK_SHA256'], c.digest(p['task_identity']))
            self.assertEqual(env['AGL_ENVIRONMENT_SHA256'], c.digest(p['data_manifest']))
            self.assertEqual(env['AGL_SOURCE_SHA256'], c.digest(p['source']))
            self.assertEqual(env['AGL_CONFIGURATION_SHA256'], p['configuration_sha256'])
            self.assertEqual(env['WANDB_PROJECT'], p['profile']['wandb']['project'])
            self.assertEqual(env['WANDB_ENTITY'], p['profile']['wandb']['entity'])
            self.assertEqual(env['AGL_MATCHED_REQUEST_FILE'], str(root/'request.json'))
            self.assertEqual(env['AGL_MATCHED_REQUEST_SHA256'], c.digest(request))
            self.assertEqual(obs['wandb'], stage['wandb'])
            self.assertEqual(obs['wandb_project'], p['profile']['wandb']['project'])
            self.assertEqual(obs['experiment_id'], job['job_id'])

    def test_cpu_receipts_on_resolved_volume_mount_and_command_failure(self):
        # Execute the actual CPU entrypoint body against a real symlinked mount.
        # Acquisition/cloud handles are mocked; filesystem ownership/receipts are real.
        tree = ast.parse((ROOT/'reader_app.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'cpu_action')
        function.decorator_list = []
        for fail in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); physical = root/'volume'; physical.mkdir()
                mount = root/'data'; mount.symlink_to(physical, target_is_directory=True)
                p = plan(); registry = Registry(); volume = Mock()
                owned = lambda _mount, relative: assets.owned_path(mount, relative)
                fake_assets = SimpleNamespace(owned_path=owned, write_new=assets.write_new,
                    read=assets.read, storage_exceeded=lambda _profile: False)
                def run_command(command, log, env, timeout, **kwargs):
                    self.assertGreater(timeout, 0)
                    if fail:
                        raise RuntimeError('Synthetic acquisition failure')
                    assets.write_new(Path(command[-1])/'assets.json', {'synthetic_fixture': True})
                namespace = {'check': Mock(), 'profile': p['profile'], 'time': time, 'c': c,
                    'modal': SimpleNamespace(current_function_call_id=lambda: 'fc-synthetic'),
                    'registry': registry, 'volume': volume, 'assets': fake_assets, 'BUNDLE': ROOT/'experiment',
                    'remote': SimpleNamespace(write=lambda path, data: path.write_text(json.dumps(data)),
                        verify_package=Mock(), capture_build=Mock()),
                    'runtime': SimpleNamespace(PYTHON=Path(sys.executable), environment=lambda _settings: {}, run_command=run_command)}
                exec(compile(ast.Module(body=[function], type_ignores=[]), 'reader_app.py', 'exec'), namespace)
                invoke = lambda: namespace['cpu_action'](p, 'prepare', time.time()+300, image_id='im-synthetic')
                if fail:
                    with self.assertRaisesRegex(RuntimeError, 'Synthetic acquisition failure'): invoke()
                else:
                    self.assertEqual(invoke()['status'], 'completed')
                receipt = assets.read(physical/p['root']/'prepare/receipt.json')
                self.assertEqual(receipt['output'], p['root']+'/prepare')
                self.assertEqual(receipt['status'], 'failed' if fail else 'completed')
                self.assertEqual(registry['result:prepare:'+p['root']], receipt)
                self.assertEqual(volume.commit.call_count, 2)

    def test_existing_launcher_failure_with_explicitly_mocked_health_response(self):
        from test_launcher_failure import LauncherFailureTests
        from experiment import launch
        ready = Mock(); ready.__enter__ = Mock(return_value=SimpleNamespace(status=200)); ready.__exit__ = Mock(return_value=False)
        with patch.object(launch.urllib.request, 'urlopen', return_value=ready):
            LauncherFailureTests().test_trainer_traceback_reaches_launcher_without_error_masking()

    def test_real_sdk_discovers_all_remote_functions_without_builds_or_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); bundle = root/'bundle/experiment'; bundle.mkdir(parents=True)
            for path in (ROOT/'experiment').glob('*.py'): shutil.copy(path, bundle/path.name)
            text = (ROOT/'reader_app.py').read_text().replace("Path('/opt/agent-lightning-lab/experiment')", f'Path({str(bundle)!r})')
            (root/'reader_app.py').write_text(text)
            p = profile(); p['fixture'] = False
            script = '''
import sys, json
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
def deny(event, args):
    if event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto'): raise AssertionError('Network forbidden')
sys.addaudithook(deny)
import modal
from modal._runtime.user_code_imports import import_single_function_service
from modal_proto import api_pb2
with patch.object(modal, 'is_local', return_value=False), patch.object(modal.Image, 'from_registry', side_effect=AssertionError('build forbidden')):
    for name in ('stage_worker', 'supervise', 'cpu_action'):
        service = import_single_function_service(api_pb2.Function(module_name='reader_app', function_name=name), None)
        assert service._user_defined_callable.__name__ == name
assert 'modal_app' not in sys.modules and 'matched_app' not in sys.modules
assert not any(n in sys.modules for n in ('torch', 'ray', 'transformers', 'pyarrow'))
print('Real Modal SDK discovery passed; no hydration, build, RPC or submission')
'''
            result = subprocess.run([sys.executable, '-I', '-c', script, str(root)], cwd=root,
                env={'PATH': os.environ['PATH'], 'MODAL_CONFIG_PATH': '/dev/null', 'AGL_READER_PROFILE': json.dumps(p)},
                capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)


class CompletionTests(unittest.TestCase):
    def test_evaluation_updates_refreshes_and_missing_critic_fail_shared_completion(self):
        p = plan(); job = p['jobs'][1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'audit').mkdir()
            def check(stage, events, message):
                (root/'audit/events-1.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
                request = c.stage_request(p, job, stage)
                with self.assertRaisesRegex(ValueError, message): matched_launch.complete_stage(root, request, root)
            base = [{'event': 'wandb_finished'}, {'event': 'trainer_returned', 'global_step': 0}, {'event': 'weight_refresh', 'step': 0}]
            check(job['stages'][0], base+[{'event': 'optimizer_step', 'role': 'actor', 'count': 1}], 'No-update')
            check(job['stages'][0], base+[{'event': 'weight_refresh', 'step': 0}], 'refresh')
            train = [{'event': 'wandb_finished'}, {'event': 'trainer_returned', 'global_step': 2}]
            train += [{'event': 'weight_refresh', 'step': i} for i in range(3)]
            train += [{'event': 'optimizer_step', 'role': 'actor', 'count': i} for i in (1, 2)]
            check(job['stages'][1], train, 'critic optimizer')
            check(job['stages'][1], train+[{'event': 'weight_refresh', 'step': 2}], 'refresh')

    def test_collection_rejects_missing_duplicate_and_mismatched_durable_evidence(self):
        p = plan(); files = {}; index = {'plan': p, 'status': 'completed', 'jobs': {}}
        for job in p['jobs']:
            receipts = []; previous = None
            for stage in job['stages']:
                call = 'fc-'+stage['wandb']['id']
                proof = {'execution_schema': c.SCHEMA, 'plan_sha256': c.digest(p), 'stage': stage['name'], 'worker': {'call_id': call, 'hostname': 'modal', 'container_id': 'ta-'+c.digest(call)[:16]},
                         'source_sha256': c.digest(p['source']), 'configuration_sha256': p['configuration_sha256'],
                         'assets_sha256': c.digest(p['assets']), 'task_sha256': c.digest(p['task_identity']),
                         'wandb_id': stage['wandb']['id'], 'checkpoint_sha256': 'd'*64}
                if stage['name'] == 'training':
                    previous = proof['worker']
                    proof['learning_diagnostics'] = {'informative_reward_groups': 0}
                if stage['name'] == 'checkpoint-eval': proof['training_worker'] = previous
                receipt = {'call_id': call, 'container_id': proof['worker']['container_id'], 'status': 'completed', 'completion': proof, 'output': stage['output']}
                files[stage['output']+'/receipt.json'] = receipt
                files[stage['output']+'/completion.json'] = proof
                files[stage['output']+'/observability.json'] = {'wandb': stage['wandb']}
                receipts.append({'stage': stage['name'], 'status': 'completed', 'call_id': call, 'result': receipt, 'output': stage['output']})
            index['jobs'][job['job_id']] = {'stages': receipts}
        report = evidence.validate_collection(p, index, files.__getitem__)
        self.assertTrue(report['evidence_complete'])
        self.assertEqual(report['operational_completion'], 'completed')
        self.assertEqual(report['live_qualification'], 'gaps remain')
        grpo = report['algorithms'][c.ALGORITHMS[0]]
        self.assertEqual(grpo['informative_grpo_reward_groups'], 0)
        self.assertFalse(grpo['harness_integration_qualified'])
        self.assertIn('Informative group-relative task-advantage evidence was not observed', grpo['qualification_gaps'])
        # Synthetic receipt aggregation: GRPO integration cannot qualify PPO-MC.
        stage = p['jobs'][0]['stages'][0]
        proof = files[stage['output']+'/completion.json']
        proof['model_tool_integration'] = {'schema': 'reader-integration-v2', 'algorithm': c.ALGORITHMS[0],
            'stage': stage['name'], 'qualified': True, 'episodes_with_valid_chain': 1, 'trace_files': 1,
            'coverage': [{'chains': [{'synthetic_test_only': True}]}]}
        report = evidence.validate_collection(p, index, files.__getitem__)
        self.assertFalse(report['algorithms'][c.ALGORITHMS[0]]['harness_integration_qualified'])  # all three stages required in v2
        self.assertFalse(report['algorithms'][c.ALGORITHMS[1]]['harness_integration_qualified'])
        self.assertEqual(report['live_qualification'], 'gaps remain')
        # Read-only historical interpretation preserves the published v1 any-stage rule.
        legacy_plan, legacy_index, legacy_files = copy.deepcopy((p, index, files))
        legacy_plan['schema'] = c.LEGACY_SCHEMA
        legacy_index['plan'] = legacy_plan
        for name, item in legacy_files.items():
            if name.endswith('/completion.json'):
                item['execution_schema'] = c.LEGACY_SCHEMA
                item['plan_sha256'] = c.digest(legacy_plan)
                if item.get('model_tool_integration'):
                    item['model_tool_integration']['schema'] = 'reader-integration-v1'
        legacy_report = evidence.validate_collection(legacy_plan, legacy_index, legacy_files.__getitem__)
        self.assertTrue(legacy_report['algorithms'][c.ALGORITHMS[0]]['harness_integration_qualified'])
        self.assertEqual(legacy_report['algorithms'][c.ALGORITHMS[0]]['integration_rule'], 'any-stage-v1')
        bad = copy.deepcopy(index); bad['jobs'][p['jobs'][0]['job_id']]['stages'].append(bad['jobs'][p['jobs'][0]['job_id']]['stages'][0])
        with self.assertRaisesRegex(ValueError, 'duplicate'): evidence.validate_collection(p, bad, files.__getitem__)
        file = p['jobs'][0]['stages'][2]['output']+'/completion.json'
        bad_files = copy.deepcopy(files); bad_files[file]['checkpoint_sha256'] = 'e'*64
        with self.assertRaises(ValueError): evidence.validate_collection(p, index, bad_files.__getitem__)
        del bad_files[file]
        with self.assertRaises(KeyError): evidence.validate_collection(p, index, bad_files.__getitem__)

    def fixture(self, root):
        p = plan(); j = p['jobs'][0]; request = c.stage_request(p, j, j['stages'][1])
        events = [{'event': 'wandb_finished', 'id': request['stage']['wandb']['id']},
                  {'event': 'trainer_returned', 'validation_only': False, 'pid': 123},
                  {'event': 'native_checkpoint_saved', 'step': 2}]
        for i in (1, 2):
            events += [{'event': 'optimizer_step', 'role': 'actor', 'count': i},
                       {'event': 'gradient_step', 'role': 'actor', 'preclip_finite': True, 'optimizer_called': True},
                       {'event': 'matched_advantages', 'iteration': i, 'real_rollouts': 32, 'generated_tokens': 100, 'constant_groups': 8}]
        (root/'audit').mkdir(); (root/'metrics.jsonl').write_text(''.join(json.dumps({'step': i, 'metrics': {'actor/pg_loss': 0.0}})+'\n' for i in (1, 2)))
        assets.write_new(root/'checkpoints/global_step_2/matched_identity.json', {'roles': {'actor': {
            f'{kind}_world_size_1_rank_0.pt': {'bytes': 1, 'sha256': 'f'*64} for kind in ('model', 'optim', 'extra_state')}}})
        return request, events

    def test_completion_duplicates_counts_finite_losses_and_no_informative_signal(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'AGL_READER_CALL_ID': 'fc-synthetic', 'MODAL_TASK_ID': 'ta-synthetic'}):
            root = Path(tmp); request, events = self.fixture(root)
            def save(es): (root/'audit/events-1.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in es))
            save(events)
            result = evidence.complete(root, request, {})
            self.assertEqual(result['learning_diagnostics']['informative_reward_groups'], 0)
            self.assertEqual(result['learning_diagnostics']['informative_group_relative_task_advantage'], 'not observed')
            self.assertFalse(result['model_tool_integration']['qualified'])
            saved_metrics = (root/'metrics.jsonl').read_text()
            (root/'metrics.jsonl').write_text(saved_metrics.splitlines()[0]+'\n')
            with self.assertRaisesRegex(ValueError, 'Missing update/role'): evidence.complete(root, request, {})
            (root/'metrics.jsonl').write_text(saved_metrics)
            for edit in (lambda es: es.append(es[0]), lambda es: es[3].update(count=2),
                         lambda es: es[4].update(preclip_finite=False), lambda es: es[5].update(real_rollouts=31)):
                changed = copy.deepcopy(events); edit(changed); save(changed)
                with self.assertRaises(ValueError): evidence.complete(root, request, {})
            save(events)
            (root/'metrics.jsonl').write_text(json.dumps({'metrics': {'actor/pg_loss': float('nan')}})+'\n')
            with self.assertRaises(ValueError): evidence.complete(root, request, {})

    def test_ppo_completion_requires_both_critic_losses(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'AGL_READER_CALL_ID': 'fc-synthetic', 'MODAL_TASK_ID': 'ta-synthetic'}):
            root = Path(tmp); request, events = self.fixture(root)
            job = request['plan']['jobs'][1]
            request = c.stage_request(request['plan'], job, job['stages'][1])
            events[0]['id'] = request['stage']['wandb']['id']
            events += [{**e, 'role': 'critic'} for e in events if e['event'] in ('optimizer_step', 'gradient_step')]
            checkpoint = root/'checkpoints/global_step_2/matched_identity.json'
            identity = assets.read(checkpoint); identity['roles']['critic'] = identity['roles']['actor']
            checkpoint.write_text(json.dumps(identity))
            (root/'audit/events-1.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
            with self.assertRaisesRegex(ValueError, 'Missing update/role'): evidence.complete(root, request, {})
            with (root/'metrics.jsonl').open('a') as f:
                f.write(json.dumps({'step': 1, 'metrics': {'critic/vf_loss': 0.0}})+'\n')
            with self.assertRaisesRegex(ValueError, 'Missing update/role'): evidence.complete(root, request, {})
            with (root/'metrics.jsonl').open('a') as f:
                f.write(json.dumps({'step': 2, 'metrics': {'critic/vf_loss': 0.0}})+'\n')
            self.assertEqual(len(evidence.complete(root, request, {})['learning_diagnostics']['losses_by_update']), 4)

    def test_reload_requires_new_provider_container_not_unique_hostname(self):
        p = plan(); job = p['jobs'][0]; request = c.stage_request(p, job, job['stages'][2])
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'AGL_READER_CALL_ID': 'fc-reload', 'MODAL_TASK_ID': 'ta-reload'}), patch.object(evidence.socket, 'gethostname', return_value='modal'):
            root = Path(tmp)/'checkpoint-eval'; (root/'audit').mkdir(parents=True)
            events = [{'event': 'wandb_finished', 'id': request['stage']['wandb']['id']},
                      {'event': 'trainer_returned', 'validation_only': True, 'pid': 123}]
            (root/'audit/events-1.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
            training = {'plan_sha256': request['plan_sha256'], 'checkpoint_sha256': 'a'*64,
                'worker': {'call_id': 'fc-training', 'hostname': 'modal', 'container_id': 'ta-training', 'trainer_pid': 123}}
            path = root.parent/'training/completion.json'; assets.write_new(path, training)
            proof = evidence.complete(root, request, {'checkpoint_sha256': 'a'*64})
            self.assertEqual(proof['worker']['hostname'], training['worker']['hostname'])
            self.assertNotEqual(proof['worker']['container_id'], training['worker']['container_id'])
            training['worker'].update(container_id='ta-reload', hostname='different-hostname')
            path.write_text(json.dumps(training))
            with self.assertRaisesRegex(ValueError, 'distinct worker'): evidence.complete(root, request, {'checkpoint_sha256': 'a'*64})
            for invalid in (None, '', 'modal', 'fc-notacontainer'):
                with self.assertRaises(ValueError): evidence.container_identity(invalid)

    def test_shared_checkpoint_inventory_rejects_missing_role_and_changed_bytes(self):
        p = plan(); j = p['jobs'][1]; request = c.stage_request(p, j, j['stages'][1])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); ckpt = root/'checkpoints/global_step_2'
            (root/'resolved_config.json').write_text('{}')
            for role in ('actor', 'critic'):
                (ckpt/role).mkdir(parents=True)
                for name in ('model_0.pt', 'optim_0.pt', 'extra_state_0.pt'):
                    (ckpt/role/name).write_bytes(b'synthetic')
            identity = matched_launch.checkpoint_identity(ckpt, request, root)
            assets.write_new(ckpt/'matched_identity.json', identity)
            (ckpt.parent/'latest_checkpointed_iteration.txt').write_text('2')
            matched_launch.verify_checkpoint(ckpt, request)
            (ckpt/'critic/model_0.pt').write_bytes(b'changed')
            with self.assertRaises(ValueError): matched_launch.verify_checkpoint(ckpt, request)
            (ckpt/'critic/model_0.pt').unlink()
            with self.assertRaisesRegex(ValueError, 'Incomplete'): matched_launch.checkpoint_identity(ckpt, request, root)


if __name__ == '__main__':
    unittest.main()
