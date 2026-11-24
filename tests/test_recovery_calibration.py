"""Recovery development calibration: plans, routing, completion rules, report and admission.

Offline and credential-free. Cloud boundaries are explicitly mocked; adapter events
here are simulated (the real adapter runs in test_matched_recovery_calibration).
"""
from collections import Counter
import copy
from contextlib import nullcontext
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiment import reader_contract as c, reader_assets as assets, reader_evidence as evidence, reader_modes as modes
from experiment import matched_contract as historical, matched_launch, matched_supervisor as supervisor
from experiment import reader_integration, reader_source, recovery_calibration as calibration, recovery_task as task
from scripts import reader_training as cli
import recovery_fixtures as fx
from test_reader_training import source, Registry, AssetsTests
from test_recovery_task import OpenRecorder

ROOT = Path(__file__).resolve().parents[1]
GRPO, PPO = c.ALGORITHMS
STAGES = ['greedy-development', 'sampled-packet-1', 'sampled-packet-2', 'sampled-packet-3']


def plan(**options):
    return fx.calibration_plan(source(), **options)


def all_correct(*_):
    return 'correct'


class ContractTests(unittest.TestCase):
    def test_plan_binds_panel_packets_modes_counts_seeds_and_zero_updates(self):
        p = plan(); c.validate(p)
        declared = p['calibration']
        self.assertEqual((p['phase'], declared['schema'], declared['split']), ('development-calibration', 'recovery-calibration-v1', 'development'))
        self.assertEqual((declared['total_episodes'], declared['episodes_per_algorithm'], declared['optimizer_updates']), (240, 120, 0))
        self.assertEqual(declared, calibration.declaration())
        panels = task.panels()
        self.assertEqual(declared['stages']['greedy-development']['case_ids'], panels['development']['case_ids'])
        self.assertEqual([declared['stages'][f'sampled-packet-{n}']['case_ids'] for n in (1, 2, 3)], task.packets())
        self.assertEqual(declared['final_evaluation']['panel_id'], panels['final']['panel_id'])
        final_ids = {r['case_id'] for r in task.rows('final')} | {r['case_id'] for r in task.rows('train')}
        self.assertEqual([j['algorithm'] for j in p['jobs']], [GRPO, PPO])
        self.assertEqual((p['bounds']['max_gpu_seconds'], p['bounds']['supervisor_seconds']), (8*1500, 8*1500+600))
        for job in p['jobs']:
            self.assertEqual([s['name'] for s in job['stages']], STAGES)
            # The established seed derivation is retained; no arbitrary seed or update count.
            self.assertEqual((job['run_seed'], job['seed_streams']), (9001, historical.streams(9001, job['algorithm'])))
            for stage in job['stages']:
                config = stage['wandb']['config']
                bound = declared['stages'][stage['name']]
                self.assertEqual((config['steps'], config['evaluation'], config['calibration']), (0, 'development-calibration', bound))
                self.assertFalse(set(bound['case_ids']) & final_ids)
                self.assertEqual(bound['split'], 'development')
                self.assertEqual((bound['mode'], bound['route'], bound['temperature'], bound['rollouts_per_question'], bound['expected_episodes']),
                                 ('greedy', 'val', 0, 1, 24) if stage['name'] == STAGES[0] else ('sampled', 'train', 1, 4, 32))
                request = c.stage_request(p, job, stage)
                cfg = matched_launch.configuration(request, Path('/model'), Path('/run'))
                greedy = stage['name'] == STAGES[0]
                self.assertEqual((cfg['trainer.val_only'], cfg['trainer.val_before_train']), (greedy, greedy))
                self.assertEqual((cfg['data.train_batch_size'], cfg['trainer.resume_mode']), (8, 'disable'))
                self.assertEqual(cfg['agentlightning.local.agent_class'], 'recovery_agent.Agent')
                self.assertEqual(cfg['agentlightning.trace_aggregator.trajectory_max_prompt_length'], 3072)
                self.assertEqual(cfg['agentlightning.trace_aggregator.trajectory_max_response_length'], 2048)
                self.assertEqual(cfg['actor_rollout_ref.rollout.max_model_len'], 4096)
                self.assertEqual(cfg['algorithm.adv_estimator'], 'grpo' if job['algorithm'] == GRPO else 'gae')
                self.assertEqual(cfg['critic.enable'], job['algorithm'] == PPO)
        ids = [s['wandb']['id'] for j in p['jobs'] for s in j['stages']]
        self.assertEqual(len(set(ids)), 8)
        self.assertEqual(p['task_identity']['episode_policy'], task.EPISODE_POLICY)
        self.assertEqual(p['task_identity']['context_limits'], task.CONTEXT_LIMITS)
        for edit in (lambda x: x['calibration']['stages']['sampled-packet-1']['case_ids'].reverse(),
                     lambda x: x['calibration']['stages']['sampled-packet-1'].update(case_ids=x['calibration']['stages']['sampled-packet-2']['case_ids']),
                     lambda x: x['calibration']['stages']['greedy-development'].update(case_ids=[r['case_id'] for r in task.rows('final')][:24]),
                     lambda x: x['calibration']['stages']['greedy-development'].update(temperature=1),
                     lambda x: x['calibration'].update(optimizer_updates=1), lambda x: x['calibration'].update(total_episodes=120),
                     lambda x: x['calibration']['criteria'].update(min_semantic_evidence_informative_groups=1),
                     lambda x: x['calibration']['final_evaluation'].update(model_use='allowed'),
                     lambda x: x['calibration'].update(panel_id='rr1-dev-0000000000000000'),
                     lambda x: x['jobs'][0]['stages'].pop(), lambda x: x['jobs'][0].update(run_seed=9002),
                     lambda x: x['jobs'][0]['stages'][1]['wandb']['config']['calibration'].update(packet_id='other'),
                     lambda x: x.update(phase='qualification'), lambda x: x['bounds'].update(max_gpu_seconds=1),
                     lambda x: x['task_identity']['context_limits'].update(trajectory_max_prompt_length=8192),
                     lambda x: x['task_identity']['episode_policy'].update(max_tool_iterations=9)):
            bad = copy.deepcopy(p); edit(bad)
            with self.assertRaises(ValueError): c.validate(bad)

    def test_profile_rules_and_closed_dispatch(self):
        base = plan()['profile']
        asset = assets.read(ROOT/'examples/reader-profile.asset.synthetic.json')
        calculator = assets.read(ROOT/'examples/reader-profile.synthetic.json')
        without = {k: v for k, v in base.items() if k != 'purpose'}
        for bad in (without, dict(base, purpose='collection-only'), dict(base, purpose='qualification'),
                    dict(asset, purpose='development-calibration'), dict(calculator, purpose='development-calibration'),
                    dict(base, task='recovery-readiness-v2'), dict(without, task='incident-simulator'),
                    dict(base, bounds=dict(base['bounds'], training_budget_usd=60)),  # six-stage budget does not cover eight
                    dict(base, bounds=dict(base['bounds'], concurrency=2))):
            with self.assertRaises(ValueError): c.validate_profile(bad)
        c.validate_profile(base)
        self.assertGreaterEqual(base['bounds']['training_budget_usd'], (16*1500+600)*10/3600)
        # Phase and stage dispatch is closed for reader-task-v2 and name-based for older schemas.
        p = plan()
        self.assertEqual([modes.kind(p, s) for s in STAGES], ['greedy', 'collection', 'collection', 'collection'])
        self.assertEqual([modes.route(p, s) for s in STAGES], ['val', 'train', 'train', 'train'])
        for name in ('training', 'baseline', 'checkpoint-eval', 'sampled-packet-4', 'final-evaluation'):
            with self.assertRaises(ValueError): modes.kind(p, name)
        for phase in ('pilot', 'campaign', 'final-evaluation', None):
            with self.assertRaises(ValueError): modes.phase(dict(p, phase=phase))
        with self.assertRaises(ValueError): modes.phase(dict(p, profile=dict(p['profile'], task='asset-status-v1')))
        qualification = c.make_plan(asset, source())
        with self.assertRaises(ValueError): modes.phase(dict(qualification, phase='development-calibration'))
        with self.assertRaises(ValueError): modes.kind(qualification, 'greedy-development')
        self.assertEqual([modes.kind(qualification, s) for s in historical.STAGES], list(historical.STAGES))
        for legacy in ({'schema': 'matched-rollout-mean-v2', 'phase': 'campaign'}, {'schema': 'reader-reference-v1'}, {}):
            self.assertIsNone(modes.phase(legacy))
            self.assertEqual((modes.kind(legacy, 'training'), modes.validation_only(legacy, 'training'), modes.collection_only(legacy, 'training')),
                             ('training', False, False))
            self.assertTrue(modes.validation_only(legacy, 'baseline'))
        with self.assertRaises(ValueError): c.task_module('calculator-v1')
        with self.assertRaises(ValueError): c.task_module('unknown-task')
        with self.assertRaises(ValueError): c.task_identity(source(), 'unknown-task')
        with self.assertRaises((ValueError, KeyError)): assets.validate_manifest({'schema': 'reader-assets-v1', 'fixture': False,
            'model': {'id': c.MODEL, 'revision': c.BASE_REVISION, 'parser': 'llama3_json', 'files': assets.model_lock(), 'manifest_sha256': 'a'*64},
            'data': {}, 'historical_match': {}}, 'unknown-task')
        with self.assertRaises(ValueError): assets.inspect('/missing', '/missing', task='calculator-v1', scope='development')
        request = c.stage_request(p, p['jobs'][0], p['jobs'][0]['stages'][0])
        with self.assertRaisesRegex(ValueError, 'Unknown task'):
            reader_integration.chains([], {**request, 'plan': {**p, 'task_identity': {**p['task_identity'], 'name': 'unknown'}}})
        with self.assertRaisesRegex(ValueError, 'Unowned stage'):
            c.stage_request(p, p['jobs'][0], {**p['jobs'][0]['stages'][0], 'name': 'training'})
        # The historical contract cannot admit a calibration plan.
        with self.assertRaises((ValueError, KeyError)): historical.validate(p)

    def test_existing_calculator_and_asset_plans_and_launch_configurations_are_unchanged(self):
        # Golden digests computed on unmodified main (daed9a5) with this fixed source identity.
        names = ('asset_task.py', 'asset_server.py', 'asset_stdio.py', 'asset_agent.py', 'asset_probe.py', 'owned_processes.py',
                 'observed_runtime.py', 'observed_calc_agent.py', 'launch.py', 'ppo_source_lock.json')
        files = {f'experiment/{n}': 'c'*64 for n in names}; files['reader_app.py'] = 'c'*64
        fixed = {'commit': 'b'*40, 'dirty': False, 'repository': 'amulyaparthasarathy/agent-lightning-grpo-ppo',
                 'baseline': dict(reader_source.BASELINE), 'files': files, 'runtime_files': dict(files)}
        asset = assets.read(ROOT/'examples/reader-profile.asset.synthetic.json')
        profiles = {'calculator': assets.read(ROOT/'examples/reader-profile.synthetic.json'), 'asset': asset,
                    'asset-collection-only': dict(asset, purpose='collection-only')}
        golden = {'calculator': '02bfbcbe4e51d4ef2df2f62c0492fb0e2066166e0faa2c1d1fabfacc37019f5e',
                  'asset': 'c0ca5fe76df1343c64f544c8a33f7a12feb9ff12f18dddbdcefa57d9e900ee04',
                  'asset-collection-only': '9a74ec6365053f783500652b5c273c72efd1c95f7c906b89be660aae246c7eab'}
        launches = {}
        for name, profile in profiles.items():
            p = c.make_plan(profile, fixed); c.validate(p)
            self.assertEqual(c.digest(p), golden[name], name)
            self.assertNotIn('calibration', p)
            for job in p['jobs']:
                for stage in job['stages']:
                    request = c.stage_request(p, job, stage)
                    configured, commands = matched_launch.make_plan(request, Path('/bundle'), Path('/repo'), Path('/data/model'),
                                                                    Path('/data/data'), Path('/python'), Path('/data')/stage['output'], 8000)
                    launches[f"{name}/{job['algorithm']}/{stage['name']}"] = (
                        c.digest({k: v for k, v in configured.items() if k != 'request'}) + ':' + c.digest(commands))
                    cfg = {k: json.loads(v) for k, v in (x.split('=', 1) for x in configured['overrides'])}
                    self.assertEqual(cfg['agentlightning.trace_aggregator.trajectory_max_prompt_length'], 2048)
        self.assertEqual(len(launches), 14)
        self.assertEqual(c.digest(launches), 'bc1b83a6417b1faeaea5fb60a84faf67efbf3fc6e70b561bbbe1a4e0861016c0')
        from experiment import asset_task
        self.assertEqual(asset_task.digest(asset_task.environment()), 'd683b4ca993f4d014de260fe8ba04afa901b1488be036a633c37b4fec7ef4089')
        self.assertEqual(asset_task.EPISODE_POLICY['version'], 'asset-episode-v2')


class RoutingTests(unittest.TestCase):
    def launch(self, request, data, run):
        return matched_launch.make_plan(request, Path('/bundle'), Path('/repo'), Path('/data/model'), data, Path('/python'), run, 8000)

    def resolved(self, configured, request):
        config = {}
        expected = matched_launch.configuration(request, Path(configured['model']), Path(configured['resolved_config']).parent)
        for key, value in {**{k: json.loads(v) for k, v in (x.split('=', 1) for x in configured['overrides'])}, **expected}.items():
            node = config
            for part in key.split('.')[:-1]:
                node = node.setdefault(part, {})
            node[key.split('.')[-1]] = value
        config['actor_rollout_ref'].setdefault('model', {})['path'] = configured['model']
        return config

    def test_both_dataset_arguments_route_to_development_and_final_records_are_never_opened(self):
        import pyarrow.parquet as pq
        p = plan()
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)/'data'; task.generate(data)
            closed = [data/n for n in task.FILES if task.SPLIT_OF_FILE[n] != 'development']
            for path in closed:
                path.chmod(0)  # unreadable sentinels: opening one raises PermissionError
            with self.assertRaises(PermissionError): (data/'final_eval.parquet').read_bytes()
            try:
                with OpenRecorder() as opened:
                    for job in p['jobs']:
                        for stage in job['stages']:
                            request = c.stage_request(p, job, stage)
                            run = Path(tmp)/job['algorithm']/stage['name']; run.mkdir(parents=True)
                            task.verify(data, scope='development')  # the model-facing asset check
                            configured, commands = self.launch(request, data, run)
                            declared = p['calibration']['stages'][stage['name']]
                            # Neither upstream argument keeps a launcher default; both are run-local.
                            self.assertEqual([configured['train_file'], configured['eval_file']],
                                             [str(run/'calibration-train.parquet'), str(run/'calibration-val.parquet')])
                            for value in (configured['train_file'], configured['eval_file']):
                                for forbidden in ('eval_main', 'train_main', 'final_eval', str(data)):
                                    self.assertNotIn(forbidden, value)
                            self.assertEqual(calibration.materialize(request, data, run, commands),
                                             [configured['train_file'], configured['eval_file']])
                            for path in (configured['train_file'], configured['eval_file']):
                                rows = pq.read_table(path).to_pylist()
                                self.assertEqual([r['case_id'] for r in rows], declared['case_ids'])
                                self.assertEqual({r['split'] for r in rows}, {'development'})
                            configured['resolved_config'] = str(run/'resolved_config.json')
                            matched_launch.validate_resolved(self.resolved(configured, request), configured)
                            # The upstream entry point's own loader, on exactly the two routed files.
                            for wrong in (str(data/'final_eval.parquet'), str(data/'train_main.parquet'), str(data/'dev_panel.parquet')):
                                for key in ('train_file', 'eval_file'):
                                    with self.assertRaisesRegex(ValueError, 'routing changed'):
                                        matched_launch.validate_resolved(self.resolved(configured, request), {**configured, key: wrong})
                seen = [path for path in opened if str(data) in path]
                self.assertTrue(any('dev_packet_3.parquet' in path for path in seen))
                self.assertFalse([path for path in seen if 'final_eval' in path or 'train_main' in path], seen)
            finally:
                for path in closed:
                    path.chmod(0o600)
            # A changed or swapped development file, or changed gateway temperatures, is rejected.
            request = c.stage_request(p, p['jobs'][0], p['jobs'][0]['stages'][1])
            run = Path(tmp)/'swap'; run.mkdir()
            configured, commands = self.launch(request, data, run)
            shutil.copyfile(data/'dev_packet_2.parquet', data/'dev_packet_1.parquet')
            with self.assertRaisesRegex(ValueError, 'identity, membership'): calibration.materialize(request, data, run, commands)
            shutil.copyfile(data/'final_eval.parquet', data/'dev_packet_1.parquet')
            with self.assertRaisesRegex(ValueError, 'identity, membership'): calibration.materialize(request, data, run, commands)
            with self.assertRaises(ValueError): task.verify(data, scope='development')
            hot = {**commands, 'gateway': [x.replace('val.temperature=0.0', 'val.temperature=0.7') for x in commands['gateway']]}
            with self.assertRaisesRegex(ValueError, 'temperatures'): calibration.materialize(request, data, run, hot)

    def test_worker_mode_selection_and_trace_route_follow_the_declared_stage_kind(self):
        p = plan()
        asset = c.make_plan(dict(assets.read(ROOT/'examples/reader-profile.asset.synthetic.json'), purpose='collection-only'), source())
        qualification = c.make_plan(assets.read(ROOT/'examples/reader-profile.asset.synthetic.json'), source())
        self.assertEqual([(modes.collection_only(p, s), modes.validation_only(p, s)) for s in STAGES],
                         [(False, True), (True, False), (True, False), (True, False)])
        self.assertEqual((modes.collection_only(asset, 'training'), modes.validation_only(asset, 'training')), (True, False))
        self.assertEqual([(modes.collection_only(qualification, s), modes.validation_only(qualification, s)) for s in historical.STAGES],
                         [(False, True), (False, False), (False, True)])
        self.assertIsNone(matched_launch.calibration(c.stage_request(asset, asset['jobs'][0], asset['jobs'][0]['stages'][0])))
        self.assertIs(matched_launch.calibration(c.stage_request(p, p['jobs'][0], p['jobs'][0]['stages'][0])), calibration)
        self.assertEqual(reader_integration.TOOLS['recovery-readiness-v1'], task.TOOLS)


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.plan = plan()

    def stage(self, name, algorithm=0):
        job = self.plan['jobs'][algorithm]
        stage = next(s for s in job['stages'] if s['name'] == name)
        return c.stage_request(self.plan, job, stage), self.plan['calibration']['stages'][name]

    def test_admitted_stages_and_retained_constant_groups(self):
        for algorithm in (0, 1):
            request, declared = self.stage('greedy-development', algorithm)
            episodes = fx.rows(declared, lambda case_id, sample: 'semantic' if case_id.endswith(('0', '1', '2', '3')) else 'correct')
            done = calibration.assemble(request, fx.events(declared, episodes), episodes)
            self.assertEqual((done['optimizer_updates'], len(done['calibration']['episodes'])), ({'actor': 0, 'critic': 0}, 24))
            self.assertNotIn('collection', done['calibration'])
            for behavior, constant in ((all_correct, 8), (lambda *_: 'semantic', 8), (lambda case_id, sample: ('correct', 'format')[sample % 2], 0)):
                request, declared = self.stage('sampled-packet-2', algorithm)
                episodes = fx.rows(declared, behavior)
                done = calibration.assemble(request, fx.events(declared, episodes), episodes)
                report = calibration.validate_stage(self.plan, self.plan['jobs'][algorithm], request['stage'], done)
                # Constant groups, including all-zero ones, are retained and counted.
                self.assertEqual((report['collection']['adapter_constant_groups'], len(report['episodes'])), (constant, 32))
                self.assertEqual(report['collection']['generated_tokens'], sum(e['sampled_tokens'] for e in report['episodes']))
                self.assertEqual(report['audit'], {'optimizer_step': 0, 'gradient_step': 0, 'matched_advantages': 0, 'native_checkpoint_saved': 0,
                                                   'weight_refresh_steps': [0], 'trainer_returned': 1, 'wandb_finished': 1,
                                                   'collection_manifest': 1, 'development_collection_admitted': 1})

    def test_optimization_checkpoint_refresh_and_mode_violations_are_rejected(self):
        for name in STAGES[:2]:
            request, declared = self.stage(name)
            episodes = fx.rows(declared, all_correct)
            good = fx.events(declared, episodes)
            calibration.assemble(request, good, episodes)
            edits = {'optimizer': lambda e: e.append({'event': 'optimizer_step', 'role': 'actor', 'count': 1}),
                     'gradient': lambda e: e.append({'event': 'gradient_step', 'role': 'actor'}),
                     'advantages': lambda e: e.append({'event': 'matched_advantages'}),
                     'checkpoint event': lambda e: e.append({'event': 'native_checkpoint_saved', 'step': 1}),
                     'second refresh': lambda e: e.append({'event': 'weight_refresh', 'step': 1}),
                     'no refresh': lambda e: e.remove(next(x for x in e if x['event'] == 'weight_refresh')),
                     'no return': lambda e: e.remove(next(x for x in e if x['event'] == 'trainer_returned')),
                     'two returns': lambda e: e.append(next(x for x in e if x['event'] == 'trainer_returned')),
                     'wrong step': lambda e: next(x for x in e if x['event'] == 'trainer_returned').update(global_step=2),
                     'wrong mode': lambda e: next(x for x in e if x['event'] == 'trainer_returned').update(
                         validation_only=name != STAGES[0], collection_only=name == STAGES[0])}
            for label, edit in edits.items():
                bad = copy.deepcopy(good); edit(bad)
                with self.subTest(stage=name, edit=label), self.assertRaises(ValueError):
                    calibration.assemble(request, bad, episodes)
            with self.assertRaises(ValueError): calibration.assemble(request, good, episodes, checkpoints=True)
        # A greedy stage must not contain a training collection, and a sampled stage needs one.
        request, declared = self.stage(STAGES[0])
        episodes = fx.rows(declared, all_correct)
        sampled_request, sampled = self.stage(STAGES[1])
        with self.assertRaises(ValueError):
            calibration.assemble(request, fx.events(declared, episodes) + [{'event': 'collection_manifest', 'trajectories': []}], episodes)
        sampled_rows = fx.rows(sampled, all_correct)
        with self.assertRaises(ValueError): calibration.assemble(sampled_request, fx.events(declared, episodes), sampled_rows)

    def test_missing_duplicate_misrouted_and_token_evidence_is_rejected_without_filling(self):
        request, declared = self.stage('sampled-packet-1')
        other = self.plan['calibration']['stages']['sampled-packet-2']
        episodes = fx.rows(declared, lambda case_id, sample: ('correct', 'semantic', 'format', 'correct')[sample])
        good = fx.events(declared, episodes)
        self.assertEqual(len(calibration.assemble(request, good, episodes)['calibration']['episodes']), 32)

        def manifest(events):
            return next(e for e in events if e['event'] == 'collection_manifest')['trajectories']

        def measured(events):
            return next(e for e in events if e['event'] == 'calibration_collection_tokens')

        row_edits = {'missing reward row': lambda r: r.pop(),
                     'duplicate rollout id': lambda r: r[1].update(rollout_id=r[0]['rollout_id']),
                     'duplicate trace': lambda r: r[1].update(trace_id=r[0]['trace_id']),
                     'wrong packet': lambda r: r.__setitem__(slice(None), fx.rows(other, all_correct)),
                     'one case from another packet': lambda r: r[0].update(case_id=other['case_ids'][0]),
                     'altered counts': lambda r: r[4].update(case_id=r[0]['case_id']),
                     'missing reward': lambda r: r[0].update(reward=None),
                     'unposted reward': lambda r: r[0].update(reward_posted=False),
                     'nonbinary reward': lambda r: r[0].update(reward=0.5),
                     'validation route': lambda r: r[0].update(route='val'),
                     'mislabelled family': lambda r: r[0].update(family='empty_history')}
        for label, edit in row_edits.items():
            bad = copy.deepcopy(episodes); edit(bad)
            with self.subTest(rows=label), self.assertRaises((ValueError, TypeError)):
                calibration.assemble(request, good, bad)
        event_edits = {'reward differs from trace': lambda e: manifest(e)[0].update(reward=1.0 - manifest(e)[0]['reward']),
                       'cross-source rollout': lambda e: manifest(e)[0].update(rollout='another-run'),
                       'case differs': lambda e: manifest(e)[0].update(case_id=declared['case_ids'][-1]),
                       'duplicate manifest rollout': lambda e: manifest(e).append(dict(manifest(e)[0])),
                       'missing manifest rollout': lambda e: manifest(e).pop(),
                       'group spans two cases': lambda e: manifest(e)[0].update(group=manifest(e)[-1]['group']),
                       'token loss': lambda e: measured(e)['rollouts'][0].update(mask_tokens=measured(e)['rollouts'][0]['mask_tokens'] - 1),
                       'token total differs': lambda e: next(x for x in e if x['event'] == 'development_collection_admitted').update(generated_tokens=1),
                       'truncated row': lambda e: measured(e).update(truncated_rows=1),
                       'dropped prompt': lambda e: measured(e).update(dropped_prompt_rows=1),
                       'unmeasured rollout': lambda e: measured(e)['rollouts'].pop(),
                       'incomplete collection': lambda e: next(x for x in e if x['event'] == 'development_collection_admitted').update(real_rollouts=31),
                       'seven groups': lambda e: next(x for x in e if x['event'] == 'development_collection_admitted').update(groups=7),
                       'no token evidence': lambda e: e.remove(measured(e)),
                       'two collections': lambda e: e.append(copy.deepcopy(next(x for x in e if x['event'] == 'collection_manifest')))}
        for label, edit in event_edits.items():
            bad = copy.deepcopy(good); edit(bad)
            with self.subTest(events=label), self.assertRaises(ValueError):
                calibration.assemble(request, bad, episodes)
        # A stale or different plan cannot admit evidence.
        stale = copy.deepcopy(request); stale['plan']['calibration']['stages']['sampled-packet-1']['case_ids'].reverse()
        with self.assertRaises(ValueError): calibration.assemble(stale, good, episodes)
        done = calibration.assemble(request, good, episodes)
        for edit in (lambda d: d['calibration']['episodes'].pop(), lambda d: d['calibration'].update(packet_id='other'),
                     lambda d: d['calibration']['audit'].update(optimizer_step=1), lambda d: d.update(optimizer_updates={'actor': 1, 'critic': 0}),
                     lambda d: d['calibration']['audit'].update(weight_refresh_steps=[0, 1]),
                     lambda d: d['calibration']['episodes'][0].update(mask_tokens=1), lambda d: d['calibration'].update(algorithm=PPO),
                     lambda d: d['calibration'].update(evidence_class='simulated-trace'), lambda d: d['calibration']['collection'].update(truncated_rows=2)):
            bad = copy.deepcopy(done); edit(bad)
            with self.assertRaises((ValueError, KeyError)):
                calibration.validate_stage(self.plan, self.plan['jobs'][0], request['stage'], bad)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.plan = plan()

    def report(self, behavior, **options):
        return calibration.report(self.plan, fx.proofs(self.plan, behavior, **options))

    def test_separate_diagnostics_denominators_and_group_classification(self):
        result = self.report(fx.informative)
        self.assertEqual((result['schema'], result['evidence_class'], result['live_policy_measurement']), ('recovery-calibration-report-v1', 'live-model', True))
        for algorithm in (GRPO, PPO):
            r = result['routes'][algorithm]
            self.assertEqual(r['episodes'], {'expected': 120, 'attempted_and_admitted': 120, 'complete': 120, 'not_admitted': 0,
                                             'rewards_confirmed': 120, 'rewards_missing': 0, 'note': r['episodes']['note']})
            s, g, groups = r['sampled'], r['greedy'], r['sampled']['groups']
            self.assertEqual((g['episodes'], s['episodes'], groups['groups']), (24, 96, 24))
            # Format and action failures are not counted as known semantic answers.
            self.assertEqual(s['decision_assessable'] + s['semantics_not_assessable'], 96)
            self.assertEqual(s['semantics_not_assessable'], s['failure_classes'].get('format', 0) + s['failure_classes'].get('action', 0))
            self.assertEqual(s['contract_format_valid'], 96 - s['failure_classes'].get('format', 0))
            # Decision and evidence correctness are reported separately.
            self.assertEqual(s['decision_correct'], s['failure_classes']['correct'] + s['failure_classes']['evidence'])
            self.assertEqual(s['decision_assessable'] - s['decision_correct'], s['failure_classes']['semantic'])
            self.assertEqual(s['decision_correct_evidence_invalid'], s['failure_classes']['evidence'])
            self.assertEqual(s['reward_sum'], s['failure_classes']['correct'])
            self.assertEqual(sum(v['episodes'] for v in s['by_family'].values()), 96)
            self.assertEqual(groups['constant_zero'] + groups['constant_one'] + groups['mixed'], 24)
            self.assertEqual(groups['mixed'], groups['mixed_semantic_evidence'] + groups['mixed_format_action_only'])
            self.assertEqual(len(s['per_question']), 24)
            self.assertTrue(all(len(q['rewards']) == 4 == len(q['failure_classes']) for q in s['per_question']))
            self.assertEqual(Counter(q['group_type'] for q in s['per_question'])['mixed_semantic_evidence'], groups['mixed_semantic_evidence'])
            self.assertEqual(sorted(s['by_packet']), STAGES[1:])
            self.assertEqual(sum(p['groups'] for p in s['by_packet'].values()), 24)
            self.assertEqual(s['tokens']['sampled_tokens'], s['tokens']['mask_tokens'])
            self.assertEqual(s['recovered_episodes'], s['outcomes'].get('complete_recovered', 0))
            self.assertEqual(r['optimization_events'], {'optimizer_step': 0, 'gradient_step': 0, 'matched_advantages': 0, 'native_checkpoint_saved': 0})
            self.assertEqual(set(r['weight_refresh_steps_by_stage'].values().__iter__().__next__()), {0})
            d = r['decision']
            self.assertEqual((d['operational_validity'], d['scientific_suitability']), ('valid', 'screen_passed'))
            self.assertGreaterEqual(d['semantic_evidence_informative_groups'], 6)
            self.assertGreaterEqual(len(d['informative_case_families']), 2)
            self.assertTrue(d['greedy_successes_and_failures_both_observed'])
        text = calibration.summary(result)
        self.assertIn('95% Wilson', text); self.assertIn('engineering heuristic', text); self.assertNotIn('Not a live policy measurement', text)
        self.assertEqual(calibration.wilson(0, 0), None)
        self.assertEqual(calibration.wilson(12, 24), [0.3143, 0.6857])  # z=1.96, checked by hand
        low, high = calibration.wilson(24, 24)
        self.assertTrue(0.85 < low < 1 == high)

    def test_negative_screens_are_valid_results_and_routes_are_never_pooled(self):
        def isolated_format(algorithm, stage, case_id, sample):
            return 'format' if stage == 'sampled-packet-1' and case_id == self.plan['calibration']['stages'][stage]['case_ids'][0] and sample == 0 else 'correct'

        def interface(algorithm, stage, case_id, sample):
            return ('format', 'action', 'correct', 'semantic')[sample] if task.family(case_id)[0] != 'empty_history' else 'format'

        def five_groups(algorithm, stage, case_id, sample):
            # Five informative groups, nine constant-zero groups, ten constant-one groups.
            position = self.plan['calibration']['stages']['greedy-development']['case_ids'].index(case_id)
            if stage == 'greedy-development':
                return 'correct'
            return 'semantic' if (position < 5 and sample == 0) or 5 <= position < 14 else 'correct'

        def one_family(algorithm, stage, case_id, sample):
            return 'semantic' if task.family(case_id)[0] == 'recent_usable' and sample == 0 else 'correct'
        expected = {'saturated': all_correct, 'too_difficult': lambda *_: 'semantic', 'interface_dominated': interface}
        for verdict, behavior in expected.items():
            for r in self.report(behavior)['routes'].values():
                self.assertEqual((r['decision']['operational_validity'], r['decision']['scientific_suitability']), ('valid', verdict))
                self.assertIn('do not repeat an unchanged collection', r['decision']['next_step'])
        # One formatting failure makes a mixed group but not a semantically informative one.
        r = self.report(isolated_format)['routes'][GRPO]
        self.assertEqual((r['sampled']['groups']['mixed'], r['sampled']['groups']['mixed_format_action_only'],
                          r['decision']['semantic_evidence_informative_groups'], r['decision']['scientific_suitability']),
                         (1, 1, 0, 'interface_dominated'))
        self.assertEqual(self.report(five_groups)['routes'][GRPO]['decision']['scientific_suitability'], 'insufficient_semantic_variation')
        r = self.report(one_family)['routes'][GRPO]['decision']
        self.assertEqual((r['semantic_evidence_informative_groups'], r['informative_case_families'], r['scientific_suitability']),
                         (6, ['recent_usable'], 'insufficient_semantic_variation'))
        # A passing route cannot rescue a failing one.
        mixed = self.report(lambda algorithm, *rest: fx.informative(algorithm, *rest) if algorithm == GRPO else 'correct')
        self.assertEqual({a: r['decision']['scientific_suitability'] for a, r in mixed['routes'].items()},
                         {GRPO: 'screen_passed', PPO: 'saturated'})

    def test_incomplete_routes_are_not_admitted_zero_filled_or_interpreted(self):
        result = self.report(fx.informative, skip={(PPO, 'sampled-packet-3')})
        ppo, grpo = result['routes'][PPO], result['routes'][GRPO]
        self.assertEqual((ppo['episodes']['attempted_and_admitted'], ppo['episodes']['not_admitted'], ppo['episodes']['rewards_missing']), (88, 32, 32))
        self.assertEqual((ppo['sampled']['episodes'], ppo['sampled']['groups']['groups']), (64, 16))  # nothing is padded to 24 groups
        self.assertEqual(ppo['decision'], {'operational_validity': 'invalid', 'scientific_suitability': 'not_assessed', 'reason': ppo['decision']['reason']})
        self.assertEqual(ppo['stage_problems'], ['sampled-packet-3: no admitted completion'])
        self.assertEqual(grpo['decision']['operational_validity'], 'valid')
        # A tampered stage proof is a stage problem, not a silently accepted count.
        proofs = fx.proofs(self.plan, fx.informative)
        proofs[GRPO]['sampled-packet-1']['calibration']['episodes'].pop()
        proofs[GRPO]['greedy-development']['calibration']['audit']['optimizer_step'] = 1
        broken = calibration.report(self.plan, proofs)['routes'][GRPO]
        self.assertEqual((len(broken['stage_problems']), broken['episodes']['attempted_and_admitted'], broken['decision']['operational_validity']), (2, 64, 'invalid'))
        with self.assertRaises(ValueError): calibration.report(dict(self.plan, phase='qualification'), proofs)

    def test_scripted_or_simulated_evidence_is_labelled_and_rejected_as_live(self):
        result = self.report(fx.informative, evidence_class='scripted-inference')
        self.assertEqual((result['evidence_class'], result['live_policy_measurement']), ('scripted-inference', False))
        for r in result['routes'].values():
            self.assertEqual((r['decision']['operational_validity'], r['decision']['scientific_suitability']), ('mechanics_only_not_live', 'not_assessed'))
            self.assertIn('never a policy measurement', r['decision']['interpretation'])
        self.assertIn('**Not a live policy measurement.**', calibration.summary(result))
        mixed = fx.proofs(self.plan, fx.informative)
        mixed[PPO]['sampled-packet-2']['calibration']['evidence_class'] = 'scripted-inference'
        self.assertEqual(calibration.report(self.plan, mixed)['evidence_class'], 'scripted-inference')
        # Durable collection: the same joins as qualification, then the calibration report.
        admitted = fx.calibration_plan(source(), fixture=False)
        index, files = fx.durable(admitted, fx.proofs(admitted, fx.informative))
        collected = evidence.validate_collection(admitted, index, files.__getitem__)
        self.assertEqual((collected['evidence_complete'], collected['qualification_evidence']), (True, False))
        self.assertIn('zero optimization', collected['live_qualification'])
        self.assertEqual(collected['calibration']['routes'][GRPO]['decision']['scientific_suitability'], 'screen_passed')
        self.assertEqual(collected['calibration']['stage_timings_seconds'][f'{GRPO}/greedy-development'], 360.0)
        self.assertEqual(collected['algorithms'][GRPO]['integration_rule'], 'all-calibration-stages-v1')
        self.assertNotIn('stage_proofs', collected['algorithms'][GRPO])
        index, files = fx.durable(admitted, fx.proofs(admitted, fx.informative, evidence_class='scripted-inference'))
        with self.assertRaisesRegex(ValueError, 'not a complete live collection'):
            evidence.validate_collection(admitted, index, files.__getitem__)
        registry = Registry({'index:'+supervisor.key(admitted): index})
        status = cli.status(admitted, registry, files.__getitem__)
        self.assertFalse(status['evidence_complete']); self.assertIn('not a complete live collection', status['incomplete_reason'])
        index, files = fx.durable(admitted, fx.proofs(admitted, fx.informative))
        for edit in (lambda f, s: f[s['output']+'/completion.json']['calibration']['episodes'].pop(),
                     lambda f, s: f[s['output']+'/completion.json'].update(optimizer_updates={'actor': 1, 'critic': 0}),
                     lambda f, s: f[s['output']+'/completion.json'].update(task_sha256='0'*64),
                     lambda f, s: f[s['output']+'/completion.json'].pop('calibration')):
            bad_files = copy.deepcopy(files); edit(bad_files, admitted['jobs'][1]['stages'][2])
            with self.assertRaises((ValueError, KeyError)): evidence.validate_collection(admitted, index, bad_files.__getitem__)
        short = copy.deepcopy(index); short['jobs'][admitted['jobs'][0]['job_id']]['stages'].pop()
        with self.assertRaisesRegex(ValueError, 'Missing/duplicate stage receipts'): evidence.validate_collection(admitted, short, files.__getitem__)


class OperationTests(unittest.TestCase):
    def test_supervisor_runs_eight_stages_in_one_lane_and_stops_a_job_at_its_first_failure(self):
        p = plan(); now = [0]; order = []; saved = []
        def clock(): return now[0]
        def sleep(seconds): now[0] += seconds
        def dispatch(job, stage):
            order.append((job['algorithm'], stage['name'])); return f'fc-{len(order)}'
        def result(call):
            job, stage = order[int(call.split('-')[1])-1]
            return {'status': 'completed', 'call_id': call, 'output': next(s['output'] for j in p['jobs'] if j['algorithm'] == job
                    for s in j['stages'] if s['name'] == stage), 'completion': {'plan_sha256': c.digest(p), 'stage': stage}}
        index = supervisor.run(p, dispatch, result, Mock(), lambda _: None, lambda: False, saved.append, clock=clock, sleep=sleep)
        self.assertEqual(index['status'], 'completed')
        self.assertEqual(order, [(a, s) for a in (GRPO, PPO) for s in STAGES])
        self.assertNotIn('dependency_gate', index['jobs'][p['jobs'][0]['job_id']])  # the baseline gate is a qualification rule
        order.clear()
        def failing(call):
            job, stage = order[int(call.split('-')[1])-1]
            if (job, stage) == (GRPO, 'sampled-packet-1'):
                raise RuntimeError('collection rejected')
            return result(call)
        index = supervisor.run(p, dispatch, failing, Mock(), lambda _: None, lambda: False, saved.append, clock=clock, sleep=sleep)
        self.assertEqual(index['status'], 'failed')
        self.assertEqual(order, [(GRPO, STAGES[0]), (GRPO, STAGES[1]), *[(PPO, s) for s in STAGES]])
        self.assertEqual([s['status'] for s in index['jobs'][p['jobs'][0]['job_id']]['stages']], ['completed', 'failed', 'skipped', 'skipped'])
        # An uncertain spawn is never retried and blocks everything after it.
        order.clear()
        index = supervisor.run(p, Mock(side_effect=TimeoutError()), Mock(), Mock(), lambda _: None, lambda: False, saved.append, clock=clock, sleep=sleep)
        self.assertEqual(index['status'], 'unknown')

    def test_mocked_provider_dispatch_requires_same_plan_preflight_and_reserves_once(self):
        p = fx.calibration_plan(source(), fixture=False)
        registry = Registry()
        receipt = {'status': 'completed', 'plan_sha256': c.digest(p), 'result': {'status': 'passed'}}
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
            with self.assertRaisesRegex(ValueError, 'preflight'): cli.online(args, p, p['profile'])
            app.supervise.with_options.assert_not_called()
            registry['result:preflight:'+p['root']] = receipt
            result = cli.online(args, p, p['profile'])
            self.assertEqual((result['status'], result['call_id'], result['automatic_retry']), ('submitted', 'fc-mocked', False))
            app.supervise.with_options.assert_called_once_with(timeout=p['bounds']['supervisor_seconds'])
            app.supervise.with_options.return_value.spawn.assert_called_once_with(p, image_id='im-synthetic')
            # The reservation is durable: no second dispatch for this registry or this plan.
            args = SimpleNamespace(command='launch', out=Path(tmp)/'retry.json')
            with self.assertRaisesRegex(ValueError, 'already reserved'): cli.online(args, p, p['profile'])
            self.assertEqual(app.supervise.with_options.return_value.spawn.call_count, 1)
        # Fixture profiles never cross the online boundary.
        with self.assertRaisesRegex(ValueError, 'fixture'):
            cli.online(SimpleNamespace(command='launch', out=None), plan(), plan()['profile'])

    def test_preparation_generates_and_verifies_recovery_data_without_other_acquisition(self):
        p = plan()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); seed = root/'model-fixture'
            lock = AssetsTests().model_fixture(seed)
            calls = []
            def acquire(command, *_):
                calls.append(command)
                if not str(command[1]).endswith('prepare_model.py'):
                    raise AssertionError('Unexpected calculator acquisition')
                shutil.copytree(seed, command[-1])
            with patch.object(assets, 'model_lock', return_value=lock):
                result = assets.prepare(p, root, acquire, root, {})
                assets.validate_manifest(result, task.TASK)
                self.assertEqual(len(calls), 1)
                self.assertEqual(result['data'], task.verify(root/p['profile']['paths']['data']))
                self.assertEqual(result['historical_match'], assets.synthetic_claim(task.TASK))
                for other in ('calculator-v1', 'asset-status-v1'):
                    with self.assertRaises(ValueError): assets.validate_manifest(result, other)
                bound = c.make_plan(p['profile'], source(), result); c.validate(bound)
                self.assertEqual(bound['data_manifest']['files']['final_eval.parquet']['rows'], 64)
                self.assertFalse(bound['training_admissible'])  # synthetic fixture profile
                # The model-facing stage check verifies development files and returns the same manifest.
                model, data = root/p['profile']['paths']['model'], root/p['profile']['paths']['data']
                for name in ('final_eval.parquet', 'final_eval.jsonl', 'train_main.parquet'):
                    (data/name).chmod(0)
                try:
                    self.assertEqual(assets.inspect(model, data, task=task.TASK, scope='development'), result)
                    with self.assertRaises(PermissionError): assets.inspect(model, data, task=task.TASK)
                finally:
                    for name in ('final_eval.parquet', 'final_eval.jsonl', 'train_main.parquet'):
                        (data/name).chmod(0o600)

    def test_new_runtime_files_are_admitted_and_package_drift_is_rejected(self):
        new = {'experiment/'+n for n in ('reader_modes.py', 'recovery_agent.py', 'recovery_calibration.py', 'recovery_probe.py',
                                         'recovery_prompt.py', 'recovery_server.py', 'recovery_solver.py', 'recovery_stdio.py', 'recovery_task.py')}
        inventory = json.loads((ROOT/'experiment/reader_runtime.json').read_text())
        self.assertTrue(new <= set(inventory))
        self.assertEqual(inventory, sorted(set(inventory)))
        # Every packaged experiment module is declared; nothing is uploaded implicitly.
        modules = {'experiment/'+p.name for p in (ROOT/'experiment').glob('*.py')}
        self.assertEqual(modules - set(inventory), set())
        admitted = c.source_identity()  # real Git admission of this checkout
        self.assertTrue(new <= set(admitted['runtime_files']))
        identity = task.identity(admitted)
        self.assertEqual(set(identity['source_sha256']), {'recovery_task.py', 'recovery_server.py', 'recovery_stdio.py', 'recovery_agent.py',
                         'recovery_probe.py', 'recovery_solver.py', 'recovery_calibration.py', 'recovery_prompt.py', 'reader_modes.py',
                         'owned_processes.py', 'observed_runtime.py'})
        changed = copy.deepcopy(admitted); changed['files']['experiment/recovery_task.py'] = 'f'*64
        self.assertNotEqual(c.digest(task.identity(changed)), c.digest(identity))
        with tempfile.TemporaryDirectory() as tmp:
            package = Path(tmp)
            for name in admitted['runtime_files']:
                (package/name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT/name, package/name)
            expected = admitted['runtime_files']
            reader_source.verify_package(package, expected, c.file_hash)
            target = package/'experiment/recovery_calibration.py'
            original = target.read_bytes()
            target.write_bytes(original + b'\n# changed\n')
            with self.assertRaisesRegex(ValueError, 'changed'): reader_source.verify_package(package, expected, c.file_hash)
            target.unlink()
            with self.assertRaisesRegex(ValueError, 'Missing'): reader_source.verify_package(package, expected, c.file_hash)
            target.write_bytes(original)
            (package/'experiment/recovery_extra.py').write_text('extra')
            with self.assertRaisesRegex(ValueError, 'Undeclared'): reader_source.verify_package(package, expected, c.file_hash)
            (package/'experiment/recovery_extra.py').unlink()
            (package/'tests').mkdir(); (package/'tests/recovery_harness_support.py').write_text('test-only')
            with self.assertRaisesRegex(ValueError, 'Undeclared'): reader_source.verify_package(package, expected, c.file_hash)
        # Test-only scripted inference is never part of the image.
        self.assertFalse([n for n in inventory if n.startswith('tests/')])


if __name__ == '__main__':
    unittest.main()
