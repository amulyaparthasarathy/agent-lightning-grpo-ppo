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
                     'no wandb finish': lambda e: e.remove(next(x for x in e if x['event'] == 'wandb_finished')),
                     'two wandb finishes': lambda e: e.append(next(x for x in e if x['event'] == 'wandb_finished')),
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
        # Server-reported completion tokens, when present, must equal the adapter's sampled tokens.
        reported = copy.deepcopy(episodes)
        self.assertTrue(all(None not in e['completion_tokens'] for e in reported))
        reported[3]['completion_tokens'][0] += 1
        with self.assertRaises(ValueError): calibration.assemble(request, good, reported)
        # Where usage was not reported the comparison cannot be made, and the row says so.
        reported = copy.deepcopy(episodes)
        reported[3]['completion_tokens'] = [None] * reported[3]['llm_calls']
        done = calibration.assemble(request, good, reported)
        self.assertEqual(sum(None in e['completion_tokens'] for e in done['calibration']['episodes']), 1)
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
                     lambda d: d['calibration'].update(seed_streams=self.plan['jobs'][1]['seed_streams']),
                     lambda d: d['calibration']['audit'].update(trainer_returned=0), lambda d: d['calibration']['audit'].update(wandb_finished=0),
                     lambda d: d['calibration']['episodes'][0].update(evidence_class='scripted-inference'),
                     lambda d: d['calibration']['collection'].update(rollouts_with_token_evidence=31),
                     lambda d: d['calibration'].update(evidence_class='simulated-trace'), lambda d: d['calibration']['collection'].update(truncated_rows=2)):
            bad = copy.deepcopy(done); edit(bad)
            with self.assertRaises((ValueError, KeyError)):
                calibration.validate_stage(self.plan, self.plan['jobs'][0], request['stage'], bad)


class ReportTests(unittest.TestCase):
    """Report arithmetic on SIMULATED admitted reviews (fx.reviews bypasses admission).

    Admission itself, with production-format traces, receipts and audit events, is
    tested in test_recovery_evidence. Summary-only evidence is tested below.
    """
    def setUp(self):
        self.plan = fx.executable_plan(source())

    def report(self, behavior, **options):
        return calibration.report(self.plan, fx.reviews(self.plan, behavior, **options))

    def test_separate_diagnostics_denominators_and_group_classification(self):
        result = self.report(fx.informative)
        self.assertEqual((result['schema'], result['evidence_class'], result['live_policy_measurement']), ('recovery-calibration-report-v1', 'live-model', True))
        for algorithm in (GRPO, PPO):
            r = result['routes'][algorithm]
            a = r['accounting']
            self.assertEqual({k: a[k] for k in ('expected', 'admitted', 'observed_attempts', 'completed', 'rewards_confirmed',
                                                'rewards_missing', 'rejected_untrusted', 'unattempted', 'unknown')},
                             {'expected': 120, 'admitted': 120, 'observed_attempts': 120, 'completed': 120, 'rewards_confirmed': 120,
                              'rewards_missing': 0, 'rejected_untrusted': 0, 'unattempted': 0, 'unknown': 0})
            self.assertEqual({s['packet_status'] for s in r['stages'].values()}, {'admitted'})
            self.assertEqual(r['failure_diagnostics'], {})
            s, g, groups = r['results']['sampled'], r['results']['greedy'], r['results']['sampled']['groups']
            self.assertEqual((g['episodes'], s['episodes'], groups['groups']), (24, 96, 24))
            # Format and action failures are not counted as known semantic answers.
            self.assertEqual(s['decision_assessable'] + s['decision_not_assessable'], 96)
            self.assertEqual(s['decision_not_assessable'], s['failure_classes'].get('format', 0) + s['failure_classes'].get('action', 0))
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
            self.assertEqual(sorted(s['by_packet']), STAGES[1:])
            self.assertEqual(sum(p['groups'] for p in s['by_packet'].values()), 24)
            # Token and truncation numbers are measured values with their coverage, never defaults.
            for name in STAGES[1:]:
                t = r['stages'][name]['token_evidence']
                self.assertEqual((t['rollouts_expected'], t['rollouts_with_token_evidence'], t['truncated_rows'], t['dropped_prompt_rows']), (32, 32, 0, 0))
                self.assertEqual((t['sampled_tokens'], t['adapter_evidence']), (t['mask_tokens'], 'measured by the collection adapter'))
            self.assertEqual(r['stages'][STAGES[0]]['token_evidence']['adapter_evidence'], 'not applicable to a greedy stage')
            self.assertEqual(s['recovered_episodes'], s['outcomes'].get('complete_recovered', 0))
            self.assertEqual(s['unknown_tool_names_as_sampled'], {})
            self.assertEqual(r['optimization_events'], {'optimizer_step': 0, 'gradient_step': 0, 'matched_advantages': 0, 'native_checkpoint_saved': 0})
            self.assertEqual(r['optimization_events_coverage'], '4 of 4 stages (admitted stages only)')
            d = r['decision']
            self.assertEqual((d['operational_validity'], d['scientific_suitability'], d['reasons']), ('valid', 'screen_passed', []))
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
        # One formatting failure makes a mixed group but not a semantically informative one,
        # and 95 of 96 correct is saturation, not an interface problem.
        d = self.report(isolated_format)['routes'][GRPO]
        groups = d['results']['sampled']['groups']
        self.assertEqual((groups['mixed'], groups['mixed_format_action_only'], d['decision']['semantic_evidence_informative_groups'],
                          d['decision']['interface_share_of_zero_rewards'], d['decision']['scientific_suitability']), (1, 1, 0, 1.0, 'saturated'))
        self.assertEqual(self.report(five_groups)['routes'][GRPO]['decision']['scientific_suitability'], 'insufficient_semantic_variation')
        d = self.report(one_family)['routes'][GRPO]['decision']
        self.assertEqual((d['semantic_evidence_informative_groups'], d['informative_case_families'], d['scientific_suitability']),
                         (6, ['recent_usable'], 'insufficient_semantic_variation'))
        # A passing route cannot rescue a failing one.
        mixed = self.report(lambda algorithm, *rest: fx.informative(algorithm, *rest) if algorithm == GRPO else 'correct')
        self.assertEqual({a: r['decision']['scientific_suitability'] for a, r in mixed['routes'].items()},
                         {GRPO: 'screen_passed', PPO: 'saturated'})

    def test_incomplete_routes_keep_diagnostics_and_get_no_scientific_decision(self):
        result = self.report(fx.informative, skip={(PPO, 'sampled-packet-3')})
        ppo, grpo = result['routes'][PPO], result['routes'][GRPO]
        self.assertIsNone(ppo['results'])  # nothing partial is presented as a result, padded or zero-filled
        self.assertEqual({k: ppo['accounting'][k] for k in ('expected', 'admitted', 'observed_attempts', 'rewards_confirmed', 'rewards_missing', 'unattempted', 'unknown')},
                         {'expected': 120, 'admitted': 88, 'observed_attempts': 88, 'rewards_confirmed': 88, 'rewards_missing': 0,
                          'unattempted': 0, 'unknown': 32})  # no local evidence is unknown, never "missing" and never "unattempted"
        self.assertEqual(ppo['accounting']['stages_with_trace_evidence'], 3)
        self.assertEqual((ppo['decision']['operational_validity'], ppo['decision']['scientific_suitability'], ppo['decision']['reasons']),
                         ('invalid', 'not_assessed', ['sampled-packet-3: packet unknown']))
        self.assertNotIn('semantic_evidence_informative_groups', ppo['decision'])
        lost = ppo['failure_diagnostics']['sampled-packet-3']
        self.assertEqual((lost['packet_status'], lost['episodes']['unknown'], lost['episodes']['observed_attempts'], lost['token_evidence']['truncated_rows']),
                         ('unknown', 32, None, None))
        self.assertEqual((grpo['decision']['operational_validity'], result['live_policy_measurement'], result['evidence_class']), ('valid', False, 'live-model'))
        text = calibration.summary(result)
        self.assertIn('**Not a live policy measurement.**', text); self.assertIn('32 unknown', text)
        self.assertIn('Diagnostic, sampled-packet-3 (unknown): unknown attempts', text)
        with self.assertRaises(ValueError): calibration.report(dict(self.plan, phase='qualification'), fx.reviews(self.plan, fx.informative))
        with self.assertRaises(ValueError): calibration.report(self.plan, {GRPO: [], PPO: []})

    def test_scripted_mixed_and_plan_level_evidence_never_becomes_live(self):
        result = self.report(fx.informative, evidence_class='scripted-inference')
        self.assertEqual((result['evidence_class'], result['live_policy_measurement']), ('scripted-inference', False))
        for r in result['routes'].values():
            d = r['decision']
            self.assertEqual((d['operational_validity'], d['scientific_suitability']), ('mechanics_only_not_live', 'not_assessed'))
            self.assertEqual(d['screen_on_unadmitted_or_non_live_evidence'], 'screen_passed')  # shown, never a decision
            self.assertIn('never a policy measurement', d['interpretation'])
            self.assertIn('none:', d['next_step'])
        self.assertIn('**Not a live policy measurement.**', calibration.summary(result))
        # Evidence class is decided per route and never borrowed, in either job order.
        for scripted_route, live_route in ((GRPO, PPO), (PPO, GRPO)):
            reviews = fx.reviews(self.plan, fx.informative)
            reviews[scripted_route] = fx.reviews(self.plan, fx.informative, evidence_class='scripted-inference')[scripted_route]
            result = calibration.report(self.plan, reviews)
            self.assertEqual((result['evidence_class'], result['live_policy_measurement']), ('mixed', False))
            self.assertEqual((result['routes'][live_route]['evidence_class'], result['routes'][live_route]['decision']['operational_validity'],
                              result['routes'][live_route]['decision']['scientific_suitability']), ('live-model', 'valid', 'screen_passed'))
            self.assertEqual((result['routes'][scripted_route]['evidence_class'], result['routes'][scripted_route]['decision']['operational_validity'],
                              result['routes'][scripted_route]['decision']['scientific_suitability']),
                             ('scripted-inference', 'mechanics_only_not_live', 'not_assessed'))
        # One scripted packet makes its own route mixed and unassessed.
        reviews = fx.reviews(self.plan, fx.informative)
        reviews[PPO][2] = fx.reviews(self.plan, fx.informative, evidence_class='scripted-inference')[PPO][2]
        result = calibration.report(self.plan, reviews)
        self.assertEqual((result['routes'][PPO]['evidence_class'], result['routes'][PPO]['decision']['operational_validity'],
                          result['routes'][PPO]['decision']['scientific_suitability'], result['routes'][GRPO]['evidence_class']),
                         ('mixed', 'mechanics_only_not_live', 'not_assessed', 'live-model'))
        # A plan that could not have executed, missing usage, or an over-limit prompt is never valid.
        reviews = fx.reviews(self.plan, fx.informative)
        problem = calibration.report(self.plan, reviews, plan_problems=['the plan is not executable'])
        self.assertEqual({r['decision']['operational_validity'] for r in problem['routes'].values()}, {'invalid'})
        self.assertFalse(problem['live_policy_measurement'])
        for edit, reason in ((lambda e: e.update(completion_tokens=[None] * e['llm_calls']), 'no reported usage'),
                             # One generation of one episode without a served prompt count is enough.
                             (lambda e: e['prompt_tokens'].__setitem__(0, None), 'no reported usage'),
                             (lambda e: e['prompt_tokens'].__setitem__(0, 0), 'no reported usage'),
                             (lambda e: e.update(prompt_tokens=[4000] * e['llm_calls']), 'exceeds the declared trajectory prompt limit')):
            reviews = fx.reviews(self.plan, fx.informative)
            edit(reviews[GRPO][0]['rows'][0])
            d = calibration.report(self.plan, reviews)['routes'][GRPO]['decision']
            self.assertEqual((d['operational_validity'], d['scientific_suitability']), ('invalid', 'not_assessed'))
            self.assertIn(reason, d['reasons'][0])

    def test_summary_only_evidence_is_explicitly_unverified_whatever_its_fields_say(self):
        # SIMULATED completion summaries with consistent receipts, but no traces or audit files.
        admitted = self.plan
        completions = fx.proofs(admitted, fx.informative)
        index, files = fx.durable(admitted, completions)
        self.assertEqual({p['calibration']['evidence_class'] for route in completions.values() for p in route.values()}, {'live-model'})
        for reviewed in (evidence.validate_collection(admitted, index, files.__getitem__),
                         evidence.calibration_collection(admitted, index, files.__getitem__)):
            report = reviewed['calibration']
            self.assertEqual((reviewed['evidence_complete'], report['live_policy_measurement'], report['evidence_class']), (False, False, 'none admitted'))
            self.assertIn('Traces and audit events are required', reviewed['incomplete_reason'])
            for r in report['routes'].values():
                self.assertIsNone(r['results'])
                self.assertEqual((r['decision']['operational_validity'], r['decision']['scientific_suitability']), ('invalid', 'not_assessed'))
                self.assertEqual({k: r['accounting'][k] for k in ('expected', 'admitted', 'unknown', 'unattempted')},
                                 {'expected': 120, 'admitted': 0, 'unknown': 120, 'unattempted': 0})
                for s in r['stages'].values():
                    self.assertEqual((s['packet_status'], s['checks']['completion_summary'], s['checks']['durable_identity'], s['checks']['traces'],
                                      s['checks']['recomputed_from_traces_and_audit']), ('rejected', 'valid', 'verified', 'absent locally', False))
                    self.assertIn('supporting traces or audit events are not available', s['problems'][0])
                    self.assertIsNone(s['episodes']['observed_attempts'])
        # Without a supervisor index nothing is even identity-checked.
        report = evidence.calibration_collection(admitted, None, files.__getitem__)['calibration']
        self.assertIn('not verified: no supervisor index was supplied', report['provenance'])
        self.assertEqual({s['checks']['durable_identity'] for r in report['routes'].values() for s in r['stages'].values()},
                         {'not checked: no supervisor index or receipts were available'})
        self.assertFalse(report['live_policy_measurement'])
        # Wrong identities, schema or completion events are named, and change nothing about admission.
        stage = admitted['jobs'][1]['stages'][2]
        edits = {'plan hash': lambda p: p.update(plan_sha256='0'*64), 'source hash': lambda p: p.update(source_sha256='0'*64),
                 'configuration hash': lambda p: p.update(configuration_sha256='0'*64), 'task hash': lambda p: p.update(task_sha256='0'*64),
                 'assets hash': lambda p: p.update(assets_sha256='0'*64), 'schema': lambda p: p.update(execution_schema='reader-reference-v1'),
                 'wandb': lambda p: p.update(wandb_id='other')}
        for label, edit in edits.items():
            bad_index, bad_files = copy.deepcopy((index, files))
            proof = bad_files[stage['output']+'/completion.json']; edit(proof)
            bad_files[stage['output']+'/receipt.json']['completion'] = proof
            bad_index['jobs'][admitted['jobs'][1]['job_id']]['stages'][2]['result'] = bad_files[stage['output']+'/receipt.json']
            s = evidence.calibration_collection(admitted, bad_index, bad_files.__getitem__)['calibration']['routes'][PPO]['stages'][stage['name']]
            self.assertEqual((s['checks']['durable_identity'], s['packet_status']), ('failed', 'rejected'), label)
            self.assertIn('durable identity: ValueError: Durable receipt/completion/observability identity mismatch', s['problems'])
        for label, edit in {'no trainer return': lambda p: p['calibration']['audit'].update(trainer_returned=0),
                            'no wandb finish': lambda p: p['calibration']['audit'].update(wandb_finished=0),
                            'optimizer event': lambda p: p['calibration']['audit'].update(optimizer_step=1),
                            'missing episode': lambda p: p['calibration']['episodes'].pop(),
                            'no calibration summary': lambda p: p.pop('calibration')}.items():
            bad_index, bad_files = copy.deepcopy((index, files))
            edit(bad_files[stage['output']+'/completion.json'])
            s = evidence.calibration_collection(admitted, bad_index, bad_files.__getitem__)['calibration']['routes'][PPO]['stages'][stage['name']]
            self.assertEqual((s['checks']['completion_summary'], s['packet_status']), ('invalid', 'rejected'), label)
        # The plan is validated on every path; a non-executable plan is a stated problem.
        with self.assertRaises(ValueError): evidence.calibration_collection(dict(admitted, configuration_sha256='0'*64), index, files.__getitem__)
        with self.assertRaises(ValueError): evidence.calibration_collection(c.make_plan(assets.read(ROOT/'examples/reader-profile.asset.synthetic.json'), source()), None, files.__getitem__)
        fixture_plan = plan()
        fixture_index, fixture_files = fx.durable(fixture_plan, fx.proofs(fixture_plan, fx.informative))
        problems = evidence.calibration_collection(fixture_plan, fixture_index, fixture_files.__getitem__)['calibration']['plan_problems']
        self.assertEqual(len(problems), 1); self.assertIn('not executable', problems[0])
        # The status command shows the same review and never reports completeness.
        registry = Registry({'index:'+supervisor.key(admitted): index})
        status = cli.status(admitted, registry, files.__getitem__)
        self.assertFalse(status['evidence_complete'])
        self.assertEqual(status['calibration']['routes'][GRPO]['decision']['scientific_suitability'], 'not_assessed')
        # Calculator and asset collections keep their strict raising behavior.
        with self.assertRaisesRegex(ValueError, 'Incomplete or mismatched supervisor index'):
            evidence.validate_collection(c.make_plan(assets.read(ROOT/'examples/reader-profile.asset.synthetic.json'), source()),
                                         {'plan': {}, 'status': 'failed', 'jobs': {}}, files.__getitem__)


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
