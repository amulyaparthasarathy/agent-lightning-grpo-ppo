"""Offline ports, contract tests and honest mocks; no Torch, inference or Modal RPC."""
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from types import SimpleNamespace
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import Mock, patch

from experiment import launch, ppo_contract as c, ppo_math as m, ppo_runtime as runtime
import modal_app as runner

ROOT = Path(__file__).resolve().parents[1]


def flat(rows, masks):
    return [v for row, mask in zip(rows, masks, strict=True) for v, valid in zip(row, mask, strict=True) if valid]


def terminal(values, masks, rewards):
    result = []
    for vs, mask, reward in zip(values, masks, rewards, strict=True):
        row = [0.0]*len(vs)
        row[max(i for i, valid in enumerate(mask) if valid)] = reward
        result.append(row)
    return result


def checkpoint_fixture():
    lock = json.loads((ROOT/'experiment/ppo_source_lock.json').read_text())
    return {**c.identity(c.PPO), 'complete': True, 'step': 2, 'base_revision': c.BASE_REVISION,
            'resolved_config_sha256': 'c'*64, 'source_identity': {'lock_sha256': c.digest_json(lock)},
            'bundle_source_sha256': {name: hashlib.sha256((ROOT/'experiment'/name).read_bytes()).hexdigest()
                                    for name in ('launch.py', 'observed_calc_agent.py', 'ppo_contract.py',
                                                 'ppo_runtime.py', 'ppo_source_lock.json')},
            'roles': {role: {name: {'bytes': 1, 'sha256': hashlib.sha256(b'x').hexdigest()}
                             for name in ('model_world_size_1_rank_0.pt', 'optim_world_size_1_rank_0.pt',
                                          'extra_state_world_size_1_rank_0.pt')}
                      for role in ('actor', 'critic')}}


class EstimatorTests(unittest.TestCase):
    def assertCloseList(self, a, b, tol=1e-10):
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b, strict=True):
            self.assertAlmostEqual(x, y, delta=tol)

    def fixture(self, reward):
        # Same actual prefix at each generated position. Observation 90 is context.
        merged = {'prefix': [10, 11], 'tokens': [20, 21, 90, 22, 23, 0],
                  'mask': [1, 1, 0, 1, 1, 0], 'values': [.1, .2, 17, .4, .7, 81]}
        fragments = [
            {'prefix': [10, 11], 'tokens': [20, 21, 0], 'mask': [1, 1, 0], 'values': [.1, .2, 81]},
            {'prefix': [10, 11, 20, 21, 90], 'tokens': [22, 23, 0], 'mask': [1, 1, 0], 'values': [.4, .7, 81]}]
        def contexts(rows):
            return [(tuple(r['prefix']+r['tokens'][:i]), len(r['prefix'])+i, tok)
                    for r in rows for i, (tok, mask) in enumerate(zip(r['tokens'], r['mask'])) if mask]
        self.assertEqual(contexts([merged]), contexts(fragments))
        def compute(rows, lam=1):
            values, masks = [r['values'] for r in rows], [r['mask'] for r in rows]
            rewards = terminal(values, masks, [reward]*len(rows))
            return [flat(x, masks) for x in m.gae(rewards, values, masks, lam=lam)]
        return merged, fragments, compute

    def test_context_preserving_fragment_equivalence_zero_and_positive(self):
        for reward in (0., 1.):
            merged, fragments, compute = self.fixture(reward)
            for x, y in zip(compute([merged]), compute(fragments), strict=True):
                self.assertCloseList(x, y)
            self.assertCloseList(compute([merged])[1], [reward]*4)

    def test_lambda_less_than_one_is_not_representation_invariant(self):
        merged, fragments, compute = self.fixture(1.)
        self.assertNotEqual(compute([merged], .95)[0], compute(fragments, .95)[0])

    def test_changed_context_and_value_invalidates_equivalence(self):
        merged, fragments, compute = self.fixture(1.)
        fragments[1]['prefix'] = [99]  # a stitched/replaced prompt is not the original context
        fragments[1]['values'][0] = .9
        self.assertNotEqual(compute([merged])[0], compute(fragments)[0])

    def test_masked_observation_values_have_no_effect_on_valid_gae(self):
        masks = [[1, 0, 1, 0]]
        first = m.gae([[0, 0, 1, 0]], [[.2, 1, .4, 2]], masks)
        second = m.gae([[0, 0, 1, 0]], [[.2, -10000, .4, 90000]], masks)
        for a, b in zip(first, second):
            self.assertCloseList(flat(a, masks), flat(b, masks))

    def test_all_failure_nonconstant_values_raw_signs_and_whitening_scale(self):
        raw, returns, adv = m.gae([[0, 0]], [[.2, .8]], [[1, 1]])
        self.assertCloseList(raw[0], [-.2, -.8])
        self.assertCloseList(returns[0], [0, 0])
        self.assertGreater(adv[0][0], 0); self.assertLess(adv[0][1], 0)
        self.assertAlmostEqual(statistics.variance(adv[0]), .18/(.18+1e-8))

    def test_constant_values_center_to_zero_including_fp_residuals(self):
        raw, returns, adv = m.gae([[0]*5], [[.7]*5], [[1]*5])
        self.assertLess(max(map(abs, adv[0])), 1e-10)
        tiny = m.whiten([[1, 1+1e-12]], [[1, 1]])[0]
        self.assertLess(statistics.variance(tiny), 1e-12)

    def test_all_success_mixed_and_different_constant_question_rewards(self):
        for rewards in ([1, 1], [0, 1], [0, 0]):
            vs, masks = [[.1, .8], [.3, .2]], [[1, 1], [1, 1]]
            raw, ret, adv = m.gae(terminal(vs, masks, rewards), vs, masks)
            self.assertCloseList(flat(ret, masks), [r for r in rewards for _ in range(2)])
            self.assertAlmostEqual(statistics.mean(flat(adv, masks)), 0, places=12)
        rows = [{'group': g, 'rollout': f'{g}{i}', 'reward': r}
                for g, r in [('a', 0), ('b', 1)] for i in range(4)]
        self.assertEqual(set(m.grpo_diagnostic(rows).values()), {0.0})
        raw, ret, adv = m.gae([[0, 0], [0, 1]], [[.5, .5]]*2, [[1, 1]]*2)
        self.assertLess(adv[0][0], 0); self.assertGreater(adv[1][0], 0)

    def test_whitening_requires_two_generated_tokens(self):
        for mask in ([[0, 0]], [[1, 0]]):
            with self.assertRaises(ValueError):
                m.whiten([[.1, .2]], mask)

    def test_grpo_is_rollout_level_and_cannot_mutate_ppo(self):
        rows = [{'group': 'q', 'rollout': 'a', 'reward': 0, 'advantages': [.3, -.3]}]*3
        rows += [{'group': 'q', 'rollout': 'b', 'reward': 1, 'advantages': [.7]}]
        old = copy.deepcopy(rows)
        diag = m.grpo_diagnostic(rows)
        self.assertEqual(rows, old)
        self.assertAlmostEqual(diag['a'], -.5/(math.sqrt(.5)+1e-6))
        self.assertAlmostEqual(diag['a'], -diag['b'])
        with self.assertRaises(ValueError):
            m.grpo_diagnostic(rows+[{'group': 'q', 'rollout': 'a', 'reward': 1}])

    def test_ev_constant_wrong_predictor_is_invalid_despite_raw_one(self):
        metrics = m.prediction_metrics([.8, .8], [0, 0])
        self.assertEqual(metrics['vf_explained_var_raw'], 1)
        self.assertFalse(metrics['ev_valid'])
        self.assertIsNone(metrics['vf_explained_var_qualified'])
        self.assertAlmostEqual(metrics['mse'], .64)
        self.assertAlmostEqual(metrics['bias'], .8)

    def test_prediction_baseline_and_unbounded_values(self):
        metrics = m.prediction_metrics([-.2, 1.4], [0., 1.], earlier_constant=.25)
        self.assertTrue(metrics['ev_valid'])
        self.assertEqual(metrics['out_of_unit_interval'], 2)
        self.assertAlmostEqual(metrics['earlier_constant_mse'], .3125)
        self.assertIsNone(m.prediction_metrics([.3, .5], [0, 1])['earlier_constant_mse'])

    def test_summaries_keep_token_rollout_and_group_weighting_distinct(self):
        rows = [{'rollout': 'a', 'group': 'q', 'reward': 0}, {'rollout': 'b', 'group': 'r', 'reward': 1}]
        out = m.stage_summaries([[0, 0, 0], [1]], [[1, 1, 1], [1]], rows)
        self.assertEqual(out['token_weighted']['mean'], .25)
        self.assertEqual(out['rollout_weighted']['mean'], .5)
        self.assertEqual(out['prompt_group_weighted']['mean'], .5)


class ReductionTests(unittest.TestCase):
    def test_actor_normalization_does_not_modify_critic_targets(self):
        raw, returns, adv = m.gae([[0, 1], [0, 0]], [[.2, .4], [.6, .8]], [[1, 1]]*2)
        before = copy.deepcopy((raw, returns, adv))
        effective, coeff = m.normalize_actor(adv, [[1, 1]]*2, ['a', 'b'])
        self.assertEqual((raw, returns, adv), before)
        self.assertEqual(coeff, [.25, .25])
        self.assertNotEqual(effective, adv)

    def test_actor_sum_reduction_gradient_accumulation_and_row_count_dependence(self):
        # A has four tokens, B one. Fragment A unevenly: same raw advantages/token loss derivatives.
        merged, masks, ids = [[1, 2, 3, 4], [8]], [[1]*4, [1]], ['a', 'b']
        split, smasks, sids = [[1], [2, 3, 4], [8]], [[1], [1]*3, [1]], ['a', 'a', 'b']
        a, _ = m.normalize_actor(merged, masks, ids)
        b, _ = m.normalize_actor(split, smasks, sids)
        # pinned actor loss sums tokens, engine accumulates backward with NO divide by microbatches.
        grad_a, grad_b = -sum(flat(a, masks)), -sum(flat(b, smasks))
        self.assertAlmostEqual(grad_a/grad_b, 3/2)
        self.assertAlmostEqual(sum(-sum(row) for row in b), grad_b)
        # Finite-difference at ratio=1, away from clipping boundary.
        loss = lambda theta: -math.exp(theta)*sum(flat(b, smasks))
        self.assertAlmostEqual((loss(1e-6)-loss(-1e-6))/2e-6, grad_b, places=8)
        self.assertNotAlmostEqual(grad_b/len(b), grad_b)

    def test_critic_microbatch_means_duplicate_fragment_weight_and_unequal_lengths(self):
        # At old=current prediction, clip inactive: d(0.5*(v-R)^2)/dv=v-R.
        merged_errors = [[1, 2, 3, 4], [8]]
        split_errors = [[1], [2, 3, 4], [8]]
        merged_gradient = sum(statistics.mean(row) for row in merged_errors)
        split_gradient = sum(statistics.mean(row) for row in split_errors)
        self.assertEqual(merged_gradient, 10.5)
        self.assertEqual(split_gradient, 12)
        self.assertNotEqual(split_gradient, statistics.mean([1, 2, 3, 4, 8]))
        self.assertNotEqual(split_gradient, merged_gradient)


class ConfigurationTests(unittest.TestCase):
    def plan(self, algorithm='grpo', **changes):
        args = SimpleNamespace(phase='smoke', profile='llama3b', hooks=None, algorithm=algorithm,
                               protocol=None)
        for key, value in changes.items():
            setattr(args, key, value)
        plan, commands = launch.make_plan(args, ROOT/'experiment', Path('/opt/agent-lightning'),
            Path('/data/restored/models/llama3b'), Path('/data/restored/data-small'), Path('/opt/python'),
            Path('/data/phase2/runs/ppo-mc-v1-test'), None, 10000)
        config = {k: json.loads(v) for k, v in (item.split('=', 1) for item in plan['overrides'])}
        return plan, config

    def test_default_grpo_does_not_install_ppo_or_change_legacy_overrides(self):
        plan, config = self.plan()
        self.assertEqual(plan['algorithm'], 'grpo')
        self.assertNotIn('critic.enable', config)
        self.assertNotIn('algorithm.adv_estimator', config)
        self.assertNotIn('ray_kwargs.ray_init.runtime_env.worker_process_setup_hook', config)
        req = runner.request('smoke', 'legacy-smoke', wandb_entity='team')
        cmd = runner.launch_command(req)
        self.assertNotIn('--algorithm', cmd)
        self.assertEqual(cmd[cmd.index('--model-path')+1], '/data/models/llama3b')

    def test_ppo_mc_fixed_contract_effective_microbatches_and_actor_recipe(self):
        plan, cfg = self.plan(c.PPO)
        self.assertEqual(plan['protocol'], c.PROTOCOL)
        self.assertEqual((cfg['algorithm.gamma'], cfg['algorithm.lam']), (1., 1.))
        self.assertFalse(cfg['algorithm.enable_rollout_level_advantage'])
        self.assertEqual(cfg['critic.model.path'], '/data/restored/models/llama3b')
        self.assertEqual(cfg['critic.fsdp.micro_batch_size_per_gpu'], 1)
        self.assertEqual(cfg['critic.fsdp.infer_micro_batch_size_per_gpu'], 1)
        self.assertFalse(cfg['critic.fsdp.use_dynamic_bsz'])
        self.assertEqual(cfg['critic.optim.lr'], 1e-5)
        self.assertEqual(cfg['critic.ppo_epochs'], 1)
        self.assertEqual(cfg['trainer.critic_warmup'], 0)
        self.assertEqual(cfg['actor_rollout_ref.actor.optim.lr'], 1e-6)
        self.assertEqual(cfg['trainer.max_critic_ckpt_to_keep'], 2)
        self.assertFalse(cfg['trainer.del_local_ckpt_after_load'])

    def test_ppo_reload_is_validation_only_and_keeps_both_native_load_contents(self):
        _, cfg = self.plan(c.PPO, phase='smoke-reload-eval')
        self.assertTrue(cfg['trainer.val_only'])
        self.assertEqual(cfg['critic.checkpoint.load_contents'], ['model', 'optimizer', 'extra'])
        self.assertEqual(cfg['trainer.total_training_steps'], 2)

    def test_resolved_config_cannot_silently_change_estimator_or_paths(self):
        _, flat_cfg = self.plan(c.PPO)
        flat_cfg['actor_rollout_ref.model.path'] = flat_cfg['critic.model.path']
        config = {}
        for key, value in flat_cfg.items():
            node = config
            parts = key.split('.')
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        c.validate_resolved(config, '/data/restored/models/llama3b')
        for key, value in [('lam', .95), ('gamma', .99)]:
            changed = copy.deepcopy(config); changed['algorithm'][key] = value
            with self.assertRaises(ValueError):
                c.validate_resolved(changed, '/data/restored/models/llama3b')
        changed = copy.deepcopy(config); changed['critic']['model']['path'] = '~/models/deepseek-llm-7b-chat'
        with self.assertRaises(ValueError):
            c.validate_resolved(changed, '/data/restored/models/llama3b')

    def test_invalid_selection_refused_before_spawn(self):
        invalid = [{'algorithm': 'ppo'}, {'algorithm': c.PPO, 'protocol': c.LEGACY_PROTOCOL},
                   {'algorithm': c.PPO, 'action': 'main'}, {'algorithm': c.PPO, 'model_profile': 'reference-qwen'},
                   {'algorithm': c.PPO, 'data_layout': 'root'}, {'algorithm': c.PPO, 'name': 'train-01'}]
        for change in invalid:
            kwargs = dict(action='smoke', name='ppo-mc-v1-test', wandb_entity='team', data_layout='restored')
            kwargs.update(change)
            with self.subTest(change=change), patch.object(runner.gpu_phase, 'spawn') as spawn:
                with self.assertRaises(ValueError):
                    runner.main(**kwargs)
                spawn.assert_not_called()

    def test_ppo_no_freeform_hooks(self):
        with self.assertRaises(ValueError):
            self.plan(c.PPO, hooks=ROOT/'experiment/ppo_math.py')

    def test_volume_binding_and_no_original_volume_dispatch(self):
        req = runner.request('smoke', 'ppo-mc-v1-test', wandb_entity='team', algorithm=c.PPO, data_layout='restored')
        cmd = runner.launch_command(req)
        self.assertEqual(cmd[cmd.index('--model-path')+1], '/data/restored/models/llama3b')
        self.assertEqual(cmd[cmd.index('--data-dir')+1], '/data/restored/data-small')
        self.assertEqual(cmd[cmd.index('--output-root')+1], '/data/phase2/runs')
        with (patch.object(runner, 'WORKING_VOLUME', ''), patch.object(runner.gpu_phase, 'spawn') as spawn,
              patch.object(runner.sys, 'argv', ['modal', '--detach'])):
            with self.assertRaisesRegex(ValueError, 'AGL_WORKING_VOLUME'):
                runner.main(action='smoke', name='ppo-mc-v1-test', wandb_entity='team', algorithm=c.PPO, data_layout='restored')
            spawn.assert_not_called()

    def test_missing_restored_assets_never_fall_back_to_root_or_start_commands(self):
        req = runner.request('smoke', 'ppo-mc-v1-test', wandb_entity='team', algorithm=c.PPO, data_layout='restored')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'models/llama3b').mkdir(parents=True)
            (root/'models/llama3b/model_manifest.json').write_text('{}')
            with patch.object(runner, 'STATE', root), patch.object(runner, 'run_command') as command:
                with self.assertRaisesRegex(ValueError, 'Missing restored'):
                    runner.train_or_check(req, root/'receipt')
                command.assert_not_called()

    def test_restored_input_directories_and_manifests_fail_closed(self):
        req = runner.request('smoke', 'ppo-mc-v1-test', wandb_entity='team', algorithm=c.PPO, data_layout='restored')
        directories = ('models/llama3b', 'data-small')
        manifests = ('models/llama3b/model_manifest.json', 'data-small/data_manifest.json')
        for relative in directories + manifests:
            for kind in ('missing', 'wrong-type'):
                with self.subTest(path=relative, kind=kind), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    # Valid legacy paths must never mask a broken restored layout.
                    for name in manifests:
                        legacy = root / name
                        legacy.parent.mkdir(parents=True, exist_ok=True)
                        legacy.write_text('{}')
                    for name in directories:
                        target = root/'restored'/name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if name != relative:
                            target.mkdir()
                        elif kind == 'wrong-type':
                            target.write_text('owner-reported directory-download artifact')
                    for name in manifests:
                        target = root/'restored'/name
                        if not target.parent.is_dir():
                            continue
                        if name != relative:
                            target.write_text('{}')
                        elif kind == 'wrong-type':
                            target.mkdir()
                    before = {str(p.relative_to(root)): p.read_bytes()
                              for p in root.rglob('*') if p.is_file()}
                    with patch.object(runner, 'STATE', root), patch.object(runner, 'run_command') as command:
                        with self.assertRaisesRegex(ValueError, 'Missing restored') as error:
                            runner.train_or_check(req, root/'receipt')
                        self.assertIn(str(root/'restored'/relative), str(error.exception))
                        command.assert_not_called()  # includes nvidia-smi and check_gpu.py
                    self.assertEqual(before, {str(p.relative_to(root)): p.read_bytes()
                                             for p in root.rglob('*') if p.is_file()})

    def test_restored_manifest_gate_allows_complete_layout_offline(self):
        req = runner.request('smoke', 'ppo-mc-v1-test', wandb_entity='team', algorithm=c.PPO, data_layout='restored')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('models/llama3b/model_manifest.json', 'data-small/data_manifest.json'):
                target = root/'restored'/name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('{}')  # content/identity is validated later by launch.py
            result = root/'phase2/runs'/req['name']/'result.json'
            result.parent.mkdir(parents=True)
            result.write_text('{"fixture": true}')
            with patch.object(runner, 'STATE', root), patch.object(runner, 'run_command') as command:
                self.assertEqual(runner.train_or_check(req, root/'receipt'), {'fixture': True})
                self.assertEqual(command.call_count, 3)
                for call in command.call_args_list:
                    self.assertEqual(call.args[2]['HF_HUB_OFFLINE'], '1')
                    self.assertEqual(call.args[2]['TRANSFORMERS_OFFLINE'], '1')

    def test_critic_head_loading_contract_rejects_backbone_missing_and_fallback(self):
        info = {'missing_keys': ['score.weight', 'score.bias'], 'unexpected_keys': ['lm_head.weight']}
        c.validate_loading_info(info)
        for changed in ({**info, 'missing_keys': ['score.weight']},
                        {**info, 'unexpected_keys': ['model.layers.0.foo']},
                        {**info, 'mismatched_keys': ['score.weight']}):
            with self.assertRaises(ValueError):
                c.validate_loading_info(changed)
        source = (ROOT/'experiment/ppo_runtime.py').read_text()
        self.assertIn('torch.random.fork_rng()', source)
        self.assertIn('torch.manual_seed(contract.CRITIC_SEED)', source)
        self.assertIn('local_files_only=True', source)
        self.assertNotIn('from trl', source)

    def test_worker_and_overlay_parse_without_importing_training_stack(self):
        ast.parse(launch.WORKER)
        ast.parse((ROOT/'experiment/ppo_runtime.py').read_text())
        self.assertLess(launch.WORKER.index('runtime_assertions()'), launch.WORKER.index('train(train_file='))

    def test_runtime_environment_preserves_upstream_defaults_without_mutation(self):
        defaults = {'env_vars': {'NCCL_DEBUG': 'WARN', 'VLLM_DISABLE_COMPILE_CACHE': '1'}}
        before = copy.deepcopy(defaults)
        args = ['ray_kwargs.ray_init.runtime_env.env_vars={"HF_HUB_OFFLINE":"1"}']
        result = c.runtime_env_overrides(args, defaults)
        self.assertEqual(defaults, before)
        self.assertEqual(len(args), 1)
        merged = json.loads(result[-1].split('=', 1)[1])
        self.assertEqual(merged, {**defaults['env_vars'], 'HF_HUB_OFFLINE': '1'})

    def test_ppo_detach_and_submission_receipt_are_not_completion(self):
        kwargs = dict(action='smoke', name='ppo-mc-v1-submit', wandb_entity='team',
                      algorithm=c.PPO, data_layout='restored')
        with patch.object(runner.sys, 'argv', ['modal']), patch.object(runner.gpu_phase, 'spawn') as spawn:
            with self.assertRaisesRegex(ValueError, 'detach'):
                runner.main(**kwargs)
            spawn.assert_not_called()
        with tempfile.TemporaryDirectory() as tmp:
            with (patch.object(runner, 'LOCAL', Path(tmp)), patch.object(runner, 'WORKING_VOLUME', 'working-fixture'),
                  patch.object(runner.sys, 'argv', ['modal', '--detach']), patch.object(runner, 'sha256', return_value='f'*64),
                  patch.object(runner.subprocess, 'check_output', side_effect=['b'*40, '', '']),
                  patch.object(runner.gpu_phase, 'with_options', return_value=runner.gpu_phase) as options,
                  patch.object(runner.gpu_phase, 'spawn', return_value=SimpleNamespace(object_id='fc-fixture')) as spawn,
                  redirect_stdout(StringIO()) as output):
                runner.main(**kwargs)
            options.assert_called_once_with(memory=196608)
            spawn.assert_called_once()
            receipts = list((Path(tmp)/'.local/experiment-submissions').glob('*.json'))
            self.assertEqual(len(receipts), 1)
            receipt = json.loads(receipts[0].read_text())
            self.assertEqual(receipt['status'], 'submitted_not_validated')
            self.assertEqual(receipt['request']['algorithm'], c.PPO)
            self.assertIn('No later phase is queued', output.getvalue())

    def test_grpo_keeps_static_resources(self):
        tree = ast.parse((ROOT/'modal_app.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'gpu_phase')
        decorator = fn.decorator_list[0]
        values = {k.arg: ast.literal_eval(k.value) for k in decorator.keywords
                  if k.arg in {'gpu', 'cpu', 'memory', 'timeout', 'retries', 'max_containers', 'single_use_containers'}}
        self.assertEqual(values, dict(gpu='H100!', cpu=16, memory=131072, timeout=14400,
                                      retries=0, max_containers=1, single_use_containers=True))


class MemoryDispatchTests(unittest.TestCase):
    def dispatch(self, action, algorithm=c.GRPO):
        gpu, prepare, collect, variant = (Mock() for _ in range(4))
        gpu.with_options.return_value = variant
        for function in (gpu, prepare, collect, variant):
            function.spawn.return_value = SimpleNamespace(object_id='fc-resource-fixture')
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = dict(action=action, name='ppo-mc-v1-resource-fixture' if algorithm == c.PPO else 'resource-fixture',
                          algorithm=algorithm, wandb_entity='team',
                          data_layout='restored' if algorithm == c.PPO else 'root')
            if action == 'smoke-reload-eval':
                kwargs['checkpoint'] = ('phase2/' if algorithm == c.PPO else '') + 'runs/ppo-mc-v1-smoke/checkpoints/global_step_2'
                if algorithm == c.PPO:
                    identity = Path(tmp)/'identity.json'
                    identity.write_text(json.dumps(checkpoint_fixture()))
                    kwargs['checkpoint_identity'] = str(identity)
            if action == 'summarize':
                kwargs.update(baseline_name='baseline-fixture', post_name='post-fixture')
            with (patch.multiple(runner, LOCAL=Path(tmp), gpu_phase=gpu, prepare_assets=prepare,
                                 collect_evidence=collect, WORKING_VOLUME='working-fixture' if algorithm == c.PPO else ''),
                  patch.object(runner.sys, 'argv', ['modal', '--detach']),
                  patch.object(runner, 'sha256', return_value='f'*64),
                  patch.object(runner.subprocess, 'check_output', side_effect=['b'*40, '', '']),
                  redirect_stdout(StringIO())):
                if algorithm == c.PPO and action not in {'smoke', 'smoke-reload-eval'}:
                    with self.assertRaises(ValueError):
                        runner.main(**kwargs)
                else:
                    runner.main(**kwargs)
        return gpu, prepare, collect, variant

    def test_ppo_smoke_and_reload_spawn_memory_variant_only(self):
        for action in ('smoke', 'smoke-reload-eval'):
            with self.subTest(action=action):
                gpu, prepare, collect, variant = self.dispatch(action, c.PPO)
                gpu.with_options.assert_called_once_with(memory=196608)
                variant.spawn.assert_called_once()
                self.assertEqual(variant.spawn.call_args.args[0]['action'], action)
                self.assertEqual(variant.spawn.call_args.args[0]['algorithm'], c.PPO)
                gpu.spawn.assert_not_called()
                self.assertEqual(prepare.mock_calls, [])
                self.assertEqual(collect.mock_calls, [])

    def test_grpo_smoke_and_reload_spawn_static_function(self):
        for action in ('smoke', 'smoke-reload-eval'):
            with self.subTest(action=action):
                gpu, prepare, collect, variant = self.dispatch(action)
                gpu.with_options.assert_not_called()
                gpu.spawn.assert_called_once()
                self.assertEqual(gpu.spawn.call_args.args[0]['algorithm'], c.GRPO)
                self.assertEqual(prepare.mock_calls, [])
                self.assertEqual(collect.mock_calls, [])
                self.assertEqual(variant.mock_calls, [])

    def test_non_gpu_actions_never_override_resources(self):
        for action in ('prepare', 'summarize', 'export'):
            with self.subTest(action=action):
                gpu, prepare, collect, variant = self.dispatch(action)
                chosen, unused = (prepare, collect) if action == 'prepare' else (collect, prepare)
                chosen.spawn.assert_called_once()
                chosen.with_options.assert_not_called()
                self.assertEqual(unused.mock_calls, [])
                self.assertEqual(gpu.mock_calls, [])
                self.assertEqual(variant.mock_calls, [])

    def test_ppo_non_gpu_actions_rejected_before_resource_override(self):
        for action in ('prepare', 'summarize', 'export'):
            with self.subTest(action=action):
                for function in self.dispatch(action, c.PPO):
                    self.assertEqual(function.mock_calls, [])


class CheckpointTests(unittest.TestCase):
    def test_historical_identity_is_grpo_only(self):
        c.verify_identity({}, c.GRPO)
        with self.assertRaises(ValueError):
            c.verify_identity({}, c.PPO)
        for algorithm in c.ALGORITHMS:
            with self.assertRaises(ValueError):
                c.verify_identity(c.identity(c.PPO if algorithm == c.GRPO else c.GRPO), algorithm)

    def test_native_checkpoint_requires_critic_optimizer_and_source_identity(self):
        valid = checkpoint_fixture()
        c.validate_checkpoint(valid, c.PPO)
        for key, value in [('complete', False), ('algorithm', 'grpo'), ('protocol', 'other'),
                           ('step', 1), ('base_revision', 'wrong'), ('roles', {'actor': valid['roles']['actor']}),
                           ('source_identity', {'lock_sha256': 'wrong'})]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                c.validate_checkpoint({**valid, key: value}, c.PPO)
        del valid['roles']['critic']['optim_world_size_1_rank_0.pt']
        with self.assertRaises(ValueError):
            c.validate_checkpoint(valid, c.PPO)

    def test_reload_digest_checked_before_dispatch_and_forwarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'identity.json'; path.write_text(json.dumps(checkpoint_fixture()))
            req = runner.request('smoke-reload-eval', 'ppo-mc-v1-reload', wandb_entity='team', algorithm=c.PPO,
                data_layout='restored', checkpoint='phase2/runs/ppo-mc-v1-smoke/checkpoints/global_step_2',
                checkpoint_identity=str(path))
            cmd = runner.launch_command(req)
            self.assertEqual(cmd[cmd.index('--checkpoint-identity-sha256')+1], runner.sha256(path))
            path.write_text(json.dumps(c.identity(c.GRPO)))
            with patch.object(runner.gpu_phase, 'spawn') as spawn, self.assertRaises(ValueError):
                runner.main(action='smoke-reload-eval', name='ppo-mc-v1-reload', wandb_entity='team', algorithm=c.PPO,
                    data_layout='restored', checkpoint=req['checkpoint'], checkpoint_identity=str(path))
            spawn.assert_not_called()

    def test_checkpoint_inventory_hash_verification_detects_corruption(self):
        record = checkpoint_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for role, files in record['roles'].items():
                (path/role).mkdir()
                for filename in files:
                    (path/role/filename).write_bytes(b'x')
            c.verify_inventory(path, record)
            (path/'critic/model_world_size_1_rank_0.pt').write_bytes(b'y')
            with self.assertRaises(ValueError):
                c.verify_inventory(path, record)

    def test_retention_cannot_touch_historical_or_source_checkpoints(self):
        run = Path('/data/phase2/runs/ppo-mc-v1-new')
        self.assertEqual(runtime.retention_path(run/'checkpoints/global_step_1/actor', run),
                         run/'checkpoints/global_step_1/actor')
        for path in ('/data/runs/train-01/checkpoints/global_step_20/actor',
                     '/data/restored/runs/train-01/checkpoints/global_step_20/critic',
                     '/data/phase2/runs/ppo-mc-v1-old/checkpoints/global_step_2/critic',
                     str(run/'checkpoints')):
            with self.assertRaises(ValueError):
                runtime.retention_path(path, run)


if __name__ == '__main__':
    unittest.main()
