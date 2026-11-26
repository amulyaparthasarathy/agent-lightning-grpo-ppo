"""Run unmodified VERL 0.8.0 config modules through real Hydra 1.3.2 on CPU.

Only package initializers that import accelerator/distributed services and model
asset I/O are substituted. Dataclass definitions, constructors, __post_init__,
OmegaConf conversion, YAML composition and Calc-X build_config are unmodified.
This process never imports CUDA/vLLM/Ray or downloads model/tokenizer assets.
"""
import ast
import asyncio
import collections.abc
import hashlib
import importlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCES = Path(os.environ.get('AGL_PINNED_SOURCES', ROOT/'.local/ppo-review'))
LOCK = json.loads((ROOT/'experiment/ppo_source_lock.json').read_text())
VERL = SOURCES/'verl/verl'
AGL = SOURCES/'agent-lightning/agentlightning'


def bootstrap():
    # Verify the complete extracted package, including dataclasses AND composed
    # YAML defaults, against the exact released wheel, never upstream main.
    wheel = SOURCES/LOCK['verl_wheel']['filename']
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == LOCK['verl_wheel']['digests']['sha256']
    with zipfile.ZipFile(wheel) as archive:
        checked = 0
        for name in archive.namelist():
            if name.startswith('verl/') and name.endswith(('.py', '.yaml')):
                assert (VERL/name.removeprefix('verl/')).read_bytes() == archive.read(name), name
                checked += 1
    for name, expected in LOCK['files']['agentlightning'].items():
        assert hashlib.sha256((AGL/name).read_bytes()).hexdigest() == expected
    example = SOURCES/'agent-lightning/examples/calc_x/train_calc_agent.py'
    assert hashlib.sha256(example.read_bytes()).hexdigest() == LOCK['calc_example_sha256']
    print(f'Verified VERL wheel SHA256 and {checked} Python/YAML files; pinned AGL config/example hashes.')

    def package(name, path):
        spec = importlib.util.spec_from_file_location(name, path/'__init__.py',
                                                     submodule_search_locations=[str(path)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        if '.' in name:
            parent, attr = name.rsplit('.', 1)
            setattr(sys.modules[parent], attr, module)
        return module

    def seam(name, **attributes):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        sys.modules[name] = module
        if '.' in name:
            parent, attr = name.rsplit('.', 1)
            setattr(sys.modules[parent], attr, module)
        return module

    for name, path in [('verl', VERL), ('verl.utils', VERL/'utils'),
                       ('verl.trainer', VERL/'trainer'), ('verl.workers', VERL/'workers'),
                       ('verl.utils.profiler', VERL/'utils/profiler'),
                       ('verl.utils.qat', VERL/'utils/qat'),
                       ('agentlightning', AGL), ('agentlightning.verl', AGL/'verl')]:
        package(name, path)
    from verl.utils.profiler.config import ProfilerConfig
    from verl.utils.qat.core import QATConfig
    sys.modules['verl.utils.profiler'].ProfilerConfig = ProfilerConfig
    sys.modules['verl.utils.qat'].QATConfig = QATConfig
    # Explicit model-asset seam: native HFModelConfig.__post_init__ still runs,
    # including its validation; only tokenizer/config filesystem loading is fake.
    seam('transformers', AutoConfig=NS(from_pretrained=lambda *a, **kw:
        NS(model_type='llama', architectures=['LlamaForCausalLM'], tie_word_embeddings=True)))
    sys.modules['verl.utils'].hf_tokenizer = lambda *a, **kw: NS(
        chat_template='CPU fixture', bos_token_id=1, eos_token_id=2, pad_token_id=2)
    sys.modules['verl.utils'].hf_processor = lambda *a, **kw: None
    seam('verl.utils.fs', copy_to_local=lambda path, **kw: path)
    seam('verl.utils.import_utils', import_external_libs=lambda value: None)
    seam('verl.utils.model', get_generation_config=lambda *a, **kw: None,
         update_model_config=lambda config, override_config_kwargs: config.__dict__.update(override_config_kwargs))
    seam('datasets', Dataset=object)  # imported by Calc-X; never used by build_config
    # constants_ppo needs only a Ray environment-variable name and the device
    # capability. Supply the declared H100 capability without a Ray/CUDA import.
    for name in ('ray', 'ray._private', 'ray._private.runtime_env'):
        seam(name)
    seam('ray._private.runtime_env.constants', RAY_JOB_CONFIG_JSON_ENV_VAR='RAY_JOB_CONFIG_JSON')
    seam('verl.utils.device', get_device_capability=lambda: (9, 0))
    import hydra
    assert hydra.__version__ == '1.3.2'
    spec = importlib.util.spec_from_file_location('train_calc_agent', example)
    calc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calc)
    sys.modules[spec.name] = calc
    return calc


def native_functions(relative, names, namespace, class_name=None):
    """Load entire verbatim function bodies; bypass only accelerator imports."""
    path = VERL/relative
    tree = ast.parse(path.read_text())
    nodes = tree.body if class_name is None else next(
        n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    selected = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    assert {n.name for n in selected} == set(names)
    assert all(not n.decorator_list for n in selected)
    tree.body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + selected
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace


calc = bootstrap()
from hydra.errors import InstantiationException
from omegaconf import OmegaConf, open_dict
from verl.utils.config import omega_conf_to_dataclass
from verl.workers import config as wc
from experiment import matched_contract as c, matched_launch, launch, ppo_contract
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env


def generated():
    for phase in c.exploratory.PHASES:
        b = c.exploratory.bounds(phase)
        plan = c.make_plan('schema-fixture', phase, 1, b['stage_seconds'], b['startup_seconds'],
            b['budget_usd'], 10, json.loads((ROOT/'evidence/phase2-prepared-data-manifest.json').read_text()),
            {'commit': 'b'*40, 'dirty': False, 'files': {'docs/PHASE2_MATCHED_STUDY_CONTRACT.md': 'c'*64}})
        for job in plan['jobs']:
            for stage in job['stages']:
                req = c.stage_request(plan, job, stage)
                rp, _ = matched_launch.make_plan(req, ROOT/'experiment', SOURCES/'agent-lightning',
                    Path('/prepared/model'), Path('/prepared/data'), Path(sys.executable),
                    Path('/run')/job['job_id']/stage['name'], 12345)
                overrides = ppo_contract.runtime_env_overrides(rp['overrides'], get_ppo_ray_runtime_env())
                cfg = calc.build_config(model=rp['model'], agl_base_url=rp['gateway'], agl_key=rp['local_key'],
                                        run_name=rp['run_name'], config_overrides=overrides)
                # train() adds only these dataset paths after build_config.
                cfg.data.train_files, cfg.data.val_files = [rp['train_file']], [rp['eval_file']]
                matched_launch.validate_resolved(OmegaConf.to_container(cfg, resolve=True), rp)
                yield req, cfg


class SchemaChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs = list(generated())

    def test_old_seed_override_reproduces_the_observed_exception(self):
        for req, cfg in self.configs[:6]:
            with self.subTest(algorithm=req['job']['algorithm'], stage=req['stage']['name']):
                old = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
                sampling = old.actor_rollout_ref.rollout.engine_kwargs.vllm.pop('seed')
                old.actor_rollout_ref.rollout.seed = sampling
                with self.assertRaisesRegex(InstantiationException, "unexpected keyword argument 'seed'") as error:
                    omega_conf_to_dataclass(old.actor_rollout_ref.rollout)
                self.assertIsInstance(error.exception.__cause__, TypeError)

    def test_all_24_generated_worker_configs_instantiate(self):
        self.assertEqual(len(self.configs), 24)
        for req, cfg in self.configs:
            job = req['job']; streams = job['seed_streams']
            with self.subTest(algorithm=job['algorithm'], seed=job['run_seed'], stage=req['stage']['name']):
                model = omega_conf_to_dataclass(cfg.actor_rollout_ref.model)
                # vLLMHttpServer._init_model_config uses the explicit-type path.
                self.assertIsInstance(omega_conf_to_dataclass(cfg.actor_rollout_ref.model, wc.HFModelConfig), wc.HFModelConfig)
                actor = omega_conf_to_dataclass(cfg.actor_rollout_ref.actor)
                critic = omega_conf_to_dataclass(cfg.critic)  # also validate disabled GRPO critic
                rollout = omega_conf_to_dataclass(cfg.actor_rollout_ref.rollout)
                checkpoint = omega_conf_to_dataclass(cfg.actor_rollout_ref.rollout.checkpoint_engine)
                self.assertIsInstance(actor, wc.FSDPActorConfig)
                self.assertIsInstance(critic, wc.FSDPCriticConfig)
                self.assertIsInstance(rollout, wc.RolloutConfig)
                self.assertIsInstance(rollout.val_kwargs, wc.SamplingConfig)
                self.assertIsInstance(actor.engine, wc.FSDPEngineConfig)
                self.assertIsInstance(critic.engine, wc.FSDPEngineConfig)
                self.assertIsInstance(checkpoint, wc.CheckpointEngineConfig)
                self.assertEqual(actor.engine.seed, streams['actor_optimizer'])
                self.assertEqual(critic.engine.seed, streams['critic_init'] or 0)
                self.assertEqual(actor.data_loader_seed, streams['row_batching'])
                self.assertEqual(critic.data_loader_seed, streams['row_batching'])
                self.assertEqual(rollout.engine_kwargs['vllm']['seed'], streams['actor_sampling'])
                self.assertEqual(cfg.data.seed, streams['paired_question_order'])
                # Same TrainingWorkerConfig construction paths as engine_workers
                # and RayPPOTrainer. No TrainingWorker/GPU construction follows.
                actor.model_config = model
                for role, obj, model_cfg in [('language_model', actor, model), ('value_model', critic, critic.model)]:
                    worker = wc.TrainingWorkerConfig(model_type=role, model_config=model_cfg,
                        engine_config=obj.engine, optimizer_config=obj.optim, checkpoint_config=obj.checkpoint)
                    self.assertIsInstance(worker, wc.TrainingWorkerConfig)
                # Reference settings are not used by this no-KL protocol, but
                # validate their native conversion after the worker's renames.
                ref = OmegaConf.create(OmegaConf.to_container(cfg.actor_rollout_ref.ref, resolve=True))
                with open_dict(ref):
                    ref.ppo_mini_batch_size = cfg.actor_rollout_ref.actor.ppo_mini_batch_size
                    for old, new, default in [('log_prob_micro_batch_size','ppo_micro_batch_size',None),
                        ('log_prob_micro_batch_size_per_gpu','ppo_micro_batch_size_per_gpu',None),
                        ('log_prob_use_dynamic_bsz','use_dynamic_bsz',False),
                        ('log_prob_max_token_len_per_gpu','ppo_max_token_len_per_gpu',None)]:
                        ref[new] = ref.pop(old, default)
                self.assertIsInstance(omega_conf_to_dataclass(ref), wc.FSDPActorConfig)
        print('24 complete composed configs: actor, critic, rollout, model (both conversion paths), reference and training/checkpoint configs passed.')

    def test_other_unknown_fields_are_rejected_not_filtered(self):
        _, base = self.configs[0]
        for path in ('actor_rollout_ref.actor', 'actor_rollout_ref.actor.fsdp_config', 'critic',
                     'critic.fsdp', 'actor_rollout_ref.rollout', 'actor_rollout_ref.rollout.val_kwargs',
                     'actor_rollout_ref.model'):
            with self.subTest(path=path):
                cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
                node = OmegaConf.select(cfg, path)
                node.unsupported_runtime_field = 123
                with self.assertRaisesRegex(InstantiationException, 'unsupported_runtime_field'):
                    omega_conf_to_dataclass(node)

    def test_native_vllm_server_receives_the_declared_sampling_seed(self):
        from packaging import version
        converter = native_functions('workers/rollout/vllm_rollout/utils.py',
            ['build_cli_args_from_config'], {'json': json})['build_cli_args_from_config']
        class Captured(Exception):
            pass
        def capture(args):
            raise Captured(converter(args))
        ns = dict(json=json, logger=logging.getLogger('seed-proof'), version=version,
                  _VLLM_VERSION=version.parse('0.22.0'), build_cli_args_from_config=capture,
                  build_vllm_profiler_args=lambda *args: {})
        native_functions('workers/rollout/vllm_rollout/vllm_async_server.py',
            ['launch_server', '_get_engine_kwargs_key', '_preprocess_engine_kwargs',
             '_get_override_generation_config'], ns, 'vLLMHttpServer')
        for req, cfg in self.configs:
            server = NS(config=omega_conf_to_dataclass(cfg.actor_rollout_ref.rollout),
                        model_config=omega_conf_to_dataclass(cfg.actor_rollout_ref.model),
                        node_rank=0, replica_rank=0, nnodes=1, gpus_per_node=1,
                        profiler_controller=NS(config=None, tool_config=None),
                        _apply_quantization=lambda: (None, {}), _get_worker_extension_cls=lambda: 'CPU-boundary')
            for name in ('_get_engine_kwargs_key','_preprocess_engine_kwargs','_get_override_generation_config'):
                setattr(server, name, ns[name].__get__(server))
            with self.assertRaises(Captured) as result:
                asyncio.run(ns['launch_server'](server))
            argv = result.exception.args[0]
            self.assertEqual(argv[argv.index('--seed')+1], str(req['job']['seed_streams']['actor_sampling']))
            self.assertEqual(argv.count('--seed'), 1)
        print('24 native VERL launch_server paths emit --seed=<actor_sampling>; stopped before vLLM parser/server/GPU startup.')

    def test_paired_seed_design_and_ray_setup_are_preserved(self):
        for phase in c.exploratory.PHASES:
            jobs = {req['job']['job_id']: req['job'] for req, _ in self.configs if req['plan']['phase']==phase}
            for seed in c.exploratory.SEEDS[phase]:
                a,b = [j for j in jobs.values() if j['run_seed']==seed]
                self.assertEqual(a['seed_streams']['paired_question_order'], b['seed_streams']['paired_question_order'])
                self.assertNotEqual(a['seed_streams']['actor_sampling'], b['seed_streams']['actor_sampling'])
        for req, cfg in self.configs:
            env = cfg.ray_kwargs.ray_init.runtime_env
            self.assertEqual(env.worker_process_setup_hook, 'matched_runtime.register_in_worker')
            self.assertEqual(json.loads(env.env_vars.AGL_MATCHED_REQUEST), req)


class ReaderTaskConfigs(unittest.TestCase):
    def test_two_tasks_two_algorithms_three_stages_composed(self):
        from experiment import reader_contract as reader
        from test_reader_training import profile, source
        checked = 0
        for task in ('calculator-v1', 'asset-status-v1'):
            p = profile(); p['task'] = task
            plan = reader.make_plan(p, source())
            for job in plan['jobs']:
                for stage in job['stages']:
                    req = reader.stage_request(plan, job, stage)
                    rp, _ = matched_launch.make_plan(req, ROOT/'experiment', SOURCES/'agent-lightning',
                        Path('/data')/p['paths']['model'], Path('/data')/p['paths']['data'], Path(sys.executable),
                        Path('/data')/stage['output'], 12345)
                    cfg = calc.build_config(model=rp['model'], agl_base_url=rp['gateway'], agl_key=rp['local_key'],
                        run_name=rp['run_name'], config_overrides=ppo_contract.runtime_env_overrides(rp['overrides'], get_ppo_ray_runtime_env()))
                    resolved = OmegaConf.to_container(cfg, resolve=True)
                    matched_launch.validate_resolved(resolved, rp)
                    self.assertEqual(resolved['agentlightning']['local']['agent_class'], plan['task_identity']['agent_class'])
                    self.assertEqual(resolved['agentlightning']['local']['env_map'], plan['task_identity']['data_mapping'])
                    self.assertEqual(set(resolved['agentlightning']['local']['env_map']), {'QUESTION', 'RESULT', 'AGL_CASE_ID', 'AGL_DATA_SPLIT', 'EVAL_SPLIT'})
                    from experiment.request_transport import digest
                    runtime_env = resolved['ray_kwargs']['ray_init']['runtime_env']['env_vars']
                    self.assertNotIn('AGL_MATCHED_REQUEST', runtime_env)
                    self.assertEqual(runtime_env['AGL_MATCHED_REQUEST_SHA256'], digest(req))
                    self.assertEqual(runtime_env['AGL_MATCHED_REQUEST_FILE'], str(Path('/data')/stage['output']/'request.json'))
                    self.assertLess(len(json.dumps(runtime_env)), 8192)
                    self.assertTrue(rp['train_file'].endswith('/train_main.parquet'))
                    self.assertTrue(rp['eval_file'].endswith('/eval_main.parquet'))
                    self.assertEqual(cfg.trainer.val_only, stage['name'] != 'training')
                    self.assertEqual(cfg.trainer.resume_mode, 'resume_path' if stage['name'] == 'checkpoint-eval' else 'disable')
                    if stage['name'] == 'checkpoint-eval':
                        self.assertTrue(cfg.trainer.resume_from_path.endswith('/training/checkpoints/global_step_2'))
                    # Instantiate the real native config classes for both tasks and roles.
                    for node in (cfg.actor_rollout_ref.actor, cfg.actor_rollout_ref.rollout, cfg.critic):
                        omega_conf_to_dataclass(node)
                    bad = OmegaConf.to_container(cfg, resolve=True)
                    bad['agentlightning']['local']['env_map']['OBSOLETE'] = 'input.result'
                    with self.assertRaisesRegex(ValueError, 'env_map'):
                        matched_launch.validate_resolved(bad, rp)
                    checked += 1
        self.assertEqual(checked, 12)
        print('12 reader configs: complete task agent/env_map, datasets, native roles and fresh-reload routing passed.')

    def test_recovery_learning_resolved_dispatch(self):
        from recovery_learning_support import request
        from experiment import reader_contract as reader
        for selection in ('grpo','ppo-mc'):
            for name in reader.modes.STAGES[reader.modes.RECOVERY_LEARNING]:
                req = request(name,selection)
                from experiment import qualification_paths
                binding = qualification_paths.describe(req,'/fixture-canonical-root','vo-test')
                locations = qualification_paths.locations(req,binding)
                run = locations['run']
                rp, _ = matched_launch.make_plan(req, ROOT/'experiment', SOURCES/'agent-lightning',
                    locations['model'], locations['data'], Path(sys.executable), run, 12345, binding=binding)
                cfg = calc.build_config(model=rp['model'], agl_base_url=rp['gateway'], agl_key=rp['local_key'],
                    run_name=rp['run_name'], config_overrides=ppo_contract.runtime_env_overrides(rp['overrides'], get_ppo_ray_runtime_env()))
                resolved = OmegaConf.to_container(cfg, resolve=True)
                matched_launch.validate_resolved(resolved,rp)
                self.assertEqual(cfg.trainer.val_only,name in ('baseline','checkpoint-eval'))
                self.assertEqual(cfg.trainer.test_freq,-1)
                self.assertEqual(cfg.trainer.resume_mode,'resume_path' if name=='checkpoint-eval' else 'disable')
                self.assertEqual(cfg.data.shuffle,name=='training')
                critic = omega_conf_to_dataclass(cfg.critic)
                self.assertEqual((critic.engine.micro_batch_size_per_gpu,critic.engine.infer_micro_batch_size_per_gpu),(1,1))
                self.assertEqual((cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                  cfg.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu),(1,1))
                for key in ('train_file','eval_file'):
                    with self.assertRaisesRegex(ValueError,'routing changed'):
                        matched_launch.validate_resolved(resolved,dict(rp,**{key:'/data/final_eval.parquet'}))

    def test_cpu_preflight_composes_all_eight_future_configs_without_allocating_outputs(self):
        from tempfile import TemporaryDirectory
        from experiment import reader_preflight
        from recovery_learning_support import request
        with TemporaryDirectory() as tmp:
            root = Path(tmp)/'physical'
            root.mkdir()
            mount = Path(tmp)/'logical'
            mount.symlink_to(root)
            records = reader_preflight.qualification_paths(request(selection='paired')['plan'],mount,'vo-test')
            self.assertEqual(len(records),8)
            from study_support import profile, fixture_source
            from experiment import reader_contract
            study = reader_contract.make_plan(profile(), fixture_source())
            records = reader_preflight.qualification_paths(study, mount, 'vo-test')
            self.assertEqual(len(records),11)
            self.assertEqual(list(root.iterdir()),[])

    def test_recovery_calibration_two_algorithms_four_stages_composed(self):
        from experiment import reader_contract as reader, reader_assets, recovery_task
        from test_reader_training import source
        p = reader_assets.read(ROOT/'examples/reader-profile.recovery-calibration.synthetic.json')
        plan = reader.make_plan(p, source())
        checked = 0
        for job in plan['jobs']:
            for stage in job['stages']:
                req = reader.stage_request(plan, job, stage)
                run = Path('/data')/stage['output']
                rp, commands = matched_launch.make_plan(req, ROOT/'experiment', SOURCES/'agent-lightning',
                    Path('/data')/p['paths']['model'], Path('/data')/p['paths']['data'], Path(sys.executable), run, 12345)
                cfg = calc.build_config(model=rp['model'], agl_base_url=rp['gateway'], agl_key=rp['local_key'],
                    run_name=rp['run_name'], config_overrides=ppo_contract.runtime_env_overrides(rp['overrides'], get_ppo_ray_runtime_env()))
                resolved = OmegaConf.to_container(cfg, resolve=True)
                matched_launch.validate_resolved(resolved, rp)
                greedy = stage['name'] == 'greedy-development'
                self.assertEqual((cfg.trainer.val_only, cfg.trainer.val_before_train), (greedy, greedy))
                self.assertEqual((cfg.trainer.resume_mode, cfg.trainer.test_freq), ('disable', -1))
                # Both upstream dataset arguments are run-local development copies.
                self.assertEqual([rp['train_file'], rp['eval_file']], [str(run/'calibration-train.parquet'), str(run/'calibration-val.parquet')])
                self.assertEqual(resolved['agentlightning']['local']['agent_class'], 'recovery_agent.Agent')
                self.assertEqual(resolved['agentlightning']['local']['env_map'], plan['task_identity']['data_mapping'])
                aggregator = resolved['agentlightning']['trace_aggregator']
                limits = recovery_task.CONTEXT_LIMITS
                self.assertEqual((aggregator['level'], aggregator['trajectory_max_prompt_length'], aggregator['trajectory_max_response_length']),
                                 ('trajectory', limits['trajectory_max_prompt_length'], limits['trajectory_max_response_length']))
                self.assertEqual(cfg.actor_rollout_ref.rollout.max_model_len, limits['max_model_len'])
                self.assertEqual((cfg.actor_rollout_ref.rollout.n, cfg.actor_rollout_ref.rollout.val_kwargs.n, cfg.data.train_batch_size), (4, 1, 8))
                self.assertEqual(cfg.algorithm.adv_estimator, 'gae' if job['algorithm'] == c.ALGORITHMS[1] else 'grpo')
                self.assertIn('default_proxy.train.temperature=1.0', commands['gateway'])
                self.assertIn('default_proxy.val.temperature=0.0', commands['gateway'])
                runtime_env = resolved['ray_kwargs']['ray_init']['runtime_env']['env_vars']
                self.assertNotIn('AGL_MATCHED_REQUEST', runtime_env)
                self.assertLess(len(json.dumps(runtime_env)), 8192)
                for node in (cfg.actor_rollout_ref.actor, cfg.actor_rollout_ref.rollout, cfg.critic):
                    omega_conf_to_dataclass(node)
                for key, value in (('train_file', '/data/'+p['paths']['data']+'/final_eval.parquet'),
                                   ('eval_file', '/data/'+p['paths']['data']+'/eval_main.parquet')):
                    with self.assertRaisesRegex(ValueError, 'routing changed'):
                        matched_launch.validate_resolved(resolved, {**rp, key: value})
                bad = OmegaConf.to_container(cfg, resolve=True)
                bad['agentlightning']['trace_aggregator']['trajectory_max_prompt_length'] = 2048
                with self.assertRaisesRegex(ValueError, 'drift'):
                    matched_launch.validate_resolved(bad, rp)
                checked += 1
        self.assertEqual(checked, 8)
        print('8 recovery calibration configs: development-only dataset routing, modes, context limits and native roles passed.')


if __name__ == '__main__':
    def deny_network(event, args):
        if event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto'):
            raise AssertionError('Network forbidden in pinned config checks')
    sys.addaudithook(deny_network)
    unittest.main(verbosity=2)
