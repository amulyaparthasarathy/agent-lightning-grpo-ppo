"""CPU-only execution of exact wheel code; sockets and Ray are never mocked."""
import ast
import functools
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from uuid import uuid4
import zipfile

import psutil
import zmq
import zmq.asyncio
from urllib3.util import parse_url

ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads((ROOT/'tests/pinned_ipc_lock.json').read_text())
WHEEL = Path(os.environ.get('AGL_VLLM_WHEEL', ROOT/'.local/ppo-review'/LOCK['filename']))


def wheel_sources():
    assert hashlib.file_digest(WHEEL.open('rb'), 'sha256').hexdigest() == LOCK['sha256']
    with zipfile.ZipFile(WHEEL) as archive:
        result = {name: archive.read(name).decode() for name in LOCK['files']}
    for name, code in result.items():
        assert hashlib.sha256(code.encode()).hexdigest() == LOCK['files'][name]
    return result


def native_functions(code, names, namespace):
    tree = ast.parse(code)
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in tree.body} == set(names)
    tree.body.insert(0, ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0))
    exec(compile(ast.fix_missing_locations(tree), 'pinned-wheel-functions', 'exec'), namespace)
    return namespace


def native_network():
    sources = wheel_sources()
    envs = types.ModuleType('pinned_vllm_envs')
    # Entire original envs module, including its lazy RPC_BASE_PATH getter.
    exec(compile(sources['vllm/envs.py'], 'vllm/envs.py', 'exec'), envs.__dict__)
    # Same cache wrapper as enable_envs_cache; avoid eagerly evaluating unrelated
    # CUDA-specific variables. The RPC getter itself is unchanged.
    envs.__getattr__ = functools.cache(envs.__getattr__)
    return native_functions(sources['vllm/utils/network_utils.py'],
        ['get_open_zmq_ipc_path', 'make_zmq_socket', 'split_zmq_path', 'is_valid_ipv6_address'],
        dict(envs=envs, uuid4=uuid4, psutil=psutil, zmq=zmq, parse_url=parse_url, ipaddress=ipaddress))


def snapshot():
    return dict(pid=os.getpid(), tempfile=tempfile.gettempdir(),
                env={k:os.environ.get(k) for k in ('TMPDIR','RAY_TMPDIR','VLLM_RPC_BASE_PATH','AGL_TRACE_DIR','WANDB_DIR')})


def connect(endpoint):
    n = native_network()
    with zmq.Context() as ctx:
        with n['make_zmq_socket'](ctx, endpoint, zmq.PUSH, linger=0) as sock:
            sock.setsockopt(zmq.SNDTIMEO, 10000)
            sock.send_json(snapshot())


def probe(expect_failure=False):
    initial = snapshot()
    n = native_network()
    # Cached tempfile state cannot override an explicit VLLM_RPC_BASE_PATH.
    if not expect_failure:
        tempfile.tempdir = '/an/intentionally/stale/' + 'cached-long-directory/'*9
    endpoint = n['get_open_zmq_ipc_path']()
    result = dict(initial, endpoint=endpoint, path_bytes=len(os.fsencode(endpoint.removeprefix('ipc://'))))
    ctx = zmq.Context()
    try:
        try:
            sock = n['make_zmq_socket'](ctx, endpoint, zmq.PULL, linger=0)
        except zmq.ZMQError as error:
            assert expect_failure and 'longer than 107 characters' in str(error), str(error)
            result['old_error'] = str(error)
            return result
        assert not expect_failure, 'Old long path unexpectedly bound'
        with sock:
            sock.setsockopt(zmq.RCVTIMEO, 30000)
            # A real exec child starts with the environment inherited from the
            # Ray server process, as do spawned vLLM engine processes.
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'connect', endpoint])
            try:
                result['peer'] = sock.recv_json()
                assert child.wait(timeout=30) == 0
            finally:
                if child.poll() is None: child.kill(); child.wait()
        return result
    finally:
        ctx.destroy(linger=0)


def verl_environments():
    lock = json.loads((ROOT/'experiment/ppo_source_lock.json').read_text())
    wheel = Path(os.environ.get('AGL_PINNED_SOURCES',ROOT/'.local/ppo-review'))/lock['verl_wheel']['filename']
    assert hashlib.file_digest(wheel.open('rb'),'sha256').hexdigest() == lock['verl_wheel']['digests']['sha256']
    with zipfile.ZipFile(wheel) as z:
        constants = ast.parse(z.read('verl/trainer/constants_ppo.py'))
        server = ast.parse(z.read('verl/workers/rollout/vllm_rollout/vllm_async_server.py'))
    from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR
    names={'PPO_RAY_RUNTIME_ENV'}
    constants.body=[n for n in constants.body if (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets)) or (isinstance(n,ast.FunctionDef) and n.name=='get_ppo_ray_runtime_env')]
    ns=dict(os=os,json=json,_gb200_nccl_env={},RAY_JOB_CONFIG_JSON_ENV_VAR=RAY_JOB_CONFIG_JSON_ENV_VAR)
    exec(compile(constants,'native-verl-runtime-env','exec'),ns)
    launch = next(n for n in ast.walk(server) if isinstance(n,ast.AsyncFunctionDef) and n.name=='launch_servers')
    env = next(n.value for n in ast.walk(launch) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='env_vars' for t in n.targets))
    return ns['get_ppo_ray_runtime_env'](), ast.literal_eval(env)


def ray_driver(request):
    import ray
    from experiment import matched_launch, ppo_contract
    assert ray.__version__ == '2.60.0'
    defaults, server_env = verl_environments()
    cfg=matched_launch.configuration(request,Path('/prepared/model'),Path(os.environ['AGL_PPO_RUN_DIR']))
    overrides=ppo_contract.runtime_env_overrides([k+'='+json.dumps(v) for k,v in cfg.items()],defaults)
    env=json.loads(overrides[-1].split('=',1)[1])
    # Actual pinned defaults and nested server env must not replace launcher IPC paths.
    assert not set(server_env)&{'TMPDIR','RAY_TMPDIR','VLLM_RPC_BASE_PATH'}
    assert not set(env)&{'TMPDIR','RAY_TMPDIR','VLLM_RPC_BASE_PATH'}
    ray.init(num_cpus=2, include_dashboard=False, object_store_memory=80*1024*1024,
             runtime_env={'env_vars':env}, log_to_driver=False)
    try:
        @ray.remote(num_cpus=0)
        class Server:
            def check(self):
                from pinned_ipc_support import probe
                return probe()
        @ray.remote(num_cpus=0)
        class TaskRunner:
            def check(self):
                # Native VERL server env overrides in a real nested Ray actor.
                server=Server.options(runtime_env={'env_vars':server_env}).remote()
                return ray.get(server.check.remote(), timeout=60)
        actor=TaskRunner.remote()
        result=ray.get(actor.check.remote(),timeout=90)
        result['driver']=snapshot()
        return result
    finally:
        ray.shutdown()


if __name__=='__main__':
    assert zmq.__version__=='27.3.0'
    if sys.argv[1]=='connect': connect(sys.argv[2])
    elif sys.argv[1]=='old': print(json.dumps(probe(expect_failure=True)))
    elif sys.argv[1]=='ray':
        result=ray_driver(json.loads(os.environ['AGL_MATCHED_REQUEST']))
        Path(os.environ['AGL_PPO_RUN_DIR'],'socket-proof.json').write_text(json.dumps(result,indent=2)+'\n')
