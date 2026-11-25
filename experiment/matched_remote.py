"""Common remote lifecycle, parameterized by a closed contract and owned handles.

No resource declarations. Historical and reader apps share these mechanisms.
"""
import asyncio
import json
import os
from pathlib import Path
import shutil
import threading
import time
from . import matched_container as runtime, matched_supervisor as supervisor, matched_contract as historical


def write(path, data, exclusive=False):
    with path.open('x' if exclusive else 'w') as f:
        json.dump(data, f, indent=2)
        f.write('\n')


def verify_package(plan):
    from . import reader_contract as reader
    from .reader_source import verify_package as verify
    verify(runtime.BUNDLE.parent, plan['source']['runtime_files'], reader.file_hash)


def capture_build(run, image_id=None):
    shutil.copytree('/opt/agl-build/evidence', run/'image-build')
    # Image ID is supplied by the explicit client submission, not inferred from a cache hit.
    write(run/'runtime-identity.json', {'image_id': image_id or 'unavailable',
          'cache_hit': 'unverified', 'python': str(runtime.PYTHON), 'container_id': os.environ['MODAL_TASK_ID'], 'worker_hostname': __import__('socket').gethostname()})


def stage_worker(request, deadline_epoch, *, c, registry, volume, call_id, image_id=None, mount=Path('/data')):
    entered_at = time.time()
    c.validate(request['plan'])
    reader = request['plan'].get('schema') in ('reader-reference-v1', 'reader-task-v2')
    if request != c.stage_request(request['plan'], request['job'], request['stage']):
        raise ValueError('Invalid stage request')
    if registry.get('stop:'+supervisor.key(request['plan'])):
        raise ValueError('Attempt stopped before worker entry')
    limit = c.stage_limit(request['plan'],request['stage'])
    if (reader or historical.exploratory.is_exploratory(request['plan'])):
        if deadline_epoch is None or not entered_at < deadline_epoch <= entered_at+limit:
            raise ValueError('Missing, expired or extended exploratory stage deadline')
    stage_key = 'stage:'+request['plan_sha256']+':'+request['stage']['wandb']['id']
    if not registry.put(stage_key, {'call_id':call_id}, skip_if_exists=True):
        raise ValueError('Stage already entered; automatic restart forbidden')
    output = request['stage']['output']
    volume.reload()
    run = mount/output
    if reader:
        from .reader_assets import owned_path
        run = owned_path(mount, output)
    binding = None
    if reader and request['plan']['phase'] == c.modes.RECOVERY_LEARNING:
        from . import qualification_paths as paths
        binding = paths.observe(request, mount, volume.object_id)
        preflight = registry.get('result:preflight:'+request['plan']['root'])
        if not preflight or binding != paths.preflight_binding(request, preflight):
            raise ValueError('Worker mount differs from same-plan CPU preflight')
        from . import matched_launch
        local = paths.locations(request, binding)
        matched_launch.make_plan(request, runtime.BUNDLE, Path('/opt/agent-lightning'), local['model'],
                                 local['data'], runtime.PYTHON, run, 8000, binding=binding)
    # Unique stage marker is a restart guard. Restarting the same attempt is forbidden.
    run.mkdir(parents=True, exist_ok=False)
    receipt = {'status': 'started', 'call_id': call_id, 'output': output, 'started_at': time.time()}
    if reader:
        receipt['container_id'] = os.environ['MODAL_TASK_ID']
    if binding:
        write(run/'path-binding.json', binding, exclusive=True)
        receipt['path_binding_sha256'] = c.digest(binding)
    write(run/'receipt.json', receipt)
    write(run/'request.json', request, exclusive=True)
    volume.commit()
    stopped = threading.Event()
    owner_lost = threading.Event()
    def heartbeat():
        while not stopped.is_set():
            try:
                lease = registry.get('lease:'+supervisor.key(request['plan'])) or {}
                if registry.get('stop:'+supervisor.key(request['plan'])) or time.time()-lease.get('time',0)>180:
                    owner_lost.set()
                if reader:
                    from .reader_assets import storage_exceeded
                    if storage_exceeded(request['plan']['profile']):
                        owner_lost.set()
                registry.put('heartbeat:'+call_id, {'call_id': call_id, 'time': time.time(), 'output': output})
            except Exception:
                owner_lost.set()  # Lost control-plane access must stop the trainer.
                return
            stopped.wait(15)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        if binding:
            from . import request_transport
            env_binding = {**request_transport.environment(request, run), 'AGL_PPO_RUN_DIR': str(run)}
            if request_transport.load(env_binding) != request:
                raise ValueError('Worker request transport identity differs')
        scratch = shutil.disk_usage('/tmp').free
        write(run/'resource-preflight.json', {'scratch_free_bytes':scratch, 'requested':
            {'gpu':'H100!', 'cpu':16, 'memory_mib':196608}, 'concurrency_limit':request['plan']['bounds']['concurrency']})
        if (request['plan']['phase']=='campaign' or (reader or historical.exploratory.is_exploratory(request['plan']))) and scratch < 100*1024**3:
            raise ValueError('Worker scratch below approved 100 GiB minimum')
        if reader:
            verify_package(request['plan'])
            capture_build(run, image_id)
        else:
            for relative, expected in request['plan']['source']['files'].items():
                if relative.startswith('experiment/'):
                    if c.file_hash(runtime.BUNDLE/relative.removeprefix('experiment/')) != expected:
                        raise ValueError(f'Packaged source differs: {relative}')
        entity = request['stage']['wandb']['entity']
        env = runtime.environment({'wandb_entity': entity, 'weave': False})
        if reader:
            env.update(AGL_READER_CALL_ID=call_id, WANDB_PROJECT=request['stage']['wandb']['project'])
            runtime.run_command([runtime.PYTHON, runtime.BUNDLE/'check_gpu.py', '--output', run/'gpu-preflight.json'],
                                run/'gpu-preflight.log', env, min(600, deadline_epoch-time.time()-120),
                                cancelled=owner_lost.is_set)
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        command_timeout = limit-60
        if (reader or historical.exploratory.is_exploratory(request['plan'])):
            env['AGL_MATCHED_DEADLINE'] = str(deadline_epoch)
            command_timeout = deadline_epoch-time.time()-60
            if command_timeout <= 0:
                raise TimeoutError('Stage budget exhausted during startup/source hashing')
        runtime.run_command([runtime.PYTHON, runtime.BUNDLE/'matched_launch.py', '--request', run/'request.json'],
                         run/'launcher.log', env, command_timeout,
                         cancelled=owner_lost.is_set)
        completion = json.loads((run/'completion.json').read_text())
        if completion.get('plan_sha256') != request['plan_sha256']:
            raise ValueError('Completion source/plan mismatch')
        receipt.update(status='completed', completion=completion)
    except BaseException as exc:
        receipt.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        stopped.set(); thread.join(timeout=20)
        receipt['finished_at'] = time.time()
        write(run/'receipt.json', receipt)
        volume.commit()
    return receipt


def supervise(plan, *, c, registry, volume, modal, stage_worker, image_id=None):
    c.validate(plan)
    reader = plan.get('schema') in ('reader-reference-v1', 'reader-task-v2')
    study_key = supervisor.key(plan)
    owner = {'call_id': modal.current_function_call_id(), 'plan_sha256': c.digest(plan)}
    if reader and plan['phase'] == c.modes.RECOVERY_LEARNING:
        preflight = registry.get('result:preflight:'+plan['root'])
        owner.update(volume_id=volume.object_id, preflight_sha256=c.digest(preflight))
    if not registry.put('owner:'+study_key, owner, skip_if_exists=True):
        raise ValueError('Attempt already owned; inspect it and use a new attempt ID')
    path = Path('/data/phase2/studies')/plan['study_id']/plan['phase']/f"supervisor-a{plan['attempt']:02d}"
    if reader:
        from .reader_assets import owned_path
        path = owned_path('/data', plan['root']+'/supervisor')
    volume.reload(); path.mkdir(parents=True, exist_ok=False)
    write(path/'plan.json', plan, exclusive=True)
    def persist(index):
        index['owner'] = owner
        registry.put('lease:'+study_key, {'time':time.time(), 'owner':owner})
        registry.put('index:'+study_key, index)
        write(path/'index.json', index)
        volume.commit()
    def dispatch(job, stage):
        async def bounded_spawn():
            limit = c.stage_limit(plan,stage)
            options = {'timeout':limit, 'max_containers':plan['bounds']['concurrency']}
            kwargs = {}
            if (reader or historical.exploratory.is_exploratory(plan)):
                options.update(cpu=(16,16), memory=(196608,196608))
                kwargs['deadline_epoch'] = c.dispatch_deadline(limit, time.time()) if reader else time.time()+limit
            if reader:
                kwargs['image_id'] = image_id
            call = await asyncio.wait_for(stage_worker.with_options(**options).spawn.aio(
                c.stage_request(plan, job, stage), **kwargs), timeout=30)
            return call.object_id
        return asyncio.run(bounded_spawn())
    return supervisor.run(plan, dispatch,
        lambda call: modal.FunctionCall.from_id(call).get(timeout=0),
        lambda call: modal.FunctionCall.from_id(call).cancel(terminate_containers=True),
        lambda call: registry.get('heartbeat:'+call),
        lambda: bool(registry.get('stop:'+study_key)), persist)
