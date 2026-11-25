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
    if reader and c.modes.recovery(request['plan']):
        from . import qualification_paths as paths
        from . import matched_launch
        projected = matched_launch.mounted_stage_plan(request, mount, volume.object_id, runtime.BUNDLE,
                                                     Path('/opt/agent-lightning'), runtime.PYTHON)
        binding = projected['path_binding']
        preflight = registry.get('result:preflight:'+request['plan']['root'])
        if not preflight or binding != paths.preflight_binding(request, preflight):
            raise ValueError('Worker mount differs from same-plan CPU preflight')
    release = None
    study = reader and request['plan']['phase'] == c.modes.STUDY
    if study:
        # The supervisor commits the exact reservation and release before spawning.
        control = owned_path(mount, request['plan']['root']+'/supervisor')
        index = json.loads((control/'index.json').read_text())
        job = request['job']
        entry, = index['jobs'][job['job_id']]['stages']
        owner = registry.get('owner:'+supervisor.key(request['plan']))
        seconds = c.stage_limit(request['plan'], request['stage'])
        if (not owner or index['owner'] != owner or owner['plan_sha256'] != request['plan_sha256']
                or owner['preflight_sha256'] != c.digest(preflight) or owner['volume_id'] != volume.object_id or index['plan'] != request['plan']
                or entry['output'] != output or entry['wandb'] != request['stage']['wandb']
                or entry['status'] not in ('planned','submitted','started') or entry['call_id'] not in (None, call_id)
                or entry['reservation'] != dict(seconds=seconds, usd=seconds*request['plan']['bounds']['all_resource_hourly_ceiling_usd']/3600)):
            raise ValueError('Missing owned study reservation')
        if request['stage']['key'] == 'training-2002' and index.get('first_seed_gate') != c.study.gate(request['plan'], index, 3):
            raise ValueError('First seed technical gate differs')
        if request['stage']['panel'] == 'final':
            release = c.study.read_release(request['plan'], control, index)
            if registry.get('release:'+request['plan_sha256']) != c.digest(release):
                raise ValueError('Missing or inconsistent durable final release')
    # Unique stage marker is a restart guard. Restarting the same attempt is forbidden.
    run.mkdir(parents=True, exist_ok=False)
    receipt = {'status': 'started', 'call_id': call_id, 'output': output, 'started_at': time.time()}
    if reader:
        receipt['container_id'] = os.environ['MODAL_TASK_ID']
    if release is not None:
        receipt['final_release_sha256'] = c.digest(release)
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
    # Long runs keep full episode and tensor inventories on the volume.
    if study:
        return dict(call_id=call_id, output=output, receipt_sha256=c.digest(receipt))
    return receipt


def supervise(plan, *, c, registry, volume, modal, stage_worker, image_id=None, mount=Path('/data')):
    c.validate(plan)
    reader = plan.get('schema') in ('reader-reference-v1', 'reader-task-v2')
    study_key = supervisor.key(plan)
    owner = {'call_id': modal.current_function_call_id(), 'plan_sha256': c.digest(plan)}
    if reader and c.modes.recovery(plan):
        preflight = registry.get('result:preflight:'+plan['root'])
        owner.update(volume_id=volume.object_id, preflight_sha256=c.digest(preflight))
    if not registry.put('owner:'+study_key, owner, skip_if_exists=True):
        raise ValueError('Attempt already owned; inspect it and use a new attempt ID')
    path = Path('/data/phase2/studies')/plan['study_id']/plan['phase']/f"supervisor-a{plan['attempt']:02d}"
    if reader:
        from .reader_assets import owned_path
        path = owned_path(mount, plan['root']+'/supervisor')
    volume.reload(); path.mkdir(parents=True, exist_ok=False)
    write(path/'plan.json', plan, exclusive=True)
    def persist(index):
        index['owner'] = owner
        registry.put('lease:'+study_key, {'time':time.time(), 'owner':owner})
        if reader and plan['phase'] == c.modes.STUDY:
            release = index.get('final_release')
            if release and not (path/'final-release.json').exists():
                write(path/'final-release.json', release, exclusive=True)
                volume.commit()
                if not registry.put('release:'+c.digest(plan), c.digest(release), skip_if_exists=True):
                    raise ValueError('Release already exists')
            write(path/'index.pending', index)
            (path/'index.pending').replace(path/'index.json')
            registry.put('index:'+study_key, {'plan':plan, 'volume_index':True, 'owner':owner})
        else:
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
    def get_result(call):
        result = modal.FunctionCall.from_id(call).get(timeout=0)
        if reader and plan['phase'] == c.modes.STUDY:
            outputs = {s['output'] for j in plan['jobs'] for s in j['stages']}
            if result['output'] not in outputs or result['call_id'] != call:
                raise ValueError('Unowned study worker result')
            volume.reload()
            receipt = json.loads((owned_path(mount, result['output'])/'receipt.json').read_text())
            if c.digest(receipt) != result['receipt_sha256']:
                raise ValueError('Durable study worker receipt changed')
            return receipt
        return result
    return supervisor.run(plan, dispatch,
        get_result,
        lambda call: modal.FunctionCall.from_id(call).cancel(terminate_containers=True),
        lambda call: registry.get('heartbeat:'+call),
        lambda: bool(registry.get('stop:'+study_key)), persist)
