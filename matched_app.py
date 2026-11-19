"""Separately launched matched study. Import never submits jobs or creates Secrets."""
from __future__ import annotations
import json
import asyncio
import os
from pathlib import Path
import shlex
import subprocess
import shutil
import threading
import time

import modal
# Explicit existing Stanford binding, never the protected source Volume.
if os.environ.get('AGL_WORKING_VOLUME', 'cs329z-agent-lightning-working-stanford-v2') != 'cs329z-agent-lightning-working-stanford-v2':
    raise ValueError('Matched studies require the existing Stanford v2 working Volume')
os.environ['AGL_WORKING_VOLUME'] = 'cs329z-agent-lightning-working-stanford-v2'
# Runtime imports must work without the local application or build scripts.
import sys
BUNDLE = Path('/opt/agent-lightning-lab/experiment')
if not (Path(__file__).resolve().parent/'experiment').is_dir():
    sys.path.insert(0, str(BUNDLE.parent))
from experiment import matched_container as runtime
from experiment import matched_contract as c, matched_supervisor as supervisor

LOCAL = Path(__file__).resolve().parent
app = modal.App('cs329z-matched-study-v2')
image = None
if modal.is_local():
    # Reuse the existing CUDA/VERL layers; apply matched-only changes afterward.
    import modal_app as base
    image = base.image
    # The cached Stanford stack resolved Transformers 5.23.0, but the reviewed
    # model-loading code is 5.22.2. Restore that exact wheel only for matched
    # jobs, after the CUDA layers; never relax the runtime source hash guard.
    wheel = json.loads((LOCAL/'experiment/ppo_source_lock.json').read_text())['transformers_wheel']
    image = image.run_commands(
        'uv pip install --python /opt/agent-lightning/.venv/bin/python --no-deps --no-build '
        + shlex.quote(wheel['url']+'#sha256='+wheel['digests']['sha256']))
    for path in sorted((LOCAL/'experiment').rglob('*')):
        if (path.is_file() and path.suffix in ('.py', '.json', '.jinja', '.txt')
                and '__pycache__' not in path.parts and str(path.relative_to(LOCAL/'experiment')) not in base.RUNTIME_FILES):
            image = image.add_local_file(path, str(BUNDLE/path.relative_to(LOCAL/'experiment')), copy=True)
credentials = modal.Secret.from_name('agent-lightning-credentials', required_keys=['HF_TOKEN', 'WANDB_API_KEY'])
volume = modal.Volume.from_name(c.VOLUME, environment_name='main', create_if_missing=False)
registry = modal.Dict.from_name('cs329z-matched-study-v2', environment_name='main', create_if_missing=True)


def write(path, data, exclusive=False):
    with path.open('x' if exclusive else 'w') as f:
        json.dump(data, f, indent=2)
        f.write('\n')


@app.function(image=image, gpu='H100!', cpu=16, memory=196608, timeout=14400,
              retries=0, max_containers=10, single_use_containers=True,
              volumes={'/data': volume}, secrets=[credentials])
def stage_worker(request, deadline_epoch=None):
    entered_at = time.time()
    c.validate(request['plan'])
    if request != c.stage_request(request['plan'], request['job'], request['stage']):
        raise ValueError('Invalid stage request')
    if registry.get('stop:'+supervisor.key(request['plan'])):
        raise ValueError('Attempt stopped before worker entry')
    limit = c.stage_limit(request['plan'],request['stage'])
    if c.exploratory.is_exploratory(request['plan']):
        if deadline_epoch is None or not entered_at < deadline_epoch <= entered_at+limit:
            raise ValueError('Missing, expired or extended exploratory stage deadline')
    call_id = modal.current_function_call_id()
    stage_key = 'stage:'+request['plan_sha256']+':'+request['stage']['wandb']['id']
    if not registry.put(stage_key, {'call_id':call_id}, skip_if_exists=True):
        raise ValueError('Stage already entered; automatic restart forbidden')
    output = request['stage']['output']
    run = Path('/data')/output
    # Unique stage marker is a restart guard. Restarting the same attempt is forbidden.
    volume.reload()
    run.mkdir(parents=True, exist_ok=False)
    receipt = {'status': 'started', 'call_id': call_id, 'output': output, 'started_at': time.time()}
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
                registry.put('heartbeat:'+call_id, {'call_id': call_id, 'time': time.time(), 'output': output})
            except Exception:
                owner_lost.set()  # Lost control-plane access must stop the trainer.
                return
            stopped.wait(15)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        scratch = shutil.disk_usage('/tmp').free
        write(run/'resource-preflight.json', {'scratch_free_bytes':scratch, 'requested':
            {'gpu':'H100!', 'cpu':16, 'memory_mib':196608}, 'concurrency_limit':request['plan']['bounds']['concurrency']})
        if (request['plan']['phase']=='campaign' or c.exploratory.is_exploratory(request['plan'])) and scratch < 100*1024**3:
            raise ValueError('Worker scratch below approved 100 GiB minimum')
        for relative, expected in request['plan']['source']['files'].items():
            if relative.startswith('experiment/'):
                if c.file_hash(runtime.BUNDLE/relative.removeprefix('experiment/')) != expected:
                    raise ValueError(f'Packaged source differs: {relative}')
        env = runtime.environment({'wandb_entity': 'amulyap-stanford-university', 'weave': False})
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        command_timeout = limit-60
        if c.exploratory.is_exploratory(request['plan']):
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


@app.function(image=image, cpu=1, memory=2048, timeout=43800, retries=0,
              max_containers=1, single_use_containers=True, volumes={'/data': volume})
def supervise(plan):
    c.validate(plan)
    study_key = supervisor.key(plan)
    owner = {'call_id': modal.current_function_call_id(), 'plan_sha256': c.digest(plan)}
    if not registry.put('owner:'+study_key, owner, skip_if_exists=True):
        raise ValueError('Attempt already owned; inspect it and use a new attempt ID')
    path = Path('/data/phase2/studies')/plan['study_id']/plan['phase']/f"supervisor-a{plan['attempt']:02d}"
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
            if c.exploratory.is_exploratory(plan):
                options.update(cpu=(16,16), memory=(196608,196608))
                kwargs['deadline_epoch'] = time.time()+limit
            call = await asyncio.wait_for(stage_worker.with_options(**options).spawn.aio(
                c.stage_request(plan, job, stage), **kwargs), timeout=30)
            return call.object_id
        return asyncio.run(bounded_spawn())
    return supervisor.run(plan, dispatch,
        lambda call: modal.FunctionCall.from_id(call).get(timeout=0),
        lambda call: modal.FunctionCall.from_id(call).cancel(terminate_containers=True),
        lambda call: registry.get('heartbeat:'+call),
        lambda: bool(registry.get('stop:'+study_key)), persist)
