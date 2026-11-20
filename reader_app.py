"""Explicit online reader adapter. Import only after authenticated ownership checks.

The CLI supplies the validated non-secret profile. Nothing falls back to author
resources. Existing Volume/registry/Secret handles never create resources here.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import time

import modal

BUNDLE = Path('/opt/agent-lightning-lab/experiment')
if not (Path(__file__).parent/'experiment').is_dir():
    sys.path.insert(0, str(BUNDLE.parent))
from experiment import reader_contract as c, reader_assets as assets, matched_remote as remote
from experiment import matched_container as runtime

profile = json.loads(os.environ['AGL_READER_PROFILE'])
c.validate_profile(profile)
if profile['fixture']:
    raise ValueError('Synthetic profiles cannot declare remote apps')
binding = profile['modal']
app = modal.App(binding['app'])
image = None
if modal.is_local():
    from modal_image import matched_image
    image = matched_image(reader=True).env({'AGL_READER_PROFILE': json.dumps(profile)})
    for relative in c.packaged(c.source_identity(clean=True)['files']):
        if not relative.startswith('experiment/'):
            image = image.add_local_file(Path(__file__).parent/relative, str(BUNDLE.parent/relative), copy=True)
volume = modal.Volume.from_name(binding['volume'], environment_name=binding['environment'], create_if_missing=False)
registry = modal.Dict.from_name(binding['registry'], environment_name=binding['environment'], create_if_missing=False)
credentials = modal.Secret.from_name(binding['secret'], environment_name=binding['environment'], required_keys=['HF_TOKEN', 'WANDB_API_KEY'])


def check(plan):
    c.validate(plan)
    if plan['profile'] != profile or modal.Workspace.from_context().hydrate().name != binding['workspace']:
        raise ValueError('Remote profile/authenticated workspace differs')
    if registry.get('stop:reader:'+c.digest(plan)):
        raise ValueError('Owned attempt stopped')


@app.function(image=image, gpu='H100!', cpu=(16, 16), memory=(196608, 196608),
              timeout=7200, retries=0, max_containers=1, single_use_containers=True,
              volumes={'/data': volume}, secrets=[credentials])
def stage_worker(request, deadline_epoch, image_id=None):
    check(request['plan'])
    if not request['plan']['training_admissible']:
        raise ValueError('Stage requires a clean non-fixture asset-bound training plan')
    return remote.stage_worker(request, deadline_epoch, c=c, registry=registry, volume=volume,
                               call_id=modal.current_function_call_id(), image_id=image_id)


@app.function(image=image, cpu=(1, 1), memory=(2048, 2048), timeout=43800,
              retries=0, max_containers=1, single_use_containers=True, volumes={'/data': volume})
def supervise(plan, image_id=None):
    check(plan)
    if not plan['training_admissible']:
        raise ValueError('Supervisor requires a clean asset-bound training plan')
    return remote.supervise(plan, c=c, registry=registry, volume=volume, modal=modal, stage_worker=stage_worker, image_id=image_id)


@app.function(image=image, cpu=(4, 4), memory=(16384, 16384), timeout=3600,
              retries=0, max_containers=1, single_use_containers=True,
              volumes={'/data': volume}, secrets=[credentials])
def cpu_action(plan, action, deadline_epoch, image_id=None):
    check(plan)
    if action not in ('prepare', 'preflight'):
        raise ValueError('Unsupported CPU action')
    limit = profile['bounds']['preparation_seconds' if action == 'prepare' else 'preflight_seconds']
    entered_at = time.time()
    if not entered_at < deadline_epoch <= entered_at+limit:
        raise ValueError(f'Missing/expired/extended CPU deadline: deadline={deadline_epoch!r}, entered={entered_at}, limit={limit}')
    call_id = modal.current_function_call_id()
    key = action+':'+plan['root']
    if not registry.put('entered:'+key, {'call_id': call_id, 'plan_sha256': c.digest(plan)}, skip_if_exists=True):
        raise ValueError('CPU action already entered; no restart')
    volume.reload()
    run = assets.owned_path('/data', plan['root']+'/'+action)
    run.mkdir(parents=True, exist_ok=False)
    # owned_path resolves Modal's /data symlink; receipts retain the logical Volume path.
    receipt = {'status': 'started', 'action': action, 'plan_sha256': c.digest(plan),
               'call_id': call_id, 'output': plan['root']+'/'+action,
               'deadline': {'epoch': deadline_epoch, 'entered_at': entered_at, 'limit_seconds': limit}}
    assets.write_new(run/'plan.json', plan)
    remote.write(run/'receipt.json', receipt)
    volume.commit()
    try:
        remote.verify_package(plan)
        remote.capture_build(run, image_id)
        env = runtime.environment({'wandb_entity': profile['wandb']['entity'], 'weave': False})
        env['WANDB_PROJECT'] = profile['wandb']['project']
        env['HF_HOME'] = str(run/'temporary/huggingface')
        script = 'reader_assets.py' if action == 'prepare' else 'reader_preflight.py'
        runtime.run_command([runtime.PYTHON, BUNDLE/script, run/'plan.json', run], run/'action.log',
                            env, deadline_epoch-time.time()-30,
                            cancelled=lambda: bool(registry.get('stop:reader:'+c.digest(plan))) or assets.storage_exceeded(profile))
        result = assets.read(run/('assets.json' if action == 'prepare' else 'preflight.json'))
        receipt.update(status='completed', result=result)
    except BaseException as exc:
        receipt.update(status='failed', error_type=type(exc).__name__)
        raise
    finally:
        remote.write(run/'receipt.json', receipt)
        volume.commit()
        registry.put('result:'+key, receipt)
    return receipt
