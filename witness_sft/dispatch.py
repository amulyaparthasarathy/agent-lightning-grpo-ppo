"""Offline plan and disabled, one-use owned dispatch for exactly three SFT workers."""
import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import shutil
import tempfile
import time

from experiment import reader_assets, reader_contract
from . import contract, data, evaluation


def owner_fields(owner):
    reader_contract.exact(owner, 'modal base_path image_id registry_id output_id base_id', 'SFT owner')
    reader_contract.exact(owner['modal'], 'profile workspace environment volume base_volume registry app', 'SFT handles')
    for value in owner['modal'].values():
        reader_contract.slug(value)
    for key, prefix in [('image_id', 'im'), ('registry_id', 'di'), ('output_id', 'vo'), ('base_id', 'vo')]:
        if not re.fullmatch(prefix + '-[A-Za-z0-9]+', owner[key]):
            raise ValueError('Invalid provider identity')
    from experiment.reader_source import safe_path
    safe_path(owner['base_path'])
    if owner['output_id'] == owner['base_id'] or owner['modal']['volume'] == owner['modal']['base_volume']:
        raise ValueError('Output volume must be dedicated to the SFT allocation')


def dispatch(plan, owner, permit, out):
    contract.authorize(plan, owner, permit)
    owner_fields(owner)
    if plan != contract.current_plan():
        raise ValueError('Reservation requires current clean source')
    import modal
    from scripts.reader_training import account
    ownership = account(modal, owner)
    p = owner['modal']
    def handle(cls, name, key):
        obj = cls.from_name(p[name], environment_name=p['environment'], create_if_missing=False).hydrate()
        if obj.object_id != owner[key]:
            raise ValueError('Authenticated handle differs from reservation')
        return obj

    volume = handle(modal.Volume, 'volume', 'output_id')
    base_volume = handle(modal.Volume, 'base_volume', 'base_id')
    registry = handle(modal.Dict, 'registry', 'registry_id')
    commitment = data.task.digest(plan)
    prefix = 'witness-sft:' + commitment
    out.mkdir(parents=True, exist_ok=False)
    opened = datetime.now(timezone.utc)
    opening = {'status': 'reserved', 'gross_usd': plan['ceiling_usd'], 'ownership': ownership,
               'opened_utc': opened.isoformat(), 'retention_review_utc': (opened+timedelta(days=7)).isoformat(),
               'plan_sha256': commitment, 'replacement': False}
    reader_assets.write_new(out / 'opening.json', opening)
    if not registry.put(prefix, opening, skip_if_exists=True):
        raise ValueError('Allocation already reserved. Reconcile the existing calls.')
    app = modal.App(p['app'])
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp)
        for name, expected in plan['files'].items():
            dest = bundle / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(contract.ROOT / name, dest)
            if reader_contract.file_hash(dest) != expected:
                raise ValueError('Packaged bytes changed')
        # Runtime mount on an identified prebuilt image. No image builder or HF credential.
        image = modal.Image.from_id(owner['image_id']).add_local_dir(bundle, '/opt/witness')
        @app.function(image=image, serialized=True, include_source=False, single_use_containers=True,
                      cpu=(16, 16), memory=(196608, 196608), gpu='H100!', retries=0, timeout=1800, startup_timeout=420,
                      max_containers=1, block_network=True,
                      volumes={'/sft': volume, '/original': base_volume.read_only()})
        def stage_worker(request):
            import json
            import os
            from pathlib import Path
            import sys
            import time
            import modal
            sys.path.insert(0, '/opt/witness')
            from experiment import matched_container, reader_assets, reader_contract
            from witness_sft import artifact, contract, data
            plan, stage, previous = (request[k] for k in ('plan', 'stage', 'previous'))
            contract.authorize(plan, request['owner'], request['permit'])
            contract.check_stage(plan, stage, previous)
            identity = artifact.worker(modal.current_function_call_id(), os.environ.get('MODAL_TASK_ID'))
            prefix = 'witness-sft:' + data.task.digest(plan)
            opening = registry.get(prefix)
            if (not opening or opening['status'] != 'reserved' or registry.get(prefix + ':stop')
                    or request['retention_review_utc'] != opening['retention_review_utc']
                    or time.time() >= request['deadline']):
                raise ValueError('Missing or expired allocation')
            if not registry.put(prefix + ':' + stage, identity, skip_if_exists=True):
                raise ValueError('Stage already claimed')
            for r in previous:
                if registry.get(prefix + ':result:' + r['stage']) != r:
                    raise ValueError('Unconfirmed predecessor receipt')
            volume.reload()
            run = Path('/sft') / data.task.digest(plan) / stage
            run.mkdir(parents=True, exist_ok=False)
            for name, expected in plan['files'].items():
                if reader_contract.file_hash(Path('/opt/witness') / name) != expected:
                    raise ValueError('Worker source mismatch')
            reader_assets.write_new(run / 'request.json', {**request, 'worker': identity})
            def cancelled():
                occupied = reader_assets.storage_bytes({'paths': {'allocation': run.parent.name}}, Path('/sft'))
                return occupied > plan['output_gib'] * 1024**3 or bool(registry.get(prefix + ':stop'))

            env = dict(os.environ, PYTHONPATH='/opt/witness', PYTHONDONTWRITEBYTECODE='1',
                       WITNESS_BASE=str(Path('/original') / request['owner']['base_path']),
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONHASHSEED='2001')
            # -P prevents an unrelated current directory taking import precedence.
            started = time.time()
            try:
                matched_container.run_command([matched_container.PYTHON, '-P', '-m', 'witness_sft.worker', run],
                        run / 'worker.log', env, min(1740, request['deadline'] - time.time()),
                        cancelled=cancelled)
                receipt = {'worker': identity, 'stage': stage, 'plan_sha256': data.task.digest(plan),
                           'started_epoch': started, 'finished_epoch': time.time(),
                           'result': json.loads((run/'result.json').read_text())}
                reader_assets.write_new(run / 'receipt.json', receipt)
            finally:
                volume.commit()
            registry.put(prefix + ':result:' + stage, receipt)
            return receipt

        receipts = []
        with app.run(environment_name=p['environment'], detach=False):
            for stage in contract.STAGES:
                contract.authorize(plan, owner, permit)
                contract.check_stage(plan, stage, receipts)
                start = time.time()
                limit = contract.STARTUP + contract.LIMITS[stage]
                deadline = min(reader_contract.dispatch_deadline(limit, start), permit['body']['expires_epoch'],
                               opened.timestamp() + plan['allocation_seconds'])
                if deadline <= start:
                    raise TimeoutError('Reservation has expired')
                request = dict(plan=plan, owner=owner, permit=permit, previous=receipts, stage=stage,
                               deadline=deadline, retention_review_utc=opening['retention_review_utc'])
                reader_assets.write_new(out / (stage + '-intent.json'), {'issued': start, 'deadline': deadline})
                async def bounded_spawn():
                    return await asyncio.wait_for(stage_worker.spawn.aio(request), min(60, deadline-time.time()))

                call = None
                try:
                    call = asyncio.run(bounded_spawn())
                    reader_assets.write_new(out / (stage + '-call.json'), {'call_id': call.object_id})
                    result = call.get(timeout=max(0, deadline-time.time()))
                    if (result['worker']['call'] != call.object_id or result['stage'] != stage
                            or result != registry.get(prefix + ':result:' + stage)):
                        raise ValueError('Call and receipt identities differ')
                    reader_assets.write_new(out / (stage + '-result.json'), result)
                    receipts.append(result)
                except BaseException as exc:
                    failure = {'status': 'incomplete', 'stage': stage, 'error_type': type(exc).__name__,
                               'call_id': call.object_id if call is not None else None, 'dispatch_remaining': 0,
                               'billing_status': 'hold entire reservation pending exact-call reconciliation',
                               'retention_review_utc': opening['retention_review_utc']}
                    try:
                        reader_assets.write_new(out / 'incomplete.json', failure)
                        registry.put(prefix + ':stop', True)
                    finally:
                        if call is not None:
                            call.cancel(terminate_containers=True)
                    raise
        reload = receipts[-1]['result']
        if (reload['loaded_artifact_sha256'] != receipts[1]['result']['artifact_sha256']
                or reload['summary'] != evaluation.summarize(reload['episodes'])):
            raise ValueError('Terminal reload result differs')
        finished = datetime.now(timezone.utc)
        close = {'stage_receipts_sha256': data.task.digest(receipts), 'behavior': reload['summary'],
                 'dispatch_remaining': 0, 'finished_utc': finished.isoformat(),
                 'retention_review_utc': opening['retention_review_utc'],
                 'billing_status': 'pending reconciliation; retain full reservation'}
        reader_assets.write_new(out / 'completion.json', close)
        registry.put(prefix + ':completion', close)
        return close


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('plan')
    run = sub.add_parser('run')
    for name in ('owner', 'permit', 'out'):
        run.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'plan':
        print(json.dumps(contract.current_plan(), indent=2))
        return
    if contract.OWNER_ED25519_PUBLIC_PEM is None:
        raise SystemExit('No paid witness-SFT reservation')
    print(json.dumps(dispatch(contract.current_plan(), json.loads(args.owner.read_text()),
                              json.loads(args.permit.read_text()), args.out), indent=2))


if __name__ == '__main__':
    main()
