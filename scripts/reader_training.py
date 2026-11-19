#!/usr/bin/env python3
"""Reader-owned calculator workflow. plan/validate are strictly offline."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment import reader_contract as c, reader_assets as assets, matched_supervisor as s


def account(modal, profile):
    """A local profile name is never accepted as proof of remote ownership."""
    if os.environ.get('MODAL_PROFILE') != profile['modal']['profile']:
        raise ValueError('Select the explicit Modal profile from this configuration')
    workspace = modal.Workspace.from_context().hydrate()
    if workspace.name != profile['modal']['workspace']:
        raise ValueError('Authenticated workspace differs; no resource operation attempted')
    return {'workspace': workspace.name, 'verified_at': time.time(), 'method': 'Workspace.from_context().hydrate()'}


def handles(modal, profile):
    p = profile['modal']
    volume = modal.Volume.from_name(p['volume'], environment_name=p['environment'], create_if_missing=False).hydrate()
    registry = modal.Dict.from_name(p['registry'], environment_name=p['environment'], create_if_missing=False).hydrate()
    modal.Secret.from_name(p['secret'], environment_name=p['environment'], required_keys=['HF_TOKEN', 'WANDB_API_KEY']).hydrate()
    return registry, volume


def submit(plan, action, registry, spawn, out, ownership):
    """Persist intent before spawn. Lost acknowledgement remains unknown, never retried."""
    if out.exists():
        raise ValueError('Submission receipt exists; inspect it instead of resubmitting')
    key = action+':'+plan['root']
    row = {'status': 'unknown', 'action': action, 'plan_sha256': c.digest(plan), 'ownership': ownership,
           'time': time.time(), 'call_id': None, 'automatic_retry': False}
    if not registry.put('submission:'+key, row, skip_if_exists=True):
        raise ValueError('Durable submission reservation exists; outcome may be unknown')
    assets.write_new(out, row)
    try:
        submission = spawn()
        row.update(status='submitted', **(submission if isinstance(submission, dict) else {'call_id': submission}))
        registry.put('submission:'+key, row)
    except BaseException as exc:
        row['error_type'] = type(exc).__name__  # never copy credential-bearing provider errors
        try:
            registry.put('submission:'+key, row)
        except Exception:
            pass
        raise
    finally:
        # Replace only the receipt just exclusively created by this invocation.
        out.write_text(json.dumps(row, indent=2)+'\n')
    return row


def cancel_owned(plan, registry, cancel):
    sha = c.digest(plan)
    calls = set()
    # Preparation uses an earlier plan without assets: cancel it with that exact plan.
    for action in ('prepare', 'preflight', 'launch'):
        for prefix in ('submission:', 'entered:'):
            row = registry.get(prefix+action+':'+plan['root']) or {}
            if row.get('plan_sha256') == sha and row.get('call_id'):
                calls.add(row['call_id'])
    index = registry.get('index:'+s.key(plan))
    if index:
        if index['plan'] != plan:
            raise ValueError('Remote supervisor identity differs')
        calls.update(s.owned_calls(index))
        if index.get('owner', {}).get('plan_sha256') == sha:
            calls.add(index['owner']['call_id'])
    for job in plan['jobs']:
        for stage in job['stages']:
            row = registry.get('stage:'+sha+':'+stage['wandb']['id']) or {}
            if row.get('call_id'):
                calls.add(row['call_id'])
    if any(not isinstance(x, str) or not x.startswith('fc-') for x in calls):
        raise ValueError('Invalid owned call ID')
    registry.put('stop:reader:'+sha, {'plan_sha256': sha, 'time': time.time()})
    training = registry.get('submission:launch:'+plan['root']) or {}
    if index or training.get('plan_sha256') == sha:
        registry.put('stop:'+s.key(plan), {'plan_sha256': sha, 'time': time.time()})
    errors = []
    for call in sorted(calls):
        try:
            cancel(call)
        except Exception as exc:
            errors.append({'call_id': call, 'error_type': type(exc).__name__})
    return {'status': 'cancellation_requested', 'owned_calls': sorted(calls), 'errors': errors,
            'unknown_submissions': 'Reservation retained; no call ID is not proof of no work'}


def status(plan, registry, read):
    report = {'plan_sha256': c.digest(plan), 'evidence_complete': False, 'operations': {}}
    for action in ('prepare', 'preflight', 'launch'):
        key = action+':'+plan['root']
        report['operations'][action] = {'submission': registry.get('submission:'+key), 'result': registry.get('result:'+key)}
    index = registry.get('index:'+s.key(plan))
    report['index'] = index
    if index:
        try:
            from experiment.reader_evidence import validate_collection
            report.update(validate_collection(plan, index, read))
        except (ValueError, KeyError, OSError, TypeError) as exc:
            report['incomplete_reason'] = type(exc).__name__+': '+str(exc)
    return report


def collect(plan, volume, out, report):
    out.mkdir(parents=True, exist_ok=False)
    size, errors = 0, []
    ceiling = plan['profile']['bounds']['collection_gib']*1024**3
    for entry in volume.iterdir(plan['root'], recursive=True):
        relative = Path(entry.path.lstrip('/'))
        if '..' in relative.parts or not relative.is_relative_to(plan['root']):
            raise ValueError('Volume returned an unowned path')
        if relative.suffix not in ('.json', '.jsonl', '.log', '.txt') or 'temporary' in relative.parts:
            continue
        if size+getattr(entry, 'size', 0) > ceiling:
            raise ValueError('Collection ceiling reached; partial evidence retained')
        dest = out/relative.relative_to(plan['root'])
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with dest.open('xb') as f:
                for chunk in volume.read_file(entry.path):
                    size += len(chunk)
                    if size > ceiling:
                        raise ValueError('Collection ceiling reached')
                    f.write(chunk)
        except Exception as exc:
            errors.append({'path': str(relative), 'error_type': type(exc).__name__})
    report.update(collected_bytes=size, collection_errors=errors)
    report['evidence_complete'] = report['evidence_complete'] and not errors
    assets.write_new(out/'collection.json', report)


def online(args, plan, profile):
    if profile['fixture']:
        raise ValueError('Synthetic fixture cannot cross the online boundary')
    if args.command in ('setup', 'prepare', 'preflight', 'launch'):
        c.validate_current(plan, profile, training=args.command in ('preflight', 'launch'))
    import modal  # Explicit online boundary; app import happens much later.
    ownership = account(modal, profile)
    binding = profile['modal']
    if args.command == 'setup':
        c.validate_current(plan, profile)
        # No Secret creation or credential reads. The owner configures the named Secret normally.
        modal.Secret.from_name(binding['secret'], environment_name=binding['environment'], required_keys=['HF_TOKEN', 'WANDB_API_KEY']).hydrate()
        modal.Volume.objects.create(binding['volume'], environment_name=binding['environment'], allow_existing=False)
        modal.Dict.objects.create(binding['registry'], environment_name=binding['environment'], allow_existing=False)
        return {'status': 'resources_created', 'ownership': ownership, 'secret': 'existing handle verified'}
    registry, volume = handles(modal, profile)
    ownership.update(environment=binding['environment'], volume_id=volume.object_id,
                     registry_id=registry.object_id, volume_backend_version=volume._get_metadata().version)
    def read(name):
        return json.loads(b''.join(volume.read_file(name)))
    if args.command == 'stop-owned':
        return cancel_owned(plan, registry, lambda call: modal.FunctionCall.from_id(call).cancel(terminate_containers=True))
    if args.command in ('status', 'collect'):
        report = status(plan, registry, read)
        if args.command == 'collect':
            if args.out is None:
                raise ValueError('collect requires a fresh --out directory')
            collect(plan, volume, args.out, report)
        return report
    c.validate_current(plan, profile, training=args.command in ('preflight', 'launch'))
    if args.out is None or args.out.exists():
        raise ValueError('Execution requires a new --out submission receipt')
    if args.command == 'launch':
        preflight = registry.get('result:preflight:'+plan['root']) or {}
        if (preflight.get('status') != 'completed' or preflight.get('plan_sha256') != c.digest(plan)
                or preflight.get('result', {}).get('status') != 'passed'
                or read(plan['root']+'/preflight/receipt.json') != preflight):
            raise ValueError('Requires durable same-plan CPU preflight; first qualification needs no prior qualification')
        if not registry.put('qualification-allocation', {'plan_sha256': c.digest(plan)}, skip_if_exists=True):
            raise ValueError('Registry qualification allocation already reserved; no blind retries')
    elif args.command == 'prepare':
        if plan['assets'] is not None:
            raise ValueError('Preparation needs the preparation-only plan, without assets')
        if not registry.put('asset-allocation', {'plan_sha256': c.digest(plan), 'paths': profile['paths']}, skip_if_exists=True):
            raise ValueError('Asset paths already reserved; inspect partial preparation')
    if registry.get('stop:reader:'+c.digest(plan)):
        raise ValueError('Attempt stopped; cannot restart')
    # Full inventory rejects ambiguous preexisting outputs before image resolution/submission.
    existing = [x.path.lstrip('/') for x in volume.iterdir('/', recursive=True)]
    targets = ([profile['paths'][k] for k in ('model', 'data', 'staging')] if args.command == 'prepare'
               else [j['output_root'] for j in plan['jobs']] if args.command == 'launch'
               else [plan['root']+'/preflight'])
    if any(n == t or n.startswith(t+'/') for n in existing for t in targets):
        raise ValueError('Remote path collision; preserve existing evidence and choose a fresh allocation')
    os.environ['AGL_READER_PROFILE'] = json.dumps(profile)
    def spawn():
        import reader_app
        with modal.enable_output(), reader_app.app.run(detach=True, environment_name=binding['environment']):
            image_id = reader_app.image.object_id
            if args.command == 'launch':
                call = reader_app.supervise.with_options(timeout=plan['bounds']['supervisor_seconds']).spawn(plan, image_id=image_id)
            else:
                limit = profile['bounds']['preparation_seconds' if args.command == 'prepare' else 'preflight_seconds']
                call = reader_app.cpu_action.with_options(timeout=limit).spawn(plan, args.command, c.dispatch_deadline(limit, time.time()), image_id=image_id)
            return {'call_id': call.object_id, 'image_id': image_id, 'app_id': reader_app.app.app_id}
    return submit(plan, args.command, registry, spawn, args.out, ownership)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'validate', 'setup', 'prepare', 'preflight', 'launch', 'status', 'collect', 'stop-owned'))
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    profile = assets.read(args.profile)
    c.validate_profile(profile)
    if args.command == 'plan':
        if not args.out or args.plan:
            raise ValueError('plan requires new --out, not --plan')
        plan = c.make_plan(profile, c.source_identity(), assets.read(args.assets) if args.assets else None)
        assets.write_new(args.out, plan)
    else:
        if not args.plan or args.assets:
            raise ValueError('Command requires --plan; --assets is only accepted by plan')
        plan = assets.read(args.plan)
    c.validate(plan)
    if profile != plan['profile']:
        raise ValueError('Configuration changed; regenerate plan')
    if args.command in ('plan', 'validate'):
        current = c.source_identity()
        if plan['source'] != current:
            raise ValueError('Stale source; regenerate plan')
        result = {'valid': True, 'offline': True, 'plan_sha256': c.digest(plan),
                  'configuration_sha256': plan['configuration_sha256'], 'source_sha256': c.digest(plan['source']),
                  'training_admissible': plan['training_admissible'], 'source_clean': not current['dirty'],
                  'ownership': 'UNVERIFIED', 'availability': 'UNVERIFIED',
                  'assets': 'manifest-bound; actual bytes must pass remote preflight' if plan['assets'] else 'not acquired/verified'}
    else:
        result = online(args, plan, profile)
    print(json.dumps(result, indent=2))
    return 0 if args.command not in ('status', 'collect') or result['evidence_complete'] else 2


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(2)
