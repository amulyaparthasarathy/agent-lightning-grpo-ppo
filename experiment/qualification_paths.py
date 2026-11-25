"""Mount observations and portable validation for recovery qualification stages."""
from pathlib import Path
import re

try:
    from . import reader_contract as contract, recovery_learning
except ImportError:
    import reader_contract as contract
    import recovery_learning


def describe(request, root, volume_id):
    """Reconstruct from the frozen request and a separately observed mount. Never resolves provider paths on the reviewer."""
    plan, job, stage = request['plan'], request['job'], request['stage']
    if request != contract.stage_request(plan, job, stage):
        raise ValueError('Unowned qualification request')
    if (not isinstance(root, str) or not root.startswith('/') or root.startswith('//')
            or str(Path(root)) != root or '..' in Path(root).parts or root == '/'
            or not isinstance(volume_id, str) or not re.fullmatch(r'vo-[a-zA-Z0-9]+', volume_id)):
        raise ValueError('Invalid qualification volume mount')
    run = contract.path(stage['output'])
    assets = plan['profile']['paths']
    data = contract.path(assets['data'])
    training = next(s for s in job['stages'] if s['name'] == 'training')
    checkpoint = contract.path(training['output'])/'checkpoints'/f"global_step_{job['steps']}"
    train, val = recovery_learning.routes(request, data, run)
    paths = dict(model=assets['model'], chat_template=str(contract.path(assets['model'])/'chat_template.jinja'),
                 data=str(data), staging=assets['staging'], run=str(run), binding=str(run/'path-binding.json'),
                 request_file=str(run/'request.json'), audit=str(run/'audit'),
                 resolved_config=str(run/'resolved_config.json'), checkpoints=str(run/'checkpoints'),
                 training_checkpoint=str(checkpoint), training_completion=str(checkpoint.parents[1]/'completion.json'),
                 latest_step=str(checkpoint.parent/'latest_checkpointed_iteration.txt'), train_file=train, eval_file=val)
    return {'root': root, 'volume_id': volume_id, 'volume_name': plan['profile']['modal']['volume'],
            'plan_sha256': request['plan_sha256'], 'request_sha256': contract.digest(request), 'paths': paths}


def observe(request, mount, volume_id):
    root = Path(mount).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('Expected a mounted volume directory')
    record = describe(request, str(root), volume_id)
    for relative in record['paths'].values():
        path = root/relative
        if path.resolve() != path or not path.is_relative_to(root):
            raise ValueError('Qualification path escapes or aliases its owned location')
    return record


def locations(request, record):
    if not isinstance(record, dict) or record != describe(request, record.get('root'), record.get('volume_id')):
        raise ValueError('Qualification path binding differs from the exact stage request')
    return {k: Path(record['root'])/v for k, v in record['paths'].items()}


def preflight_binding(request, receipt):
    if (receipt['status'] != 'completed' or receipt['plan_sha256'] != request['plan_sha256']
            or receipt['result']['status'] != 'passed' or receipt['result']['plan_sha256'] != request['plan_sha256']):
        raise ValueError('Missing same-plan qualification preflight binding')
    return receipt['result']['stage_paths'][request['stage']['wandb']['id']]


def launch_context(request, binding):
    paths = locations(request, binding)
    return {**{key: str(paths[key]) for key in ('model', 'resolved_config', 'train_file', 'eval_file')},
            'request': request, 'path_binding': binding}
