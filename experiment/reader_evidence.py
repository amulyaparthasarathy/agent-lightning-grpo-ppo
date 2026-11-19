"""Additional reader evidence checks, after the shared native completion checks."""
import json
import math
import os
from pathlib import Path
import socket

try:
    from . import reader_contract as c
except ImportError:
    import reader_contract as c


def complete(run, request, completion):
    events = [json.loads(line) for p in (run/'audit').glob('events-*.jsonl') for line in p.read_text().splitlines()]
    stage, job = request['stage'], request['job']
    finished = [e for e in events if e['event'] == 'wandb_finished']
    if len(finished) != 1 or finished[0]['id'] != stage['wandb']['id']:
        raise ValueError('Missing/duplicate/wrong W&B completion identity')
    terminal = [e for e in events if e['event'] == 'trainer_returned']
    if len(terminal) != 1 or terminal[0]['validation_only'] != (stage['name'] != 'training'):
        raise ValueError('Wrong trainer evaluation mode')
    result = {**completion, 'execution_schema': c.SCHEMA,
              'source_sha256': c.digest(request['plan']['source']),
              'configuration_sha256': request['plan']['configuration_sha256'],
              'assets_sha256': c.digest(request['plan']['assets']),
              'task_sha256': c.digest(request['plan']['task_identity']),
              'worker': {'call_id': os.environ['AGL_READER_CALL_ID'], 'hostname': socket.gethostname(),
                         'launcher_pid': os.getpid(), 'trainer_pid': terminal[0]['pid']},
              'wandb_id': stage['wandb']['id']}
    if stage['name'] == 'training':
        roles = ('actor', 'critic') if job['algorithm'] == c.ALGORITHMS[1] else ('actor',)
        checkpoint = json.loads((run/'checkpoints/global_step_2/matched_identity.json').read_text())
        if set(checkpoint['roles']) != set(roles):
            raise ValueError('Unexpected native checkpoint roles')
        for role in roles:
            files = checkpoint['roles'][role]
            required = {f'{kind}_world_size_1_rank_0.pt' for kind in ('model', 'optim', 'extra_state')}
            if not required <= files.keys() or any(files[n]['bytes'] <= 0 for n in required):
                raise ValueError('Incomplete single-rank native checkpoint role inventory')
        if any(e['role'] not in roles for e in events if e['event'] == 'optimizer_step'):
            raise ValueError('Unexpected optimizer role')
        for role in roles:
            counts = sorted(e['count'] for e in events if e['event'] == 'optimizer_step' and e['role'] == role)
            gradients = [e for e in events if e['event'] == 'gradient_step' and e['role'] == role]
            if counts != [1, 2] or len(gradients) != 2 or any(not e['preclip_finite'] or not e['optimizer_called'] for e in gradients):
                raise ValueError('Wrong optimizer sequence or missing/nonfinite gradient evidence')
        collections = [e for e in events if e['event'] == 'matched_advantages']
        if (len(collections) != 2 or len({e['iteration'] for e in collections}) != 2
                or any(e['real_rollouts'] != 32 or e['generated_tokens'] <= 0
                       or not 0 <= e['constant_groups'] <= 8 for e in collections)):
            raise ValueError('Missing/duplicate/incomplete rollout accounting')
        if [e['step'] for e in events if e['event'] == 'native_checkpoint_saved'] != [2]:
            raise ValueError('Missing/extra final native checkpoint save')
        metrics = [json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()]
        losses = [v for row in metrics for k, v in row['metrics'].items() if 'loss' in k]
        if not losses or any(type(v) not in (int, float) or not math.isfinite(v) for v in losses):
            raise ValueError('Missing or nonfinite loss evidence')
        result['learning_diagnostics'] = {'collections': [{k: e[k] for k in ('iteration', 'generated_tokens', 'real_rollouts', 'constant_groups')} for e in collections],
            'informative_reward_groups': sum(8-e['constant_groups'] for e in collections),
            'finite_loss_values': len(losses), 'accuracy_improvement_required': False}
    if stage['name'] == 'checkpoint-eval':
        training = json.loads((run.parent/'training/completion.json').read_text())
        if (training['plan_sha256'] != request['plan_sha256']
                or training['checkpoint_sha256'] != result['checkpoint_sha256']
                or training['worker']['call_id'] == result['worker']['call_id']
                or training['worker']['hostname'] == result['worker']['hostname']):
            raise ValueError('Reload requires the exact training checkpoint in a distinct worker/process')
        result['training_worker'] = training['worker']
    return result


def validate_collection(plan, index, read):
    """Join durable proof with exact supervisor calls; do not trust directory existence."""
    if index['plan'] != plan or index['status'] != 'completed' or set(index['jobs']) != {j['job_id'] for j in plan['jobs']}:
        raise ValueError('Incomplete or mismatched supervisor index')
    calls = set()
    checkpoints = {}
    for job in plan['jobs']:
        receipts = index['jobs'][job['job_id']]['stages']
        if len(receipts) != 3 or [r['stage'] for r in receipts] != ['baseline', 'training', 'checkpoint-eval']:
            raise ValueError('Missing/duplicate stage receipts')
        for stage, entry in zip(job['stages'], receipts, strict=True):
            receipt, proof, obs = (read(stage['output']+'/'+n) for n in ('receipt.json', 'completion.json', 'observability.json'))
            if (entry['status'] != 'completed' or receipt['status'] != 'completed'
                    or entry['output'] != stage['output'] or receipt['output'] != stage['output']
                    or receipt != entry['result'] or proof['execution_schema'] != c.SCHEMA
                    or receipt['call_id'] != entry['call_id'] or receipt['call_id'] in calls
                    or receipt['completion'] != proof or proof != entry['result']['completion']
                    or proof['plan_sha256'] != c.digest(plan) or proof['stage'] != stage['name']
                    or proof['worker']['call_id'] != receipt['call_id']
                    or proof['source_sha256'] != c.digest(plan['source'])
                    or proof['configuration_sha256'] != plan['configuration_sha256']
                    or proof['assets_sha256'] != c.digest(plan['assets'])
                    or proof['task_sha256'] != c.digest(plan['task_identity'])
                    or proof['wandb_id'] != stage['wandb']['id'] or obs['wandb'] != stage['wandb']):
                raise ValueError('Durable receipt/completion/observability identity mismatch')
            calls.add(receipt['call_id'])
            if stage['name'] == 'training':
                checkpoints[job['job_id']] = proof
            elif stage['name'] == 'checkpoint-eval':
                previous = checkpoints[job['job_id']]
                if (proof['checkpoint_sha256'] != previous['checkpoint_sha256']
                        or proof['training_worker'] != previous['worker']
                        or proof['worker']['hostname'] == previous['worker']['hostname']):
                    raise ValueError('Training/reload evidence differs')
    return {'evidence_complete': True, 'live_qualification': 'completed', 'hosted_wandb_history': 'not independently queried'}
