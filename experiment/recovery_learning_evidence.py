"""Recompute qualification from the complete episode and event streams on worker and reader."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import hashlib

try:
    from . import reader_contract as c, reader_evidence, reader_integration, recovery_calibration as cal, recovery_learning as learning, summarize
except ImportError:
    import reader_contract as c
    import reader_evidence
    import reader_integration
    import recovery_calibration as cal
    import recovery_learning as learning
    import summarize


def collection(event, manifest, batch, episodes):
    """Check the adapter's ordered question slots against the raw, independently graded traces."""
    entries = event['episodes']
    rows = {e['rollout_id']: e for e in episodes if e['case_id'] in batch}
    original = {r['rollout']: r for r in manifest['trajectories']}
    rep = event['representation']
    groups = defaultdict(set)
    if (len(entries) != 32 or len(rows) != 32 or len(manifest['trajectories']) != 32
            or set(original) != set(rows) or {r['rollout'] for r in entries} != set(rows)
            or set(rep['fragments_per_rollout']) != set(rows)):
        raise ValueError('Unexpected collection membership or rollout count')
    for e in entries:
        rid = e['rollout']
        row, sampled = rows[rid], original[rid]
        i = e['slot']
        if (type(i) != int or not 0 <= i < 8 or batch[i] != e['case_id'] or row['case_id'] != e['case_id']
                or e['reward'] != row['reward'] or any(e[k] != sampled[k] for k in ('case_id', 'reward', 'group'))
                or sum(row['completion_tokens']) != sampled['original_generated_tokens']
                or sampled['fragments'] != rep['fragments_per_rollout'][rid] or sampled['fragments'] < 1):
            raise ValueError('Collection/trace slot, probability target count or reward differs')
        groups[e['group']].add(row['case_id'])
    total = sum(sum(r['completion_tokens']) for r in rows.values())
    lengths = rep['active_lengths']
    tensors = rep['tensors']
    n = rep['fragment_rows']
    layout = {'input_ids': (5120, 'int64'), 'attention_mask': (5120, 'int64'), 'position_ids': (5120, 'int64'),
              'prompts': (3072, 'int64'), 'responses': (2048, 'int64'), 'response_mask': (2048, 'int64'),
              'token_level_scores': (2048, 'float32'), 'rollout_log_probs': (2048, 'float32'), 'is_drop_mask': (1, 'bool')}
    for key, (width, dtype) in layout.items():
        size = {'bool': 1, 'int64': 8, 'float32': 4}[dtype]
        expected = {'shape': [n] if key == 'is_drop_mask' else [n,width],
                    'dtype': 'torch.'+dtype, 'bytes': n*width*size}
        if tensors[key] != expected:
            raise ValueError('Invalid adapter tensor inventory: '+key)
    context = rep['masked_response_context_tokens']
    if type(context) != int or not 0 <= context <= sum(lengths)-total-n:
        raise ValueError('Invalid attended prompt/context accounting')
    if (len(groups) != 8 or any(len(g) != 1 for g in groups.values())
            or rep['target_tokens'] != total or rep['generated_tokens'] != total
            or rep['fragment_rows'] != sum(rep['fragments_per_rollout'].values())
            or len(lengths) != rep['fragment_rows'] or not 0 < min(lengths) <= max(lengths) <= 4096
            or rep['max_active_position'] != max(lengths)-1 or manifest['adapter_metrics']['training/n_truncated_sample']):
        raise ValueError('Invalid fragment or group accounting')
    return sum(len({e['reward'] for e in rows.values() if e['case_id'] == cid}) > 1 for cid in batch)


def stage_report(run, request, events=None, evidence_class='live-model'):
    run = Path(run)
    events = cal.read_events(run) if events is None else events
    spec = learning.stage_spec(request)
    if json.loads((run/'qualification-inputs.json').read_text()) != spec['inputs']:
        raise ValueError('Eager loader input identities differ')
    try:
        from . import matched_launch
    except ImportError:
        import matched_launch
    paths = request['plan']['profile']['paths']
    remote = Path('/data')/request['stage']['output']
    train, val = learning.routes(request, Path('/data')/paths['data'], remote)
    resolved = run/'resolved_config.json'
    matched_launch.validate_resolved(json.loads(resolved.read_text()), {
        'request': request, 'model': str(Path('/data')/paths['model']), 'train_file': train,
        'eval_file': val, 'resolved_config': str(remote/'resolved_config.json')})
    name = request['stage']['name']
    training, probe = name == 'training', name == 'collection-probe'
    steps = spec['updates']
    rows = cal.episode_table(run, request, evidence_class)
    cal.check_table(spec, rows)
    by_event = defaultdict(list)
    for event in sorted(events, key=lambda e: e['time']):
        by_event[event['event']].append(event)
    returns = by_event['trainer_returned']
    if ([e['id'] for e in by_event['wandb_finished']] != [request['stage']['wandb']['id']]
            or len(returns) != 1 or returns[0]['global_step'] != (1 if probe else spec['initial_step']+steps)
            or returns[0]['validation_only'] is not (spec['route'] == 'val')
            or bool(returns[0].get('collection_only')) != probe
            or (probe and returns[0].get('optimizer_updates') != 0)
            or by_event['advantages'] or (not training and by_event['critic_preupdate'])
            or [e['step'] for e in by_event['weight_refresh']] != list(range(spec['initial_step'], spec['initial_step']+steps+1))
            or [e['step'] for e in by_event['native_checkpoint_saved']] != ([2] if training else [])
            or len(by_event['development_collection_admitted']) != int(probe)):
        raise ValueError('Unexpected trainer, refresh, finish or checkpoint sequence')
    roles = ('actor', 'critic') if request['job']['algorithm'] == c.ALGORITHMS[1] else ('actor',)
    critic = by_event['critic_preupdate']
    if [e['iteration'] for e in critic] != (list(range(1, steps+1)) if 'critic' in roles else []):
        raise ValueError('Missing or unexpected critic pre-update diagnostics')
    for event in critic:
        metrics = event['prediction']
        if (metrics['prediction_time'] != 'fresh-rollout pre-update'
                or any(not math.isfinite(metrics[k]) for k in ('mae', 'mse', 'bias', 'target_variance', 'vf_explained_var_raw'))):
            raise ValueError('Invalid critic prediction diagnostics')
    for key in ('optimizer_step', 'gradient_step'):
        if Counter(e['role'] for e in by_event[key]) != Counter({role: steps for role in roles}):
            raise ValueError('Unexpected gradient or optimizer role/count')
    for key in ('matched_advantages', 'recovery_old_probabilities'):
        if [e['iteration'] for e in by_event[key]] != list(range(1, steps+1)):
            raise ValueError('Unexpected optimization event sequence')
    collections, manifests = by_event['recovery_collection'], by_event['collection_manifest']
    informative = []
    for i, (batch, event, manifest) in enumerate(zip(spec['batches'], collections, manifests, strict=True), 1):
        if event['iteration'] != i or manifest['iteration'] != i or event['policy_step'] != i-1:
            raise ValueError('Unexpected collection or policy version')
        informative.append(collection(event, manifest, batch, rows))
        refresh = by_event['weight_refresh'][i-1]['time']
        if not refresh <= event['time'] <= manifest['time']:
            raise ValueError('Collection precedes its policy refresh')
        if training:
            adv, prob = by_event['matched_advantages'][i-1], by_event['recovery_old_probabilities'][i-1]
            if (adv['constant_groups'] != 8-informative[-1] or adv['real_rollouts'] != 32
                    or adv['generated_tokens'] != event['representation']['target_tokens']
                    or prob['target_tokens'] != adv['generated_tokens'] or not math.isfinite(prob['mean_delta'])
                    or not math.isfinite(prob['max_abs_delta']) or prob['max_abs_delta'] < 0
                    or not manifest['time'] <= prob['time'] <= adv['time']):
                raise ValueError('Advantages or probability evidence differs from admitted collection')
            for role in roles:
                opt = [e for e in by_event['optimizer_step'] if e['role'] == role][i-1]
                grad = [e for e in by_event['gradient_step'] if e['role'] == role][i-1]
                if (opt['count'] != i or not grad['preclip_finite'] or not grad['optimizer_called']
                        or not math.isfinite(grad['preclip_norm'])
                        or not adv['time'] <= opt['time'] <= grad['time'] <= by_event['weight_refresh'][i]['time']):
                    raise ValueError('Gradient/optimizer evidence outside declared update')
    if probe:
        collected = by_event['development_collection_admitted'][0]
        rep = collections[0]['representation']
        if (collected['iteration'] != 1 or collected['real_rollouts'] != 32 or collected['groups'] != 8
                or collected['generated_tokens'] != rep['target_tokens'] or collected['fragment_rows'] != rep['fragment_rows']
                or collected['constant_groups'] != 8-informative[0]):
            raise ValueError('Probe admission summary differs from the single collection')
    losses = []
    if training:
        losses = reader_evidence.loss_evidence([json.loads(l) for l in (run/'metrics.jsonl').read_text().splitlines()], roles)
    return {'schema': 'recovery-qualification-evidence-v1', 'stage': name, 'plan_sha256': request['plan_sha256'],
            'resolved_config_sha256': c.file_hash(resolved), 'declared': spec, 'episodes': rows, 'collections': collections, 'old_probabilities': by_event['recovery_old_probabilities'],
            'optimizer_calls': {role: steps for role in roles}, 'losses': losses, 'critic_preupdate': critic, 'informative_groups': informative,
            'strict_successes': sum(int(e['reward']) for e in rows), 'episodes_admitted': len(rows)}


def review(plan, index, read, local):
    c.validate(plan)
    stages, errors, calls = {}, [], set()
    jobs = index.get('jobs') if isinstance(index, dict) else None
    if (not plan['training_admissible'] or not isinstance(jobs, dict) or index.get('plan') != plan
            or index.get('status') != 'completed' or set(jobs) != {j['job_id'] for j in plan['jobs']}):
        errors.append('Execution plan or supervisor index incomplete/unverified')
    try:
        owner = index['owner']
        if (index['study_key'] != f"{plan['study_id']}/{plan['phase']}/a{plan['attempt']:02d}"
                or owner['plan_sha256'] != c.digest(plan) or not owner['call_id'].startswith('fc-')
                or read(plan['root']+'/supervisor/plan.json') != plan
                or read(plan['root']+'/supervisor/index.json') != index):
            raise ValueError('Supervisor ownership or persisted index differs')
        calls.add(owner['call_id'])
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as exc:
        errors.append('Unverified supervisor: '+str(exc))
    for job in plan['jobs']:
        state = (jobs or {}).get(job['job_id'], {}) if isinstance(jobs, dict) else {}
        entries = state.get('stages', []) if isinstance(state, dict) else []
        if (not isinstance(entries, list) or not isinstance(state, dict) or state.get('status') != 'completed'
                or [e.get('stage') for e in entries if isinstance(e, dict)] != [s['name'] for s in job['stages']]):
            errors.append(job['job_id']+' has incomplete or misordered stage entries')
        training = None
        for stage in job['stages']:
            name = stage['name']
            item = stages[job['algorithm']+'/'+name] = {'admitted': False}
            request = c.stage_request(plan, job, stage)
            entry = next((e for e in entries if isinstance(e, dict) and e.get('stage') == name), {}) if isinstance(entries,list) else {}
            item['supervisor_status'] = entry.get('status', 'unknown')
            try:
                if local is None:
                    raise ValueError('Collected raw traces and audit files required')
                run = Path(local)/Path(stage['output']).relative_to(plan['root'])
                facts, _, attempts = cal.inspect_traces(run, request, 'live-model')
                item['trace_diagnostics'] = facts
                item['observed_slots'], item['completed_slots'], item['confirmed_slots'] = cal.slot_accounting(learning.stage_spec(request), attempts)
                proof, integration = reader_evidence.durable_stage(plan, job, stage, entry, read, calls)
                if not proof['worker']['call_id'].startswith('fc-'):
                    raise ValueError('Missing provider stage call identity')
                obs = read(stage['output']+'/observability.json')
                if (read(stage['output']+'/request.json') != request or entry.get('wandb') != stage['wandb']
                        or obs['experiment_id'] != job['job_id'] or obs['wandb_run_id'] != stage['wandb']['id']
                        or obs['request_sha256'] != hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
                        or integration != reader_integration.report(run, request)):
                    raise ValueError('Observability or raw integration identity differs')
                report = stage_report(run, request)
                if proof['protocol'] != c.historical.PROTOCOL or report != proof['recovery_qualification']:
                    raise ValueError('Worker summary differs from raw traces and audit')
                if name in ('baseline', 'checkpoint-eval'):
                    expected = summarize.summarize(run, name, 'val')[0]
                    actual = proof['evaluation'].copy()
                    # The collector relocates run directories from the volume to local storage.
                    actual['run_dir'] = expected['run_dir']
                    if actual != expected:
                        raise ValueError('Evaluation summary differs from raw trajectories')
                if name == 'training':
                    counts = report['informative_groups']
                    diags = proof['learning_diagnostics']
                    expected = [{'iteration': i+1, 'generated_tokens': e['representation']['generated_tokens'],
                                 'real_rollouts': 32, 'constant_groups': 8-counts[i]}
                                for i, e in enumerate(report['collections'])]
                    if (diags['collections'] != expected or diags['losses_by_update'] != report['losses']
                            or diags['finite_loss_values'] != len(report['losses'])
                            or diags['informative_reward_groups'] != sum(counts)
                            or diags['accuracy_improvement_required'] is not False):
                        raise ValueError('Learning diagnostics differ from collected outcomes or metrics')
                expected_dirs = ['global_step_2'] if name == 'training' else []
                if sorted(p.name for p in (run/'checkpoints').glob('global_step_*')) != expected_dirs:
                    raise ValueError('Unexpected native checkpoint set')
                if name == 'training':
                    checkpoint = json.loads((run/'checkpoints/global_step_2/matched_identity.json').read_text())
                    if (c.digest(checkpoint) != proof['checkpoint_sha256'] or checkpoint['step'] != 2
                            or checkpoint['algorithm'] != job['algorithm'] or checkpoint['plan_sha256'] != c.digest(plan)
                            or checkpoint['resolved_config_sha256'] != c.file_hash(run/'resolved_config.json')
                            or set(checkpoint['roles']) != set(report['optimizer_calls'])):
                        raise ValueError('Native checkpoint manifest identity differs')
                    for files in checkpoint['roles'].values():
                        if not all(files[k+'_world_size_1_rank_0.pt']['bytes'] > 0 for k in ('model', 'optim', 'extra_state')):
                            raise ValueError('Incomplete checkpoint inventory')
                    training = proof
                if name == 'checkpoint-eval' and (proof['checkpoint_before_load_sha256'] != proof['checkpoint_sha256'] or not training or proof['training_worker'] != training['worker']
                        or proof['worker']['container_id'] == training['worker']['container_id']
                        or proof['checkpoint_sha256'] != training['checkpoint_sha256']):
                    raise ValueError('Fresh-worker checkpoint identity differs')
                item.update(admitted=True, performance={'strict_successes': report['strict_successes'],
                    'episodes': report['episodes_admitted']}, optimization={'informative_groups': report['informative_groups'],
                    'losses': report['losses'], 'optimizer_calls': report['optimizer_calls']})
            except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, RuntimeError) as exc:
                item['reason'] = f'{type(exc).__name__}: {exc}'
                errors.append(job['algorithm']+'/'+name+': '+item['reason'])
    return {'evidence_complete': not errors, 'operational_completion': 'completed' if not errors else 'incomplete or unverified',
            'qualification_evidence': not errors, 'stages': stages, 'problems': errors}
