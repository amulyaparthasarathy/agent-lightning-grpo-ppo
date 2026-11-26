"""Replay the two collected seed-2001 allocations offline. Export allowlisted evidence only."""
import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
from statistics import median, stdev
from struct import pack, unpack
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment import reader_contract as contract, recovery_calibration as calibration
from experiment import recovery_learning_evidence as replay, recovery_solver as solver, recovery_task as task

PUBLIC = ROOT / 'evidence/phase4c-functional-pilot'
CLASSES = ('ready', 'not_ready')
FLAGS = ('json_syntax', 'format_compliant', 'target_correct', 'time_correct', 'decision_correct', 'evidence_correct',
         'asset_record_correct', 'policy_record_correct', 'inventory_correct', 'supporting_runs_correct')


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique(rows, key):
    result = {row[key]: row for row in rows}
    require(len(result) == len(rows), 'Duplicate ' + key)
    return result


def parse_answer(raw, terminal):
    if terminal:
        return None
    try:
        return task.parse(raw)
    except (TypeError, ValueError, RecursionError):
        return None


def case_facts(case):
    label = json.loads(case['result'])
    answer, ledger = solver.solve(case['question'], task.call)
    ref = task.parse(answer)
    require(ref['decision'] == label['decision']
            and sorted(ref['supporting_run_ids']) == label['qualifying_run_ids'], 'Solver and label disagree')
    rpo = json.loads(ledger[1]['content'])['rpo']
    limit = rpo['value'] * task.UNIT_MINUTES[rpo['unit']]
    inventory = {}
    for run in json.loads(ledger[2]['content'])['runs']:
        age = task.minutes(ref['as_of']) - task.minutes(run['recovery_point_at'])
        inventory[run['run_id']] = dict(age_minutes=age, status=run['status'], integrity=run['integrity'])
    return dict(actual=label['decision'], qualifying=label['qualifying_run_ids'], rpo_minutes=limit,
                as_of=ref['as_of'], asset_id=ref['asset_id'], inventory=inventory)


def describe(row, records, facts, digest):
    episode, = (r for r in records if r['kind'] == 'episode')
    require(episode['rollout_id'] == row['rollout_id'] and episode['case_id'] == row['case_id'],
            'Episode/trace join differs')
    raw = episode['outputs']['raw_answer']
    answer = parse_answer(raw, row['outcome'] == 'terminal_policy_failure')
    require(bool(answer) == bool(row['format_compliant']), 'Parser differs from replay')
    if row['decision_correct'] is not None:
        require(answer is not None and (answer['decision'] == facts['actual']) == row['decision_correct'],
                'Raw decision and verifier disagree')
    # Only publish IDs that belong to this synthetic target inventory.
    selected = None if answer is None else [
        rid if rid in facts['inventory'] else 'outside_target_inventory' for rid in answer['supporting_run_ids']]
    reasons = set()
    for rid in selected or []:
        if rid == 'outside_target_inventory':
            reasons.add(rid)
            continue
        run = facts['inventory'][rid]
        if run['age_minutes'] > facts['rpo_minutes']:
            reasons.add('outside_rpo')
        if run['status'] != 'succeeded':
            reasons.add(run['status'])
        elif run['integrity'] != 'passed':
            reasons.add('integrity_' + run['integrity'])
    calls = [(v['name'], v['arguments_sha256']) for v in row['sampled_calls']]
    dev = {}
    if episode['eval_split'] == 'development':
        actions = [r['inputs'] for r in records if r['kind'] == 'action_outcome']
        first = next(r for r in records if r['kind'] == 'llm')
        dev = dict(first_input_sha256=contract.digest(first['inputs']),
                   completion_tokens_by_turn=row['completion_tokens'],
                   observation_contents_sha256=contract.digest([a['observation']['content'] for a in actions]),
                   parsed_calls_sha256=contract.digest([(a['call']['name'], json.loads(a['call']['arguments']))
                                                       for a in actions]))
    return dict(case_id=row['case_id'], family=row['family'], variant=row['variant'], actual=facts['actual'],
                decision=answer['decision'] if answer else None, reward=int(row['reward']),
                failure=row['failure_class'], **{k: row[k] for k in FLAGS}, selected=selected,
                selected_issues=sorted(reasons), observed=row['observed'], outcome=row['outcome'],
                policy_errors=row['policy_error_codes'], tokens=sum(row['completion_tokens']),
                llm_calls=row['llm_calls'], trace_sha256=digest, answer_sha256=contract.digest(raw),
                calls_sha256=contract.digest(calls), **dev)


def float32(value):
    return unpack('f', pack('f', value))[0]


def advantages(rewards):
    """Frozen GRPO arithmetic for four binary rewards (unbiased SD, float32 operations)."""
    require(len(rewards) == 4 and set(rewards) <= {0, 1}, 'Expected four binary rewards')
    mean = sum(rewards) / 4
    denominator = float32(float32(stdev(rewards)) + float32(1e-6))
    return [float32((r - mean) / denominator) for r in rewards]


def summarize(rows):
    groups = by_case(rows)
    bins = Counter(sum(r['reward'] for r in group) for group in groups.values())
    adv = [r['advantage'] for r in rows if r['advantage'] > 0]
    return dict(questions=len(groups), trajectories=len(rows),
                families=dict(Counter(group[0]['family'] for group in groups.values())),
                ready=sum(r['decision'] == 'ready' for r in rows),
                not_ready=sum(r['decision'] == 'not_ready' for r in rows),
                unassessable=sum(r['decision'] is None for r in rows),
                format_assessed=sum(r['format_compliant'] is not None for r in rows),
                decision_assessed=sum(r['decision_correct'] is not None for r in rows),
                decision_wrong=sum(r['decision_correct'] is False for r in rows),
                evidence_assessed=sum(r['evidence_correct'] is not None for r in rows),
                evidence_wrong=sum(r['evidence_correct'] is False for r in rows),
                **{k: sum(r['failure'] == k for r in rows) for k in ('correct', 'semantic', 'evidence', 'format', 'action')},
                all_zero=bins[0], mixed=bins[1] + bins[2] + bins[3], all_one=bins[4],
                positive_advantages=len(adv), positive_advantage_sum=sum(adv),
                tokens=sum(r['tokens'] for r in rows), median_tokens=median(r['tokens'] for r in rows) if rows else None)


def by_case(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['case_id']].append(row)
    return groups


def training(report, episodes, metrics):
    samples, collections = [], []
    metrics = unique(metrics, 'step')
    require(set(metrics) == set(report['actor_calls']), 'Unpaired update metrics')
    for event, update, prob in zip(report['collections'], report['actor_calls'], report['old_probabilities'], strict=True):
        members = [dict(episodes[e['rollout']], collection=event['iteration']) for e in event['episodes']]
        groups = by_case(members)
        require(len(groups) == 8 and update == event['policy_step'] + 1 == event['iteration'],
                'Unpaired behavior policy or update')
        for group in groups.values():
            for row, a in zip(group, advantages([r['reward'] for r in group]), strict=True):
                row['advantage'] = a
        samples.extend(members)
        m = metrics[update]['metrics']
        collections.append(dict(collection=event['iteration'], policy_step=event['policy_step'], actor_call=update,
                                by_class={k: summarize([r for r in members if r['actual'] == k]) for k in CLASSES},
                                actor={k: m['actor/'+k] for k in ('pg_loss', 'grad_norm', 'pg_clipfrac',
                                                                  'pg_clipfrac_lower', 'ppo_kl')},
                                old_minus_sampler={k: prob[k] for k in ('target_tokens', 'mean_delta', 'max_abs_delta')}))
    require(len({r['trace_sha256'] for r in samples}) == len(episodes) == len(samples), 'Training rollout coverage')
    return sorted(samples, key=lambda r: (r['collection'], r['case_id'], r['trace_sha256'])), collections


def allocation(name, directory, facts):
    source_file = PUBLIC / (name + '.json')
    public = read(source_file)
    root = directory / 'pilot-evidence'
    plan = read(root/'supervisor/plan.json')
    idx = read(root/'supervisor/index.json')
    review_file = directory/'independent-review/report.json'
    review = read(review_file)
    commitments = dict(plan_sha256=contract.digest(plan), assets_sha256=contract.digest(plan['assets']),
                       runtime_manifest_sha256=plan['packaged_sha256'], source=plan['source']['commit'],
                       data_manifest_sha256=contract.digest(plan['data_manifest']),
                       model_manifest_sha256=plan['assets']['model']['manifest_sha256'],
                       independent_report_sha256=contract.file_hash(review_file))
    require(all(public[k] == v for k, v in commitments.items()) and idx['plan'] == plan, 'Allocation identity differs')
    commitments.update(public_summary_sha256=contract.file_hash(source_file),
                       model_files_sha256=contract.digest(plan['assets']['model']['files']),
                       checkpoint_inventory_sha256=public['checkpoint']['digest'],
                       study_sha256=contract.digest(plan['study']), stages={})
    preflight = read(root/'preflight/receipt.json')
    require(contract.digest(preflight) == idx['owner']['preflight_sha256'], 'Preflight receipt differs')
    if name == 'P02':
        require(contract.digest(idx['first_seed_gate']) == public['first_seed_gate_sha256']
                and review['pilot_completion']['admitted_episodes'] == review['pilot_completion']['confirmed_rewards'] == 560,
                'Terminal gate differs')
    result, configs, workers = {}, {}, {}
    for job in plan['jobs']:
        stage, = job['stages']
        key = stage['name']
        published = public['stages'][key]
        admission = review['stages'][stage['key']]
        if not admission['admitted']:
            require((name, key) == ('P01', 'checkpoint-eval') and published['admitted_episodes'] == 0,
                    'Unexpected incomplete stage')
            continue
        run = root/Path(stage['output']).relative_to(plan['root'])
        req = contract.stage_request(plan, job, stage)
        require(read(run/'request.json') == req, 'Request mismatch')
        events = calibration.read_events(run)
        report = replay.stage_report(run, req, events, preflight=preflight)
        completion = read(run/'completion.json')
        entry, = idx['jobs'][job['job_id']]['stages']
        receipt = read(run/'receipt.json')
        require(receipt == entry['result'] and receipt['call_id'] == entry['call_id']
                and receipt['completion'] == completion, 'Supervisor/worker receipt differs')
        worker = public['provider']['calls'][key]
        call_digest = sha256(receipt['call_id'].encode()).hexdigest()
        container_digest = sha256(receipt['container_id'].encode()).hexdigest()
        require(call_digest == worker['call_sha256'] and container_digest in worker['container_digests'],
                'Raw worker differs from published identity')
        workers[key] = (call_digest, container_digest)
        require(report == completion['recovery_qualification'] and
                len(report['episodes']) == admission['confirmed_slots'] == published['admitted_episodes'] and
                report['strict_successes'] == published['strict_successes'], 'Stage replay differs from receipts')
        if key != 'baseline':
            require(completion['checkpoint_sha256'] == public['checkpoint']['digest'], 'Checkpoint differs')
        if key == 'training':
            inventory = read(run/'checkpoints/global_step_16/matched_identity.json')
            require(contract.digest(inventory) == public['checkpoint']['digest'], 'Inventory differs')
        if key == 'checkpoint-eval':
            require(all(a != b for a, b in zip(workers[key], workers['training'], strict=True)),
                    'Training and reload must have distinct calls and workers')
            require(completion['checkpoint_before_load_sha256'] == completion['checkpoint_sha256'] and
                    report['refresh_steps'] == [16], 'Reload policy differs')
        registrations = [e['streams'] for e in events if e['event'] == 'matched_registered']
        require(registrations and all(v == job['seed_streams'] for v in registrations), 'Configured streams differ')
        configs[key] = read(run/'resolved_config.json')
        rows = unique(report['episodes'], 'trace_id')
        episodes = {}
        for filename, records in calibration.read_traces(run):
            row = rows.pop(records[0]['trace_id'])
            episodes[row['rollout_id']] = describe(row, records, facts[row['case_id']], contract.file_hash(run/'traces'/filename))
        require(not rows, 'Missing raw answer trace')
        commitments['stages'][key] = dict(
            traces_sha256=contract.digest(sorted(r['trace_sha256'] for r in episodes.values())),
            audit_sha256=contract.digest(sorted(contract.file_hash(p) for p in (run/'audit').glob('events-*.jsonl'))),
            resolved_config_sha256=report['resolved_config_sha256'], streams_sha256=contract.digest(job['seed_streams']),
            completion_sha256=contract.file_hash(run/'completion.json'))
        if key == 'training':
            path = run/'metrics.jsonl'
            commitments['stages'][key]['metrics_sha256'] = contract.file_hash(path)
            result[key], result['collections'] = training(report, episodes, [json.loads(l) for l in path.read_text().splitlines()])
        else:
            result[key] = unique(list(episodes.values()), 'case_id')
    return result, commitments, configs, plan


def changed_keys(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        return [field for k in sorted(a.keys() | b.keys())
                for field in changed_keys(a.get(k), b.get(k), prefix + ('.' if prefix else '') + k)]
    return [] if a == b else [prefix]


def pairs(p01, base, reload, published, facts):
    cases = unique(published, 'case_id')
    require(p01.keys() == base.keys() == reload.keys() == cases.keys(), 'Unpaired development cases')
    result = []
    for cid in sorted(cases):
        a, b = base[cid], reload[cid]
        for prefix, row in [('baseline', a), ('reload', b)]:
            for key in ('reward', 'failure', 'decision_correct', 'evidence_correct'):
                require(row[key] == cases[cid][prefix+'_'+key], 'Published paired flags disagree: '+cid)
        for field, flag in [('answer_sha256', 'final_answer_changed'), ('calls_sha256', 'tool_sequence_or_arguments_changed')]:
            require((a[field] != b[field]) == cases[cid][flag], 'Published change differs: '+cid)
        common = ('case_id', 'actual', 'family', 'variant')
        result.append({**{k: a[k] for k in common}, **facts[cid],
                       **{stage: {k: v for k, v in row.items() if k not in common}
                          for stage, row in [('p01_base', p01[cid]), ('base', a), ('reload', b)]}})
    return result


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True, separators=(',',':')) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def save(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')


def analyze(p01, p02, output):
    facts = {r['case_id']: case_facts(r) for split in ('train', 'development') for r in task.rows(split)}
    first, src1, cfg1, plan1 = allocation('P01', p01, facts)
    second, src2, cfg2, plan2 = allocation('P02', p02, facts)
    paired = pairs(first['baseline'], second['baseline'], second['checkpoint-eval'],
                   read(PUBLIC/'P02.json')['paired_development']['cases'], facts)
    summary = dict(schema='functional-pilot-diagnosis-v1', analysis_script_sha256=contract.file_hash(Path(__file__)),
                   sources={'P01': src1, 'P02': src2}, training={}, development={},
                   resolved_config_changed_fields={k: changed_keys(v, cfg2[k]) for k, v in cfg1.items()},
                   model_manifest_changed_fields=changed_keys(plan1['assets']['model']['manifest'],
                                                             plan2['assets']['model']['manifest']))
    for k, data in [('P01', first), ('P02', second)]:
        summary['training'][k] = {c: summarize([r for r in data['training'] if r['actual'] == c]) for c in CLASSES}
    family_totals = Counter(r['family'] for r in paired)
    for key in ('p01_base', 'base', 'reload'):
        summary['development'][key] = dict(
            confusion={c: {d: sum(r['actual'] == c and (r[key]['decision'] or 'unassessable') == d for r in paired)
                           for d in (*CLASSES, 'unassessable')} for c in CLASSES},
            decision_correct=sum(r[key]['decision_correct'] is True for r in paired),
            evidence_correct=sum(r[key]['evidence_correct'] is True for r in paired),
            family_cases=family_totals,
            family_successes={f: sum(r[key]['reward'] for r in paired if r['family'] == f) for f in family_totals})
    summary['strict_transitions'] = dict(Counter(f"{r['base']['reward']}_to_{r['reload']['reward']}" for r in paired))
    summary['baseline_pair_changes'] = {
        k: sum(r['p01_base'][k] != r['base'][k] for r in paired) for k in
        ('reward', 'decision', 'answer_sha256', 'parsed_calls_sha256', 'calls_sha256', 'first_input_sha256',
         'observation_contents_sha256', 'completion_tokens_by_turn')}
    output.mkdir(parents=True, exist_ok=True)
    save(output/'development.json', paired)
    write_csv(output/'training.csv', second['training'])
    save(output/'collections.json', second['collections'])
    summary['tables_sha256'] = {f: contract.file_hash(output/f) for f in ('development.json', 'training.csv', 'collections.json')}
    save(output/'summary.json', summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('p01', 'p02', 'output'):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    analyze(args.p01, args.p02, args.output)
