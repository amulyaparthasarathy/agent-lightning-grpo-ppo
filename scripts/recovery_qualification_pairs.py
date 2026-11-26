"""Redacted case-paired comparison of retained A03 greedy traces. Offline only."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment import reader_contract as contract, recovery_calibration as calibration, recovery_learning as learning


def compare(directory):
    plan = json.loads((directory/'supervisor/plan.json').read_text())
    if contract.digest(plan) != 'a899410f14650150917e0a68d9d9d2897f50d1c560106baf41e8528163cf6a39':
        raise ValueError('Expected exact asset-bound A03 plan')
    job, = plan['jobs']
    policies = []
    for stage in (job['stages'][0],job['stages'][-1]):
        req = contract.stage_request(plan,job,stage)
        run = directory/Path(stage['output']).relative_to(plan['root'])
        table = {}
        rows = []
        for _, records in calibration.read_traces(run):
            row = calibration.episode_row(records,req)
            rows.append(row)
            answer = next(r['outputs']['raw_answer'] for r in records if r['kind'] == 'episode')
            calls = [(c['name'],c['arguments_sha256']) for c in row['sampled_calls']]
            actions = [r['inputs']['call'] for r in records if r['kind'] == 'action_outcome']
            normalized = [(a['name'],json.loads(a['arguments'])) for a in actions]
            table[row['case_id']] = dict(reward=row['reward'],failure=row['failure_class'],raw_answer=answer,
                                         calls=calls, normalized_calls=normalized, tool_names=[a['name'] for a in actions])
        calibration.check_table(learning.stage_spec(req),rows)
        policies.append(table)
    before, after = policies
    if before.keys() != after.keys():
        raise ValueError('Unpaired development cases')
    changes = {k:sum(before[c][k] != after[c][k] for c in before) for k in next(iter(before.values()))}
    transitions = Counter((int(before[c]['reward']),int(after[c]['reward'])) for c in before)
    return dict(cases=len(before), changed=changes,
                reward_transitions={f'{a}_to_{b}':n for (a,b),n in sorted(transitions.items())},
                any_answer_or_call_change=sum(any(before[c][k] != after[c][k] for k in ('raw_answer','calls')) for c in before))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('collected',type=Path)
    print(json.dumps(compare(parser.parse_args().collected),indent=2))
