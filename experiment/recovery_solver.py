"""Reference solver for recovery-readiness-v1 that uses only the public tools.

It receives the request text and a `call(name, arguments)` function returning the
parsed tool observation. It never imports the generator, the trusted world, case
families or labels, and its arithmetic is written independently of the verifier
(calendar datetimes here; integer epoch minutes there). Agreement between the two
on every generated case is a dataset-consistency check, not a model result.
"""
from datetime import datetime, timedelta, timezone
import json
import re

REQUEST = re.compile(r'(ast-[0-9a-f]{8})')
MOMENT = re.compile(r'([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}Z)')


def moment(text):
    return datetime.strptime(text, '%Y-%m-%dT%H:%MZ').replace(tzinfo=timezone.utc)


def request(question):
    assets, times = REQUEST.findall(question), MOMENT.findall(question)
    if len(assets) != 1 or len(times) != 1:
        raise ValueError('Request must name one asset and one evaluation time')
    return assets[0], times[0]


def window(rpo):
    return timedelta(**{rpo['unit']: rpo['value']})


def qualifying(inventory, rpo, as_of):
    now, limit = moment(as_of), window(rpo)
    return [run['run_id'] for run in inventory['runs']
            if run['status'] == 'succeeded' and run['integrity'] == 'passed'
            and timedelta(0) <= now - moment(run['recovery_point_at']) <= limit]


def steps(question):
    """Generator protocol: yields (tool, arguments), receives the observation, returns the answer."""
    asset_id, as_of = request(question)
    asset = yield 'get_asset', {'asset_id': asset_id}
    policy = yield 'get_policy', {'policy_id': asset['policy_id']}
    inventory = yield 'list_backup_runs', {'asset_id': asset_id, 'as_of': as_of}
    usable = qualifying(inventory, policy['rpo'], as_of)
    return {'asset_id': asset_id, 'as_of': as_of, 'decision': 'ready' if usable else 'not_ready',
            'asset_record_id': asset['record_id'], 'policy_record_id': policy['record_id'],
            'inventory_id': inventory['inventory_id'], 'supporting_run_ids': usable}


def solve(question, call):
    """Run the three public calls; return (answer text, ledger of what was observed)."""
    plan, ledger, observation = steps(question), [], None
    try:
        while True:
            name, arguments = plan.send(observation)
            observation = call(name, arguments)
            ledger.append({'tool': name, 'arguments': arguments,
                           'content': json.dumps(observation, sort_keys=True, separators=(',', ':'))})
    except StopIteration as done:
        return json.dumps(done.value, sort_keys=True, separators=(',', ':')), ledger
