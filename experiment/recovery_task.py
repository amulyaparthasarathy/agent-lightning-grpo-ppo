"""recovery-readiness-v1: deterministic synthetic world, three read-only tools, strict verifier.

Only QUESTION (a request string naming the asset and the evaluation time) enters
the agent prompt. RESULT is a verifier-only label that is itself checked against
the regenerated world. Tools return raw facts as known at the supplied as_of.
They never return readiness, run eligibility, case families, labels or rewards.

Pure stdlib except generate/verify, which need PyArrow. No wall-clock time is read.
"""
import calendar
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import time

TASK = 'recovery-readiness-v1'
SEED = 'recovery-readiness-v1/2026-11-24/329'
DECISIONS = ('ready', 'not_ready')
FAMILIES = ('recent_usable', 'failed_newer_usable_older', 'stale_success',
            'recent_incomplete', 'recent_integrity_failed', 'empty_history')
READY_FAMILIES = FAMILIES[:2]
TOOLS = ('get_asset', 'get_policy', 'list_backup_runs')
MAX_RUNS = 6
# Prospective harness contract; separate from the dataset identity below.
EPISODE_POLICY = {
    'version': 'recovery-episode-v1', 'max_tool_iterations': 5, 'policy_error_limit': 2,
    'parallel_tool_calls': False, 'final_generation': 'context-json-reminder-v1',
    'final_answer': 'strict-evidence-verifier', 'recovered_final': 'strict-evidence-verifier',
    'policy_limit': 'terminal-zero', 'missing_required_evidence': 'complete-zero',
    'multiple_calls_in_one_generation': 'incomplete-no-reward',
    'infrastructure_or_missing_evidence': 'incomplete-no-reward',
    # Typed server codes, one harness code, and the two exact AutoGen 0.7.5 branches.
    'policy_error_codes': ['invalid_arguments', 'invalid_asset_id', 'unknown_asset', 'invalid_policy_id',
                           'unknown_policy', 'invalid_as_of', 'non_object_arguments',
                           'invalid_argument_json', 'unknown_tool']}
SERVER_ERROR_CODES = tuple(EPISODE_POLICY['policy_error_codes'][:6])
# Task-specific context binding. Only the trajectory prompt limit differs from the shared
# launcher value (2048): in live Phase 3 collections every model call became its own
# training fragment, so each call's whole prompt must fit this limit, and the bounded
# worst case here (five rounds of the largest inventory) measures 2739 tokens with the
# pinned tokenizer. 3072 is max_model_len minus max_tokens, the largest prompt the
# server can accept with the full generation budget. Calculator and asset settings are
# unchanged. Collection-only calibration performs no forward or backward pass.
CONTEXT_LIMITS = {'version': 'recovery-context-v1', 'trajectory_max_prompt_length': 3072,
                  'trajectory_max_response_length': 2048, 'max_model_len': 4096, 'max_tokens': 1024}
ASSET_ID, POLICY_ID, RUN_ID = '^ast-[0-9a-f]{8}$', '^pol-[0-9a-f]{8}$', '^run-[0-9a-f]{8}$'
TIMESTAMP = '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}Z$'
TOOL_SCHEMAS = {
    'get_asset': {'type': 'object', 'properties': {'asset_id': {'type': 'string', 'pattern': ASSET_ID}},
                  'required': ['asset_id'], 'additionalProperties': False},
    'get_policy': {'type': 'object', 'properties': {'policy_id': {'type': 'string', 'pattern': POLICY_ID}},
                   'required': ['policy_id'], 'additionalProperties': False},
    'list_backup_runs': {'type': 'object', 'properties': {'asset_id': {'type': 'string', 'pattern': ASSET_ID},
                                                           'as_of': {'type': 'string', 'pattern': TIMESTAMP}},
                         'required': ['asset_id', 'as_of'], 'additionalProperties': False}}
TOOL_DESCRIPTIONS = {
    'get_asset': 'Return the asset record, including the policy_id that applies to the asset.',
    'get_policy': 'Return the backup policy record, including its recovery point objective (RPO).',
    'list_backup_runs': 'Return the complete backup-run inventory of an asset as known at the UTC time as_of.'}
ANSWER_FIELDS = ('asset_id', 'as_of', 'decision', 'asset_record_id', 'policy_record_id', 'inventory_id',
                 'supporting_run_ids')
REQUESTS = ('Determine whether asset {asset_id} is recovery-ready as of {as_of} (UTC).',
            'As of {as_of} (UTC), does asset {asset_id} meet its recovery point objective?',
            'Assess the recovery readiness of asset {asset_id} at evaluation time {as_of} (UTC).')
PROMPT = (
    'Decide whether the asset is recovery-ready at the evaluation time (as_of) given in this request. '
    'That as_of is the only current time; ignore any other date. '
    'Call get_asset for the asset, get_policy for the policy_id it names, and list_backup_runs with the '
    'asset_id and the same as_of. '
    'A backup run is usable only if its status is "succeeded", its integrity is "passed", and its '
    'recovery_point_at is no later than as_of and no more than the policy RPO before as_of. '
    'A recovery point exactly one RPO old still counts. Measure from recovery_point_at, not completed_at. '
    'The decision is "ready" if at least one run is usable and "not_ready" otherwise. '
    'Return exactly one JSON object and no other text, with exactly the keys asset_id, as_of, decision, '
    'asset_record_id, policy_record_id, inventory_id, supporting_run_ids. '
    'Copy record_id and inventory_id values from the tool results. supporting_run_ids is a list of one or '
    'more usable run_id values when the decision is ready, and [] when it is not_ready.')
RPO_OPTIONS = ((90, 'minutes'), (240, 'minutes'), (4, 'hours'), (8, 'hours'), (12, 'hours'), (24, 'hours'),
               (36, 'hours'), (1, 'days'), (2, 'days'), (3, 'days'))
UNIT_MINUTES = {'minutes': 1, 'hours': 60, 'days': 1440}
# Ordered slot recipes. Ready and not-ready decisions are balanced in every panel.
TRAIN_COUNTS = {'recent_usable': 32, 'failed_newer_usable_older': 32, 'stale_success': 22,
                'recent_incomplete': 16, 'recent_integrity_failed': 16, 'empty_history': 10}
FINAL_COUNTS = {'recent_usable': 16, 'failed_newer_usable_older': 16, 'stale_success': 11,
                'recent_incomplete': 8, 'recent_integrity_failed': 8, 'empty_history': 5}
DEVELOPMENT_PACKETS = (
    {'recent_usable': 2, 'failed_newer_usable_older': 2, 'stale_success': 2,
     'recent_incomplete': 1, 'recent_integrity_failed': 1},
    {'recent_usable': 2, 'failed_newer_usable_older': 2, 'stale_success': 1,
     'recent_incomplete': 1, 'recent_integrity_failed': 1, 'empty_history': 1},
    {'recent_usable': 2, 'failed_newer_usable_older': 2, 'stale_success': 1,
     'recent_incomplete': 1, 'recent_integrity_failed': 1, 'empty_history': 1})
# Same-asset counterfactual pairs (usable at one evaluation time, stale at a later one).
PAIRS = {'train': 8, 'final': 4, 'development': (1, 0, 0)}
VARIANTS = {'recent_usable': ('plain', 'boundary', 'multi_witness', 'plain'),
            'failed_newer_usable_older': ('plain', 'multi_witness', 'boundary', 'plain'),
            'stale_success': ('plain', 'just_stale', 'completed_recent', 'plain'),
            'recent_incomplete': ('plain',),
            'recent_integrity_failed': ('plain', 'integrity_pending'),
            'empty_history': ('plain',)}
FILES = {'train_main.parquet': 128, 'dev_panel.parquet': 24, 'dev_panel.jsonl': 24,
         'dev_packet_1.parquet': 8, 'dev_packet_2.parquet': 8, 'dev_packet_3.parquet': 8,
         'final_eval.parquet': 64, 'final_eval.jsonl': 64}
SPLIT_OF_FILE = {'train_main.parquet': 'train', 'final_eval.parquet': 'final', 'final_eval.jsonl': 'final',
                 **{n: 'development' for n in FILES if n.startswith('dev_')}}
# Model-facing calibration inputs. The final-evaluation files are deliberately not
# named like any launcher default (train_main/eval_main) and are never listed here.
CALIBRATION_FILES = {'greedy': 'dev_panel.parquet',
                     'packets': ('dev_packet_1.parquet', 'dev_packet_2.parquet', 'dev_packet_3.parquet')}


def dumps(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def minutes(stamp):
    """Explicit UTC minute timestamp to integer minutes; rejects impossible dates."""
    if not isinstance(stamp, str) or not re.fullmatch(TIMESTAMP, stamp):
        raise ValueError('invalid timestamp')
    return calendar.timegm(time.strptime(stamp, '%Y-%m-%dT%H:%MZ')) // 60


def stamp(value):
    return time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime(value * 60))


def _n(*parts):
    # Hash sampling: no PRNG/version dependence and no information in visible IDs.
    return int(digest([SEED, *parts]), 16)


def _between(low, high, *parts):
    if high < low:
        raise ValueError('empty deterministic range')
    return low + _n(*parts) % (high - low + 1)


def _id(prefix, width, *parts):
    return prefix + digest([SEED, 'id', prefix, *parts])[:width]


def _history(family, variant, rpo, key):
    """Run facts relative to the evaluation minute; age is as_of - recovery point."""
    def usable(i, exact=False):
        duration, check = _between(5, 30, *key, 'u-d', i), _between(1, 15, *key, 'u-c', i)
        age = rpo if exact else _between(duration + check, rpo - 1, *key, 'u-a', i)
        return {'age': age, 'duration': duration, 'check': check, 'status': 'succeeded', 'integrity': 'passed'}

    def stale(i, kind='plain', beyond=0):
        if kind == 'just_stale':
            age = rpo + 1
        elif kind == 'completed_recent':
            # Long-running backup: completion is inside the RPO, the recovery point is not.
            age = rpo + _between(1, 30, *key, 's-a', i)
            duration = _between(age - rpo + 1, age - rpo + 20, *key, 's-d', i)
            return {'age': age, 'duration': duration, 'check': _between(1, age - duration, *key, 's-c', i),
                    'status': 'succeeded', 'integrity': 'passed'}
        else:
            # Half of ordinary stale points are near the boundary.
            near = _n(*key, 's-near', i) % 2 == 0
            age = rpo + beyond + _between(2, max(3, rpo // 4) if near else 3 * rpo, *key, 's-a', i)
        return {'age': age, 'duration': _between(5, 40, *key, 's-d', i), 'check': _between(1, 15, *key, 's-c', i),
                'status': 'succeeded', 'integrity': 'passed'}

    def failed(i, low, high):
        age = _between(low, max(low, high), *key, 'f-a', i)
        return {'age': age, 'duration': _between(1, min(age, 45), *key, 'f-d', i), 'check': 0,
                'status': 'failed', 'integrity': 'not_applicable'}

    def bad_integrity(i, low, high, pending=False):
        age = _between(low, max(low, high), *key, 'i-a', i)
        duration = _between(1, max(1, min(age - 1, 30)), *key, 'i-d', i)
        # A pending check finishes (and passes) only after the evaluation time.
        check = age - duration + _between(5, 60, *key, 'i-c', i) if pending else _between(1, max(1, age - duration), *key, 'i-c', i)
        return {'age': age, 'duration': duration, 'check': check, 'status': 'succeeded',
                'integrity': 'passed' if pending else 'failed'}

    def running(i):
        age = _between(1, min(rpo - 1, 120), *key, 'r-a', i)
        return {'age': age, 'duration': age + _between(5, 90, *key, 'r-d', i), 'check': _between(1, 15, *key, 'r-c', i),
                'status': 'succeeded', 'integrity': 'passed'}

    def older(count, after, salt):
        # Non-qualifying records strictly older than the family's defining record.
        rows = []
        for i in range(count):
            kind = ('stale', 'failed', 'stale', 'integrity')[_n(*key, salt, 'kind', i) % 4]
            if kind == 'stale':
                rows.append(stale(f'{salt}{i}'))
            elif kind == 'failed':
                rows.append(failed(f'{salt}{i}', after + 1, after + 2 * rpo))
            else:
                rows.append(bad_integrity(f'{salt}{i}', max(after + 1, 3), after + 2 * rpo))
        return rows

    if family == 'empty_history':
        return []
    if family == 'recent_usable':
        first = usable(0, exact=variant == 'boundary')
        rows = [first] + ([usable(1), *([usable(2)] if _n(*key, 'three') % 2 else [])] if variant == 'multi_witness' else [])
        newest = min(r['age'] for r in rows)
        # Keep the newest record usable so the family label stays truthful.
        rows += [r for r in older(_between(0, 3, *key, 'extra'), max(r['age'] for r in rows), 'o')]
        assert min(r['age'] for r in rows) == newest
        return rows
    if family == 'failed_newer_usable_older':
        witnesses = [usable(0, exact=variant == 'boundary')] + ([usable(1)] if variant == 'multi_witness' else [])
        # A usable point is at least six minutes old, so a newer failed run always fits.
        rows = [failed('new', 1, min(r['age'] for r in witnesses) - 1)] + witnesses
        return rows + older(_between(0, 3, *key, 'extra'), max(r['age'] for r in witnesses), 'o')
    if family == 'stale_success':
        rows = [stale(0, variant)] + [stale(i, beyond=31) for i in range(1, 1 + _between(0, 2, *key, 'more'))]
        oldest = max(r['age'] for r in rows)
        return rows + [failed(f'o{i}', oldest + 1, oldest + 2 * rpo) for i in range(_between(0, 2, *key, 'extra'))]
    if family == 'recent_incomplete':
        rows = [running(0)]
    elif family == 'recent_integrity_failed':
        rows = [bad_integrity(0, 3, rpo - 1, pending=variant == 'integrity_pending')]
    else:
        raise ValueError('unknown case family')
    rows += [stale(i) for i in range(1, 2 + _between(0, 1, *key, 'more'))]
    oldest = max(r['age'] for r in rows)
    return rows + [failed(f'o{i}', oldest + 1, oldest + 2 * rpo) for i in range(_between(0, 2, *key, 'extra'))]


def _distinct(family, variant, rpo, key):
    # Recovery points of one asset are distinct; resample deterministically otherwise.
    for salt in range(64):
        rows = _history(family, variant, rpo, (*key, 'salt', salt))
        if len({r['age'] for r in rows}) == len(rows) <= MAX_RUNS:
            return rows
    raise ValueError('could not build a distinct bounded history')


def _blocks(counts, pairs, offset, *key):
    """Four-case blocks sharing one policy. A pair is one asset evaluated at two times."""
    counts = dict(counts)
    counts['recent_usable'] -= pairs
    counts['stale_success'] -= pairs
    singles = [(family, VARIANTS[family][(i + offset) % len(VARIANTS[family])]) for family in FAMILIES
               for i in range(counts.get(family, 0))]
    singles = [singles[i] for i in sorted(range(len(singles)), key=lambda i: digest([SEED, 'slot', *key, i]))]
    total = len(singles) + 2 * pairs
    if total % 4 or pairs > total // 4:
        raise ValueError('split recipe does not fill four-case blocks')
    blocks = [[] for _ in range(total // 4)]
    for index in sorted(range(len(blocks)), key=lambda i: digest([SEED, 'pair-block', *key, i]))[:pairs]:
        blocks[index] += [('recent_usable', 'time_sibling'), ('stale_success', 'time_sibling')]
    for block in blocks:
        while len(block) < 4:
            block.append(singles.pop(0))
    return blocks


@lru_cache(maxsize=1)
def _world():
    assets, policies, runs, cases = {}, {}, {}, {'train': [], 'development': [], 'final': []}
    packets = []

    def add_runs(asset_id, as_of, specs):
        rows = []
        for spec in specs:
            point = as_of - spec['age']
            completed = point + spec['duration']
            rows.append({'run_id': _id('run-', 8, asset_id, point), 'asset_id': asset_id,
                         'recovery_point_at': point, 'completed_at': completed, 'status': spec['status'],
                         'integrity': spec['integrity'],
                         'integrity_checked_at': completed + spec['check'] if spec['status'] == 'succeeded' else None})
        runs[asset_id] = sorted(rows, key=lambda r: digest([SEED, 'run-order', r['run_id']]))

    def block(split, index, slots):
        value, unit = RPO_OPTIONS[_n('rpo', split, index) % len(RPO_OPTIONS)]
        rpo = value * UNIT_MINUTES[unit]
        policy_id = _id('pol-', 8, split, index)
        policies[policy_id] = {'record_id': _id('rec-', 10, 'policy', policy_id), 'policy_id': policy_id,
                               'rpo': {'value': value, 'unit': unit}}
        made = []
        # Counterfactual pair: the same asset and history, evaluated while its newest
        # recovery point is usable and again after that point has gone stale.
        early = next((i for i, s in enumerate(slots) if s == ('recent_usable', 'time_sibling')), None)
        for position, (family, variant) in enumerate(slots):
            key = (split, index, position)
            if position == early:
                continue
            asset_id = _id('ast-', 8, *key)
            as_of = minutes('2026-02-01T00:00Z') + _between(0, 330 * 1440, *key, 'as-of')
            if variant == 'time_sibling':
                duration, check = _between(5, 30, *key, 'p-d'), _between(1, 15, *key, 'p-c')
                late_age = rpo + _between(1, rpo, *key, 'p-late')
                early_age = _between(duration + check, rpo, *key, 'p-early')
                shift = late_age - early_age
                specs = [{'age': late_age, 'duration': duration, 'check': check, 'status': 'succeeded', 'integrity': 'passed'}]
                for i in range(_between(0, 2, *key, 'p-more')):
                    specs.append({'age': rpo + shift + _between(1, 2 * rpo, *key, 'p-old', i) + i,
                                  'duration': _between(5, 40, *key, 'p-od', i), 'check': _between(1, 15, *key, 'p-oc', i),
                                  'status': 'succeeded', 'integrity': 'passed'})
                if len({s['age'] for s in specs}) != len(specs):
                    specs = specs[:1]
                made.append((early, 'recent_usable', 'time_sibling', asset_id, as_of - shift))
            else:
                specs = _distinct(family, variant, rpo, key)
            assets[asset_id] = {'record_id': _id('rec-', 10, 'asset', asset_id), 'asset_id': asset_id, 'policy_id': policy_id}
            add_runs(asset_id, as_of, specs)
            made.append((position, family, variant, asset_id, as_of))
        rows = []
        for position, family, variant, asset_id, as_of in sorted(made):
            text = REQUESTS[_n('request', split, index, position) % len(REQUESTS)]
            rows.append({'case_id': f"{TASK}/{split}/{_id('c-', 10, split, index, position)}", 'split': split,
                         'asset_id': asset_id, 'as_of': stamp(as_of), 'family': family, 'variant': variant,
                         'question': text.format(asset_id=asset_id, as_of=stamp(as_of))})
        return rows

    def build(split, blocks, first):
        rows = []
        for i, slots in enumerate(blocks):
            rows += block(split, first + i, slots)
        return sorted(rows, key=lambda r: digest([SEED, 'order', r['case_id']]))

    cases['train'] = build('train', _blocks(TRAIN_COUNTS, PAIRS['train'], 0, 'train'), 0)
    cases['final'] = build('final', _blocks(FINAL_COUNTS, PAIRS['final'], 0, 'final'), 0)
    for number, counts in enumerate(DEVELOPMENT_PACKETS):
        packet = build('development', _blocks(counts, PAIRS['development'][number], number, 'development', number), 2 * number)
        packets.append([r['case_id'] for r in packet])
        cases['development'] += packet
    ids = [r['case_id'] for rows in cases.values() for r in rows]
    if len(set(ids)) != len(ids) or len({(r['asset_id'], r['as_of']) for rows in cases.values() for r in rows}) != len(ids):
        raise ValueError('colliding synthetic identities')
    return {'assets': assets, 'policies': policies, 'runs': runs, 'cases': cases, 'packets': packets,
            'by_question': {r['question']: r for rows in cases.values() for r in rows},
            'by_id': {r['case_id']: r for rows in cases.values() for r in rows}}


def registry():
    """Trusted raw facts, including what becomes known only after an evaluation time."""
    w = _world()
    def show(run):
        return {**run, **{k: None if run[k] is None else stamp(run[k])
                          for k in ('recovery_point_at', 'completed_at', 'integrity_checked_at')}}
    return {'assets': w['assets'], 'policies': w['policies'],
            'runs': {a: [show(r) for r in rows] for a, rows in w['runs'].items()}}


class PolicyActionError(ValueError):
    """Only explicit model input validation failures; never transport/server errors."""


def policy_error(code):
    """Typed model-visible error observation; shared by the server and harness."""
    return {'error_type': 'recovery_policy_action_v1', 'code': code,
            'instruction': 'Use get_asset(asset_id), get_policy(policy_id) or list_backup_runs(asset_id, as_of) '
                           'with exactly those string arguments.'}


def _arguments(arguments, names):
    if (not isinstance(arguments, dict) or set(arguments) != set(names)
            or any(type(arguments[n]) is not str for n in names)):
        raise PolicyActionError('invalid_arguments')


def _asset(asset_id):
    if not re.fullmatch(ASSET_ID, asset_id):
        raise PolicyActionError('invalid_asset_id')
    if asset_id not in _world()['assets']:
        raise PolicyActionError('unknown_asset')
    return _world()['assets'][asset_id]


def inventory_id(asset_id, as_of):
    return _id('inv-', 12, asset_id, as_of)


def call(name, arguments):
    """The complete public tool surface. Raises PolicyActionError for invalid input."""
    if name == 'get_asset':
        _arguments(arguments, ('asset_id',))
        return dict(_asset(arguments['asset_id']))
    if name == 'get_policy':
        _arguments(arguments, ('policy_id',))
        if not re.fullmatch(POLICY_ID, arguments['policy_id']):
            raise PolicyActionError('invalid_policy_id')
        if arguments['policy_id'] not in _world()['policies']:
            raise PolicyActionError('unknown_policy')
        policy = _world()['policies'][arguments['policy_id']]
        return {**policy, 'rpo': dict(policy['rpo'])}
    if name == 'list_backup_runs':
        _arguments(arguments, ('asset_id', 'as_of'))
        asset = _asset(arguments['asset_id'])
        try:
            now = minutes(arguments['as_of'])
        except ValueError:
            raise PolicyActionError('invalid_as_of') from None
        rows = []
        for run in _world()['runs'][asset['asset_id']]:
            if run['recovery_point_at'] > now:
                continue  # The run had not started; it does not exist yet.
            done = run['completed_at'] <= now
            checked = done and run['integrity_checked_at'] is not None and run['integrity_checked_at'] <= now
            rows.append({'run_id': run['run_id'], 'recovery_point_at': stamp(run['recovery_point_at']),
                         'completed_at': stamp(run['completed_at']) if done else None,
                         'status': run['status'] if done else 'running',
                         'integrity': ('not_applicable' if not done or run['status'] != 'succeeded'
                                       else run['integrity'] if checked else 'pending')})
        return {'inventory_id': inventory_id(asset['asset_id'], arguments['as_of']), 'asset_id': asset['asset_id'],
                'as_of': arguments['as_of'], 'complete': True, 'runs': rows}
    raise PolicyActionError('unknown_tool')


def classify_error(name, raw_arguments, content):
    """Typed policy-error code for an error observation, or None if unattributable.

    A code is returned only when the retained sampled call itself fails the declared
    validation with that same code. An error flag alone proves nothing: the MCP
    workbench converts every exception into an error result. Pure, so that a
    reviewer can re-derive every classification from a recorded trace.
    """
    try:
        sampled = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        # Exact AutoGen 0.7.5 branch; MCP is never invoked.
        return 'invalid_argument_json' if content == f'Error: {exc}' else None
    if name not in TOOLS:
        return 'unknown_tool' if content == f"Error: tool '{name}' not found in any workbench" else None
    try:
        error = json.loads(content)
    except (ValueError, TypeError):
        return None
    if not isinstance(error, dict) or error != policy_error(error.get('code')):
        return None
    if error['code'] == 'non_object_arguments':
        return error['code'] if not isinstance(sampled, dict) else None
    if error['code'] not in SERVER_ERROR_CODES or not isinstance(sampled, dict):
        return None
    try:
        call(name, sampled)
    except PolicyActionError as exc:
        return str(exc) if str(exc) == error['code'] else None
    return None


def usable(run, as_of, rpo, asset_id):
    """Trusted eligibility over raw integer-minute facts; the RPO boundary is inclusive."""
    return (run['asset_id'] == asset_id and run['status'] == 'succeeded' and run['completed_at'] <= as_of
            and run['integrity'] == 'passed' and run['integrity_checked_at'] is not None
            and run['integrity_checked_at'] <= as_of and run['recovery_point_at'] <= as_of
            and 0 <= as_of - run['recovery_point_at'] <= rpo)


def truth(asset_id, as_of):
    """Enumerate every trusted record of the asset; never a tool or model conclusion."""
    w = _world()
    asset = w['assets'][asset_id]
    policy = w['policies'][asset['policy_id']]
    rpo = policy['rpo']['value'] * UNIT_MINUTES[policy['rpo']['unit']]
    now = minutes(as_of)
    qualifying = sorted(r['run_id'] for r in w['runs'][asset_id] if usable(r, now, rpo, asset_id))
    return {'asset_id': asset_id, 'as_of': as_of, 'decision': 'ready' if qualifying else 'not_ready',
            'asset_record_id': asset['record_id'], 'policy_id': policy['policy_id'],
            'policy_record_id': policy['record_id'], 'inventory_id': inventory_id(asset_id, as_of),
            'qualifying_run_ids': qualifying}


def label(case):
    t = truth(case['asset_id'], case['as_of'])
    return dumps({k: t[k] for k in ('asset_id', 'as_of', 'decision', 'qualifying_run_ids')})


def target(question):
    for template in REQUESTS:
        pattern = re.escape(template).replace(r'\{asset_id\}', '(?P<asset>ast-[0-9a-f]{8})').replace(
            r'\{as_of\}', '(?P<time>[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}Z)')
        match = re.fullmatch(pattern, question)
        if match:
            return match['asset'], match['time']
    raise ValueError('Unknown versioned request format')


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def expected(question, privileged):
    """Verify the privileged row label against the independently regenerated world."""
    case = _world()['by_question'].get(question)
    if case is None or target(question) != (case['asset_id'], case['as_of']):
        raise ValueError('Request is not a generated case')
    if json.loads(privileged, object_pairs_hook=unique_object) != json.loads(label(case)):
        raise ValueError('Untrusted or mismatched verifier label')
    return dict(case)


def strict_json(text):
    """One JSON value: unique keys and no NaN/Infinity constants."""
    return json.loads(text, object_pairs_hook=unique_object,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def parse(text):
    """Whole response is one JSON object with exactly the contract fields and types."""
    value = strict_json(text)
    if not isinstance(value, dict) or set(value) != set(ANSWER_FIELDS):
        raise ValueError('invalid final object')
    runs = value['supporting_run_ids']
    if (any(type(value[k]) is not str for k in ANSWER_FIELDS[:-1]) or type(runs) is not list
            or any(type(r) is not str for r in runs)
            or not re.fullmatch(ASSET_ID, value['asset_id']) or not re.fullmatch(TIMESTAMP, value['as_of'])
            or value['decision'] not in DECISIONS):
        raise ValueError('invalid final object')
    return value


class EvidenceIntegrityError(RuntimeError):
    """A recorded observation differs from the trusted environment: untrusted execution."""


def observed(ledger):
    """Trusted successful observations of this episode, keyed by tool and exact arguments.

    ledger rows are {'tool', 'arguments', 'content'} for calls whose observation was
    joined to the sampled call in a later model input. Content must equal what the
    trusted environment returns for that call; anything else is not a policy outcome.
    """
    seen = {}
    for row in ledger:
        if set(row) != {'tool', 'arguments', 'content'} or row['tool'] not in TOOLS:
            raise EvidenceIntegrityError('Malformed observation ledger')
        try:
            trusted = call(row['tool'], row['arguments'])
        except PolicyActionError:
            raise EvidenceIntegrityError('Ledger records an invalid call as successful') from None
        if row['content'] != dumps(trusted):
            raise EvidenceIntegrityError('Observation differs from the trusted environment')
        seen[(row['tool'], dumps(row['arguments']))] = trusted
    return seen


def assess(text, case, ledger):
    """Independent terminal verification plus privileged diagnostics.

    The reward is 1 only for a contract-valid, correct and evidence-grounded answer.
    Diagnostics are computed after it, are never model inputs, and add no reward terms.
    None means not assessable; a malformed answer has no known semantic content.
    """
    t = truth(case['asset_id'], case['as_of'])
    seen = observed(ledger)
    grounded = {'asset': ('get_asset', dumps({'asset_id': t['asset_id']})) in seen,
                'policy': ('get_policy', dumps({'policy_id': t['policy_id']})) in seen,
                'inventory': ('list_backup_runs', dumps({'as_of': t['as_of'], 'asset_id': t['asset_id']})) in seen}
    coverage = {name: sum(k[0] == name for k in seen) for name in TOOLS}
    d = {'json_syntax': False, 'format_compliant': False, 'target_correct': None, 'time_correct': None,
         'decision_correct': None, 'asset_record_correct': None, 'policy_record_correct': None,
         'inventory_correct': None, 'supporting_runs_correct': None, 'evidence_correct': None,
         'required_evidence_observed': all(grounded.values()), 'observed': grounded,
         'distinct_successful_observations': coverage, 'failure_class': 'format'}
    try:
        strict_json(text)
        d['json_syntax'] = True
        answer = parse(text)
    except (ValueError, TypeError, RecursionError):
        return 0.0, d
    runs = answer['supporting_run_ids']
    d.update(format_compliant=True, target_correct=answer['asset_id'] == t['asset_id'],
             time_correct=answer['as_of'] == t['as_of'])
    if not (d['target_correct'] and d['time_correct']):
        d['failure_class'] = 'semantic'
        return 0.0, d
    d['decision_correct'] = answer['decision'] == t['decision']
    d['asset_record_correct'] = answer['asset_record_id'] == t['asset_record_id'] and grounded['asset']
    d['policy_record_correct'] = answer['policy_record_id'] == t['policy_record_id'] and grounded['policy']
    d['inventory_correct'] = answer['inventory_id'] == t['inventory_id'] and grounded['inventory']
    if answer['decision'] == 'ready':
        # Any nonempty duplicate-free subset of the qualifying runs of the observed inventory.
        d['supporting_runs_correct'] = (bool(runs) and len(set(runs)) == len(runs) and grounded['inventory']
                                        and set(runs) <= set(t['qualifying_run_ids']))
    else:
        # Minimal v1 contract: the observed complete inventory is the evidence; cite no run.
        d['supporting_runs_correct'] = runs == [] and grounded['inventory']
    d['evidence_correct'] = all(d[k] for k in ('asset_record_correct', 'policy_record_correct',
                                               'inventory_correct', 'supporting_runs_correct'))
    correct = d['decision_correct'] and d['evidence_correct']
    d['failure_class'] = 'correct' if correct else 'semantic' if not d['decision_correct'] else 'evidence'
    return float(correct), d


def reward(text, case, ledger):
    return assess(text, case, ledger)[0]


def rows(split=None):
    """Transport rows: request, verifier-only label, opaque case ID and split only."""
    w = _world()
    out = {s: [{'question': r['question'], 'result': label(r), 'case_id': r['case_id'], 'split': s} for r in rs]
           for s, rs in w['cases'].items()}
    return out if split is None else out[split]


def packets():
    return [list(p) for p in _world()['packets']]


def cases():
    r = rows()
    by_id = {x['case_id']: x for x in r['development']}
    out = {'train_main.parquet': r['train'], 'dev_panel.parquet': r['development'], 'dev_panel.jsonl': r['development'],
           'final_eval.parquet': r['final'], 'final_eval.jsonl': r['final']}
    for number, ids in enumerate(packets(), 1):
        out[f'dev_packet_{number}.parquet'] = [by_id[i] for i in ids]
    return out


def family(case_id):
    """Trusted diagnostic metadata; not a row field, tool output or model input."""
    case = _world()['by_id'][case_id]
    return case['family'], case['variant']


def panels():
    """Frozen ordered development panel and packets, and the final-evaluation commitment.

    Identifiers bind ordered membership and content (case ID, request and label).
    """
    r = rows()
    by_id = {x['case_id']: x for x in r['development']}
    return {'development': {'panel_id': 'rr1-dev-' + digest(r['development'])[:16], 'split': 'development',
                            'file': 'dev_panel.parquet', 'case_ids': [x['case_id'] for x in r['development']],
                            'packets': [{'packet_id': f'rr1-dev-p{n}-' + digest([by_id[i] for i in ids])[:16],
                                         'file': f'dev_packet_{n}.parquet', 'case_ids': ids}
                                        for n, ids in enumerate(packets(), 1)]},
            'final': {'panel_id': 'rr1-final-' + digest(r['final'])[:16], 'split': 'final', 'cases': len(r['final']),
                      'ordered_case_ids_sha256': digest([x['case_id'] for x in r['final']]),
                      'use': 'no model evaluation during calibration; frozen for a later controlled study'}}


@lru_cache(maxsize=1)
def _environment():
    files = cases()
    return dumps({'task': TASK, 'recipe': SEED, 'world_sha256': digest(registry()),
            'ordered_cases_sha256': {n: digest(v) for n, v in files.items()},
            'tools': [{'name': n, 'description': TOOL_DESCRIPTIONS[n], 'input_schema': TOOL_SCHEMAS[n]} for n in TOOLS],
            'requests': list(REQUESTS), 'prompt': PROMPT, 'max_runs_per_inventory': MAX_RUNS,
            'output': {'fields': list(ANSWER_FIELDS), 'decision': list(DECISIONS),
                       'parser': 'whole-response-json-unique-keys-no-extra-fields-v1',
                       'ready': 'nonempty duplicate-free subset of qualifying observed runs, any order',
                       'not_ready': 'supporting_run_ids is []; the observed complete inventory is the evidence'},
            'reward': 'binary: contract-valid, correct and evidence-grounded; incomplete has no reward',
            'splits': {'train': 128, 'development': 24, 'final': 64}, 'panels': panels()})


def environment():
    return json.loads(_environment())


def identity(source):
    return {'name': TASK, 'visible_fields': ['question'], 'label_fields': ['result'],
            'episode_policy': EPISODE_POLICY, 'context_limits': CONTEXT_LIMITS,
            'data_mapping': {'QUESTION': 'input.question', 'RESULT': 'input.result',
                             'AGL_CASE_ID': 'input.case_id', 'AGL_DATA_SPLIT': 'input.split', 'EVAL_SPLIT': 'input.split'},
            'agent_class': 'recovery_agent.Agent', 'cwd': 'examples/calc_x',
            'mcp': {'command': 'training-python', 'args': ['-I', 'recovery_stdio.py'],
                    'environment': {'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1'},
                    'ownership_receipt': 'optional absolute directory argument; never a model input'},
            'local_probe': 'recovery_probe.check',
            'integration_rule': ('reader-integration-v2; chain coverage is reported for every calibration stage and every '
                                 'successful call of an admitted episode must be joined; a stage without chains is a '
                                 'result, not an admission failure'),
            'environment': environment(), 'environment_sha256': digest(environment()),
            'source_sha256': {n: source['files'][f'experiment/{n}'] for n in
                              ('recovery_task.py', 'recovery_server.py', 'recovery_stdio.py', 'recovery_agent.py',
                               'recovery_probe.py', 'recovery_solver.py', 'recovery_calibration.py',
                               'recovery_prompt.py', 'reader_modes.py',
                               'owned_processes.py', 'observed_runtime.py')},
            'entrypoint_lock_sha256': source['files']['experiment/ppo_source_lock.json'],
            'launcher_sha256': source['files']['experiment/launch.py']}


def _world_bytes():
    return (dumps(registry()) + '\n').encode()


def generate(directory):
    import pyarrow as pa
    import pyarrow.parquet as pq
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'world.json').write_bytes(_world_bytes())
    manifest = {'schema': 'recovery-data-v1', 'task': TASK, 'environment': environment(),
                'environment_sha256': digest(environment()), 'world_file_sha256': hash_file(directory/'world.json'),
                'files': {}}
    for name, data in cases().items():
        p = directory/name
        if p.suffix == '.parquet':
            pq.write_table(pa.Table.from_pylist(data), p, compression='NONE', version='2.6')
        else:
            p.write_text(''.join(dumps(r) + '\n' for r in data))
        manifest['files'][name] = {'rows': len(data), 'split': SPLIT_OF_FILE[name],
                                   'case_ids': [r['case_id'] for r in data], 'sha256': hash_file(p)}
    (directory/'data_manifest.json').write_text(dumps(manifest) + '\n')
    return verify(directory)


def validate_manifest(m):
    """Stdlib-only identity check; membership, order and split of every panel are frozen."""
    if (set(m) != {'schema', 'task', 'environment', 'environment_sha256', 'world_file_sha256', 'files'}
            or m['schema'] != 'recovery-data-v1' or m['task'] != TASK or m['environment'] != environment()
            or m['environment_sha256'] != digest(environment()) or set(m['files']) != set(FILES)
            or m['world_file_sha256'] != hashlib.sha256(_world_bytes()).hexdigest()):
        raise ValueError('Incompatible synthetic environment manifest')
    for name, data in cases().items():
        item = m['files'][name]
        if (set(item) != {'rows', 'split', 'case_ids', 'sha256'} or len(data) != FILES[name]
                or item['rows'] != FILES[name] or item['split'] != SPLIT_OF_FILE[name]
                or item['case_ids'] != [r['case_id'] for r in data]
                or not re.fullmatch('[a-f0-9]{64}', item['sha256'])):
            raise ValueError('Changed ordered dataset membership, split or hash')


def read_rows(path):
    path = Path(path)
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    return [json.loads(s, object_pairs_hook=unique_object) for s in path.read_text().splitlines()]


DEVELOPMENT_FILES = tuple(n for n in FILES if SPLIT_OF_FILE[n] == 'development')


def verify(directory, scope='all'):
    """Verify the manifest and the bytes and rows of the files in scope.

    scope='all' is the dataset-consistency check used by generation, preparation and
    CPU preflight; it may inspect every split. scope='development' is the model-facing
    calibration check: it opens only world.json and the development files. The other
    declared files must exist as regular files but are never opened, so no
    final-evaluation record can reach a calibration stage through this path.
    """
    if scope not in ('all', 'development'):
        raise ValueError('Unknown dataset verification scope')
    directory = Path(directory)
    m = json.loads((directory/'data_manifest.json').read_text(), object_pairs_hook=unique_object)
    validate_manifest(m)
    if {p.name for p in directory.iterdir()} != set(FILES) | {'world.json', 'data_manifest.json'}:
        raise ValueError('Undeclared dataset inputs')
    for p in directory.iterdir():
        if p.is_symlink() or not p.is_file():
            raise ValueError('Dataset must contain regular files')
    if hash_file(directory/'world.json') != m['world_file_sha256']:
        raise ValueError('Changed world bytes')
    trusted = cases()
    for name in (FILES if scope == 'all' else DEVELOPMENT_FILES):
        if hash_file(directory/name) != m['files'][name]['sha256']:
            raise ValueError('Changed dataset bytes')
        if read_rows(directory/name) != trusted[name]:
            raise ValueError('Changed cases/labels/order, duplicate records, or split leakage')
    return m
