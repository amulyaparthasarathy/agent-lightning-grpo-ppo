"""Linux MCP ownership receipts. Never signal from malformed or stale identity."""
import json
import os
from pathlib import Path
import signal
import time
from uuid import uuid4


def start_ticks(pid):
    # comm can contain spaces and parentheses; fields after the final ')' start at field 3.
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]


def register(directory):
    pid = os.getpid()
    if os.getpgrp() != pid:
        raise RuntimeError('MCP server requires its own process group')
    receipt = {'pid': pid, 'start_ticks': start_ticks(pid), 'environment_keys': sorted(os.environ)}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = uuid4().hex
    # A crash leaves a .pending diagnostic, never a partially published receipt.
    pending = directory/(name+'.pending')
    with pending.open('x') as f:
        json.dump(receipt, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(pending, directory/(name+'.json'))
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _unique(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError('Duplicate receipt key')
        result[k] = v
    return result


def _live_group(pid):
    # Zombies need reaping by their parent, but are no longer executing.
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = path.read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == pid and fields[0] != 'Z':
                return True
        except (FileNotFoundError, ProcessLookupError):
            continue
    return False


def cleanup(directory):
    """Best effort per receipt; structured anomalies survive independent cleanup.

    No receipt contents are included in parse errors. An interrupted publication
    cannot establish ownership, so it is retained and reported as unresolved.
    SIGKILL delivery is recorded separately from observed group termination;
    reaping remains the direct parent's responsibility.
    """
    report = {'complete': True, 'signalled': 0, 'receipts': [], 'anomalies': []}
    try:
        paths = sorted(Path(directory).iterdir())
    except Exception as exc:
        paths = []
        report['anomalies'].append({'reason': 'receipt_directory_unreadable', 'error_type': type(exc).__name__})
    rows = []
    for path in paths:
        try:
            if path.suffix != '.json' or path.is_symlink() or not path.is_file():
                raise ValueError('Unpublished or unsafe receipt')
            row = json.loads(path.read_text(), object_pairs_hook=_unique)
            if (not isinstance(row, dict) or set(row) != {'pid', 'start_ticks', 'environment_keys'}
                    or type(row['pid']) is not int or row['pid'] <= 1
                    or not isinstance(row['start_ticks'], str) or not row['start_ticks'].isascii()
                    or not row['start_ticks'].isdigit() or not isinstance(row['environment_keys'], list)
                    or any(not isinstance(k, str) for k in row['environment_keys'])):
                raise ValueError('Invalid receipt fields')
            rows.append((path, row))
        except Exception as exc:
            report['anomalies'].append({'file': path.name, 'reason': 'invalid_or_unpublished_receipt', 'error_type': type(exc).__name__})
    for path, row in rows:
        entry = {'file': path.name, 'pid': row['pid'], 'start_ticks': row['start_ticks']}
        report['receipts'].append(entry)
        try:
            if sum(r['pid'] == row['pid'] for _, r in rows) != 1:
                raise ValueError('Ambiguous duplicate PID receipts')
            try:
                ticks = start_ticks(row['pid'])
            except (FileNotFoundError, ProcessLookupError):
                if _live_group(row['pid']):
                    raise ValueError('Leader absent with unresolved process group')
                entry['status'] = 'already_exited'
                continue
            if ticks != row['start_ticks'] or os.getpgid(row['pid']) != row['pid']:
                raise ValueError('Ownership no longer verifiable')
            os.killpg(row['pid'], signal.SIGKILL)
            report['signalled'] += 1
            entry['signal'] = 'SIGKILL'
            deadline = time.monotonic()+2
            while _live_group(row['pid']):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Owned group termination unconfirmed')
                time.sleep(.01)
            entry['status'] = 'group_terminated'
        except Exception as exc:
            entry['status'] = 'unresolved'
            report['anomalies'].append({'file': path.name, 'reason': 'ownership_or_cleanup_unresolved', 'error_type': type(exc).__name__})
    report['complete'] = not report['anomalies']
    return report
