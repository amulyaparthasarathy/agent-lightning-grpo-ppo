"""Linux MCP process receipts; PID start ticks prevent signaling a reused PID."""
import json
import os
from pathlib import Path
import signal
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
    with (directory/(uuid4().hex+'.json')).open('x') as f:
        json.dump(receipt, f)


def cleanup(directory):
    count = 0
    for path in Path(directory).glob('*.json'):
        row = json.loads(path.read_text())
        try:
            if start_ticks(row['pid']) == row['start_ticks'] and os.getpgid(row['pid']) == row['pid']:
                os.killpg(row['pid'], signal.SIGKILL)
                count += 1
        except (FileNotFoundError, ProcessLookupError):
            pass
    return count
