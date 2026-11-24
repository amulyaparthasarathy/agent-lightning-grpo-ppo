"""Keep reader requests out of Ray's single serialized runtime-env argument.

The complete request remains in the owned stage directory and is content-bound.
Historical matched runs retain their inline environment representation.
"""
import hashlib
import json
import os
from pathlib import Path
import re


def digest(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def environment(request, run):
    if request['plan'].get('schema') != 'reader-task-v2':
        return {'AGL_MATCHED_REQUEST': json.dumps(request)}
    return {'AGL_MATCHED_REQUEST_FILE': str(Path(run)/'request.json'),
            'AGL_MATCHED_REQUEST_SHA256': digest(request)}


def load(environ=None):
    env = os.environ if environ is None else environ
    if 'AGL_MATCHED_REQUEST_FILE' not in env:
        if 'AGL_MATCHED_REQUEST_SHA256' in env:
            raise ValueError('Incomplete request file binding')
        request = json.loads(env['AGL_MATCHED_REQUEST'])
        if request['plan'].get('schema') == 'reader-task-v2':
            raise ValueError('Reader workers require a content-bound request file')
        return request
    if 'AGL_MATCHED_REQUEST' in env:
        raise ValueError('Ambiguous inline and file request bindings')
    path = Path(env['AGL_MATCHED_REQUEST_FILE'])
    run = Path(env['AGL_PPO_RUN_DIR'])
    sha = env.get('AGL_MATCHED_REQUEST_SHA256', '')
    if (not run.is_absolute() or path != run/'request.json' or path.resolve() != path
            or not path.is_file() or path.stat().st_size > 8*1024**2
            or not re.fullmatch('[0-9a-f]{64}', sha)):
        raise ValueError('Unsafe or incomplete owned request file binding')
    request = json.loads(path.read_text())
    if digest(request) != sha or request['plan'].get('schema') != 'reader-task-v2':
        raise ValueError('Request content identity differs')
    return request
