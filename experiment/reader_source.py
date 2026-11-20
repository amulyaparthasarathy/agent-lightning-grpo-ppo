"""Offline source admission. Git ancestry is not GitHub ownership or push proof."""
from pathlib import Path, PurePosixPath
import json
import os
import re
import subprocess

BASELINE = {'repository': 'amulyaparthasarathy/agent-lightning-grpo-ppo',
            'commit': '5c069f7e9cc23d03050d13360ce43e27e47e54aa'}
INVENTORY = 'experiment/reader_runtime.json'


def repository(value):
    # Deliberately do not parse or echo arbitrary URLs in diagnostics.
    patterns = (r'https://github\.com/([^/]+)/([^/]+)',
                r'git@github\.com:([^/]+)/([^/]+)',
                r'ssh://git@github\.com/([^/]+)/([^/]+)')
    match = next((m for p in patterns if (m := re.fullmatch(p, value or ''))), None)
    if not match:
        raise ValueError('Unsupported origin: use credential-free GitHub HTTPS or git SSH')
    owner, repo = match.groups()
    repo = repo.removesuffix('.git')
    if (not re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?', owner)
            or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]{0,99}', repo)
            or '..' in repo or repo.endswith('.')):
        raise ValueError('Unsafe GitHub repository identity')
    return f'{owner.lower()}/{repo.lower()}'


def safe_path(value):
    if (not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_./-]+', value)
            or PurePosixPath(value).is_absolute() or any(x in ('', '.', '..') for x in value.split('/'))):
        raise ValueError('Unsafe source inventory path')
    return value


def run_git(root, *args, input=None, check=True):
    """Read the checkout's raw objects, never inherited alternate Git views.

    NO_REPLACE_OBJECTS does not disable grafts. Override the graft file as well,
    and bypass cached commit graphs when proving ancestry. Linked worktrees are
    discovered normally through -C; no reader metadata is changed.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    env.update(GIT_NO_REPLACE_OBJECTS='1', GIT_GRAFT_FILE=os.devnull)
    result = subprocess.run(['git', '--no-replace-objects', '-c', 'core.commitGraph=false',
                             '-C', str(root), *args], input=input, env=env,
                            capture_output=True, text=True)
    if check and result.returncode:
        raise ValueError('Local Git check failed; verify checkout and origin (details suppressed)')
    return result


def git(root, *args):
    return run_git(root, *args).stdout.strip()


def inventory(root):
    names = json.loads((Path(root)/INVENTORY).read_text())
    if not isinstance(names, list) or names != sorted(set(names)) or INVENTORY not in names:
        raise ValueError('Runtime inventory must be a sorted unique list including itself')
    tracked = set(git(root, 'ls-files', '-z').split('\0'))
    for name in names:
        safe_path(name)
        p = Path(root)/name
        if name not in tracked or not p.is_file() or p.resolve() != p.absolute():
            raise ValueError('Runtime inventory requires tracked regular files without symlinks')
    return names


def identity(root, file_hash, clean=False):
    root = Path(root).resolve()
    dirty = bool(git(root, 'status', '--porcelain', '--untracked-files=all'))
    if clean and dirty:
        raise ValueError('Requires clean committed source; commit and regenerate the plan')
    origin = repository(git(root, 'remote', 'get-url', 'origin'))
    head = git(root, 'rev-parse', 'HEAD')
    if run_git(root, 'cat-file', '-e', BASELINE['commit']+'^{commit}', check=False).returncode:
        raise ValueError('Canonical baseline object missing (possibly shallow clone). Fetch public main/history '
                         'from the canonical repository, then regenerate; fork HEAD is not a baseline.')
    if run_git(root, 'merge-base', '--is-ancestor', BASELINE['commit'], head, check=False).returncode:
        raise ValueError('Execution commit does not descend from the pinned canonical baseline; '
                         'fetch full history if shallow, or rebase the derivative onto public main')
    runtime = inventory(root)
    files = {}
    for name in git(root, 'ls-files', '-z').split('\0'):
        if not name:
            continue
        safe_path(name)
        p = root/name
        if p.is_symlink() or p.resolve() != p:
            raise ValueError('Source symlinks are not admitted')
        if p.suffix in ('.py', '.sh', '.json', '.jinja', '.txt', '.toml', '.lock', '.md', '.yaml', '.yml'):
            files[name] = file_hash(p)
    # Status can be fooled by assume-unchanged/skip-worktree flags. Compare raw
    # working bytes to HEAD blobs as well, without filters or external diff commands.
    head_files = {}
    for entry in git(root, 'ls-tree', '-r', '-z', 'HEAD').split('\0'):
        if entry:
            info, name = entry.split('\t', 1)
            head_files[name] = info.split()[2]
    hashes = run_git(root, 'hash-object', '--no-filters', '--stdin-paths',
                     input=''.join(n+'\n' for n in files))
    dirty = dirty or any(head_files.get(n) != h for n, h in zip(files, hashes.stdout.splitlines(), strict=True))
    if clean and dirty:
        raise ValueError('Requires clean committed source bytes; commit and regenerate the plan')
    return {'commit': head, 'dirty': dirty, 'repository': origin, 'baseline': dict(BASELINE),
            'files': files, 'runtime_files': {n: files[n] for n in runtime}}


def verify_package(root, expected, file_hash):
    """Closed runtime tree. No undeclared files, including sourceless bytecode."""
    root = Path(root)
    for name, sha in expected.items():
        safe_path(name)
        p = root/name
        if p.resolve() != p.absolute() or not p.is_file() or file_hash(p) != sha:
            raise ValueError('Missing, changed or symlinked runtime input: '+name)
    if json.loads((root/INVENTORY).read_text()) != sorted(expected):
        raise ValueError('Declared runtime inventory differs from plan')
    actual = set()
    for p in root.rglob('*'):
        if p.is_symlink():
            raise ValueError('Symlink in runtime package')
        if p.is_file():
            # Interpreter caches are disallowed; worker environment disables writes.
            actual.add(p.relative_to(root).as_posix())
    if actual != set(expected):
        raise ValueError('Undeclared runtime input set; rebuild from the tracked inventory')
