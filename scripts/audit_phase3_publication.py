#!/usr/bin/env python3
"""Redacted heuristic publication audit: tree, index, patch, branch blobs and metadata.

This is a review aid, not proof that arbitrary secrets or restricted content are absent.
Never reads ignored files or credential stores. --text can include a prepared PR body.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    'private-key': rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'github-token': rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})',
    'provider-token': rb'\b(?:sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{25,}|hf_[A-Za-z0-9]{25,}|AKIA[A-Z0-9]{16}|xox[baprs]-[A-Za-z0-9-]{20,})',
    'authenticated-url': rb'https?://[^\s/\"\'<>]+:[^\s/\"\'<>]+@',
    'long-credential-literal': rb'(?i)(?:api_key|password|access_token|secret_key)\s*[=:]\s*[\"\'][A-Za-z0-9_+/=-]{24,}[\"\']',
}
FORBIDDEN = {'.safetensors', '.gguf', '.pt', '.pth', '.pem', '.key', '.zip', '.tgz', '.whl'}


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='5c069f7e9cc23d03050d13360ce43e27e47e54aa')
    parser.add_argument('--text', action='append', type=Path, default=[])
    args = parser.parse_args()
    findings = []; counts = {'worktree_files': 0, 'index_blobs': 0, 'branch_blobs': 0, 'branch_commits': 0}
    def scan(label, data, name=None):
        if name and (Path(name).suffix in FORBIDDEN or (Path(name).name.startswith('.env') and Path(name).name != '.env.example')):
            findings.append({'location': label, 'rule': 'restricted-file-type'})
        if len(data) > 5*1024**2:
            findings.append({'location': label, 'rule': 'large-blob-review'})
        for rule, pattern in PATTERNS.items():
            for match in re.finditer(pattern, data):
                # Explicit negative-test URL is not a credential; still report for manual adjudication.
                findings.append({'location': label, 'rule': rule, 'line': data[:match.start()].count(b'\n')+1})
    names = git('ls-files', '-z', '--cached', '--others', '--exclude-standard').decode().split('\0')
    for name in sorted(set(names)-{''}):
        path = ROOT/name
        if path.is_symlink():
            findings.append({'location': name, 'rule': 'symlink-review'})
        elif path.is_file():
            scan('worktree:'+name, path.read_bytes(), name); counts['worktree_files'] += 1
    for entry in git('ls-files', '-s', '-z').decode().split('\0'):
        if not entry: continue
        info, name = entry.split('\t', 1); sha = info.split()[1]
        scan('index:'+name, git('cat-file', 'blob', sha), name); counts['index_blobs'] += 1
    scan('staged-patch', git('diff', '--cached', '--binary'))
    for commit in git('rev-list', args.base+'..HEAD').decode().splitlines():
        scan('commit-metadata:'+commit, git('show', '-s', '--format=fuller', commit))
        counts['branch_commits'] += 1
    for entry in git('rev-list', '--objects', args.base+'..HEAD').decode().splitlines():
        sha, _, name = entry.partition(' ')
        if git('cat-file', '-t', sha).strip() == b'blob':
            scan('branch-blob:'+sha+':'+name, git('cat-file', 'blob', sha), name)
            counts['branch_blobs'] += 1
    for path in args.text:
        scan('publication-text:'+path.name, path.read_bytes())
    print(json.dumps({'scope': counts, 'findings': findings, 'secret_values_printed': False,
                      'limitations': 'Heuristic patterns; manual diff/content/license review still required'}, indent=2))
    return bool(findings)


if __name__ == '__main__':
    raise SystemExit(main())
