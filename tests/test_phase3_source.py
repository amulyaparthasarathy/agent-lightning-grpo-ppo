"""Real temporary Git histories and closed image trees; no provider calls."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from experiment import reader_source as source, reader_contract as c
from test_reader_training import ROOT, plan


class SourceTests(unittest.TestCase):
    def git(self, root, *args):
        return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.DEVNULL, text=True).strip()

    def checkout(self, root):
        self.git(root, 'init', '-q')
        self.git(root, 'config', 'user.email', 'test@example.invalid')
        self.git(root, 'config', 'user.name', 'Synthetic Test')
        self.git(root, 'remote', 'add', 'origin', 'git@github.com:Reader/lab.git')
        (root/'experiment').mkdir()
        names = ['experiment/reader_runtime.json', 'experiment/task.py']
        (root/names[0]).write_text(json.dumps(names))
        (root/names[1]).write_text('value = 1\n')
        (root/'.gitignore').write_text('ignored.py\noutputs/\n')
        self.git(root, 'add', '.')
        self.git(root, 'commit', '-qm', 'Synthetic baseline')
        baseline = dict(source.BASELINE, commit=self.git(root, 'rev-parse', 'HEAD'))
        return baseline

    def test_calculator_agent_body_preserved(self):
        import ast
        old = self.git(ROOT, 'show', source.BASELINE['commit']+':experiment/observed_calc_agent.py')
        current = (ROOT/'experiment/observed_calc_agent.py').read_text()
        agent = lambda text: ast.dump(next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == 'Agent'))
        self.assertEqual(agent(old), agent(current))

    def test_supported_normalizations_and_redacted_rejections(self):
        for url in ('https://github.com/Reader/Lab.git', 'https://github.com/reader/lab',
                    'git@github.com:Reader/Lab.git', 'ssh://git@github.com/Reader/Lab.git'):
            self.assertEqual(source.repository(url), 'reader/lab')
        for value in ('https://secret@github.com/u/r', 'https://u:secret@github.com/u/r',
                      'https://github.com/u/r?secret', 'https://github.com/u/r#secret',
                      'ssh://secret@github.com/u/r', 'git://github.com/u/r', 'file:///secret',
                      'https://other.example/u/r', 'https://github.com/u/../r',
                      'https://github.com/u/r;secret', 'git@github.com:u/$(secret)',
                      'https://github.com/u/r\nsecret', 'https://github.com/-u/r',
                      'https://github.com/u/r%2Fsecret', '--upload-pack=secret'):
            with self.subTest(value=value), self.assertRaises(ValueError) as result:
                source.repository(value)
            self.assertNotIn('secret', str(result.exception))

    def test_real_fork_direct_clone_dirty_staged_untracked_and_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); baseline = self.checkout(root)
            with patch.object(source, 'BASELINE', baseline):
                first = source.identity(root, c.file_hash, clean=True)
                self.assertEqual(first['repository'], 'reader/lab')
                self.assertEqual(first['baseline'], baseline)
                for origin in ('https://github.com/amulyaparthasarathy/agent-lightning-grpo-ppo.git', 'git@github.com:Reader/lab.git'):
                    self.git(root, 'remote', 'set-url', 'origin', origin)
                    self.assertEqual(source.identity(root, c.file_hash)['repository'], source.repository(origin))
                (root/'experiment/task.py').write_text('value = 2\n')
                with self.assertRaisesRegex(ValueError, 'clean committed'): source.identity(root, c.file_hash, clean=True)
                self.git(root, 'add', '.')
                with self.assertRaisesRegex(ValueError, 'clean committed'): source.identity(root, c.file_hash, clean=True)
                self.git(root, 'commit', '-qm', 'Derivative implementation')
                current = source.identity(root, c.file_hash, clean=True)
                self.assertNotEqual(current['commit'], first['commit'])
                self.git(root, 'update-index', '--assume-unchanged', 'experiment/task.py')
                (root/'experiment/task.py').write_text('hidden modification')
                with self.assertRaisesRegex(ValueError, 'clean committed source bytes'):
                    source.identity(root, c.file_hash, clean=True)
                (root/'experiment/task.py').write_text('value = 2\n')
                self.git(root, 'update-index', '--no-assume-unchanged', 'experiment/task.py')
                (root/'extra.py').write_text('untracked')
                with self.assertRaisesRegex(ValueError, 'clean committed'): source.identity(root, c.file_hash, clean=True)
                (root/'extra.py').unlink()
                (root/'experiment/ignored.py').write_text('ignored executable')
                self.assertEqual(source.identity(root, c.file_hash, clean=True), current)
                self.assertNotIn('experiment/ignored.py', source.inventory(root))
                (root/'experiment/task.py').unlink(); (root/'experiment/task.py').symlink_to('/etc/hosts')
                with self.assertRaisesRegex(ValueError, 'symlink'): source.identity(root, c.file_hash)

    def test_missing_unrelated_and_shallow_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'repo'; root.mkdir(); baseline = self.checkout(root)
            with patch.object(source, 'BASELINE', dict(baseline, commit='1'*40)), self.assertRaisesRegex(ValueError, 'baseline object missing'):
                source.identity(root, c.file_hash)
            original_branch = self.git(root, 'branch', '--show-current')
            self.git(root, 'checkout', '--orphan', 'unrelated')
            self.git(root, 'commit', '-qm', 'Unrelated root')
            with patch.object(source, 'BASELINE', baseline), self.assertRaisesRegex(ValueError, 'does not descend'):
                source.identity(root, c.file_hash)
            self.git(root, 'checkout', original_branch)
            (root/'experiment/task.py').write_text('next = True')
            self.git(root, 'commit', '-qam', 'Next source')
            shallow = Path(tmp)/'shallow'
            subprocess.run(['git', 'clone', '-q', '--depth', '1', root.as_uri(), str(shallow)], check=True)
            self.git(shallow, 'remote', 'set-url', 'origin', 'https://github.com/reader/lab')
            with patch.object(source, 'BASELINE', baseline), self.assertRaisesRegex(ValueError, 'possibly shallow'):
                source.identity(shallow, c.file_hash)

    def test_replacement_cannot_admit_different_runtime_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); baseline = self.checkout(root)
            (root/'experiment/task.py').write_text('value = 999\n')
            self.git(root, 'commit', '-qam', 'Replacement bytes')
            replacement = self.git(root, 'rev-parse', 'HEAD')
            self.git(root, 'checkout', '--detach', baseline['commit'])
            self.git(root, 'replace', baseline['commit'], replacement)
            self.git(root, 'read-tree', '--reset', '-u', 'HEAD')
            self.assertEqual((root/'experiment/task.py').read_text(), 'value = 999\n')
            with patch.object(source, 'BASELINE', baseline), self.assertRaisesRegex(ValueError, 'clean committed'):
                source.identity(root, c.file_hash, clean=True)
            self.assertEqual(self.git(root, 'replace', '-l'), baseline['commit'])

    def test_grafts_cannot_fabricate_baseline_ancestry(self):
        for inherited in (False, True):
            with self.subTest(inherited=inherited), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)/'repo'; root.mkdir(); baseline = self.checkout(root)
                self.git(root, 'checkout', '--orphan', 'unrelated')
                self.git(root, 'commit', '-qm', 'Unrelated root')
                head = self.git(root, 'rev-parse', 'HEAD')
                with patch.object(source, 'BASELINE', baseline):
                    with self.assertRaisesRegex(ValueError, 'does not descend'):
                        source.identity(root, c.file_hash, clean=True)
                    graft = Path(tmp)/'grafts' if inherited else root/'.git/info/grafts'
                    content = head+' '+baseline['commit']+'\n'
                    graft.write_text(content)
                    with patch.dict(os.environ, {'GIT_GRAFT_FILE': str(graft)} if inherited else {}):
                        # The uncontrolled Git view accepts this fabricated relationship.
                        self.git(root, 'merge-base', '--is-ancestor', baseline['commit'], head)
                        with self.assertRaisesRegex(ValueError, 'does not descend'):
                            source.identity(root, c.file_hash, clean=True)
                    self.assertEqual(graft.read_text(), content)

    def test_linked_worktree_raw_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'repo'; root.mkdir(); baseline = self.checkout(root)
            linked = Path(tmp)/'linked'
            self.git(root, 'worktree', 'add', '-b', 'reader', str(linked))
            with patch.object(source, 'BASELINE', baseline):
                admitted = source.identity(linked, c.file_hash, clean=True)
            self.assertEqual(admitted['commit'], baseline['commit'])
            self.assertEqual(admitted['repository'], 'reader/lab')

    def test_package_closed_set_hashes_and_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); baseline = self.checkout(root)
            with patch.object(source, 'BASELINE', baseline):
                admitted = source.identity(root, c.file_hash)
            package = root/'package'; package.mkdir()
            for name in admitted['runtime_files']:
                (package/name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root/name, package/name)
            expected = admitted['runtime_files']
            source.verify_package(package, expected, c.file_hash)
            for name in ('experiment/ignored.py', 'experiment/__pycache__/task.cpython-312.pyc', 'rogue.py'):
                p = package/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('extra')
                with self.assertRaisesRegex(ValueError, 'Undeclared'): source.verify_package(package, expected, c.file_hash)
                p.unlink()
            p = package/'experiment/task.py'; p.write_text('changed')
            with self.assertRaises(ValueError): source.verify_package(package, expected, c.file_hash)
            p.unlink()
            with self.assertRaises(ValueError): source.verify_package(package, expected, c.file_hash)
            p.symlink_to(root/'experiment/task.py')
            with self.assertRaises(ValueError): source.verify_package(package, expected, c.file_hash)

    def test_plan_tampering_and_legacy_regeneration(self):
        p = plan()
        for edit in (lambda x: x['source']['baseline'].update(commit='a'*40),
                     lambda x: x['source'].update(repository='reader/lab?credential'),
                     lambda x: x['source']['files'].update({'../escape.py': 'a'*64}),
                     lambda x: x['source']['runtime_files'].update({'reader_app.py': 'a'*64}),
                     lambda x: x.update(configuration_sha256='a'*64),
                     lambda x: x.update(schema='reader-reference-v1'),
                     lambda x: x['upstream'].update(agentlightning_commit='a'*40)):
            bad = copy.deepcopy(p); edit(bad)
            with self.assertRaises(ValueError): c.validate(bad)
        with patch.object(c, 'source_identity', return_value={}), self.assertRaisesRegex(ValueError, 'Stale'):
            c.validate_current(p, p['profile'])
