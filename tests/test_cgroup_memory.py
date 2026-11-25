import tempfile
from pathlib import Path
import unittest

from experiment.ppo_runtime import _cgroup_memory


class CgroupMemoryTests(unittest.TestCase):
    def fixture(self, membership, filesystem):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        proc = Path(tmp.name)/'proc'
        proc.mkdir()
        mount = Path(tmp.name)/'memory mount'
        group = mount/'worker'
        group.mkdir(parents=True)
        encoded_mount = str(mount).replace(' ', r'\040')
        (proc/'mountinfo').write_text(f'31 1 0:6 /jobs {encoded_mount} rw - {filesystem}\n')
        (proc/'cgroup').write_text(membership)
        return proc, group

    def test_v1_memory_controller_in_hybrid_subtree(self):
        proc, group = self.fixture('0::/other\n5:cpu:/shared\n9:memory:/jobs/worker', 'cgroup cgroup rw,memory')
        (group.parent/'memory.max_usage_in_bytes').write_text('99999')
        (group/'memory.usage_in_bytes').write_text('0')
        (group/'memory.max_usage_in_bytes').write_text('7291')
        (group/'memory.failcnt').write_text('1')
        (group/'cgroup.procs').write_text('10\n11\n')
        result = _cgroup_memory(proc)
        self.assertEqual((result['cgroup_version'],result['memory.current'],result['memory.peak'],
                          result['memory.failcnt'],result['cgroup_process_count']), (1,'0','7291','1',2))
        self.assertTrue(result['cgroup_available'])
        self.assertIn('emitting process', result['scope'])

    def test_v2_unified_and_malformed_counters(self):
        proc, group = self.fixture('0::/jobs/worker', 'cgroup2 cgroup rw')
        (group/'memory.current').write_text('20')
        (group/'memory.peak').write_text('40')
        (group/'memory.events').write_text('oom 0\noom_kill 0')
        self.assertEqual(_cgroup_memory(proc)['memory.peak'], '40')
        (group/'memory.current').write_text('max')
        (group/'memory.peak').write_text('-1')
        (group/'memory.events').write_text('fatal')
        (group/'cgroup.procs').write_text('abc')
        result = _cgroup_memory(proc)
        self.assertFalse(result['cgroup_available'])
        self.assertEqual(result['cgroup_version'],2)
        self.assertTrue(all(result[k] is None for k in ('memory.current','memory.peak','memory.events','cgroup_process_count')))

    def test_missing_membership_or_unrelated_mount(self):
        proc, group = self.fixture('0::/other/worker', 'cgroup2 cgroup rw')
        (group/'memory.peak').write_text('100')
        for content in ('0::/other/worker', 'malformed', '0::/jobs/../other', ''):
            (proc/'cgroup').write_text(content)
            self.assertIsNone(_cgroup_memory(proc)['memory.peak'])
        (proc/'cgroup').unlink()
        self.assertFalse(_cgroup_memory(proc)['cgroup_available'])
