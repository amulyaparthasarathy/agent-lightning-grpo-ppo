"""All passing quotas here are synthetic fixtures, never live capacity evidence."""
import copy
import unittest
from experiment import matched_contract as c, matched_exploratory as e, matched_capacity as formal


def fixture(version=1):
    return {'workspace':'amulyasp','environment':'main','observed_at':100,
            'evidence_reference':'SYNTHETIC TEST ONLY', 'resources':formal.RESOURCES,
            'volume_version':version,
            'volume_metadata':{'volume_name':c.VOLUME,'volume_id':'vo-fixture','source':'SYNTHETIC VolumeList',
                               'backend_version_raw':version,'backend_version_enum':f'VOLUME_FS_VERSION_V{version}'},
            'gpu_limit':6,'gpu_used':0,'gpu_limit_source':'SYNTHETIC quota',
            'container_limit':7,'containers_used':0,'container_limit_source':'SYNTHETIC quota',
            'cpu_ram_quotas':{'scope':'aggregate','source':'SYNTHETIC quota','cpu_limit':97,'cpu_used':0,
                              'memory_mib_limit':1181696,'memory_mib_used':0},
            'storage':{'capacity_kind':'elastic-no-fixed-byte-capacity','inventory_method':'modal-volume-iterdir',
                       'source':'SYNTHETIC inventory','listed_entries':10,'listed_bytes':100,
                       'planned_incremental_gib':600},
            'scratch':{'minimum_free_gib':100,'enforcement':'existing-worker-startup-check'},
            'billing':{'source':'SYNTHETIC billing','remaining_gross_usage_usd':225,'remaining_funded_spending_usd':225},
            'cuda_layers_cached':True,'cuda_cache_source':'SYNTHETIC historical image','funding':e.funding()}


class CapacityReviewTests(unittest.TestCase):
    def test_explicit_v1_and_v2_qualification_compatibility_leaves_input_unchanged(self):
        for version in (1,2):
            data=fixture(version);before=copy.deepcopy(data)
            e.validate_capacity(data,'exploratory-qualification',now=100,bounded_admission=True)
            self.assertEqual(data,before)

    def test_missing_metadata_or_fabricated_v2_attestation_fails(self):
        for edit in (lambda x:x.update(volume_version=2),lambda x:x.pop('volume_metadata'),
                     lambda x:x['volume_metadata'].update(volume_name='other-volume'),
                     lambda x:x['volume_metadata'].update(backend_version_raw=0),
                     lambda x:x['volume_metadata'].update(backend_version_raw=True),
                     lambda x:x['volume_metadata'].update(backend_version_raw=3)):
            data=fixture();edit(data)
            with self.assertRaisesRegex(ValueError,'backend metadata'):
                e.validate_capacity(data,'exploratory-qualification',now=100,bounded_admission=True)

    def test_formal_gate_still_rejects_v1(self):
        data=fixture()
        with self.assertRaisesRegex(ValueError,'version'):formal.validate_capacity(data,2,now=100,storage=False)

    def test_two_lanes_have_three_v1_writers_and_no_probe(self):
        e.validate_capacity(fixture(),'exploratory',now=100)
        self.assertEqual(e.bounds('exploratory')['concurrency'],2)
        self.assertFalse(e.funding()['admission_probe_required'])

    def test_statfs_placeholder_and_missing_inventory_are_not_capacity(self):
        edits=(lambda x:x.update(volume_free_gib=999999),
               lambda x:x['storage'].update(inventory_method='df'),
               lambda x:x['storage'].update(inventory_method='statfs'),
               lambda x:x['storage'].update(listed_entries=500000),
               lambda x:x['storage'].update(listed_bytes=float('nan')))
        for edit in edits:
            data=fixture();edit(data)
            with self.assertRaises(ValueError):e.validate_capacity(data,'exploratory-qualification',now=100,bounded_admission=True)

    def test_unknown_billing_and_stale_observation_still_fail_closed(self):
        edits=(lambda x:x['billing'].update(remaining_gross_usage_usd=None),
               lambda x:x['billing'].update(remaining_gross_usage_usd=49),
               lambda x:x.update(observed_at=None),lambda x:x.update(observed_at=-801))
        for edit in edits:
            data=fixture();edit(data)
            with self.assertRaises(ValueError):e.validate_capacity(data,'exploratory-qualification',now=100,bounded_admission=True)
        data=fixture();data['billing']['remaining_funded_spending_usd']=None
        with self.assertRaises(ValueError):e.validate_capacity(data,'exploratory-qualification',now=100)

    def test_documented_per_container_only_quotas_need_actual_provider_confirmation(self):
        data=fixture();data['cpu_ram_quotas']={'scope':'per-container-only-confirmed','source':'SYNTHETIC provider confirmation',
                                            'max_cpu':16,'max_memory_mib':196608}
        e.validate_capacity(data,'exploratory-qualification',now=100,bounded_admission=True)
        data['cpu_ram_quotas']['source']=''
        with self.assertRaises(ValueError):e.validate_capacity(data,'exploratory',now=100)

    def test_only_explicit_bounded_mode_defers_unknown_capacity_and_cache(self):
        data=fixture(2)
        data.update(gpu_limit=None,container_limit=None,cpu_ram_quotas={'scope':'unknown'},
                    cuda_layers_cached=None,cuda_cache_source=None)
        before=copy.deepcopy(data)
        e.validate_capacity(data,'exploratory-qualification',now=100,bounded_admission=True)
        self.assertEqual(data,before)  # No fabricated attestations or mutation.
        e.validate_capacity(data,'exploratory',now=100,bounded_admission=True)
        with self.assertRaises(ValueError):e.validate_capacity(data,'exploratory-qualification',now=100)
        with self.assertRaises(ValueError):e.validate_capacity(data,'exploratory',now=100)
        for edit in (lambda x:x.update(gpu_limit=None),lambda x:x.update(container_limit=None),
                     lambda x:x.update(cpu_ram_quotas={'scope':'unknown'}),
                     lambda x:x.update(cuda_layers_cached=None)):
            data=fixture(2);edit(data)
            e.validate_capacity(data,'exploratory-qualification',now=100,bounded_admission=True)
            with self.assertRaises(ValueError):e.validate_capacity(data,'exploratory',now=100)
        # A cache miss is resolved by Modal, not a local claim of incompatibility.
        data=fixture();data['cuda_layers_cached']=False
        e.validate_capacity(data,'exploratory-qualification',now=100,bounded_admission=True)

    def test_qualification_rejects_known_insufficiency_and_malformed_capacity(self):
        edits=(lambda x:x.update(gpu_limit=1),lambda x:x.update(container_limit=2),
               lambda x:x.update(gpu_limit=2,gpu_used=1),
               lambda x:x.update(gpu_limit=float('nan')),lambda x:x.update(gpu_limit=None,gpu_used=-1),
               lambda x:x['cpu_ram_quotas'].update(cpu_limit=32),
               lambda x:x['cpu_ram_quotas'].update(memory_mib_limit=395263),
               lambda x:x.update(cpu_ram_quotas={'scope':'per-container-only-confirmed','max_cpu':8}),
               lambda x:x.update(cpu_ram_quotas={'scope':'per-container-only-confirmed','max_memory_mib':196607}),
               lambda x:x['resources'].update(gpu='A100'),
               lambda x:x['funding'].update(reserve_released_usd=75))
        for edit in edits:
            data=copy.deepcopy(fixture());edit(data)
            with self.assertRaises(ValueError):e.validate_capacity(data,'exploratory-qualification',now=100,bounded_admission=True)

    def test_partial_unknown_limits_do_not_require_fabricated_numbers(self):
        data=fixture();data.update(gpu_limit=None,gpu_used=None,containers_used=None)
        data['cpu_ram_quotas'].update(cpu_limit=None,memory_mib_used=None)
        e.validate_capacity(data,'exploratory-qualification',now=100,bounded_admission=True)
        data['cpu_ram_quotas']['memory_mib_limit']=1
        with self.assertRaises(ValueError):e.validate_capacity(data,'exploratory-qualification',now=100,bounded_admission=True)
