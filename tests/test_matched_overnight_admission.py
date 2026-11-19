"""Synthetic admission tests: no provider calls, GPUs or capacity attestations."""
import copy
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
from experiment import matched_admission as admission, matched_contract as c, matched_exploratory as e, matched_supervisor as s
from scripts import phase2_study as cli
from test_matched_exploratory import plan, simulate


@contextmanager
def harness(phase='exploratory-qualification'):
    p=plan(phase); cloud=Mock();registry=Mock();store={}
    q,_=simulate(plan('exploratory-qualification'))
    store['index:'+s.key(q['plan'])]=q
    if phase=='exploratory-qualification':store.clear()
    def put(k,v,skip_if_exists=False):
        if skip_if_exists and k in store:return False
        store[k]=v;return True
    registry.get.side_effect=store.get;registry.put.side_effect=put
    cloud.Dict.from_name.return_value=registry
    workspace=NS(name='amulyasp',billing=NS(summary=Mock(return_value=NS(metered_cost=1.55,billed_cost=0))))
    cloud.Workspace.from_context.return_value.hydrate.return_value=workspace
    volume=NS(object_id=admission.VOLUME_ID,_get_metadata=lambda:NS(version=1),iterdir=lambda *a,**k:[NS(size=100)])
    cloud.Volume.from_name.return_value.hydrate.return_value=volume
    cloud.enable_output.side_effect=lambda:nullcontext()
    app=Mock();app.app.run.side_effect=lambda **kw:nullcontext()
    app.supervise.with_options.return_value.spawn.return_value=NS(object_id='fc-synthetic')
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);file=root/'plan.json';file.write_text(json.dumps(p))
        qfile=root/'qualification.json';qfile.write_text(json.dumps(q))
        args=['launch','--plan',str(file),'--bounded-admission','--owner-gross-headroom-usd','598.46',
              '--reviewed-usage-limit-usd','600','--authorize',s.key(p)+':'+c.digest(p),'--out',str(root/'submit.json')]
        if phase=='exploratory':args+=['--qualification',str(qfile)]
        with patch.dict(sys.modules,{'modal':cloud,'matched_app':app}),patch.dict(os.environ,MODAL_PROFILE='amulyasp'),patch.object(c,'source_identity',return_value=p['source']) as identity:
            yield NS(plan=p,cloud=cloud,workspace=workspace,volume=volume,registry=registry,store=store,app=app,
                     root=root,file=file,args=args,identity=identity)


class BoundedAdmissionTests(unittest.TestCase):
    def test_both_exploratory_phases_use_actual_admission_without_capacity_file(self):
        for phase in e.PHASES:
            with harness(phase) as h:
                self.assertEqual(cli.main(h.args),0)
                proof=json.loads((h.root/'submit.admission.json').read_text())
                self.assertIsNone(proof['gpu_limit']);self.assertIsNone(proof['cuda_layers_cached'])
                self.assertIn('UNVERIFIED',proof['capacity_status'])
                self.assertAlmostEqual(proof['billing']['remaining_gross_usage_usd'],598.45)
                h.app.app.run.assert_called_once_with(detach=True,environment_name='main')
                h.app.supervise.with_options.assert_called_once_with(timeout=e.bounds(phase)['supervisor_seconds'],cpu=(1,1),memory=(2048,2048))
                h.app.supervise.with_options.return_value.spawn.assert_called_once_with(h.plan)

    def test_forbidden_phases_and_unknown_owner_budget_fail_before_cloud(self):
        for phase in ('qualification','pilot','campaign'):
            p=plan();p=c.make_plan('formal',phase,1,3600,600,1000,10,p['data_manifest'],p['source'])
            with self.assertRaises(ValueError):admission.validate_request(p,598.46,600)
        for headroom,limit in ((None,600),(49,600),(float('nan'),600),(600,601),(601,600)):
            with self.assertRaises(ValueError):admission.validate_request(plan('exploratory-qualification'),headroom,limit)
        with harness() as h:
            p=c.make_plan('formal','qualification',1,3600,600,1000,10,h.plan['data_manifest'],h.plan['source'])
            h.file.write_text(json.dumps(p))
            with self.assertRaisesRegex(ValueError,'exploratory launch'):cli.main(h.args)
            h.cloud.Workspace.from_context.assert_not_called();h.app.app.run.assert_not_called()

    def test_no_authorization_dirty_mismatched_source_or_reserve_bypass(self):
        for mode in ('authorization','dirty','mismatch','reserve','resources'):
            with harness() as h:
                if mode=='authorization':h.args[h.args.index('--authorize')+1]=''
                elif mode=='dirty':h.identity.side_effect=ValueError('dirty source')
                elif mode=='mismatch':h.identity.return_value={}
                else:
                    bad=copy.deepcopy(h.plan)
                    if mode=='reserve':bad['funding']['reserve_released_usd']=75
                    else:bad['jobs'][0]['stages'][0]['wandb']['config']['resources']['memory_mib']=1
                    h.file.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):cli.main(h.args)
                h.cloud.Workspace.from_context.assert_not_called();h.registry.put.assert_not_called()

    def test_known_workspace_volume_and_billing_incompatibility_rejected(self):
        for mode in ('workspace','volume','version','billing'):
            with harness() as h:
                if mode=='workspace':h.workspace.name='other'
                elif mode=='volume':h.volume.object_id='vo-other'
                elif mode=='version':h.volume._get_metadata=lambda:NS(version=2)
                else:h.workspace.billing.summary.return_value.metered_cost=599
                with self.assertRaises(ValueError):cli.main(h.args)
                h.app.app.run.assert_not_called();h.registry.put.assert_not_called()
                self.assertEqual(json.loads((h.root/'submit.admission.json').read_text())['status'],'rejected')

    def test_unavailable_billing_uses_owner_confirmation_not_invented_provider_numbers(self):
        with harness() as h:
            h.workspace.billing.summary.side_effect=ConnectionError('unavailable')
            proof=admission.inspect(h.cloud,h.plan,598.46,600)
            self.assertEqual(proof['billing']['remaining_gross_usage_usd'],598.46)
            self.assertNotIn('live_metered_usd',proof['billing'])
            self.assertIn('unavailable',proof['billing']['live_summary'])

    def test_provider_failure_keeps_reservation_evidence_and_never_resubmits(self):
        for location in ('admission','spawn'):
            with harness() as h:
                if location=='admission':h.app.app.run.side_effect=RuntimeError('provider rejected resource')
                else:h.app.supervise.with_options.return_value.spawn.side_effect=ConnectionError('response lost')
                with self.assertRaises(Exception):cli.main(h.args)
                failure=json.loads((h.root/'submit.failure.json').read_text())
                self.assertEqual(failure['status'],'unknown');self.assertFalse(failure['automatic_retry'])
                self.assertIn('exploratory-allocation:'+h.plan['study_id']+':'+h.plan['phase'],h.store)
                retry=list(h.args);retry[retry.index('--out')+1]=str(h.root/'second.json')
                with self.assertRaisesRegex(ValueError,'already reserved'):cli.main(retry)
                h.app.app.run.assert_called_once()
                self.assertLessEqual(h.app.supervise.with_options.return_value.spawn.call_count,1)

    def test_duplicate_success_cannot_reserve_another_submission(self):
        with harness() as h:
            cli.main(h.args)
            retry=list(h.args);retry[retry.index('--out')+1]=str(h.root/'second.json')
            with self.assertRaisesRegex(ValueError,'already reserved'):cli.main(retry)
            h.app.supervise.with_options.return_value.spawn.assert_called_once()
