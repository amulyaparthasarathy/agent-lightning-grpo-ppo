"""Offline review of the supplied ignored runner; never execute a cloud command."""
from contextlib import contextmanager
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch
from test_matched_exploratory import plan,simulate

ROOT=Path(__file__).resolve().parents[1]
RUNNER=Path(os.environ.get('AGL_OVERNIGHT_RUNNER',ROOT/'.local/exploratory/phase2_overnight.py'))
if not RUNNER.exists():raise unittest.SkipTest('Place the reviewed downloadable runner in ignored .local/exploratory first')
spec=importlib.util.spec_from_file_location('overnight',RUNNER);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

@contextmanager
def directory():
    before=Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:yield Path(tmp)
        finally:os.chdir(before)


def completed(root,p):
    index,_=simulate(p);proofs=[]
    for job in p['jobs']:
        native={'step':job['steps'],'roles':{'actor':{'TEST_ONLY':True}},'plan_sha256':r.digest(p)}
        for stage,receipt in zip(job['stages'],index['jobs'][job['job_id']]['stages']):
            completion=receipt['result']['completion']
            if stage['name']!='training':completion['evaluation']={'TEST_ONLY':True}
            if stage['name']!='baseline':completion['checkpoint_sha256']=r.digest(native)
            dest=root/stage['output']
            r.write_json_new(dest/'completion.json',completion)
            r.write_json_new(dest/'receipt.json',receipt['result'])
            r.write_json_new(dest/'observability.json',{'wandb':stage['wandb']})
            proofs.append({'output':stage['output'],'available':True})
        r.write_json_new(root/job['output_root']/'training/checkpoints'/f"global_step_{job['steps']}"/'matched_identity.json',native)
    report=copy.deepcopy(index);report.update(evidence_complete=True,supervisor_actual='completed',volume_evidence=proofs,collection_errors=[])
    r.write_json_new(root/'index.json',index);r.write_json_new(root/'collection.json',report)
    return index,report


class RunnerTests(unittest.TestCase):
    def test_two_explicit_allocations_required_before_execution(self):
        with directory(),patch.object(r,'verify'),patch.object(r,'validate_offline'),patch.object(r,'check_launch_option'),patch.object(r,'execute') as execute:
            r.write_json_new(r.QUAL,plan('exploratory-qualification'));r.write_json_new(r.STUDY,plan())
            for flags in ([],['--approve-qualification-usd','50'],['--approve-qualification-usd','50','--approve-study-usd','176']):
                with self.assertRaises(r.Abort):r.main(['--execute',*flags])
            execute.assert_not_called()

    def test_dry_run_never_calls_execute(self):
        with directory(),patch.object(r,'verify'),patch.object(r,'validate_offline'),patch.object(r,'check_launch_option'),patch.object(r,'execute') as execute:
            r.write_json_new(r.QUAL,plan('exploratory-qualification'));r.write_json_new(r.STUDY,plan())
            self.assertEqual(r.main([]),0);execute.assert_not_called()

    def test_ambiguous_submission_intent_prevents_second_cli_launch(self):
        with directory(),patch.object(r,'invoke',side_effect=r.Abort('response lost')) as invoke:
            r.write_json_new(r.QUAL,plan('exploratory-qualification'))
            with self.assertRaises(r.Abort):r.ensure_submission(r.QUAL,r.QUAL_RECEIPT,9999999999)
            with self.assertRaisesRegex(r.Abort,'Prior submission intent'):r.ensure_submission(r.QUAL,r.QUAL_RECEIPT,9999999999)
            invoke.assert_called_once()

    def test_existing_verified_receipt_attaches_but_unknown_never_resubmits(self):
        with directory(),patch.object(r,'invoke') as invoke:
            p=plan('exploratory-qualification');r.write_json_new(r.QUAL,p)
            receipt={'status':'submitted','plan_sha256':r.digest(p),'call_id':'fc-test'}
            r.write_json_new(r.QUAL_RECEIPT,receipt)
            r.ensure_submission(r.QUAL,r.QUAL_RECEIPT,9999999999);invoke.assert_not_called()
            receipt['status']='unknown';r.QUAL_RECEIPT.write_text(json.dumps(receipt))
            with self.assertRaises(r.Abort):r.ensure_submission(r.QUAL,r.QUAL_RECEIPT,9999999999)
            invoke.assert_not_called()

    def test_deadline_persists_across_resume_and_cannot_extend(self):
        with directory(),patch.object(r.time,'time',return_value=100):
            end=r.execution_deadline(plan('exploratory-qualification'),plan(),12)
            with patch.object(r.time,'time',return_value=20000):
                self.assertEqual(r.execution_deadline(plan('exploratory-qualification'),plan(),12),end)
            with self.assertRaises(r.Abort):r.execution_deadline(plan('exploratory-qualification'),plan(),11)
            self.assertEqual(end,43300)

    def test_complete_evidence_requires_native_reload_and_durable_identities(self):
        with directory() as tmp:
            p=plan();root=tmp/'collected';completed(root,p)
            self.assertTrue(r.check_evidence(root,p))
            receipt=root/p['jobs'][0]['stages'][0]['output']/'receipt.json'
            receipt.write_text('{}')
            with self.assertRaisesRegex(r.Abort,'Collected receipt'):r.check_evidence(root,p)

    def test_partial_collection_never_repeats_downloads(self):
        with directory(),patch.object(r,'invoke') as invoke:
            p=plan();r.write_json_new(r.STUDY,p);r.STUDY_EVIDENCE.mkdir(parents=True)
            with self.assertRaises(r.Abort):r.collect(r.STUDY,r.STUDY_EVIDENCE,9999999999)
            invoke.assert_not_called()

    def test_phase_failure_stops_owned_and_never_starts_report(self):
        with directory(),patch.object(r,'ensure_submission') as submit,patch.object(r,'wait_complete',side_effect=r.Abort('provider failed')),patch.object(r,'owned_stop') as stop,patch.object(r,'collect') as collect:
            args=Mock(max_hours=12,poll_seconds=60,owner_gross_headroom_usd=598.46)
            with self.assertRaises(r.Abort):r.execute(plan('exploratory-qualification'),plan(),args)
            self.assertEqual(submit.call_count,1);self.assertEqual(submit.call_args.args[0],r.QUAL)
            self.assertEqual(stop.call_args.args[0],r.QUAL);collect.assert_not_called()

    def test_success_orders_qualification_collection_before_single_report_submission(self):
        with directory(),patch.object(r,'ensure_submission') as submit,patch.object(r,'wait_complete') as wait,patch.object(r,'report_summary',return_value={'ok':True}):
            args=Mock(max_hours=12,poll_seconds=60,owner_gross_headroom_usd=598.46)
            def collect(path,out,deadline):
                p=plan('exploratory-qualification') if path==r.QUAL else plan()
                completed(out,p)
            with patch.object(r,'collect',side_effect=collect):
                self.assertEqual(r.execute(plan('exploratory-qualification'),plan(),args),{'ok':True})
            self.assertEqual([call.args[0] for call in submit.call_args_list],[r.QUAL,r.STUDY])
            self.assertEqual(wait.call_count,2)

    def test_local_timeout_preserves_partial_output(self):
        with directory(),patch.object(r.subprocess,'Popen') as popen,patch.object(r.os,'killpg') as kill:
            proc=popen.return_value;proc.pid=123
            proc.communicate.side_effect=[subprocess.TimeoutExpired('synthetic',1),('partial stdout','partial stderr')]
            with self.assertRaises(r.Abort):r.invoke(['TEST_ONLY'],'launch-test',r.time.time()+10,timeout=3)
            self.assertIn('partial stdout',next(r.LOGROOT.glob('launch-test-*.txt')).read_text())
            kill.assert_called_once()

    def test_pending_supervisor_timeout_is_not_a_failed_admission(self):
        with directory(),patch.object(r,'invoke') as invoke,patch.object(r.time,'sleep'):
            r.write_json_new(r.QUAL,plan('exploratory-qualification'))
            statuses=[{'status':'started','supervisor_actual':'unavailable_or_failed: TimeoutError'},
                      {'status':'completed','evidence_complete':True,'supervisor_actual':'completed'}]
            invoke.side_effect=[subprocess.CompletedProcess([],2,json.dumps(row),'') for row in statuses]
            result=r.wait_complete(r.QUAL,r.time.time()+100,10)
            self.assertTrue(result['evidence_complete']);self.assertEqual(invoke.call_count,2)
