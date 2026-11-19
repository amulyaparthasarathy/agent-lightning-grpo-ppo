"""Offline transport, recovery, budget and conversion planning tests. No tensors."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import preservation_transfer as transfer
from scripts import preservation_workflow as workflow
from scripts.convert_fsdp1 import shard_plan, validate_tensor_schema
from scripts.preservation_preflight import capacity

REPO='amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup'


class FakeHub:
    def __init__(self):
        self.revision='0'*40
        self.files={}
        self.history={self.revision:{}}
    def repo_info(self,repo_id,**kw):
        return SimpleNamespace(id=repo_id,private=True,sha=self.revision)
    def list_repo_files(self,repo_id,revision=None):return sorted(self.history[revision or self.revision])
    def create_commit(self,repo_id,operations,parent_commit,**kw):
        if parent_commit!=self.revision:raise RuntimeError('optimistic conflict')
        for operation in operations:
            value=operation.path_or_fileobj
            self.files[operation.path_in_repo]=Path(value).read_bytes() if isinstance(value,str) else value
        self.revision=f'{int(self.revision,16)+1:040x}'
        self.history[self.revision]=dict(self.files)
        return SimpleNamespace(oid=self.revision)
    def auth_check(self,*args,**kwargs):pass
    def stream(self,api,repo,revision,path):
        data=self.history[revision][path]
        for i in range(0,len(data),3):yield data[i:i+3]


def plan():
    p={'remote_execution_approved':'YES','owner':'amulyaparthasarathy','limits':workflow.LIMITS,'execution_commit':'e'*40,
       'rights_review':{'fixture':True},'capacity_evidence':{'fixture':True},
       'source_volume_id':'source','validation_volume_id':'validation','working_volume_id':'work',
       'initial_accounting':dict.fromkeys(workflow.LIMITS,0),'phases':[]}
    for name in workflow.ORDER:
        gpu=name in workflow.GPU_PHASES
        p['phases'].append({'name':name,'kind':'gpu' if gpu else 'cpu','cpu':16 if gpu else 8,'memory_gib':128,'timeout_seconds':10,
                           'startup_reserve_seconds':2,'reservation':dict.fromkeys(workflow.LIMITS,0)})
        p['phases'][-1]['reservation']['gpu_seconds' if gpu else 'cpu_seconds']=12
    return p


class TransportTests(unittest.TestCase):
    def test_recovery_before_snapshot_claim_creates_once_then_refuses_completed_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'record.json').write_bytes(b'{}')
            manifests={'evidence':{'files':[{'path':'record.json','bytes':2,'sha256':hashlib.sha256(b'{}').hexdigest()}]}}
            hub=FakeHub()
            result=transfer.upload_snapshot(hub,REPO,'snapshot',{'evidence':root},manifests,hub.revision,lambda r:None,resume=True)
            self.assertEqual(result['bytes'],2)
            with self.assertRaisesRegex(ValueError,'only incomplete'):
                transfer.upload_snapshot(hub,REPO,'snapshot',{'evidence':root},manifests,hub.revision,lambda r:None,resume=True)

    def test_metadata_semantics_accepts_only_lossless_rope_and_verified_tokenizer_override(self):
        from scripts.convert_fsdp1 import validate_metadata_semantics
        ids={'bos_token_id':128000,'eos_token_id':128009,'pad_token_id':128009}
        base={'bos_token_id':128000,'eos_token_id':[128001,128008,128009],
              'rope_scaling':{'rope_type':'llama3','factor':32.0},'rope_theta':500000.0,
              'hidden_size':3072,'tie_word_embeddings':True,'transformers_version':'old'}
        native={'hidden_size':3072,'tie_word_embeddings':True,**ids,
                'rope_parameters':{'rope_type':'llama3','factor':32.0,'rope_theta':500000.0},'transformers_version':'new'}
        generation={'bos_token_id':128000,'eos_token_id':[128001,128008,128009],'temperature':0.6,'top_p':0.9}
        saved=copy.deepcopy((native,base,generation))
        report=validate_metadata_semantics(native,base,dict(generation,transformers_version='new'),generation,ids)
        self.assertTrue(report['matched']);self.assertEqual((native,base,generation),saved)
        for key,value in [('hidden_size',2048),('tie_word_embeddings',False),('eos_token_id',128001),
                          ('pad_token_id',0),('rope_parameters',{'rope_theta':123})]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_metadata_semantics(dict(native,**{key:value}),base,generation,generation,ids)
        with self.assertRaisesRegex(ValueError,'generation semantics'):
            validate_metadata_semantics(native,base,dict(generation,eos_token_id=[128009]),generation,ids)
        with self.assertRaisesRegex(ValueError,'tokenizer IDs'):
            validate_metadata_semantics(native,base,generation,generation,dict(ids,pad_token_id=0))
        with self.assertRaisesRegex(ValueError,'ambiguous RoPE'):
            validate_metadata_semantics(native,dict(base,rope_parameters={}),generation,generation,ids)

    def test_parallel_reads_are_bounded_and_keep_manifest_order(self):
        from threading import Barrier, Lock
        barrier=Barrier(transfer.READERS); lock=Lock(); active=0; peak=0
        def read(index, stopped):
            nonlocal active, peak
            with lock:
                active+=1;peak=max(peak,active)
            if index<transfer.READERS:barrier.wait(timeout=5)
            data=str(index).encode()
            row={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
            result=transfer.download_file([data],row,stopped=stopped)
            with lock:active-=1
            return index,result[1]
        result=transfer.parallel_read(range(19),read)
        self.assertEqual([r[0] for r in result],list(range(19)))
        self.assertEqual(peak,transfer.READERS)
        self.assertEqual([r[1] for r in result],[hashlib.sha256(str(i).encode()).hexdigest() for i in range(19)])

    def test_parallel_failure_cancels_stream_and_closes_response(self):
        from threading import Barrier
        barrier=Barrier(2); closed=[]; consumed=[]
        def read(index, stopped):
            barrier.wait(timeout=5)
            if index==0:raise transfer.GateFailure('authentic fixture mismatch')
            def chunks():
                try:
                    self.assertTrue(stopped.wait(timeout=5))
                    for i in range(100):
                        consumed.append(i);yield b'x'
                finally:closed.append(True)
            return transfer.download_file(chunks(),{'bytes':100,'sha256':hashlib.sha256(b'x'*100).hexdigest()},stopped=stopped)
        with self.assertRaisesRegex(ValueError,'authentic fixture mismatch'):
            transfer.parallel_read(range(2),read)
        # Cancellation may happen before opening the stream or on its first chunk.
        self.assertLessEqual(len(consumed),1)
        if consumed:self.assertEqual(closed,[True])

    def test_parallel_restore_checks_every_file_and_duplicate_before_read(self):
        hub=FakeHub();data=b'fixture';rows=[]
        for i in range(12):
            path=f'data/{i:02}.bin'
            hub.files['snapshot/source/'+path]=data
            rows.append({'namespace':'source','path':path,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        hub.history[hub.revision]=dict(hub.files)
        target={'snapshot_prefix':'snapshot','repo_id':REPO,'revision':hub.revision}
        with tempfile.TemporaryDirectory() as tmp, patch.object(transfer,'byte_stream',hub.stream):
            out=Path(tmp)/'restored';receipt=transfer.restore(hub,target,rows,out)
            self.assertEqual([r['path'] for r in receipt['files']],[r['path'] for r in rows])
            self.assertTrue(all((out/r['path']).read_bytes()==data for r in rows))
            with patch.object(transfer,'byte_stream') as read:
                with self.assertRaisesRegex(ValueError,'duplicate'):
                    transfer.restore(hub,target,[rows[0],rows[0]],Path(tmp)/'duplicate')
                read.assert_not_called()

    def test_small_files_use_bounded_commits_and_detect_batch_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i in range(53):(root/f'{i:03}.json').write_bytes(b'{}')
            rows=[{'path':p.name,'bytes':2,'sha256':hashlib.sha256(b'{}').hexdigest()} for p in sorted(root.iterdir())]
            hub=FakeHub()
            transfer.upload_snapshot(hub,REPO,'snapshot',{'n':root},{'n':{'files':rows}},hub.revision,lambda r:None)
            self.assertEqual(int(hub.revision,16),5) # claim, two bounded batches, manifest, terminal
            original=hub.create_commit
            def mutate(**kwargs):
                value=original(**kwargs)
                if any(o.path_in_repo.endswith('/000.json') for o in kwargs['operations']):
                    (root/'000.json').write_bytes(b'changed')
                return value
            hub.create_commit=mutate
            with self.assertRaisesRegex(ValueError,'source changed during upload'):
                transfer.upload_snapshot(hub,REPO,'another',{'n':root},{'n':{'files':rows}},hub.revision,lambda r:None)
            self.assertNotIn('another/transfer.json',hub.files)

    def test_stream_mismatch_and_partial_files_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'restored.pt';data=b'opaque bytes'
            row={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
            with self.assertRaisesRegex(ValueError,'mismatch'):
                transfer.download_file([b'broken'],row,out)
            self.assertEqual(out.read_bytes(),b'broken')
            with self.assertRaises(FileExistsError):transfer.download_file([data],row,out)
            size,sha,_=transfer.download_file([data[:2],data[2:]],row)
            self.assertEqual((size,sha),(row['bytes'],row['sha256']))

    def test_exclusive_upload_validated_resume_and_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'a.pt').write_bytes(b'opaque state'); (root/'b.json').write_bytes(b'{}')
            rows=[{'path':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                  for p in sorted(root.iterdir())]
            manifests={'experiment':{'files':rows}}
            hub=FakeHub();progress=[]
            def stopped(value):
                progress.append(value)
                if value['uploaded_bytes']>0: raise RuntimeError('disconnect fixture')
            with patch.object(transfer,'byte_stream',hub.stream):
                with self.assertRaises(RuntimeError):
                    transfer.upload_snapshot(hub,REPO,'snapshots/one',{'experiment':root},manifests,hub.revision,stopped)
                with self.assertRaisesRegex(ValueError,'already exists'):
                    transfer.upload_snapshot(hub,REPO,'snapshots/one',{'experiment':root},manifests,hub.revision,lambda r:None)
                before=hub.revision;member='snapshots/one/experiment/a.pt'
                hub.history[before][member]=b'corrupt'
                with self.assertRaisesRegex(ValueError,'byte mismatch'):
                    transfer.upload_snapshot(hub,REPO,'snapshots/one',{'experiment':root},manifests,before,lambda r:None,resume=True)
                self.assertEqual(hub.revision,before)
                self.assertNotIn('snapshots/one/transfer.json',hub.files)
                hub.history[before][member]=hub.files[member]
                saved=transfer.upload_snapshot(hub,REPO,'snapshots/one',{'experiment':root},manifests,hub.revision,lambda r:None,resume=True)
                self.assertEqual(saved['bytes'],sum(r['bytes'] for r in rows))
                self.assertIn('snapshots/one/transfer.json',hub.files)
                with self.assertRaisesRegex(ValueError,'incomplete'):
                    transfer.upload_snapshot(hub,REPO,'snapshots/one',{'experiment':root},manifests,hub.revision,lambda r:None,resume=True)
                (root/'a.pt').write_bytes(b'changed source')
                with self.assertRaisesRegex(ValueError,'source changed'):
                    transfer.upload_snapshot(hub,REPO,'snapshots/two',{'experiment':root},manifests,hub.revision,lambda r:None)

    def test_restore_rejects_traversal_before_reading_and_refuses_existing_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(transfer,'byte_stream') as read:
                with self.assertRaises(ValueError):
                    transfer.restore(None,{},[{'path':'../escape'}],Path(tmp)/'fresh')
                read.assert_not_called()
                with self.assertRaises(FileExistsError):transfer.restore(None,{},[],Path(tmp))

    def test_fresh_verifier_reads_bytes_and_rejects_extra_or_changed_files(self):
        from scripts import preservation_manifest as pm
        hub=FakeHub();data=b'{"case":"fixture"}'
        row={'path':'runs/new/trace.jsonl','group':'evidence','bytes':len(data),
             'sha256':hashlib.sha256(data).hexdigest(),'observation_class':'synthetic_observation',
             'observation_origin':'fixture-source','hash_basis':'bytes_read'}
        selection={'selections':[{'group':'evidence','path':'runs'}]}
        sources={'experiment':pm.manifest([row],{},'2026-11-14T00:00:00+00:00',
                   selections=selection['selections'],missing_selections=[])}
        prefix='snapshots/fixture'
        additions=[(prefix+'/experiment/'+row['path'],data)]+[(prefix+'/'+n,b'{}') for n in
                    ('attempt.json','source-manifests.json','transfer.json')]
        revision=transfer.commit_files(hub,REPO,hub.revision,additions,'fixture')
        target={'repo_id':REPO,'revision':revision,'snapshot_prefix':prefix}
        with patch.object(transfer,'byte_stream',hub.stream):
            copies,_=transfer.verify_snapshot(hub,target,sources,{'experiment':selection},None,[],
                                               'fresh-fixture',require_bundle=False)
            self.assertEqual(copies['experiment']['files'][0]['sha256'],row['sha256'])
            self.assertNotEqual(copies['experiment']['files'][0]['observation_origin'],row['observation_origin'])
            hub.history[revision][prefix+'/extra']=b'extra'
            with self.assertRaisesRegex(ValueError,'missing/extra'):
                transfer.verify_snapshot(hub,target,sources,{'experiment':selection},None,[],'fresh',require_bundle=False)
            del hub.history[revision][prefix+'/extra']
            hub.history[revision][prefix+'/experiment/'+row['path']]=b'changed'
            with self.assertRaisesRegex(ValueError,'byte mismatch'):
                transfer.verify_snapshot(hub,target,sources,{'experiment':selection},None,[],'fresh',require_bundle=False)

    def test_git_bundle_recovery_checks_actual_commits(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);repo=root/'repo';repo.mkdir()
            subprocess.run(['git','init',str(repo)],check=True,capture_output=True)
            (repo/'a').write_text('fixture')
            subprocess.run(['git','-C',str(repo),'add','a'],check=True,capture_output=True)
            subprocess.run(['git','-C',str(repo),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid',
                            'commit','-m','fixture'],check=True,capture_output=True)
            commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
            bundle=root/'history.bundle'
            subprocess.run(['git','-C',str(repo),'bundle','create',str(bundle),'--all'],check=True,capture_output=True)
            self.assertEqual(transfer.verify_bundle(bundle,[commit])['restored_commits'],[commit])
            with self.assertRaises(subprocess.CalledProcessError):transfer.verify_bundle(bundle,['f'*40])


class ReleaseTests(unittest.TestCase):
    def test_only_validated_inference_files_and_small_summaries_publish_then_readback(self):
        import test_export_eval as fixture
        from experiment.export_artifact import sha256
        from scripts import preservation_release as release
        from scripts.backup_qualification import digest
        f=fixture.ExportEvaluationTests();f.setUp();self.addCleanup(f.doCleanups)
        for name,value in {'LICENSE':'fixture license','Notice':'fixture notice','README.md':'# Llama fixture\nBuilt with Llama'}.items():
            path=f.model/name;path.write_text(value)
            f.manifest['files'][name]={'bytes':path.stat().st_size,'sha256':sha256(path)}
        f.manifest['tensor_validation']['export_files_sha256']=digest(f.manifest['files']);f.save()
        hub=FakeHub();repo=REPO.replace('-Backup','-GRPO')
        evaluation={'baseline':{'observed_reward':{'count':64,'sum':35}},'post':{'observed_reward':{'count':64,'sum':35}},
                    'comparison':{'available':True,'paired_cases':64},'conversion_defect_resolved':True,
                    'cases':[{'raw_answer':'PRIVATE FIXTURE MUST NOT BE RELEASED'}]}
        target=release.publish(hub,repo,f.model,f.data_manifest,f.harness,'main',hub.revision,evaluation)
        self.assertNotIn('PRIVATE FIXTURE',hub.files['export_manifest.json'].decode())
        self.assertEqual(hub.files['model.safetensors'],(f.model/'model.safetensors').read_bytes())
        with patch.object(transfer,'byte_stream',hub.stream),patch.object(release,'byte_stream',hub.stream):
            report=release.retrieve(hub,target,f.root/'fresh-retrieval',f.data_manifest,f.harness)
            self.assertEqual(report['status'],'retrieved_bytes_verified')
        with self.assertRaisesRegex(ValueError,'already exists'):
            release.publish(hub,repo,f.model,f.data_manifest,f.harness,'main',hub.revision,evaluation)



class DurableTests(unittest.TestCase):
    def test_small_source_profile_preserves_tensor_gpu_and_historical_limits(self):
        p=plan();workflow.validate_plan(p)  # Historical requests stay interpretable.
        for phase in p['phases']:
            if phase['name'] in workflow.SOURCE_STREAM_PHASES:
                phase['cpu'],phase['memory_gib']=workflow.SOURCE_STREAM_RESOURCES
        workflow.validate_plan(p)
        source=p['phases'][0]
        usage=workflow.observed_usage(source,10)
        self.assertAlmostEqual(usage['cost_usd'],12*(2*0.0000131+4*0.00000222)*1.2)
        tensor=next(v for v in p['phases'] if v['name']=='convert')
        tensor['cpu'],tensor['memory_gib']=workflow.SOURCE_STREAM_RESOURCES
        with self.assertRaisesRegex(ValueError,'resource cap'):workflow.validate_plan(p)

    def test_observed_cost_keeps_startup_egress_and_does_not_clamp_overruns(self):
        phase=plan()['phases'][0];phase['reservation']['egress_bytes']=1024**3
        short=workflow.observed_usage(phase,1);long=workflow.observed_usage(phase,100)
        self.assertEqual(short['cpu_seconds'],3)
        self.assertGreater(long['cost_usd'],short['cost_usd'])
        self.assertGreater(long['cpu_seconds'],phase['reservation']['cpu_seconds'])
        self.assertIsNone(short['hf_bytes'])

    def test_terminal_fallback_survives_remote_receipt_failure(self):
        saved=[]
        def unavailable(record):raise transfer.GateFailure('HTTP transient status 429; fixture')
        record={'success':False,'diagnostic':'fixture transport failure'}
        with self.assertRaisesRegex(ValueError,'429'):
            workflow.persist_before_publish(record,lambda r:saved.append(copy.deepcopy(r)),unavailable)
        self.assertEqual(saved,[record])

    def test_snapshot_resume_reserves_transfer_again_but_not_same_retained_files(self):
        revision,ledger=self.store.load()
        phase=next(p for p in ledger['plan']['phases'] if p['name']=='backup')
        phase['reservation']['hf_bytes']=160*1024**3
        phase['reservation']['egress_bytes']=170*1024**3
        from scripts.backup_qualification import digest
        ledger['plan_sha256']=digest(ledger['plan']);self.store.save(revision,ledger)
        self.store.claim();self.store.started('inventory','fc-inventory')
        self.store.terminal('inventory','fc-inventory',{},dict.fromkeys(workflow.LIMITS,None),True)
        self.store.claim();self.store.started('backup','fc-backup')
        self.store.terminal('backup','fc-backup',{},dict.fromkeys(workflow.LIMITS,None),False)
        recovered,ledger=self.store.recover('backup','fixture diagnosed resume',True)
        self.assertEqual(ledger['status'],'running')
        self.assertEqual(self.store.load()[1]['status'],'running')
        self.assertEqual(ledger['phases']['backup']['status'],'launching')
        self.assertEqual(ledger['accounting']['hf_bytes'],160*1024**3)
        self.assertEqual(ledger['accounting']['egress_bytes'],340*1024**3)
        self.assertEqual(recovered['reservation']['hf_bytes'],160*1024**3)

    def test_difference_review_cannot_waive_changed_evidence_or_tensor_failure(self):
        from scripts.preservation_release import reviewed_comparison
        from scripts.backup_qualification import digest
        evaluation={'comparison':{'post_wins':1},'raw_answer_differences':2}
        converted={'manifest_sha256':'a'*64,'manifest':{'tensor_validation':{'all_tensors_equal':True}}}
        review={'comparison_sha256':digest(evaluation),'export_manifest_sha256':'a'*64,
                'disposition':'no_conversion_or_loading_defect','reviewer':'fixture',
                'reviewed_at':'synthetic','analysis':'Fixture investigation',
                'evidence_locations':['fixture://tensor-receipt']}
        self.assertTrue(reviewed_comparison(evaluation,converted,review)['conversion_defect_resolved'])
        with self.assertRaisesRegex(ValueError,'different evidence'):
            reviewed_comparison(dict(evaluation,raw_answer_differences=3),converted,review)
        converted['manifest']['tensor_validation']['all_tensors_equal']=False
        with self.assertRaisesRegex(ValueError,'cannot waive'):
            reviewed_comparison(evaluation,converted,review)

    def setUp(self):
        self.hub=FakeHub()
        self.patch=patch.object(transfer,'byte_stream',self.hub.stream);self.patch.start();self.addCleanup(self.patch.stop)
        self.store=workflow.HubLedger(self.hub,REPO,'workflows/fixture/ledger.json')
        self.store.initialize(plan())

    def test_lost_ack_duplicate_claim_and_completed_resume(self):
        with self.assertRaisesRegex(ValueError,'project workflow'):
            workflow.HubLedger(self.hub,REPO,'workflows/duplicate/ledger.json').initialize(plan())
        phase,_=self.store.claim()
        with self.assertRaisesRegex(ValueError,'already claimed'):self.store.claim()
        self.store.started('inventory','fc-fixture')
        self.assertEqual(self.store.load()[1]['phases']['inventory']['execution_commit'],'e'*40)
        with self.assertRaisesRegex(ValueError,'duplicate'):self.store.started('inventory','fc-other')
        self.store.terminal('inventory','fc-fixture',{'fixture':True},dict.fromkeys(workflow.LIMITS,None),True)
        next_phase,_=self.store.claim()
        self.assertEqual(next_phase['name'],'backup')

    def test_failure_accounting_and_diagnosed_retry_bound(self):
        self.store.claim();self.store.started('inventory','fc-0')
        self.store.terminal('inventory','fc-0',{'error':'fixture'},dict.fromkeys(workflow.LIMITS,None),False)
        with self.assertRaisesRegex(ValueError,'already claimed'):self.store.claim()
        with self.assertRaisesRegex(ValueError,'active'):self.store.recover('inventory','diagnosis',False)
        for i in (1,2):
            self.store.recover('inventory','diagnosis '+str(i),True)
            self.store.started('inventory','fc-'+str(i))
            self.store.terminal('inventory','fc-'+str(i),{},dict.fromkeys(workflow.LIMITS,None),False)
        with self.assertRaisesRegex(ValueError,'retry limit'):self.store.recover('inventory','third',True)
        _,ledger=self.store.load();self.assertEqual(ledger['accounting']['cpu_seconds'],36)

    def test_native_bootstrap_recovery_requires_bound_remote_absence_evidence(self):
        from scripts.backup_qualification import digest
        revision,ledger=self.store.load()
        failure_path='receipts/native-failed.json';diagnostic_path='receipts/startup.json'
        revision=transfer.put_record(self.hub,REPO,revision,failure_path,{'error_type':'ChildDidNotReturn'})
        failure_ref={'revision':revision,'receipt_path':failure_path}
        report={'kind':'import_only_diagnostic','probe_sha256':'a'*64,'function_call_id':'fc-probe',
                'native_run_exists':False,'native_submission_exists':False,
                'client':{'transport_configuration':{'configured':True}},
                'training':{'imports':{'scripts.preservation_worker':{'module_name':'scripts.preservation_worker'}}}}
        revision=transfer.put_record(self.hub,REPO,revision,diagnostic_path,report)
        ledger['phases']['native_eval']={'status':'failed','function_call_id':'fc-native','result':failure_ref}
        ledger['diagnostics']={'native-eval-startup-0':{'status':'observed','prior_native_call':'fc-native',
            'function_call_id':'fc-probe','probe_sha256':'a'*64,'report_sha256':digest(report),
            'result':{'revision':revision,'receipt_path':diagnostic_path}}}
        original=copy.deepcopy(ledger)
        self.store.save(revision,ledger)
        reason='client-orchestration-before-native-launch'
        with self.assertRaisesRegex(ValueError,'active'):self.store.recover('native_eval',reason,False)
        for field in ('native_run_exists','native_submission_exists'):
            revision,_=self.store.load();changed=dict(report,**{field:True})
            revision=transfer.put_record(self.hub,REPO,revision,diagnostic_path,changed)
            ledger=copy.deepcopy(original);claim=ledger['diagnostics']['native-eval-startup-0']
            claim['report_sha256']=digest(changed);claim['result']['revision']=revision
            self.store.save(revision,ledger)
            with self.assertRaisesRegex(ValueError,'output exists'):self.store.recover('native_eval',reason,True)
        revision,_=self.store.load();ledger=copy.deepcopy(original)
        ledger['diagnostics']['native-eval-startup-0']['prior_native_call']='fc-other'
        self.store.save(revision,ledger)
        with self.assertRaisesRegex(ValueError,'another call'):self.store.recover('native_eval',reason,True)
        revision,_=self.store.load();self.store.save(revision,original)
        _,recovered=self.store.recover('native_eval',reason,True)
        self.assertEqual(recovered['phases']['native_eval']['attempt'],1)
        self.assertEqual(recovered['phases']['native_eval']['recoveries'][0]['prior']['result'],failure_ref)
        self.store.started('native_eval','fc-next')
        self.store.terminal('native_eval','fc-next',{},dict.fromkeys(workflow.LIMITS,None),False)
        with self.assertRaisesRegex(ValueError,'scientific-run'):self.store.recover('native_eval','different',True)

    def test_budget_and_capacity_fail_closed(self):
        ledger=workflow.new_ledger(plan());ledger['accounting']['cost_usd']=100
        phase=copy.deepcopy(plan()['phases'][0]);phase['reservation']['cost_usd']=0.01
        with self.assertRaisesRegex(ValueError,'cost_usd'):workflow.reserve(ledger,phase)
        p=plan();p['phases'][0]['cpu']=9
        with self.assertRaises(ValueError):workflow.validate_plan(p)
        with self.assertRaises(ValueError):capacity({'capacity_evidence':{},'projected_hf_bytes':100,'projected_modal_bytes':100})
        p={'capacity_evidence':{'observed_at':'fixture','source':'synthetic dashboard',
                               'hf_available_bytes':299*1024**3,'modal_available_bytes':1000},
           'projected_hf_bytes':100,'projected_modal_bytes':100,'projected_monthly_usd':9}
        with self.assertRaisesRegex(ValueError,'300 GiB'):capacity(p)
        p['capacity_evidence']['hf_available_bytes']=300*1024**3
        capacity(p)

    def test_rounded_owner_capacity_is_not_a_provider_measurement(self):
        evidence={'observed_at':'fixture','source':'owner-attested dashboard','hf_available_bytes':None,
                  'modal_available_bytes':200*1024**3,'hf_owner_attestation':{
                  'owner':'amulyaparthasarathy','subscription':'PRO','private_used_display':'0 TB',
                  'private_total_display':'1 TB','rounded':True,'approved_for_workflow':True}}
        p={'capacity_evidence':evidence,'projected_hf_bytes':270*1024**3,
           'projected_modal_bytes':180*1024**3,'projected_monthly_usd':9}
        capacity(p)
        self.assertIsNone(evidence['hf_available_bytes'])
        p['projected_hf_bytes']=301*1024**3
        with self.assertRaisesRegex(ValueError,'allocation insufficient'):capacity(p)
        p['projected_hf_bytes']=270*1024**3
        evidence['hf_available_bytes']=10**12
        with self.assertRaisesRegex(ValueError,'masquerade'):capacity(p)
        evidence['hf_available_bytes']=None
        evidence['hf_owner_attestation']['private_used_display']='0.9 TB'
        with self.assertRaisesRegex(ValueError,'attestation'):capacity(p)


class DispatchTests(unittest.TestCase):
    def test_startup_probe_sanitizes_errors_and_survives_shadowed_scripts(self):
        import os,sys
        from scripts import preservation_startup as probe
        secret='fixture-credential-must-not-appear'
        self.assertEqual(probe.failure(ImportError(secret,name='huggingface_hub')),
                         {'error_type':'ImportError','module_name':'huggingface_hub'})
        self.assertNotIn(secret,json.dumps(probe.failure(ImportError(secret,name=secret))))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'scripts').mkdir();(root/'scripts/__init__.py').write_text('# unrelated package')
            with patch.dict(os.environ,PYTHONPATH=tmp,HF_HUB_DISABLE_XET='1'):
                previous=Path.cwd()
                try:
                    os.chdir(tmp)
                    result=probe.probe(sys.executable)
                finally:
                    os.chdir(previous)
        self.assertEqual(result['imports']['scripts.preservation_worker']['error_type'],'ModuleNotFoundError')
        self.assertEqual(result['imports']['scripts.preservation_worker']['module_name'],'scripts.preservation_worker')
        self.assertNotIn('environ',result)

    def test_explicit_packages_win_over_installed_scripts_and_tensor_errors_are_sanitized(self):
        import os,sys,subprocess
        from scripts import preservation_tensor
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);shadow=root/'installed/scripts';shadow.mkdir(parents=True)
            (shadow/'__init__.py').write_text("raise RuntimeError('unrelated scripts package')")
            request=root/'input.json';output=root/'output.json'
            request.write_text(json.dumps({'action':'fixture-invalid-action-secret','arguments':{}}))
            env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1])+os.pathsep+str(shadow.parent))
            child=subprocess.run([sys.executable,'-m','scripts.preservation_tensor','--input',str(request),
                                  '--output',str(output)],cwd=tmp,env=env,capture_output=True,text=True)
            self.assertEqual(child.returncode,1)
            self.assertEqual(json.loads(output.read_text()),{'success':False,'error_type':'ValueError'})
            self.assertNotIn('secret',child.stderr+output.read_text())

    def test_tensor_subprocess_has_no_publisher_credential_and_requires_success_receipt(self):
        import os
        from scripts import preservation_tensor as tensor
        calls=[]
        def child(argv,**kw):
            calls.append((argv,kw))
            Path(argv[argv.index('--output')+1]).write_text(json.dumps({'success':True,'result':{'fixture':True}}))
            return SimpleNamespace(returncode=0)
        with patch.dict(os.environ,HF_TOKEN='fixture-secret',WANDB_API_KEY='fixture-secret'), \
             patch.object(tensor.subprocess,'run',side_effect=child):
            self.assertEqual(tensor.invoke('/qualified/python','clean_load',{'root':'fixture'},60),{'fixture':True})
        argv,kw=calls[0]
        self.assertEqual(argv[0],'/qualified/python')
        self.assertEqual(kw['timeout'],60)
        self.assertEqual(kw['env']['HF_HUB_OFFLINE'],'1')
        self.assertNotIn('HF_TOKEN',kw['env']);self.assertNotIn('WANDB_API_KEY',kw['env'])
        with patch.object(tensor.subprocess,'run',return_value=SimpleNamespace(returncode=1)):
            with self.assertRaisesRegex(ValueError,'before result'):tensor.invoke('/qualified/python','convert',{},60)

    def test_worker_bootstrap_failure_is_a_sanitized_terminal_receipt(self):
        import sys
        from scripts import preservation_worker as worker
        from scripts import preservation_http
        with tempfile.TemporaryDirectory() as tmp:
            request=Path(tmp)/'input.json';output=Path(tmp)/'output.json'
            request.write_text(json.dumps({'phase':'native_eval','plan':{'phases':[
                {'name':'native_eval','reservation':{'egress_bytes':128*1024**2}}]}}))
            with patch.object(sys,'argv',['worker','--input',str(request),'--output',str(output)]), \
                 patch.object(preservation_http,'configure',side_effect=RuntimeError('fixture-secret')):
                self.assertEqual(worker.main(),1)
            result=json.loads(output.read_text())
            self.assertEqual(result['result']['stage'],'bootstrap')
            self.assertNotIn('fixture-secret',output.read_text())

    def test_conversion_diagnostic_reads_only_small_metadata_and_source(self):
        from scripts.preservation_startup import inspect_conversion
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);actor=root/'runs/train-01/checkpoints/global_step_20/actor'
            (actor/'huggingface').mkdir(parents=True);base=root/'models/llama3b';base.mkdir(parents=True)
            paths=[actor/'fsdp_config.json',actor/'huggingface/config.json',actor/'huggingface/generation_config.json']
            paths += [base/n for n in ('config.json','generation_config.json','tokenizer_config.json','model_manifest.json')]
            for path in paths:path.write_text('{}')
            package=root/'verl';(package/'model_merger').mkdir(parents=True)
            for name in ('fsdp_model_merger.py','base_model_merger.py'):(package/'model_merger'/name).write_text('# fixture')
            (package/'workers/config').mkdir(parents=True)
            (package/'workers/config/model.py').write_text('eos_token_id = tokenizer.eos_token_id\n')
            # Invalid tensor contents must never be opened by this metadata probe.
            (actor/'model_world_size_1_rank_0.pt').write_text('not a tensor')
            with patch('importlib.metadata.version',return_value='0.8.0'):
                report=inspect_conversion(root,package)
                self.assertEqual(report['files']['fsdp_config']['bytes'],2)
                self.assertFalse(report['output_exists']);self.assertEqual(len(report['installed_merger']),2)
                self.assertIn('1',report['installed_worker']['token_config_lines'])
                (base/'config.json').write_bytes(b'x'*(1024**2+1))
                with self.assertRaisesRegex(ValueError,'exceeds'):inspect_conversion(root,package)

    def test_evidence_diagnostic_records_root_failure_without_tensor_scan(self):
        from scripts.preservation_startup import inspect_evidence
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for n in ('native','export'):
                for group in ('runs','submissions'):
                    path=root/group/n/'record.json';path.parent.mkdir(parents=True);path.write_text('{}')
            path=root/'evidence/fixture/compare.json';path.parent.mkdir(parents=True);path.write_text('{}')
            result=inspect_evidence(root,['native','export'],'fixture')
            self.assertTrue(result['scan_succeeded']);self.assertEqual(result['files'],5)
            with patch.object(transfer,'inventory',side_effect=ValueError('source root differs from approved runtime target')):
                result=inspect_evidence(root,['native','export'],'fixture')
            self.assertFalse(result['scan_succeeded'])
            self.assertEqual(result['diagnostic'],'source root differs from approved runtime target')
            with patch.object(transfer,'inventory',side_effect=ValueError('fixture-secret')):
                result=inspect_evidence(root,['native','export'],'fixture')
            self.assertNotIn('fixture-secret',json.dumps(result))

    def test_pinned_sdk_source_worker_has_small_request_and_read_only_sources(self):
        import os,subprocess,sys
        env=dict(os.environ,MODAL_CONFIG_PATH='/dev/null',AGL_WORKING_VOLUME='offline-fixture')
        env.pop('AGL_REPOSITORY_PACKAGE',None)
        code="""import json, preservation_app as p
x=p.source_cpu.info()
print(json.dumps({'cpu':x.cpu,'memory':x.memory_mib,'disk':x.ephemeral_disk_mib,
 'gpus':x.gpus,'retries':x.max_retries,'single_use':x.single_use_containers,
 'mounts':{k:v.read_only for k,v in x.volumes.items()},
 'tensor_memory':p.work_cpu.info().memory_mib,'gpu_memory':p.gpu.info().memory_mib,
 'diagnostic':{'cpu':p.startup_probe.info().cpu,'memory':p.startup_probe.info().memory_mib,
 'gpus':p.startup_probe.info().gpus,'mounts':{k:v.read_only for k,v in p.startup_probe.info().volumes.items()}}}))"""
        result=json.loads(subprocess.check_output([sys.executable,'-c',code],env=env,text=True))
        self.assertEqual(result,{'cpu':2,'memory':4096,'disk':None,'gpus':[],
             'retries':0,'single_use':True,'mounts':{'/source':True,'/validation':True,'/work':False},
             'tensor_memory':131072,'gpu_memory':131072,
             'diagnostic':{'cpu':2,'memory':4096,'gpus':[],'mounts':{'/work':True}}})

    def test_phase_eval_uses_existing_launcher_and_frozen_settings(self):
        from scripts import preservation_phases as phases
        from experiment.export_artifact import sha256
        from experiment import summarize
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle=root/'bundle';bundle.mkdir()
            (bundle/'observed_calc_agent.py').write_text('# fixture')
            run=root/'runs/new-native';run.mkdir(parents=True)
            (run/'manifest.json').write_text('{}')
            config={}
            for path,value in phases.FROZEN_DTYPES.items():
                node=config
                for key in path.split('.')[:-1]:node=node.setdefault(key,{})
                node[path.split('.')[-1]]=value
            (run/'resolved_config.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'runtime dtype'):phases.check_runtime_dtypes({})
            report={'completed':64,'reward_posts_confirmed':64}
            p={'run_names':{'native_eval':'new-native'},'wandb_entity':'fixture',
               'execution_commit':'a'*40,'harness_sha256':sha256(bundle/'observed_calc_agent.py')}
            with patch.object(phases,'DATA',root),patch.object(phases,'BUNDLE',bundle), \
                 patch.object(phases.subprocess,'run') as launch, \
                 patch.object(summarize,'summarize',return_value=(report,[])), \
                 patch.object(summarize,'compare',return_value={'available':True}):
                result=phases.eval_phase(p,'native_eval',native=True)
            argv=launch.call_args.args[0]
            self.assertNotIn('--execute',argv)
            self.assertNotIn('--weave',argv)
            self.assertEqual(argv[argv.index('--phase')+1],'reload-eval')
            self.assertEqual(argv[argv.index('--checkpoint')+1],str(root/'runs/train-01/checkpoints/global_step_20'))
            self.assertEqual(result['run_name'],'new-native')

    def test_evaluation_evidence_keeps_distinct_execution_commits(self):
        from scripts import preservation_phases as phases
        p=plan();p.update(run_names={'native_eval':'native','export_eval':'export'},
                         selections={'experiment':{'provenance':{}}},working_source_identity='working',workflow_id='fixture')
        ledger={'phases':{'native_eval':{'execution_commit':'a'*40},'export_eval':{'execution_commit':'b'*40}}}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name,commit in [('native','a'*40),('export','b'*40)]:
                path=root/'submissions'/name/'receipt.json';path.parent.mkdir(parents=True)
                path.write_text(json.dumps({'request':{'source':{'commit':commit}}}))
            with patch.object(phases,'DATA',root),patch.object(phases,'working_data_root',return_value=root), \
                 patch.object(phases,'inventory',side_effect=ValueError('fixture stop')) as scan:
                with self.assertRaisesRegex(ValueError,'fixture stop'):
                    phases.execute('original_evidence_backup',p,ledger,None,None)
                provenance=scan.call_args.args[1]['evidence']['provenance']
                self.assertEqual(provenance['lab_execution_commit'],
                                 {'submissions/native/receipt.json':'a'*40,'submissions/export/receipt.json':'b'*40})
                ledger['phases']['native_eval']['execution_commit']='b'*40
                with self.assertRaisesRegex(ValueError,'identity differs'):
                    phases.execute('original_evidence_backup',p,ledger,None,None)

    def test_working_inventory_resolves_only_approved_mount_and_rejects_descendant_links(self):
        from scripts import preservation_phases as phases
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);physical=base/'volume';physical.mkdir()
            restored=physical/'restored';(restored/'evidence').mkdir(parents=True)
            (restored/'evidence/record.json').write_bytes(b'{}')
            mount=base/'work';mount.symlink_to(physical,target_is_directory=True)
            with patch.object(phases,'WORK',mount):
                root=phases.working_data_root();self.assertEqual(root,restored)
                manifest=transfer.inventory({'evidence':root},{'evidence':{'selections':[{'group':'evidence','path':'evidence'}]}},'fixture')['evidence']
                self.assertEqual([r['path'] for r in manifest['files']],['evidence/record.json'])
                self.assertEqual(phases.source_roots({},True)['experiment'],str(restored))
                (restored/'evidence/escape').symlink_to(base/'outside')
                with self.assertRaises((ValueError,OSError)):
                    transfer.inventory({'evidence':root},{'evidence':{'selections':[{'group':'evidence','path':'evidence'}]}},'fixture')
                restored.rename(physical/'retained-fixture')
                restored.symlink_to(physical/'retained-fixture',target_is_directory=True)
                with self.assertRaises(OSError):phases.working_data_root()

    def test_metadata_scope_excludes_caches_but_accounts_native_remnants(self):
        from scripts.preservation_plan import selected_metadata
        rows=[{'path':p,'type':t,'bytes':1} for p,t in (
            ('runs','2'),('runs/train-01','2'),('runs/train-01/checkpoints','2'),
            ('runs/train-01/checkpoints/global_step_5','2'),('runs/train-01/checkpoints/global_step_5/data.pt','1'),
            ('runs/train-01/log.txt','1'),('models/llama3b','2'),('models/llama3b/.cache','2'),
            ('models/llama3b/.cache/token','1'),('models/llama3b/config.json','1'))]
        files,excluded=selected_metadata(rows,[{'group':'evidence','path':'runs'},
            {'group':'native_state','path':'runs/train-01/checkpoints'},
            {'group':'prepared_base','path':'models/llama3b'}])
        self.assertIn('runs/train-01/checkpoints/global_step_5/data.pt',files)
        self.assertFalse(any('.cache' in p for p in files))
        self.assertEqual(len(excluded),2)

    def test_opaque_workers_have_no_torch_import_at_module_load(self):
        # Actual import succeeds in the client where Torch is not installed.
        import importlib.util
        self.assertIsNone(importlib.util.find_spec('torch'))
        import scripts.convert_fsdp1
        import scripts.preservation_phases



class HttpBudgetTests(unittest.TestCase):
    def test_stream_budget_and_retryable_responses_fail_without_retry(self):
        import httpx
        from scripts.preservation_http import client_factory
        seen=[]
        def receive(request):
            seen.append(request.read())
            return httpx.Response(200,content=b'ok')
        with client_factory(9000,httpx.MockTransport(receive))() as client:
            with self.assertRaisesRegex(ValueError,'byte reservation'):
                client.post('https://fixture.invalid',content=b'x'*2000)
        with client_factory(20000,httpx.MockTransport(lambda r:httpx.Response(503)))() as client:
            with self.assertRaisesRegex(ValueError,'diagnosed recovery'):
                client.get('https://fixture.invalid')
        def fail(request):raise httpx.ConnectError('secret-like exception that must not propagate')
        with client_factory(20000,httpx.MockTransport(fail))() as client:
            with self.assertRaisesRegex(ValueError,'transport failure') as error:
                client.get('https://fixture.invalid')
            self.assertNotIn('secret-like',str(error.exception))



class ConverterLogicTests(unittest.TestCase):
    def test_shard_plan_and_alias_semantics(self):
        self.assertEqual(shard_plan({'b':4,'a':3,'c':6},7),[['a','b'],['c']])
        schema={k:{'shape':[2,3],'dtype':'float32'} for k in ('lm_head.weight','model.embed_tokens.weight')}
        expected={k:v['shape'] for k,v in schema.items()}
        aliases=validate_tensor_schema(schema,expected,True,lambda a,b:True)
        self.assertEqual(aliases,{'lm_head.weight':'model.embed_tokens.weight'})
        with self.assertRaisesRegex(ValueError,'unequal'):validate_tensor_schema(schema,expected,True,lambda a,b:False)
        changed=copy.deepcopy(schema);changed['lm_head.weight']['dtype']='bfloat16'
        with self.assertRaisesRegex(ValueError,'unequal'):validate_tensor_schema(changed,expected,True,lambda a,b:True)
        with self.assertRaisesRegex(ValueError,'keys'):validate_tensor_schema({},expected,False,lambda a,b:True)

if __name__=='__main__':unittest.main()
