"""Bounded detached preservation chain. Import only with an approved working Volume.

All original mounts are read-only. Verification/retrieval workers have no Volume
mounts. There is no create-if-missing path or credential-file upload in this app.
"""
from __future__ import annotations
import json
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import modal

# Reuse the qualified build layers, binding its default Volume only to working storage.
if not os.environ.get('AGL_WORKING_VOLUME'):
    raise ValueError('Explicit AGL_WORKING_VOLUME required')
import modal_app as runtime
from scripts.preservation_workflow import HubLedger, validate_plan, persist_before_publish, observed_usage, SOURCE_STREAM_RESOURCES
from scripts.preservation_transfer import require, identity, put_record, GateFailure

LOCAL=Path(__file__).parent
BACKUP='amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup'
ENVIRONMENT=os.environ.get('AGL_MODAL_ENVIRONMENT','main')
source=modal.Volume.from_name('cs329z-agent-lightning',environment_name=ENVIRONMENT,create_if_missing=False)
validation=modal.Volume.from_name('agent-lightning-validation',environment_name=ENVIRONMENT,create_if_missing=False)
working=modal.Volume.from_name(os.environ['AGL_WORKING_VOLUME'],environment_name=ENVIRONMENT,create_if_missing=False)
publisher=modal.Secret.from_name('agent-lightning-publisher',environment_name=ENVIRONMENT,required_keys=['HF_TOKEN'])
training=modal.Secret.from_name('agent-lightning-credentials',environment_name=ENVIRONMENT,required_keys=['HF_TOKEN','WANDB_API_KEY'])
image=runtime.image.uv_pip_install('huggingface-hub==1.33.0')
for name in ('preservation_manifest','backup_qualification','preservation_transfer','preservation_workflow',
             'preservation_phases','preservation_release','preservation_worker','preservation_http','preservation_tensor','convert_fsdp1','__init__'):
    image=image.add_local_file(LOCAL/'scripts'/f'{name}.py',f'/opt/preservation/scripts/{name}.py',copy=True)
image=image.add_local_file(LOCAL/'scripts/preservation_startup.py','/opt/preservation/scripts/preservation_startup.py',copy=True)
for name in ('export_artifact','summarize','__init__'):
    image=image.add_local_file(LOCAL/'experiment'/f'{name}.py',f'/opt/preservation/experiment/{name}.py',copy=True)
# Explicit small repository recovery package, created/verified after integration merge.
package=os.environ.get('AGL_REPOSITORY_PACKAGE')
if package:
    root=Path(package)
    require(set(p.name for p in root.iterdir())=={'history.bundle','Notice','provenance.json'},'repository package is not closed')
    for name in ('history.bundle','Notice','provenance.json'):
        require((root/name).is_file() and not (root/name).is_symlink(),'invalid repository package member')
        image=image.add_local_file(root/name,'/repository/'+name,copy=True)
image=image.env({'PYTHONPATH':'/opt/preservation','HF_DEBUG':'0','HF_HUB_DISABLE_TELEMETRY':'1',
                 'HF_HUB_DISABLE_XET':'1','AGL_WORKING_VOLUME':os.environ['AGL_WORKING_VOLUME'],'AGL_MODAL_ENVIRONMENT':ENVIRONMENT})
app=modal.App('cs329z-agent-lightning-preservation')


def api_client():
    from scripts.preservation_http import configure
    configure(64*1024**2)
    from huggingface_hub import HfApi
    import logging
    logging.disable(logging.CRITICAL)
    return HfApi(token=os.environ['HF_TOKEN'])


def worker(ledger_path,phase,has_work=False):
    api=api_client();store=HubLedger(api,BACKUP,ledger_path)
    _,ledger=store.load();plan=validate_plan(ledger['plan'])
    call_id=modal.current_function_call_id()
    working.hydrate()
    require(working.object_id==plan['working_volume_id'],'working Volume identity differs')
    if phase['name'] in {'inventory','backup'}:
        source.hydrate();validation.hydrate()
        require(source.object_id==plan['source_volume_id'] and validation.object_id==plan['validation_volume_id'],
                'source Volume identity differs')
    store.started(phase['name'],call_id)
    started=time.monotonic()
    if has_work: working.reload()
    success=False; value={'error_type':'ChildDidNotReturn'}
    try:
        with tempfile.TemporaryDirectory(prefix='preservation-worker-') as scratch:
            request=Path(scratch)/'request.json';output=Path(scratch)/'result.json'
            request.write_text(json.dumps({'phase':phase['name'],'plan':plan,'ledger':ledger}))
            env=dict(os.environ,HF_HOME=str(Path(scratch)/'hf-cache'))
            # All Hub transport uses the pinned client SDK. Scientific launch and
            # tensor operations create their own qualified training subprocesses.
            child=subprocess.run([sys.executable,'-m','scripts.preservation_worker','--input',str(request),'--output',str(output)],
                                 env=env,timeout=phase['timeout_seconds'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            if output.exists():
                response=json.loads(output.read_text());value=response['result']
                success=child.returncode==0 and response['success'] is True
            else:
                value={'error_type':'ChildDidNotReturn','returncode':child.returncode,'phase':phase['name']}
    except Exception as exc:
        value={'error_type':type(exc).__name__,'phase':phase['name']}
    elapsed=time.monotonic()-started
    # Persist each full receipt independently; the ledger remains small.
    attempt=ledger['phases'][phase['name']].get('attempt',0)
    receipt_path=ledger_path.rsplit('/',1)[0]+'/receipts/'+phase['name']+f'-{attempt}.json'
    def fallback(record):
        target='/receipts/'+plan['workflow_id']+'/'+phase['name']+f'-{attempt}-terminal.json'
        content=json.dumps({'phase':phase['name'],'function_call_id':call_id,'success':success,
                            'elapsed_seconds':elapsed,'execution_commit':plan['execution_commit'],
                            'result':record},sort_keys=True).encode()
        if has_work:
            path=Path('/work'+target);path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as handle:handle.write(content)
            working.commit()
        else:
            # Write only the receipt via the Volume API; verifier has no Volume mounts.
            with working.batch_upload(force=False) as batch:
                batch.put_file(io.BytesIO(content),target)
    revision=persist_before_publish(value,fallback,
        lambda record:put_record(api,BACKUP,identity(api,BACKUP),receipt_path,record))
    usage=observed_usage(phase,elapsed)
    terminal=store.terminal(phase['name'],call_id,{'receipt_path':receipt_path,'revision':revision},usage,success)
    if success and terminal['status']!='complete':
        coordinate.spawn(ledger_path,call_id)
    return {'status':terminal['status'],'phase':phase['name'],'receipt_path':receipt_path,'revision':revision}


def guarded_worker(ledger_path,phase,has_work=False):
    try:
        return worker(ledger_path,phase,has_work)
    except Exception as exc:
        # Suppress SDK bodies, headers and signed URLs even if receipt persistence fails.
        detail=str(exc) if isinstance(exc,GateFailure) else type(exc).__name__
        raise RuntimeError('Preservation '+phase['name']+' stopped: '+detail+'; reconcile durable state') from None


@app.function(image=image,volumes={'/source':source.read_only(),'/validation':validation.read_only(),'/work':working},
              secrets=[publisher],cpu=SOURCE_STREAM_RESOURCES[0],memory=SOURCE_STREAM_RESOURCES[1]*1024,timeout=10860,retries=0,
              max_containers=1,single_use_containers=True)
def source_cpu(ledger_path,phase):
    return guarded_worker(ledger_path,phase,True)


@app.function(image=image,volumes={'/work':working,'/validation':validation.read_only()},secrets=[publisher],
              cpu=8,memory=131072,ephemeral_disk=524288,timeout=10860,retries=0,max_containers=1,single_use_containers=True)
def work_cpu(ledger_path,phase):
    return guarded_worker(ledger_path,phase,True)


@app.function(image=image,secrets=[publisher],cpu=8,memory=131072,ephemeral_disk=524288,
              timeout=10860,retries=0,max_containers=1,single_use_containers=True)
def isolated_cpu(ledger_path,phase):
    return guarded_worker(ledger_path,phase)


@app.function(image=image,volumes={'/work':working},secrets=[training,publisher],gpu='H100!',cpu=16,memory=131072,
              timeout=12000,retries=0,max_containers=1,single_use_containers=True)
def gpu(ledger_path,phase):
    return guarded_worker(ledger_path,phase,True)


@app.function(image=image,secrets=[publisher],cpu=0.125,memory=512,timeout=180,retries=0,
              max_containers=1,single_use_containers=True)
def coordinate(ledger_path,previous_call='',recover_phase='',diagnosis=''):
    try:
        if previous_call:
            # Previous GPU must have exited before any next worker is launched.
            modal.FunctionCall.from_id(previous_call).get(timeout=120)
        store=HubLedger(api_client(),BACKUP,ledger_path)
        if recover_phase:
            _,ledger=store.load()
            old=ledger['phases'][recover_phase]
            require(old['status']=='failed' and old.get('function_call_id'),'failed call identity required')
            def terminal(nodes):
                for node in nodes:
                    if node.function_call_id==old['function_call_id']:
                        return node.status.name in {'SUCCESS','FAILURE','INIT_FAILURE','TERMINATED','TIMEOUT'}
                    if terminal(node.children):return True
                return False
            require(terminal(modal.FunctionCall.from_id(old['function_call_id']).get_call_graph()),
                    'prior call is not terminal; never duplicate')
            phase,ledger=store.recover(recover_phase,diagnosis,True)
        else:
            phase,ledger=store.claim()
        if phase is None:return {'status':'complete'}
        require(ledger['plan']['working_volume_name']==os.environ['AGL_WORKING_VOLUME'],'working binding differs')
        if phase['kind']=='gpu': function=gpu
        elif phase['name'] in {'inventory','backup'}: function=source_cpu
        elif phase['name'] in {'verify_backup','replication_verify','retrieve','replication_retrieve','original_evidence_verify','replication_evidence_verify'}: function=isolated_cpu
        else:function=work_cpu
        # The worker durably records its own ID before any artifact operation. A lost
        # spawn acknowledgement leaves a launching claim, never an automatic duplicate.
        call=function.spawn(ledger_path,phase)
        return {'phase':phase['name'],'function_call_id':call.object_id,'status':'submitted_not_complete'}
    except Exception as exc:
        raise RuntimeError('Coordinator stopped: '+type(exc).__name__+'; reconcile durable state') from None



@app.local_entrypoint()
def main(plan_file:str='',ledger_path:str='',recover_phase:str='',diagnosis:str='',diagnostic_id:str=''):
    require('--detach' in sys.argv or '-d' in sys.argv,'modal run --detach required')
    require(bool(ledger_path),'explicit durable ledger path required')
    if plan_file:
        # Initialize through a remote coordinator? Client credentials are never read
        # implicitly by this entrypoint. Initialize with the separate secure preflight.
        raise ValueError('Initialize the approved ledger with scripts/preservation_preflight.py first')
    require(not diagnostic_id or not recover_phase,'diagnostic and recovery are separate invocations')
    call=startup_probe.spawn(ledger_path,diagnostic_id) if diagnostic_id else coordinate.spawn(ledger_path,recover_phase=recover_phase,diagnosis=diagnosis)
    print(json.dumps({'coordinator_call_id':call.object_id,'app_id':app.app_id,'ledger_path':ledger_path}),flush=True)


@app.function(image=image,volumes={'/work':working.read_only()},secrets=[publisher],cpu=2,memory=4096,
              timeout=180,retries=0,max_containers=1,single_use_containers=True)
def startup_probe(ledger_path,diagnostic_id):
    """A preclaimed, CPU-only observation; never launches a scientific phase."""
    try:
        return _startup_probe_impl(ledger_path,diagnostic_id)
    except Exception as exc:
        raise RuntimeError('Startup diagnostic failed: '+type(exc).__name__+'; reconcile durable claim') from None


def _startup_probe_impl(ledger_path,diagnostic_id):
    from scripts.preservation_startup import probe
    from scripts.backup_qualification import digest
    from scripts.preservation_manifest import member
    require('/' not in member(diagnostic_id),'plain diagnostic name required')
    api=api_client();store=HubLedger(api,BACKUP,ledger_path)
    revision,ledger=store.load();claim=ledger['diagnostics'][diagnostic_id]
    phase=claim.get('phase','native_eval')
    require(phase in {'native_eval','convert','original_evidence_backup'} and claim['status']=='launching'
            and ledger['phases'][phase]['status']=='failed','diagnostic claim differs')
    working.hydrate()
    require(working.object_id==ledger['plan']['working_volume_id'],'working Volume identity differs')
    claim.update(status='running',function_call_id=modal.current_function_call_id())
    store.save(revision,ledger)
    started=time.monotonic()
    kind={'native_eval':'import_only_diagnostic','convert':'conversion_metadata_diagnostic',
          'original_evidence_backup':'evidence_inventory_diagnostic'}[phase]
    report={'kind':kind,'diagnostic_id':diagnostic_id,
            'function_call_id':modal.current_function_call_id(),
            'implementation_commit':claim['implementation_commit']}
    if phase=='native_eval':
        report.update(client=probe(sys.executable),training=probe(runtime.PYTHON))
        name=ledger['plan']['run_names']['native_eval']
        report['native_run_exists']=(Path('/work/restored/runs')/name).exists()
        report['native_submission_exists']=(Path('/work/restored/submissions')/name).exists()
    elif phase=='convert':
        report['conversion']=probe(runtime.PYTHON,'inspect_conversion')
    else:
        report['evidence']=probe(sys.executable,'inspect_evidence',{'root':'/work/restored',
            'names':[ledger['plan']['run_names'][p] for p in ('native_eval','export_eval')],
            'workflow_id':ledger['plan']['workflow_id']})
    report['elapsed_seconds']=time.monotonic()-started
    report['probe_sha256']=__import__('hashlib').sha256(Path('/opt/preservation/scripts/preservation_startup.py').read_bytes()).hexdigest()
    require(report['probe_sha256']==claim['probe_sha256'],'diagnostic source differs')
    # Only the new working Volume receives a fallback; its mounted view is read-only.
    with working.batch_upload(force=False) as batch:
        batch.put_file(io.BytesIO(json.dumps(report).encode()),'/receipts/'+ledger['plan']['workflow_id']+'/'+diagnostic_id+'.json')
    path=ledger_path.rsplit('/',1)[0]+'/receipts/'+diagnostic_id+'.json'
    receipt_revision=put_record(api,BACKUP,identity(api,BACKUP),path,report)
    revision,ledger=store.load();claim=ledger['diagnostics'][diagnostic_id]
    require(claim['function_call_id']==modal.current_function_call_id(),'diagnostic identity changed')
    claim.update(status='observed',result={'revision':receipt_revision,'receipt_path':path},report_sha256=digest(report))
    store.save(revision,ledger)
    return {'receipt_revision':receipt_revision,'receipt_path':path,'scientific_phase_submitted':False}
