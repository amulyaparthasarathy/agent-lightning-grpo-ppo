"""Ordered phase implementations; invoked by the bounded remote chain only."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import tempfile

from scripts import preservation_manifest as pm
from scripts.backup_qualification import qualify, digest, EXPECTED_BASE, BASE, DATA_FILES
from scripts.preservation_transfer import (require, now, inventory, upload_snapshot, verify_snapshot,
                                           restore, identity, put_record, read_small, byte_stream,
                                           download_file, verify_bundle)
from scripts.preservation_release import publish, retrieve, reviewed_comparison
from scripts.preservation_tensor import invoke

WORK = Path('/work')
DATA = WORK/'restored'
PYTHON = '/opt/agent-lightning/.venv/bin/python'
BUNDLE = Path('/opt/agent-lightning-lab/experiment')


FROZEN_DTYPES = {
    'actor_rollout_ref.actor.fsdp_config.model_dtype':'fp32',
    'actor_rollout_ref.actor.fsdp_config.dtype':'bfloat16',
    'actor_rollout_ref.ref.fsdp_config.model_dtype':'fp32',
    'actor_rollout_ref.ref.fsdp_config.dtype':'bfloat16',
    'actor_rollout_ref.rollout.dtype':'bfloat16',
}


def check_runtime_dtypes(config):
    observed={}
    for path,expected in FROZEN_DTYPES.items():
        value=config
        for key in path.split('.'):value=value.get(key,{}) if isinstance(value,dict) else None
        require(value==expected,'resolved runtime dtype differs: '+path)
        observed[path]=value
    return observed


def save(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f: json.dump(value,f,indent=2,sort_keys=True)


def result(ledger, name):
    p=ledger['phases'][name]
    require(p['status']=='passed','predecessor not passed: '+name)
    value=p['result']
    return ledger['_receipt_reader'](value) if 'receipt_path' in value else value


def source_roots(plan, replication=False):
    return {'experiment':str(working_data_root() if replication else Path('/source')),
            'validation':'/validation', 'repository':'/repository'}


def working_data_root():
    """Resolve only the approved platform mount, then reject linked descendants.

    The worker has already checked the mounted working Volume identity. Namespace
    member paths stay relative; no platform-internal Volume prefix is hardcoded.
    """
    root=WORK.resolve(strict=True)/'restored'
    with pm.directory(root):
        pass  # O_NOFOLLOW on every component, including restored itself.
    return root


def reconcile(scope_template, manifests):
    scope=json.loads(json.dumps(scope_template))
    for name, manifest in manifests.items():
        spec=scope['namespaces'][name]
        # Expected paths came from independently reviewed metadata, not this scan.
        require({r['path'] for r in manifest['files']}==set(spec['approved_files']), 'source scope changed: '+name)
        require(not manifest['missing_selections'], 'source selection missing: '+name)
        require(manifest['exclusions']==spec['reviewed_exclusions'], 'source exclusions changed: '+name)
        require(set(spec['required_files'])<=set(spec['approved_files']), 'required source scope absent')
        rows={r['path']:r for r in manifest['files']}
        require(all(manifest['provenance'].get(k)==v for k,v in spec['expected_provenance'].items()),'source provenance differs')
        for path,sha in spec.get('expected_sha256',{}).items():
            require(rows.get(path,{}).get('sha256')==sha,'frozen source identity differs: '+path)
        for path,commit in spec.get('run_lineage',{}).items():
            require(manifest['checked_metadata'].get(path,{}).get('request',{}).get('source',{}).get('commit')==commit,
                    'source execution lineage differs: '+path)
    primary=manifests[scope['primary_namespace']]
    require(primary['required_file_findings']['structurally_complete'],'source native structure/marker invalid')
    metadata=primary['checked_metadata'];model=metadata.get(BASE+'model_manifest.json',{})
    require(all(model.get(k)==v for k,v in EXPECTED_BASE.items()),'source prepared base identity differs')
    require(metadata.get(BASE+'config.json',{}).get('architectures')==[EXPECTED_BASE['architecture']],'source architecture differs')
    rows={r['path']:r for r in primary['files']}
    weights=model.get('weights_files',[])
    require(bool(weights) and set(weights)=={p[len(BASE):] for p in rows if p.startswith(BASE) and p.endswith('.safetensors')},
            'source prepared shard coverage differs')
    require(sum(rows[BASE+n]['bytes'] for n in weights)==model.get('weight_bytes'),'source prepared weight bytes differ')
    for name in DATA_FILES:
        require(rows.get('data-small/'+name,{}).get('sha256')==metadata.get('data-small/data_manifest.json',{}).get('files',{}).get(name,{}).get('sha256'),
                'source frozen manifest differs')
    return scope


def eval_phase(plan, phase, native=False, exported=False, replication=False, training=False):
    from experiment import summarize
    name=plan['run_names'][phase]
    train_name=plan['run_names']['replication_train'] if replication else 'train-01'
    mode='main' if training else 'export-eval' if exported else 'reload-eval' if native else 'baseline-eval'
    model=DATA/'exports'/('replication' if replication else 'original') if exported else DATA/'models/llama3b'
    argv=[PYTHON,str(BUNDLE/'launch.py'),'--repo','/opt/agent-lightning','--python',PYTHON,
          '--model-path',str(model),'--data-dir',str(DATA/'data-small'),'--output-root',str(DATA/'runs'),
          '--name',name,'--profile','llama3b','--phase',mode,
          '--walltime-minutes','180' if training else '45']
    if native: argv+=['--checkpoint',str(DATA/'runs'/train_name/'checkpoints/global_step_20')]
    # Preserve the qualified runner environment and W&B configuration.
    env=dict(os.environ,HF_HOME='/tmp/hf-cache',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
    env['PATH']=str(Path(PYTHON).parent)+':'+env.get('PATH','')
    env['WANDB_ENTITY']=plan['wandb_entity']
    env.pop('AGL_WEAVE_PROJECT',None)
    env['HF_HUB_OFFLINE']='1';env['TRANSFORMERS_OFFLINE']='1'
    # No original Volume mounted; all output paths are confined to the restored working tree.
    from experiment.export_artifact import sha256
    require(sha256(BUNDLE/'observed_calc_agent.py')==plan['harness_sha256'],'frozen harness identity differs')
    subprocess.run(argv,env=env,check=True,timeout=11340 if training else 3240)
    run=DATA/'runs'/name
    dtypes=check_runtime_dtypes(json.loads((run/'resolved_config.json').read_text()))
    report, attempts=summarize.summarize(run/'traces',phase=mode,route='val')
    if not training:
        comparison=summarize.compare((report,attempts),(report,attempts),DATA/'data-small/eval_main.jsonl')
        require(comparison['available'] and report['completed']==64 and report['reward_posts_confirmed']==64,
                'evaluation incomplete or wrong heldout IDs')
    else:
        require((run/'checkpoints/latest_checkpointed_iteration.txt').read_text().strip()=='20', 'replication did not finish step 20')
    save(DATA/'submissions'/name/'receipt.json',{'status':'succeeded','request':{'source':{'commit':plan['execution_commit']},
          'action':mode,'name':name},'result':report,'at':now()})
    return {'run_name':name,'phase':mode,'report':report,'runtime_dtypes':dtypes,'execution_commit':plan['execution_commit'],
            'hosted_wandb_readback':'not_executed',
            'run_manifest_sha256':digest(json.loads((run/'manifest.json').read_text()))}


def paired(native, exported, output):
    from experiment import summarize
    a=summarize.summarize(DATA/'runs'/native['run_name']/'traces',native['phase'],'val')
    b=summarize.summarize(DATA/'runs'/exported['run_name']/'traces',exported['phase'],'val')
    comparison=summarize.compare(a,b,DATA/'data-small/eval_main.jsonl')
    require(comparison['available'],'paired evaluation incomplete')
    before={m['case_id']:e['outputs'] for m,e in a[1]}
    after={m['case_id']:e['outputs'] for m,e in b[1]}
    cases=[{'case_id':c,'baseline':before[c],'post':after[c],
            'raw_answer_changed':before[c].get('raw_answer')!=after[c].get('raw_answer')} for c in sorted(before)]
    report={'baseline':a[0],'post':b[0],'comparison':comparison,'cases':cases,
            'raw_answer_differences':sum(c['raw_answer_changed'] for c in cases)}
    save(output,report)
    return report


def execute(name, plan, ledger, api, progress):
    ledger['_receipt_reader']=lambda value: read_small(api,plan['backup_repo'],value['revision'],value['receipt_path'])
    replication=name.startswith('replication_')
    key='replication' if replication else 'original'
    reservation=next(p['reservation'] for p in plan['phases'] if p['name']==name)
    child_timeout=next(p['timeout_seconds'] for p in plan['phases'] if p['name']==name)-60
    if name.endswith('_evidence_backup'):
        scientific_phases=('replication_export_eval',) if replication else ('native_eval','export_eval')
        names=[plan['run_names'][n] for n in scientific_phases]
        lineage={}
        for phase,run_name in zip(scientific_phases,names):
            path=f'submissions/{run_name}/receipt.json'
            commit=json.loads((DATA/path).read_text())['request']['source']['commit']
            require(commit==ledger['phases'][phase].get('execution_commit'), 'evaluation receipt/ledger execution identity differs')
            lineage[path]=commit
        provenance=dict(plan['selections']['experiment']['provenance'],
                        source_identity=plan['working_source_identity'],
                        preservation_implementation_commit=plan['execution_commit'],
                        lab_execution_commit=lineage)
        selection={'provenance':provenance,
                   'selections':[{'group':'evidence','path':f'runs/{n}'} for n in names]
                     +[{'group':'evidence','path':f'submissions/{n}'} for n in names]
                     +[{'group':'evidence','path':f'evidence/{plan["workflow_id"]}'}]}
        root=working_data_root()
        manifests=inventory({'evidence':root},{'evidence':selection},name+'/'+plan['workflow_id'])
        total=sum(r['bytes'] for r in manifests['evidence']['files'])
        require(total<=min(reservation['hf_bytes'],reservation['egress_bytes']),'evidence bytes exceed reservation')
        target=upload_snapshot(api,plan['backup_repo'],plan['snapshot_prefixes'][key]+'-evaluation',{'evidence':root},
                               manifests,identity(api,plan['backup_repo']),progress,
                               resume=ledger['phases'][name].get('attempt',0)>0)
        return {'target':target,'manifests':manifests,'selections':{'evidence':selection}}
    if name.endswith('_evidence_verify'):
        backup=result(ledger,name.replace('_verify','_backup'))
        copies,_=verify_snapshot(api,backup['target'],backup['manifests'],backup['selections'],None,[],
                                  name+'/'+plan['workflow_id'],require_bundle=False)
        comparison=pm.compare(backup['manifests']['evidence'],copies['evidence'],now())
        require(comparison['content_match'],'evaluation evidence readback differs')
        return {'evidence_verified':True,'target':backup['target'],'comparison':comparison,'destination_manifests':copies}
    if name=='inventory':
        manifests=inventory(source_roots(plan),plan['selections'],'source/'+plan['workflow_id'])
        scope=reconcile(plan['scope'],manifests)
        # Retained license must match reviewed primary terms before any tensor transfer.
        license_row=next(r for r in manifests['experiment']['files'] if r['path']=='models/llama3b/LICENSE.txt')
        require(license_row['sha256']==plan['rights_review']['retained_license_sha256'], 'retained license identity differs')
        save(WORK/'receipts'/plan['workflow_id']/'source-manifests.json',manifests)
        return {'manifests':manifests,'scope':scope,'bytes':sum(r['bytes'] for m in manifests.values() for r in m['files'])}
    if name in {'backup','replication_backup'}:
        if replication:
            # Approved replication selections are scoped to new run names. Inventory every retained checkpoint/remnant.
            manifests=inventory(source_roots(plan,True),plan['replication_selections'],'replication/'+plan['workflow_id'])
            scope=json.loads(json.dumps(plan['replication_scope']))
            for namespace,m in manifests.items():
                scope['namespaces'][namespace]['approved_files']=[r['path'] for r in m['files']]
                scope['namespaces'][namespace]['reviewed_exclusions']=m['exclusions']
            # Required paths/selections remain predeclared; never regenerate those from missing output.
            scope=reconcile(scope,manifests)
        else:
            prepared=result(ledger,'inventory');manifests,scope=prepared['manifests'],prepared['scope']
        total=sum(r['bytes'] for m in manifests.values() for r in m['files'])
        require(total<=min(plan['snapshot_byte_caps'][key],reservation['hf_bytes'],reservation['egress_bytes']), 'measured snapshot exceeds reservation')
        target=upload_snapshot(api,plan['backup_repo'],plan['snapshot_prefixes'][key],source_roots(plan,replication),
                               manifests,identity(api,plan['backup_repo']),progress,
                               resume=ledger['phases'][name].get('attempt',0)>0)
        return {'target':target,'manifests':manifests,'scope':scope}
    if name in {'verify_backup','replication_verify'}:
        backup=result(ledger,'replication_backup' if replication else 'backup')
        selections=plan['replication_selections'] if replication else plan['selections']
        destinations,bundle=verify_snapshot(api,backup['target'],backup['manifests'],selections,
                                            plan['bundle_path'],plan['required_git_commits'],plan['workflow_id']+'/'+name)
        recovery_history=[]
        # The initial snapshot/manifest stays immutable during resumed transfer.
        # Archive later recovery implementations separately and verify their bytes
        # and reachable commits in this same source-free verifier.
        for addition in plan.get('recovery_bundles',[]):
            require(addition['repo_id']==plan['backup_repo'],'unapproved recovery history repository')
            with tempfile.TemporaryDirectory(prefix='recovery-history-') as temporary:
                path=Path(temporary)/'history.bundle'
                download_file(byte_stream(api,addition['repo_id'],addition['revision'],addition['path']),addition,path)
                recovery_history.append(dict(verify_bundle(path,addition['required_commits']),
                    destination=addition))
        scope=backup['scope'];scope['destination']=backup['target']
        scope['destination'].pop('bytes',None)
        receipt={'destination':scope['destination'],'method':'destination_bytes','fresh_process':True,
                 'original_source_mounted':False,'verifier_id':plan['workflow_id']+'/'+name,
                 'receipt_location':plan['backup_repo']+'/'+plan['ledger_path'],
                 'scope_sha256':digest(scope),'git_bundle':bundle,
                 'recovery_history':recovery_history,
                 'source_manifests':{n:digest(m) for n,m in backup['manifests'].items()},
                 'destination_manifests':{n:digest(m) for n,m in destinations.items()}}
        qualification=qualify(scope,backup['manifests'],destinations,receipt,now())
        require(qualification['backup_verified'],'independent scope qualification failed: '+str(qualification['failures']))
        return {'qualification':qualification,'destination_manifests':destinations,'target':scope['destination']}
    if name=='restore':
        verified=result(ledger,'verify_backup')
        manifest=verified['destination_manifests']['experiment']
        rows=[dict(r,namespace='experiment') for r in manifest['files'] if any(
            r['path']==p or r['path'].startswith(p+'/') for p in plan['restore_selections'])
              and ('checkpoints' not in Path(r['path']).parts or r['path'].startswith(pm.CHECKPOINT+'/') or r['path']==pm.MARKER)]
        require({r['path'] for r in rows}>=set(plan['restore_required_files']), 'required restored scope absent')
        require(sum(r['bytes'] for r in rows)<=reservation['modal_bytes'],'restore exceeds retained-byte reservation')
        restored=restore(api,verified['target'],rows,DATA)
        save(WORK/'receipts'/plan['workflow_id']/'restore.json',restored)
        return restored
    if name in {'native_eval','export_eval','replication_baseline','replication_train','replication_native','replication_export_eval'}:
        return eval_phase(plan,name,native=name in {'native_eval','replication_native'},
                          exported=name in {'export_eval','replication_export_eval'},replication=replication,
                          training=name=='replication_train')
    if name in {'convert','replication_convert'}:
        from experiment.export_artifact import sha256
        native_name=plan['run_names']['replication_train'] if replication else 'train-01'
        selected=DATA/'runs'/native_name/'checkpoints/global_step_20'
        verified=result(ledger,'replication_verify' if replication else 'verify_backup')
        source=verified['destination_manifests']['experiment']
        native_path=f'runs/{native_name}/checkpoints/global_step_20/actor/model_world_size_1_rank_0.pt'
        row=next(r for r in source['files'] if r['path']==native_path)
        contract=dict(plan['conversion_contracts'][key],restore_status='restored_bytes_verified',max_output_bytes=reservation['modal_bytes'],
                      source_checkpoint={'path':str(Path(native_path).parent.parent),'model_sha256':row['sha256'],'manifest_sha256':digest(source)},
                      prepared_base_manifest_sha256=next(r['sha256'] for r in source['files'] if r['path']=='models/llama3b/model_manifest.json'))
        output=DATA/'exports'/key;output.parent.mkdir(exist_ok=True)
        converted=invoke(PYTHON,'convert',dict(actor=str(selected/'actor'),base=str(DATA/'models/llama3b'),
                         data=str(DATA/'data-small'),output=str(output),contract=contract),timeout=child_timeout)
        return {'path':str(output),'manifest':converted,'manifest_sha256':sha256(output/'export_manifest.json')}
    if name in {'compare','replication_compare','replication_export_compare'}:
        first='replication_baseline' if name=='replication_compare' else 'replication_native' if replication else 'native_eval'
        second='replication_native' if name=='replication_compare' else 'replication_export_eval' if replication else 'export_eval'
        report=paired(result(ledger,first),result(ledger,second),DATA/'evidence'/plan['workflow_id']/f'{name}.json')
        # Legitimate sampling variation is retained, but publication stops for review on score/output differences.
        # A later diagnosed continuation may approve differences; never rerun for a preferred score.
        if name!='replication_compare':
            report['conversion_defect_resolved']=(report['comparison'].get('post_wins')==0
                and report['comparison'].get('post_losses')==0 and report['raw_answer_differences']==0)
        return report
    if name in {'publish','replication_publish'}:
        converted=result(ledger,'replication_convert' if replication else 'convert')
        comparison=result(ledger,'replication_export_compare' if replication else 'compare')
        require(result(ledger,key+'_evidence_verify')['evidence_verified'],'new evaluation evidence unverified')
        if not comparison['conversion_defect_resolved']:
            # A reviewer records the actual investigation without modifying original
            # outputs or rerunning evaluation. Read from one immutable Hub revision.
            revision=identity(api,plan['backup_repo'])
            path=plan['ledger_path'].rsplit('/',1)[0]+'/reviews/'+key+'-differences.json'
            require(path in api.list_repo_files(plan['backup_repo'],revision=revision),
                    'native/export differences require a bound investigation receipt: '+path)
            review=read_small(api,plan['backup_repo'],revision,path)
            comparison=reviewed_comparison(comparison,converted,review)
            comparison['difference_review']=dict(review,receipt_revision=revision,receipt_path=path)
        total=sum(r['bytes'] for r in converted['manifest']['files'].values())
        require(total<=min(reservation['hf_bytes'],reservation['egress_bytes']),'release exceeds storage/egress reservation')
        branch=plan['release_branches'][key]
        if replication:
            original_revision=result(ledger,'publish')['revision']
            branches={b.name:b.target_commit for b in api.list_repo_refs(plan['release_repo']).branches}
            if branch in branches:
                require(branches[branch]==original_revision,'replication branch advanced; reconcile publication')
            else:
                api.create_branch(plan['release_repo'],branch=branch,revision=original_revision,exist_ok=False)
        parent=api.repo_info(plan['release_repo'],revision=branch).sha
        return publish(api,plan['release_repo'],converted['path'],json.loads((DATA/'data-small/data_manifest.json').read_text()),
                       plan['harness_sha256'],branch,parent,comparison)
    if name in {'retrieve','replication_retrieve'}:
        target=result(ledger,'replication_publish' if replication else 'publish')
        # Fresh scratch, no source or working Volume mounted in this worker.
        with tempfile.TemporaryDirectory(prefix='release-retrieval-') as scratch:
            root=Path(scratch)/'release'
            report=retrieve(api,target,root,plan['frozen_data_manifest'],plan['harness_sha256'])
            report['load']=invoke(PYTHON,'clean_load',dict(root=str(root),compute_dtype=plan['compute_dtype']),timeout=child_timeout)
            return report
    raise ValueError('unknown preservation phase')
