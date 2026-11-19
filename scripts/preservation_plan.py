"""Build the explicit reviewed plan from small preflight facts, never credentials.

Missing capacity, source inventory, rights or exact integration identity is an
error. This tool does not probe accounts or invent successful observations.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
from scripts import preservation_manifest as pm
from scripts.backup_qualification import EXPECTED_BASE, ORIGINAL_LINEAGE
from scripts.preservation_workflow import ORDER, LIMITS, validate_plan, SOURCE_STREAM_PHASES, SOURCE_STREAM_RESOURCES
from scripts.preservation_preflight import capacity
from scripts.preservation_transfer import require

FROZEN = dict(zip(('train_smoke.parquet','eval_smoke.parquet','eval_smoke.jsonl','train_main.parquet','eval_main.parquet','eval_main.jsonl'), (
 '065b218b8b2bb16f80edf7b4721e2cd98fc6188f603af02146182591ccba6182',
 'd0ceb9f4daea9af580e09b932493ee35ade4ad945127603da2c0f272d54c074e',
 '703e6203df93c3741f0de843f22a22cf66237b44fb3267193484f79a7ad0f960',
 '6010acaa4d4be314101a537d8fa3b3d0f7f3811ddc53b0162ff8a61300620178',
 'e90950dcb3daedea284091a54d40de5f2f59a8530e28bdd8e4e4f8ee53ecc0ba',
 '27445758e42fd93c062ce297515869af2a372e1edd66cd7e1f853c37fbc96850')))
EXPECTED_HASHES={'data-small/'+n:h for n,h in FROZEN.items()} | {
 'models/llama3b/tokenizer_config.json':'7fe399d0739ac28d562a04f0d294093d5d38ad3f00e6ea0e624f932ccf397116',
 'models/llama3b/chat_template.jinja':'a718674808440b537a553c15e0557b5b45d6d76e49ad9ddbc1a9e92447843e2c'}


def selected_metadata(entries, selections, excluded_symlinks=()):
    """Reconcile metadata paths using the same exclusions, without reading bytes."""
    selected,excluded={},{}
    indexed={pm.member(e['path']):e for e in entries}
    for selection in selections:
        root=pm.member(selection['path']);group=selection['group'];pruned=[]
        require(root in indexed,'required selection absent in metadata: '+root)
        for path,e in sorted(indexed.items(),key=lambda item:(len(Path(item[0]).parts),item[0])):
            if not(path==root or path.startswith(root+'/')):continue
            if any(path.startswith(p+'/') for p in pruned):continue
            reason=pm.exclusion(path)
            if group=='evidence' and 'checkpoints' in Path(path).parts:
                reason='checkpoint subtree outside evidence group; select native state separately'
            if reason:
                excluded[path]={'path':path,'reason':reason};pruned.append(path);continue
            if path in excluded_symlinks:
                require(group=='evidence' and e['type']=='3','reviewed exclusion is not an evidence symlink')
                excluded[path]={'path':path,'reason':'reviewed evidence symlink excluded; target not read'}
                continue
            require(e['type'] in ('1','2'),'unsupported source metadata type')
            if e['type']=='1':
                require(path not in selected,'duplicate selected path')
                selected[path]=e
    return sorted(selected), sorted(excluded.values(),key=lambda e:e['path'])


def build(facts):
    plan=copy.deepcopy(facts)
    require(plan['remote_execution_approved']=='YES','approval missing')
    require(len(plan['execution_commit'])==40,'exact integrated commit required')
    require(plan['existing_token_scope_exception'] is True,'token exception must be explicit')
    require(plan['harness_sha256']=='e802a2324dab779ffaa49af678196a7cc5f02e3bb89936bd0ed1f1ed8c3a02fc','frozen harness anchor differs')
    require(plan['compute_dtype']=='bfloat16','retrieval serving dtype differs from frozen rollout')
    plan['owner']='amulyaparthasarathy';plan['limits']=LIMITS
    plan['backup_repo']='amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-Backup'
    plan['release_repo']='amulyaparthasarathy/Llama-3.2-3B-AgentLightning-Lab-GRPO'
    label=pm.member(plan['workflow_id']);require('/' not in label,'plain workflow name required')
    plan['ledger_path']=f'workflows/{label}/ledger.json'
    plan['snapshot_prefixes']={k:f'snapshots/{label}-{k}' for k in ('original','replication')}
    plan['release_branches']={'original':'main','replication':'replication-'+label}
    plan['bundle_path']='repository/history.bundle'
    commits=list(dict.fromkeys([*ORIGINAL_LINEAGE.values(),'377b721cb0719ef1bee3521fadf4621675f8e4ec',
                              'ab93992daf21999ba658bd038ffe9ba565621431',plan['execution_commit']]))
    plan['required_git_commits']=commits
    plan['run_names']={name:label+'-'+name.replace('_','-') for name in ORDER if name in {
        'native_eval','export_eval','replication_baseline','replication_train','replication_native','replication_export_eval'}}
    primary=[{'group':'evidence','path':p} for p in ('runs','evidence','submissions','data-small','data-source','staging')]
    primary += [{'group':'prepared_base','path':'models/llama3b'}]
    checkpoints=[e['path'] for e in facts['metadata']['experiment'] if e['path'].startswith('runs/') and e['path'].endswith('/checkpoints')]
    primary += [{'group':'native_state','path':p} for p in sorted(checkpoints)]
    selections={'experiment':primary,'validation':[{'group':'evidence','path':'runs'}],
                'repository':[{'group':'evidence','path':n} for n in ('history.bundle','Notice','provenance.json')]}
    provenance={'lab_execution_commit':ORIGINAL_LINEAGE,'documentation_commit':'377b721cb0719ef1bee3521fadf4621675f8e4ec',
                'agent_lightning_commit':'cbc56e41e0eab643aa71d195e92e5bba652fa683',
                'preservation_implementation_commit':plan['execution_commit']}
    plan['selections']={};namespaces={}
    for namespace,items in selections.items():
        prov=dict(provenance,source_identity=facts['source_identities'][namespace],
                  lab_execution_commit=facts['namespace_execution_provenance'][namespace])
        plan['selections'][namespace]={'selections':items,'provenance':prov,
                                       'provenance_records':list(ORIGINAL_LINEAGE) if namespace=='experiment' else [],
                                       'excluded_symlinks':facts.get('excluded_symlinks',{}).get(namespace,[])}
        files,exclusions=selected_metadata(facts['metadata'][namespace],items,
                                          plan['selections'][namespace]['excluded_symlinks'])
        namespaces[namespace]={'source_identity':prov['source_identity'],'expected_provenance':prov,'selections':items,
                              'approved_files':files,'required_files':files,'reviewed_exclusions':exclusions,
                              'expected_sha256':EXPECTED_HASHES if namespace=='experiment' else {},
                              'run_lineage':ORIGINAL_LINEAGE if namespace=='experiment' else {}}
    plan['scope']={'schema_version':1,'approval_id':plan['approval_id'],'approved_by':'amulyaparthasarathy',
                   'remote_execution_approved':'YES','expected_base':EXPECTED_BASE,'namespaces':namespaces,
                   'primary_namespace':'experiment','validation_namespace':'validation','repository_namespace':'repository',
                   'native_checkpoint':pm.CHECKPOINT,'retained_notices':['models/llama3b/LICENSE.txt'],
                   'required_git_commits':commits}
    plan['restore_selections']=['models/llama3b','data-small','runs/train-01']
    # Only step 20 and source-run metadata, not step-15 optimizer state, are restored.
    plan['restore_required_files']=[pm.CHECKPOINT+'/'+n for n in pm.NATIVE_FILES]+[pm.MARKER]+['runs/train-01/'+n for n in ('manifest.json','model_manifest.json','data_manifest.json','resolved_config.json')]
    replica=plan['run_names']['replication_train']
    plan['replication_selections']=copy.deepcopy(plan['selections'])
    new_runs=[plan['run_names'][n] for n in ('replication_baseline','replication_train','replication_native')]
    items=[{'group':'evidence','path':f'runs/{n}'} for n in new_runs]
    items += [{'group':'evidence','path':f'submissions/{n}'} for n in new_runs]
    items += [{'group':'evidence','path':'data-small'},{'group':'evidence','path':f'evidence/{label}/replication_compare.json'},
              {'group':'prepared_base','path':'models/llama3b'},
              {'group':'native_state','path':f'runs/{replica}/checkpoints/global_step_20'},
              {'group':'native_state','path':f'runs/{replica}/checkpoints/latest_checkpointed_iteration.txt'}]
    rep=plan['replication_selections']['experiment'];rep['selections']=items
    rep['native_checkpoint']=f'runs/{replica}/checkpoints/global_step_20'
    rep['provenance_records']=[f'submissions/{n}/receipt.json' for n in new_runs]
    rep['provenance']['source_identity']=facts['working_source_identity']
    rep['provenance']['lab_execution_commit']={p:plan['execution_commit'] for p in rep['provenance_records']}
    plan['replication_scope']=copy.deepcopy(plan['scope']);rs=plan['replication_scope']
    rs['native_checkpoint']=rep['native_checkpoint']
    rs['namespaces']['experiment'].update(selections=items,source_identity=rep['provenance']['source_identity'],expected_provenance=rep['provenance'],
        run_lineage=rep['provenance']['lab_execution_commit'],
        required_files=[rep['native_checkpoint']+'/'+n for n in pm.NATIVE_FILES]+[f'runs/{replica}/checkpoints/latest_checkpointed_iteration.txt']
        +[f'runs/{run}/'+n for run in new_runs for n in ('manifest.json','result.json','data_manifest.json','resolved_config.json')])
    plan['conversion_contracts']={}
    for k in ('original','replication'):
        plan['conversion_contracts'][k]={'training_execution_commit':plan['execution_commit'] if k=='replication' else ORIGINAL_LINEAGE['submissions/train-01/receipt.json'],
             'integration_execution_commit':plan['execution_commit'],'converter_commit':plan['execution_commit'],
             'harness_sha256':plan['harness_sha256'],'model_card':facts['model_cards'][k]}
    plan['phases']=[]
    for name in ORDER:
        gpu=name in plan['run_names']
        cpu,memory = SOURCE_STREAM_RESOURCES if name in SOURCE_STREAM_PHASES else (16 if gpu else 8,128)
        seconds=11400 if name=='replication_train' else 3300 if gpu else facts['cpu_timeouts'][name]
        startup=600
        reservation=dict.fromkeys(LIMITS,0)
        reservation['egress_bytes']=1024**3  # conservative request/log/receipt traffic
        reservation['hf_bytes']=16*1024**2  # independent phase receipt storage
        reservation['gpu_seconds' if gpu else 'cpu_seconds']=seconds+startup
        # Conservative full allocation plus 20% cost contingency, no free credits assumed.
        rate=(0.001097 if gpu else 0)+cpu*0.0000131+memory*0.00000222
        reservation['cost_usd']=(seconds+startup)*rate*1.2
        for metric,amount in facts['phase_storage_and_transfer_reservations'].get(name,{}).items():reservation[metric]=amount
        reservation['cost_usd']+=reservation['egress_bytes']/1024**3*0.04*1.2
        plan['phases'].append({'name':name,'kind':'gpu' if gpu else 'cpu','cpu':cpu,'memory_gib':memory,
                              'timeout_seconds':seconds,'startup_reserve_seconds':startup,'reservation':reservation})
    capacity(plan);validate_plan(plan,require_existing=False)
    plan.pop('metadata');return plan


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--facts',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();pm.write_output(a.output,build(pm.read_json(a.facts)))

if __name__=='__main__':main()
