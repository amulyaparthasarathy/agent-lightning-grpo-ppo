"""Private inference publication and isolated retrieval, never raw evidence."""
from pathlib import Path
import json
import hashlib
from experiment.export_artifact import verify_export, sha256
from scripts.preservation_transfer import identity, require, byte_stream, download_file, read_small, now
from scripts.backup_qualification import digest


def reviewed_comparison(evaluation, converted, review):
    """Bind a recorded investigation to these exact outputs and validated tensors."""
    require(review.get('comparison_sha256')==digest(evaluation)
            and review.get('export_manifest_sha256')==converted['manifest_sha256'],
            'difference review identifies different evidence')
    require(review.get('disposition')=='no_conversion_or_loading_defect'
            and review.get('reviewer') and review.get('reviewed_at')
            and review.get('analysis') and review.get('evidence_locations'),
            'difference investigation unresolved or undocumented')
    require(converted['manifest']['tensor_validation']['all_tensors_equal'] is True,
            'difference review cannot waive tensor equality')
    return dict(evaluation,conversion_defect_resolved=True,difference_review=review)


def publish(api, repo, root, data_manifest, harness_hash, branch, expected_parent, evaluation):
    from huggingface_hub import CommitOperationAdd
    root=Path(root)
    manifest=verify_export(root,data_manifest,harness_hash)
    require(evaluation.get("comparison",{}).get("available") is True, "native/export comparison incomplete")
    require(evaluation.get("conversion_defect_resolved") is True, "unresolved conversion/loading investigation")
    require((root/"LICENSE").is_file() and (root/"Notice").is_file(), "release agreement/Notice absent")
    card=(root/"README.md").read_text()
    require(card.startswith('# Llama') and 'Built with Llama' in card, "release attribution missing")
    identity(api,repo)
    require(api.repo_info(repo,revision=branch).sha==expected_parent, "release branch changed")
    existing=set(api.list_repo_files(repo,revision=expected_parent))
    approved=set(manifest['files'])|{'export_manifest.json'}
    require(existing<=(approved|{'.gitattributes'}), "unrelated release files present")
    if branch=='main':
        require(existing<={'.gitattributes'}, "original release already exists; never overwrite")
    else:
        require(branch.startswith('replication-'), "only distinct replication branches allowed")
    api.auth_check(repo,write=True)
    # Add small measured summaries to the manifest only. Inference files and the
    # all-tensor receipt remain byte-identical to those evaluated. Raw cases stay private.
    manifest['evaluated_export_manifest_sha256']=sha256(root/'export_manifest.json')
    manifest['release_evaluation']={
        'native':evaluation['baseline']['observed_reward'],
        'export':evaluation['post']['observed_reward'],
        'comparison':evaluation['comparison'],
        'raw_answer_differences':evaluation.get('raw_answer_differences'),
        'difference_review':evaluation.get('difference_review')}
    content=json.dumps(manifest,indent=2,sort_keys=True).encode()
    operations=[CommitOperationAdd(path_in_repo=n,path_or_fileobj=str(root/n)) for n in sorted(manifest['files'])]
    operations.append(CommitOperationAdd(path_in_repo='export_manifest.json',path_or_fileobj=content))
    result=api.create_commit(repo_id=repo,operations=operations,revision=branch,parent_commit=expected_parent,
                             commit_message='Validated Llama trained inference release',num_threads=1)
    return {'repo_id':repo,'revision':result.oid,'private':True,'branch':branch,
            'export_manifest_sha256':hashlib.sha256(content).hexdigest(),'at':now()}



def retrieve(api, target, output, data_manifest, harness_hash):
    root=Path(output)
    root.mkdir(parents=True,exist_ok=False)
    identity(api,target['repo_id'])
    manifest=read_small(api,target['repo_id'],target['revision'],'export_manifest.json')
    # Serialize only after checking the actual downloaded manifest bytes separately.
    chunks=byte_stream(api,target['repo_id'],target['revision'],'export_manifest.json')
    content=bytearray()
    for chunk in chunks:
        content.extend(chunk)
        require(len(content)<=8*1024**2,'release manifest exceeds bound')
    require(len(content)<=8*1024**2 and hashlib.sha256(content).hexdigest()==target['export_manifest_sha256'], 'release manifest differs')
    (root/'export_manifest.json').write_bytes(content)
    actual=set(api.list_repo_files(target['repo_id'],revision=target['revision']))
    require(actual<=set(manifest['files'])|{'export_manifest.json','.gitattributes'}, 'unlisted release files')
    for name,row in manifest['files'].items():
        from scripts.preservation_manifest import exclusion
        require(not exclusion(name),'excluded release member')
        require('/' not in name and '\\' not in name and name not in {'.','..'}, 'invalid release member')
        download_file(byte_stream(api,target['repo_id'],target['revision'],name),row,root/name)
    verify_export(root,data_manifest,harness_hash)
    return {'status':'retrieved_bytes_verified','destination':target,'files':manifest['files'],'at':now()}


def clean_load(root, compute_dtype):
    """Called only in a fresh remote CPU process with a fresh cache."""
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    require(compute_dtype in {'bfloat16','float32'},'unsupported runtime dtype')
    tokenizer=AutoTokenizer.from_pretrained(root,local_files_only=True,trust_remote_code=False)
    model=AutoModelForCausalLM.from_pretrained(root,local_files_only=True,trust_remote_code=False,
                                              dtype=getattr(torch,compute_dtype),device_map='cpu')
    model.eval()
    prompt=tokenizer.apply_chat_template([{'role':'user','content':'What is 2 + 2? Answer briefly.'}],
                                         tokenize=False,add_generation_prompt=True)
    inputs=tokenizer(prompt,return_tensors='pt')
    with torch.inference_mode():
        output=model.generate(**inputs,max_new_tokens=32,do_sample=False)
    return {'model_loaded':True,'tokenizer_loaded':True,'compute_dtype':compute_dtype,
            'generation_token_ids':output[0,inputs['input_ids'].shape[1]:].tolist(),
            'generation_text':tokenizer.decode(output[0,inputs['input_ids'].shape[1]:]),'at':now()}
