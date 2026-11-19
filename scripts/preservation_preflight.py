"""Explicit, sanitized resource preflight. Never sources .env or writes login caches."""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout
import io
import json
import logging
from pathlib import Path
import os

from scripts.check_validation_env import check
from scripts.preservation_transfer import require, identity
from scripts.preservation_workflow import HubLedger, validate_plan

OWNER='amulyaparthasarathy'
REPOS=tuple(OWNER+'/Llama-3.2-3B-AgentLightning-Lab-'+suffix for suffix in ('Backup','GRPO'))


def read_hf_token(path=Path('.env')):
    """Existing schema/checker; non-executing parser consumes only HF_TOKEN."""
    path=Path(path)
    with redirect_stdout(io.StringIO()):check(path)
    before=path.stat()
    flags=os.O_RDONLY|os.O_NOFOLLOW
    with os.fdopen(os.open(path,flags),'r') as f:
        opened=os.fstat(f.fileno())
        require((before.st_ino,before.st_mtime_ns,before.st_size)==(opened.st_ino,opened.st_mtime_ns,opened.st_size),
                'credential file changed')
        token=None
        for line in f:
            key,sep,value=line.strip().partition('=')
            if sep and key.strip()=='HF_TOKEN':
                value=value.strip()
                token=value[1:-1] if value[:1] in {'"',"'"} else value
        after=os.fstat(f.fileno())
        require((opened.st_mtime_ns,opened.st_size)==(after.st_mtime_ns,after.st_size),'credential file changed')
    require(bool(token),'required credential unavailable')
    return token


def capacity(plan):
    evidence=plan['capacity_evidence']
    require(evidence.get('observed_at') and evidence.get('source'),'capacity evidence lacks origin/date')
    available=evidence.get('hf_available_bytes')
    attestation=evidence.get('hf_owner_attestation')
    if attestation is not None:
        require(available is None,'rounded owner evidence must not masquerade as measured available bytes')
        require(attestation.get('owner')==OWNER and attestation.get('subscription')=='PRO'
                and attestation.get('private_used_display')=='0 TB'
                and attestation.get('private_total_display')=='1 TB'
                and attestation.get('rounded') is True and attestation.get('approved_for_workflow') is True,
                'unsupported or incomplete owner capacity attestation')
        # Owner approved the full <=300 GiB workflow against the reported 0/1 TB
        # dashboard. This is an allocation ceiling, NOT a provider byte measurement.
        available=300*1024**3
    require(type(available) is int and available>=300*1024**3,
            'owner-required 300 GiB available private storage not established')
    require(available>=plan['projected_hf_bytes'],'approved HF capacity allocation insufficient')
    require(type(evidence.get('modal_available_bytes')) is int
            and evidence['modal_available_bytes']>=plan['projected_modal_bytes'],'available account capacity insufficient: modal_available_bytes')
    require(plan['projected_monthly_usd']<=30 and plan['projected_hf_bytes']<=300*1024**3
            and plan['projected_modal_bytes']<=200*1024**3,'project storage/recurring cap exceeded')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['inspect','create-publisher','create-repositories','create-working','initialize'])
    p.add_argument('--plan');p.add_argument('--env-file',default='.env');p.add_argument('--environment',default='main')
    a=p.parse_args()
    logging.disable(logging.CRITICAL);os.environ['HF_DEBUG']='0'
    from huggingface_hub import HfApi
    token=read_hf_token(Path(a.env_file));api=HfApi(token=token)
    info=api.whoami()
    require(info.get('name')==OWNER,'authenticated owner mismatch')
    if a.action=='inspect':
        print(json.dumps({'identity':OWNER,'is_pro':info.get('isPro'),'capacity_bytes':'unknown'}))
        for repo in REPOS:
            try:
                revision=identity(api,repo);api.auth_check(repo,write=True)
                print(json.dumps({'repo_id':repo,'private':True,'revision':revision,'write_access':'confirmed'}))
            except Exception as e:print(json.dumps({'repo_id':repo,'status':'unknown_or_denied','error_type':type(e).__name__}))
        return
    require(a.plan is not None,'approved plan required for mutations')
    plan=validate_plan(json.loads(Path(a.plan).read_text()),require_existing=a.action=='initialize')
    require(plan.get('existing_token_scope_exception') is True,'explicit existing-token exception absent')
    if a.action=='create-publisher':
        import modal
        modal.Secret.objects.create('agent-lightning-publisher',{'HF_TOKEN':token},allow_existing=False,environment_name=a.environment)
        print('Created agent-lightning-publisher; original training Secret unchanged')
    elif a.action=='create-repositories':
        capacity(plan)
        from huggingface_hub.errors import RepositoryNotFoundError
        for repo in REPOS:
            try:
                revision=identity(api,repo)
                require(set(api.list_repo_files(repo,revision=revision))<={'.gitattributes'},
                        'nonempty existing repository requires project reconciliation')
            except RepositoryNotFoundError:
                # Inaccessible is unknown, not unused: exclusive creation may fail,
                # and that failure never falls back to overwriting existing content.
                api.create_repo(repo_id=repo,repo_type='model',private=True,exist_ok=False)
            identity(api,repo);api.auth_check(repo,write=True)
        print('Created and checked the two approved private repositories')
    elif a.action=='create-working':
        capacity(plan)
        import modal
        require(plan['working_volume_name'] not in {'cs329z-agent-lightning','agent-lightning-validation'},'original Volume protected')
        modal.Volume.objects.create(plan['working_volume_name'],environment_name=a.environment,allow_existing=False)
        print('Created isolated working Volume')
    elif a.action=='initialize':
        capacity(plan)
        import subprocess
        head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip()
        require(head==plan['execution_commit'] and not dirty,'integrated execution checkout is not exact/clean')
        for repo in REPOS:identity(api,repo);api.auth_check(repo,write=True)
        HubLedger(api,REPOS[0],plan['ledger_path']).initialize(plan)
        print(json.dumps({'status':'initialized_not_launched','ledger_path':plan['ledger_path']}))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        # Never print SDK exceptions, raw account responses, URLs or credentials.
        raise SystemExit('preflight failed: '+type(exc).__name__)
