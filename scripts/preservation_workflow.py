"""Small durable phase ledger with optimistic Hub revision guards.

A claim is committed before spawning. A lost spawn acknowledgement is ambiguous,
never an invitation to launch twice. Workers report their own call IDs. No clock
expiry steals a claim and no automatic application retry is performed.
"""
from __future__ import annotations
import copy
import json
import time
from scripts.preservation_transfer import require, identity, read_small, put_record, commit_files, now
from scripts.backup_qualification import digest

LIMITS = {"cost_usd":100, "cpu_seconds":36000, "gpu_seconds":28800,
          "egress_bytes":500*1024**3, "hf_bytes":300*1024**3, "modal_bytes":200*1024**3,
          "monthly_usd":30}
GPU_PHASES = {"native_eval","export_eval","replication_baseline","replication_train","replication_native","replication_export_eval"}
SOURCE_STREAM_PHASES = {'inventory', 'backup'}
SOURCE_STREAM_RESOURCES = (2, 4)  # CPU cores, GiB; byte streams never load tensors.
ORDER = ("inventory", "backup", "verify_backup", "restore", "native_eval", "convert", "export_eval",
         "compare", "original_evidence_backup", "original_evidence_verify", "publish", "retrieve", "replication_baseline", "replication_train", "replication_native",
         "replication_compare", "replication_backup", "replication_verify", "replication_convert",
         "replication_export_eval", "replication_export_compare", "replication_evidence_backup", "replication_evidence_verify", "replication_publish", "replication_retrieve")


def validate_plan(plan, require_existing=True):
    require(plan.get("remote_execution_approved") == "YES", "remote execution not approved")
    require(plan.get("owner") == "amulyaparthasarathy", "unapproved owner")
    require(plan.get("limits") == LIMITS, "global limits differ from authorization")
    require(tuple(p["name"] for p in plan["phases"]) == ORDER, "phase order differs")
    require(plan.get("rights_review") and plan.get("capacity_evidence"), "rights/capacity preflight missing")
    require(plan.get("source_volume_id") and plan.get("validation_volume_id"), "explicit source Volume identities required")
    if require_existing:
        require(plan.get("working_volume_id"), "confirmed working Volume identity required")
        require(plan["working_volume_id"] not in {plan["source_volume_id"],plan["validation_volume_id"]}, "working/source overlap")
    for p in plan["phases"]:
        r=p["reservation"]
        require(set(r)==set(LIMITS) and all(type(v) in (int,float) and v>=0 for v in r.values()), "incomplete resource reservation")
        require(0<p["timeout_seconds"]<=11400, "invalid bounded phase timeout")
        require(p['kind'] == ('gpu' if p['name'] in GPU_PHASES else 'cpu'), 'phase worker kind differs')
        gpu=p["kind"]=="gpu"
        permitted = {(16,128)} if gpu else {(8,128)}
        if p['name'] in SOURCE_STREAM_PHASES:
            permitted.add(SOURCE_STREAM_RESOURCES)  # Retain readable historical 8/128 plans.
        require((p['cpu'],p['memory_gib']) in permitted, "worker resource cap")
        require(not gpu or p["memory_gib"]==128, "qualified GPU recipe changed")
        require(r["gpu_seconds" if gpu else "cpu_seconds"]>=p["timeout_seconds"]+p.get("startup_reserve_seconds",600), "timeout unreserved")
    return plan


def reserve(ledger, phase):
    r=phase["reservation"]
    # Reservations are maxima, including startup, failures and network allowances.
    totals={k:ledger["accounting"].get(k,0)+r[k] for k in LIMITS}
    for key,limit in LIMITS.items():
        require(totals[key]<=limit, "budget reservation exceeds "+key)
    return totals


def new_ledger(plan):
    validate_plan(plan)
    initial=plan.get("initial_accounting",{})
    require(set(initial)==set(LIMITS), "build/preflight accounting must be explicit")
    require(all(0<=initial[k]<=LIMITS[k] for k in LIMITS), "initial budget exhausted")
    return {"schema_version":1,"plan_sha256":digest(plan),"plan":plan,"status":"ready","phases":{},
            "accounting":initial,"created_at":now()}


def persist_before_publish(record, fallback, publish):
    """Keep an independent terminal copy even when the Hub control write fails."""
    fallback(record)
    return publish(record)


def observed_usage(phase, elapsed):
    """Conservative rate estimate, not a provider bill; no allowance credits."""
    require(elapsed>=0,'invalid elapsed time')
    usage=dict.fromkeys(LIMITS,None)
    seconds=elapsed+phase.get('startup_reserve_seconds',600)
    gpu=phase['kind']=='gpu'
    usage['gpu_seconds' if gpu else 'cpu_seconds']=seconds
    rate=(0.001097 if gpu else 0)+phase['cpu']*0.0000131+phase['memory_gib']*0.00000222
    usage['cost_usd']=(seconds*rate+phase['reservation']['egress_bytes']/1024**3*0.04)*1.2
    return usage


class HubLedger:
    def __init__(self, api, repo, path):
        self.api,self.repo,self.path=api,repo,path

    def load(self):
        revision=identity(self.api,self.repo)
        return revision,read_small(self.api,self.repo,revision,self.path)

    def save(self, revision, record):
        # No retry of an optimistic conflict. Read the current ledger to reconcile.
        return put_record(self.api,self.repo,revision,self.path,record)

    def initialize(self, plan):
        revision=identity(self.api,self.repo)
        files=self.api.list_repo_files(self.repo,revision=revision)
        claim='project-preservation-claim.json'
        require(self.path not in files and claim not in files, "project workflow already exists; reconnect to its ledger")
        record=new_ledger(plan)
        return commit_files(self.api,self.repo,revision,[
            (claim,json.dumps({'ledger_path':self.path,'plan_sha256':record['plan_sha256']}).encode()),
            (self.path,json.dumps(record,sort_keys=True,allow_nan=False).encode())],'Claim one preservation workflow')

    def claim(self):
        revision,ledger=self.load()
        validate_plan(ledger["plan"])
        require(digest(ledger["plan"])==ledger["plan_sha256"], "approved plan changed")
        for phase in ledger["plan"]["phases"]:
            previous=ledger["phases"].get(phase["name"])
            if previous and previous["status"]=="passed": continue
            require(previous is None, "phase already claimed/failed; reconcile, never duplicate")
            totals=reserve(ledger,phase)
            ledger["accounting"]=totals
            ledger["phases"][phase["name"]]={"status":"launching","claimed_at":now(),"reservation":phase["reservation"]}
            ledger["status"]="running"
            self.save(revision,ledger)
            return phase,ledger
        return None,ledger

    def recover(self, name, diagnosis, previous_call_terminal):
        """Explicit diagnosed recovery only, never a timeout-based claim takeover."""
        revision,ledger=self.load()
        old=ledger['phases'][name]
        require(old['status']=='failed' and previous_call_terminal is True,'unresolved or active invocation')
        require(bool(diagnosis) and diagnosis not in [x['diagnosis'] for x in old.get('recoveries',[])],
                'missing/repeated identical failure diagnosis')
        require(old.get('attempt',0)<2,'phase retry limit exhausted')
        if name in GPU_PHASES:
            # One narrowly diagnosed bootstrap recovery, not a scientific rerun.
            # Read the independently executed CPU diagnostic from its immutable
            # receipt; caller-supplied booleans cannot waive this gate.
            require(name=='native_eval' and old.get('attempt',0)==0
                    and diagnosis=='client-orchestration-before-native-launch',
                    'no automatic scientific-run retries')
            claim=ledger.get('diagnostics',{}).get('native-eval-startup-0',{})
            require(claim.get('status')=='observed' and claim.get('prior_native_call')==old.get('function_call_id'),
                    'startup diagnostic is absent or belongs to another call')
            ref=claim['result'];report=read_small(self.api,self.repo,ref['revision'],ref['receipt_path'])
            require(digest(report)==claim.get('report_sha256') and report.get('probe_sha256')==claim.get('probe_sha256')
                    and report.get('function_call_id')==claim.get('function_call_id'), 'startup diagnostic identity differs')
            require(report.get('kind')=='import_only_diagnostic' and report.get('native_run_exists') is False
                    and report.get('native_submission_exists') is False, 'scientific output exists or absence unverified')
            require(report.get('client',{}).get('transport_configuration',{}).get('configured') is True
                    and report.get('training',{}).get('imports',{}).get('scripts.preservation_worker',{}).get('module_name')
                    =='scripts.preservation_worker', 'bootstrap import diagnosis differs')
            failed=old['result'];failure=read_small(self.api,self.repo,failed['revision'],failed['receipt_path'])
            require(failure.get('error_type')=='ChildDidNotReturn','failure was not pre-result bootstrap')
        phase=next(p for p in ledger['plan']['phases'] if p['name']==name)
        incremental=copy.deepcopy(phase['reservation'])
        if name in {'backup','replication_backup','original_evidence_backup','replication_evidence_backup'}:
            # Same immutable source manifest and snapshot prefix; readback validates
            # every reused part. Retained content was already reserved in attempt 0.
            incremental['hf_bytes']=0
        ledger['accounting']=reserve(ledger,dict(phase,reservation=incremental))
        history=old.get('recoveries',[])+[{'diagnosis':diagnosis,'prior':{k:v for k,v in old.items() if k!='recoveries'}}]
        ledger['phases'][name]={'status':'launching','reservation':incremental,'attempt':old.get('attempt',0)+1,
                                'recoveries':history,'claimed_at':now()}
        ledger['status']='running'
        self.save(revision,ledger)
        return phase,ledger

    def started(self, name, call_id):
        revision,ledger=self.load()
        phase=ledger["phases"][name]
        require(phase["status"]=="launching", "worker already started; duplicate refused")
        phase.update(status="running",function_call_id=call_id,started_at=now(),
                     execution_commit=ledger['plan']['execution_commit'])
        self.save(revision,ledger)

    def terminal(self, name, call_id, result, usage, success):
        revision,ledger=self.load()
        phase=ledger["phases"][name]
        require(phase["status"]=="running" and phase["function_call_id"]==call_id, "terminal receipt identity differs")
        require(set(usage)==set(LIMITS), "resource accounting incomplete")
        # Unknown usage keeps the full reservation, never assumes zero after failure.
        for key in LIMITS:
            used=usage[key] if usage[key] is not None else phase["reservation"][key]
            require(used>=0, "invalid usage")
            ledger["accounting"][key]+=used-phase["reservation"][key]
        if any(ledger["accounting"][k]>LIMITS[k] for k in LIMITS): success=False
        phase.update(status="passed" if success else "failed",result=result,usage=usage,finished_at=now())
        ledger["status"]="ready" if success else "stopped"
        if success and name==ORDER[-1]: ledger["status"]="complete"
        self.save(revision,ledger)
        return ledger
