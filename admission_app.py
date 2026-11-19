"""Separately authorized admission measurement: no model, CUDA build, Secret or Volume."""
import asyncio
import json
import time
import subprocess
import os
import shutil
from pathlib import Path
import modal

app = modal.App('cs329z-tenway-admission')
image = modal.Image.debian_slim(python_version='3.12')
registry = modal.Dict.from_name('cs329z-tenway-admission', environment_name='main', create_if_missing=True)
RESOURCES = {'gpu':'H100!', 'cpu':16, 'memory_mib':196608}


@app.function(image=image, gpu='H100!', cpu=16, memory=196608, timeout=300,
              retries=0, max_containers=10, single_use_containers=True)
def worker(plan, slot):
    key = plan['probe_id']
    call = modal.current_function_call_id()
    record = {'slot':slot, 'call_id':call, 'started_at':time.time(), 'status':'started'}
    if not registry.put(f'{key}:entered:{slot}', record, skip_if_exists=True):
        raise ValueError('Probe worker restart forbidden')
    try:
        record['resources_requested'] = RESOURCES
        record['scratch_free_bytes'] = shutil.disk_usage('/tmp').free
        record['logical_cpus_visible'] = os.cpu_count()
        record['cgroup'] = {name:Path('/sys/fs/cgroup',name).read_text().strip()
                            for name in ('cpu.max','memory.max') if Path('/sys/fs/cgroup',name).is_file()}
        record['gpu'] = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total',
            '--format=csv,noheader'], text=True, timeout=15).strip()
        if 'H100' not in record['gpu'] or len(record['gpu'].splitlines()) != 1:
            raise ValueError('Expected exactly one H100')
        while time.time()-record['started_at'] < 270:
            record['heartbeat_at'] = time.time()
            registry.put(f'{key}:worker:{slot}', record)
            if registry.get(f'{key}:release') or registry.get(f'{key}:stop'):
                break
            time.sleep(1)
        record['status'] = 'completed'
    except BaseException as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        record['finished_at'] = time.time()
        registry.put(f'{key}:worker:{slot}', record)
    return record


@app.function(image=image, cpu=1, memory=2048, timeout=420, retries=0, max_containers=1)
def supervise(plan):
    key = plan['probe_id']; start = time.time()
    if not registry.put(key+':owner', {'call_id':modal.current_function_call_id()}, skip_if_exists=True):
        raise ValueError('Probe already owned')
    report = {'probe_id':key, 'source_commit':plan['source_commit'], 'resources':RESOURCES,
              'status':'started', 'started_at':start, 'calls':{}, 'workers':{},
              'max_simultaneous':0, 'samples':[], 'billing_reconciled':False}
    def save():
        registry.put(key+':report', report)
    async def submit(slot):
        report['calls'][str(slot)] = {'status':'submitting', 'submitted_at':time.time()}
        save()
        try:
            call = await asyncio.wait_for(worker.spawn.aio(plan, slot), timeout=30)
            report['calls'][str(slot)].update(status='submitted', call_id=call.object_id)
        except Exception as exc:
            report['calls'][str(slot)].update(status='unknown', error=str(exc))
        save()
    async def submit_all():
        await asyncio.gather(*(submit(slot) for slot in range(10)))
    try:
        save(); asyncio.run(submit_all())
        if any(c['status']=='unknown' for c in report['calls'].values()):
            raise RuntimeError('Ambiguous submission; no retry')
        overlap_start = None
        while time.time()-start < 180:
            now = time.time(); alive = []
            for slot in range(10):
                record = registry.get(f'{key}:worker:{slot}')
                if record:
                    report['workers'][str(slot)] = record
                    if record['status']=='started' and now-record.get('heartbeat_at',0) <= 3:
                        alive.append(slot)
            report['samples'].append({'time':now, 'alive_slots':alive})
            report['max_simultaneous'] = max(report['max_simultaneous'],len(alive))
            save()
            if registry.get(key+':stop'):
                raise RuntimeError('Owner cancellation')
            if len(alive)==10:
                overlap_start = overlap_start or now
                if now-overlap_start >= 30:
                    break
            else:
                overlap_start = None
            time.sleep(1)
        registry.put(key+':release',True)
        # Receipts, not submission counts, establish simultaneous GPU intervals.
        for slot, call in report['calls'].items():
            record = modal.FunctionCall.from_id(call['call_id']).get(timeout=max(0,240-(time.time()-start)))
            if record['call_id'] != call['call_id'] or record['status'] != 'completed':
                raise ValueError('Missing worker completion')
            record['startup_latency_seconds'] = record['started_at']-call['submitted_at']
            report['workers'][slot] = record
        rows = list(report['workers'].values())
        events = sorted([(r['started_at'],1) for r in rows]+[(r['finished_at'],-1) for r in rows])
        current = peak = 0
        for _, delta in events:
            current += delta; peak = max(peak,current)
        report['interval_max_simultaneous'] = peak
        report['overlap_seconds'] = max(0,min(r['finished_at'] for r in rows)-max(r['started_at'] for r in rows))
        unique_gpu = len({r['gpu'].split(',')[0] for r in rows})==10
        report['status'] = 'completed' if peak==10 and unique_gpu and report['overlap_seconds']>=30 else 'failed'
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    finally:
        registry.put(key+':release',True)
        for slot in range(10):
            call = report['calls'].get(str(slot),{}).get('call_id')
            # Recover IDs recorded by workers when spawn responses were lost.
            entered = registry.get(f'{key}:entered:{slot}') or {}
            call = call or entered.get('call_id')
            if call:
                try:
                    modal.FunctionCall.from_id(call).cancel(terminate_containers=True)
                except Exception as exc:
                    report.setdefault('cancellation_errors',[]).append({'call_id':call,'error':str(exc)})
        report['finished_at'] = time.time()
        report['measured_gpu_seconds'] = sum(r.get('finished_at',time.time())-r['started_at'] for r in report['workers'].values())
        report['cost_upper_bound_usd'] = plan['budget_upper_bound_usd']
        report['billing_note'] = 'Measured function intervals exclude provisioning; reconcile provider usage before campaign.'
        save()
    return report
