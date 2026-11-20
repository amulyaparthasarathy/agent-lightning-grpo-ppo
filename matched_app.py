"""Separately launched matched study. Import never submits jobs or creates Secrets."""
from __future__ import annotations
import os
from pathlib import Path

import modal
# Explicit existing Stanford binding, never the protected source Volume.
if os.environ.get('AGL_WORKING_VOLUME', 'cs329z-agent-lightning-working-stanford-v2') != 'cs329z-agent-lightning-working-stanford-v2':
    raise ValueError('Matched studies require the existing Stanford v2 working Volume')
os.environ['AGL_WORKING_VOLUME'] = 'cs329z-agent-lightning-working-stanford-v2'
# Runtime imports must work without the local application or build scripts.
import sys
BUNDLE = Path('/opt/agent-lightning-lab/experiment')
if not (Path(__file__).resolve().parent/'experiment').is_dir():
    sys.path.insert(0, str(BUNDLE.parent))
from experiment import matched_contract as c

app = modal.App('cs329z-matched-study-v2')
image = None
if modal.is_local():
    from modal_image import matched_image
    image = matched_image()
credentials = modal.Secret.from_name('agent-lightning-credentials', required_keys=['HF_TOKEN', 'WANDB_API_KEY'])
volume = modal.Volume.from_name(c.VOLUME, environment_name='main', create_if_missing=False)
registry = modal.Dict.from_name('cs329z-matched-study-v2', environment_name='main', create_if_missing=True)


@app.function(image=image, gpu='H100!', cpu=16, memory=196608, timeout=14400,
              retries=0, max_containers=10, single_use_containers=True,
              volumes={'/data': volume}, secrets=[credentials])
def stage_worker(request, deadline_epoch=None):
    from experiment import matched_remote
    return matched_remote.stage_worker(request, deadline_epoch, c=c, registry=registry,
                                       volume=volume, call_id=modal.current_function_call_id())


@app.function(image=image, cpu=1, memory=2048, timeout=43800, retries=0,
              max_containers=1, single_use_containers=True, volumes={'/data': volume})
def supervise(plan):
    from experiment import matched_remote
    return matched_remote.supervise(plan, c=c, registry=registry, volume=volume,
                                    modal=modal, stage_worker=stage_worker)
