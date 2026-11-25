"""Run with PYTHONPATH pointing to the failed A01 checkout and its tests directory."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from experiment import matched_launch, reader_assets, recovery_learning, request_transport
from recovery_learning_support import request


def reproduce():
    req = request('baseline')
    with TemporaryDirectory() as tmp:
        base = Path(tmp)
        physical = base/'physical-volume'
        physical.mkdir()
        mount = base/'data'
        mount.symlink_to(physical, target_is_directory=True)
        run = reader_assets.owned_path(mount, req['stage']['output'])
        run.mkdir(parents=True)
        file = run/'request.json'
        file.write_text(json.dumps(req))
        env = dict(request_transport.environment(req,run), AGL_PPO_RUN_DIR=str(run))
        loaded = request_transport.load(env)
        model = Path('/data')/req['plan']['profile']['paths']['model']
        data = Path('/data')/req['plan']['profile']['paths']['data']
        rp, _ = matched_launch.make_plan(loaded,base,base,model,data,base/'python',file.parent,12345)
        cfg = {'actor_rollout_ref':{'model':{'path':str(model)}}, 'agentlightning':{'async_rollout':{'enabled':False}}}
        for override in rp['overrides']:
            key, value = override.split('=', 1)
            node = cfg
            for part in key.split('.')[:-1]:
                node = node.setdefault(part,{})
            node[key.split('.')[-1]] = json.loads(value)
        matched_launch.validate_resolved(cfg,rp)
        (run/'resolved_config.json').write_text(json.dumps(cfg))
        (run/'qualification-inputs.json').write_text(json.dumps(recovery_learning.stage_spec(req)['inputs']))
        (run/'audit').mkdir()
        event = {'event':'wandb_finished', 'time':1, 'id':req['stage']['wandb']['id']}
        (run/'audit/events-1.jsonl').write_text(json.dumps(event)+'\n')
        matched_launch.complete_stage(run,loaded,data)


if __name__ == '__main__':
    reproduce()
