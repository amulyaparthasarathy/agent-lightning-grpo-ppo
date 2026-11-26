"""Single-stage training-image child process. Input is verified by the owner wrapper."""
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile

from experiment import reader_assets
from . import artifact, contract, data, evaluation, training


def execute(request, run, base):
    plan, stage, previous, worker = (request[k] for k in ('plan', 'stage', 'previous', 'worker'))
    contract.authorize(plan, request['owner'], request['permit'])
    contract.check_stage(plan, stage, previous)
    artifact.worker(**worker)
    if any(any(r['worker'][k] == worker[k] for k in ('call', 'container')) for r in previous):
        raise ValueError('A fresh provider worker is required')
    from importlib.metadata import version
    import torch
    from transformers import LlamaForCausalLM
    if (version('transformers') != '5.22.2' or version('tokenizers') != '0.23.3'
            or torch.__version__.split('+')[0] != '2.11.0' or not torch.cuda.is_bf16_supported()):
        raise ValueError('Incompatible training stack or accelerator')
    reader_assets.verify_model(base)
    if (base / 'adapter_config.json').exists():
        raise ValueError('Unexpected adapter on original base')
    if stage == 'fresh-base':
        return evaluation.evaluate(base, base, run, data.task.digest(plan))
    model = LlamaForCausalLM.from_pretrained(base, dtype=torch.bfloat16, attn_implementation='sdpa',
                                             local_files_only=True, use_safetensors=True)
    producer = worker if stage == 'supervised-pass' else previous[-1]['worker']
    binding = artifact.identity(artifact.LIVE, plan['source'], producer, reader_assets.model_lock()['config.json']['sha256'])
    path = run.parent / 'supervised-pass' / 'artifact'
    if stage == 'supervised-pass':
        device = torch.device('cuda:0')
        stats = training.fit(model.to(device), training.tokenizer(base / 'tokenizer.json'), device)
        digest = artifact.save(path, model, binding, stats, request['retention_review_utc'])
        return {'artifact_sha256': digest, 'identity': binding, 'stats': stats,
                'peak_allocated': torch.cuda.max_memory_allocated(), 'peak_reserved': torch.cuda.max_memory_reserved()}
    if stage != 'terminal-reload':
        raise ValueError('Unknown SFT stage')
    digest = previous[-1]['result']['artifact_sha256']
    artifact.restore(path, model, binding, digest, worker)
    with tempfile.TemporaryDirectory() as scratch:
        merged = Path(scratch)
        model.save_pretrained(merged)
        del model
        # Serve using the original architecture and decoding metadata and the
        # original tokenizer. Only the two projection tensors have been replaced.
        for filename in ('config.json', 'generation_config.json'):
            shutil.copyfile(base / filename, merged / filename)
        return {'loaded_artifact_sha256': digest, **evaluation.evaluate(merged, base, run, data.task.digest(plan))}


if __name__ == '__main__':
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    run = Path(sys.argv[1])
    request = json.loads((run / 'request.json').read_text())
    result = execute(request, run, Path(os.environ['WITNESS_BASE']))
    reader_assets.write_new(run / 'result.json', result)
