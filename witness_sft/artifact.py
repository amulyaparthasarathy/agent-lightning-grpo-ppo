"""Terminal projection replacements, anchored by a separately collected training receipt."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
from pathlib import Path
import re

from experiment import reader_assets, reader_contract, reader_evidence
from . import data, training

CPU = 'tiny-cpu-fixture'
LIVE = 'pinned-base-production'
WEIGHTS = 'projections.safetensors'


def worker(call, container):
    if not isinstance(call, str) or not re.fullmatch(r'fc-[A-Za-z0-9]+', call):
        raise ValueError('Invalid provider call identity')
    return {'call': call, 'container': reader_evidence.container_identity(container)}


def identity(kind, source, producer, config):
    if not re.fullmatch('[0-9a-f]{40}', source) or kind not in (LIVE, CPU):
        raise ValueError('Wrong source or artifact class')
    if kind == LIVE:
        worker(producer['call'], producer['container'])
        if config != reader_assets.model_lock()['config.json']['sha256']:
            raise ValueError('Wrong production architecture')
    elif producer != {'fixture': 'cpu'}:
        raise ValueError('Wrong fixture producer')
    return {'schema': 'witness-sft-projections-v1', 'class': kind, 'producer': producer, 'source': source,
            'config_sha256': config, 'model': reader_contract.MODEL if kind == LIVE else 'local-random-LlamaForCausalLM',
            'revision': reader_contract.BASE_REVISION if kind == LIVE else None,
            'base_files_sha256': data.task.digest(reader_assets.model_lock()) if kind == LIVE else None,
            'tokenizer_sha256': reader_assets.model_lock()['tokenizer.json']['sha256'],
            'manifest_sha256': data.MANIFEST_SHA256, 'dataset_sha256': data.DATASET_SHA256,
            'recipe': training.RECIPE}


def validate_stats(stats):
    if (stats['examples'] != 32 or stats['updates'] != training.UPDATES
            or len(stats['preclip_norms']) != training.UPDATES or len(stats['example_losses']) != 32
            or stats['supervised_tokens'] != 2540
            or not all(math.isfinite(n) and n >= 0 for n in stats['preclip_norms'] + stats['example_losses'])):
        raise ValueError('Incomplete fixed SFT pass')


def save(path, model, binding, stats):
    from safetensors.torch import save_file
    validate_stats(stats)
    params = training.projections(model)
    if binding['class'] == LIVE and (next(model.parameters()).device.type != 'cuda' or model.config.num_hidden_layers != 28):
        raise ValueError('Production artifact requires the GPU base training path')
    path.mkdir(parents=True, exist_ok=False)
    tensors = {name: p.detach().cpu().contiguous() for name, p in params.items()}
    save_file(tensors, path / WEIGHTS)
    created = datetime.now(timezone.utc)
    manifest = {**binding, 'complete': True, 'stats': stats, 'created_utc': created.isoformat(),
                'retention_review_utc': (created + timedelta(days=7)).isoformat(),
                'file': WEIGHTS, 'bytes': (path / WEIGHTS).stat().st_size,
                'sha256': reader_contract.file_hash(path / WEIGHTS),
                'tensors': {name: {'shape': list(t.shape), 'dtype': str(t.dtype)} for name, t in tensors.items()}}
    # The completion record is written last. Callers receive its external commitment
    # only after the weights and manifest are closed.
    reader_assets.write_new(path / 'manifest.json', manifest)
    return data.task.digest(manifest)


def restore(path, model, binding, expected_digest, consumer):
    import torch
    from safetensors.torch import load_file
    if binding['class'] == LIVE:
        worker(consumer['call'], consumer['container'])
        if any(binding['producer'][k] == consumer[k] for k in ('call', 'container')):
            raise ValueError('Reload requires a distinct call and container')
    elif binding['class'] != CPU or consumer != {'fixture': 'cpu-reload'}:
        raise ValueError('Wrong artifact consumer')
    if path.is_symlink() or {p.name for p in path.iterdir() if p.is_file() and not p.is_symlink()} != {WEIGHTS, 'manifest.json'} or len(list(path.iterdir())) != 2:
        raise ValueError('Incomplete or unexpected artifact files')
    m = json.loads((path / 'manifest.json').read_text())
    if (data.task.digest(m) != expected_digest or any(m.get(k) != v for k, v in binding.items())
            or m['complete'] is not True or m['file'] != WEIGHTS or m['bytes'] != (path / WEIGHTS).stat().st_size
            or m['sha256'] != reader_contract.file_hash(path / WEIGHTS)):
        raise ValueError('Artifact identity or inventory differs')
    validate_stats(m['stats'])
    tensors = load_file(path / WEIGHTS)
    params = training.projections(model)
    if tensors.keys() != params.keys() or m['tensors'].keys() != params.keys():
        raise ValueError('Incomplete trainable weight inventory')
    for name, t in tensors.items():
        if (t.shape != params[name].shape or t.dtype != torch.float32 or not torch.isfinite(t).all()
                or m['tensors'][name] != {'shape': list(t.shape), 'dtype': str(t.dtype)}):
            raise ValueError('Projection shape, dtype or values differ')
    with torch.no_grad():
        for name, t in tensors.items():
            params[name].copy_(t)
    return model.eval()
