"""Sanitized import-only probe. No tensor loading, network calls or env dumps."""
import json
import re
import subprocess


def failure(exc):
    result = {'error_type': type(exc).__name__}
    name = getattr(exc, 'name', None)
    if (isinstance(name, str) and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.]*', name)
            and name.split('.')[0] in {'scripts','experiment','huggingface_hub','httpx','httpx2'}):
        result['module_name'] = name
    return result  # Never serialize exception messages, traceback locals or argv.


def inspect_imports():
    import importlib
    import importlib.metadata
    import platform
    import sys
    result = {'python': sys.version.split()[0], 'architecture': platform.machine(), 'packages': {}, 'imports': {}}
    for name in ('huggingface-hub', 'httpx', 'httpx2', 'torch'):
        try:
            result['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result['packages'][name] = None
    for name in ('scripts.preservation_worker', 'scripts.preservation_http', 'huggingface_hub'):
        try:
            module = importlib.import_module(name)
            result['imports'][name] = {'imported': True, 'path': getattr(module, '__file__', None)}
        except Exception as exc:
            result['imports'][name] = failure(exc)
    try:
        from scripts.preservation_http import configure
        configure(1024 * 1024)  # Configure only; no HTTP request.
        result['transport_configuration'] = {'configured': True}
    except Exception as exc:
        result['transport_configuration'] = failure(exc)
    return result


def inspect_conversion(root='/work/restored', package_root=None):
    """Read only small restored metadata and installed merger sources; no tensors."""
    import hashlib
    import importlib.util
    import importlib.metadata
    from pathlib import Path
    root=Path(root)
    actor=root/'runs/train-01/checkpoints/global_step_20/actor'
    base=root/'models/llama3b'
    result={'kind':'conversion_metadata_diagnostic','files':{},'installed_merger':{}}
    for label,path in {'fsdp_config':actor/'fsdp_config.json', 'native_config':actor/'huggingface/config.json',
                       'native_generation':actor/'huggingface/generation_config.json',
                       'base_config':base/'config.json','base_generation':base/'generation_config.json',
                       'prepared_tokenizer_config':base/'tokenizer_config.json',
                       'prepared_manifest':base/'model_manifest.json'}.items():
        if path.stat().st_size>1024*1024:raise ValueError('metadata exceeds diagnostic limit')
        content=path.read_bytes()
        result['files'][label]={'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content),'json':json.loads(content)}
    package_root=Path(package_root) if package_root else Path(importlib.util.find_spec('verl').origin).parent
    result['verl_version']=importlib.metadata.version('verl')
    for name in ('fsdp_model_merger.py','base_model_merger.py'):
        path=package_root/'model_merger'/name
        content=path.read_bytes()
        result['installed_merger'][name]={'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)}
    worker=package_root/'workers/config/model.py'
    content=worker.read_bytes();lines=content.decode().splitlines()
    selected=set()
    for i,line in enumerate(lines):
        if any(key in line for key in ('eos_token_id','pad_token_id','bos_token_id')):
            selected.update(range(max(0,i-3),min(len(lines),i+4)))
    result['installed_worker']={'sha256':hashlib.sha256(content).hexdigest(),
                                'token_config_lines':{str(i+1):lines[i] for i in sorted(selected)}}
    result['output_exists']=(root/'exports/original').exists()
    return result


def inspect_evidence(root, names, workflow_id):
    """Diagnose only the selected small evaluation evidence, never checkpoint bytes."""
    from pathlib import Path
    import traceback
    from scripts.preservation_transfer import inventory
    root=Path(root)
    report={'logical_root':str(root),'resolved_root':str(root.resolve(strict=True)),
            'root_is_symlink':root.is_symlink(),'parent_is_symlink':root.parent.is_symlink()}
    selection={'selections':[{'group':'evidence','path':f'runs/{n}'} for n in names]
              +[{'group':'evidence','path':f'submissions/{n}'} for n in names]
              +[{'group':'evidence','path':f'evidence/{workflow_id}'}]}
    try:
        result=inventory({'evidence':root},{'evidence':selection},'diagnostic/'+workflow_id)['evidence']
        report.update(scan_succeeded=True,files=len(result['files']),bytes=sum(r['bytes'] for r in result['files']),
                      exclusions=result['exclusions'],missing_selections=result['missing_selections'])
    except Exception as exc:
        report.update(scan_succeeded=False,error_type=type(exc).__name__,
                      frames=[{'file':Path(f.filename).name,'function':f.name,'line':f.lineno}
                              for f in traceback.extract_tb(exc.__traceback__)])
        if type(exc) is ValueError and str(exc).startswith(('source root ','source directory mutation:',
                'unsupported file type or hardlink:','overlapping selection:','unsafe member path:')):
            report['diagnostic']=str(exc)
    return report


def probe(python, inspector='inspect_imports', arguments=None):
    # Absolute file loading keeps the probe usable even if a training package
    # shadows the generic scripts namespace that we are diagnosing.
    if inspector not in {'inspect_imports','inspect_conversion','inspect_evidence'}:raise ValueError('unknown inspector')
    code = f'import json,runpy; print(json.dumps(runpy.run_path({__file__!r})[{inspector!r}](**{arguments or {}!r})))'
    child = subprocess.run([str(python), '-c', code], capture_output=True, text=True, timeout=60)
    if child.returncode:
        return {'returncode': child.returncode, 'diagnostic': 'Probe bootstrap failed; raw stderr retained only in process memory'}
    return json.loads(child.stdout)
