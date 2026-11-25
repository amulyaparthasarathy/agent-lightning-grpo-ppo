"""Explicit local CPU test preparation: download pinned code, never model assets."""
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
lock=json.loads((ROOT/'experiment/ppo_source_lock.json').read_text())
dest=ROOT/'.local/ppo-review'
dest.mkdir(parents=True,exist_ok=True)
agl=dest/'agent-lightning'
if not agl.exists():
    subprocess.run(['git','clone','--no-checkout','https://github.com/microsoft/agent-lightning.git',str(agl)],check=True)
    subprocess.run(['git','-C',str(agl),'checkout','--detach',lock['agentlightning_commit']],check=True)
wheel=dest/lock['verl_wheel']['filename']
if not wheel.exists():
    with urllib.request.urlopen(lock['verl_wheel']['url'],timeout=120) as response, wheel.open('xb') as out:
        while chunk:=response.read(1024*1024):out.write(chunk)
if hashlib.sha256(wheel.read_bytes()).hexdigest()!=lock['verl_wheel']['digests']['sha256']:
    raise ValueError('VERL wheel digest mismatch')
if not (dest/'verl/verl').exists():
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            path=Path(member.filename)
            if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe source archive path')
            if member.filename.startswith('verl/'):
                archive.extract(member,dest/'verl')
ipc=json.loads((ROOT/'tests/pinned_ipc_lock.json').read_text())
vllm=dest/ipc['filename']
if not vllm.exists():
    with urllib.request.urlopen(ipc['url'],timeout=120) as response, vllm.open('xb') as out:
        while chunk:=response.read(1024*1024):out.write(chunk)
with vllm.open('rb') as handle:
    if hashlib.file_digest(handle,'sha256').hexdigest()!=ipc['sha256']:
        raise ValueError('vLLM wheel digest mismatch')
print('Prepared pinned sources (vLLM wheel is not installed); tests verify hashes before execution.')

# Only this pure-Torch indexing module is needed to execute the VERL unpadding path on CPU.
padding = ROOT/'.local/bert_padding.py'
if not padding.exists():
    with urllib.request.urlopen('https://raw.githubusercontent.com/Dao-AILab/flash-attention/v2.8.3/flash_attn/bert_padding.py', timeout=60) as response:
        body = response.read()
    if hashlib.sha256(body).hexdigest() != '805d449ac749f87a50f3a3114385710f0f926ff45521275000b74d9e80721e5c':
        raise ValueError('FlashAttention padding source differs')
    padding.write_bytes(body)
