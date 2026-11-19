"""Load hash-verified upstream code, without importing the CUDA/Ray model stack.

Only imports/decorators and accelerator model construction are replaced. Function
bodies for adapter, padding, losses, minibatch iteration, backward and optimizer
are executed verbatim. No equations are reimplemented in this loader.
"""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SOURCES = Path(os.environ.get('AGL_PINNED_SOURCES', ROOT/'.local/ppo-review'))
ROOTS = {'agentlightning': SOURCES/'agent-lightning/agentlightning', 'verl': SOURCES/'verl/verl'}
LOCK = json.loads((ROOT/'experiment/ppo_source_lock.json').read_text())['files']
for package, files in json.loads((ROOT/'experiment/matched_source_lock.json').read_text()).items():
    LOCK[package].update(files)


def source(package, relative):
    path = ROOTS[package]/relative
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != LOCK[package][relative]:
        raise AssertionError(f'Unreviewed pinned test source: {path}')
    return path, data.decode()


def load(package, relative, namespace, names=None, class_name=None):
    path, text = source(package, relative)
    tree = ast.parse(text)
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name==class_name).body
    if names:
        nodes = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
        if len(nodes) != len(names):
            raise AssertionError(f'Functions missing in {path}: {names}')
    else:
        nodes = [n for n in nodes if not isinstance(n, (ast.Import, ast.ImportFrom))]
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.decorator_list = []
    tree.body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + nodes
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(path), 'exec'), namespace)
    return SimpleNamespace(**namespace)


class Proto:
    """CPU transport seam only. TensorDict storage/slicing remains real."""
    def __init__(self, batch=None, non_tensor_batch=None, meta_info=None):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch or {}
        self.meta_info = meta_info or {}
    def __len__(self):
        return len(self.batch)
    def __getitem__(self, indices):
        return Proto(self.batch[indices], {k:v[indices] for k,v in self.non_tensor_batch.items()}, self.meta_info.copy())
    def to_tensordict(self):
        return self.batch.clone()
    @classmethod
    def from_single_dict(cls, data, meta_info=None):
        return cls(data, meta_info=meta_info)
