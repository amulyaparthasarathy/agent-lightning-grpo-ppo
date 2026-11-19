"""Cold builder discovery through Modal 1.6.3, without RPC or installation."""
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from modal._utils.function_utils import FunctionSourceInfo
import modal_app

ROOT = Path(__file__).resolve().parents[1]


class ColdBuildImportTests(unittest.TestCase):
    def test_builder_discovery_without_application_or_experiment(self):
        for name, installer in (
            ('build_training_stack', 'install_modal_training.sh'),
            ('finalize_training_stack', 'finalize_modal_training.sh'),
        ):
            with self.subTest(builder=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                # The recipe explicitly copies this module before both steps.
                # No modal_app, experiment, scripts or repository on sys.path.
                shutil.copyfile(ROOT/'modal_build.py', root/'modal_build.py')
                function = getattr(modal_app, name)
                info = FunctionSourceInfo(function)
                self.assertEqual(info.module_name, 'modal_build')
                self.assertFalse(info.is_serialized())
                script = r'''
import builtins
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

def deny_network(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo', 'socket.sendto'}:
        raise AssertionError('Network access forbidden during offline discovery')

sys.addaudithook(deny_network)
from modal._runtime.user_code_imports import import_single_function_service
from modal_proto import api_pb2
import modal

root, module_name, function_name, installer = sys.argv[1:]
sys.path.insert(0, root)
assert importlib.util.find_spec('experiment') is None
assert importlib.util.find_spec('modal_app') is None
assert not (Path(root)/'scripts').exists()
original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'modal_app', 'experiment', 'ppo_contract', 'ppo_runtime', 'ppo_math'}:
        raise AssertionError('Forbidden runtime dependency: ' + name)
    return original_import(name, *args, **kwargs)

with (patch('builtins.__import__', side_effect=guarded_import),
      patch.object(modal.Volume, 'from_name', side_effect=AssertionError('Volume dependency')),
      patch.object(modal.Secret, 'from_name', side_effect=AssertionError('Secret dependency'))):
    service = import_single_function_service(
        api_pb2.Function(module_name=module_name, function_name=function_name,
                         implementation_name=function_name, is_builder_function=True), None)
    function = service._user_defined_callable
    assert function.__module__ == 'modal_build'
    assert function.__name__ == function_name
    # Fail on any non-stdlib import made by the function, even an indirect one.
    def stdlib_only(name, *args, **kwargs):
        assert name.split('.')[0] in sys.stdlib_module_names, name
        return original_import(name, *args, **kwargs)
    with patch('builtins.__import__', side_effect=stdlib_only), patch('subprocess.run') as run:
        function()
        run.assert_called_once_with(['bash', '/opt/agl-build/' + installer], check=True)
    # Shell errors must still fail the build; never execute a real installer.
    failure = subprocess.CalledProcessError(1, ['bash', installer])
    with patch('subprocess.run', side_effect=failure):
        try:
            function()
        except subprocess.CalledProcessError as error:
            assert error is failure
        else:
            raise AssertionError('Installer failure swallowed')
assert 'modal_app' not in sys.modules
assert not any(name == 'experiment' or name.startswith('experiment.') for name in sys.modules)
print(json.dumps({'builder': function_name, 'module': module_name,
                  'runtime_package': 'absent', 'network': 'blocked', 'installer': 'mocked'}))
'''
                env = {k: v for k, v in os.environ.items()
                       if not k.startswith(('MODAL_', 'HF_', 'WANDB_', 'AGL_', 'PYTHON'))}
                env.update(MODAL_CONFIG_PATH='/dev/null', HF_HUB_OFFLINE='1')
                result = subprocess.run(
                    [sys.executable, '-I', '-c', script, str(root), info.module_name,
                     info.implementation_name, installer],
                    cwd=root, env=env, text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout), {
                    'builder': name, 'module': 'modal_build', 'runtime_package': 'absent',
                    'network': 'blocked', 'installer': 'mocked'})

    def test_build_module_has_only_stdlib_entrypoints(self):
        tree = ast.parse((ROOT/'modal_build.py').read_text())
        self.assertIsInstance(tree.body[0], ast.Expr)  # module docstring
        self.assertEqual([node.name for node in tree.body[1:]],
                         ['build_training_stack', 'finalize_training_stack'])
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertEqual([alias.name for alias in node.names], ['subprocess'])
            self.assertNotIsInstance(node, ast.ImportFrom)

    def test_build_module_is_copied_before_builders_and_runtime_after(self):
        tree = ast.parse((ROOT/'modal_app.py').read_text())
        recipe = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'image' for t in node.targets))
        calls = []
        while isinstance(recipe, ast.Call) and isinstance(recipe.func, ast.Attribute):
            calls.append(recipe)
            recipe = recipe.func.value
        calls.reverse()
        copy_index = next(i for i, call in enumerate(calls)
                          if call.func.attr == 'add_local_file'
                          and ast.literal_eval(call.args[1]) == '/root/modal_build.py')
        module_copy = calls[copy_index]
        self.assertEqual(ast.unparse(module_copy.args[0]), "LOCAL / 'modal_build.py'")
        self.assertTrue(ast.literal_eval(module_copy.keywords[0].value))
        builders = [(i, call) for i, call in enumerate(calls) if call.func.attr == 'run_function']
        self.assertEqual([call.args[0].id for _, call in builders],
                         ['build_training_stack', 'finalize_training_stack'])
        for (index, call), resources in zip(builders, (
            dict(cpu=8, memory=65536, timeout=7200, include_source=False),
            dict(cpu=4, memory=16384, timeout=1800, include_source=False),
        )):
            self.assertLess(copy_index, index)
            # Explicit source only; no implicit app imports, GPU, Volume or Secret.
            self.assertEqual({kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}, resources)
        runtime_copy = next(node for node in tree.body if isinstance(node, ast.For)
                            and isinstance(node.iter, ast.Name) and node.iter.id == 'RUNTIME_FILES')
        self.assertGreater(runtime_copy.lineno, builders[-1][1].end_lineno)
