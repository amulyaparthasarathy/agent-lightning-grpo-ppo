"""Bounded real local MCP probe; no inference or training qualification."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
try:
    from . import asset_task as task
    from .owned_processes import cleanup
except ImportError:
    import asset_task as task
    from owned_processes import cleanup


async def probe():
    from autogen_ext.tools.mcp import McpWorkbench
    try:
        from .asset_agent import server_params
    except ImportError:
        from asset_agent import server_params
    checks = []
    async with McpWorkbench(server_params()) as workbench:
        tools = await workbench.list_tools()
        if len(tools) != 1 or tools[0]['name'] != 'lookup_asset' or tools[0]['parameters'] != task.TOOL_SCHEMA:
            raise ValueError('MCP schema differs')
        for args, error in (({'asset_id': 'asset-047'}, False), ({}, True),
                            ({'asset_id': 47}, True), ({'asset_id': 'asset-047', 'extra': 1}, True),
                            ({'asset_id': '../047'}, True), ({'asset_id': 'asset-999'}, True)):
            result = await workbench.call_tool('lookup_asset', args)
            valid = result.is_error == error
            if not error:
                valid = valid and json.loads(result.to_text()) == task.lookup(args)
            checks.append({'arguments': args, 'expected_error': error, 'valid': valid})
    return {'schema': 'asset-local-probe-v1', 'task': task.TASK, 'environment_sha256': task.digest(task.environment()),
            'transport': 'real stdio MCP', 'checks': checks, 'status': 'passed' if all(c['valid'] for c in checks) else 'failed',
            'workbench_context_exited': True, 'model_inference': 'not executed', 'live_qualification': False}


def check(output, timeout=90):
    with Path(output).open('x') as out, tempfile.TemporaryDirectory(prefix='asset-probe-') as owners:
        env = dict(os.environ, AGL_MCP_OWNERS=owners)
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker'],
                                env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            stdout, _ = proc.communicate(timeout=timeout)
            result = json.loads(stdout) if proc.returncode == 0 else {'status': 'failed', 'reason': 'MCP worker failed'}
        except (subprocess.TimeoutExpired, ValueError):
            result = {'status': 'failed', 'reason': 'MCP worker timeout or invalid receipt'}
        finally:
            import signal
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.communicate(timeout=10)
            cleanup(owners)
        result['owned_process_cleanup_completed'] = True
        json.dump(result, out, indent=2)
        out.write('\n')
    if result['status'] != 'passed':
        raise ValueError('Asset MCP probe failed; inspect local receipt')
    return result


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        print(json.dumps(asyncio.run(probe())))
    else:
        raise SystemExit('Use scripts/task_local.py probe --out outputs/asset-probe.json')
