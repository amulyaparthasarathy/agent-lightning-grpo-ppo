"""Bounded CPU-only probe of the pinned calculator's actual stdio API."""
import argparse
import asyncio
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile


async def probe(workbench):
    tools = await workbench.list_tools()
    expected = {'type': 'object', 'properties': {'expression': {'title': 'Expression', 'type': 'string'}},
                'required': ['expression'], 'additionalProperties': False}
    receipt = {'schema': 'reader-calculator-preflight-v1', 'transport': 'stdio',
               'tools': tools, 'checks': [], 'status': 'failed',
               'model_inference': 'unqualified', 'training': 'unqualified', 'reload': 'unqualified'}
    matches = [t for t in tools if t['name'] == 'calculate']
    receipt['schema_valid'] = len(matches) == 1 and matches[0]['parameters'] == expected
    if not receipt['schema_valid']:
        return receipt
    for request, expected_text, expected_error in (
        ({'expression': '(2 + 3) * 4'}, '20', False),
        ({'expression': 'unknown_reader_symbol + 1'},
         'Error executing tool calculate: Unknown identifier: unknown_reader_symbol', True),
    ):
        result = await workbench.call_tool('calculate', request)
        observed = result.model_dump(mode='json')
        valid = (observed['name'] == 'calculate' and observed['is_error'] is expected_error
                 and observed['result'] == [{'type': 'TextResultContent', 'content': expected_text}])
        receipt['checks'].append({'tool': 'calculate', 'request': request,
                                  'observed': observed, 'valid': valid})
    receipt['status'] = 'passed' if all(c['valid'] for c in receipt['checks']) else 'failed'
    return receipt


async def worker(pid_file):
    from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
    versions = {p: importlib.metadata.version(p) for p in ('autogen-ext', 'autogen-core', 'mcp', 'mcp-server-calculator')}
    if versions['autogen-ext'] != '0.7.5' or versions['autogen-core'] != '0.7.5' or versions['mcp-server-calculator'] != '0.2.1':
        raise ValueError('Calculator/AutoGen package identity differs')
    async with McpWorkbench(StdioServerParams(command=sys.executable,
        args=[str(Path(__file__).resolve()), '--server', str(pid_file)], read_timeout_seconds=15)) as workbench:
        receipt = await probe(workbench)
    return {**receipt, 'packages': versions, 'workbench_context_exited': True}


def check(output):
    """Bound startup, calls and teardown together; kill the owned process group on timeout."""
    output = Path(output)
    # Reserve exclusively before starting, including failed probes; never overwrite evidence.
    with output.open('x') as log, tempfile.TemporaryDirectory(prefix='reader-calculator-') as temporary:
        pid_file = Path(temporary)/'server.pid'
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', '--pid-file', str(pid_file)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            stdout, _stderr = process.communicate(timeout=90)
            try:
                receipt = json.loads(stdout) if process.returncode == 0 else {
                    'status': 'failed', 'error': 'calculator child failed', 'returncode': process.returncode}
            except ValueError:
                receipt = {'status': 'failed', 'error': 'invalid calculator child receipt'}
        except subprocess.TimeoutExpired:
            receipt = {'status': 'failed', 'error': 'calculator timeout', 'deadline_seconds': 90}
        finally:
            # MCP may start its server in a separate session. The owned wrapper
            # records that PID before exec, so timeout cleanup reaches both groups.
            pids = [process.pid]
            if pid_file.exists():
                pids.append(int(pid_file.read_text()))
            for pid in pids:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.communicate(timeout=10)
        receipt['owned_process_group_cleaned'] = True
        json.dump(receipt, log, indent=2, allow_nan=False)
        log.write('\n')
    if receipt['status'] != 'passed':
        raise ValueError(f'Calculator CPU probe failed; see {output}')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--pid-file', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--server', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.server:
        args.server.write_text(str(os.getpid()))
        os.execv(sys.executable, [sys.executable, '-m', 'mcp_server_calculator'])
    elif args.worker:
        print(json.dumps(asyncio.run(worker(args.pid_file))))
    elif args.out:
        check(args.out)
    else:
        parser.error('--out is required')
