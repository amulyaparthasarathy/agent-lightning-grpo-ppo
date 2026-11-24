"""Bounded real local MCP probe for recovery-readiness-v1; no inference or training.

Checks the three discovered schemas, valid facts and every typed error payload in
this runtime's MCP version, then runs the reference solver through the real stdio
transport on the frozen development panel and verifies each answer independently.
Final-evaluation cases are not used.
"""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
try:
    from . import recovery_task as task, recovery_solver as solver
    from .owned_processes import cleanup
except ImportError:
    import recovery_task as task
    import recovery_solver as solver
    from owned_processes import cleanup


async def probe():
    from autogen_ext.tools.mcp import McpWorkbench
    try:
        from .recovery_agent import server_params
    except ImportError:
        from recovery_agent import server_params
    checks, solved = [], []
    rows = task.rows('development')
    asset_id, as_of = task.target(rows[0]['question'])
    policy_id = task.call('get_asset', {'asset_id': asset_id})['policy_id']
    async with McpWorkbench(server_params()) as workbench:
        tools = await workbench.list_tools()
        if ([t['name'] for t in tools] != list(task.TOOLS)
                or any(t['parameters'] != task.TOOL_SCHEMAS[t['name']] for t in tools)):
            raise ValueError('MCP schema differs')
        for name, args, code in (
                ('get_asset', {'asset_id': asset_id}, None),
                ('get_policy', {'policy_id': policy_id}, None),
                ('list_backup_runs', {'asset_id': asset_id, 'as_of': as_of}, None),
                ('get_asset', {}, 'invalid_arguments'),
                ('get_asset', {'asset_id': 7}, 'invalid_arguments'),
                ('get_asset', {'asset_id': asset_id, 'as_of': as_of}, 'invalid_arguments'),
                ('get_asset', {'asset_id': '../'+asset_id}, 'invalid_asset_id'),
                ('get_asset', {'asset_id': 'ast-00000000'}, 'unknown_asset'),
                ('get_policy', {'policy_id': asset_id}, 'invalid_policy_id'),
                ('get_policy', {'policy_id': 'pol-00000000'}, 'unknown_policy'),
                ('list_backup_runs', {'asset_id': asset_id}, 'invalid_arguments'),
                ('list_backup_runs', {'asset_id': asset_id, 'as_of': '2026-02-30T00:00Z'}, 'invalid_as_of'),
                ('list_backup_runs', {'asset_id': asset_id, 'as_of': 'now'}, 'invalid_as_of')):
            result = await workbench.call_tool(name, args)
            error = code is not None
            # The episode policy depends on the exact typed error in this runtime's MCP version.
            valid = result.is_error == error and json.loads(result.to_text()) == (
                task.policy_error(code) if error else task.call(name, args))
            checks.append({'tool': name, 'arguments': args, 'expected_error': error, 'expected_code': code, 'valid': valid})
        for row in rows:
            ledger = []
            plan, observation = solver.steps(row['question']), None
            try:
                while True:
                    name, args = plan.send(observation)
                    result = await workbench.call_tool(name, args)
                    if result.is_error:
                        raise ValueError('Reference solver call failed')
                    ledger.append({'tool': name, 'arguments': args, 'content': result.to_text()})
                    observation = json.loads(result.to_text())
            except StopIteration as done:
                answer = task.dumps(done.value)
            case = task.expected(row['question'], row['result'])
            solved.append({'case_id': row['case_id'], 'calls': len(ledger), 'verified': task.reward(answer, case, ledger) == 1.0})
    passed = all(c['valid'] for c in checks) and len(solved) == 24 and all(s['verified'] and s['calls'] == 3 for s in solved)
    return {'schema': 'recovery-local-probe-v1', 'task': task.TASK, 'environment_sha256': task.digest(task.environment()),
            'transport': 'real stdio MCP', 'checks': checks, 'reference_solver': {
                'panel_id': task.panels()['development']['panel_id'], 'split': 'development', 'cases': len(solved),
                'verified': sum(s['verified'] for s in solved), 'evidence_class': 'reference-solver; not model inference'},
            'status': 'passed' if passed else 'failed', 'workbench_context_exited': True,
            'model_inference': 'not executed', 'live_qualification': False}


def check(output, timeout=120):
    with Path(output).open('x') as out:
        # Retain ownership records beside the result, especially when cleanup is unresolved.
        owners = tempfile.mkdtemp(prefix=Path(output).name+'-mcp-', dir=Path(output).parent)
        env = dict(os.environ, AGL_MCP_OWNERS=owners)
        proc = None
        result = {'status': 'failed', 'reason': 'MCP worker did not complete'}
        anomalies = []
        try:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker'],
                                    env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
            stdout, _ = proc.communicate(timeout=timeout)
            candidate = json.loads(stdout) if proc.returncode == 0 else {'status': 'failed', 'reason': 'MCP worker failed'}
            if not isinstance(candidate, dict) or candidate.get('status') not in ('passed', 'failed'):
                raise ValueError('Invalid worker receipt')
            result = candidate
        except Exception as exc:
            result = {'status': 'failed', 'reason': 'MCP worker startup, execution, timeout or receipt failure',
                      'error_type': type(exc).__name__}
        finally:
            import signal
            if proc is not None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except Exception as exc:
                    anomalies.append({'action': 'worker_signal', 'error_type': type(exc).__name__})
                try:
                    proc.communicate(timeout=10)
                except Exception as exc:
                    anomalies.append({'action': 'worker_reap', 'error_type': type(exc).__name__})
            try:
                ownership = cleanup(owners)
            except Exception as exc:
                ownership = {'complete': False, 'anomalies': [{'error_type': type(exc).__name__}]}
            result['ownership_receipts_directory'] = owners
            result['owned_process_cleanup'] = ownership
            result['worker_cleanup_anomalies'] = anomalies
            result['owned_process_cleanup_completed'] = ownership['complete'] and not anomalies
            if not result['owned_process_cleanup_completed']:
                result['status'] = 'failed'
                result.setdefault('reason', 'Owned process cleanup unresolved')
            # Even interrupted execution retains failure evidence before propagation.
            json.dump(result, out, indent=2)
            out.write('\n')
    if result['status'] != 'passed':
        raise ValueError('Recovery MCP probe failed; inspect local receipt')
    return result


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']:
        print(json.dumps(asyncio.run(probe())))
    else:
        raise SystemExit('Use scripts/task_local.py probe --task recovery-readiness-v1 --out outputs/recovery-probe.json')
