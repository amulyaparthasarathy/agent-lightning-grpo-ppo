"""Evidence-bound release of completed CPU startup allowances, never new limits.

This client-side helper does not launch workers or change the approved plan.
The caller archives the evidence and pre-adjustment ledger in the same guarded
Hub commit as the revised ledger. Historical phase usage stays untouched.
"""
from __future__ import annotations

import copy
import math
import re

from scripts.backup_qualification import digest
from scripts.preservation_transfer import require
from scripts.preservation_workflow import LIMITS, ORDER, reserve, validate_plan

REMAINING = ("replication_publish", "replication_retrieve")
MARGIN_SECONDS = 120


def reconcile_cpu(ledger, evidence, *, revision, reviewed_at):
    """Return a new ledger and append-only audit; evidence is Modal call metadata.

    Only successful, unretried CPU phases with a measured startup duration qualify.
    Retain the larger of worker elapsed or full Modal execution plus startup,
    round upward, and add two minutes. Other resource accounting is unchanged.

    This is a pure calculation, not evidence authentication or permission to write.
    The caller must read the ledger/evidence at verified immutable revisions and
    use an expected-parent guard when archiving the result. A completed workflow
    cannot be reconciled again through this helper.
    """
    validate_plan(ledger['plan'])
    require(digest(ledger['plan']) == ledger['plan_sha256'], 'plan identity differs')
    require(isinstance(revision, str) and re.fullmatch(r'[0-9a-f]{40}', revision)
            and isinstance(reviewed_at, str) and reviewed_at.strip(),
            'immutable ledger reference required')
    accounting = ledger['accounting']
    require(set(accounting) == set(LIMITS)
            and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= LIMITS[k]
                    for k, v in accounting.items()), 'invalid resource accounting')
    require(ledger['status'] == 'ready', 'workflow is not ready')
    require(tuple(n for n in ORDER if ledger['phases'].get(n, {}).get('status') != 'passed') == REMAINING,
            'only publication and retrieval may remain')
    require(all(n not in ledger['phases'] for n in REMAINING), 'remaining phase already claimed')
    require(not any(r.get('kind') == 'completed_cpu_startup_closeout' for r in ledger.get('accounting_reconciliations', [])),
            'closeout allowance already reconciled')
    require(bool(evidence), 'execution evidence missing')
    rows = []
    specs = {p['name']: p for p in ledger['plan']['phases']}
    for name, observation in sorted(evidence.items()):
        phase, spec = ledger['phases'][name], specs[name]
        require(spec['kind'] == 'cpu' and phase['status'] == 'passed'
                and not phase.get('recoveries') and phase.get('attempt', 0) == 0,
                'only unretried successful CPU phases qualify')
        require(observation['function_call_id'] == phase['function_call_id'], 'call identity differs')
        require(observation['total_inputs'] == 1 and observation['succeeded_inputs']['count'] == 1
                and len(observation['succeeded_inputs']['latest']) == 1
                and all(observation[k]['count'] == 0 for k in
                        ('failed_inputs', 'pending_inputs', 'timeout_inputs', 'cancelled_inputs')),
                'execution is ambiguous or not terminal')
        timing = observation['succeeded_inputs']['latest'][0]
        old = phase['usage']['cpu_seconds']
        allowance = spec.get('startup_reserve_seconds', 600)
        values = [old, allowance, timing['started_at'], timing['finished_at'],
                  timing['execution_seconds'], timing['startup_seconds']]
        require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values),
                'finite nonnegative measurements required')
        require(timing['finished_at'] > timing['started_at']
                and abs(timing['execution_seconds'] - (timing['finished_at'] - timing['started_at'])) < 1e-6
                and old >= allowance, 'duration identity differs')
        worker_elapsed = old - allowance
        measured = timing['execution_seconds'] + timing['startup_seconds']
        retained = math.ceil(max(worker_elapsed, measured)) + MARGIN_SECONDS
        require(retained < old, 'no releasable allowance')
        rows.append({'phase': name, 'function_call_id': phase['function_call_id'],
                     'phase_receipt': copy.deepcopy(phase['result']),
                     'modal_observation_sha256': digest(observation),
                     'original_accounted_seconds': old, 'original_startup_allowance_seconds': allowance,
                     'worker_elapsed_seconds': worker_elapsed, 'modal_execution_seconds': timing['execution_seconds'],
                     'modal_startup_seconds': timing['startup_seconds'], 'retained_safety_margin_seconds': MARGIN_SECONDS,
                     'reconciled_seconds': retained, 'released_seconds': old - retained})
    revised = copy.deepcopy(ledger)
    release = sum(row['released_seconds'] for row in rows)
    revised['accounting']['cpu_seconds'] -= release
    require(0 <= revised['accounting']['cpu_seconds'] <= LIMITS['cpu_seconds'], 'invalid revised total')
    projected = copy.deepcopy(revised)
    for name in REMAINING:
        projected['accounting'] = reserve(projected, specs[name])
    audit = {'kind': 'completed_cpu_startup_closeout', 'at': reviewed_at,
             'prior_ledger_revision': revision, 'prior_ledger_sha256': digest(ledger),
             'original_accounting': copy.deepcopy(ledger['accounting']), 'rows': rows,
             'adjustment_cpu_seconds': -release, 'reconciled_accounting': copy.deepcopy(revised['accounting']),
             'projected_with_both_remaining_reservations': projected['accounting'],
             'retained_build_control_cpu_seconds': ledger['plan']['initial_accounting']['cpu_seconds'],
             'basis': 'Completed Modal call timestamps and measured startup; round upward plus 120 seconds per call. '
                      'All failed attempts, diagnostic reservations, build/control allowance, GPU usage and USD/byte '
                      'reservations remain unchanged. CPU seconds retain the existing CPU-worker elapsed-time metric, '
                      'not core-seconds. Provider invoice totals are unknown.'}
    revised.setdefault('accounting_reconciliations', []).append(audit)
    return revised, audit
