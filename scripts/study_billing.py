"""Read-only study cost attribution for the collector. Gross hourly observations can arrive late."""
from datetime import datetime, timedelta, timezone
from decimal import Decimal


def attribution(rows, submissions, volume_id, registry_id):
    owned = {volume_id:'volume', registry_id:'registry'}
    for action, submission in submissions.items():
        if submission and submission.get('app_id'):
            obj = submission['app_id']
            if obj in owned:
                raise ValueError('Ambiguous billing object attribution')
            owned[obj] = action
    seen, totals = set(), {}
    for row in rows:
        if row.object_id not in owned:
            continue
        key = (row.object_id,row.interval_start)
        amount = Decimal(str(row.cost))
        if key in seen or not amount.is_finite() or amount < 0:
            raise ValueError('Invalid or duplicated provider hour')
        seen.add(key)
        role = owned[row.object_id]
        totals[role] = totals.get(role,Decimal(0)) + amount
    return dict(posted_before_credits_usd=str(sum(totals.values(),Decimal(0))),
                by_owned_object_type_usd={key:str(value) for key,value in totals.items()}, object_hours=len(seen),
                coverage='Owned app, volume and registry rows only. Shared image attribution and delayed charges require reconciliation.')


def snapshot(workspace, plan, operations, volume_id, registry_id):
    bounds = plan['profile']['bounds']
    report = dict(allocation_reservation_usd=bounds['total_budget_usd'],
                  storage_allowance_usd=bounds['storage_budget_usd'], delayed_billing_hold_usd=bounds['delayed_billing_usd'],
                  posted_before_credits_usd=None)
    try:
        submissions = {action:op['submission'] for action,op in operations.items()}
        first = min(s['time'] for s in submissions.values() if s)
        start = datetime.fromtimestamp(first,timezone.utc).replace(minute=0,second=0,microsecond=0)
        now = datetime.now(timezone.utc)
        end = now.replace(minute=0,second=0,microsecond=0)+timedelta(hours=1)
        rows = workspace.billing.report(start=start,end=end,resolution='h')
        report.update(attribution(rows,submissions,volume_id,registry_id),as_of=now.isoformat(),start=start.isoformat())
    except Exception as exc:
        report['unavailable_reason'] = type(exc).__name__
    return report
