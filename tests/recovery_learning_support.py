"""Synthetic qualification profiles. The dataset fixture uses real frozen Parquet bytes."""
import json
from pathlib import Path

from experiment import reader_contract as c, recovery_learning as learning
from test_reader_training import source


def profile(selection='grpo'):
    base = json.loads((c.ROOT/'examples/reader-profile.recovery-learning.synthetic.json').read_text())
    base.update(selection=selection, algorithms=learning.ROUTES[selection].copy())
    if selection == 'paired':
        base['bounds'].update(training_budget_usd=162, total_budget_usd=193)
    return base


def request(name='training', selection='grpo', assets=None):
    plan = c.make_plan(profile(selection), source(), assets)
    job = plan['jobs'][0]
    return c.stage_request(plan, job, next(s for s in job['stages'] if s['name'] == name))
