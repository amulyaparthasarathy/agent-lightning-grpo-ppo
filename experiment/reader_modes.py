"""Closed dispatch for reader plan phases and stage kinds. Pure stdlib; fails closed.

Historical matched plans and the archived reader-reference-v1 schema keep their
original name-based behavior. A reader-task-v2 plan must be in one of the declared
phases, and each of its stages must be one that phase declares; nothing falls
through to calculator, qualification or weaker historical handling.
"""
READER = 'reader-task-v2'
QUALIFICATION, COLLECTION_ONLY, CALIBRATION = 'qualification', 'collection-only', 'development-calibration'
RECOVERY_LEARNING = 'recovery-learning-qualification-v1'
TASKS = ('calculator-v1', 'asset-status-v1', 'recovery-readiness-v1')
# phase -> ordered stage names
STAGES = {RECOVERY_LEARNING: ('baseline', 'collection-probe', 'training', 'checkpoint-eval'),
          QUALIFICATION: ('baseline', 'training', 'checkpoint-eval'),
          COLLECTION_ONLY: ('training',),
          CALIBRATION: ('greedy-development', 'sampled-packet-1', 'sampled-packet-2', 'sampled-packet-3')}
# phase -> tasks allowed to use it
PHASE_TASKS = {RECOVERY_LEARNING: TASKS[2:], QUALIFICATION: TASKS[:2], COLLECTION_ONLY: TASKS[1:2], CALIBRATION: TASKS[2:]}


def phase(plan):
    """The reader-task-v2 phase, or None for historical and archived-v1 plans."""
    if plan.get('schema') != READER:
        return None
    value = plan.get('phase')
    task = plan.get('profile', {}).get('task')
    if value not in STAGES or task not in PHASE_TASKS[value]:
        raise ValueError('Unknown or unsupported reader phase/task combination')
    return value


def kind(plan, stage):
    """One of baseline, training, checkpoint-eval, collection, greedy."""
    name = stage['name'] if isinstance(stage, dict) else stage
    current = phase(plan)
    if current is None:
        return name  # historical dispatch is by stage name, unchanged
    if name not in STAGES[current]:
        raise ValueError('Stage is not declared by this reader phase')
    if current == COLLECTION_ONLY or (current == RECOVERY_LEARNING and name == 'collection-probe'):
        return 'collection'
    if current == CALIBRATION:
        return 'greedy' if name == 'greedy-development' else 'collection'
    return name


def validation_only(plan, stage):
    """True for stages that only evaluate; they use the gateway's temperature-0 route."""
    return kind(plan, stage) != 'training' and kind(plan, stage) != 'collection'


def collection_only(plan, stage):
    """True for a sampled training collection that stops before any advantage or update."""
    return kind(plan, stage) == 'collection'


def route(plan, stage):
    return 'val' if validation_only(plan, stage) else 'train'
