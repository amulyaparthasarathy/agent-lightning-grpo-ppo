#!/usr/bin/env python3
"""Offline task checks. Never imports Modal, acquires models, or creates cloud resources."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment import asset_task

ASSET, CALCULATOR, RECOVERY = asset_task.TASK, 'calculator-v1', 'recovery-readiness-v1'


def recovery_prompts(args):
    """Effective served prompts for one development case, and the token budget if a tokenizer is given."""
    from experiment import recovery_prompt as prompt, recovery_task as task
    row = task.rows('development')[args.case]
    asset_id, as_of = task.target(row['question'])
    policy_id = task.call('get_asset', {'asset_id': asset_id})['policy_id']
    solved = [('get_asset', {'asset_id': asset_id}), ('get_policy', {'policy_id': policy_id}),
              ('list_backup_runs', {'asset_id': asset_id, 'as_of': as_of})]
    paths = {'reference: three calls, answer on a tool-enabled turn': prompt.episode(row['question'], solved),
             'typed error then recovery': prompt.episode(row['question'], [('get_asset', {'asset_id': asset_id, 'as_of': as_of})] + solved),
             'five rounds, then tool-less reflection with the reminder': prompt.episode(
                 row['question'], solved + solved[2:] * 2, reflect=True)}
    count = prompt.tokenizer(args.tokenizer) if args.tokenizer else None
    for views in paths.values():
        for view in views:
            view['prompt_tokens'] = count(view['prompt']) if count else 'not measured: pass --tokenizer'
    result = {'schema': 'recovery-effective-prompts-v1', 'task': task.TASK, 'case_id': row['case_id'], 'split': 'development',
              'rendering': 'prepared served template (fixed date header) rendered offline; no model inference',
              'template_date_header': 'Today Date: 11 Nov 2026', 'authoritative_time': as_of,
              'context_limits': task.CONTEXT_LIMITS, 'paths': paths}
    if count:
        result['budget'] = prompt.budget(count)
    if args.replay:
        if not count:
            raise SystemExit('--replay requires --tokenizer')
        result['replay'] = prompt.replay(args.replay, count)
    return result


def calibration_report(args):
    """Review collected calibration evidence and write the report.

    The plan is validated on every path. A live decision requires the supervisor
    index (collection.json, written by `reader_training.py collect`), the durable
    receipts, the traces and the audit events, all under --evidence. With less, the
    report is still written, as diagnostics: explicitly unverified, never live.
    """
    from experiment import reader_assets as assets, reader_evidence as evidence
    plan = assets.read(args.plan)
    collection = args.evidence/'collection.json'
    index = assets.read(collection).get('index') if collection.is_file() else None
    if evidence.modes.recovery(plan):
        reviewed = evidence.validate_collection(plan, index, evidence.local_reader(plan, args.evidence), local=args.evidence)
        if plan['phase'] == evidence.modes.STUDY and collection.is_file():
            reviewed['billing'] = assets.read(collection).get('billing')
        assets.write_new(args.out/'report.json', reviewed)
        return reviewed
    reviewed = evidence.calibration_collection(plan, index, evidence.local_reader(plan, args.evidence), local=args.evidence)
    result = reviewed['calibration']
    args.out.mkdir(parents=True, exist_ok=False)
    assets.write_new(args.out/'report.json', result)
    from experiment.recovery_calibration import summary
    (args.out/'summary.md').write_text(summary(result))
    return {'report': str(args.out/'report.json'), 'summary': str(args.out/'summary.md'),
            'evidence_class': result['evidence_class'], 'live_policy_measurement': result['live_policy_measurement'],
            'provenance': result['provenance'], 'plan_problems': result['plan_problems'],
            'episodes_admitted_after_recomputation_from_traces': sum(r['accounting']['admitted'] for r in result['routes'].values()),
            # Counts of expected (case, rollout) slots. None means unknown: no trace evidence was available.
            'accounting': {a: {k: r['accounting'][k] for k in ('expected', 'admitted', 'observed_attempts', 'rewards_confirmed',
                                                               'rewards_missing', 'unattempted', 'unknown', 'rejected_untrusted',
                                                               'unexpected_extra_attempts')}
                           for a, r in result['routes'].items()},
            'decisions': {a: {k: r['decision'][k] for k in ('operational_validity', 'scientific_suitability', 'reasons')}
                          for a, r in result['routes'].items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('generate', 'verify', 'probe', 'scripted', 'prompts', 'calibration-demo', 'report'))
    p.add_argument('--task', choices=(ASSET, CALCULATOR, RECOVERY), default=ASSET)
    p.add_argument('--out', type=Path)
    p.add_argument('--data', type=Path)
    p.add_argument('--case', type=int, default=0, help='prompts: index into the development panel')
    p.add_argument('--tokenizer', type=Path, help='prompts: pinned tokenizer.json (verified against the model lock)')
    p.add_argument('--replay', type=Path, help='prompts: collected evidence directory whose recorded prompt tokens to reproduce')
    p.add_argument('--evidence', type=Path, help='report: directory written by reader_training.py collect, or by calibration-demo')
    p.add_argument('--plan', type=Path, help='report: the plan the evidence was produced under')
    args = p.parse_args()
    if args.command in ('prompts', 'calibration-demo', 'report') and args.task != RECOVERY:
        p.error(f'{args.command} requires --task {RECOVERY}')
    if args.command in ('generate', 'verify'):
        if args.task == CALCULATOR or not args.data:
            p.error('generate/verify require --data and a synthetic task')
        if args.task == RECOVERY:
            from experiment import recovery_task as task
        else:
            task = asset_task
        result = task.generate(args.data) if args.command == 'generate' else task.verify(args.data)
        if args.task == RECOVERY:
            result = {'schema': result['schema'], 'task': result['task'], 'environment_sha256': result['environment_sha256'],
                      'world_file_sha256': result['world_file_sha256'], 'panels': result['environment']['panels'],
                      'files': {n: {k: f[k] for k in ('rows', 'split', 'sha256')} for n, f in result['files'].items()}}
    elif args.command == 'report':
        if not (args.evidence and args.plan and args.out):
            p.error('report requires --evidence, --plan and a fresh --out directory')
        result = calibration_report(args)
    else:
        if not args.out:
            p.error(f'{args.command} requires a fresh --out')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.command == 'probe':
            if args.task == ASSET:
                from experiment.asset_probe import check
            elif args.task == RECOVERY:
                from experiment.recovery_probe import check
            else:
                from experiment.reader_calculator import check
            result = check(args.out)
            if args.task == RECOVERY:
                result = {k: v for k, v in result.items() if k != 'checks'} | {'checks_passed': sum(c['valid'] for c in result['checks'])}
        elif args.command == 'prompts':
            result = recovery_prompts(args)
            with args.out.open('x') as handle:
                json.dump(result, handle, indent=2)
                handle.write('\n')
            result = {'written': str(args.out), 'paths': {k: [(v['kind'], v['prompt_tokens']) for v in views] for k, views in result['paths'].items()},
                      **{k: result[k] for k in ('budget', 'replay') if k in result}}
        else:
            if args.task == CALCULATOR:
                p.error('scripted harness demonstrations support the synthetic tasks')
            # Explicit test-only boundary; excluded from the production image inventory.
            sys.path.insert(0, str(ROOT/'tests'))
            if args.command == 'calibration-demo':
                from recovery_harness_support import calibration_demonstration
                result = calibration_demonstration(args.out)
            elif args.task == RECOVERY:
                from recovery_harness_support import demonstration
                result = demonstration(args.out)
            else:
                from asset_harness_support import demonstration
                result = demonstration(args.out)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
