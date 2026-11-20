#!/usr/bin/env python3
"""Offline task checks. Never imports Modal, acquires models, or creates cloud resources."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment import asset_task


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('generate', 'verify', 'probe', 'scripted'))
    p.add_argument('--task', choices=('asset-status-v1', 'calculator-v1'), default='asset-status-v1')
    p.add_argument('--out', type=Path)
    p.add_argument('--data', type=Path)
    args = p.parse_args()
    if args.command in ('generate', 'verify'):
        if args.task != asset_task.TASK or not args.data:
            p.error('generate/verify require --data and asset-status-v1')
        result = asset_task.generate(args.data) if args.command == 'generate' else asset_task.verify(args.data)
    else:
        if not args.out:
            p.error('probe/scripted require a fresh --out')
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.command == 'probe':
            if args.task == asset_task.TASK:
                from experiment.asset_probe import check
            else:
                from experiment.reader_calculator import check
            result = check(args.out)
        else:
            if args.task != asset_task.TASK:
                p.error('scripted harness demonstration supports asset-status-v1')
            # Explicit test-only boundary; excluded from the production image inventory.
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tests'))
            from asset_harness_support import demonstration
            result = demonstration(args.out)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
