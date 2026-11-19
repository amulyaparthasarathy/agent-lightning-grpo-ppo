#!/usr/bin/env python3
"""Offline independent holdout audit and blinded final-answer grading."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment.matched_holdout import audit_rows
from experiment.matched_grading import grade
from scripts.phase2_study import write_new


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('audit','grade'))
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--training',type=Path);p.add_argument('--development',type=Path)
    p.add_argument('--predictions',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    def read(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if a.command=='audit':
        if not a.training or not a.development:p.error('audit requires training and all historical development rows')
        result=audit_rows(read(a.training),read(a.development),read(a.cases))
    else:
        if not a.predictions:p.error('grade requires independently extracted blinded predictions')
        result=grade(read(a.cases),read(a.predictions))
    write_new(a.out,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
