"""Fail-closed holdout manifest and independent contamination audit.

No data is downloaded or fabricated. A reviewed external panel is required.
"""
import hashlib
import json
import re
from difflib import SequenceMatcher


def normalized(question):
    return ' '.join(question.casefold().split())


def audit_rows(training, development, holdout):
    if len(holdout) != 256 or len({r['case_id'] for r in holdout}) != 256:
        raise ValueError('Holdout must contain 256 distinct cases')
    prior = training + development
    if {r['case_id'] for r in prior} & {r['case_id'] for r in holdout}:
        raise ValueError('Holdout IDs overlap training/development')
    old = {normalized(r['question']) for r in prior}
    new = [normalized(r['question']) for r in holdout]
    if len(set(new)) != 256 or old & set(new):
        raise ValueError('Normalized question contamination')
    # Flag templates and near duplicates for independent human adjudication.
    template = lambda q: re.sub(r'\d+(?:[.,]\d+)*', '<number>', q)
    flags = []
    for i, row in enumerate(holdout):
        for other in prior + holdout[:i]:
            a, b = new[i], normalized(other['question'])
            if template(a) == template(b) or SequenceMatcher(None, a, b, autojunk=False).ratio() >= .90:
                flags.append([row['case_id'], other['case_id']])
    return {'cases': 256, 'exact_disjoint': True, 'near_duplicate_pairs': flags,
            'normalized_question_sha256': [hashlib.sha256(q.encode()).hexdigest() for q in new]}


def validate_manifest(manifest, frozen):
    h = manifest['holdout']; files = manifest['files']
    ids = files['eval_main.parquet']['case_ids']
    excluded = set(frozen['train_main.parquet']['case_ids']) | set(frozen['eval_main.parquet']['case_ids'])
    if len(ids) != 256 or len(set(ids)) != 256 or excluded & set(ids):
        raise ValueError('New holdout must be disjoint from training and legacy development IDs')
    if not re.fullmatch(r'phase2/holdouts/[a-z0-9-]{1,64}', h['volume_path']):
        raise ValueError('Invalid immutable holdout path')
    if (h['untouched_attested'] is not True or not h['source_uri'] or not h['selection_rule']
            or not h['independent_reviewer'] or h['training_author'] == h['independent_reviewer']
            or h['grading_protocol'] != 'blind-independent-numeric-v1'
            or h['contamination_review'] != 'passed'):
        raise ValueError('Unreviewed or reused holdout')
    for name in ('source_sha256', 'contamination_report_sha256', 'grading_rubric_sha256'):
        if not re.fullmatch('[a-f0-9]{64}', h[name]):
            raise ValueError('Missing holdout provenance')
    hashes = h['normalized_question_sha256']
    if len(hashes) != 256 or len(set(hashes)) != 256 or any(not re.fullmatch('[a-f0-9]{64}', x) for x in hashes):
        raise ValueError('Missing normalized question commitments')
