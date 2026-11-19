"""Independent final-panel grading; no trainer reward/scorer imports.

Input is a blinded extraction of the final answer, never a trainer reward bit.
Ambiguous units, malformed text and extraction failures require adjudication.
"""
from fractions import Fraction
import re

NUMBER = re.compile(r'[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:/[+-]?\d+)?\Z')


def numeric(value):
    value = str(value).strip()
    if ',' in value:
        if not re.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?', value):
            raise ValueError('Ambiguous numeric separators')
        value = value.replace(',', '')
    if not NUMBER.fullmatch(value):
        raise ValueError('Not an unambiguous numeric final answer')
    return Fraction(value)


def grade(cases, predictions):
    expected={r['case_id']:r for r in cases}
    if len(expected)!=len(cases) or len(predictions)!=len(cases) or len({r['case_id'] for r in predictions})!=len(cases):
        raise ValueError('Incomplete or duplicated independent grading denominator')
    if set(expected)!={r['case_id'] for r in predictions}:raise ValueError('Case set mismatch')
    rows=[]
    for row in predictions:
        target=expected[row['case_id']]
        record={'case_id':row['case_id'],'status':'adjudication_required','correct':None}
        try:
            if row.get('unit','')!=target.get('unit','') or row.get('extraction_ambiguous',False):
                raise ValueError('Unit or extraction ambiguity')
            record.update(status='graded',correct=numeric(row['final_answer'])==numeric(target['answer']))
        except (ValueError,ZeroDivisionError,KeyError) as exc:
            record['reason']=str(exc)
        rows.append(record)
    return {'protocol':'blind-independent-numeric-v1','denominator':len(cases),'cases':rows,
            'adjudication_required':sum(r['correct'] is None for r in rows),
            'correct':sum(r['correct'] is True for r in rows)}
