import json
from experiment import reader_contract as c
from test_reader_training import source


def profile():
    return json.loads((c.ROOT/'examples/reader-profile.recovery-study.synthetic.json').read_text())


def fixture_source():
    result = source()
    declaration = json.loads((c.ROOT/c.study.DECLARATION).read_text())
    for name in c.study.ANALYSIS_SOURCES:
        result['files'][name] = c.file_hash(c.ROOT/name)
    for name in [c.study.DECLARATION, *('experiment/'+n for n in declaration['frozen_sources'])]:
        result['files'][name] = result['runtime_files'][name] = c.file_hash(c.ROOT/name)
    return result
