"""Read-only Phase 4B prompt replay. Print counts after verifying public commitments."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment import matched_contract as c, recovery_prompt


def replay(directory, tokenizer):
    record = json.loads((ROOT/'evidence/phase4b-live-20261125/calibration.json').read_text())
    count = recovery_prompt.tokenizer(tokenizer)
    results = {}
    for stage, identities in record['stage_artifact_commitments'].items():
        run = directory/stage
        for name, pattern in (('traces', 'traces/trajectory-*.jsonl'), ('audit', 'audit/events-*.jsonl'),
                              ('receipt', 'receipt.json'), ('completion', 'completion.json'),
                              ('configuration', 'resolved_config.json'), ('observability', 'observability.json')):
            hashes = sorted(c.file_hash(path) for path in run.glob(pattern))
            if identities[name] != {'files': len(hashes), 'sorted_file_hashes_sha256': c.digest(hashes)}:
                raise ValueError('Historical artifact commitment mismatch')
        results[stage] = recovery_prompt.replay(run/'traces', count)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('tokenizer', type=Path)
    args = parser.parse_args()
    print(json.dumps(replay(args.directory, args.tokenizer), indent=2))
