#!/usr/bin/env python3
"""Synthetic arithmetic illustration, not an AGL training reproduction.

He et al., Agent Lightning v1.0 (2026), Sec. 2.3, Eqs. 15 and 16:
https://arxiv.org/html/2608.17528v1#S2.SS3
Uniform sample averaging weights rollouts by their sample counts; Eq. 16
averages token losses within each rollout, then weights rollouts equally.
"""
import argparse
import json
from pathlib import Path


def scenario(fragment):
    # Identical underlying token losses in both representations; no model ran.
    token_losses = {"A": [1, 1, 2, 2, 3, 3], "B": [8] * 6}
    rewards = {"A": 1, "B": 0}
    rows = [(key, tokens[i:i + width]) for key, tokens in token_losses.items()
            for width in [2 if fragment and key == "A" else 6]
            for i in range(0, len(tokens), width)]
    return {"rows": [{"rollout": key, "response_tokens": len(tokens),
                      "predefined_mean_token_loss": sum(tokens) / len(tokens)} for key, tokens in rows],
            "rewards": rewards, "global_row_mean_loss": sum(sum(t) / len(t) for _, t in rows) / len(rows),
            "equal_rollout_token_mean_loss": sum(sum(t) / len(t) for t in token_losses.values()) / 2,
            "global_row_rollout_weights": {k: sum(key == k for key, _ in rows) / len(rows) for k in token_losses},
            "equal_rollout_weights": {"A": 0.5, "B": 0.5}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", help="New directory for cpu-accounting.json")
    args = parser.parse_args()
    result = {"synthetic_fixture": True, "source": "https://arxiv.org/html/2608.17528v1#S2.SS3",
              "scope": "Mathematical weight change only; no training, policy improvement, or stability measured.",
              "unsplit": scenario(False), "fragmented": scenario(True)}
    text = json.dumps(result, indent=2) + "\n"
    if args.output_dir:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=False)
        (output / "cpu-accounting.json").write_text(text, encoding="utf-8")
    else:
        print(text, end="")
if __name__ == "__main__":
    main()
