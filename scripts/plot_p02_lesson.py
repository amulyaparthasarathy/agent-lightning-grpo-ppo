# /// script
# requires-python = "==3.12.*"
# dependencies = ["matplotlib==3.10.8"]
# ///
"""Draw Chapter 7 figures from the public P02 diagnosis tables."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "evidence/phase4c-functional-pilot/diagnosis"
FIGURES = ROOT / "docs/learning/figures"
CLASSES = ("ready", "not_ready")
TEAL, BLUE = "#176d68", "#425b89"


def read_table(name):
    return json.loads((DATA / name).read_text())


def development():
    cases = read_table("development.json")
    panels = [[r for r in cases if r["actual"] == label] for label in CLASSES]
    panels.append(cases)
    fig, ax = plt.subplots(figsize=(7.3, 3.3), layout="constrained")
    for stage, label, color, offset in (
        ("base", "Baseline", "#606776", -0.18), ("reload", "Step 16 reload", "#9b5521", 0.18)
    ):
        counts = [sum(r[stage]["reward"] for r in panel) for panel in panels]
        fractions = [f"{n}/{len(panel)}" for n, panel in zip(counts, panels)]
        print(label, "strict (ready, not_ready, overall):", *fractions)
        for field in ("decision_correct", "evidence_correct"):
            correct = sum(r[stage][field] for r in cases)
            print(f"  {field}: {correct}/{len(cases)}")
        decisions = Counter(r[stage]["decision"] for r in cases)
        print(f"  not_ready answers: {decisions['not_ready']}/{len(cases)}")
        bars = ax.bar([i + offset for i in range(3)],
                      [100 * n / len(panel) for n, panel in zip(counts, panels)],
                      width=0.34, label=label, color=color)
        ax.bar_label(bars, fractions, padding=3)
    ax.set(xticks=range(3), xticklabels=["Actually ready", "Actually not_ready", "All cases"],
           ylabel="Strict success (%)", ylim=(0, 116), yticks=range(0, 101, 25))
    ax.legend(frameon=False, loc="upper left")
    transitions = Counter((r["base"]["reward"], r["reload"]["reward"]) for r in cases)
    for pair, count in sorted(transitions.items()):
        print(f"Strict {pair[0]} -> {pair[1]}: {count}")
    hidden = [r["case_id"].split("/")[-1] for r in cases
              if r["base"]["reward"] == r["reload"]["reward"] == 0
              and r["base"]["decision_correct"] and not r["reload"]["decision_correct"]]
    print("Decision regressions among retained failures:", *hidden)
    return fig


def groups():
    collections = read_table("collections.json")
    with (DATA / "training.csv").open(newline="") as stream:
        samples = list(csv.DictReader(stream))
    fig, ax = plt.subplots(figsize=(7.3, 3.1), layout="constrained")
    for label, offset, color in zip(CLASSES, (-0.18, 0.18), (BLUE, TEAL)):
        counts = [sum(row["by_class"][label][kind] for row in collections)
                  for kind in ("all_zero", "mixed", "all_one")]
        trials = [s for s in samples if s["actual"] == label]
        strict = sum(int(s["reward"]) for s in trials)
        positive = sum(float(s["advantage"]) > 0 for s in trials)
        print(f"{label}: groups {counts}, strict {strict}/{len(trials)}, positive advantage {positive}")
        bars = ax.bar([i + offset for i in range(3)], counts, width=0.34,
                      label=label, color=color)
        ax.bar_label(bars, padding=3)
    ax.set(xticks=range(3), xticklabels=["All zero", "Mixed", "All one"],
           ylabel="Four-attempt groups", ylim=(0, 52))
    ax.legend(title="Actual class", frameon=False)
    return fig


def collection_rates():
    rows = read_table("collections.json")
    # One row per collection. Print each fraction beside a bar on a common scale.
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 5.7), sharey=True, layout="constrained")
    steps = [r["collection"] for r in rows]
    for ax, label, color in zip(axes, CLASSES, (BLUE, TEAL)):
        values = [r["by_class"][label] for r in rows]
        rates = [100 * v["correct"] / v["trajectories"] for v in values]
        bars = ax.barh(steps, rates, height=0.56, color=color)
        ax.bar_label(bars, [f"{v['correct']}/{v['trajectories']}" for v in values], padding=4)
        ax.set(title=f"Actually {label}", xlabel="Strict success (%)",
               xlim=(0, 127), xticks=[0, 50, 100])
    axes[0].set(yticks=steps, yticklabels=[f"{r['collection']}  /  {r['policy_step']}" for r in rows],
                ylabel="Collection  /  sampled policy step", ylim=(steps[-1] + 0.7, steps[0] - 0.7))
    for row in rows:
        fractions = []
        for label in CLASSES:
            cell = row["by_class"][label]
            fractions.append(f"{label} {cell['correct']}/{cell['trajectories']}")
        print(f"Collection {row['collection']}, policy {row['policy_step']}:", ", ".join(fractions))
    return fig


def updates():
    rows = read_table("collections.json")
    calls = [r["actor_call"] for r in rows]
    losses = [r["actor"]["pg_loss"] for r in rows]
    norms = [r["actor"]["grad_norm"] for r in rows]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.3, 4.2), sharex=True, layout="constrained")
    top.axhline(0, color="#888888", linewidth=0.8)
    top.scatter(calls, [v * 1e9 for v in losses], color=BLUE)
    top.set_ylabel("Policy loss\n(× 10⁻⁹)")
    bottom.scatter(calls, norms, color=TEAL)
    bottom.set(ylim=(0, 2.1), ylabel="Gradient norm\n(before clipping)",
               xlabel="Actor optimizer call", xticks=calls)
    print(f"{len(rows)} actor calls; policy loss {min(losses):.3g} to {max(losses):.3g}; "
          f"preclip norm {min(norms):.3f} to {max(norms):.3f}")
    return fig


def main():
    plots = {"development": development, "groups": groups, "collections": collection_rates, "updates": updates}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", choices=["all", *plots], default="all", nargs="?")
    choice = parser.parse_args().figure
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 11})
    for name, draw in plots.items():
        if choice in ("all", name):
            fig = draw()
            fig.savefig(FIGURES / f"p02-{name}.png", dpi=160)
            plt.close(fig)


if __name__ == "__main__":
    main()
