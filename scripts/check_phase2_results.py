#!/usr/bin/env python3
"""Check committed r03 derived evidence only; standard library, no cloud access."""
import csv
import hashlib
import json
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "evidence/phase2-r03"
ALGORITHMS = ("grpo-matched-v2", "ppo-mc-matched-v2")
SEEDS = ("2001", "2002", "2003")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rows(name):
    with (BUNDLE / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def count(value):
    numerator, denominator = map(int, value.split("/"))
    require(denominator == 64 and 0 <= numerator <= denominator,
            "Expected a binary-score count out of 64")
    return numerator


def main():
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    paths = set()
    for entry in manifest["artifacts"]:
        path = (BUNDLE / entry["path"]).resolve()
        require(path.is_relative_to(BUNDLE.resolve()) and path not in paths,
                "Invalid or duplicate manifest path")
        paths.add(path)
        data = path.read_bytes()
        require(hashlib.sha256(data).hexdigest() == entry["sha256"]
                and len(data) == entry["bytes"], "Manifest mismatch: " + entry["path"])
    per = rows("per_seed_results.csv")
    expected = {(algorithm, seed) for algorithm in ALGORITHMS for seed in SEEDS}
    require(len(per) == 6 and {(r["algorithm"], r["seed"]) for r in per} == expected,
            "Expected all six unique study runs; no qualification seed")
    results = {}
    for r in per:
        key = r["algorithm"], r["seed"]
        base, final = count(r["baseline_correct/64"]), count(r["final_correct/64"])
        require(Fraction(r["baseline_accuracy"]) == Fraction(base, 64)
                and Fraction(r["final_accuracy"]) == Fraction(final, 64)
                and Fraction(r["delta_percentage_points"]) == Fraction(100 * (final-base), 64),
                "Inconsistent score arithmetic")
        wins, losses, ties = (int(r[k]) for k in ("wins", "losses", "ties"))
        require(min(wins, losses, ties) >= 0 and wins + losses + ties == 64
                and wins - losses == final-base, "Inconsistent paired outcomes")
        require(int(r["actor_updates"]) == 20 and int(r["critic_updates"]) ==
                (20 if key[0] == ALGORITHMS[1] else 0), "Unexpected update count")
        require(r["stage_verification"] == r["W&B_verification"] == "PASS"
                and int(r["invalid_or_incomplete_episodes"]) == 0, "Incomplete recorded run")
        results[key] = base, final
        print(f"{key[0]} seed {key[1]}: {base}/64 -> {final}/64; "
              f"{100*(final-base)/64:+.6f} pp")

    pairs = rows("paired_differences.csv")
    require(len(pairs) == 3 and {r["seed"] for r in pairs} == set(SEEDS),
            "Expected three unique seed pairs")
    for r in pairs:
        gb, gf = results[ALGORITHMS[0], r["seed"]]
        pb, pf = results[ALGORITHMS[1], r["seed"]]
        require(Fraction(r["grpo_minus_ppo_final_percentage_points"]) == Fraction(100*(gf-pf), 64)
                and Fraction(r["difference_in_improvements_percentage_points"]) ==
                Fraction(100*((gf-gb)-(pf-pb)), 64), "Inconsistent algorithm contrast")
        go, po, both, neither = (int(r[k]) for k in ("final_grpo_only_correct",
            "final_ppo_only_correct", "final_both_correct", "final_both_incorrect"))
        require(min(go, po, both, neither) >= 0 and go+po+both+neither == 64
                and go+both == gf and po+both == pf, "Inconsistent case contrasts")

    stages = rows("stage_receipts.csv")
    keys = {(r["algorithm"], r["seed"], r["stage"]) for r in stages}
    require(len(stages) == 24 and keys == {(a, s, stage) for a in ALGORITHMS
            for s in (*SEEDS, "9101") for stage in ("baseline", "training", "checkpoint-eval")},
            "Expected exactly 24 unique qualification/study stages")
    for r in stages:
        require(r["source_commit"] == manifest["executed_source_commit"]
                and r["completeness_verdict"] == r["W&B_verification"] == "PASS"
                and not r["failed_checks"] and int(r["invalid_or_incomplete_episodes"]) == 0,
                "Stage source/verdict mismatch")
        require(all(r[k] == "completed" for k in ("supervisor_status",
                "supervisor_stage_status", "durable_receipt_status")), "Incomplete stage")
        qualification = r["seed"] == "9101"
        require(r["phase"] == ("exploratory-qualification" if qualification else "exploratory")
                and r["plan_sha256"] == manifest["qualification_plan_sha256" if qualification
                                                else "study_plan_sha256"], "Stage plan mismatch")
        steps = 2 if qualification else 20
        updates = steps if r["stage"] == "training" else 0
        require(int(r["actor_updates"]) == updates and int(r["critic_updates"]) ==
                (updates if r["algorithm"] == ALGORITHMS[1] else 0), "Stage optimizer mismatch")
        require(int(r["terminal_global_step"]) == (0 if r["stage"] == "baseline" else steps),
                "Stage terminal step mismatch")
        if r["stage"] == "training":
            require(int(r["case_count"]) == steps * 8 * 4, "Trajectory denominator mismatch")
            reload = next(s for s in stages if (s["algorithm"], s["seed"], s["stage"]) ==
                          (r["algorithm"], r["seed"], "checkpoint-eval"))
            require(len(r["checkpoint_sha256"]) == 64
                    and r["checkpoint_sha256"] == reload["checkpoint_sha256"]
                    and r["worker_call_id"] != reload["worker_call_id"],
                    "Recorded checkpoint/reload identity mismatch")
        if r["stage"] != "training":
            require(int(r["case_count"]) == 64, "Evaluation denominator mismatch")
            if not qualification:
                expected_correct = results[r["algorithm"], r["seed"]][r["stage"] == "checkpoint-eval"]
                require(Fraction(r["correct"]) == expected_correct, "Stage score mismatch")
    for a in ALGORITHMS:
        base = sum(results[a, s][0] for s in SEEDS)
        final = sum(results[a, s][1] for s in SEEDS)
        print(f"{a} mean: baseline {100*base/192:.6f}%, final {100*final/192:.6f}%, "
              f"improvement {100*(final-base)/192:+.6f} pp")
    print(f"PASS: {len(paths)} artifact hashes; 24 stage summaries; six runs; three contrasts.")
    print("Derived CSV consistency only; no raw trajectories, model/data bytes, or tensors revalidated.")


if __name__ == "__main__":
    main()
