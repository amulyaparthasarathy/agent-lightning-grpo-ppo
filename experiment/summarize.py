#!/usr/bin/env python3
"""Summarize observed_calc_agent JSONL, never filling missing rewards or usage.

--run-dir is the post run when --baseline-dir is supplied. Comparison requires
the frozen --heldout-file and one completed, rewarded val attempt per case.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def stats(values):
    values = [v for v in values if number(v)]
    return {"count": len(values), "sum": sum(values) if values else None,
            "mean": sum(values) / len(values) if values else None,
            "max": max(values) if values else None}


def summarize(directory, phase=None, route=None):
    files = sorted(Path(directory).rglob("trajectory-*.jsonl"))
    if not files:
        raise ValueError(f"No trajectory-*.jsonl files under {directory}")
    attempts, spans, errors, unclassified, seen = [], [], [], 0, set()
    for path in files:
        trace_id = path.stem.removeprefix("trajectory-")
        if trace_id in seen:
            raise ValueError(f"Duplicate trace file ID: {trace_id}")
        seen.add(trace_id)
        records = []
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                record = json.loads(line)
                if not isinstance(record, dict) or record.get("schema_version") != 1 or record.get("trace_id") != trace_id:
                    raise ValueError("Invalid schema or trace ID")
                records.append(record)
            except (ValueError, TypeError):
                errors.append(f"{path.name}:{lineno}")
        meta = records[0] if records else {}
        unclassified += not bool(meta)
        if phase is not None and meta.get("phase") != phase:
            continue
        if route is not None and meta.get("route_mode") != route:
            continue
        episodes = [r for r in records if r.get("kind") == "episode"]
        episode = episodes[0] if len(episodes) == 1 else {}
        attempts.append((meta, episode))
        spans.extend(records)
    episodes = [episode for _, episode in attempts]
    outputs = [e.get("outputs") or {} for e in episodes]
    llms = [r for r in spans if r.get("kind") == "llm"]
    tools = [r for r in spans if r.get("kind") == "tool"]
    usage = [(r.get("outputs") or {}).get("usage") for r in llms]
    completed = sum(e.get("execution_status") == "ok" for e in episodes)
    failed = sum(e.get("execution_status") == "exception" for e in episodes)
    rewards = stats([o.get("reward") for o in outputs])
    report = {
        "run_dir": str(Path(directory).resolve()), "phase_filter": phase, "route_filter": route,
        "attempt_files_total": len(files), "attempts_selected": len(attempts),
        "unclassified_attempt_files": unclassified, "invalid_records": errors,
        "completed": completed, "failed": failed,
        "incomplete_or_ambiguous": len(attempts) - completed - failed,
        "phase_route_counts": dict(Counter(f"{m.get('phase')}/{m.get('route_mode')}" for m, _ in attempts)),
        "observed_reward": rewards, "attempts_without_numeric_reward": len(attempts) - rewards["count"],
        "reward_denominator": "Only observed numeric episode rewards, including failed reward posts; no zero fills.",
        "reward_posts_confirmed": sum(o.get("reward_posted") is True for o in outputs),
        "model_calls_recorded": len(llms),
        "model_errors_recorded": sum(r.get("execution_status") == "exception" for r in llms),
        "tool_calls_recorded": len(tools),
        "tool_errors_recorded": sum(r.get("execution_status") in {"exception", "tool_reported_error"} for r in tools),
        "call_count_scope": "Recorded spans, not transport retries; unfinished or dropped spans may be absent.",
        "model_calls_with_usage": sum(isinstance(u, dict) for u in usage),
        "model_calls_without_usage": sum(not isinstance(u, dict) for u in usage),
        "usage_source": "Recorded response usage only; AutoGen zero/absent ambiguity remains unknown.",
        "usage": {key: stats([u.get(key) for u in usage if isinstance(u, dict)])
                  for key in ("prompt_tokens", "completion_tokens")},
        "episode_elapsed_ms": stats([e.get("elapsed_ms") for e in episodes]),
        "model_operation_ms": stats([r.get("operation_elapsed_ms") for r in llms]),
        "tool_operation_ms": stats([r.get("operation_elapsed_ms") for r in tools]),
        "timing_scope": "Inclusive episode duration; model/tool operation durations are separate and may overlap.",
    }
    # Prospective fields only: do not reinterpret historical failures or rewards.
    if any(e.get('episode_policy') in ('asset-episode-v2', 'recovery-episode-v1') for e in episodes):
        report['episode_outcomes'] = dict(Counter(
            (e.get('outputs') or {}).get('outcome', 'incomplete')
            if e.get('execution_status') == 'ok' and (e.get('outputs') or {}).get('reward_posted') is True
            else 'incomplete' for e in episodes))
        report['format_compliant_finals'] = sum(o.get('diagnostics', {}).get('format_compliant') is True for o in outputs)
        report['policy_errors'] = sum(o.get('policy_errors', 0) for o in outputs)
        report['diagnostic_scope'] = 'Final-format and policy diagnostics; no added reward terms'
    return report, attempts


def compare(baseline, post, heldout):
    expected = []
    for line in Path(heldout).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        expected.append(str(row.get("case_id") or hashlib.sha256(row["question"].encode()).hexdigest()[:16]))
    reasons, rewards = [], []
    if not expected or len(set(expected)) != len(expected):
        reasons.append("Heldout manifest is empty or contains duplicate case IDs")
    for label, (report, attempts) in (("baseline", baseline), ("post", post)):
        if report["invalid_records"] or report["unclassified_attempt_files"]:
            reasons.append(f"{label}: invalid records or unclassified attempts")
        if Counter(m.get("case_id") for m, _ in attempts) != Counter(expected):
            reasons.append(f"{label}: expected case IDs do not appear exactly once")
        valid = all(m.get("eval_split") == "heldout" and m.get("route_mode") == "val"
                    and m.get("synthetic_fixture") is False and e.get("execution_status") == "ok"
                    and (e.get("outputs") or {}).get("reward_posted") is True
                    and number((e.get("outputs") or {}).get("reward"))
                    and e["outputs"]["reward"] in (0, 1) for m, e in attempts)
        if not valid:
            reasons.append(f"{label}: requires complete real heldout val episodes with posted binary rewards")
        rewards.append({m.get("case_id"): (e.get("outputs") or {}).get("reward") for m, e in attempts})
    if reasons:
        return {"available": False, "reasons": reasons}
    before, after = rewards
    return {"available": True, "paired_cases": len(expected),
            "post_wins": sum(after[c] > before[c] for c in expected),
            "post_losses": sum(after[c] < before[c] for c in expected),
            "ties": sum(after[c] == before[c] for c in expected),
            "interpretation": "Descriptive paired outcomes only; no significance or causal claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--phase")
    parser.add_argument("--route-mode", choices=("train", "val"))
    parser.add_argument("--baseline-dir")
    parser.add_argument("--baseline-phase", default="baseline-eval")
    parser.add_argument("--heldout-file")
    parser.add_argument("--output-dir", help="New directory; refuses to overwrite existing evidence")
    args = parser.parse_args()
    if args.baseline_dir and not args.heldout_file:
        parser.error("--baseline-dir requires --heldout-file")
    phase = args.phase or ("reload-eval" if args.baseline_dir else None)
    route = args.route_mode or ("val" if args.baseline_dir else None)
    try:
        post = summarize(args.run_dir, phase, route)
        result = {"schema_version": 1, "run": post[0]}
        if args.baseline_dir:
            baseline = summarize(args.baseline_dir, args.baseline_phase, "val")
            result.update(baseline=baseline[0], comparison=compare(baseline, post, args.heldout_file))
        text = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output_dir:
            output = Path(args.output_dir)
            output.mkdir(parents=True, exist_ok=False)
            (output / "summary.json").write_text(text, encoding="utf-8")
        else:
            print(text, end="")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
