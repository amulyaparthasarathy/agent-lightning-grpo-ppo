#!/usr/bin/env python3
"""Offline PPO/GRPO matched-study planning; NO Modal operations or cloud execution.

The current lab implements 2-step PPO smoke/reload only. This tool produces
validated, collision-free *proposals* and deliberately exposes no launch path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_STUDY = "agl-ppo-grpo-20261114-v1"
ENTITY = "amulyap-stanford-university"
PROJECT = "cs329z-agent-lightning"
WORKSPACE = "amulyasp"
ENVIRONMENT = "main"
VOLUME = "cs329z-agent-lightning-working-stanford-v2"
BASE_REVISION = "0cb88a4f764b7a12671c53f0838cd831a0843b95"
RUN_ID_RE = re.compile(r"[a-z0-9][a-z0-9-]{0,27}\Z")
SAFE_PATH_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}\Z")
STAGE_NAMES = ("baseline", "train", "checkpoint-eval")
ALGORITHMS = ("grpo", "ppo-mc-v1")
BLOCKERS = (
    "common_grpo_ppo_rollout_retention_and_fragment_invariant_actor_critic_reduction",
    "proven_run_seed_streams_and_paired_question_order",
    "qualified_20_step_ppo_runtime_with_native_reload",
    "study_evaluation_panel_hashes_and_independent_verifier",
    "remote_wandb_group_id_job_type_config_and_terminal_flush",
    "detached_supervisor_lifetime_startup_watchdog_and_targeted_cancel",
    "exclusive_job_ownership_storage_budget_timeout_and_volume_safety",
    "explicit_owner_approval_for_billable_execution",
)


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ident(*parts: object) -> str:
    return _digest(":".join(map(str, parts)))[:16]


def _seed(*parts: object) -> int:
    return int(_digest(":".join(map(str, parts)))[:8], 16) % (2**31 - 1)


def make_plan(study: str, phase: str, attempt: int = 1) -> dict[str, Any]:
    """Produce a fixed protocol *proposal*. Derived seeds are NOT wired to trainers."""
    _require(bool(RUN_ID_RE.fullmatch(study)), "study id must be 1-28 lowercase alphanumerics/hyphens")
    _require(phase in ("pilot", "campaign"), "phase must be pilot or campaign")
    _require(1 <= attempt <= 99, "attempt must be 1..99")
    seeds = [9001] if phase == "pilot" else [1001, 1002, 1003, 1004, 1005]
    steps, concurrency = (20, 2) if phase == "pilot" else (60, 4)
    group = f"{study}-{phase}"
    jobs: list[dict[str, Any]] = []
    for seed in seeds:
        pair_id = f"{study}-{phase}-s{seed}"
        # Independent streams prevent critic draws from shifting actor draws.
        common_schedule = _seed(study, phase, seed, "paired_question_order")
        for algorithm in ALGORITHMS:
            job_id = f"{study}-{phase}-{algorithm}-s{seed}-a{attempt:02d}"
            _require(bool(SAFE_PATH_RE.fullmatch(job_id)), "job id exceeds Modal's name contract")
            root = f"phase2/studies/{study}/{phase}/{job_id}"
            streams = {
                "paired_question_order": common_schedule,
                "python": _seed(job_id, "python"),
                "numpy": _seed(job_id, "numpy"),
                "torch_cpu": _seed(job_id, "torch_cpu"),
                "torch_cuda": _seed(job_id, "torch_cuda"),
                "actor_sampling": _seed(job_id, "actor_sampling"),
                "data_order": _seed(job_id, "data_order"),
                "critic_init": _seed(job_id, "critic_init") if algorithm == "ppo-mc-v1" else None,
            }
            stages: list[dict[str, Any]] = []
            for kind, job_type in (("baseline", "baseline"), ("train", "training"),
                                   ("checkpoint-eval", "checkpoint-eval")):
                stage_id = f"{job_id}-{kind}"
                _require(bool(SAFE_PATH_RE.fullmatch(stage_id)), "stage id exceeds Modal's name contract")
                stage_root = f"{root}/{kind}"
                previous = None if kind == "baseline" else ("baseline" if kind == "train" else "train")
                stages.append({
                    "name": kind,
                    "run_id": stage_id,
                    "depends_on": [] if previous is None else [previous],
                    "output": stage_root,
                    "logs": {n: f"{stage_root}/logs/{n}.log" for n in ("trainer", "controller", "gateway")},
                    "wandb": {
                        "entity": ENTITY, "project": PROJECT, "group": group,
                        "id": _ident(stage_id, "wandb"), "name": stage_id,
                        "job_type": job_type, "resume": "never",
                        "tags": [phase, algorithm, f"seed-{seed}", "h100", "comparison-unqualified"],
                        "config": {
                            "study_id": study, "study_phase": phase, "algorithm": algorithm,
                            "seed": seed, "pair_id": pair_id, "attempt": attempt,
                            "stage": kind, "training_steps": steps,
                            "dataset_selection_seed": 329,
                            "seed_streams_proposed_not_wired": streams,
                            "common_protocol_sha256": None,
                            "evaluation_manifest_sha256": None,
                        },
                    },
                })
            jobs.append({
                "job_id": job_id, "pair_id": pair_id, "algorithm": algorithm,
                "run_seed": seed, "proposed_seed_streams_not_wired": streams,
                "training_steps": steps, "initial_weights": {"source": "prepared-base", "revision": BASE_REVISION},
                "gpu": "H100!", "cpu": 16, "host_memory_mib": 196608,
                "output_root": root, "stages": stages,
                "retention": {"preserve": ["final", "step-20"], "policy_qualification_pending": True},
            })
    plan = {
        "schema": "agl-paired-study-planning-v1",
        "status": "PLANNING_ONLY_NOT_LAUNCHABLE",
        "execution_allowed": False,
        "study_id": study, "study_phase": phase, "attempt": attempt,
        "workspace": WORKSPACE, "environment": ENVIRONMENT,
        "working_volume": VOLUME, "wandb_group": group,
        "model_revision": BASE_REVISION, "dataset_selection_seed": 329,
        "legacy_baseline": {"name": "stanford-v2-base-01", "correct": 26, "total": 64,
                            "role": "infrastructure-reference-not-a-matched-scientific-control"},
        "evaluation_panel": "legacy-64-development" if phase == "pilot" else "NEW_UNTOUCHED_256_REQUIRED",
        "requested_max_parallel_gpu_jobs": concurrency,
        "jobs": jobs,
        "all_jobs_complete_required_for_report": True,
        "automatically_start_campaign_after_pilot": False,
        "automatic_training_retries": 0,
        "open_qualification_gates": list(BLOCKERS),
    }
    validate_plan(plan)
    return plan


def validate_plan(plan: dict[str, Any]) -> None:
    """Check provenance-independent structural properties; not authorization."""
    _require(plan.get("status") == "PLANNING_ONLY_NOT_LAUNCHABLE" and plan.get("execution_allowed") is False,
             "this schema must never authorize execution")
    phase, jobs = plan.get("study_phase"), plan.get("jobs", [])
    _require(phase in ("pilot", "campaign"), "invalid study phase")
    _require(len(jobs) == (2 if phase == "pilot" else 10), "incorrect planned job count")
    _require(plan.get("workspace") == WORKSPACE and plan.get("working_volume") == VOLUME,
             "unexpected cloud workspace or Volume")
    _require(bool(plan.get("open_qualification_gates")), "cannot erase unqualified gates")
    paths, ids, wb_ids, pairs, stage_ids, log_paths = set(), set(), set(), {}, set(), set()
    for job in jobs:
        jid = job["job_id"]
        _require(jid not in ids and bool(SAFE_PATH_RE.fullmatch(jid)), "job id collision/invalid name")
        ids.add(jid)
        _require(job["training_steps"] == (20 if phase == "pilot" else 60), "unreviewed step count")
        _require(job["gpu"] == "H100!" and job["host_memory_mib"] == 196608, "resource mismatch")
        pairs.setdefault(job["pair_id"], {})[job["algorithm"]] = job
        _require([x["name"] for x in job["stages"]] == list(STAGE_NAMES), "missing or reordered stage")
        for index, stage in enumerate(job["stages"]):
            _require(stage["depends_on"] == ([] if index == 0 else [STAGE_NAMES[index-1]]),
                     "stage dependency mismatch")
            _require(stage["output"] not in paths, "output path reused")
            paths.add(stage["output"])
            _require(stage["output"] == f"{job['output_root']}/{stage['name']}",
                     "stage path not scoped to job and stage")
            _require(stage["run_id"] not in stage_ids and bool(SAFE_PATH_RE.fullmatch(stage["run_id"])),
                     "stage name reused or invalid")
            stage_ids.add(stage["run_id"])
            for name, logfile in stage["logs"].items():
                _require(logfile == f"{stage['output']}/logs/{name}.log" and logfile not in log_paths,
                         "log path reused or outside stage")
                log_paths.add(logfile)
            _require(stage["wandb"]["id"] not in wb_ids, "W&B run id reused")
            wb_ids.add(stage["wandb"]["id"])
            _require(stage["wandb"]["group"] == plan["wandb_group"] and
                     stage["wandb"]["resume"] == "never", "W&B group/resume mismatch")
            _require(stage["wandb"]["name"] == stage["run_id"] and
                     stage["wandb"]["config"]["stage"] == stage["name"], "W&B identity mismatch")
    for pair in pairs.values():
        _require(set(pair) == set(ALGORITHMS), "incomplete GRPO/PPO pair")
        _require(len({item["run_seed"] for item in pair.values()}) == 1,
                 "GRPO/PPO pair must share declared run seed")
        _require(len({item["proposed_seed_streams_not_wired"]["paired_question_order"]
                      for item in pair.values()}) == 1, "paired schedules differ")
        _require(len({item["proposed_seed_streams_not_wired"]["python"]
                      for item in pair.values()}) == 2, "actor RNG streams unexpectedly shared")
    _require(len(pairs) == (1 if phase == "pilot" else 5), "incorrect number of seed pairs")


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, timeout=5, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return "UNKNOWN_NOT_VERIFIED"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("pilot", "campaign"), default="pilot")
    p.add_argument("--study-id", default=DEFAULT_STUDY)
    p.add_argument("--attempt", type=int, default=1)
    p.add_argument("--out", type=Path, required=True,
                   help="New output JSON path; refuses to overwrite existing plan")
    args = p.parse_args()
    try:
        plan = make_plan(args.study_id, args.phase, args.attempt)
        plan["planner_git_head"] = git_head()
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as handle:
            json.dump(plan, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"PLANNING ONLY: {len(plan['jobs'])} jobs, {plan['requested_max_parallel_gpu_jobs']} proposed concurrent GPUs")
        print(f"Wrote {args.out}; NO Modal jobs submitted. Backend and qualification gates remain open.")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
