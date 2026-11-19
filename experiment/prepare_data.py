#!/usr/bin/env python3
"""Prepare fixed, disjoint Calc-X subsets from an already downloaded release.

This script performs no downloads. It retains only question, result, case_id,
and split. The original worked solution (chain) is never passed to the agent.
The eight smoke-evaluation cases are separate from the 64 final held-out cases;
all 72 evaluation cases are excluded from training. Smoke training is a subset
of main training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def question_key(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def read_rows(path: Path, split: str) -> tuple[list[dict[str, str]], int]:
    import pyarrow.parquet as pq

    columns = pq.read_schema(path).names
    if not {"question", "result"}.issubset(columns):
        raise ValueError(f"{path} must contain question and result columns")
    selected = [name for name in ("id", "question", "result") if name in columns]
    raw = pq.read_table(path, columns=selected).to_pylist()
    by_id: dict[str, dict[str, str]] = {}
    by_question: dict[str, dict[str, str]] = {}
    duplicates = 0
    for position, row in enumerate(raw):
        if not isinstance(row["question"], str) or not row["question"].strip():
            raise ValueError(f"Empty/invalid question at row {position} in {path}")
        if row["result"] is None or not str(row["result"]).strip():
            raise ValueError(f"Empty result at row {position} in {path}")
        question = row["question"].strip()
        qkey = question_key(question)
        original_id = row.get("id")
        case_id = str(original_id) if original_id is not None else f"question-{qkey}"
        case = {"case_id": case_id, "question": question, "result": str(row["result"]), "split": split}
        old = by_id.get(case_id)
        if old is not None:
            if (old["question"], old["result"]) != (question, case["result"]):
                raise ValueError(f"Conflicting duplicate case ID {case_id!r} in {path}")
            duplicates += 1
            continue
        old_question = by_question.get(qkey)
        if old_question is not None:
            if old_question["result"] != case["result"]:
                raise ValueError(f"Conflicting answers for the same question in {path}")
            duplicates += 1
            continue
        by_id[case_id] = case
        by_question[qkey] = case
    return list(by_id.values()), duplicates


def select_subsets(train: list[dict[str, str]], evaluation: list[dict[str, str]], seed: int):
    def ordering(row: dict[str, str]) -> tuple[str, str]:
        key = f"{seed}\0{row['case_id']}\0{question_key(row['question'])}"
        return hashlib.sha256(key.encode()).hexdigest(), row["case_id"]

    if len(evaluation) < 72:
        raise ValueError(f"Need at least 72 distinct evaluation cases (64 final + 8 smoke); found {len(evaluation)}")
    # Reserve separate final/smoke evaluation sets, then exclude all 72 from training.
    reserved_eval = sorted(evaluation, key=ordering)[:72]
    eval_main = reserved_eval[:64]
    eval_smoke = [{**row, "split": "smoke_eval"} for row in reserved_eval[64:72]]
    held_ids = {row["case_id"] for row in reserved_eval}
    held_questions = {question_key(row["question"]) for row in reserved_eval}
    eligible = [row for row in train if row["case_id"] not in held_ids and question_key(row["question"]) not in held_questions]
    if len(eligible) < 256:
        raise ValueError(f"Need 256 distinct training cases after excluding held-out overlaps; found {len(eligible)}")
    train_main = sorted(eligible, key=ordering)[:256]
    assert not {row["case_id"] for row in train_main} & held_ids
    assert not {question_key(row["question"]) for row in train_main} & held_questions
    return {
        "train_main.parquet": train_main,
        "eval_main.parquet": eval_main,
        "train_smoke.parquet": train_main[:16],
        "eval_smoke.parquet": eval_smoke,
    }, len(train) - len(eligible)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--eval-parquet", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="A new output directory")
    parser.add_argument("--seed", type=int, default=329)
    args = parser.parse_args()
    train_path, eval_path, output = (p.expanduser().resolve() for p in (args.train_parquet, args.eval_parquet, args.out))
    if output.exists():
        parser.error(f"Output already exists; choose a new path: {output}")
    if not train_path.is_file() or not eval_path.is_file():
        parser.error("Both input parquet files must exist; download/extract the official data first")
    if train_path == eval_path:
        parser.error("Training and evaluation must use different released input files")
    train, train_duplicates = read_rows(train_path, "train")
    evaluation, eval_duplicates = read_rows(eval_path, "heldout")
    subsets, excluded = select_subsets(train, evaluation, args.seed)
    import pyarrow as pa
    import pyarrow.parquet as pq

    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".calcx-prep-", dir=output.parent))
    try:
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "selection": "SHA256 rank of seed, original case ID, and whitespace-normalized question; reserve 72 evaluation cases first: first 64 final held-out, next 8 smoke; exclude all 72 from training by ID and question",
            "evaluation_design": {"final_heldout_cases": 64, "smoke_evaluation_cases": 8, "smoke_and_final_disjoint": True, "smoke_training_nested_in_main": True},
            "sources": {
                "train": {"path": str(train_path), "sha256": sha256_file(train_path)},
                "eval": {"path": str(eval_path), "sha256": sha256_file(eval_path)},
            },
            "duplicate_rows_removed": {"train": train_duplicates, "eval": eval_duplicates},
            "training_rows_excluded_for_heldout_overlap": excluded,
            "files": {},
        }
        for name, rows in subsets.items():
            path = work / name
            pq.write_table(pa.Table.from_pylist(rows), path)
            manifest["files"][name] = {"rows": len(rows), "sha256": sha256_file(path), "case_ids": [row["case_id"] for row in rows]}
            if name.startswith("eval_"):
                jsonl_path = path.with_suffix(".jsonl")
                jsonl_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
                manifest["files"][jsonl_path.name] = {"rows": len(rows), "sha256": sha256_file(jsonl_path), "case_ids": [row["case_id"] for row in rows]}
        (work / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(f"Output appeared during preparation: {output}")
        work.rename(output)
    finally:
        if work.exists():
            shutil.rmtree(work)
    print(f"Prepared 256 training / 64 final held-out cases, 16 nested training smoke cases, and 8 separate smoke-evaluation cases: {output}")
    print(f"Manifest: {output / 'data_manifest.json'}")


if __name__ == "__main__":
    main()
