#!/usr/bin/env python3
"""Summarize official  math_eval JSONL files for one Qwen3-4B mode."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


DATASETS = ("aime24", "aime25", "hmmt25_feb", "hmmt25_nov")
KS = (1, 4, 8, 16, 32)
QUESTIONS = 30
SAMPLES = 32


def pass_at_k(n: int, c: int, k: int) -> float:
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def summarize(path: Path, dataset: str) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != QUESTIONS:
        raise ValueError(f"{path}: expected {QUESTIONS} rows, got {len(rows)}")
    correct_by_problem = []
    lengths = []
    extracted = 0
    for index, row in enumerate(rows, start=1):
        acc = row.get("acc_list")
        responses = row.get("responses")
        predictions = row.get("pred_answers")
        if not isinstance(acc, list) or len(acc) != SAMPLES:
            raise ValueError(f"{path}: row {index} invalid acc_list length")
        if not isinstance(responses, list) or len(responses) != SAMPLES:
            raise ValueError(f"{path}: row {index} invalid responses length")
        if not isinstance(predictions, list) or len(predictions) != SAMPLES:
            raise ValueError(f"{path}: row {index} invalid pred_answers length")
        if any(not isinstance(value, bool) for value in acc):
            raise ValueError(f"{path}: row {index} has non-boolean accuracy")
        correct_by_problem.append(sum(acc))
        extracted += sum(value is not None for value in predictions)
        lengths.extend(len(value) for value in responses)

    total = QUESTIONS * SAMPLES
    correct = sum(correct_by_problem)
    result = {
        "dataset": dataset,
        "questions": QUESTIONS,
        "samples_per_question": SAMPLES,
        "generations": total,
        "correct_generations": correct,
        "sample_accuracy": correct / total,
        "boxed_extraction_rate": extracted / total,
        "mean_response_characters": sum(lengths) / len(lengths),
        "problem_correct_counts": correct_by_problem,
        "raw_output": str(path),
    }
    for k in KS:
        value = sum(pass_at_k(SAMPLES, count, k) for count in correct_by_problem) / QUESTIONS
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{path}: invalid pass@{k}: {value}")
        result[f"pass@{k}"] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--model", default="Qwen3-4B")
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()
    root = args.eval_root.resolve()
    rows = [summarize(root / "raw" / f"{dataset}.jsonl", dataset) for dataset in DATASETS]
    payload = {
        "model": args.model,
        "evaluation": " official math_eval.py",
        "datasets": list(DATASETS),
        "questions_per_dataset": QUESTIONS,
        "samples_per_question": SAMPLES,
        "max_tokens": 16384,
        "temperature": 1.0,
        "top_p": 1.0,
        "seed": 42,
        "thinking": bool(args.thinking),
        "pass_k": list(KS),
        "results": rows,
    }
    (root / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    columns = ["dataset", "questions", "samples_per_question", "generations", "correct_generations", "sample_accuracy", *[f"pass@{k}" for k in KS], "boxed_extraction_rate", "mean_response_characters", "raw_output"]
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in columns})
    lines = [
        f"# Qwen3-4B Math evaluation ({'think' if args.thinking else 'no-think'})",
        "",
        f"Official  math_eval.py; n=32, max_tokens=16384, temperature=1, top_p=1, seed=42, thinking={bool(args.thinking)}.",
        "",
        "| Dataset | Correct/960 | Accuracy | Pass@1 | Pass@4 | Pass@8 | Pass@16 | Pass@32 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['correct_generations']}/960 | {row['sample_accuracy']:.6f} | "
            + " | ".join(f"{row[f'pass@{k}']:.6f}" for k in KS)
            + " |"
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
