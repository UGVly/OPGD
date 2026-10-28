#!/usr/bin/env python3
"""Summarize  math_eval outputs for Math500, AMC23, and OlympiadBench."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ALL_KS = (1, 4, 8, 16, 32)


def pass_at_k(n: int, correct: int, k: int) -> float:
    if k > n:
        raise ValueError(f"pass@{k} is undefined for n={n}")
    if correct <= 0:
        return 0.0
    if n - correct < k:
        return 1.0
    return 1.0 - math.comb(n - correct, k) / math.comb(n, k)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def summarize_rows(rows: list[dict[str, Any]], dataset: str, n: int, expected: int, source_files: list[Path]) -> dict[str, Any]:
    if len(rows) != expected:
        raise ValueError(f"{dataset}: expected {expected} questions, got {len(rows)}")
    correct_counts = []
    response_lengths = []
    extracted = 0
    for index, row in enumerate(rows, 1):
        acc = row.get("acc_list")
        responses = row.get("responses")
        predictions = row.get("pred_answers")
        if not isinstance(acc, list) or len(acc) != n:
            raise ValueError(f"{dataset} row {index}: acc_list length is not {n}")
        if not isinstance(responses, list) or len(responses) != n:
            raise ValueError(f"{dataset} row {index}: responses length is not {n}")
        if not isinstance(predictions, list) or len(predictions) != n:
            raise ValueError(f"{dataset} row {index}: pred_answers length is not {n}")
        if any(not isinstance(value, bool) for value in acc):
            raise ValueError(f"{dataset} row {index}: non-boolean accuracy value")
        correct_counts.append(sum(acc))
        extracted += sum(value is not None for value in predictions)
        response_lengths.extend(len(str(value)) for value in responses)

    total = expected * n
    result: dict[str, Any] = {
        "dataset": dataset,
        "questions": expected,
        "samples_per_question": n,
        "generations": total,
        "correct_generations": sum(correct_counts),
        "sample_accuracy": sum(correct_counts) / total,
        "boxed_extraction_rate": extracted / total,
        "mean_response_characters": sum(response_lengths) / len(response_lengths),
        "problem_correct_counts": correct_counts,
        "raw_outputs": [str(path) for path in source_files],
    }
    for k in ALL_KS:
        result[f"pass@{k}"] = None
        if k <= n:
            value = sum(pass_at_k(n, count, k) for count in correct_counts) / expected
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{dataset}: invalid pass@{k}: {value}")
            result[f"pass@{k}"] = value
    return result


def load_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("format") != "extra_math_eval_v1":
        raise ValueError(f"Unsupported manifest: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.eval_root.resolve()
    manifest = load_manifest(root)

    results = []
    for spec in manifest["datasets"]:
        files = [root / name for name in spec["outputs"]]
        rows: list[dict[str, Any]] = []
        for path in files:
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.extend(load_rows(path))
        results.append(
            summarize_rows(
                rows,
                spec["name"],
                int(spec["samples_per_question"]),
                int(spec["questions"]),
                files,
            )
        )

    payload = {
        "model": manifest["model"],
        "evaluation": manifest["evaluation"],
        "protocol": manifest["protocol"],
        "pass_k": list(ALL_KS),
        "results": results,
    }
    (root / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    columns = [
        "dataset", "questions", "samples_per_question", "generations",
        "correct_generations", "sample_accuracy", *[f"pass@{k}" for k in ALL_KS],
        "boxed_extraction_rate", "mean_response_characters", "raw_outputs",
    ]
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key) for key in columns})

    lines = [
        f"# {manifest['model']} extra math evaluation",
        "",
        f"{manifest['evaluation']}; {json.dumps(manifest['protocol'], ensure_ascii=False, sort_keys=True)}",
        "",
        "| Dataset | Questions | n | Correct | Accuracy | Pass@1 | Pass@4 | Pass@8 | Pass@16 | Pass@32 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        values = []
        for k in ALL_KS:
            value = result[f"pass@{k}"]
            values.append("n/a" if value is None else f"{value:.6f}")
        lines.append(
            f"| {result['dataset']} | {result['questions']} | {result['samples_per_question']} | "
            f"{result['correct_generations']}/{result['generations']} | {result['sample_accuracy']:.6f} | "
            + " | ".join(values)
            + " |"
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
