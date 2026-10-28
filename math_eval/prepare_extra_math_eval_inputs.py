#!/usr/bin/env python3
"""Normalize Math500, AMC23, and OlympiadBench for  math_eval.py.

The official evaluator consumes JSONL records with ``problem`` and
``answer`` fields.  Math500 and AMC23 already have that shape; OlympiadBench
uses ``question`` and ``final_answer``, so it is converted here while keeping
the source metadata in each normalized record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_basic(rows: list[dict[str, Any]], dataset: str) -> list[dict[str, Any]]:
    normalized = []
    for index, row in enumerate(rows):
        if "problem" not in row or "answer" not in row:
            raise ValueError(f"{dataset} row {index + 1} lacks problem/answer")
        item = dict(row)
        item["problem"] = str(row["problem"])
        item["answer"] = str(row["answer"])
        item["_source_dataset"] = dataset
        normalized.append(item)
    return normalized


def normalize_olympiad(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for index, row in enumerate(rows):
        question = row.get("question")
        context = row.get("context")
        answers = row.get("final_answer")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"olympiadbench row {index + 1} lacks question")
        if not isinstance(answers, list) or len(answers) != 1:
            raise ValueError(
                f"olympiadbench row {index + 1}: expected one final_answer entry, got {answers!r}"
            )
        # OlympiadBench wraps many answers in dollar delimiters.  The 
        # evaluator adds its own boxed delimiters, so remove only those outer
        # TeX math markers and preserve the mathematical expression itself.
        answer = str(answers[0]).strip().replace("$", "")
        item = dict(row)
        item["problem"] = (
            f"{context.strip()}\n\n{question}" if isinstance(context, str) and context.strip() else question
        )
        item["answer"] = answer
        item["_source_dataset"] = "olympiadbench"
        item["_source_final_answer"] = answers
        normalized.append(item)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--math500-input", type=Path, required=True)
    parser.add_argument("--amc23-input", type=Path, required=True)
    parser.add_argument("--olympiadbench-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    inputs = {
        "math500": args.math500_input.resolve(),
        "amc23": args.amc23_input.resolve(),
        "olympiadbench": args.olympiadbench_input.resolve(),
    }
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")

    datasets = {
        "math500": normalize_basic(read_jsonl(inputs["math500"]), "math500"),
        "amc23": normalize_basic(read_jsonl(inputs["amc23"]), "amc23"),
        "olympiadbench": normalize_olympiad(read_jsonl(inputs["olympiadbench"])),
    }
    if len(datasets["math500"]) != 500:
        raise ValueError(f"Math500 expected 500 rows, got {len(datasets['math500'])}")
    if len(datasets["amc23"]) != 40:
        raise ValueError(f"AMC23 expected 40 rows, got {len(datasets['amc23'])}")
    if len(datasets["olympiadbench"]) != 675:
        raise ValueError(
            f"OlympiadBench expected 675 rows, got {len(datasets['olympiadbench'])}"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in datasets.items():
        write_jsonl(output_dir / f"{name}.jsonl", rows)

    manifest = {
        "format": "math_eval_jsonl_v1",
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path), "rows": len(datasets[name])}
            for name, path in inputs.items()
        },
        "outputs": {
            name: {"path": str((output_dir / f"{name}.jsonl").resolve()), "rows": len(rows)}
            for name, rows in datasets.items()
        },
        "olympiadbench_answer_conversion": {
            "source_field": "final_answer[0]",
            "removed_characters": ["$"],
            "multiple_answer_rows": sum(bool(row.get("is_multiple_answer")) for row in datasets["olympiadbench"]),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
