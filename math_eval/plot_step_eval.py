#!/usr/bin/env python3
"""Summarize math eval JSONL files and draw step curves."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATASETS = ("aime24", "aime25", "hmmt25_feb", "hmmt25_nov")
COLORS = ("#1565c0", "#2e7d32", "#ef6c00", "#6a1b9a", "#00838f", "#ad1457")


@dataclass(frozen=True)
class EvalPoint:
    dataset: str
    step: int
    source_file: Path
    num_items: int
    total_predictions: int
    correct_predictions: int
    accuracy: float
    pass_at_k: float
    avg_responses_per_item: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, default=Path("eval_outputs"))
    parser.add_argument("--model-prefix", default="Qwen3-4B-GRPO-DeepMath")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--steps", nargs="*", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("../outputs/grpo_step_eval_curves.svg"))
    parser.add_argument("--summary-output", type=Path, default=Path("../outputs/grpo_step_eval_summary.csv"))
    parser.add_argument(
        "--allow-missing-step0",
        action="store_true",
        help="Draw partial curves even when the base-model step0 eval is absent.",
    )
    return parser.parse_args()


def summarize_jsonl(path: Path, dataset: str, step: int) -> EvalPoint:
    num_items = 0
    total_predictions = 0
    correct_predictions = 0
    pass_at_k_count = 0

    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            accs = [bool(value) for value in item.get("acc_list", [])]
            num_items += 1
            total_predictions += len(accs)
            correct_predictions += sum(accs)
            pass_at_k_count += int(any(accs))

    accuracy = correct_predictions / total_predictions if total_predictions else 0.0
    pass_at_k = pass_at_k_count / num_items if num_items else 0.0
    avg_responses = total_predictions / num_items if num_items else 0.0
    return EvalPoint(
        dataset=dataset,
        step=step,
        source_file=path,
        num_items=num_items,
        total_predictions=total_predictions,
        correct_predictions=correct_predictions,
        accuracy=accuracy,
        pass_at_k=pass_at_k,
        avg_responses_per_item=avg_responses,
    )


def result_file_for_step(dataset_dir: Path, model_prefix: str, step: int) -> Path | None:
    exact = dataset_dir / f"{model_prefix}-step{step}.jsonl"
    if exact.exists():
        return exact

    matches = sorted(dataset_dir.glob(f"{model_prefix}-step{step}-*.jsonl"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(str(path) for path in matches)
        raise SystemExit(f"Ambiguous result files for step {step}: {names}")
    return None


def discover_steps(eval_dir: Path, datasets: list[str], model_prefix: str) -> list[int]:
    step_re = re.compile(rf"^{re.escape(model_prefix)}-step(\d+)(?:-.+)?\.jsonl$")
    steps: set[int] = set()
    for dataset in datasets:
        dataset_dir = eval_dir / dataset
        if not dataset_dir.exists():
            continue
        for path in dataset_dir.glob(f"{model_prefix}-step*.jsonl"):
            match = step_re.match(path.name)
            if match:
                steps.add(int(match.group(1)))
    return sorted(steps)


def collect_points(args: argparse.Namespace) -> list[EvalPoint]:
    steps = args.steps if args.steps is not None and len(args.steps) > 0 else discover_steps(
        args.eval_dir, args.datasets, args.model_prefix
    )
    if not steps:
        raise SystemExit(f"No eval files found under {args.eval_dir} for prefix {args.model_prefix!r}")
    if 0 not in steps and not args.allow_missing_step0:
        raise SystemExit(
            "Missing step0 eval result. Run base-model eval first, or pass --allow-missing-step0 for a partial plot."
        )

    points: list[EvalPoint] = []
    for dataset in args.datasets:
        dataset_dir = args.eval_dir / dataset
        for step in steps:
            result_file = result_file_for_step(dataset_dir, args.model_prefix, step)
            if result_file is not None:
                points.append(summarize_jsonl(result_file, dataset, step))
    if not points:
        raise SystemExit("No usable eval result files found.")
    return sorted(points, key=lambda point: (point.dataset, point.step))


def write_summary(points: list[EvalPoint], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "dataset",
                "step",
                "source_file",
                "num_items",
                "total_predictions",
                "correct_predictions",
                "accuracy",
                "pass_at_k",
                "avg_responses_per_item",
            ]
        )
        for point in points:
            writer.writerow(
                [
                    point.dataset,
                    point.step,
                    point.source_file,
                    point.num_items,
                    point.total_predictions,
                    point.correct_predictions,
                    f"{point.accuracy:.8f}",
                    f"{point.pass_at_k:.8f}",
                    f"{point.avg_responses_per_item:.4f}",
                ]
            )


def scale(value: float, domain_min: float, domain_max: float, range_min: float, range_max: float) -> float:
    if domain_max == domain_min:
        return (range_min + range_max) / 2
    ratio = (value - domain_min) / (domain_max - domain_min)
    return range_min + ratio * (range_max - range_min)


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def draw_panel(
    metric_name: str,
    metric_attr: str,
    points: list[EvalPoint],
    datasets: list[str],
    steps: list[int],
    top: float,
    height: float,
    width: float,
) -> list[str]:
    left = 82.0
    right = width - 38.0
    bottom = top + height
    x_min = min(steps)
    x_max = max(steps)
    if x_min == x_max:
        x_min -= 1
        x_max += 1

    lines = [
        f'<text x="{left}" y="{top - 16}" class="panel-title">{html.escape(metric_name)}</text>',
        f'<rect x="{left}" y="{top}" width="{right-left}" height="{height}" fill="#fff" stroke="#dfe5e8"/>',
    ]

    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        y = scale(tick, 0, 1, bottom, top)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="small">{tick:.2f}</text>')

    x_ticks = sorted(set([0, *steps]))
    for step in x_ticks:
        if step < min(steps) or step > max(steps):
            continue
        x = scale(step, x_min, x_max, left, right)
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" class="grid"/>')
        lines.append(f'<text x="{x:.1f}" y="{bottom + 20}" text-anchor="middle" class="small">{step}</text>')

    for index, dataset in enumerate(datasets):
        dataset_points = [point for point in points if point.dataset == dataset]
        if not dataset_points:
            continue
        color = COLORS[index % len(COLORS)]
        coords = [
            (
                scale(point.step, x_min, x_max, left, right),
                scale(getattr(point, metric_attr), 0, 1, bottom, top),
            )
            for point in dataset_points
        ]
        lines.append(f'<polyline points="{polyline(coords)}" class="line" stroke="{color}"/>')
        for point, (x, y) in zip(dataset_points, coords):
            value = getattr(point, metric_attr)
            title = f"{dataset} {metric_name} step={point.step} value={value:.4f}"
            lines.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" class="dot">'
                f"<title>{html.escape(title)}</title></circle>"
            )

    return lines


def write_svg(points: list[EvalPoint], path: Path, model_prefix: str, allow_missing_step0: bool) -> None:
    width = 1120.0
    height = 720.0
    datasets = sorted({point.dataset for point in points})
    steps = sorted({point.step for point in points})
    title = f"{model_prefix} Math Eval Curves"
    note = "Step 0 is the base model eval." if 0 in steps else "Partial plot: step0 eval is missing."

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(width)}" height="{int(height)}" viewBox="0 0 {int(width)} {int(height)}">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        "<style>"
        "text{font-family:Inter,Arial,sans-serif;fill:#263238}"
        ".title{font-size:22px;font-weight:700}.panel-title{font-size:16px;font-weight:700}"
        ".small{font-size:12px}.note{font-size:12px;fill:#546e7a}.grid{stroke:#eceff1;stroke-width:1}"
        ".line{fill:none;stroke-width:2.8}.dot{stroke:#fff;stroke-width:1.4}.legend{font-size:13px}"
        "</style>",
        f'<text x="82" y="34" class="title">{html.escape(title)}</text>',
        f'<text x="82" y="55" class="note">{html.escape(note)} Summary generated from math_eval/eval_outputs.</text>',
    ]

    lines.extend(draw_panel("Accuracy", "accuracy", points, datasets, steps, top=96, height=230, width=width))
    lines.extend(draw_panel("Pass@k", "pass_at_k", points, datasets, steps, top=410, height=230, width=width))

    legend_x = 82
    legend_y = 674
    for index, dataset in enumerate(datasets):
        x = legend_x + index * 180
        color = COLORS[index % len(COLORS)]
        lines.append(f'<line x1="{x}" y1="{legend_y}" x2="{x + 26}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        lines.append(f'<text x="{x + 34}" y="{legend_y + 4}" class="legend">{html.escape(dataset)}</text>')

    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    points = collect_points(args)
    write_summary(points, args.summary_output)
    write_svg(points, args.output, args.model_prefix, args.allow_missing_step0)
    print(f"Wrote {args.summary_output}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

