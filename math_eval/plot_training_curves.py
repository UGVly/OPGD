#!/usr/bin/env python3
"""Parse verl GRPO training logs and draw step-wise training curves as SVG."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"\bstep:(\d+)\b")
METRIC_RE = re.compile(r"([A-Za-z0-9_./@+-]+):(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")

COLORS = (
    "#1565c0",
    "#2e7d32",
    "#c62828",
    "#ef6c00",
    "#6a1b9a",
    "#00838f",
    "#ad1457",
    "#455a64",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("logs/qwen3_4b_grpo_deepmath_table4_bs128_n8_len16k_500steps_kl0.log"),
        help="Training log to parse.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/grpo_step_curves.svg"))
    parser.add_argument(
        "--also-output",
        action="append",
        type=Path,
        default=[],
        help="Additional SVG path(s) to receive the same plot.",
    )
    parser.add_argument("--csv-output", type=Path, default=Path("outputs/grpo_training_step_metrics_1_150.csv"))
    parser.add_argument("--max-step", type=int, default=None, help="Optional inclusive upper step bound.")
    parser.add_argument(
        "--base-eval-dir",
        type=Path,
        default=Path("math_eval/eval_outputs"),
        help="Directory containing AIME step0 JSONL eval outputs to merge into validation curves.",
    )
    parser.add_argument("--eval-model-prefix", default="Qwen3-4B-GRPO-DeepMath")
    parser.add_argument("--teacher-eval-prefix", default="Keven16-Qwen3-4B-Non-Thinking-RL-Math-Step500-n8")
    return parser.parse_args()


def parse_log(path: Path, max_step: int | None = None) -> list[dict[str, float]]:
    if not path.exists():
        raise SystemExit(f"Log file not found: {path}")

    by_step: dict[int, dict[str, float]] = {}
    with path.open(errors="ignore") as file:
        for raw_line in file:
            line = ANSI_RE.sub("", raw_line).replace("\r", "")
            match = STEP_RE.search(line)
            if not match:
                continue
            step = int(match.group(1))
            if max_step is not None and step > max_step:
                continue

            row: dict[str, float] = {}
            for key, value in METRIC_RE.findall(line):
                if key == "step":
                    continue
                row[key] = float(value)
            row["step"] = float(step)
            by_step[step] = row

    rows = [by_step[step] for step in sorted(by_step)]
    if not rows:
        raise SystemExit(f"No step metrics found in {path}")
    return rows


def summarize_first_k(path: Path, k: int = 8) -> tuple[float, float]:
    total_items = 0
    correct = 0
    pass_count = 0
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            accs = [bool(value) for value in item.get("acc_list", [])[:k]]
            if not accs:
                continue
            total_items += 1
            correct += sum(accs)
            pass_count += int(any(accs))

    if total_items == 0:
        raise SystemExit(f"No usable accuracy rows in {path}")
    return correct / (total_items * k), pass_count / total_items


def merge_base_aime_eval(rows: list[dict[str, float]], eval_dir: Path, model_prefix: str) -> list[dict[str, float]]:
    dataset_specs = {
        "aime24": "AIME2024",
        "aime25": "AIME2025",
    }
    step0: dict[str, float] = {"step": 0.0}
    for dataset, metric_name in dataset_specs.items():
        path = eval_dir / dataset / f"{model_prefix}-step0.jsonl"
        if not path.exists():
            continue
        mean_at_8, pass_at_8 = summarize_first_k(path, k=8)
        step0[f"val-core/{metric_name}/reward/mean@8"] = mean_at_8
        step0[f"val-core/{metric_name}/reward/best@8/mean"] = pass_at_8

    if len(step0) == 1:
        return rows

    filtered = [row for row in rows if int(row["step"]) != 0]
    return [step0, *filtered]


def collect_teacher_aime_refs(eval_dir: Path, model_prefix: str) -> list[tuple[str, float, str]]:
    dataset_specs = (
        ("aime24", "AIME2024", COLORS[0], COLORS[1]),
        ("aime25", "AIME2025", COLORS[2], COLORS[3]),
    )
    refs: list[tuple[str, float, str]] = []
    for dataset, metric_name, mean_color, best_color in dataset_specs:
        path = eval_dir / dataset / f"{model_prefix}.jsonl"
        if not path.exists():
            continue
        mean_at_8, pass_at_8 = summarize_first_k(path, k=8)
        refs.append((f"teacher {metric_name} mean@8", mean_at_8, mean_color))
        refs.append((f"teacher {metric_name} best@8", pass_at_8, best_color))
    return refs


def write_csv(rows: list[dict[str, float]], path: Path) -> None:
    keys = sorted({key for row in rows for key in row if key != "step"})
    header = ["step", *keys]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in rows:
            writer.writerow([format_value(row.get(key)) for key in header])


def format_value(value: float | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return f"{value:.10g}"


def scale(value: float, domain_min: float, domain_max: float, range_min: float, range_max: float) -> float:
    if domain_max == domain_min:
        return (range_min + range_max) / 2
    return range_min + (value - domain_min) / (domain_max - domain_min) * (range_max - range_min)


def format_tick(value: float) -> str:
    value = 0.0 if abs(value) < 1e-12 else value
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if 0 < abs(value) < 0.001:
        return f"{value:.1e}"
    if abs(value) < 1:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if abs(value) < 10:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.0f}"


def nice_domain(values: list[float], include_zero: bool = False, ymin: float | None = None, ymax: float | None = None) -> tuple[float, float]:
    low = min(values) if ymin is None else ymin
    high = max(values) if ymax is None else ymax
    if include_zero:
        low = min(low, 0.0)
        high = max(high, 0.0)
    if low == high:
        pad = abs(low) * 0.1 or 1.0
        return low - pad, high + pad
    pad = (high - low) * 0.08
    if ymin is None:
        low -= pad
    if ymax is None:
        high += pad
    if include_zero and low > 0:
        low = 0.0
    return low, high


def line_path(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    head, *tail = points
    parts = [f"M {head[0]:.1f} {head[1]:.1f}"]
    parts.extend(f"L {x:.1f} {y:.1f}" for x, y in tail)
    return " ".join(parts)


def panel(
    title: str,
    rows: list[dict[str, float]],
    series: list[tuple[str, str, str]],
    left: float,
    top: float,
    width: float,
    height: float,
    max_step: int,
    include_zero: bool = False,
    ymin: float | None = None,
    ymax: float | None = None,
    note: str | None = None,
    reference_lines: list[tuple[str, float, str]] | None = None,
) -> list[str]:
    right = left + width
    bottom = top + height
    values = [row[key] for row in rows for key, _, _ in series if key in row and math.isfinite(row[key])]
    if reference_lines:
        values.extend(value for _, value, _ in reference_lines if math.isfinite(value))
    if not values:
        return []

    y_min, y_max = nice_domain(values, include_zero=include_zero, ymin=ymin, ymax=ymax)
    lines = [
        f'<text x="{left}" y="{top - 14}" class="panel-title">{html.escape(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="#fff" stroke="#dfe5e8"/>',
    ]
    if note:
        lines.append(f'<text x="{right}" y="{top - 14}" text-anchor="end" class="note">{html.escape(note)}</text>')

    for i in range(5):
        tick_value = y_min + (y_max - y_min) * i / 4
        y = scale(tick_value, y_min, y_max, bottom, top)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{left - 9}" y="{y + 4:.1f}" text-anchor="end" class="small">{format_tick(tick_value)}</text>')

    x_ticks = [0, 25, 50, 75, 100, 125, max_step]
    for step in sorted(set(tick for tick in x_ticks if 0 <= tick <= max_step)):
        x = scale(step, 0, max_step, left, right)
        lines.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" class="grid"/>')
        lines.append(f'<text x="{x:.1f}" y="{bottom + 17}" text-anchor="middle" class="small">{step}</text>')

    legend_x = left + 12
    legend_y = top + 19
    for index, (_, label, color) in enumerate(series):
        lx = legend_x + index * 168
        lines.append(f'<line x1="{lx}" y1="{legend_y}" x2="{lx + 24}" y2="{legend_y}" stroke="{color}" stroke-width="2.8"/>')
        lines.append(f'<text x="{lx + 31}" y="{legend_y + 4}" class="legend">{html.escape(label)}</text>')

    if reference_lines:
        ref_legend_y = legend_y + 19
        for index, (label, _, color) in enumerate(reference_lines):
            lx = legend_x + (index % 2) * 280
            ly = ref_legend_y + (index // 2) * 17
            lines.append(
                f'<line x1="{lx}" y1="{ly}" x2="{lx + 24}" y2="{ly}" stroke="{color}" '
                'stroke-width="2.4" stroke-dasharray="7 5"/>'
            )
            lines.append(f'<text x="{lx + 31}" y="{ly + 4}" class="legend">{html.escape(label)}</text>')

    important_steps = {0, 1, max_step, 50, 100, 150}
    for key, label, color in series:
        points: list[tuple[int, float, float, float]] = []
        for row in rows:
            if key not in row:
                continue
            step = int(row["step"])
            value = row[key]
            x = scale(step, 0, max_step, left, right)
            y = scale(value, y_min, y_max, bottom, top)
            points.append((step, value, x, y))
        if not points:
            continue

        lines.append(
            f'<path d="{line_path([(x, y) for _, _, x, y in points])}" '
            f'fill="none" stroke="{color}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        show_all_points = len(points) <= 12
        for step, value, x, y in points:
            if show_all_points or step in important_steps or step % 25 == 0:
                title_text = f"{label}: step {step}, value {format_tick(value)}"
                lines.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6" fill="{color}" class="dot">'
                    f"<title>{html.escape(title_text)}</title></circle>"
                )

    for label, value, color in reference_lines or []:
        y = scale(value, y_min, y_max, bottom, top)
        lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" '
            f'stroke="{color}" stroke-width="2.2" stroke-dasharray="8 6" opacity="0.82">'
            f"<title>{html.escape(label)}: {format_tick(value)}</title></line>"
        )

    return lines


def last_value(rows: list[dict[str, float]], key: str) -> str:
    for row in reversed(rows):
        if key in row:
            return format_tick(row[key])
    return "n/a"


def write_svg(
    rows: list[dict[str, float]],
    path: Path,
    log_path: Path,
    teacher_refs: list[tuple[str, float, str]] | None = None,
) -> None:
    max_step = int(max(row["step"] for row in rows))
    min_step = int(min(row["step"] for row in rows))
    width = 1480.0
    height = 1660.0
    left_col = 88.0
    right_col = 796.0
    panel_width = 610.0
    panel_height = 245.0
    row_tops = [116.0, 460.0, 804.0, 1148.0]
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if min_step == 0:
        subtitle = (
            f"source: {log_path} + AIME step0 eval outputs | "
            f"training steps 1-{max_step}, AIME eval starts at 0 | generated {generated}"
        )
    else:
        subtitle = (
            f"source: {log_path} | logged steps {min_step}-{max_step} "
            f"({len(rows)} points) | generated {generated}"
        )
    summary = (
        f"final train score {last_value(rows, 'critic/score/mean')} | "
        f"AIME24 mean@8 {last_value(rows, 'val-core/AIME2024/reward/mean@8')} | "
        f"AIME25 mean@8 {last_value(rows, 'val-core/AIME2025/reward/mean@8')}"
    )

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(width)}" height="{int(height)}" viewBox="0 0 {int(width)} {int(height)}">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        "<style>"
        "text{font-family:Inter,Arial,sans-serif;fill:#263238}"
        ".title{font-size:24px;font-weight:700}.subtitle{font-size:12px;fill:#546e7a}"
        ".panel-title{font-size:16px;font-weight:700}.small{font-size:11px}.note{font-size:11px;fill:#607d8b}"
        ".grid{stroke:#edf1f3;stroke-width:1}.dot{stroke:#fff;stroke-width:1.2}.legend{font-size:12px}"
        "</style>",
        '<text x="88" y="38" class="title">GRPO Step-wise Training Curves, Step 0-150</text>',
        f'<text x="88" y="60" class="subtitle">{html.escape(subtitle)}</text>',
        f'<text x="88" y="80" class="subtitle">{html.escape(summary)}</text>',
    ]

    lines.extend(
        panel(
            "Training Reward",
            rows,
            [("critic/score/mean", "score mean", COLORS[0]), ("critic/rewards/mean", "reward mean", COLORS[1])],
            left_col,
            row_tops[0],
            panel_width,
            panel_height,
            max_step,
            ymin=0.45,
            ymax=0.92,
        )
    )
    lines.extend(
        panel(
            "Validation Reward",
            rows,
            [
                ("val-core/AIME2024/reward/mean@8", "AIME24 mean@8", COLORS[0]),
                ("val-core/AIME2025/reward/mean@8", "AIME25 mean@8", COLORS[2]),
                ("val-core/AIME2024/reward/best@8/mean", "AIME24 best@8", COLORS[1]),
                ("val-core/AIME2025/reward/best@8/mean", "AIME25 best@8", COLORS[3]),
            ],
            right_col,
            row_tops[0],
            panel_width,
            panel_height,
            max_step,
            ymin=0.0,
            ymax=1.0,
            note="base eval at 0; validation at 50/100/150; dashed teacher refs",
            reference_lines=teacher_refs,
        )
    )
    lines.extend(
        panel(
            "Response Length",
            rows,
            [
                ("response_length/mean", "mean", COLORS[0]),
                ("response_length/min", "min", COLORS[1]),
                ("response_length/max", "max", COLORS[2]),
            ],
            left_col,
            row_tops[1],
            panel_width,
            panel_height,
            max_step,
            ymin=0.0,
            ymax=17000.0,
        )
    )
    lines.extend(
        panel(
            "Response Clip Ratio",
            rows,
            [("response_length/clip_ratio", "clip ratio", COLORS[3]), ("response/aborted_ratio", "aborted ratio", COLORS[5])],
            right_col,
            row_tops[1],
            panel_width,
            panel_height,
            max_step,
            ymin=0.0,
            ymax=0.09,
        )
    )
    lines.extend(
        panel(
            "Actor Dynamics",
            rows,
            [
                ("actor/entropy", "entropy", COLORS[0]),
                ("actor/grad_norm", "grad norm", COLORS[2]),
                ("perf/mfu/actor", "actor MFU", COLORS[4]),
            ],
            left_col,
            row_tops[2],
            panel_width,
            panel_height,
            max_step,
            ymin=0.0,
            ymax=0.45,
        )
    )
    lines.extend(
        panel(
            "KL Loss",
            rows,
            [("actor/kl_loss", "actor kl loss", COLORS[4])],
            right_col,
            row_tops[2],
            panel_width,
            panel_height,
            max_step,
            ymin=0.0,
            ymax=0.0007,
        )
    )
    lines.extend(
        panel(
            "Timing",
            rows,
            [
                ("timing_s/step", "step", COLORS[0]),
                ("timing_s/gen", "generate", COLORS[1]),
                ("timing_s/update_actor", "update actor", COLORS[2]),
                ("timing_s/testing", "testing", COLORS[3]),
            ],
            left_col,
            row_tops[3],
            panel_width,
            panel_height,
            max_step,
            ymin=0.0,
            ymax=1300.0,
            note="seconds",
        )
    )
    lines.extend(
        panel(
            "Throughput",
            rows,
            [("perf/throughput", "throughput", COLORS[5])],
            right_col,
            row_tops[3],
            panel_width,
            panel_height,
            max_step,
        )
    )

    lines.append(f'<text x="{width / 2:.1f}" y="{height - 36:.1f}" text-anchor="middle" class="subtitle">x-axis starts at 0 for context; first logged training metrics are at step 1.</text>')
    lines.append("</svg>")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = parse_log(args.log, max_step=args.max_step)
    rows = merge_base_aime_eval(rows, args.base_eval_dir, args.eval_model_prefix)
    teacher_refs = collect_teacher_aime_refs(args.base_eval_dir, args.teacher_eval_prefix)
    write_csv(rows, args.csv_output)
    write_svg(rows, args.output, args.log, teacher_refs=teacher_refs)
    for extra_path in args.also_output:
        extra_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.output, extra_path)
    print(f"parsed_steps={len(rows)} range={int(rows[0]['step'])}-{int(rows[-1]['step'])}")
    print(f"svg={args.output}")
    for extra_path in args.also_output:
        print(f"svg={extra_path}")
    print(f"csv={args.csv_output}")


if __name__ == "__main__":
    main()
