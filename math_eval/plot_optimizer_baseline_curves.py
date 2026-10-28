#!/usr/bin/env python3
"""Plot optimizer baseline curves from verl GRPO logs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"\bstep:(\d+)\b")
METRIC_RE = re.compile(r"([A-Za-z0-9_./@+-]+):(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")

METRICS = {
    "AIME2024 mean@8": "val-core/AIME2024/reward/mean@8",
    "AIME2024 best@8": "val-core/AIME2024/reward/best@8/mean",
    "AIME2025 mean@8": "val-core/AIME2025/reward/mean@8",
    "AIME2025 best@8": "val-core/AIME2025/reward/best@8/mean",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-log",
        type=Path,
        default=Path("logs/qwen3_4b_grpo_deepmath_table4_bs128_n8_len16k_500steps_kl0.log"),
    )
    parser.add_argument(
        "--optimizer-log",
        type=Path,
        default=Path("logs/grpo_qwen3_4b_muon_rmsprop_baselines_tmux_20260628_1039.log"),
    )
    parser.add_argument("--out-prefix", type=Path, default=Path("outputs/optimizer_baseline_curves"))
    return parser.parse_args()


def parse_log(path: Path, method: str, max_step: int | None = None) -> list[dict[str, float | str]]:
    if not path.exists():
        raise FileNotFoundError(path)

    rows_by_step: dict[int, dict[str, float | str]] = {}
    with path.open(errors="ignore") as file:
        for raw_line in file:
            line = ANSI_RE.sub("", raw_line).replace("\r", "")
            step_match = STEP_RE.search(line)
            if not step_match:
                continue
            step = int(step_match.group(1))
            if max_step is not None and step > max_step:
                continue

            metrics = dict(METRIC_RE.findall(line))
            row: dict[str, float | str] = {"method": method, "step": step}
            for label, key in METRICS.items():
                if key in metrics:
                    row[label] = float(metrics[key])
            if len(row) > 2:
                rows_by_step[step] = row

    return [rows_by_step[step] for step in sorted(rows_by_step)]


def main() -> None:
    args = parse_args()

    rows = []
    baseline_rows = parse_log(args.baseline_log, "GRPO baseline")
    rows.extend(baseline_rows)

    # The optimizer queue log currently contains Muon first. RMSprop will be parsed
    # by this script later if the queue reaches that run and writes validation rows.
    rows.extend(parse_log(args.optimizer_log, "Muon", max_step=None))

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No validation metrics found.")

    # Keep the comparison fair by plotting only steps where the baseline has data.
    common_steps = sorted(set(df.loc[df["method"] == "GRPO baseline", "step"]))
    df = df[df["step"].isin(common_steps)].copy()

    long_df = df.melt(id_vars=["method", "step"], value_vars=list(METRICS), var_name="metric", value_name="score")
    long_df = long_df.dropna(subset=["score"])

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_prefix.with_suffix(".csv")
    png_path = args.out_prefix.with_suffix(".png")
    pdf_path = args.out_prefix.with_suffix(".pdf")
    long_df.to_csv(csv_path, index=False)

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    palette = {
        "GRPO baseline": "#2563eb",
        "Muon": "#dc2626",
        "RMSprop": "#16a34a",
    }

    for ax, metric in zip(axes, METRICS):
        metric_df = long_df[long_df["metric"] == metric]
        sns.lineplot(
            data=metric_df,
            x="step",
            y="score",
            hue="method",
            style="method",
            markers=True,
            dashes=False,
            linewidth=2.6,
            markersize=9,
            palette=palette,
            ax=ax,
        )
        ax.set_title(metric)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Reward")
        ax.set_ylim(0.15, 0.85)
        ax.set_xticks(common_steps)
        ax.legend_.remove()

        for _, row in metric_df.iterrows():
            ax.annotate(
                f"{row['score']:.3f}",
                (row["step"], row["score"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9,
            )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    fig.suptitle("GRPO Optimizer Baseline Comparison", y=0.995, fontsize=22, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
