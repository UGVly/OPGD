#!/usr/bin/env python3
"""Compare a OPGD W&B run with the validated OPGDv3 reference checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import sys

REFERENCE_RUN = "scut2024/OPGD/opdone8eb47d3748a33a137"
METRICS = {
    "aime2024": "val-core/AIME2024/reward/mean@32",
    "aime2025": "val-core/AIME2025/reward/mean@32",
}
REFERENCE = {
    10: {"aime2024": 0.28020833333333334, "aime2025": 0.225},
    20: {"aime2024": 0.4375, "aime2025": 0.33958333333333335},
    30: {"aime2024": 0.5958333333333333, "aime2025": 0.4822916666666667},
    40: {"aime2024": 0.596875, "aime2025": 0.496875},
    50: {"aime2024": 0.5895833333333333, "aime2025": 0.4875},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run", nargs="?", default=REFERENCE_RUN, help="W&B entity/project/run_id"
    )
    parser.add_argument("--tolerance", type=float, default=0.03)
    parser.add_argument(
        "--through-step", type=int, default=50, choices=sorted(REFERENCE)
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.tolerance) or args.tolerance < 0:
        raise SystemExit("--tolerance must be finite and non-negative")
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "wandb is required: install it in the veRL environment"
        ) from exc

    run = wandb.Api().run(args.run)
    wanted_steps = [step for step in sorted(REFERENCE) if step <= args.through_step]
    observed = {}
    for row in run.scan_history(keys=["_step", *METRICS.values()]):
        step = row.get("_step")
        if step not in wanted_steps:
            continue
        if all(row.get(metric) is not None for metric in METRICS.values()):
            observed[int(step)] = {
                name: float(row[metric]) for name, metric in METRICS.items()
            }

    checks = []
    passed = True
    for step in wanted_steps:
        if step not in observed:
            passed = False
            checks.append({"step": step, "status": "missing"})
            continue
        check = {"step": step, "status": "ok", "metrics": {}}
        for name, reference_value in REFERENCE[step].items():
            actual = observed[step][name]
            error = abs(actual - reference_value)
            within_tolerance = error <= args.tolerance
            passed &= within_tolerance
            if not within_tolerance:
                check["status"] = "outside_tolerance"
            check["metrics"][name] = {
                "actual": actual,
                "reference": reference_value,
                "absolute_error": error,
                "within_tolerance": within_tolerance,
            }
        checks.append(check)

    print(
        json.dumps(
            {
                "run": args.run,
                "reference_run": REFERENCE_RUN,
                "tolerance": args.tolerance,
                "through_step": args.through_step,
                "passed": passed,
                "checks": checks,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
