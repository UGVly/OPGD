#!/usr/bin/env bash
set -euo pipefail

# 1.7B Math GRPO baseline with a 16K response budget.
# The underlying launcher keeps all baseline settings; only length and budget differ.
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
export TAG="${TAG:-grpo_math_qwen3_1p7b_fullft4_b128_n8_lr1e6_16k_s200_${STAMP}}"
export RUN_DIR="${RUN_DIR:-/path/to/runs/${TAG}}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-/path/to/runs/checkpoints/${TAG}}"
export RAY_TMP="${RAY_TMP:-/tmp/${TAG}_ray}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${TAG}_wandb}"
export MAX_RESPONSE_LENGTH=16384
export TOKEN_BUDGET="${TOKEN_BUDGET:-65536}"
export TOTAL_STEPS="${TOTAL_STEPS:-200}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export MAX_CKPTS="${MAX_CKPTS:-20}"

exec "$(dirname "$0")/run_grpo_math_qwen3_1p7b.sh"
