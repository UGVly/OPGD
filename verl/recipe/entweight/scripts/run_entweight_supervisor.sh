#!/usr/bin/env bash
# Restart the entropy-weighted GRPO run after a crash, resuming from the last checkpoint.
set -u

LAUNCHER="${LAUNCHER:-/path/to/opgd/verl/recipe/entweight/scripts/run_entweight_math_qwen3_4b_a8008_s300.sh}"
TAG="${TAG_OVERRIDE:?TAG_OVERRIDE is required}"
TOTAL_STEPS="${TOTAL_STEPS:-300}"
CKPT="/path/to/opgd/verl/checkpoints/entweight/$TAG"
LOG="/path/to/opgd/logs/entweight_${TAG}.log"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-5}"

mkdir -p "$(dirname "$LOG")"
export TAG_OVERRIDE="$TAG"
export OUT_OVERRIDE="$CKPT"

for i in $(seq 1 "$MAX_ATTEMPTS"); do
  echo "=== SUPERVISOR ATTEMPT $i START $(date -u +%FT%TZ) ===" | tee -a "$LOG"
  bash "$LAUNCHER" 2>&1 | tee -a "$LOG"
  echo "=== SUPERVISOR ATTEMPT $i EXIT rc=${PIPESTATUS[0]} $(date -u +%FT%TZ) ===" | tee -a "$LOG"
  cur=$(cat "$CKPT/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
  echo "=== checkpoint now: $cur ===" | tee -a "$LOG"
  if [ "${cur:-0}" -ge "$TOTAL_STEPS" ]; then
    echo "=== TRAINING COMPLETE ===" | tee -a "$LOG"
    break
  fi
  sleep 60
done
