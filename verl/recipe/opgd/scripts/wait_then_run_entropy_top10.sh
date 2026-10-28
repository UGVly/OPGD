#!/usr/bin/env bash
set -Eeuo pipefail
while true; do
  busy=$(/usr/bin/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000 {n++} END {print n+0}')
  if [[ "$busy" == "0" ]]; then break; fi
  echo "waiting_for_gpus busy=$busy $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sleep 60
done
exec /path/to/opgd/verl/recipe/opgd/scripts/run_entropy_top10_math_qwen3_4b.sh
