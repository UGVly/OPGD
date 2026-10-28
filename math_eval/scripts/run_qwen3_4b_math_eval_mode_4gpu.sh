#!/usr/bin/env bash
set -Eeuo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REF_REPO="${REF_REPO:-/path/to/reference}"
MATH_EVAL="${MATH_EVAL:-$REF_REPO/math_eval/eval_math.py}"
MODEL_PATH="${MODEL_PATH:-$REF_REPO/models/models/Qwen/Qwen3-4B}"
SUMMARY_SCRIPT="${SUMMARY_SCRIPT:-$RUN_ROOT/records/summarize_qwen3_4b_math_passk.py}"
DATASETS=(aime24 aime25 hmmt25_feb hmmt25_nov)
GPUS=(0 1 2 3)
ENABLE_THINKING="${ENABLE_THINKING:-0}"
THINKING_ARGS=()
THINKING=false
if [[ "$ENABLE_THINKING" == "1" ]]; then THINKING_ARGS=(--enable_thinking); THINKING=true; fi

if [[ -e "$RUN_ROOT/status" || -e "$RUN_ROOT/manifest.json" ]]; then
  echo "Refusing to overwrite existing math evaluation: $RUN_ROOT" >&2
  exit 84
fi
mkdir -p "$RUN_ROOT"/{raw,logs,records}
printf '%s\n' "$$" > "$RUN_ROOT/controller.pid"

for required in "$PYTHON_BIN" "$MATH_EVAL" "$MODEL_PATH" "$SUMMARY_SCRIPT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    printf '%s\n' failed > "$RUN_ROOT/status"
    exit 81
  fi
done
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -Eq '[0-9]'; then
  echo "Refusing to start: one or more GPUs already have compute processes." >&2
  printf '%s\n' refused_busy > "$RUN_ROOT/status"
  exit 83
fi

cp -p "$SUMMARY_SCRIPT" "$RUN_ROOT/records/"
cat > "$RUN_ROOT/manifest.json" <<EOF
{
  "format": "qwen3_4b_math_eval_v1",
  "model": "$(basename "$MODEL_PATH")",
  "model_path": "$MODEL_PATH",
  "evaluation": " official math_eval.py",
  "thinking": $THINKING,
  "datasets": ["aime24", "aime25", "hmmt25_feb", "hmmt25_nov"],
  "questions_per_dataset": 30,
  "samples_per_question": 32,
  "max_tokens": 16384,
  "temperature": 1.0,
  "top_p": 1.0,
  "seed": 42,
  "tensor_parallel_size": 1,
  "gpus": [0, 1, 2, 3]
}
EOF
printf '%s\n' running > "$RUN_ROOT/status"
{
  printf 'model=%s\n' "$MODEL_PATH"
  printf 'evaluator=%s\n' "$MATH_EVAL"
  printf 'datasets=%s\n' "${DATASETS[*]}"
  printf 'gpus=%s\n' "${GPUS[*]}"
  printf 'n=32\nmax_tokens=16384\ntemperature=1.0\ntop_p=1.0\nseed=42\nthinking=%s\n' "$THINKING"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$RUN_ROOT/records/protocol.txt"

pids=()
for index in "${!DATASETS[@]}"; do
  dataset="${DATASETS[$index]}"; gpu="${GPUS[$index]}"
  input="$REF_REPO/data/$dataset/test.jsonl"; output="$RUN_ROOT/raw/$dataset.jsonl"; log="$RUN_ROOT/logs/$dataset.log"
  if [[ ! -f "$input" ]]; then echo "Missing dataset: $input" >&2; printf '%s\n' failed > "$RUN_ROOT/status"; exit 82; fi
  (
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONUNBUFFERED=1 \
      "$PYTHON_BIN" "$MATH_EVAL" --input_file "$input" --model_path "$MODEL_PATH" --output_file "$output" \
      --max_tokens 16384 --temperature 1.0 --top_p 1.0 --max_num_seqs 256 --n 32 --seed 42 \
      "${THINKING_ARGS[@]}" > "$log" 2>&1
    rc=$?; printf '%s\n' "$rc" > "$RUN_ROOT/raw/$dataset.exit_code"; exit "$rc"
  ) &
  pid=$!; pids+=("$pid"); printf '%s\n' "$pid" > "$RUN_ROOT/raw/$dataset.pid"
done

overall=0
for pid in "${pids[@]}"; do wait "$pid" || overall=1; done
for dataset in "${DATASETS[@]}"; do
  code_file="$RUN_ROOT/raw/$dataset.exit_code"
  if [[ ! -f "$code_file" || "$(<"$code_file")" != 0 ]]; then overall=1; fi
done
if [[ "$overall" -ne 0 ]]; then
  printf '%s\n' failed > "$RUN_ROOT/status"; printf '%s\n' "$overall" > "$RUN_ROOT/controller.exit"; exit "$overall"
fi

SUMMARY_THINK_ARG=()
if [[ "$ENABLE_THINKING" == "1" ]]; then SUMMARY_THINK_ARG=(--thinking); fi
"$PYTHON_BIN" "$SUMMARY_SCRIPT" --eval-root "$RUN_ROOT" --model "$(basename "$MODEL_PATH")" "${SUMMARY_THINK_ARG[@]}" > "$RUN_ROOT/logs/summarize.log" 2>&1
rc=$?; printf '%s\n' "$rc" > "$RUN_ROOT/controller.exit"
if [[ "$rc" -ne 0 ]]; then printf '%s\n' failed > "$RUN_ROOT/status"; exit "$rc"; fi
printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN_ROOT/records/protocol.txt"
printf '%s\n' complete > "$RUN_ROOT/status"
