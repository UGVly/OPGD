#!/usr/bin/env bash
set -Eeuo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
REF_REPO="${REF_REPO:-/path/to/reference}"
MATH_EVAL="${MATH_EVAL:-$REF_REPO/math_eval/eval_math.py}"
MODEL_PATH="${MODEL_PATH:-$REF_REPO/models/models/Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500}"
PEFT_DATA_ROOT="${PEFT_DATA_ROOT:-/path/to/math-eval-data}"
SCRIPT_ROOT="${SCRIPT_ROOT:-/path/to/opgd/math_eval}"
PREP_SCRIPT="${PREP_SCRIPT:-$SCRIPT_ROOT/prepare_extra_math_eval_inputs.py}"
SUMMARY_SCRIPT="${SUMMARY_SCRIPT:-$SCRIPT_ROOT/summarize_extra_math_passk.py}"
EXTRA_ROOT="${EXTRA_ROOT:-$RUN_ROOT/extra_math_eval}"

# Fixed requested protocol: Math500/OlympiadBench n=4, AMC23 n=32.
MATH500_N=4
OLYMPIAD_N=4
AMC_N=32
THINKING_ARGS=()
THINKING=false
if [[ "${ENABLE_THINKING:-0}" == "1" ]]; then
  THINKING_ARGS=(--enable_thinking)
  THINKING=true
fi
MODEL_NAME="$(basename "$MODEL_PATH")"

if [[ -e "$EXTRA_ROOT/status" || -e "$EXTRA_ROOT/manifest.json" ]]; then
  echo "Refusing to overwrite an existing extra evaluation: $EXTRA_ROOT" >&2
  exit 84
fi
mkdir -p "$EXTRA_ROOT"/{raw,logs,records,data}

for required in "$PYTHON_BIN" "$MATH_EVAL" "$MODEL_PATH" "$PREP_SCRIPT" "$SUMMARY_SCRIPT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    printf '%s\n' failed > "$EXTRA_ROOT/status"
    exit 81
  fi
done

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -Eq '[0-9]'; then
  echo "Refusing to start: one or more GPUs already have compute processes." >&2
  printf '%s\n' refused_busy > "$EXTRA_ROOT/status"
  exit 83
fi

cp -p "$PREP_SCRIPT" "$SUMMARY_SCRIPT" "$EXTRA_ROOT/records/"

"$PYTHON_BIN" "$PREP_SCRIPT" \
  --math500-input "$PEFT_DATA_ROOT/math500/test.jsonl" \
  --amc23-input "$PEFT_DATA_ROOT/amc23/test.jsonl" \
  --olympiadbench-input "$PEFT_DATA_ROOT/olympiadbench/test.jsonl" \
  --output-dir "$EXTRA_ROOT/data" > "$EXTRA_ROOT/logs/prepare.log" 2>&1

cat > "$EXTRA_ROOT/manifest.json" <<EOF
{
  "format": "extra_math_eval_v1",
  "model": "${MODEL_NAME}",
  "evaluation": " official math_eval.py",
  "protocol": {
    "max_tokens": 16384,
    "temperature": 1.0,
    "top_p": 1.0,
    "seed": 42,
  "thinking": ${THINKING},
    "tensor_parallel_size": 1,
    "gpu_count": 4,
    "amc_sampling_count": ${AMC_N}
  },
  "datasets": [
    {"name": "math500", "questions": 500, "samples_per_question": ${MATH500_N}, "outputs": ["raw/math500.jsonl"]},
    {"name": "amc23", "questions": 40, "samples_per_question": ${AMC_N}, "outputs": ["raw/amc23.part0.jsonl", "raw/amc23.part1.jsonl"]},
    {"name": "olympiadbench", "questions": 675, "samples_per_question": ${OLYMPIAD_N}, "outputs": ["raw/olympiadbench.jsonl"]}
  ],
  "normalized_data": "data/manifest.json"
}
EOF

printf '%s\n' running > "$EXTRA_ROOT/status"
{
  printf 'model=%s\n' "$MODEL_PATH"
  printf 'evaluator=%s\n' "$MATH_EVAL"
  printf 'data_root=%s\n' "$PEFT_DATA_ROOT"
  printf 'datasets=Math500(n=%s), AMC23(n=%s), OlympiadBench(n=%s)\n' "$MATH500_N" "$AMC_N" "$OLYMPIAD_N"
  printf 'gpus=0,1,2,3; tensor_parallel_size=1\n'
  printf 'max_tokens=16384\ntemperature=1.0\ntop_p=1.0\nseed=42\nthinking=%s\n' "$THINKING"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$EXTRA_ROOT/records/protocol.txt"

datasets=(math500 olympiadbench amc23.part0 amc23.part1)
gpus=(0 1 2 3)
inputs=(
  "$EXTRA_ROOT/data/math500.jsonl"
  "$EXTRA_ROOT/data/olympiadbench.jsonl"
  "$EXTRA_ROOT/data/amc23.jsonl"
  "$EXTRA_ROOT/data/amc23.jsonl"
)
outputs=(
  "$EXTRA_ROOT/raw/math500.jsonl"
  "$EXTRA_ROOT/raw/olympiadbench.jsonl"
  "$EXTRA_ROOT/raw/amc23.part0.jsonl"
  "$EXTRA_ROOT/raw/amc23.part1.jsonl"
)
logs=(
  "$EXTRA_ROOT/logs/math500.log"
  "$EXTRA_ROOT/logs/olympiadbench.log"
  "$EXTRA_ROOT/logs/amc23.part0.log"
  "$EXTRA_ROOT/logs/amc23.part1.log"
)

for output in "${outputs[@]}"; do
  if [[ -e "$output" ]]; then
    echo "Refusing to overwrite existing output: $output" >&2
    printf '%s\n' refused_existing_output > "$EXTRA_ROOT/status"
    exit 84
  fi
done

pids=()
for index in "${!datasets[@]}"; do
  dataset="${datasets[$index]}"
  gpu="${gpus[$index]}"
  input="${inputs[$index]}"
  output="${outputs[$index]}"
  log="${logs[$index]}"
  begin_args=()
  if [[ "$dataset" == "amc23.part0" ]]; then
    begin_args=(--begin_idx 0 --end_idx 20)
  elif [[ "$dataset" == "amc23.part1" ]]; then
    begin_args=(--begin_idx 20 --end_idx 40)
  fi
  if [[ "$dataset" == "math500" ]]; then
    n="$MATH500_N"
  elif [[ "$dataset" == "olympiadbench" ]]; then
    n="$OLYMPIAD_N"
  else
    n="$AMC_N"
  fi
  (
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONUNBUFFERED=1 \
      "$PYTHON_BIN" "$MATH_EVAL" \
      --input_file "$input" \
      --model_path "$MODEL_PATH" \
      --output_file "$output" \
      --max_tokens 16384 \
      --temperature 1.0 \
      --top_p 1.0 \
      --max_num_seqs 256 \
      --n "$n" \
      --seed 42 "${THINKING_ARGS[@]}" "${begin_args[@]}" > "$log" 2>&1
    rc=$?
    printf '%s\n' "$rc" > "$EXTRA_ROOT/raw/$dataset.exit_code"
    exit "$rc"
  ) &
  pid=$!
  pids+=("$pid")
  printf '%s\n' "$pid" > "$EXTRA_ROOT/raw/$dataset.pid"
done

overall=0
for pid in "${pids[@]}"; do
  wait "$pid" || overall=1
done
for dataset in "${datasets[@]}"; do
  code_file="$EXTRA_ROOT/raw/$dataset.exit_code"
  if [[ ! -f "$code_file" || "$(<"$code_file")" != 0 ]]; then
    overall=1
  fi
done

if [[ "$overall" -ne 0 ]]; then
  printf '%s\n' failed > "$EXTRA_ROOT/status"
  printf '%s\n' "$overall" > "$EXTRA_ROOT/controller.exit"
  exit "$overall"
fi

"$PYTHON_BIN" "$SUMMARY_SCRIPT" --eval-root "$EXTRA_ROOT" > "$EXTRA_ROOT/logs/summarize.log" 2>&1
printf '%s\n' "$?" > "$EXTRA_ROOT/controller.exit"
printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$EXTRA_ROOT/records/protocol.txt"
printf '%s\n' complete > "$EXTRA_ROOT/status"
