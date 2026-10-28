#!/usr/bin/env bash
set -Eeuo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REF_REPO="${REF_REPO:-/path/to/reference}"
MODEL_PATH="${MODEL_PATH:-$REF_REPO/models/models/Qwen/Qwen3-4B}"
EVALPLUS_REPO="${EVALPLUS_REPO:-$REF_REPO/code_eval/coding/evalplus}"
LCB_REPO="${LCB_REPO:-$REF_REPO/code_eval/coding/LiveCodeBench}"
SCRIPT_ROOT="${SCRIPT_ROOT:-/path/to/opgd/math_eval}"
EVALPLUS_WRAPPER="${EVALPLUS_WRAPPER:-$SCRIPT_ROOT/evalplus_codegen_mode.py}"
LCB_WRAPPER="${LCB_WRAPPER:-$SCRIPT_ROOT/lcb_runner_mode.py}"
CODE_ROOT="${CODE_ROOT:-$RUN_ROOT/code_eval}"
RUNTIME_PACKAGES="${RUNTIME_PACKAGES:-/path/to/runs/opgd0_code_eurus_a8008_vsgdlr0p17_s100_seed42_20260825T210022Z/formal_code_eval/stepwise_code3eval_n4_temp1_seed42_20260826T1950Z/records/runtime/python_packages}"
CODE_N=4
ENABLE_THINKING="${ENABLE_THINKING:-0}"

if [[ -e "$CODE_ROOT/status" || -e "$CODE_ROOT/code_eval_manifest.json" ]]; then
  echo "Refusing to overwrite existing CodeEval run: $CODE_ROOT" >&2
  exit 84
fi
mkdir -p "$CODE_ROOT"/{evalplus_results,logs,records,lcb_work}
for required in "$PYTHON_BIN" "$MODEL_PATH" "$EVALPLUS_REPO" "$LCB_REPO" "$EVALPLUS_WRAPPER" "$LCB_WRAPPER"; do
  if [[ ! -e "$required" ]]; then echo "Missing required CodeEval path: $required" >&2; printf '%s\n' failed > "$CODE_ROOT/status"; exit 81; fi
done
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -Eq '[0-9]'; then
  echo "Refusing to start CodeEval: one or more GPUs already have compute processes." >&2
  printf '%s\n' refused_busy > "$CODE_ROOT/status"; exit 83
fi
for required in "$REF_REPO/code_eval/data/HumanEvalPlus.jsonl" "$REF_REPO/code_eval/data/MbppPlus.jsonl"; do
  if [[ ! -s "$required" ]]; then echo "Missing code benchmark: $required" >&2; printf '%s\n' failed > "$CODE_ROOT/status"; exit 82; fi
done

# LiveCodeBench resolves its few-shot JSON files relative to the current
# directory.  Keep results isolated while exposing the official package tree.
if [[ ! -e "$CODE_ROOT/lcb_work/lcb_runner" ]]; then
  ln -s "$LCB_REPO/lcb_runner" "$CODE_ROOT/lcb_work/lcb_runner"
fi

THINKING=false; [[ "$ENABLE_THINKING" == "1" ]] && THINKING=true
cat > "$CODE_ROOT/code_eval_manifest.json" <<EOF
{
  "format": "qwen3_4b_code_eval_v1",
  "model": "$(basename "$MODEL_PATH")",
  "model_path": "$MODEL_PATH",
  "thinking": $THINKING,
  "components": ["EvalPlus HumanEval+", "EvalPlus MBPP+", "LiveCodeBench v6"],
  "n_samples": 4,
  "temperature": 1.0,
  "top_p": 1.0,
  "max_tokens": 16384,
  "seed": 42,
  "gpus": [0, 1, 2]
}
EOF
printf '%s\n' running > "$CODE_ROOT/status"
{
  printf 'model=%s\n' "$MODEL_PATH"
  printf 'components=EvalPlus HumanEval+, EvalPlus MBPP+, LiveCodeBench v6\n'
  printf 'n_samples=4\ntemperature=1.0\ntop_p=1.0\nmax_tokens=16384\ntensor_parallel_size=1\nseed=42\ngpus=0,1,2\nthinking=%s\n' "$THINKING"
  printf 'started_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$CODE_ROOT/records/protocol.txt"
cp -p "$EVALPLUS_WRAPPER" "$LCB_WRAPPER" "$CODE_ROOT/records/"

datasets=(humaneval mbpp); gpus=(0 1); pids=()
for index in "${!datasets[@]}"; do
  dataset="${datasets[$index]}"; gpu="${gpus[$index]}"; log="$CODE_ROOT/logs/${dataset}_codegen_eval.log"
  (
    set +e
    ENABLE_THINKING="$ENABLE_THINKING" CUDA_VISIBLE_DEVICES="$gpu" VLLM_WORKER_MULTIPROC_METHOD=spawn \
      PYTHONUNBUFFERED=1 PYTHONPATH="$RUNTIME_PACKAGES:$EVALPLUS_REPO${PYTHONPATH:+:$PYTHONPATH}" \
      "$PYTHON_BIN" "$EVALPLUS_WRAPPER" --model "$MODEL_PATH" --dataset "$dataset" \
        --root "$CODE_ROOT/evalplus_results" --backend vllm --temperature 1.0 \
        --trust_remote_code --n-samples "$CODE_N" > "$log" 2>&1
    rc=$?
    if [[ "$rc" -eq 0 ]]; then
      # EvalPlus codegen writes three JSONL variants.  Only the sanitized
      # samples file contains executable completions; the raw and
      # all_solutions files contain model explanations and must not be passed
      # to evalplus.evaluate.
      samples="$(find "$CODE_ROOT/evalplus_results/$dataset" -maxdepth 1 -type f -name '*.jsonl' ! -name '*.raw.jsonl' ! -name '*.all_solutions.jsonl' -print -quit 2>/dev/null || true)"
      if [[ -z "$samples" || ! -s "$samples" ]]; then printf 'Generated samples missing for %s\n' "$dataset" >> "$log"; rc=84
      else
        CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1 PYTHONPATH="$RUNTIME_PACKAGES:$EVALPLUS_REPO${PYTHONPATH:+:$PYTHONPATH}" \
          "$PYTHON_BIN" -m evalplus.evaluate --dataset "$dataset" --samples "$samples" \
            --output_file "$CODE_ROOT/evalplus_results/$dataset/${dataset}_eval_results.json" \
            --min-time-limit 10.0 --gt-time-limit-factor 8.0 >> "$log" 2>&1
        rc=$?
      fi
    fi
    printf '%s\n' "$rc" > "$CODE_ROOT/records/${dataset}.exit_code"; exit "$rc"
  ) &
  pid=$!; pids+=("$pid"); printf '%s\n' "$pid" > "$CODE_ROOT/records/${dataset}.pid"
done

(
  set +e
  cd "$CODE_ROOT/lcb_work"
  ENABLE_THINKING="$ENABLE_THINKING" LCB_MODEL_PATH="$MODEL_PATH" CUDA_VISIBLE_DEVICES=2 VLLM_WORKER_MULTIPROC_METHOD=spawn \
    PYTHONUNBUFFERED=1 PYTHONPATH="$RUNTIME_PACKAGES:$LCB_REPO${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" "$LCB_WRAPPER" --model Qwen3-4B-NonThinking --local_model_path "$MODEL_PATH" \
      --trust_remote_code --scenario codegeneration --release_version v6 --tensor_parallel_size 1 \
      --use_cache --n "$CODE_N" --temperature 1.0 --top_p 1.0 --max_tokens 16384 \
      --timeout 60 --num_process_evaluate 12 --evaluate > "$CODE_ROOT/logs/livecodebench_v6.log" 2>&1
  rc=$?; printf '%s\n' "$rc" > "$CODE_ROOT/records/livecodebench_v6.exit_code"; exit "$rc"
) &
lcb_pid=$!; printf '%s\n' "$lcb_pid" > "$CODE_ROOT/records/livecodebench_v6.pid"

overall=0
for pid in "${pids[@]}"; do wait "$pid" || overall=1; done
for dataset in "${datasets[@]}"; do if [[ ! -f "$CODE_ROOT/records/${dataset}.exit_code" || "$(<"$CODE_ROOT/records/${dataset}.exit_code")" != 0 ]]; then overall=1; fi; done
wait "$lcb_pid" || overall=1

LCB_SOURCE="$CODE_ROOT/lcb_work/lcb_outputs/$(basename "$MODEL_PATH")"
if [[ -d "$LCB_SOURCE" ]]; then cp -a "$LCB_SOURCE" "$CODE_ROOT/lcb_outputs"; elif [[ "$(<"$CODE_ROOT/records/livecodebench_v6.exit_code")" == 0 ]]; then echo "Missing LCB output: $LCB_SOURCE" >&2; overall=1; fi
if [[ "$overall" -ne 0 ]]; then printf '%s\n' failed > "$CODE_ROOT/status"; printf '%s\n' "$overall" > "$CODE_ROOT/controller.exit"; exit "$overall"; fi
for required in "$CODE_ROOT/evalplus_results/humaneval" "$CODE_ROOT/evalplus_results/mbpp" "$CODE_ROOT/lcb_outputs"; do if [[ ! -e "$required" ]]; then echo "Incomplete CodeEval output: $required" >&2; printf '%s\n' failed > "$CODE_ROOT/status"; exit 86; fi; done
printf '%s\n' 0 > "$CODE_ROOT/controller.exit"; printf '%s\n' complete > "$CODE_ROOT/status"
