#!/usr/bin/env bash
set -Eeuo pipefail

: "${RUN_ROOT:?RUN_ROOT is required}"
SCRIPT_ROOT="${SCRIPT_ROOT:-/path/to/opgd/math_eval}"
MODEL_PATH="${MODEL_PATH:-/path/to/Qwen3-4B}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -e "$RUN_ROOT/status" || -e "$RUN_ROOT/manifest.json" ]]; then echo "Refusing to overwrite: $RUN_ROOT" >&2; exit 84; fi
mkdir -p "$RUN_ROOT"/records
printf '%s\n' "$$" > "$RUN_ROOT/controller.pid"
cat > "$RUN_ROOT/manifest.json" <<EOF
{
  "format": "qwen3_4b_same_eval_two_modes_v1",
  "model": "$(basename "$MODEL_PATH")",
  "model_path": "$MODEL_PATH",
  "modes": ["no_think", "think"],
  "flow": ["AIME24", "AIME25", "HMMT25-Feb", "HMMT25-Nov", "Math500", "OlympiadBench", "AMC23", "HumanEval+", "MBPP+", "LiveCodeBench v6"],
  "math_protocol": {"n": 32, "temperature": 1.0, "top_p": 1.0, "max_tokens": 16384, "seed": 42},
  "extra_math_protocol": {"Math500_n": 4, "OlympiadBench_n": 4, "AMC23_n": 32, "temperature": 1.0, "top_p": 1.0, "max_tokens": 16384, "seed": 42},
  "code_protocol": {"n": 4, "temperature": 1.0, "top_p": 1.0, "max_tokens": 16384, "seed": 42}
}
EOF
printf '%s\n' running > "$RUN_ROOT/status"
printf 'model=%s\nflow=math(base)+extra_math+code\nmodes=no_think,think\nstarted_utc=%s\n' "$MODEL_PATH" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_ROOT/records/protocol.txt"

run_stage() {
  local mode="$1" thinking="$2" stage_root="$RUN_ROOT/$mode"
  mkdir -p "$stage_root/records"
  cp -p "$SCRIPT_ROOT"/run_qwen3_4b_math_eval_mode_4gpu.sh "$SCRIPT_ROOT"/run_qwen3_4b_code_eval_mode_4gpu.sh "$SCRIPT_ROOT"/run_step500_teacher_extra_math_eval_4gpu.sh "$SCRIPT_ROOT"/summarize_qwen3_4b_math_passk.py "$SCRIPT_ROOT"/summarize_extra_math_passk.py "$stage_root/records/"
  ENABLE_THINKING="$thinking" RUN_ROOT="$stage_root/math_eval" MODEL_PATH="$MODEL_PATH" SUMMARY_SCRIPT="$stage_root/records/summarize_qwen3_4b_math_passk.py" \
    bash "$SCRIPT_ROOT/run_qwen3_4b_math_eval_mode_4gpu.sh"
  ENABLE_THINKING="$thinking" RUN_ROOT="$stage_root" EXTRA_ROOT="$stage_root/extra_math_eval" MODEL_PATH="$MODEL_PATH" SUMMARY_SCRIPT="$stage_root/records/summarize_extra_math_passk.py" \
    bash "$SCRIPT_ROOT/run_step500_teacher_extra_math_eval_4gpu.sh"
  ENABLE_THINKING="$thinking" RUN_ROOT="$stage_root" CODE_ROOT="$stage_root/code_eval" MODEL_PATH="$MODEL_PATH" SCRIPT_ROOT="$SCRIPT_ROOT" \
    bash "$SCRIPT_ROOT/run_qwen3_4b_code_eval_mode_4gpu.sh"
  printf '%s\n' complete > "$stage_root/status"
}

if ! run_stage no_think 0 > "$RUN_ROOT/no_think.log" 2>&1; then printf '%s\n' failed > "$RUN_ROOT/status"; exit 1; fi
if ! run_stage think 1 > "$RUN_ROOT/think.log" 2>&1; then printf '%s\n' failed > "$RUN_ROOT/status"; exit 1; fi
printf 'finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN_ROOT/records/protocol.txt"
printf '%s\n' complete > "$RUN_ROOT/status"
