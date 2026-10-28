#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_NAME="${MODEL_NAME:-Qwen3-4B-GRPO-DeepMath}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${REPO_ROOT}/models/models/Qwen/Qwen3-4B}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/Qwen3-4B-GRPO-DeepMath}"
STEPS="${STEPS:-0 50 100 111}"
DATASETS="${DATASETS:-aime24 aime25 hmmt25_feb hmmt25_nov}"

MAX_TOKENS="${MAX_TOKENS:-16384}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
N="${N:-32}"
SEED="${SEED:-42}"
FORCE="${FORCE:-0}"
MERGE_MISSING="${MERGE_MISSING:-0}"

model_path_for_step() {
    local step="$1"
    if [[ "${step}" == "0" ]]; then
        printf '%s\n' "${BASE_MODEL_PATH}"
        return
    fi

    local step_dir="${CHECKPOINT_ROOT}/global_step_${step}"
    local merged_dir="${step_dir}/hf_merged"
    local actor_dir="${step_dir}/actor"

    if [[ -f "${merged_dir}/model.safetensors.index.json" || -f "${merged_dir}/model.safetensors" ]]; then
        printf '%s\n' "${merged_dir}"
        return
    fi

    if [[ -d "${actor_dir}" && "${MERGE_MISSING}" == "1" ]]; then
        python "${REPO_ROOT}/verl/scripts/legacy_model_merger.py" merge \
            --backend fsdp \
            --local_dir "${actor_dir}" \
            --target_dir "${merged_dir}" >&2
        printf '%s\n' "${merged_dir}"
        return
    fi

    cat >&2 <<EOF
No merged HF model found for step ${step}: ${merged_dir}
Merge it first, or rerun with MERGE_MISSING=1:
  python ${REPO_ROOT}/verl/scripts/legacy_model_merger.py merge --backend fsdp --local_dir ${actor_dir} --target_dir ${merged_dir}
EOF
    return 1
}

dataset_config() {
    case "$1" in
        aime24)
            printf '%s\t%s\t%s\n' "../data/aime24/test.jsonl" "eval_outputs/aime24" "${AIME24_GPUS:-0,1}"
            ;;
        aime25)
            printf '%s\t%s\t%s\n' "../data/aime25/test.jsonl" "eval_outputs/aime25" "${AIME25_GPUS:-2,3}"
            ;;
        hmmt25_feb)
            printf '%s\t%s\t%s\n' "../data/hmmt25_feb/test.jsonl" "eval_outputs/hmmt25_feb" "${HMMT25_FEB_GPUS:-4,5}"
            ;;
        hmmt25_nov)
            printf '%s\t%s\t%s\n' "../data/hmmt25_nov/test.jsonl" "eval_outputs/hmmt25_nov" "${HMMT25_NOV_GPUS:-6,7}"
            ;;
        *)
            echo "Unknown dataset: $1" >&2
            return 1
            ;;
    esac
}

run_one_dataset() {
    local dataset="$1"
    local step="$2"
    local model_path="$3"
    local input_file output_dir gpus

    IFS=$'\t' read -r input_file output_dir gpus < <(dataset_config "${dataset}")
    mkdir -p "${output_dir}"

    local output_file="${output_dir}/${MODEL_NAME}-step${step}.jsonl"
    if [[ -s "${output_file}" && "${FORCE}" != "1" ]]; then
        echo "Skip existing ${output_file}"
        return 0
    fi

    echo "Eval ${dataset} step ${step}: ${model_path}"
    CUDA_VISIBLE_DEVICES="${gpus}" python eval_math.py \
        --input_file "${input_file}" \
        --model_path "${model_path}" \
        --output_file "${output_file}" \
        --max_tokens "${MAX_TOKENS}" \
        --temperature "${TEMPERATURE}" \
        --top_p "${TOP_P}" \
        --max_num_seqs "${MAX_NUM_SEQS}" \
        --n "${N}" \
        --begin_idx -1 \
        --end_idx -1 \
        --seed "${SEED}"
}

cd "${SCRIPT_DIR}"

for step in ${STEPS}; do
    model_path="$(model_path_for_step "${step}")"
    pids=()
    for dataset in ${DATASETS}; do
        run_one_dataset "${dataset}" "${step}" "${model_path}" &
        pids+=("$!")
    done

    for pid in "${pids[@]}"; do
        wait "${pid}"
    done

    python plot_step_eval.py \
        --model-prefix "${MODEL_NAME}" \
        --output "${REPO_ROOT}/outputs/grpo_step_eval_curves.svg" \
        --summary-output "${REPO_ROOT}/outputs/grpo_step_eval_summary.csv"
done
