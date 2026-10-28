#!/usr/bin/env bash
# Entropy-weighted GRPO, Qwen3-4B math, 8x A800, 300 steps.
#
# Plain GRPO with actor_rollout_ref.actor.policy_loss.entropy_token_weight_mode=
# neg_entropy_minmax: the per-token advantage is multiplied by min-max normalized
# -entropy, so the least uncertain token keeps weight 1 and the most uncertain gets 0.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/run_entweight_math_qwen3_4b_a8008_s300.sh"
OPGD_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
VERL_ROOT="$OPGD_ROOT/verl"
PYTHON=python

if [[ -f "$OPGD_ROOT/.env" ]]; then
  set -a
  source "$OPGD_ROOT/.env"
  set +a
fi

MODEL_PATH="${MODEL_PATH:-/path/to/Qwen3-4B}"
TRAIN_FILE="${TRAIN_FILE:-/path/to/training-data/DeepMath-103K/train_filtered_level6.parquet}"
VAL_FILE_2024="${VAL_FILE_2024:-/path/to/training-data/AIME2024/test.parquet}"
VAL_FILE_2025="${VAL_FILE_2025:-/path/to/training-data/AIME2025/test.parquet}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
ROLLOUT_N="${ROLLOUT_N:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
TOKEN_BUDGET="${TOKEN_BUDGET:-32768}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
TOTAL_STEPS="${TOTAL_STEPS:-300}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-10}"
MAX_CKPTS="${MAX_CKPTS:-31}"
WORLD_SIZE=8
ROLLOUT_TP=4
SEED=42

ENT_WEIGHT_MODE="${ENT_WEIGHT_MODE:-pos_entropy_minmax}"
ENT_WEIGHT_SCOPE="${ENT_WEIGHT_SCOPE:-response}"
ENT_WEIGHT_RENORM="${ENT_WEIGHT_RENORM:-true}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TAG="${TAG_OVERRIDE:-entweight_${ENT_WEIGHT_MODE}_${ENT_WEIGHT_SCOPE}_renorm${ENT_WEIGHT_RENORM}_math_qwen3_4b_a8008_s${TOTAL_STEPS}_seed42_${STAMP}}"
OUT="${OUT_OVERRIDE:-$VERL_ROOT/checkpoints/entweight/$TAG}"
RAY_TMP="${RAY_TMP:-/tmp/entweight_${STAMP}}"
WANDB_PROJECT="${WANDB_PROJECT:-OPGD}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_RUN_ID="${WANDB_RUN_ID:-$TAG}"
WANDB_RESUME="${WANDB_RESUME:-allow}"
PRINT_CONFIG="${PRINT_CONFIG:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"

mkdir -p "$OUT" "$RAY_TMP"

if [[ "$PRINT_CONFIG" == "1" ]]; then
  cd "$VERL_ROOT"
  exec "$PYTHON" -m recipe.entweight.main_entweight --cfg job --resolve "$@"
fi

RECORD_DIR="$OUT/records"
mkdir -p "$RECORD_DIR"
cp "$SCRIPT_PATH" "$RECORD_DIR/run_entweight_math_qwen3_4b_a8008_s300.sh"
{
  printf 'utc_started=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'script=%s\n' "$SCRIPT_PATH"
  printf 'command='
  printf '%q ' "$SCRIPT_PATH" "$@"
  printf '\n'
  printf 'tag=%s\n' "$TAG"
  printf 'output_dir=%s\n' "$OUT"
  printf 'python=%s\n' "$PYTHON"
  printf 'model_path=%s\n' "$MODEL_PATH"
  printf 'train_file=%s\n' "$TRAIN_FILE"
  printf 'val_file_2024=%s\n' "$VAL_FILE_2024"
  printf 'val_file_2025=%s\n' "$VAL_FILE_2025"
  printf 'adv_estimator=grpo\n'
  printf 'critic=disabled\n'
  printf 'reference_model=disabled\n'
  printf 'entropy_token_weight_mode=%s\n' "$ENT_WEIGHT_MODE"
  printf 'entropy_token_weight_scope=%s\n' "$ENT_WEIGHT_SCOPE"
  printf 'entropy_token_weight_renormalize_mean=%s\n' "$ENT_WEIGHT_RENORM"
  printf 'ngpus_per_node=%s\n' "$WORLD_SIZE"
  printf 'cuda_visible_devices=0,1,2,3,4,5,6,7\n'
  printf 'rollout_tensor_parallel_size=%s\n' "$ROLLOUT_TP"
  printf 'train_batch_size=%s\n' "$TRAIN_BATCH_SIZE"
  printf 'rollout_n=%s\n' "$ROLLOUT_N"
  printf 'total_steps=%s\n' "$TOTAL_STEPS"
  printf 'max_prompt_length=%s\n' "$MAX_PROMPT_LENGTH"
  printf 'max_response_length=%s\n' "$MAX_RESPONSE_LENGTH"
  printf 'max_tokens_per_gpu=%s\n' "$TOKEN_BUDGET"
  printf 'optimizer=AdamW\n'
  printf 'learning_rate=%s\n' "$LEARNING_RATE"
  printf 'real_optimizer_weight_decay=0.01\n'
  printf 'save_freq=%s\n' "$SAVE_FREQ"
  printf 'test_freq=%s\n' "$TEST_FREQ"
  printf 'max_actor_ckpt_to_keep=%s\n' "$MAX_CKPTS"
  printf 'rollout_seed=%s\n' "$SEED"
  printf 'wandb_project=%s\n' "$WANDB_PROJECT"
  printf 'wandb_entity=%s\n' "$WANDB_ENTITY"
  printf 'wandb_mode=%s\n' "$WANDB_MODE"
  printf 'wandb_run_id=%s\n' "$WANDB_RUN_ID"
} > "$RECORD_DIR/launch.txt"

for required in "$VERL_ROOT" "$MODEL_PATH" "$TRAIN_FILE" "$VAL_FILE_2024" "$VAL_FILE_2025"; do
  if [[ ! -e "$required" ]]; then
    printf 'Missing required path: %s\n' "$required" >&2
    exit 91
  fi
done

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export PYTHONPATH="$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export RAY_TMPDIR="$RAY_TMP"
export WANDB_MODE WANDB_PROJECT WANDB_ENTITY WANDB_RUN_ID WANDB_RESUME
unset RAY_ADDRESS || true

cd "$VERL_ROOT"

CMD=(
  "$PYTHON" -m recipe.entweight.main_entweight
  algorithm.adv_estimator=grpo
  algorithm.norm_adv_by_std_in_grpo=true
  algorithm.use_kl_in_reward=false
  algorithm.rollout_correction.rollout_is=token
  algorithm.rollout_correction.rollout_is_threshold=5.0
  algorithm.rollout_correction.rollout_rs=null
  algorithm.rollout_correction.bypass_mode=false
  reward_model.reward_manager=naive
  data.train_files="$TRAIN_FILE"
  "data.val_files=[$VAL_FILE_2024,$VAL_FILE_2025]"
  data.train_batch_size="$TRAIN_BATCH_SIZE"
  +data.gen_batch_size="$TRAIN_BATCH_SIZE"
  data.dataloader_num_workers=0
  data.max_prompt_length="$MAX_PROMPT_LENGTH"
  data.max_response_length="$MAX_RESPONSE_LENGTH"
  data.filter_overlong_prompts=true
  data.truncation=error
  data.shuffle=true
  data.seed="$SEED"
  data.validation_shuffle=false
  +data.apply_chat_template_kwargs.enable_thinking=false
  actor_rollout_ref.model.path="$MODEL_PATH"
  actor_rollout_ref.model.lora_rank=0
  actor_rollout_ref.model.use_remove_padding=true
  actor_rollout_ref.model.enable_gradient_checkpointing=true
  actor_rollout_ref.nccl_timeout=14400
  actor_rollout_ref.actor.strategy=fsdp
  actor_rollout_ref.actor.optim.optimizer=AdamW
  actor_rollout_ref.actor.optim.optimizer_impl=torch.optim
  actor_rollout_ref.actor.optim.lr="$LEARNING_RATE"
  actor_rollout_ref.actor.optim.lr_scheduler_type=constant
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0
  actor_rollout_ref.actor.optim.weight_decay=0.01
  'actor_rollout_ref.actor.optim.betas=[0.9,0.999]'
  actor_rollout_ref.actor.ppo_epochs=1
  actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$TOKEN_BUDGET"
  actor_rollout_ref.actor.use_dynamic_bsz=false
  actor_rollout_ref.actor.shuffle=false
  actor_rollout_ref.actor.use_kl_loss=false
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.loss_agg_mode=token-mean
  actor_rollout_ref.actor.policy_loss.loss_mode=vanilla
  actor_rollout_ref.actor.policy_loss.entropy_token_weight_mode="$ENT_WEIGHT_MODE"
  actor_rollout_ref.actor.policy_loss.entropy_token_weight_scope="$ENT_WEIGHT_SCOPE"
  actor_rollout_ref.actor.policy_loss.entropy_token_weight_renormalize_mean="$ENT_WEIGHT_RENORM"
  actor_rollout_ref.actor.fsdp_config.param_offload=false
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=sync
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP"
  actor_rollout_ref.rollout.n="$ROLLOUT_N"
  actor_rollout_ref.rollout.seed="$SEED"
  actor_rollout_ref.rollout.calculate_log_probs=true
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$TOKEN_BUDGET"
  actor_rollout_ref.rollout.max_num_batched_tokens="$TOKEN_BUDGET"
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.top_k=-1
  actor_rollout_ref.rollout.val_kwargs.n=32
  trainer.val_before_train=false
  trainer.resume_mode="${RESUME_MODE:-auto}"
  'trainer.logger=[console,wandb]'
  trainer.project_name="$WANDB_PROJECT"
  trainer.experiment_name="$TAG"
  trainer.n_gpus_per_node="$WORLD_SIZE"
  trainer.nnodes=1
  trainer.balance_batch=true
  trainer.save_freq="$SAVE_FREQ"
  trainer.test_freq="$TEST_FREQ"
  trainer.total_epochs=1
  trainer.total_training_steps="$TOTAL_STEPS"
  trainer.max_actor_ckpt_to_keep="$MAX_CKPTS"
  trainer.default_local_dir="$OUT"
  trainer.validation_data_dir="$OUT/validation_generations"
  +ray_kwargs.ray_init._temp_dir="$RAY_TMP"
  +ray_kwargs.ray_init.include_dashboard=false
  +ray_kwargs.ray_init.num_cpus=64
)

printf '%q ' "${CMD[@]}" > "$RECORD_DIR/launch_command.txt"
printf '\n' >> "$RECORD_DIR/launch_command.txt"

"$PYTHON" - "$TRAIN_FILE" "$MODEL_PATH" "$RECORD_DIR/asset_audit.json" <<'PY'
import collections
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

train_path, model_path, output_path = map(Path, sys.argv[1:])
pf = pq.ParquetFile(train_path)
assert pf.metadata.num_rows == 57046, pf.metadata.num_rows
required = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
assert required.issubset(pf.schema_arrow.names), pf.schema_arrow.names
sources = collections.Counter()
for batch in pf.iter_batches(columns=["data_source"], batch_size=4096):
    sources.update(str(v) for v in batch.column(0).to_pylist())
assert set(sources) == {"DeepMath-103K"}, sources
shards = sorted(model_path.glob("*.safetensors"))
assert shards and all(p.stat().st_size > 0 for p in shards), shards
payload = {
    "train_rows": pf.metadata.num_rows,
    "data_sources": dict(sources),
    "model_safetensor_count": len(shards),
    "model_safetensor_bytes": sum(p.stat().st_size for p in shards),
}
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

"$PYTHON" - "$RECORD_DIR/reward_smoke.json" <<'PY'
import json
import sys
from pathlib import Path

from verl.utils.reward_score.math_verify import compute_score

correct = compute_score("The answer is \\boxed{2}.", "2")
incorrect = compute_score("The answer is \\boxed{3}.", "2")
assert float(correct) == 1.0, correct
assert float(incorrect) == 0.0, incorrect
Path(sys.argv[1]).write_text(
    json.dumps({"correct": float(correct), "incorrect": float(incorrect)}, indent=2) + "\n"
)
PY

if ! "${CMD[@]}" --cfg job --resolve > "$RECORD_DIR/resolved_config.yaml" 2> "$RECORD_DIR/preflight.stderr"; then
  printf '90\n' > "$RECORD_DIR/preflight.status"
  cat "$RECORD_DIR/preflight.stderr" >&2
  exit 90
fi

"$PYTHON" - "$RECORD_DIR/resolved_config.yaml" "$ENT_WEIGHT_MODE" "$ENT_WEIGHT_SCOPE" "$ENT_WEIGHT_RENORM" "$TOTAL_STEPS" "$TRAIN_BATCH_SIZE" "$ROLLOUT_N" "$MAX_PROMPT_LENGTH" "$MAX_RESPONSE_LENGTH" "$SAVE_FREQ" "$TEST_FREQ" <<'PY'
import sys

import yaml

cfg = yaml.safe_load(open(sys.argv[1]))
mode, scope, renorm = sys.argv[2], sys.argv[3], sys.argv[4]
(
    total_steps,
    train_batch_size,
    rollout_n,
    max_prompt_length,
    max_response_length,
    save_freq,
    test_freq,
) = (int(v) for v in sys.argv[5:12])

assert cfg["algorithm"]["adv_estimator"] == "grpo"
assert cfg["algorithm"]["use_kl_in_reward"] is False
assert cfg["algorithm"]["norm_adv_by_std_in_grpo"] is True
assert cfg["data"]["train_batch_size"] == train_batch_size
assert cfg["data"]["max_prompt_length"] == max_prompt_length
assert cfg["data"]["max_response_length"] == max_response_length
assert cfg["data"]["seed"] == 42
assert cfg["data"]["apply_chat_template_kwargs"]["enable_thinking"] is False

actor = cfg["actor_rollout_ref"]["actor"]
assert actor["strategy"] == "fsdp"
assert actor["optim"]["lr"] == 1e-6
assert actor["ppo_mini_batch_size"] == train_batch_size
assert actor["ppo_micro_batch_size_per_gpu"] == 1
assert actor["use_kl_loss"] is False
assert actor["entropy_coeff"] == 0
assert actor["loss_agg_mode"] == "token-mean"

policy_loss = actor["policy_loss"]
assert policy_loss["loss_mode"] == "vanilla"
assert policy_loss["entropy_token_weight_mode"] == mode, policy_loss
assert policy_loss["entropy_token_weight_scope"] == scope, policy_loss
assert bool(policy_loss["entropy_token_weight_renormalize_mean"]) is (renorm == "true"), policy_loss

rollout = cfg["actor_rollout_ref"]["rollout"]
assert rollout["n"] == rollout_n
assert rollout["tensor_model_parallel_size"] == 4
assert rollout["temperature"] == 1.0
assert rollout["val_kwargs"]["n"] == 32

assert cfg["trainer"]["n_gpus_per_node"] == 8
assert cfg["trainer"]["total_training_steps"] == total_steps
assert cfg["trainer"]["save_freq"] == save_freq
assert cfg["trainer"]["test_freq"] == test_freq
print("AUDIT_OK")
PY
printf '0\n' > "$RECORD_DIR/preflight.status"

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  printf 'preflight only, not launching\n'
  exit 0
fi

exec env \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  VERL_MATH_VERIFY_HARD_TIMEOUT_SECONDS=30 \
  "${CMD[@]}" "$@"
