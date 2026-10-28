#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/run_advantage_top10_math_qwen3_4b.sh"
OPGD_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
VERL_ROOT="$OPGD_ROOT/verl"

if [[ -f "$OPGD_ROOT/.env" ]]; then
  set -a
  source "$OPGD_ROOT/.env"
  set +a
fi

MODEL_PATH="${MODEL_PATH:-/path/to/Qwen3-4B}"
TRAIN_FILE="${TRAIN_FILE:-/path/to/training-data/DeepMath-103K/train_filtered_level6.parquet}"
VAL_FILE_2024="${VAL_FILE_2024:-/path/to/training-data/AIME2024/test.parquet}"
VAL_FILE_2025="${VAL_FILE_2025:-/path/to/training-data/AIME2025/test.parquet}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TAG="${TAG_OVERRIDE:-opgd_advantage_top10_math_qwen3_4b_seed42_${STAMP}}"
OUT="${OUT_OVERRIDE:-$VERL_ROOT/checkpoints/opgd/$TAG}"
WANDB_PROJECT="${WANDB_PROJECT:-OPGD}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
PRINT_CONFIG="${PRINT_CONFIG:-0}"

INSPECT_ARGS=()
if [[ "$PRINT_CONFIG" == "1" ]]; then
  INSPECT_ARGS=(--cfg job --resolve)
fi

if [[ "$PRINT_CONFIG" != "1" ]]; then
  RECORD_DIR="$OUT/records"
  mkdir -p "$RECORD_DIR"
  cp "$SCRIPT_PATH" "$RECORD_DIR/run_advantage_top10_math.sh"
  {
    printf 'utc_started=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'script=%s\n' "$SCRIPT_PATH"
    printf 'command='
    printf '%q ' "$SCRIPT_PATH" "$@"
    printf '\n'
    printf 'tag=%s\n' "$TAG"
    printf 'output_dir=%s\n' "$OUT"
    printf 'python=python\n'
    printf 'model_path=%s\n' "$MODEL_PATH"
    printf 'train_file=%s\n' "$TRAIN_FILE"
    printf 'val_file_2024=%s\n' "$VAL_FILE_2024"
    printf 'val_file_2025=%s\n' "$VAL_FILE_2025"
    printf 'ngpus_per_node=8\n'
    printf 'cuda_visible_devices=0,1,2,3,4,5,6,7\n'
    printf 'tensor_parallel_size=4\n'
    printf 'virtual_outer_dp_size=1\n'
    printf 'train_batch_size=128\n'
    printf 'rollout_n=8\n'
    printf 'total_steps=300\n'
    printf 'max_prompt_length=2048\n'
    printf 'max_response_length=16384\n'
    printf 'max_tokens_per_gpu=32768\n'
    printf 'rollout_seed=42\n'
    printf 'expert_count=1\n'
    printf 'virtual_update=sgd\n'
    printf 'virtual_sgd_learning_rate=0.1\n'
    printf 'real_optimizer=AdamW\n'
    printf 'real_optimizer_learning_rate=1e-6\n'
    printf 'real_optimizer_weight_decay=0.01\n'
    printf 'adaptive_virtual_scale=false\n'
    printf 'max_rescale_passes=0\n'
    printf 'prompt_centering=false\n'
    printf 'length_denormalization_opd=false\n'
    printf 'length_denormalization_rl=false\n'
    printf 'advantage_top_fraction=0.1\n'
    printf 'wandb_project=%s\n' "$WANDB_PROJECT"
    printf 'wandb_entity=%s\n' "$WANDB_ENTITY"
    printf 'wandb_mode=%s\n' "$WANDB_MODE"
    printf 'l0_zero_tolerance=1e-6\n'
  } > "$RECORD_DIR/launch.txt"
fi

cd "$VERL_ROOT"
export PYTHONPATH="$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE WANDB_PROJECT WANDB_ENTITY

exec env \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  VERL_MATH_VERIFY_HARD_TIMEOUT_SECONDS=30 \
  python -m recipe.opgd.main_opgd \
  "${INSPECT_ARGS[@]}" \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=true \
  algorithm.use_kl_in_reward=false \
  algorithm.filter_groups.enable=false \
  algorithm.rollout_correction.rollout_is=token \
  algorithm.rollout_correction.rollout_is_threshold=5.0 \
  algorithm.rollout_correction.rollout_rs=null \
  algorithm.opgd.enable=true \
  algorithm.opgd.expert_count=1 \
  algorithm.opgd.virtual_update=sgd \
  algorithm.opgd.virtual_step_l2=0.01 \
  algorithm.opgd.virtual_sgd_learning_rate=0.1 \
  algorithm.opgd.adaptive_virtual_scale=false \
  algorithm.opgd.target_logratio_abs_mean=0.03 \
  algorithm.opgd.target_logratio_low=0.01 \
  algorithm.opgd.target_logratio_high=0.06 \
  algorithm.opgd.max_logratio_abs_mean=0.10 \
  algorithm.opgd.max_rescale_passes=0 \
  algorithm.opgd.min_runtime_scale_factor=0.25 \
  algorithm.opgd.max_runtime_scale_factor=4.0 \
  algorithm.opgd.prompt_centering=false \
  algorithm.opgd.length_denormalization_opd=false \
  algorithm.opgd.length_denormalization_rl=false \
  algorithm.opgd.advantage_top_fraction=0.1 \
  algorithm.opgd.length_denormalization_max_tokens=16384 \
  algorithm.opgd.verify_restore=true \
  algorithm.opgd.l0_zero_tolerance=1e-6 \
  reward_model.reward_manager=naive \
  reward_model.overlong_buffer.enable=false \
  data.train_files="$TRAIN_FILE" \
  "data.val_files=[$VAL_FILE_2024,$VAL_FILE_2025]" \
  data.train_batch_size=128 \
  data.gen_batch_size=128 \
  data.max_prompt_length=2048 \
  data.max_response_length=16384 \
  data.filter_overlong_prompts=true \
  data.truncation=error \
  data.shuffle=true \
  data.seed=42 \
  data.validation_shuffle=false \
  +data.apply_chat_template_kwargs.enable_thinking=false \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.actor.optim.optimizer=AdamW \
  actor_rollout_ref.actor.optim.optimizer_impl=torch.optim \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.optim.weight_decay=0.01 \
  'actor_rollout_ref.actor.optim.betas=[0.9,0.999]' \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
  actor_rollout_ref.actor.use_dynamic_bsz=false \
  actor_rollout_ref.actor.shuffle=false \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=sync \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.virtual_outer_dp_size=1 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.seed=42 \
  actor_rollout_ref.rollout.calculate_log_probs=true \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
  actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.n=32 \
  trainer.val_before_train=false \
  trainer.resume_mode=auto \
  'trainer.logger=[console,wandb]' \
  trainer.project_name="$WANDB_PROJECT" \
  trainer.experiment_name="$TAG" \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.balance_batch=true \
  trainer.save_freq=10 \
  trainer.test_freq=10 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=300 \
  trainer.max_actor_ckpt_to_keep=31 \
  trainer.default_local_dir="$OUT" \
  trainer.validation_data_dir="$OUT/validation_generations" \
  +ray_kwargs.ray_init._temp_dir="/tmp/opgd_${STAMP}" \
  +ray_kwargs.ray_init.include_dashboard=false \
  +ray_kwargs.ray_init.num_cpus=64 \
  "$@"
