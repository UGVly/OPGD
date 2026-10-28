#!/usr/bin/env bash
set -euo pipefail

: "${TAG:?TAG is required}"
: "${RUN_DIR:?RUN_DIR is required}"
: "${CHECKPOINT_DIR:?CHECKPOINT_DIR is required}"
: "${RAY_TMP:?RAY_TMP is required}"
: "${WANDB_RUN_ID:?WANDB_RUN_ID is required}"

ROOT=/path/to/opgd
REPO="$ROOT/verl"
PYTHON=python3
MODEL=/path/to/Qwen3-1.7B
TRAIN=/path/to/training-data/DeepMath-103K/train_filtered_level6.parquet
AIME24=/path/to/training-data/AIME2024/test.parquet
AIME25=/path/to/training-data/AIME2025/test.parquet

TRAIN_BATCH_SIZE=128
ROLLOUT_N=8
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
LEARNING_RATE=1e-6
TOTAL_STEPS="${TOTAL_STEPS:-200}"
SAVE_FREQ="${SAVE_FREQ:-10}"
MAX_CKPTS="${MAX_CKPTS:-20}"
WORLD_SIZE=4
ROLLOUT_TP=1
SEED=42
TOKEN_BUDGET="${TOKEN_BUDGET:-32768}"

mkdir -p "$RUN_DIR" "$CHECKPOINT_DIR" "$RAY_TMP"
printf '%s\n' "$$" > "$RUN_DIR/controller.pid"

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export RAY_TMPDIR="$RAY_TMP"
export OPGD_CLUSTER_TAG="$TAG"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-OPGD}"
export WANDB_RUN_ID
export WANDB_RESUME="${WANDB_RESUME:-never}"
export USED_MODEL=no_api
unset RAY_ADDRESS || true

for required in "$REPO" "$MODEL" "$TRAIN" "$AIME24" "$AIME25"; do
  if [[ ! -e "$required" ]]; then
    printf 'Missing required path: %s\n' "$required" >&2
    printf '91\n' > "$RUN_DIR/preflight.status"
    exit 91
  fi
done

CMD=(
  "$PYTHON" -m verl.trainer.main_ppo
  algorithm.adv_estimator=grpo
  algorithm.rollout_correction.rollout_is=token
  algorithm.rollout_correction.rollout_is_threshold=5.0
  algorithm.rollout_correction.rollout_rs=null
  algorithm.rollout_correction.bypass_mode=false
  algorithm.use_kl_in_reward=false
  actor_rollout_ref.rollout.calculate_log_probs=true
  data.train_files="$TRAIN"
  "data.val_files=[$AIME24,$AIME25]"
  data.train_batch_size="$TRAIN_BATCH_SIZE"
  data.dataloader_num_workers=0
  data.max_prompt_length="$MAX_PROMPT_LENGTH"
  data.max_response_length="$MAX_RESPONSE_LENGTH"
  data.filter_overlong_prompts=true
  data.truncation=error
  data.shuffle=true
  data.seed="$SEED"
  +data.apply_chat_template_kwargs.enable_thinking=false
  actor_rollout_ref.model.path="$MODEL"
  actor_rollout_ref.model.lora_rank=0
  actor_rollout_ref.model.use_remove_padding=true
  actor_rollout_ref.model.enable_gradient_checkpointing=true
  actor_rollout_ref.nccl_timeout=14400
  actor_rollout_ref.actor.strategy=fsdp
  actor_rollout_ref.actor.optim.lr="$LEARNING_RATE"
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0
  actor_rollout_ref.actor.ppo_epochs=1
  actor_rollout_ref.actor.ppo_mini_batch_size="$TRAIN_BATCH_SIZE"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.use_dynamic_bsz=false
  actor_rollout_ref.actor.use_kl_loss=true
  actor_rollout_ref.actor.kl_loss_coef=0.0
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$TOKEN_BUDGET"
  actor_rollout_ref.actor.fsdp_config.param_offload=false
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=sync
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$TOKEN_BUDGET"
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP"
  actor_rollout_ref.rollout.seed="$SEED"
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6
  actor_rollout_ref.rollout.n="$ROLLOUT_N"
  actor_rollout_ref.rollout.max_num_batched_tokens="$TOKEN_BUDGET"
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$TOKEN_BUDGET"
  actor_rollout_ref.ref.fsdp_config.param_offload=true
  reward_model.reward_manager=naive
  trainer.critic_warmup=0
  trainer.val_before_train=false
  'trainer.logger=[console,wandb]'
  trainer.log_val_generations=0
  trainer.project_name=OPGD
  trainer.experiment_name="$TAG"
  trainer.n_gpus_per_node="$WORLD_SIZE"
  trainer.nnodes=1
  trainer.balance_batch=true
  trainer.save_freq="$SAVE_FREQ"
  trainer.test_freq=-1
  trainer.total_epochs=3
  trainer.total_training_steps="$TOTAL_STEPS"
  trainer.max_actor_ckpt_to_keep="$MAX_CKPTS"
  trainer.resume_mode="${RESUME_MODE:-disable}"
  trainer.default_local_dir="$CHECKPOINT_DIR"
  +ray_kwargs.ray_init._temp_dir="$RAY_TMP"
  +ray_kwargs.ray_init.include_dashboard=false
  +ray_kwargs.ray_init.num_cpus=64
)

printf '%q ' "${CMD[@]}" > "$RUN_DIR/launch_command.txt"
printf '\n' >> "$RUN_DIR/launch_command.txt"

cd "$REPO"

"$PYTHON" - "$TRAIN" "$MODEL" "$RUN_DIR/asset_audit.json" <<'PY'
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

"$PYTHON" - "$RUN_DIR/reward_smoke.json" <<'PY'
import json
import sys
from pathlib import Path
from verl.utils.reward_score.math_verify import compute_score

correct = compute_score("The answer is \\boxed{2}.", "2")
incorrect = compute_score("The answer is \\boxed{3}.", "2")
assert float(correct) == 1.0, correct
assert float(incorrect) == 0.0, incorrect
Path(sys.argv[1]).write_text(json.dumps({"correct": float(correct), "incorrect": float(incorrect)}, indent=2) + "\n")
PY

if ! "${CMD[@]}" --cfg job > "$RUN_DIR/resolved_config.yaml" 2> "$RUN_DIR/preflight.stderr"; then
  printf '90\n' > "$RUN_DIR/preflight.status"
  exit 90
fi

"$PYTHON" - "$RUN_DIR/resolved_config.yaml" "$TRAIN" "$MODEL" "$AIME24" "$AIME25" "$TOTAL_STEPS" "$SAVE_FREQ" "$MAX_CKPTS" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1]))
assert cfg["algorithm"]["adv_estimator"] == "grpo"
assert cfg["algorithm"]["use_kl_in_reward"] is False
assert cfg["data"]["train_files"] == sys.argv[2]
assert cfg["data"]["val_files"] == [sys.argv[4], sys.argv[5]]
assert cfg["data"]["train_batch_size"] == 128
assert cfg["data"]["max_prompt_length"] == 2048
import os
assert cfg["data"]["max_response_length"] == int(os.environ.get("MAX_RESPONSE_LENGTH", "2048"))
assert cfg["data"]["seed"] == 42
assert cfg["data"]["apply_chat_template_kwargs"]["enable_thinking"] is False
assert cfg["actor_rollout_ref"]["model"]["path"] == sys.argv[3]
assert cfg["actor_rollout_ref"]["model"]["lora_rank"] == 0
actor = cfg["actor_rollout_ref"]["actor"]
assert actor["strategy"] == "fsdp"
assert actor["optim"]["lr"] == 1e-6
assert actor["ppo_mini_batch_size"] == 128
assert actor["ppo_micro_batch_size_per_gpu"] == 1
assert actor["use_kl_loss"] is True
assert actor["kl_loss_coef"] == 0.0
rollout = cfg["actor_rollout_ref"]["rollout"]
assert rollout["n"] == 8
assert rollout["tensor_model_parallel_size"] == 1
assert rollout["temperature"] == 1.0
assert rollout["top_p"] == 1.0
assert cfg["trainer"]["n_gpus_per_node"] == 4
assert cfg["trainer"]["total_training_steps"] == int(sys.argv[6])
assert cfg["trainer"]["save_freq"] == int(sys.argv[7])
assert cfg["trainer"]["test_freq"] == -1
assert cfg["trainer"]["max_actor_ckpt_to_keep"] == int(sys.argv[8])
print("AUDIT_OK")
PY
printf '0\n' > "$RUN_DIR/preflight.status"

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  exit 0
fi

if [[ "${ALLOW_NONEMPTY_CHECKPOINT:-0}" != "1" ]] && find "$CHECKPOINT_DIR" -mindepth 1 -print -quit | grep -q .; then
  printf 'Fresh-run checkpoint root is not empty: %s\n' "$CHECKPOINT_DIR" >&2
  printf '92\n' > "$RUN_DIR/controller.status"
  exit 92
fi

nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader > "$RUN_DIR/gpu_state_before.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader > "$RUN_DIR/gpu_processes_before.csv"
if [[ -s "$RUN_DIR/gpu_processes_before.csv" ]]; then
  printf 'At least one GPU has an active compute process; refusing to launch.\n' >&2
  printf '93\n' > "$RUN_DIR/controller.status"
  exit 93
fi

start_epoch=$(date +%s)
set +e
"${CMD[@]}" 2>&1 | tee -a "$RUN_DIR/train.log"
status=${PIPESTATUS[0]}
set -e
end_epoch=$(date +%s)
"$PYTHON" - "$RUN_DIR/timing.json" "$start_epoch" "$end_epoch" "$status" <<'PY'
import json
import sys
from pathlib import Path
start, end, status = map(int, sys.argv[2:])
Path(sys.argv[1]).write_text(json.dumps({"start_epoch_seconds": start, "end_epoch_seconds": end, "elapsed_seconds": end-start, "exit_status": status}, indent=2) + "\n")
PY
printf '%s\n' "$status" > "$RUN_DIR/controller.status"
exit "$status"
