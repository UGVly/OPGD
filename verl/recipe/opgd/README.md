# OPGD recipe

## One step

1. Generate one rollout batch and compute GRPO advantages.
2. Backpropagate the virtual-teacher policy loss without stepping an
   optimizer. Optional RL length denormalization is disabled by default.
3. Move a temporary teacher by raw SGD `alpha * g` (the checked-in default)
   or opt-in normalized SGD `eta_v * g / ||g||`.
4. Measure `delta = log pi_teacher - log pi_student` on sampled response
   tokens. Optionally rescale the virtual move once to target
   `mean(abs(delta)) = 0.03`.
5. Optionally center delta per prompt/expert fold, rescale response-level
   advantages, and select token subsets by `|delta|`, entropy, or
   `|advantage|`.
6. Restore the student exactly, verify optimizer/scheduler state did not
   change, and perform exactly one real optimizer and scheduler update.

Only two extra batch tensors survive signal construction:

- `advantages`: the final real-update token signal;
- `opgd_update_mask`: tokens admitted to the real update.

`opgd_teacher_log_probs` and scalar safety tensors exist only across the
teacher RPC boundary. Duplicate GRPO, delta, centered-delta, length, scale, and
final-advantage copies were removed.

## Profiles

All launchers use `OPGDTrainer` and `config/opgd_trainer.yaml`. The checked-in
configuration is the no-tricks baseline: raw SGD, prompt centering off, OPD/RL
length denormalization off, adaptive scale off, and all token/reward masks off.
Dedicated launchers explicitly opt into one experiment switch. The
`run_advantage_top10_math_qwen3_4b.sh` launcher enables only
`advantage_top_fraction=0.1`, selecting the 10% largest `|final_advantage|`
tokens independently in each response. Entropy selection requests the
actor-side entropy tensor; batch export requests it automatically even when
the entropy coefficient is zero.

Multiple experts remain available through `expert_count`; `rollout.n` and the
full train batch must divide the expert/world-size layout. The trainer rejects
invalid synchronous-rollout, vLLM, log-prob, and small-batch configurations
before Ray workers start.

## Layout

- `trainer.py`: fold assignment, final OPD signal, trainer hooks.
- `controller.py`: one-pass rescale and hard teacher-validity gate.
- `main_opgd.py`: the only Hydra/Ray entry point.
- `config/opgd_trainer.yaml`: the only OPGD config.
- `scripts/run_no_tricks_math.sh`: standalone math all-tricks-off launcher.
- `scripts/run_no_tricks_code.sh`: standalone Eurus CODE all-tricks-off launcher.
- `scripts/run_no_tricks_code_17b.sh`: Qwen3-1.7B Eurus CODE launcher on four GPUs.
- `scripts/run_length_denorm_code_17b.sh`: Qwen3-1.7B CODE launcher with both
  OPD and RL length denormalization enabled on GPUs 4-7.
- `scripts/run_advantage_top10_math_qwen3_4b.sh`: 8x A800
  top-10% `|advantage|` experiment.
- `scripts/run_entropy_top10_math_qwen3_4b.sh`: 8x A800 entropy
  top-10% experiment with explicit entropy generation.
- `scripts/run_export_batch_math_qwen3_4b.sh`: 8x A800 diagnostic
  batch export with all training tricks disabled.
- `tests/test_opgd.py`: CPU contracts for both profiles and safety logic.

## Invariants

- GRPO, full-parameter FSDP1, LoRA rank zero, BF16/FP32.
- One full actor mini-batch, one PPO epoch, dynamic actor batching off.
- Entropy and actor/reward KL off.
- Synchronous rollout log probabilities required.
- The virtual path never calls optimizer or scheduler `step()`.
- A non-finite/zero gradient, excessive/non-finite teacher distance, inexact
  restore, or mutated optimizer/scheduler state prevents the real update.

Run from `verl`:

```bash
recipe/opgd/scripts/run_no_tricks_math.sh
PRINT_CONFIG=1 recipe/opgd/scripts/run_no_tricks_math.sh
PRINT_CONFIG=1 recipe/opgd/scripts/run_no_tricks_code.sh
```
