# OPGD configuration

`opgd_trainer.yaml` composes veRL's ordinary `ppo_trainer` and enables the
OPGD trainer. The checked-in configuration is the reproducible no-tricks
profile: raw virtual SGD, no prompt centering, no OPD/RL length
denormalization, no adaptive scale, and no token or reward-group masks.
Dedicated experiment launchers opt into one feature at a time.

The core controls under `algorithm.opgd` are:

- `virtual_update`: `sgd` (the default raw update) or `normalized_sgd`.
- `virtual_sgd_learning_rate` and `virtual_step_l2`: learning-rate controls
  for raw and normalized virtual updates respectively.
- `adaptive_virtual_scale` and `max_rescale_passes`: optional bounded teacher
  distance rescaling. The default is disabled.
- `prompt_centering`: optional per-prompt centering of the OPD signal.
- `length_denormalization_opd` and `length_denormalization_rl`: optional
  response-length scaling for OPD and virtual-teacher GRPO advantages.
- `expert_count`: number of cross-fitted virtual experts; `1` is the default.
  Rollout `n` must be divisible by this count.
- `mask_uniform_reward_groups`: exclude prompt groups whose rewards have zero
  variance. `uniform_reward_tolerance` controls the equality threshold.
- `scale_opgd.enable`: optional response-wise advantage rescaling. The
  `statistic` is `mean_abs`, `rms`, or `l2`; `target` is `batch_mean` or `unit`.

Token-selection controls are all disabled when their fraction is `0`:

- `delta_top_fraction` keeps tokens with the largest absolute teacher delta
  (`|delta|`) in each response.
- `entropy_top_fraction` keeps tokens with the largest entropy in each response.
  The actor policy config must request `entropy_token_top_fraction` so the
  rollout batch contains `entropys`; the entropy launcher sets both switches.
- `advantage_bottom_fraction` keeps tokens with the smallest absolute final
  advantage in each response.
- `advantage_top_fraction` keeps tokens with the largest absolute final
  advantage in each response. A value of `0.1` is the reproducible top-10%
  `|advantage|` experiment requested by the project plan.

For every fraction, the implementation keeps at least one valid token in a
non-empty response and intersects the selection with the response mask,
finite-value checks, and any enabled reward-group mask. If a selection makes
the update mask empty, the trainer skips the real optimizer step safely.

Batch export is opt-in through the Hydra override
`+algorithm.opgd.export_batch_dir=/path/to/export`. Exported tensors are
written as a self-contained diagnostic artifact; export does not enable any
training trick. Entropy is computed automatically when an entropy selection or
export path needs it, while the no-tricks profile avoids the extra dependency.

The default distance targets are `target_logratio_abs_mean=0.03`, accepted
range `[0.015, 0.06]`, and hard ceiling `0.10`. They are inactive unless
`adaptive_virtual_scale=true`.
