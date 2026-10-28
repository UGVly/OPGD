# Entropy-weighted GRPO recipe

Plain GRPO in which each response token's advantage is multiplied by a weight built
from the token entropy of the current policy:

$$
w_t = \operatorname{minmax}\!\left(s \cdot H_t\right)
     = \frac{\max(sH) - sH_t}{\max(sH) - \min(sH)},
\qquad A_t \leftarrow A_t \cdot w_t
$$

with `s = +1` (`pos_entropy_minmax`, the recipe default) or `s = -1`
(`neg_entropy_minmax`). The two settings are mirror images: the first gives the most
uncertain token in scope weight `1` and the least uncertain one `0`, the second mutes the
most uncertain token instead. Both min-max normalize into `[0, 1]`, and on valid tokens
the two weight sets sum to one.

The default emphasizes uncertainty because the response-level diagnostics below show that
higher-entropy responses carry somewhat larger `|advantage|`, so `minmax(+H)` scales up
the updates that matter most rather than cancelling them. Note that because the
normalization happens per response, "high entropy" is relative to that response: a
uniformly uncertain response keeps weight 1 everywhere, and only the spread of entropy
inside a response produces any weighting at all.

Weighting happens in the actor worker and the weighted advantage flows through the normal
PPO ratio clipping, so the recipe is a one-term change to GRPO rather than a new
objective.

## Layout

- `main_entweight.py`: the only Hydra entry point. It runs the stock veRL GRPO trainer
  (`verl.trainer.main_ppo.run_ppo`); no custom trainer is needed because the weighting
  is applied in the actor worker.
- `config/entweight_trainer.yaml`: recipe defaults on top of `ppo_trainer`.
- `scripts/run_entweight_math_qwen3_4b_a8008_s300.sh`: Qwen3-4B math launcher for
  8x A800, 300 steps, with a preflight audit of the data, model, reward, and resolved
  config before training starts.
- `scripts/run_entweight_supervisor.sh`: restarts the launcher after a crash and resumes
  from the last checkpoint.
- `tests/test_ent_weight.py`: CPU contracts for the weight function and the config
  defaults.

The weighting math lives in
`verl/trainer/ppo/core_algos.py::compute_entropy_token_weights`, because the actor worker
applies it and shared code must not import from a recipe. The config fields live in
`verl/trainer/config/actor/actor.yaml` and `verl.workers.config.PolicyLossConfig`.

## Knobs

All under `actor_rollout_ref.actor.policy_loss`:

| Field | Default | Meaning |
|---|---|---|
| `entropy_token_weight_mode` | `none` | `none` disables weighting. `pos_entropy_minmax` emphasizes the most uncertain tokens, `neg_entropy_minmax` mutes them. The recipe config sets `pos_entropy_minmax`; the shared actor default stays `none` so nothing changes unless a recipe opts in. |
| `entropy_token_weight_scope` | `response` | `response` normalizes inside each response; `batch` normalizes over the whole micro-batch. |
| `entropy_token_weight_renormalize_mean` | `false` | Rescale weights to mean 1 over valid tokens, restoring the raw gradient scale. The recipe config sets `true`; see the scale caveat below. |

Because entropy is needed on the same tokens the advantage is applied to,
`entropy_token_weight_mode != none` forces the actor to compute entropy, which adds the
full-vocabulary entropy reduction to every micro-batch forward.

## Scale caveat

Raw min-max weights average far from 1, and how far depends on the direction, because
`w_pos = 1 - w_neg` on the same tokens. Token entropy inside a response is right-skewed:
most tokens are confident, so the measured mean of the raw `neg_entropy_minmax` weights is
about 0.89, which makes the raw `pos_entropy_minmax` mean about 0.11. Every response also
has at least one token pinned at weight 0 either way, the least or the most uncertain one.

That asymmetry is why the recipe turns on `renormalize_mean`, which is the setting that
makes the two directions comparable at all: it rescales the weights to mean 1 over valid
tokens, so with lr=1e-6 the average step size matches plain GRPO instead of shrinking to
roughly a ninth of it. Leave it on unless you deliberately want the sparsifying effect of
training almost only on the most uncertain tokens, in which case raise the learning rate
by roughly the reciprocal of the raw mean (about 9x for `pos_entropy_minmax`). Watch
`actor/entropy_weight_mean` to confirm which regime a run is in: it reads 1.0 by
construction when renormalizing.

## Metrics

Logged per step under `actor/`:

- `entropy_weight_mean`, `entropy_weight_std`, `entropy_weight_min`
- `entropy_weight_zero_fraction`: share of valid tokens muted to weight 0
- `entropy_weight_advantage_rms`: RMS of the weighted advantage

Plus a set of diagnostics measured on the **unweighted** advantages, pooled over the whole
update so they do not depend on how dynamic batching cut the batch, and so they stay
comparable across modes and across runs:

- `token_entropy_advantage_corr`: Pearson correlation of token entropy and the signed
  advantage
- `token_entropy_abs_advantage_corr`: Pearson correlation of token entropy and
  `|advantage|`. A positive value means uncertain tokens carry larger-magnitude updates,
  which is what decides whether muting them is a denoiser or a mistake.
- `token_abs_advantage_slope_on_entropy`: least-squares slope of `|advantage|` on entropy,
  so the size of the effect is visible and not just its sign
- `resp_entropy_advantage_corr`, `resp_entropy_abs_advantage_corr`,
  `resp_abs_advantage_slope_on_entropy`: the same three quantities paired per response

The response-level trio is the more informative one. GRPO gives each response a single
advantage shared by all of its tokens, so a token-level correlation mixes the
between-response signal with within-response entropy variance, which only attenuates it.
Pooling also matters for a second reason: computed inside a single micro-batch the
correlation is often undefined, and averaging those undefined entries would bury the
signal.

Every key in this block is written on every micro-batch, with a neutral value when the
data cannot define it, because `reduce_metrics` means over the per-rank lists of
per-micro-batch values. A key that only some micro-batches produce leaves those lists
with different lengths across ranks and makes the step-level reduction raise. The metric
names also avoid the substrings `min` and `max`, which `reduce_metrics` treats as
aggregate-by-minimum and aggregate-by-maximum rather than by mean.

## Run

From `verl`:

```bash
recipe/entweight/scripts/run_entweight_math_qwen3_4b_a8008_s300.sh

# Resolve and print the config without training.
PRINT_CONFIG=1 recipe/entweight/scripts/run_entweight_math_qwen3_4b_a8008_s300.sh

# Preflight only: audit the assets and assert the resolved config, then exit.
PREFLIGHT_ONLY=1 recipe/entweight/scripts/run_entweight_math_qwen3_4b_a8008_s300.sh
```

Optional overrides: `TAG_OVERRIDE`, `OUT_OVERRIDE`, `ENT_WEIGHT_MODE`,
`ENT_WEIGHT_SCOPE`, `ENT_WEIGHT_RENORM`, `TOTAL_STEPS`, `WANDB_PROJECT`, `WANDB_MODE`.

## Relation to the other token-selection knobs

`entropy_token_top_fraction` and OPGD's `delta_top_fraction` / `advantage_bottom_fraction`
*hard select* tokens by dropping a fraction of them. This recipe never drops a token; it
*soft weights* every token by how confident the policy is, which keeps the response-level
normalization of the objective intact.

`neg_entropy_minmax` is the soft, per-response analogue of keeping only the most confident
tokens. It is the right direction exactly when high entropy coincides with large advantages
— read `resp_entropy_abs_advantage_corr` and the matching slope to check that instead of
assuming it. If they come out negative, high-entropy tokens carry the *small* updates and
`pos_entropy_minmax` is the setting that amplifies them.

## Tests

```bash
cd verl
python -m pytest recipe/entweight/tests/test_ent_weight.py -q
```
