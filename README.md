# OPGD

**On-Policy Gradient Distillation** — veRL-based implementation of the paper
*"Step Ahead, Distill Back: One-Step RL-Guided On-Policy Distillation"*
(Zhou Jiang, Ruotian Peng, Haoquan Zhang, Yandong Wen, Zhen Liu).

On-policy distillation (OPD) gives dense token-level supervision to reasoning
models, but it is usually supplied by a *fully reward-trained* expert in a
train-then-distill workflow. OPGD shows that a teacher only **one reward-gradient
step ahead** of the student is already enough: every iteration rebuilds the
teacher from the current actor, so it inherits accumulated learning while one
SGD step supplies fresh guidance from the latest verified outcomes.

## Method

Each training step reuses a single rollout batch for two jobs — construct the
teacher, then distill it back into the actor:

1. Sample $G$ responses per prompt from $\pi_{\theta_k}$ and evaluate verifier
   rewards.
2. Compute group-normalized (GRPO) advantages and the outcome gradient:

   $$A_i = \frac{R_i - \bar{R}_{q_i}}{\sigma_{q_i} + \varepsilon},
   \qquad
   g_k = \sum_{(i,t)\in K} w_{it}\, A_i\, s_{it},
   \quad
   s_{it} = \nabla_\theta \log \pi_{\theta_k}(x_{it}\mid c_{it}).$$

3. Construct a **virtual teacher** with one stateless SGD step
   ($\alpha$ is the lookahead rate, independent of the actor's learning rate):

   $$\theta^E_k = \theta_k + \alpha\, g_k.$$

4. Measure the teacher–student log-ratio on the sampled tokens — this is the
   signed token-level OPD advantage:

   $$d_{it} = \log \pi_{\theta^E_k}(x_{it}\mid c_{it}) - \log \pi_{\theta_k}(x_{it}\mid c_{it}).$$

5. Restore the actor exactly and optimize it with the *detached* advantages:

   $$L_{\mathrm{OPGD}}(\theta) = -\sum_{i,t} v_{it}\,\mathrm{sg}[d_{it}]\,\log \pi_\theta(x_{it}\mid c_{it}).$$

6. Advance the real optimizer and scheduler exactly once.

Linearizing step 4 gives the NTK view of **contextual credit** — a token's
coefficient sums the batch outcome signal over its score similarities with every
other sampled token:

$$d_{it} \approx \alpha\, s_{it}^\top g_k
= \alpha \sum_{j,u} w_{ju} A_j\, \underbrace{s_{it}^\top s_{ju}}_{K(it,ju)}.$$

This is why the same token can receive opposite advantages in different contexts,
and why responses with zero group advantage still get a signal from the rest of
the batch. The induced update reweights the outcome gradient through the actor's
local geometry while preserving nonnegative first-order alignment:

$$h_k = -\nabla_\theta L_{\mathrm{OPGD}}(\theta_k) \approx \alpha F_k g_k,
\qquad
F_k = \sum_{i,t} v_{it}\, s_{it} s_{it}^\top,
\qquad
g_k^\top h_k \approx \tfrac{1}{\alpha}\sum_{i,t} v_{it} d_{it}^2 \ge 0.$$

Retaining the largest $|d_{it}|$ within each response maximizes this alignment;
the checked-in config keeps all tokens (`f = 1`) and the top-10% variant sets
`advantage_top_fraction = 0.1`.

## Results

**Qwen3-4B**, 100-update budget, mean Pass@1 across AIME'24, AIME'25,
HMMT-Feb'25, HMMT-Nov'25 (32 completions/problem):

| Method                | Budget | Mean Pass@1 |
|-----------------------|--------|-------------|
| Base                  | —      | 0.157       |
| GRPO                  | 100    | 0.376       |
| GRPO (entropy 20%)    | 100    | 0.411       |
| DAPO                  | 100    | 0.429       |
| GRPO                  | 200    | 0.435       |
| GRPO (external ref.)  | 500    | 0.462       |
| **OPGD (full)**       | 100    | **0.468**   |
| **OPGD (top 10%)**    | 100    | **0.489**   |

OPGD (full) beats GRPO by 9.2 percentage points at the same 100-update budget,
and both OPGD variants surpass the 500-update external GRPO reference. Separate
training on coding prompts reaches GRPO's update-100 accuracy within 30–40
updates on HumanEval+, MBPP+, and LiveCodeBench v6. A Qwen3-8B run also shows
out-of-domain code gains, indicating the effect transfers beyond the training
task. An OPGD update is ~1.5× a GRPO update; the overhead is the virtual
backward pass and teacher scoring.

## Repository layout

- `verl/recipe/opgd/` — the OPGD recipe (trainer, controller, config,
  launchers, tests). Implementation details live in
  [`verl/recipe/opgd/README.md`](verl/recipe/opgd/README.md).
- `math_eval/` — math evaluation (AIME 2024/2025, HMMT 2025) and curve
  plotting.
- `code_eval/` — code evaluation (HumanEval+, MBPP+, LiveCodeBench).
- `src/paper.pdf` — the paper PDF.

## Environment

Training runs with the `verl` conda environment (veRL with FSDP1, synchronous
vLLM, and W&B). The launchers currently invoke
`python`; adjust that path if your environment
lives elsewhere.

## Data and model download

All datasets and models are public and downloadable from Hugging Face. No
password or API key is stored in this repository.

Expected local layout (paths shown as used on the training host; any directory
works as long as the environment variables below point at it):

```
training-data/
├── Eurus/code_train.parquet            # CODE train
├── Eurus/code_validation.parquet       # CODE validation
├── DeepMath-103K/train_filtered_level6.parquet  # math train
├── AIME2024/test.parquet               # math validation
└── AIME2025/test.parquet               # math validation
```

**CODE (Eurus).** The code train/validation parquets are the code subset of
[`PRIME-RL/Eurus-2-RL-Data`](https://huggingface.co/datasets/PRIME-RL/Eurus-2-RL-Data)
in veRL parquet format (columns `data_source`, `prompt`, `ability`,
`reward_model`, `extra_info`):

```bash
huggingface-cli download PRIME-RL/Eurus-2-RL-Data --repo-type dataset
```

**Math train (DeepMath-103K).** A difficulty-filtered subset of
[`zwhe99/DeepMath-103K`](https://huggingface.co/datasets/zwhe99/DeepMath-103K):

```bash
huggingface-cli download zwhe99/DeepMath-103K --repo-type dataset
```

**Math validation (AIME 2024/2025).** AIME 2024 and
[`math-ai/aime25`](https://huggingface.co/datasets/math-ai/aime25) test sets,
converted to the same veRL parquet format.

**Base models.** For example Qwen3-4B (math) and Qwen3-1.7B (code):

```bash
huggingface-cli download Qwen/Qwen3-4B
huggingface-cli download Qwen/Qwen3-1.7B
```

## Configuration

Host-specific paths and W&B settings are supplied through a `.env` file at the
repository root or through the environment:

```bash
MODEL_PATH=/path/to/Qwen3-4B              # math launcher (code 1.7B launcher sets its own default)
TRAIN_FILE=/path/to/train_filtered_level6.parquet   # math train
VAL_FILE_2024=/path/to/AIME2024/test.parquet        # math validation
VAL_FILE_2025=/path/to/AIME2025/test.parquet        # math validation
WANDB_API_KEY=...
WANDB_ENTITY=...
WANDB_PROJECT=OPGD
WANDB_MODE=online
```

The CODE launchers use the Eurus paths shown above directly. Optional
overrides understood by every launcher: `TAG_OVERRIDE` (experiment name),
`OUT_OVERRIDE` (checkpoint directory), `WANDB_PROJECT`, `WANDB_MODE`,
`PRINT_CONFIG`.

`.env.example` contains placeholders only. Put a rotated W&B key in the
ignored `.env` on the training host; credentials and experiment outputs are
never part of the synchronized source tree.

## Run

All launchers run from the `verl` directory. Prefix with `PRINT_CONFIG=1` to
print the resolved Hydra config without training.

```bash
recipe/opgd/scripts/run_no_tricks_math.sh
PRINT_CONFIG=1 recipe/opgd/scripts/run_no_tricks_math.sh
PRINT_CONFIG=1 recipe/opgd/scripts/run_no_tricks_code.sh
```

Each run records its launch command and resolved parameters in
`<OUT>/records/launch.txt` and checkpoints under `<OUT>/global_step_*`.
Metrics are logged to W&B (`project=OPGD` by default).

The checked-in config is the reproducible no-tricks baseline: raw SGD virtual
step, prompt centering off, OPD/RL length denormalization off, adaptive scale
off, and all token/reward-group masks off. Optional behaviors are controlled
through Hydra overrides under `algorithm.opgd.*`, including normalized SGD
(`virtual_update`), adaptive distance rescale (`adaptive_virtual_scale`),
prompt-wise centering, length denormalization, Scale-OPGD, entropy/delta
selection, uniform-reward masking, advantage-bottom selection, and multiple
cross-fitted experts (`expert_count`; `rollout.n` must be divisible by the
expert count). The `advantage_top_fraction` switch ranks `abs(final_advantage)`
independently in each response; `0.1` is the top-10% launcher setting.

`PRINT_CONFIG=1` resolves a launcher without starting training. The no-tricks
launcher must report every optional fraction as `0` and every optional mask or
rescaling switch as `false`; the top-10% launcher changes only
`algorithm.opgd.advantage_top_fraction` to `0.1`. Entropy selection requests
the actor's `entropys` field explicitly, and `export_batch_dir` can be used for
diagnostic batch snapshots without enabling an entropy trick, for example with
`+algorithm.opgd.export_batch_dir=/tmp/opgd-export`.

## Evaluation

- Math: `math_eval/scripts/run_eval_math.sh` runs `eval_math.py` over
  AIME 2024/2025 and HMMT 2025.
- Code: `code_eval/scripts/` runs HumanEval+/MBPP+ (`run_evalplus.sh`) and
  LiveCodeBench generation (`run_lcb_gen.sh`).

## Tests

CPU contracts for the profiles and safety logic:

```bash
cd verl
python -m pytest recipe/opgd/tests/test_opgd.py
```
