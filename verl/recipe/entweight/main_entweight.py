"""Single entry point for the entropy-weighted GRPO recipe.

Runs the stock veRL GRPO trainer: the entropy weighting lives in the actor worker,
driven by ``actor_rollout_ref.actor.policy_loss.entropy_token_weight_*``.
"""

import hydra
from omegaconf import DictConfig

from verl.trainer.main_ppo import run_ppo


@hydra.main(config_path="config", config_name="entweight_trainer", version_base=None)
def main(config: DictConfig) -> None:
    run_ppo(config)


if __name__ == "__main__":
    main()
