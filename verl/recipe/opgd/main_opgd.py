"""Single entry point for formula and ablation OPGD profiles."""

import hydra
import ray
from omegaconf import DictConfig

from recipe.opgd.trainer import OPGDTrainer
from verl.trainer.main_ppo import TaskRunner as BaseTaskRunner
from verl.trainer.main_ppo import run_ppo


@ray.remote(num_cpus=1)
class TaskRunner(BaseTaskRunner):
    trainer_cls = OPGDTrainer


@hydra.main(config_path="config", config_name="opgd_trainer", version_base=None)
def main(config: DictConfig):
    run_ppo(config, task_runner_class=TaskRunner)


if __name__ == "__main__":
    main()
