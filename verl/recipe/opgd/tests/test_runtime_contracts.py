"""Fail-closed runtime contracts for OPGD configuration and entropy data."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from recipe.opgd.trainer import OPGDMixin, _ensure_opgd_entropy_request
from verl import DataProto


def _config(**overrides):
    values = {
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
            "opgd": {
                "enable": True,
                "expert_count": 1,
                "virtual_update": "sgd",
                "virtual_step_l2": 1.0,
                "virtual_sgd_learning_rate": 1.0,
                "length_denormalization_max_tokens": 8,
                "verify_restore": True,
            },
        },
        "actor_rollout_ref": {
            "model": {"lora_rank": 0},
            "actor": {
                "strategy": "fsdp",
                "ppo_epochs": 1,
                "ppo_mini_batch_size": 8,
                "use_dynamic_bsz": False,
                "entropy_coeff": 0.0,
                "use_kl_loss": False,
                "policy_loss": {"entropy_token_top_fraction": 0.0},
            },
            "rollout": {
                "name": "vllm",
                "mode": "sync",
                "n": 2,
                "calculate_log_probs": True,
                "log_prob_micro_batch_size_per_gpu": 1,
            },
        },
        "data": {"train_batch_size": 8},
        "trainer": {
            "balance_batch": True,
            "n_gpus_per_node": 4,
            "nnodes": 1,
        },
    }
    return OmegaConf.merge(OmegaConf.create(values), overrides)


def _validate(config):
    mixin = object.__new__(OPGDMixin)
    mixin.config = config
    mixin.opgd_config = config.algorithm.opgd
    mixin._validate_opgd_config()


def test_entropy_request_promotes_worker_log_prob_entropy_flag():
    config = _config()
    assert _ensure_opgd_entropy_request(config, {"export_batch_dir": "/tmp/opgd"})
    assert config.actor_rollout_ref.actor.policy_loss.entropy_token_top_fraction == 1.0


def test_entropy_request_is_disabled_for_default_no_tricks():
    config = _config()
    assert not _ensure_opgd_entropy_request(config, {"entropy_top_fraction": 0.0})
    assert config.actor_rollout_ref.actor.policy_loss.entropy_token_top_fraction == 0.0


def test_entropy_request_preserves_explicit_actor_selection_fraction():
    config = _config()
    config.actor_rollout_ref.actor.policy_loss.entropy_token_top_fraction = 0.1
    assert _ensure_opgd_entropy_request(config, {"entropy_top_fraction": 0.1})
    assert config.actor_rollout_ref.actor.policy_loss.entropy_token_top_fraction == 0.1


def test_export_allows_missing_entropy_when_entropy_is_disabled(tmp_path):
    mixin = object.__new__(OPGDMixin)
    mixin.opgd_config = {"export_batch_dir": str(tmp_path)}
    mixin.config = SimpleNamespace(
        data=SimpleNamespace(train_batch_size=2),
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=1)),
    )
    mixin.global_steps = 1
    mixin.tokenizer = SimpleNamespace(decode=lambda token_ids, skip_special_tokens=False: "x")
    batch = DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[1, 2], [3, 0]]),
            "response_mask": torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
            "advantages": torch.tensor([[0.5, 0.25], [0.0, 0.0]]),
        },
        non_tensors={"uid": np.array(["a", "b"], dtype=object)},
    )
    mixin._export_opgd_batch(batch, {}, None, {})
    payload = next(tmp_path.glob("*_responses_advantages_entropy.jsonl")).read_text()
    assert '"entropy":[null,null]' in payload


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mode", "async", "mode=sync"),
        ("name", "hf", "name=vllm"),
        ("calculate_log_probs", False, "calculate_log_probs=true"),
        ("log_prob_micro_batch_size_per_gpu", 0, "micro_batch_size"),
    ],
)
def test_rollout_contracts_fail_closed(field, value, message):
    config = _config()
    setattr(config.actor_rollout_ref.rollout, field, value)
    with pytest.raises(ValueError, match=message):
        _validate(config)


def test_single_response_rollout_is_rejected():
    config = _config()
    config.actor_rollout_ref.rollout.n = 1
    with pytest.raises(ValueError, match=r"rollout\.n >= 2"):
        _validate(config)


def test_bypass_rollout_correction_is_rejected():
    config = _config(algorithm={"rollout_correction": {"bypass_mode": True}})
    with pytest.raises(ValueError, match="bypass_mode=false"):
        _validate(config)


def test_small_batch_is_rejected_before_worker_startup():
    config = _config()
    config.data.train_batch_size = 1
    config.actor_rollout_ref.actor.ppo_mini_batch_size = 1
    with pytest.raises(ValueError, match=r"train_batch_size\*rollout\.n"):
        _validate(config)


def test_expert_fold_must_divide_actor_world_size():
    config = _config()
    config.algorithm.opgd.expert_count = 2
    config.actor_rollout_ref.rollout.n = 2
    config.data.train_batch_size = 6
    config.actor_rollout_ref.actor.ppo_mini_batch_size = 6
    # Six prompts * two responses gives 12 rows: each of two folds has six,
    # which cannot be split evenly across four actor ranks.
    with pytest.raises(ValueError, match=r"expert_count\*world_size"):
        _validate(config)
