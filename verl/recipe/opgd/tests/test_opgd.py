"""CPU contracts for the single OPGD implementation."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from recipe.opgd.controller import (
    VirtualTeacherControllerConfig,
    runtime_rescale,
    validate_teacher,
)
from recipe.opgd.trainer import (
    _masked_l0_statistics,
    assign_opgd_expert_ids,
    build_opgd_final_advantages,
)
from verl import DataProto
from verl.workers.actor.dp_actor import _length_denormalize_advantages
from verl.workers.config.optimizer import FSDPOptimizerConfig, build_optimizer

VERL_ROOT = Path(__file__).resolve().parents[3]


def make_rollout_batch(*, teacher_valid: bool = True) -> DataProto:
    response_mask = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )
    student_log_probs = torch.full((4, 4), -2.0)
    delta = torch.tensor(
        [
            [10.0, 991.0, -777.0, 333.0],
            [-13.0, -9.0, 555.0, -444.0],
            [14.0, 18.0, 22.0, 777.0],
            [-5.0, -1.0, 3.0, 7.0],
        ]
    )
    state_value = 1.0 if teacher_valid else 0.0
    batch = DataProto.from_dict(
        tensors={
            "response_mask": response_mask,
            "advantages": torch.zeros((4, 4)),
            "old_log_probs": student_log_probs,
            "opgd_teacher_log_probs": student_log_probs + delta,
            "opgd_expert_id": torch.zeros(4, dtype=torch.int64),
            "opgd_teacher_valid": torch.full((4,), state_value),
            "opgd_optimizer_state_unchanged": torch.ones(4),
            "opgd_scheduler_state_unchanged": torch.ones(4),
        },
        non_tensors={"uid": np.array(["a", "b", "a", "b"], dtype=object)},
    )
    batch.meta_info["opgd"] = {
        "virtual_update": "normalized_sgd",
        "adaptive_virtual_scale": True,
        "expert_count": 1,
    }
    return batch


def controller_config(**overrides) -> VirtualTeacherControllerConfig:
    values = {
        "adaptive_virtual_scale": True,
        "target_logratio_abs_mean": 0.03,
        "target_logratio_low": 0.015,
        "target_logratio_high": 0.06,
        "max_logratio_abs_mean": 0.10,
        "min_runtime_scale_factor": 0.25,
        "max_runtime_scale_factor": 4.0,
        "max_rescale_passes": 1,
        "verify_restore": True,
    }
    values.update(overrides)
    return VirtualTeacherControllerConfig(**values)



def test_l0_metrics_and_per_token_advantage_records():
    values = torch.tensor([[1.0, 0.0, -2.0], [0.0, 3.0, 0.0]])
    mask = torch.tensor([[True, True, False], [True, True, False]])
    stats = _masked_l0_statistics(values, mask)
    assert stats["l0_total"] == pytest.approx(2.0)
    assert stats["l0_mean_per_sequence"] == pytest.approx(1.0)
    assert stats["l0_per_length_mean"] == pytest.approx(0.5)
    assert stats["l0_token_fraction"] == pytest.approx(0.5)

    batch = make_rollout_batch()
    batch.batch["responses"] = torch.tensor(
        [[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34], [41, 42, 43, 44]]
    )
    _, metrics = build_opgd_final_advantages(
        batch,
        prompt_centering=False,
        length_denormalization_opd=False,
        length_denormalization_rl=False,
        length_denormalization_max_tokens=4,
    )
    records = json.loads(metrics["opgd/qualitative_token_advantage_records"])
    assert len(records) == 10
    assert records[0]["token_id"] == 11
    assert {"delta_log_diff", "grpo_advantage", "final_advantage", "token_text"} <= set(records[0])
    assert metrics["opgd/delta_logdiff_l0"] == pytest.approx(2.5)
    assert metrics["opgd/delta_logdiff_l0_per_length"] == pytest.approx(1.0)
    assert metrics["opgd/advantage_l0"] == pytest.approx(2.5)
    assert metrics["opgd/advantage_l0_per_length"] == pytest.approx(1.0)


def test_expert_ids_are_balanced_per_prompt_and_rotate_each_step():
    prompt_uids = np.array(
        [uid for sample_index in range(8) for uid in ("a", "b", "c")],
        dtype=object,
    )
    sample_indices, expert_ids = assign_opgd_expert_ids(
        prompt_uids,
        responses_per_prompt=8,
        expert_count=2,
        global_step=1,
    )
    assert sample_indices.tolist() == [index for index in range(8) for _ in range(3)]
    assert expert_ids.tolist() == [1, 1, 1, 0, 0, 0] * 4
    for prompt_uid in ("a", "b", "c"):
        uid_mask = torch.from_numpy(prompt_uids == prompt_uid)
        assert torch.bincount(expert_ids[uid_mask]).tolist() == [4, 4]


def test_expert_assignment_rejects_unbalanced_crossfit():
    with pytest.raises(ValueError, match="divisible"):
        assign_opgd_expert_ids(
            np.array(["a"] * 8, dtype=object),
            responses_per_prompt=8,
            expert_count=3,
            global_step=0,
        )


def test_formula_profile_centers_per_prompt_and_denormalizes_by_length():
    batch, metrics = build_opgd_final_advantages(
        make_rollout_batch(),
        prompt_centering=True,
        length_denormalization_opd=True,
        length_denormalization_rl=False,
        length_denormalization_max_tokens=4,
    )
    expected = torch.tensor(
        [
            [-1.5, 0.0, 0.0, 0.0],
            [-5.0, -3.0, 0.0, 0.0],
            [-1.5, 1.5, 4.5, 0.0],
            [-2.0, 2.0, 6.0, 10.0],
        ]
    )
    assert torch.allclose(batch.batch["advantages"], expected)
    assert set(batch.batch.keys()).issuperset({"advantages", "opgd_update_mask"})
    assert "opgd_final_advantages" not in batch.batch
    assert metrics["opgd/mean_abs_delta"] == pytest.approx(10.2)
    assert metrics["opgd/trick1_grad_norm_enabled"] == 1.0
    assert metrics["opgd/trick2_adaptive_rescale_enabled"] == 1.0
    assert metrics["opgd/trick3_prompt_centering_enabled"] == 1.0
    assert metrics["opgd/trick4_length_denormalization_opd_enabled"] == 1.0
    assert metrics["opgd/trick4_length_denormalization_rl_enabled"] == 0.0
    assert metrics["opgd/trick5_multi_expert_enabled"] == 0.0
    assert metrics["opgd/no_tricks_profile"] == 0.0
    assert metrics["opgd/update_applied"] == 1.0



def test_scale_opgd_equalizes_response_level_advantage_scale():
    batch = make_rollout_batch()
    batch, metrics = build_opgd_final_advantages(
        batch,
        prompt_centering=False,
        length_denormalization_opd=False,
        length_denormalization_rl=False,
        length_denormalization_max_tokens=4,
        scale_opgd=True,
        scale_opgd_statistic="mean_abs",
        scale_opgd_target="batch_mean",
    )
    mask = batch.batch["response_mask"].bool()
    values = batch.batch["advantages"].abs()
    counts = mask.sum(dim=-1).float()
    response_scales = (values * mask).sum(dim=-1) / counts
    expected_target = torch.tensor([10.0, 11.0, 18.0, 4.0]).mean()
    assert torch.allclose(response_scales, torch.full_like(response_scales, expected_target))
    assert metrics["opgd/scale_opgd_enabled"] == pytest.approx(1.0)
    assert metrics["opgd/scale_opgd_target_scale"] == pytest.approx(float(expected_target))
    assert metrics["opgd/scale_opgd_scaled_response_scale_std"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["opgd/no_tricks_profile"] == pytest.approx(0.0)


def test_no_tricks_profile_uses_masked_raw_delta():
    batch = make_rollout_batch()
    batch.meta_info["opgd"] = {
        "virtual_update": "sgd",
        "adaptive_virtual_scale": False,
        "expert_count": 1,
    }
    batch, metrics = build_opgd_final_advantages(
        batch,
        prompt_centering=False,
        length_denormalization_opd=False,
        length_denormalization_rl=False,
        length_denormalization_max_tokens=4,
    )
    expected = (
        batch.batch["opgd_teacher_log_probs"] - batch.batch["old_log_probs"]
    ) * batch.batch["response_mask"]
    assert torch.equal(batch.batch["advantages"], expected)
    assert metrics["opgd/mean_abs_delta"] == pytest.approx(10.2)
    assert metrics["opgd/no_tricks_profile"] == 1.0
    for metric_name in (
        "trick1_grad_norm_enabled",
        "trick2_adaptive_rescale_enabled",
        "trick3_prompt_centering_enabled",
        "trick4_length_denormalization_opd_enabled",
        "trick4_length_denormalization_rl_enabled",
        "trick5_multi_expert_enabled",
    ):
        assert metrics[f"opgd/{metric_name}"] == 0.0


def test_rl_length_denormalization_is_independent_from_opd_signal():
    advantages = torch.ones((3, 4))
    response_mask = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, True],
        ]
    )
    rl_advantages = _length_denormalize_advantages(
        advantages,
        response_mask,
        enabled=True,
        max_tokens=4,
    )
    expected_scales = torch.tensor([0.25, 0.5, 1.0]).unsqueeze(-1)
    assert torch.allclose(rl_advantages, advantages * expected_scales)

    batch = make_rollout_batch()
    batch.meta_info["opgd"] = {
        "virtual_update": "sgd",
        "adaptive_virtual_scale": False,
        "expert_count": 1,
    }
    batch, metrics = build_opgd_final_advantages(
        batch,
        prompt_centering=False,
        length_denormalization_opd=False,
        length_denormalization_rl=True,
        length_denormalization_max_tokens=4,
    )
    raw_delta = (
        batch.batch["opgd_teacher_log_probs"] - batch.batch["old_log_probs"]
    ) * batch.batch["response_mask"]
    assert torch.equal(batch.batch["advantages"], raw_delta)
    assert metrics["opgd/trick4_length_denormalization_opd_enabled"] == 0.0
    assert metrics["opgd/trick4_length_denormalization_rl_enabled"] == 1.0
    assert metrics["opgd/no_tricks_profile"] == 0.0


def test_invalid_teacher_skips_the_only_real_update():
    batch, metrics = build_opgd_final_advantages(
        make_rollout_batch(teacher_valid=False),
        prompt_centering=True,
        length_denormalization_opd=True,
        length_denormalization_rl=False,
        length_denormalization_max_tokens=4,
    )
    assert not batch.batch["opgd_update_mask"].any()
    assert torch.count_nonzero(batch.batch["advantages"]) == 0
    assert metrics["opgd/update_applied"] == 0.0


def test_invalid_teacher_is_fail_closed_when_grpo_advantage_mixing_is_enabled():
    batch = make_rollout_batch(teacher_valid=False)
    batch.batch["advantages"] = torch.ones_like(batch.batch["advantages"])
    batch, metrics = build_opgd_final_advantages(
        batch,
        prompt_centering=False,
        use_grpo_advantage=True,
        grpo_advantage_scale=3.0,
        length_denormalization_opd=False,
        length_denormalization_rl=False,
        length_denormalization_max_tokens=4,
    )
    assert not batch.batch["opgd_update_mask"].any()
    assert torch.count_nonzero(batch.batch["advantages"]) == 0
    assert metrics["opgd/teacher_valid"] == 0.0
    assert metrics["opgd/update_applied"] == 0.0


def test_controller_rescales_once_and_enforces_restore():
    decision = runtime_rescale(0.01755, 0.12, controller_config())
    assert decision.rescale_required
    assert decision.scale_factor == pytest.approx(0.25)
    assert decision.applied_step_l2 == pytest.approx(0.01755 * 0.25)
    assert validate_teacher(
        0.03,
        finite_nonzero_gradient=True,
        restore_max_abs=0.0,
        config=controller_config(),
    ).valid
    invalid = validate_teacher(
        0.03,
        finite_nonzero_gradient=True,
        restore_max_abs=1e-8,
        config=controller_config(),
    )
    assert not invalid.valid
    assert invalid.reason == "restore_not_exact"


def test_single_config_has_no_tricks_defaults_and_no_legacy_modes():
    config = OmegaConf.load(VERL_ROOT / "recipe/opgd/config/opgd_trainer.yaml")
    opgd = config.algorithm.opgd
    assert opgd.virtual_step_l2 == pytest.approx(0.01755)
    assert opgd.virtual_sgd_learning_rate == pytest.approx(0.17)
    assert opgd.target_logratio_abs_mean == pytest.approx(0.03)
    assert opgd.target_logratio_low == pytest.approx(0.015)
    assert opgd.target_logratio_high == pytest.approx(0.06)
    assert opgd.virtual_update == "sgd"
    assert opgd.prompt_centering is False
    assert opgd.length_denormalization_opd is False
    assert opgd.length_denormalization_rl is False
    assert opgd.adaptive_virtual_scale is False
    assert opgd.delta_top_fraction == pytest.approx(0.0)
    assert opgd.entropy_top_fraction == pytest.approx(0.0)
    assert opgd.advantage_top_fraction == pytest.approx(0.0)
    assert opgd.advantage_bottom_fraction == pytest.approx(0.0)
    assert opgd.mask_uniform_reward_groups is False
    assert opgd.scale_opgd.enable is False
    assert "parameter_mode" not in opgd
    assert "update_mode" not in opgd
    assert "quality_gate_mode" not in opgd


def test_real_sgd_optimizer_uses_plain_sgd_arguments():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    config = FSDPOptimizerConfig(
        optimizer="SGD",
        optimizer_impl="torch.optim",
        lr=0.1,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        lr_scheduler_type="constant",
    )
    optimizer = build_optimizer([parameter], config)
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    assert optimizer.param_groups[0]["momentum"] == pytest.approx(0.0)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.0)

    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert torch.allclose(parameter.detach(), torch.tensor([0.9]))


def test_source_has_one_opgd_teacher_path_and_no_legacy_names():
    actor_source = (VERL_ROOT / "verl/workers/actor/dp_actor.py").read_text()
    worker_source = (VERL_ROOT / "verl/workers/fsdp_workers.py").read_text()
    trainer_source = (VERL_ROOT / "recipe/opgd/main_opgd.py").read_text()
    assert actor_source.count("def _compute_opgd_crossfit_teacher_log_prob") == 1
    assert "return self._compute_opgd_crossfit_teacher_log_prob(data)" in actor_source
    assert "self.actor_optimizer.step" not in actor_source.split(
        "def _compute_opgd_crossfit_teacher_log_prob", 1
    )[1].split("def _compute_opgd_teacher_log_prob", 1)[0]
    combined = actor_source + worker_source + trainer_source
    for legacy_name in (
        "only_reverse_kl_advantages",
        "multi_teacher_distill",
        "base_model_path",
        "base_plus_opd",
        "quality_gate",
    ):
        assert legacy_name not in combined


def test_retained_launcher_starts_python_directly_without_shell_wrappers():
    script_dir = VERL_ROOT / "recipe/opgd/scripts"
    assert not (script_dir / "run_opgd.sh").exists()
    assert not (script_dir / "run_formula.sh").exists()
    assert not (script_dir / "run_smoke.sh").exists()
    assert not (script_dir / "run_no_tricks.sh").exists()
    sources = [
        (script_dir / "run_no_tricks_math.sh").read_text(),
        (script_dir / "run_no_tricks_code.sh").read_text(),
        (script_dir / "run_no_tricks_code_17b.sh").read_text(),
    ]
    length_denorm_source = (script_dir / "run_length_denorm_code_17b.sh").read_text()
    for source in [*sources, length_denorm_source]:
        assert "-m recipe.opgd.main_opgd" in source
        assert "run_opgd.sh" not in source
        assert 'bash "$SCRIPT_DIR/' not in source

    for no_tricks_source in sources:
        for fixed_setting in (
            "algorithm.opgd.expert_count=1",
            "algorithm.opgd.virtual_update=sgd",
            "algorithm.opgd.adaptive_virtual_scale=false",
            "algorithm.opgd.max_rescale_passes=0",
            "algorithm.opgd.prompt_centering=false",
            "algorithm.opgd.length_denormalization_opd=false",
            "algorithm.opgd.length_denormalization_rl=false",
        ):
            assert fixed_setting in no_tricks_source
        for real_optimizer_setting in (
            "actor_rollout_ref.actor.optim.optimizer=AdamW",
            "actor_rollout_ref.actor.optim.lr=1e-6",
            "actor_rollout_ref.actor.optim.weight_decay=0.01",
        ):
            assert real_optimizer_setting in no_tricks_source

    for code_source in sources[1:]:
        assert "Eurus/code_train.parquet" in code_source
        assert "Eurus/code_validation.parquet" in code_source
        assert "data.max_response_length=8192" in code_source
        assert "trainer.total_training_steps=100" in code_source
        assert "trainer.test_freq=-1" in code_source

    code_17b_source = sources[2]
    assert "Qwen3-1.7B" in code_17b_source
    assert "CUDA_VISIBLE_DEVICES=0,1,2,3" in code_17b_source
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=2" in code_17b_source
    assert "trainer.n_gpus_per_node=4" in code_17b_source

    assert "Qwen3-1.7B" in length_denorm_source
    assert "CUDA_VISIBLE_DEVICES=4,5,6,7" in length_denorm_source
    assert "algorithm.opgd.length_denormalization_opd=true" in length_denorm_source
    assert "algorithm.opgd.length_denormalization_rl=true" in length_denorm_source
    assert "algorithm.opgd.adaptive_virtual_scale=false" in length_denorm_source
    assert "algorithm.opgd.prompt_centering=false" in length_denorm_source
    assert '/tmp/rld17_${STAMP}' in length_denorm_source


def test_delta_top_fraction_masks_each_response_by_abs_delta():
    batch = make_rollout_batch()
    batch, metrics = build_opgd_final_advantages(
        batch,
        prompt_centering=False,
        length_denormalization_opd=False,
        length_denormalization_rl=False,
        length_denormalization_max_tokens=4,
        delta_top_fraction=0.5,
    )
    # Valid response lengths are 1, 2, 3, 4; keep ceil(50%) per row.
    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, False, False, False],  # abs(delta): 13, 9
            [False, True, True, False],   # abs(delta): 14, 18, 22
            [True, False, False, True],     # abs(delta): 5, 1, 3, 7
        ]
    )
    assert torch.equal(batch.batch["opgd_update_mask"], expected)
    assert metrics["opgd/delta_top_fraction"] == pytest.approx(0.5)
    assert metrics["opgd/update_token_fraction"] == pytest.approx(6 / 10)


def test_advantage_top_fraction_masks_each_response_by_abs_final_advantage():
    batch = make_rollout_batch()
    batch.batch["advantages"] = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [100.0, 1.0, 0.0, 0.0],
            [0.0, 100.0, 0.0, 0.0],
            [0.0, 0.0, 100.0, 0.0],
        ]
    )
    batch, metrics = build_opgd_final_advantages(
        batch,
        prompt_centering=False,
        use_grpo_advantage=True,
        length_denormalization_opd=False,
        length_denormalization_rl=False,
        length_denormalization_max_tokens=4,
        advantage_top_fraction=0.5,
    )
    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, False, False, False],
            [False, True, True, False],
            [False, False, True, True],
        ]
    )
    assert torch.equal(batch.batch["opgd_update_mask"], expected)
    assert metrics["opgd/advantage_top_fraction"] == pytest.approx(0.5)
    assert metrics["opgd/update_token_fraction"] == pytest.approx(6 / 10)


@pytest.mark.parametrize(
    "setting",
    [
        {"delta_top_fraction": -0.01},
        {"entropy_top_fraction": 1.01},
        {"advantage_top_fraction": -0.01},
        {"advantage_bottom_fraction": 1.01},
    ],
)
def test_token_selection_fractions_reject_values_outside_unit_interval(setting):
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        build_opgd_final_advantages(
            make_rollout_batch(),
            prompt_centering=False,
            length_denormalization_opd=False,
            length_denormalization_rl=False,
            length_denormalization_max_tokens=4,
            **setting,
        )


def test_uniform_reward_groups_are_masked_out_of_the_opd_update():
    batch = make_rollout_batch()
    # uid "a" -> responses 0, 2 (both reward 1); uid "b" -> responses 1, 3 (0 vs 1).
    batch.batch["token_level_rewards"] = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    batch, metrics = build_opgd_final_advantages(
        batch,
        prompt_centering=False,
        length_denormalization_opd=False,
        length_denormalization_rl=False,
        length_denormalization_max_tokens=4,
        mask_uniform_reward_groups=True,
    )
    # Zero-variance group "a" (rows 0, 2) is dropped; group "b" is kept.
    expected = torch.tensor(
        [
            [False, False, False, False],
            [True, True, False, False],
            [False, False, False, False],
            [True, True, True, True],
        ]
    )
    assert torch.equal(batch.batch["opgd_update_mask"], expected)
    assert torch.equal(batch.batch["advantages"][[0, 2]], torch.zeros(2, 4))
    assert batch.batch["advantages"][[1, 3]].abs().sum() > 0
    assert metrics["opgd/mask_uniform_reward_groups_enabled"] == pytest.approx(1.0)
    assert metrics["opgd/uniform_reward_masked_sequences"] == pytest.approx(2.0)
    assert metrics["opgd/uniform_reward_masked_tokens"] == pytest.approx(4.0)


def test_uniform_reward_masking_is_off_by_default():
    batch = make_rollout_batch()
    batch.batch["token_level_rewards"] = torch.zeros(4, 4)
    batch, metrics = build_opgd_final_advantages(
        batch,
        prompt_centering=False,
        length_denormalization_opd=False,
        length_denormalization_rl=False,
        length_denormalization_max_tokens=4,
    )
    assert torch.equal(
        batch.batch["opgd_update_mask"], batch.batch["response_mask"].bool()
    )
    assert metrics["opgd/mask_uniform_reward_groups_enabled"] == pytest.approx(0.0)
    assert metrics["opgd/uniform_reward_masked_tokens"] == pytest.approx(0.0)
