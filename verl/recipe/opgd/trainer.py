"""GRPO -> virtual teacher -> OPGD update on the veRL dataflow."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.debug import marked_timer
from verl.utils.seqlen_balancing import (
    calculate_workload,
    get_seqlen_balanced_partitions,
    log_seqlen_unbalance,
)

_VIRTUAL_TEACHER_SCALARS = (
    "virtual_grad_norm",
    "virtual_update_scale",
    "virtual_weight_delta_l2",
    "teacher_logratio_abs_mean",
    "teacher_logratio_delta_mean",
    "teacher_logratio_delta_var",
    "teacher_logratio_delta_std",
    "rescale_passes",
    "restore_max_abs",
    "teacher_valid",
    "optimizer_state_unchanged",
    "scheduler_state_unchanged",
    "snapshot_local_mib",
)

_OPGD_QUALITATIVE_MAX_SEQUENCES = 4
_OPGD_QUALITATIVE_MAX_TOKENS_PER_SEQUENCE = 256
_OPGD_L0_ZERO_TOLERANCE = 1e-6
_OPGD_UNIFORM_REWARD_TOLERANCE = 1e-6


def _ensure_opgd_entropy_request(config, opgd_config) -> bool:
    """Ask actor log-prob workers for entropy when OPGD needs token entropy.

    The request is intentionally independent of ``actor.entropy_coeff``: OPGD
    uses entropy as a selection/export signal while keeping the regularization
    term disabled.  Returning the decision makes this small contract easy to
    validate without constructing a Ray worker group.
    """

    entropy_needed = bool(opgd_config.get("export_batch_dir", None)) or float(
        opgd_config.get("entropy_top_fraction", 0.0)
    ) > 0.0
    if not entropy_needed:
        return False
    with open_dict(config):
        policy_loss = config.actor_rollout_ref.actor.policy_loss
        current_fraction = float(policy_loss.get("entropy_token_top_fraction", 0.0))
        # Preserve an explicit actor-side fraction from an entropy launcher.  A
        # zero value means only that the actor would otherwise skip computing
        # entropy; in that case request all token entropies so OPGD can apply
        # its own selection mask in the trainer.
        if current_fraction <= 0.0:
            policy_loss.entropy_token_top_fraction = 1.0
    return True


def _sequence_rewards(rollout_batch: DataProto) -> torch.Tensor:
    """Per-response scalar outcome reward, mirroring GRPO's summed token reward."""
    for key in ("token_level_rewards", "token_level_scores", "rewards"):
        values = rollout_batch.batch.get(key)
        if values is not None:
            values = values.to(torch.float64)
            if values.dim() == 1:
                return values
            return values.reshape(values.shape[0], -1).sum(dim=-1)
    raise ValueError(
        "mask_uniform_reward_groups requires token_level_rewards, "
        "token_level_scores, or rewards in the rollout batch."
    )


def _uniform_reward_sequences(
    sequence_rewards: torch.Tensor,
    prompt_uids: np.ndarray,
    *,
    tolerance: float,
) -> torch.Tensor:
    """Flag responses whose prompt group has identical (zero-variance) rewards.

    When every sampled response in a prompt group earns the same outcome reward
    (all correct or all incorrect), the group carries no GRPO advantage. Its
    response tokens are excluded from the real OPD update.
    """
    rewards = sequence_rewards.detach().to(torch.float64).cpu().numpy()
    uniform = np.zeros(rewards.shape[0], dtype=bool)
    for prompt_uid in np.unique(prompt_uids):
        group_index = prompt_uids == prompt_uid
        group_rewards = rewards[group_index]
        if group_rewards.size and (
            float(group_rewards.max()) - float(group_rewards.min())
        ) <= float(tolerance):
            uniform |= group_index
    return torch.from_numpy(uniform)


def _masked_l0_statistics(
    values: torch.Tensor, mask: torch.Tensor, *, tolerance: float = _OPGD_L0_ZERO_TOLERANCE
) -> dict[str, float]:
    """Return thresholded L0 counts and response-length-normalized rates."""
    values = values.detach()
    mask = mask.bool()
    response_lengths = mask.sum(dim=-1).to(dtype=torch.float64)
    active = mask & (values.abs() > float(tolerance))
    l0_per_sequence = active.sum(dim=-1).to(dtype=torch.float64)
    nonempty = response_lengths > 0
    denominator = nonempty.sum().clamp_min(1)
    l0_mean = l0_per_sequence[nonempty].sum() / denominator
    ratio_per_sequence = l0_per_sequence / response_lengths.clamp_min(1.0)
    ratio_mean = ratio_per_sequence[nonempty].sum() / denominator
    total_tokens = response_lengths.sum().clamp_min(1.0)
    return {
        'l0_total': float(l0_per_sequence.sum().item()),
        'l0_mean_per_sequence': float(l0_mean.item()),
        'l0_per_length_mean': float(ratio_mean.item()),
        'l0_token_fraction': float((l0_per_sequence.sum() / total_tokens).item()),
    }


def _build_token_advantage_records(
    rollout_batch: DataProto,
    *,
    delta_log_diff: torch.Tensor,
    opd_signal: torch.Tensor,
    base_advantages: torch.Tensor,
    final_advantages: torch.Tensor,
    response_mask: torch.Tensor,
    update_mask: torch.Tensor,
    tokenizer: Any | None = None,
    max_sequences: int = _OPGD_QUALITATIVE_MAX_SEQUENCES,
    max_tokens_per_sequence: int = _OPGD_QUALITATIVE_MAX_TOKENS_PER_SEQUENCE,
    l0_zero_tolerance: float = _OPGD_L0_ZERO_TOLERANCE,
) -> list[dict[str, Any]]:
    """Build bounded per-token OPGD advantage records for qualitative logging."""
    if max_sequences <= 0 or max_tokens_per_sequence <= 0:
        return []
    token_ids = rollout_batch.batch.get('responses')
    uids = rollout_batch.non_tensor_batch.get('uid')
    if token_ids is None:
        token_ids = torch.full_like(response_mask, -1, dtype=torch.long)
    token_ids = token_ids.detach().cpu()
    mask_cpu = response_mask.detach().cpu()
    delta_cpu = delta_log_diff.detach().cpu()
    signal_cpu = opd_signal.detach().cpu()
    base_cpu = base_advantages.detach().cpu()
    final_cpu = final_advantages.detach().cpu()
    update_cpu = update_mask.detach().cpu()
    lengths_cpu = mask_cpu.sum(dim=-1).tolist()
    records: list[dict[str, Any]] = []
    sequence_count = min(int(mask_cpu.shape[0]), int(max_sequences))
    for row in range(sequence_count):
        positions = torch.where(mask_cpu[row])[0].tolist()
        for position in positions[:max_tokens_per_sequence]:
            token_id = int(token_ids[row, position].item())
            token_text = ''
            if tokenizer is not None:
                try:
                    token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                except Exception:
                    token_text = ''
            uid = str(uids[row]) if uids is not None else str(row)
            delta_value = float(delta_cpu[row, position].item())
            records.append(
                {
                    'sequence_index': row,
                    'uid': uid,
                    'token_position': int(position),
                    'token_id': token_id,
                    'token_text': token_text,
                    'response_length': int(lengths_cpu[row]),
                    'delta_log_diff': delta_value,
                    'opd_signal': float(signal_cpu[row, position].item()),
                    'grpo_advantage': float(base_cpu[row, position].item()),
                    'final_advantage': float(final_cpu[row, position].item()),
                    'l0_active': bool(abs(delta_value) > float(l0_zero_tolerance)),
                    'updated': bool(update_cpu[row, position].item()),
                }
            )
    return records


def assign_opgd_expert_ids(
    prompt_uids: np.ndarray,
    *,
    responses_per_prompt: int,
    expert_count: int,
    global_step: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign every response a stable sample index and a rotating expert fold."""

    if responses_per_prompt < 2:
        raise ValueError("OPGD requires rollout.n >= 2.")
    if expert_count < 1 or expert_count > responses_per_prompt:
        raise ValueError("expert_count must be in [1, rollout.n].")
    if expert_count > 1 and responses_per_prompt % expert_count:
        raise ValueError("rollout.n must be divisible by expert_count for balanced cross-fitting.")
    if len(prompt_uids) == 0:
        raise ValueError("OPGD requires at least one response.")

    uid_counters: dict[Any, int] = defaultdict(int)
    sample_indices: list[int] = []
    expert_ids: list[int] = []
    for prompt_uid in prompt_uids.tolist():
        sample_index = uid_counters[prompt_uid]
        uid_counters[prompt_uid] += 1
        sample_indices.append(sample_index)
        expert_ids.append((sample_index + global_step) % expert_count)

    expected_per_expert = responses_per_prompt // expert_count
    expert_id_array = np.asarray(expert_ids)
    for prompt_uid, response_count in uid_counters.items():
        if response_count != responses_per_prompt:
            raise ValueError(
                f"UID {prompt_uid!r} has {response_count} responses; "
                f"expected rollout.n={responses_per_prompt}."
            )
        if expert_count > 1:
            uid_mask = prompt_uids == prompt_uid
            fold_counts = np.bincount(
                expert_id_array[uid_mask], minlength=expert_count
            )
            if not np.all(fold_counts == expected_per_expert):
                raise RuntimeError(
                    f"UID {prompt_uid!r} does not have balanced OPGD expert folds."
                )

    return (
        torch.tensor(sample_indices, dtype=torch.int64),
        torch.tensor(expert_ids, dtype=torch.int64),
    )



def _response_advantage_scales(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    statistic: str,
) -> torch.Tensor:
    """Compute one positive scale statistic for every response row."""
    statistic = str(statistic).lower()
    if statistic not in {"mean_abs", "rms", "l2"}:
        raise ValueError("scale_opgd statistic must be one of: mean_abs, rms, l2.")
    mask = response_mask.bool()
    values = advantages.float()
    counts = mask.sum(dim=-1).to(dtype=values.dtype)
    safe_counts = counts.clamp_min(1.0)
    if statistic == "mean_abs":
        scales = (values.abs() * mask).sum(dim=-1) / safe_counts
    elif statistic == "rms":
        scales = ((values.square() * mask).sum(dim=-1) / safe_counts).sqrt()
    else:
        scales = (values.square() * mask).sum(dim=-1).sqrt()
    return torch.where(counts > 0, scales, torch.zeros_like(scales))


def _scale_response_level_advantages(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    statistic: str,
    target: str,
    eps: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Rescale every response to a common advantage magnitude."""
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("scale_opgd eps must be finite and positive.")
    target = str(target).lower()
    if target not in {"batch_mean", "unit"}:
        raise ValueError("scale_opgd target must be one of: batch_mean, unit.")

    mask = response_mask.bool()
    raw_scales = _response_advantage_scales(advantages, mask, statistic=statistic)
    nonempty = mask.any(dim=-1)
    active = nonempty & torch.isfinite(raw_scales) & (raw_scales > float(eps))
    if target == "batch_mean":
        target_tensor = raw_scales[active].mean() if active.any() else raw_scales.new_zeros(())
    else:
        target_tensor = raw_scales.new_ones(())
    factors = torch.zeros_like(raw_scales)
    if active.any() and float(target_tensor.item()) > 0.0:
        factors[active] = target_tensor / raw_scales[active].clamp_min(float(eps))
    scaled = advantages * factors.to(dtype=advantages.dtype).unsqueeze(-1) * mask
    scaled_scales = _response_advantage_scales(scaled, mask, statistic=statistic)

    def _mean_std(values: torch.Tensor) -> tuple[float, float]:
        valid = values[nonempty & torch.isfinite(values)]
        if not valid.numel():
            return 0.0, 0.0
        return float(valid.mean().item()), float(valid.float().std(unbiased=False).item())

    raw_mean, raw_std = _mean_std(raw_scales)
    scaled_mean, scaled_std = _mean_std(scaled_scales)
    return scaled, {
        "opgd/scale_opgd_enabled": 1.0,
        "opgd/scale_opgd_target_scale": float(target_tensor.item()),
        "opgd/scale_opgd_raw_response_scale_mean": raw_mean,
        "opgd/scale_opgd_raw_response_scale_std": raw_std,
        "opgd/scale_opgd_scaled_response_scale_mean": scaled_mean,
        "opgd/scale_opgd_scaled_response_scale_std": scaled_std,
        "opgd/scale_opgd_scaled_response_count": float(active.sum().item()),
        "opgd/scale_opgd_factor_mean": float(factors[active].mean().item()) if active.any() else 0.0,
        "opgd/scale_opgd_factor_std": (
            float(factors[active].float().std(unbiased=False).item())
            if active.any()
            else 0.0
        ),
    }

def build_opgd_final_advantages(
    rollout_batch: DataProto,
    *,
    prompt_centering: bool,
    use_grpo_advantage: bool = False,
    grpo_advantage_scale: float = 1.0,
    length_denormalization_opd: bool,
    length_denormalization_rl: bool,
    delta_top_fraction: float = 0.0,
    entropy_top_fraction: float = 0.0,
    advantage_top_fraction: float = 0.0,
    advantage_bottom_fraction: float = 0.0,
    scale_opgd: bool = False,
    scale_opgd_statistic: str = "mean_abs",
    scale_opgd_target: str = "batch_mean",
    scale_opgd_eps: float = 1.0e-6,
    length_denormalization_max_tokens: int,
    tokenizer: Any | None = None,
    l0_zero_tolerance: float = _OPGD_L0_ZERO_TOLERANCE,
    mask_uniform_reward_groups: bool = False,
    uniform_reward_tolerance: float = _OPGD_UNIFORM_REWARD_TOLERANCE,
) -> tuple[DataProto, dict[str, Any]]:
    """Build the token OPD signal with the five experiment switches explicit."""

    if not math.isfinite(float(grpo_advantage_scale)):
        raise ValueError("grpo_advantage_scale must be finite.")
    if length_denormalization_max_tokens <= 0:
        raise ValueError("length_denormalization_max_tokens must be positive.")
    if not math.isfinite(float(l0_zero_tolerance)) or float(l0_zero_tolerance) < 0.0:
        raise ValueError("l0_zero_tolerance must be finite and non-negative.")
    if (
        not math.isfinite(float(uniform_reward_tolerance))
        or float(uniform_reward_tolerance) < 0.0
    ):
        raise ValueError("uniform_reward_tolerance must be finite and non-negative.")
    if not math.isfinite(float(delta_top_fraction)) or not (0.0 <= float(delta_top_fraction) <= 1.0):
        raise ValueError("delta_top_fraction must be finite and in [0, 1].")
    if not math.isfinite(float(entropy_top_fraction)) or not (0.0 <= float(entropy_top_fraction) <= 1.0):
        raise ValueError("entropy_top_fraction must be finite and in [0, 1].")
    if not math.isfinite(float(advantage_top_fraction)) or not (0.0 <= float(advantage_top_fraction) <= 1.0):
        raise ValueError("advantage_top_fraction must be finite and in [0, 1].")
    if not math.isfinite(float(advantage_bottom_fraction)) or not (0.0 <= float(advantage_bottom_fraction) <= 1.0):
        raise ValueError("advantage_bottom_fraction must be finite and in [0, 1].")
    if str(scale_opgd_statistic).lower() not in {"mean_abs", "rms", "l2"}:
        raise ValueError("scale_opgd statistic must be one of: mean_abs, rms, l2.")
    if str(scale_opgd_target).lower() not in {"batch_mean", "unit"}:
        raise ValueError("scale_opgd target must be one of: batch_mean, unit.")
    if not math.isfinite(float(scale_opgd_eps)) or float(scale_opgd_eps) <= 0.0:
        raise ValueError("scale_opgd eps must be finite and positive.")

    response_mask = rollout_batch.batch["response_mask"].bool()
    base_advantages = rollout_batch.batch.get("advantages", torch.zeros_like(response_mask, dtype=torch.float32))
    student_log_probs = rollout_batch.batch["old_log_probs"]
    virtual_teacher_log_probs = rollout_batch.batch["opgd_teacher_log_probs"]
    expert_ids = rollout_batch.batch["opgd_expert_id"].long()
    prompt_uids = rollout_batch.non_tensor_batch["uid"]

    logratio_delta = virtual_teacher_log_probs - student_log_probs
    opd_signal = logratio_delta * response_mask
    if prompt_centering:
        opd_signal = torch.zeros_like(logratio_delta)
        expert_ids_cpu = expert_ids.cpu().numpy()
        for prompt_uid in np.unique(prompt_uids):
            uid_mask = prompt_uids == prompt_uid
            for expert_id in np.unique(expert_ids_cpu[uid_mask]):
                sequence_mask_np = uid_mask & (expert_ids_cpu == expert_id)
                sequence_mask = torch.from_numpy(sequence_mask_np).to(
                    response_mask.device
                )
                token_mask = response_mask & sequence_mask.unsqueeze(-1)
                fold_delta = logratio_delta[token_mask]
                if fold_delta.numel():
                    opd_signal[token_mask] = fold_delta - fold_delta.mean()

    teacher_is_valid = bool(
        torch.isfinite(rollout_batch.batch["opgd_teacher_valid"]).all()
        and rollout_batch.batch["opgd_teacher_valid"].min() >= 1.0
        and torch.isfinite(
            rollout_batch.batch["opgd_optimizer_state_unchanged"]
        ).all()
        and rollout_batch.batch["opgd_optimizer_state_unchanged"].min() >= 1.0
        and torch.isfinite(
            rollout_batch.batch["opgd_scheduler_state_unchanged"]
        ).all()
        and rollout_batch.batch["opgd_scheduler_state_unchanged"].min() >= 1.0
        and torch.isfinite(logratio_delta[response_mask]).all()
    )
    if not teacher_is_valid:
        opd_signal.zero_()

    response_lengths = response_mask.sum(dim=-1)
    if length_denormalization_opd:
        length_denormalization_opd_scale = response_lengths.to(
            logratio_delta.dtype
        ) / float(
            length_denormalization_max_tokens
        )
    else:
        length_denormalization_opd_scale = torch.ones_like(
            response_lengths, dtype=logratio_delta.dtype
        )
    if length_denormalization_rl:
        length_denormalization_rl_scale = response_lengths.to(
            logratio_delta.dtype
        ) / float(length_denormalization_max_tokens)
    else:
        length_denormalization_rl_scale = torch.ones_like(
            response_lengths, dtype=logratio_delta.dtype
        )
    combined_signal = opd_signal
    if use_grpo_advantage:
        combined_signal = combined_signal + float(grpo_advantage_scale) * base_advantages
    final_advantages = (
        combined_signal * length_denormalization_opd_scale.unsqueeze(-1) * response_mask
    )
    if scale_opgd:
        final_advantages, scale_metrics = _scale_response_level_advantages(
            final_advantages,
            response_mask,
            statistic=scale_opgd_statistic,
            target=scale_opgd_target,
            eps=scale_opgd_eps,
        )
    else:
        scale_metrics = {
            "opgd/scale_opgd_enabled": 0.0,
            "opgd/scale_opgd_target_scale": 0.0,
            "opgd/scale_opgd_raw_response_scale_mean": 0.0,
            "opgd/scale_opgd_raw_response_scale_std": 0.0,
            "opgd/scale_opgd_scaled_response_scale_mean": 0.0,
            "opgd/scale_opgd_scaled_response_scale_std": 0.0,
            "opgd/scale_opgd_scaled_response_count": 0.0,
            "opgd/scale_opgd_factor_mean": 0.0,
            "opgd/scale_opgd_factor_std": 0.0,
        }
    if advantage_top_fraction > 0.0:
        valid_counts = response_mask.sum(dim=-1).to(torch.long)
        keep_counts = (
            torch.ceil(valid_counts.float() * float(advantage_top_fraction))
            .to(torch.long)
            .clamp_min(1)
        )
        advantage_top_ranks = torch.argsort(
            torch.argsort(
                final_advantages.abs().masked_fill(~response_mask, -torch.inf),
                dim=-1,
                descending=True,
            ),
            dim=-1,
        )
        advantage_top_mask = (
            advantage_top_ranks < keep_counts.unsqueeze(-1)
        ) & response_mask
    else:
        advantage_top_mask = response_mask
    if delta_top_fraction > 0.0:
        valid_counts = response_mask.sum(dim=-1).to(torch.long)
        keep_counts = torch.ceil(valid_counts.float() * float(delta_top_fraction)).to(torch.long).clamp_min(1)
        ranks = torch.argsort(
            torch.argsort(
                logratio_delta.abs().masked_fill(~response_mask, -torch.inf),
                dim=-1,
                descending=True,
            ),
            dim=-1,
        )
        delta_top_mask = (ranks < keep_counts.unsqueeze(-1)) & response_mask
    else:
        delta_top_mask = response_mask
    if entropy_top_fraction > 0.0:
        entropys = rollout_batch.batch.get("entropys")
        if entropys is None:
            raise ValueError("entropy_top_fraction requires entropys in the rollout batch.")
        valid_counts = response_mask.sum(dim=-1).to(torch.long)
        keep_counts = torch.ceil(valid_counts.float() * float(entropy_top_fraction)).to(torch.long).clamp_min(1)
        entropy_ranks = torch.argsort(
            torch.argsort(
                entropys.to(logratio_delta.dtype).masked_fill(~response_mask, -torch.inf),
                dim=-1,
                descending=True,
            ),
            dim=-1,
        )
        entropy_top_mask = (entropy_ranks < keep_counts.unsqueeze(-1)) & response_mask
    else:
        entropy_top_mask = response_mask
    if advantage_bottom_fraction > 0.0:
        valid_counts = response_mask.sum(dim=-1).to(torch.long)
        keep_counts = torch.ceil(valid_counts.float() * float(advantage_bottom_fraction)).to(torch.long).clamp_min(1)
        advantage_ranks = torch.argsort(
            torch.argsort(
                final_advantages.abs().masked_fill(~response_mask, torch.inf),
                dim=-1,
                descending=False,
            ),
            dim=-1,
        )
        advantage_bottom_mask = (advantage_ranks < keep_counts.unsqueeze(-1)) & response_mask
    else:
        advantage_bottom_mask = response_mask
    update_mask = (
        response_mask
        & delta_top_mask
        & entropy_top_mask
        & advantage_top_mask
        & advantage_bottom_mask
        & torch.isfinite(final_advantages)
    )
    uniform_reward_mask = torch.zeros_like(update_mask)
    if mask_uniform_reward_groups:
        uniform_reward_mask = (
            _uniform_reward_sequences(
                _sequence_rewards(rollout_batch),
                prompt_uids,
                tolerance=uniform_reward_tolerance,
            )
            .to(update_mask.device)
            .unsqueeze(-1)
            .expand_as(update_mask)
        )
        update_mask = update_mask & ~uniform_reward_mask
        final_advantages = final_advantages.masked_fill(uniform_reward_mask, 0.0)
    # Invalid virtual-teacher state is fail-closed.  In particular, GRPO
    # advantage mixing must never turn a failed teacher check into a real
    # optimizer update.
    if not teacher_is_valid:
        final_advantages.zero_()
        update_mask.zero_()
    update_applied = bool(final_advantages[update_mask].abs().sum() > 0.0)
    if not update_applied:
        update_mask.zero_()

    rollout_batch.batch["opgd_update_mask"] = update_mask
    rollout_batch.batch["advantages"] = final_advantages

    valid_delta = logratio_delta[response_mask]
    valid_centered = opd_signal[response_mask]
    valid_signal = opd_signal[response_mask]
    valid_final = final_advantages[response_mask]
    mean_abs_delta = float(valid_delta.abs().mean()) if valid_delta.numel() else 0.0
    delta_var = (
        float(valid_delta.float().var(unbiased=False))
        if valid_delta.numel()
        else 0.0
    )
    centered_delta_var = (
        float(valid_centered.float().var(unbiased=False))
        if valid_centered.numel()
        else 0.0
    )
    normalized_sgd = str(rollout_batch.meta_info["opgd"]["virtual_update"]) == "normalized_sgd"
    adaptive_rescale = bool(
        rollout_batch.meta_info["opgd"]["adaptive_virtual_scale"]
    )
    multi_expert = int(rollout_batch.meta_info["opgd"]["expert_count"]) > 1
    metrics = {
        "opgd/trick1_grad_norm_enabled": float(normalized_sgd),
        "opgd/trick2_adaptive_rescale_enabled": float(adaptive_rescale),
        "opgd/trick3_prompt_centering_enabled": float(prompt_centering),
        "opgd/trick4_length_denormalization_opd_enabled": float(
            length_denormalization_opd
        ),
        "opgd/trick4_length_denormalization_rl_enabled": float(
            length_denormalization_rl
        ),
        "opgd/trick5_multi_expert_enabled": float(multi_expert),
        "opgd/no_tricks_profile": float(
            not normalized_sgd
            and not adaptive_rescale
            and not prompt_centering
            and not length_denormalization_opd
            and not length_denormalization_rl
            and not multi_expert
            and not use_grpo_advantage
            and not scale_opgd
            and delta_top_fraction <= 0.0
            and entropy_top_fraction <= 0.0
            and advantage_top_fraction <= 0.0
            and advantage_bottom_fraction <= 0.0
            and not mask_uniform_reward_groups
        ),
        "opgd/teacher_valid": float(teacher_is_valid),
        "opgd/expert_count": float(torch.unique(expert_ids).numel()),
        "opgd/logratio_delta_mean": float(valid_delta.mean()) if valid_delta.numel() else 0.0,
        "opgd/logratio_delta_var": delta_var,
        "opgd/logratio_delta_std": math.sqrt(max(delta_var, 0.0)),
        "opgd/mean_abs_delta": mean_abs_delta,
        "opgd/prompt_centered_delta_var": centered_delta_var,
        "opgd/prompt_centered_delta_std": math.sqrt(max(centered_delta_var, 0.0)),
        "opgd/opd_signal_mean": float(valid_signal.mean()) if valid_signal.numel() else 0.0,
        "opgd/length_denormalization_opd_scale_mean": float(
            length_denormalization_opd_scale.float().mean()
        ),
        "opgd/length_denormalization_rl_scale_mean": float(
            length_denormalization_rl_scale.float().mean()
        ),
        "opgd/final_advantage_rms": (
            float(valid_final.float().square().mean().sqrt())
            if valid_final.numel()
            else 0.0
        ),
        "opgd/update_tokens": float(update_mask.sum()),
        "opgd/update_token_fraction": float(update_mask.sum()) / float(max(int(response_mask.sum()), 1)),
        "opgd/delta_top_fraction": float(delta_top_fraction),
        "opgd/entropy_top_fraction": float(entropy_top_fraction),
        "opgd/advantage_top_fraction": float(advantage_top_fraction),
        "opgd/advantage_bottom_fraction": float(advantage_bottom_fraction),
        "opgd/mask_uniform_reward_groups_enabled": float(mask_uniform_reward_groups),
        "opgd/uniform_reward_tolerance": float(uniform_reward_tolerance),
        "opgd/uniform_reward_masked_sequences": float(
            uniform_reward_mask.any(dim=-1).sum()
        ),
        "opgd/uniform_reward_masked_tokens": float(
            (uniform_reward_mask & response_mask).sum()
        ),
        "opgd/uniform_reward_masked_token_fraction": float(
            (uniform_reward_mask & response_mask).sum()
        )
        / float(max(int(response_mask.sum()), 1)),
        "opgd/update_applied": float(update_applied),
    }
    metrics.update(scale_metrics)
    delta_l0 = _masked_l0_statistics(
        logratio_delta, response_mask, tolerance=l0_zero_tolerance
    )
    advantage_l0 = _masked_l0_statistics(
        final_advantages, response_mask, tolerance=l0_zero_tolerance
    )
    token_records = _build_token_advantage_records(
        rollout_batch,
        delta_log_diff=logratio_delta,
        opd_signal=opd_signal,
        base_advantages=base_advantages,
        final_advantages=final_advantages,
        response_mask=response_mask,
        update_mask=update_mask,
        l0_zero_tolerance=l0_zero_tolerance,
        tokenizer=tokenizer,
    )
    token_records_json = json.dumps(
        token_records, ensure_ascii=False, separators=(",", ":")
    )
    response_token_count = int(response_mask.sum().item())
    metrics.update(
        {
            "opgd/delta_logdiff_l0_total": delta_l0["l0_total"],
            "opgd/delta_logdiff_l0": delta_l0["l0_mean_per_sequence"],
            "opgd/delta_logdiff_l0_per_length": delta_l0["l0_per_length_mean"],
            "opgd/delta_logdiff_l0_token_fraction": delta_l0["l0_token_fraction"],
            "opgd/advantage_l0_total": advantage_l0["l0_total"],
            "opgd/advantage_l0": advantage_l0["l0_mean_per_sequence"],
            "opgd/advantage_l0_per_length": advantage_l0["l0_per_length_mean"],
            "opgd/advantage_l0_token_fraction": advantage_l0["l0_token_fraction"],
            "opgd/l0_zero_tolerance": float(l0_zero_tolerance),
            "opgd/qualitative_token_records_count": float(len(token_records)),
            "opgd/qualitative_token_coverage": (
                float(len(token_records)) / float(max(response_token_count, 1))
            ),
            "opgd/qualitative_token_advantage_records": token_records_json,
        }
    )
    return rollout_batch, metrics


class OPGDMixin:
    """OPGD hooks layered on veRL's ordinary GRPO trainer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opgd_config = self.config.algorithm.get("opgd", {})
        # OPGD computes token-level entropy on the driver for either an entropy
        # selection or a batch export.  The actor worker normally skips entropy
        # when the entropy coefficient is zero, so promote the worker-side
        # request automatically instead of relying on every launcher to repeat
        # this implementation detail.  A value of one keeps all tokens in the
        # actor loss; OPGD's own update mask still controls which tokens train.
        _ensure_opgd_entropy_request(self.config, self.opgd_config)
        self._validate_opgd_config()

    def _cfg(self, key: str, default: Any = None):
        return self.opgd_config.get(key, default)

    def _validate_opgd_config(self) -> None:
        actor = self.config.actor_rollout_ref.actor
        rollout = self.config.actor_rollout_ref.rollout
        expert_count = int(self._cfg("expert_count", 1))
        responses_per_prompt = int(rollout.n)
        if not bool(self._cfg("enable", False)):
            raise ValueError("OPGD requires algorithm.opgd.enable=true.")
        advantage_estimator = getattr(
            self.config.algorithm.adv_estimator,
            "value",
            self.config.algorithm.adv_estimator,
        )
        if str(advantage_estimator) != "grpo":
            raise ValueError("OPGD requires algorithm.adv_estimator=grpo.")
        if responses_per_prompt < 2:
            raise ValueError("OPGD requires rollout.n >= 2 for cross-response teacher estimation.")
        if expert_count < 1 or expert_count > responses_per_prompt:
            raise ValueError("expert_count must be in [1, rollout.n].")
        if expert_count > 1 and responses_per_prompt % expert_count:
            raise ValueError("rollout.n must be divisible by expert_count.")
        if expert_count > 1 and not bool(self.config.trainer.balance_batch):
            raise ValueError("Multi-expert OPGD requires trainer.balance_batch=true.")
        if self._cfg("virtual_update", "normalized_sgd") not in {
            "normalized_sgd",
            "sgd",
        }:
            raise ValueError("virtual_update must be normalized_sgd or sgd.")
        virtual_step_l2 = float(self._cfg("virtual_step_l2", 0.01755))
        if not math.isfinite(virtual_step_l2) or virtual_step_l2 <= 0.0:
            raise ValueError("virtual_step_l2 must be finite and positive.")
        virtual_sgd_learning_rate = float(
            self._cfg("virtual_sgd_learning_rate", 0.17)
        )
        if (
            not math.isfinite(virtual_sgd_learning_rate)
            or virtual_sgd_learning_rate <= 0.0
        ):
            raise ValueError("virtual_sgd_learning_rate must be finite and positive.")
        if int(self._cfg("length_denormalization_max_tokens")) <= 0:
            raise ValueError("length_denormalization_max_tokens must be positive.")
        uniform_reward_tolerance = float(
            self._cfg("uniform_reward_tolerance", _OPGD_UNIFORM_REWARD_TOLERANCE)
        )
        if not math.isfinite(uniform_reward_tolerance) or uniform_reward_tolerance < 0.0:
            raise ValueError(
                "uniform_reward_tolerance must be finite and non-negative."
            )
        for fraction_name in (
            "delta_top_fraction",
            "entropy_top_fraction",
            "advantage_top_fraction",
            "advantage_bottom_fraction",
        ):
            fraction = float(self._cfg(fraction_name, 0.0))
            if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
                raise ValueError(f"{fraction_name} must be finite and in [0, 1].")
        if int(self.config.actor_rollout_ref.model.get("lora_rank", 0)) != 0:
            raise ValueError("OPGD requires model.lora_rank=0.")
        if str(actor.strategy) != "fsdp":
            raise ValueError("OPGD requires FSDP1 actor training.")
        if int(actor.ppo_epochs) != 1:
            raise ValueError("OPGD requires actor.ppo_epochs=1.")
        if int(actor.ppo_mini_batch_size) != int(self.config.data.train_batch_size):
            raise ValueError(
                "OPGD requires one full-batch mini-batch and one real optimizer step."
            )
        if bool(actor.use_dynamic_bsz):
            raise ValueError("OPGD requires actor.use_dynamic_bsz=false.")
        if float(actor.entropy_coeff) != 0.0:
            raise ValueError("OPGD requires actor.entropy_coeff=0.")
        if bool(actor.use_kl_loss) or bool(self.config.algorithm.use_kl_in_reward):
            raise ValueError("OPGD requires actor and reward KL to be disabled.")
        if not bool(rollout.calculate_log_probs):
            raise ValueError("OPGD requires rollout.calculate_log_probs=true.")
        rollout_correction = self.config.algorithm.get("rollout_correction", {})
        if bool(rollout_correction.get("bypass_mode", False)):
            raise ValueError(
                "OPGD requires decoupled rollout correction (bypass_mode=false) "
                "so actor old_log_probs remain available."
            )
        if str(rollout.mode) != "sync":
            raise ValueError("OPGD requires synchronous rollout.mode=sync.")
        if str(rollout.name) != "vllm":
            raise ValueError("OPGD requires rollout.name=vllm.")
        # OPGD uses rollout log-probs as the old-policy anchor and then
        # performs a second actor-side log-prob pass for the virtual teacher.
        # Refuse an unset/empty micro-batch size because it otherwise fails deep
        # inside FSDP with an opaque empty-batch error.
        log_prob_micro_batch_size = rollout.get("log_prob_micro_batch_size_per_gpu", None)
        if log_prob_micro_batch_size is None:
            log_prob_micro_batch_size = rollout.get("log_prob_micro_batch_size", None)
        if log_prob_micro_batch_size is None or int(log_prob_micro_batch_size) <= 0:
            raise ValueError(
                "OPGD requires a positive rollout.log_prob_micro_batch_size_per_gpu "
                "(or legacy rollout.log_prob_micro_batch_size)."
            )
        try:
            world_size = int(self.config.trainer.n_gpus_per_node) * int(self.config.trainer.nnodes)
        except (AttributeError, TypeError, ValueError):
            world_size = 1
        if world_size <= 0:
            raise ValueError("OPGD requires a positive actor world size.")
        train_batch_size = int(self.config.data.train_batch_size)
        total_rows = train_batch_size * responses_per_prompt
        if total_rows < world_size:
            raise ValueError(
                "OPGD train_batch_size*rollout.n must be at least the actor world size "
                f"({total_rows} < {world_size})."
            )
        if total_rows % world_size:
            raise ValueError(
                "OPGD train_batch_size*rollout.n must be divisible by the actor world size "
                f"({total_rows} % {world_size} != 0)."
            )
        if total_rows % (expert_count * world_size):
            raise ValueError(
                "Each OPGD expert fold must divide the actor world size; require "
                "train_batch_size*rollout.n divisible by expert_count*world_size "
                f"({total_rows} % {expert_count * world_size} != 0)."
            )
        if not bool(self._cfg("verify_restore", True)):
            raise ValueError("OPGD requires verify_restore=true.")

    def _plain_config(self) -> dict[str, Any]:
        return OmegaConf.to_container(self.opgd_config, resolve=True)

    def _export_opgd_batch(
        self,
        batch: DataProto,
        timing_raw: dict,
        base_advantages_cpu: torch.Tensor | None,
        signal_metrics: dict[str, Any],
    ) -> None:
        """Persist every rollout response and token-level OPGD diagnostics."""
        export_dir = self._cfg("export_batch_dir", None)
        if not export_dir:
            return
        export_path = Path(str(export_dir))
        export_path.mkdir(parents=True, exist_ok=True)

        def cpu_tensor(name: str):
            value = batch.batch.get(name)
            return value.detach().cpu() if torch.is_tensor(value) else None

        responses = cpu_tensor("responses")
        response_mask = cpu_tensor("response_mask")
        entropys = cpu_tensor("entropys")
        final_advantages = cpu_tensor("advantages")
        old_log_probs = cpu_tensor("old_log_probs")
        teacher_log_probs = cpu_tensor("opgd_teacher_log_probs")
        update_mask = cpu_tensor("opgd_update_mask")
        sample_indices = cpu_tensor("opgd_sample_index")
        expert_ids = cpu_tensor("opgd_expert_id")
        if responses is None or response_mask is None or final_advantages is None:
            missing = [
                name
                for name, value in (
                    ("responses", responses),
                    ("response_mask", response_mask),
                    ("advantages", final_advantages),
                )
                if value is None
            ]
            raise RuntimeError(f"Cannot export complete OPGD batch; missing {missing}.")
        if base_advantages_cpu is None:
            base_advantages_cpu = torch.zeros_like(final_advantages)
        mask_cpu = response_mask.bool()
        rows = int(responses.shape[0])
        if rows != int(self.config.data.train_batch_size) * int(self.config.actor_rollout_ref.rollout.n):
            raise RuntimeError(
                f"Expected train_batch_size*rollout.n rows "
                f"({int(self.config.data.train_batch_size)}*{int(self.config.actor_rollout_ref.rollout.n)}), "
                f"got {rows}."
            )

        uids = batch.non_tensor_batch.get("uid")
        uid_counts: dict[str, int] = {}
        response_jsonl = export_path / f"step_{int(self.global_steps)}_responses_advantages_entropy.jsonl"

        def finite_float(value: Any):
            value = float(value)
            return value if math.isfinite(value) else None

        with response_jsonl.open("w", encoding="utf-8") as handle:
            for row in range(rows):
                positions = torch.where(mask_cpu[row])[0].tolist()
                token_ids = [int(v) for v in responses[row, positions].tolist()]
                uid = str(uids[row]) if uids is not None else str(row)
                response_index = uid_counts.get(uid, 0)
                uid_counts[uid] = response_index + 1
                try:
                    response_text = self.tokenizer.decode(
                        token_ids, skip_special_tokens=False
                    )
                except Exception:
                    response_text = ""
                record = {
                    "step": int(self.global_steps),
                    "sequence_index": row,
                    "uid": uid,
                    "response_index": response_index,
                    "response_length": len(token_ids),
                    "response_token_ids": token_ids,
                    "response_text": response_text,
                    "entropy": (
                        [finite_float(entropys[row, pos].item()) for pos in positions]
                        if entropys is not None
                        else [None] * len(positions)
                    ),
                    "advantage": [
                        finite_float(final_advantages[row, pos].item()) for pos in positions
                    ],
                    "grpo_advantage": [
                        finite_float(base_advantages_cpu[row, pos].item()) for pos in positions
                    ],
                    "response_mask": [1] * len(positions),
                    "update_mask": (
                        [bool(update_mask[row, pos].item()) for pos in positions]
                        if update_mask is not None
                        else None
                    ),
                    "old_log_prob": (
                        [finite_float(old_log_probs[row, pos].item()) for pos in positions]
                        if old_log_probs is not None
                        else None
                    ),
                    "opgd_teacher_log_prob": (
                        [finite_float(teacher_log_probs[row, pos].item()) for pos in positions]
                        if teacher_log_probs is not None
                        else None
                    ),
                    "opgd_sample_index": (
                        int(sample_indices[row].item()) if sample_indices is not None else None
                    ),
                    "opgd_expert_id": (
                        int(expert_ids[row].item()) if expert_ids is not None else None
                    ),
                }
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

        raw_timing = {}
        for key, value in timing_raw.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                raw_timing[str(key)] = float(value)
        step_seconds = float(raw_timing.get("step", 0.0))
        stage_keys = (
            "gen",
            "reward",
            "old_log_prob",
            "ref",
            "values",
            "adv",
            "update_critic",
            "update_actor",
        )
        stages = {key: max(float(raw_timing.get(key, 0.0)), 0.0) for key in stage_keys}
        accounted = sum(stages.values())
        stages["other"] = max(step_seconds - accounted, 0.0)
        denominator = step_seconds if step_seconds > 0.0 else sum(stages.values())
        proportions = {
            key: (value / denominator if denominator > 0.0 else 0.0)
            for key, value in stages.items()
        }
        timing_payload = {
            "step": int(self.global_steps),
            "rows": rows,
            "raw_seconds": raw_timing,
            "stages_seconds": stages,
            "proportions": proportions,
            "nested_timers_not_in_top_level_breakdown": {
                key: raw_timing[key]
                for key in ("opgd_virtual_teacher",)
                if key in raw_timing
            },
        }
        (export_path / f"step_{int(self.global_steps)}_timing.json").write_text(
            json.dumps(timing_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary = {
            "step": int(self.global_steps),
            "rows": rows,
            "expected_rows": int(self.config.data.train_batch_size)
            * int(self.config.actor_rollout_ref.rollout.n),
            "unique_prompts": len(uid_counts),
            "responses_per_prompt": int(self.config.actor_rollout_ref.rollout.n),
            "total_response_tokens": int(mask_cpu.sum().item()),
            "response_jsonl": str(response_jsonl),
            "timing_json": str(export_path / f"step_{int(self.global_steps)}_timing.json"),
            "signal_metrics": {
                key: value
                for key, value in signal_metrics.items()
                if isinstance(value, (int, float, np.integer, np.floating))
            },
        }
        (export_path / f"step_{int(self.global_steps)}_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"[OPGD export] wrote {rows} responses to {response_jsonl}; "
            f"{int(mask_cpu.sum().item())} response tokens"
        )

    def _postprocess_rollout_batch(self, batch: DataProto, metrics: dict) -> DataProto:
        expert_count = int(self._cfg("expert_count", 1))
        sample_indices, expert_ids = assign_opgd_expert_ids(
            batch.non_tensor_batch["uid"],
            responses_per_prompt=int(self.config.actor_rollout_ref.rollout.n),
            expert_count=expert_count,
            global_step=int(self.global_steps),
        )
        batch.batch["opgd_sample_index"] = sample_indices
        batch.batch["opgd_expert_id"] = expert_ids
        metrics.update(
            {
                "opgd/expert_count": expert_count,
                "opgd/crossfit_enabled": float(expert_count > 1),
                "opgd/rollout_calls": 1.0,
            }
        )
        return batch

    def _balance_batch(
        self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False
    ):
        expert_count = int(self._cfg("expert_count", 1))
        if expert_count == 1 or "opgd_expert_id" not in batch.batch:
            return super()._balance_batch(batch, metrics, logging_prefix, keep_minibatch)

        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        workloads = calculate_workload(attention_mask.view(batch_size, -1).sum(-1))
        world_size = self.actor_rollout_wg.world_size
        expert_ids = batch.batch["opgd_expert_id"].long()
        combined_partitions = [[] for _ in range(world_size)]
        for expert_id in range(expert_count):
            expert_indices = torch.where(expert_ids == expert_id)[0]
            if len(expert_indices) % world_size:
                raise ValueError(
                    "Each OPGD expert fold must divide the actor world size."
                )
            fold_partitions = get_seqlen_balanced_partitions(
                workloads[expert_indices], k_partitions=world_size, equal_size=True
            )
            for rank, fold_partition in enumerate(fold_partitions):
                combined_partitions[rank].extend(
                    expert_indices[fold_partition].tolist()
                )
        for rank, partition in enumerate(combined_partitions):
            partition.sort(key=lambda index: (workloads[index], index))
            combined_partitions[rank] = partition[::2] + partition[1::2][::-1]
        global_indices = torch.tensor(
            [index for partition in combined_partitions for index in partition]
        )
        batch.reorder(global_indices)
        metrics.update(
            log_seqlen_unbalance(
                seqlen_list=workloads,
                partitions=combined_partitions,
                prefix=f"{logging_prefix}_opgd",
            )
        )

    def _postprocess_advantage_batch(
        self, batch: DataProto, metrics: dict, timing_raw: dict
    ) -> DataProto:
        batch.meta_info["opgd"] = self._plain_config()
        with marked_timer("opgd_virtual_teacher", timing_raw, color="purple"):
            teacher_output = self.actor_rollout_wg.compute_opgd_teacher_log_prob(batch)
        batch = batch.union(teacher_output)
        export_dir = self._cfg("export_batch_dir", None)
        base_advantages_for_export = None
        if export_dir:
            base_advantages = batch.batch.get("advantages")
            if base_advantages is not None:
                base_advantages_for_export = base_advantages.detach().cpu()
        for scalar_name in _VIRTUAL_TEACHER_SCALARS:
            scalar_values = batch.batch[f"opgd_{scalar_name}"]
            finite_values = scalar_values[torch.isfinite(scalar_values)]
            metrics[f"opgd/{scalar_name}"] = (
                float(finite_values.mean()) if finite_values.numel() else float("nan")
            )
        scale_config = self._cfg("scale_opgd", {}) or {}
        batch, signal_metrics = build_opgd_final_advantages(
            batch,
            prompt_centering=bool(self._cfg("prompt_centering", False)),
            use_grpo_advantage=bool(self._cfg("use_grpo_advantage", False)),
            grpo_advantage_scale=float(self._cfg("grpo_advantage_scale", 1.0)),
            length_denormalization_opd=bool(
                self._cfg("length_denormalization_opd", False)
            ),
            length_denormalization_rl=bool(
                self._cfg("length_denormalization_rl", False)
            ),
            delta_top_fraction=float(self._cfg("delta_top_fraction", 0.0)),
            entropy_top_fraction=float(self._cfg("entropy_top_fraction", 0.0)),
            advantage_top_fraction=float(self._cfg("advantage_top_fraction", 0.0)),
            advantage_bottom_fraction=float(self._cfg("advantage_bottom_fraction", 0.0)),
            scale_opgd=bool(scale_config.get("enable", False)),
            scale_opgd_statistic=str(scale_config.get("statistic", "mean_abs")),
            scale_opgd_target=str(scale_config.get("target", "batch_mean")),
            scale_opgd_eps=float(scale_config.get("eps", 1.0e-6)),
            length_denormalization_max_tokens=int(
                self._cfg("length_denormalization_max_tokens")
            ),
            tokenizer=getattr(self, "tokenizer", None),
            l0_zero_tolerance=float(self._cfg("l0_zero_tolerance", 1e-6)),
            mask_uniform_reward_groups=bool(
                self._cfg("mask_uniform_reward_groups", False)
            ),
            uniform_reward_tolerance=float(
                self._cfg("uniform_reward_tolerance", _OPGD_UNIFORM_REWARD_TOLERANCE)
            ),
        )
        metrics.update(signal_metrics)
        if export_dir:
            self._export_opgd_batch(
                batch,
                timing_raw,
                base_advantages_for_export,
                signal_metrics,
            )
        batch.meta_info["opgd_enabled"] = True
        batch.meta_info["opgd_skip_optimizer"] = not bool(
            signal_metrics["opgd/update_applied"]
        )
        return batch


class OPGDTrainer(OPGDMixin, RayPPOTrainer):
    """The exact one-real-update OPGD trainer described in the method note."""
