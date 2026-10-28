"""Pure control logic for the OPGD one-expert virtual teacher."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VirtualTeacherControllerConfig:
    """Validated bounds for the one-pass virtual-teacher controller."""

    adaptive_virtual_scale: bool
    target_logratio_abs_mean: float
    target_logratio_low: float
    target_logratio_high: float
    max_logratio_abs_mean: float
    min_runtime_scale_factor: float
    max_runtime_scale_factor: float
    max_rescale_passes: int
    verify_restore: bool

    def __post_init__(self) -> None:
        positive = {
            "target_logratio_abs_mean": self.target_logratio_abs_mean,
            "target_logratio_low": self.target_logratio_low,
            "target_logratio_high": self.target_logratio_high,
            "max_logratio_abs_mean": self.max_logratio_abs_mean,
            "min_runtime_scale_factor": self.min_runtime_scale_factor,
            "max_runtime_scale_factor": self.max_runtime_scale_factor,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not (
            self.target_logratio_low
            <= self.target_logratio_abs_mean
            <= self.target_logratio_high
            <= self.max_logratio_abs_mean
        ):
            raise ValueError(
                "log-ratio bounds must satisfy low <= target <= high <= hard max"
            )
        if self.min_runtime_scale_factor > self.max_runtime_scale_factor:
            raise ValueError("min runtime scale factor must not exceed max")
        if self.max_rescale_passes not in {0, 1}:
            raise ValueError("OPGD permits zero or one runtime rescale pass")


@dataclass(frozen=True, slots=True)
class RuntimeScaleDecision:
    applied_step_l2: float
    scale_factor: float
    rescale_required: bool
    reason: str


@dataclass(frozen=True, slots=True)
class TeacherValidity:
    valid: bool
    reason: str


def runtime_rescale(
    step_l2: float,
    observed_logratio_abs_mean: float,
    config: VirtualTeacherControllerConfig,
) -> RuntimeScaleDecision:
    """Return the optional second-pass normalized-SGD step length."""

    if not math.isfinite(step_l2) or step_l2 <= 0.0:
        raise ValueError("step_l2 must be finite and positive")
    if (
        not math.isfinite(observed_logratio_abs_mean)
        or observed_logratio_abs_mean <= 0.0
    ):
        raise ValueError("observed_logratio_abs_mean must be finite and positive")
    in_band = (
        config.target_logratio_low
        <= observed_logratio_abs_mean
        <= config.target_logratio_high
    )
    if not config.adaptive_virtual_scale:
        return RuntimeScaleDecision(step_l2, 1.0, False, "adaptive_scaling_disabled")
    if config.max_rescale_passes == 0:
        return RuntimeScaleDecision(step_l2, 1.0, False, "rescale_passes_disabled")
    if in_band:
        return RuntimeScaleDecision(step_l2, 1.0, False, "within_target_band")

    factor = config.target_logratio_abs_mean / observed_logratio_abs_mean
    factor = min(
        config.max_runtime_scale_factor, max(config.min_runtime_scale_factor, factor)
    )
    reason = (
        "below_target_band"
        if observed_logratio_abs_mean < config.target_logratio_low
        else "above_target_band"
    )
    return RuntimeScaleDecision(step_l2 * factor, factor, True, reason)


def validate_teacher(
    observed_logratio_abs_mean: float,
    *,
    finite_nonzero_gradient: bool,
    restore_max_abs: float,
    config: VirtualTeacherControllerConfig,
) -> TeacherValidity:
    """Apply the hard safety gate before a teacher shift can update the actor."""

    if not finite_nonzero_gradient:
        reason = "invalid_gradient"
    elif (
        not math.isfinite(observed_logratio_abs_mean)
        or observed_logratio_abs_mean <= 0.0
    ):
        reason = "invalid_logratio"
    elif observed_logratio_abs_mean > config.max_logratio_abs_mean:
        reason = "hard_logratio_limit_exceeded"
    elif not math.isfinite(restore_max_abs) or restore_max_abs < 0.0:
        reason = "invalid_restore_measurement"
    elif config.verify_restore and restore_max_abs != 0.0:
        reason = "restore_not_exact"
    else:
        reason = "valid"
    return TeacherValidity(valid=reason == "valid", reason=reason)
