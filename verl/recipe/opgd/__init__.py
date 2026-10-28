"""Reinforcement On-Policy Distillation recipe."""

from .controller import (
    VirtualTeacherControllerConfig,
    runtime_rescale,
    validate_teacher,
)
from .trainer import OPGDTrainer

__all__ = [
    "OPGDTrainer",
    "VirtualTeacherControllerConfig",
    "runtime_rescale",
    "validate_teacher",
]
