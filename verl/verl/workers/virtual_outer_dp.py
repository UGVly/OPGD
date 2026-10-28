"""Utilities for time-multiplexed virtual rollout outer-DP streams.

The virtual streams share one inference engine.  They differ only in their
persistent device RNG state and are executed sequentially over contiguous
batch chunks, matching the ordering used by the normal outer-DP dispatcher.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile
from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import TypeVar

import torch


BatchT = TypeVar("BatchT")
OutputT = TypeVar("OutputT")
RNGStateT = TypeVar("RNGStateT")


VIRTUAL_OUTER_DP_RNG_CHECKPOINT_VERSION = 1
_VIRTUAL_OUTER_DP_RNG_CHECKPOINT_FIELDS = frozenset(
    {
        "version",
        "global_step",
        "virtual_outer_dp_size",
        "world_size",
        "rank",
        "rollout_seed",
        "tp_size",
        "states",
    }
)
_GLOBAL_STEP_COMPONENT = re.compile(r"global_step_(\d+)")


def _validate_virtual_outer_dp_size(virtual_outer_dp_size: int) -> int:
    if isinstance(virtual_outer_dp_size, bool) or not isinstance(virtual_outer_dp_size, int):
        raise ValueError("virtual_outer_dp_size must be an integer, not bool or float.")
    if virtual_outer_dp_size < 1:
        raise ValueError("virtual_outer_dp_size must be positive.")
    return virtual_outer_dp_size


def validate_virtual_outer_dp_topology(
    *,
    virtual_outer_dp_size: int,
    rollout_name: str,
    rollout_mode: str,
    tensor_model_parallel_size: int,
    rollout_data_parallel_size: int,
    pipeline_model_parallel_size: int,
    actual_outer_dp_size: int | None = None,
) -> None:
    """Fail closed for the one supported virtual outer-DP topology."""

    virtual_outer_dp_size = _validate_virtual_outer_dp_size(virtual_outer_dp_size)
    if virtual_outer_dp_size == 1:
        return
    if virtual_outer_dp_size != 2:
        raise ValueError("virtual_outer_dp_size currently supports only 1 or 2.")
    if rollout_name != "vllm" or rollout_mode != "sync":
        raise ValueError("virtual outer-DP requires synchronous vLLM rollout.")
    if int(tensor_model_parallel_size) != 4:
        raise ValueError("virtual outer-DP requires tensor_model_parallel_size=4.")
    if int(rollout_data_parallel_size) != 1:
        raise ValueError("virtual outer-DP requires rollout.data_parallel_size=1.")
    if int(pipeline_model_parallel_size) != 1:
        raise ValueError("virtual outer-DP requires pipeline_model_parallel_size=1.")
    if actual_outer_dp_size is not None and int(actual_outer_dp_size) != 1:
        raise ValueError("virtual outer-DP requires actual rollout outer-DP size 1.")


def validate_virtual_outer_dp_sampling_params(sampling_params) -> None:
    """Require the vLLM sampler to consume the default CUDA RNG stream.

    A per-request seed creates private ``torch.Generator`` objects inside
    vLLM.  In that mode, swapping the ambient CUDA RNG state would silently
    stop isolating the virtual outer-DP streams.
    """

    if sampling_params is None or not hasattr(sampling_params, "seed"):
        raise ValueError("virtual outer-DP requires vLLM SamplingParams.seed to be inspectable.")
    if sampling_params.seed is not None:
        raise ValueError("virtual outer-DP requires SamplingParams.seed=None (default CUDA RNG).")


def split_virtual_outer_dp_batch(batch: BatchT, virtual_outer_dp_size: int) -> list[BatchT]:
    """Split a batch into equal contiguous chunks, like DataProto.chunk()."""

    virtual_outer_dp_size = _validate_virtual_outer_dp_size(virtual_outer_dp_size)
    if virtual_outer_dp_size == 1:
        return [batch]
    if len(batch) % virtual_outer_dp_size != 0:
        raise ValueError(
            f"rollout batch size {len(batch)} is not divisible by "
            f"virtual_outer_dp_size={virtual_outer_dp_size}."
        )
    return batch.chunk(chunks=virtual_outer_dp_size)


def _clone_rng_state(state: RNGStateT) -> RNGStateT:
    clone = getattr(state, "clone", None)
    return clone() if callable(clone) else deepcopy(state)


def clone_virtual_outer_dp_rng_states(
    initial_state: RNGStateT, virtual_outer_dp_size: int
) -> list[RNGStateT]:
    """Create independent initial RNG-state snapshots for virtual streams."""

    virtual_outer_dp_size = _validate_virtual_outer_dp_size(virtual_outer_dp_size)
    return [_clone_rng_state(initial_state) for _ in range(virtual_outer_dp_size)]


def virtual_outer_dp_rng_checkpoint_path(local_path: str | os.PathLike[str], rank: int) -> Path:
    """Return the rank-local virtual outer-DP RNG sidecar path."""

    rank = _validate_checkpoint_int("rank", rank, minimum=0)
    return Path(local_path) / f"virtual_outer_dp_rng_rank_{rank:05d}.pt"


def infer_checkpoint_global_step(local_path: str | os.PathLike[str]) -> int:
    """Infer ``global_step`` from a normal ``.../global_step_N/actor`` path."""

    for component in reversed(Path(local_path).parts):
        match = _GLOBAL_STEP_COMPONENT.fullmatch(component)
        if match is not None:
            return int(match.group(1))
    raise ValueError(f"checkpoint path does not contain a global_step_N component: {local_path}")


def _validate_checkpoint_int(name: str, value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"virtual outer-DP RNG checkpoint {name} must be an integer >= {minimum}.")
    return value


def _validated_cpu_rng_states(states: object, virtual_outer_dp_size: int) -> list[torch.Tensor]:
    if not isinstance(states, list):
        raise ValueError("virtual outer-DP RNG checkpoint states must be a list.")
    if len(states) != virtual_outer_dp_size:
        raise ValueError(
            "virtual outer-DP RNG checkpoint lane mismatch: "
            f"expected {virtual_outer_dp_size} states, found {len(states)}."
        )

    validated = []
    expected_shape = None
    for lane, state in enumerate(states):
        if not isinstance(state, torch.Tensor):
            raise ValueError(f"virtual outer-DP RNG checkpoint lane {lane} is not a tensor.")
        if state.device.type != "cpu" or state.dtype != torch.uint8:
            raise ValueError(
                f"virtual outer-DP RNG checkpoint lane {lane} must be a CPU uint8 tensor."
            )
        if state.ndim != 1 or state.numel() == 0:
            raise ValueError(
                f"virtual outer-DP RNG checkpoint lane {lane} must be a non-empty 1-D tensor."
            )
        if expected_shape is None:
            expected_shape = state.shape
        elif state.shape != expected_shape:
            raise ValueError(
                "virtual outer-DP RNG checkpoint lane mismatch: all state tensors must have the same shape."
            )
        validated.append(state.detach().contiguous().clone())
    return validated


def _build_virtual_outer_dp_rng_checkpoint(
    *,
    global_step: int,
    virtual_outer_dp_size: int,
    world_size: int,
    rank: int,
    rollout_seed: int,
    tp_size: int,
    states: object,
) -> dict[str, object]:
    global_step = _validate_checkpoint_int("global_step", global_step)
    virtual_outer_dp_size = _validate_checkpoint_int(
        "virtual_outer_dp_size", virtual_outer_dp_size, minimum=2
    )
    world_size = _validate_checkpoint_int("world_size", world_size, minimum=1)
    rank = _validate_checkpoint_int("rank", rank)
    rollout_seed = _validate_checkpoint_int("rollout_seed", rollout_seed)
    tp_size = _validate_checkpoint_int("tp_size", tp_size, minimum=1)
    if rank >= world_size:
        raise ValueError(
            f"virtual outer-DP RNG checkpoint rank {rank} is outside world_size={world_size}."
        )
    return {
        "version": VIRTUAL_OUTER_DP_RNG_CHECKPOINT_VERSION,
        "global_step": global_step,
        "virtual_outer_dp_size": virtual_outer_dp_size,
        "world_size": world_size,
        "rank": rank,
        "rollout_seed": rollout_seed,
        "tp_size": tp_size,
        "states": _validated_cpu_rng_states(states, virtual_outer_dp_size),
    }


def save_virtual_outer_dp_rng_checkpoint(
    local_path: str | os.PathLike[str],
    *,
    global_step: int,
    virtual_outer_dp_size: int,
    world_size: int,
    rank: int,
    rollout_seed: int,
    tp_size: int,
    states: object,
) -> Path:
    """Atomically save one rank's persistent virtual rollout RNG lanes."""

    payload = _build_virtual_outer_dp_rng_checkpoint(
        global_step=global_step,
        virtual_outer_dp_size=virtual_outer_dp_size,
        world_size=world_size,
        rank=rank,
        rollout_seed=rollout_seed,
        tp_size=tp_size,
        states=states,
    )
    checkpoint_dir = Path(local_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_path = virtual_outer_dp_rng_checkpoint_path(checkpoint_dir, rank)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=checkpoint_dir, prefix=f".{final_path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(file_descriptor, "wb") as temporary_file:
            torch.save(payload, temporary_file)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_name, final_path)
    except BaseException:
        try:
            os.close(file_descriptor)
        except OSError:
            pass
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return final_path


def load_virtual_outer_dp_rng_checkpoint(
    local_path: str | os.PathLike[str],
    *,
    expected_global_step: int,
    expected_virtual_outer_dp_size: int,
    expected_world_size: int,
    expected_rank: int,
    expected_rollout_seed: int,
    expected_tp_size: int,
) -> list[torch.Tensor]:
    """Strictly validate and return detached CPU lane states from one sidecar."""

    expected = _build_virtual_outer_dp_rng_checkpoint(
        global_step=expected_global_step,
        virtual_outer_dp_size=expected_virtual_outer_dp_size,
        world_size=expected_world_size,
        rank=expected_rank,
        rollout_seed=expected_rollout_seed,
        tp_size=expected_tp_size,
        states=[torch.zeros(1, dtype=torch.uint8) for _ in range(expected_virtual_outer_dp_size)],
    )
    checkpoint_path = virtual_outer_dp_rng_checkpoint_path(local_path, expected_rank)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing virtual outer-DP RNG checkpoint sidecar: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"failed to load virtual outer-DP RNG checkpoint sidecar {checkpoint_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("virtual outer-DP RNG checkpoint payload must be a dict.")
    payload_fields = frozenset(payload)
    if payload_fields != _VIRTUAL_OUTER_DP_RNG_CHECKPOINT_FIELDS:
        missing = sorted(_VIRTUAL_OUTER_DP_RNG_CHECKPOINT_FIELDS - payload_fields)
        unexpected = sorted(payload_fields - _VIRTUAL_OUTER_DP_RNG_CHECKPOINT_FIELDS)
        raise ValueError(
            "virtual outer-DP RNG checkpoint schema mismatch: "
            f"missing={missing}, unexpected={unexpected}."
        )
    _validate_checkpoint_int("version", payload["version"], minimum=1)
    if payload["version"] != VIRTUAL_OUTER_DP_RNG_CHECKPOINT_VERSION:
        raise ValueError(
            "virtual outer-DP RNG checkpoint version mismatch: "
            f"expected {VIRTUAL_OUTER_DP_RNG_CHECKPOINT_VERSION}, found {payload['version']!r}."
        )
    for field in (
        "global_step",
        "virtual_outer_dp_size",
        "world_size",
        "rank",
        "rollout_seed",
        "tp_size",
    ):
        _validate_checkpoint_int(field, payload[field], minimum=2 if field == "virtual_outer_dp_size" else 0)
        if payload[field] != expected[field]:
            raise ValueError(
                f"virtual outer-DP RNG checkpoint {field} mismatch: "
                f"expected {expected[field]}, found {payload[field]}."
            )
    return _validated_cpu_rng_states(payload["states"], expected_virtual_outer_dp_size)


def run_virtual_outer_dp_streams(
    chunks: Sequence[BatchT],
    rng_states: list[RNGStateT],
    *,
    set_rng_state: Callable[[RNGStateT], None],
    get_rng_state: Callable[[], RNGStateT],
    generate: Callable[[BatchT], OutputT],
    between_streams: Callable[[], None] | None = None,
) -> list[OutputT]:
    """Generate transactionally, then commit independently advanced RNGs."""

    if len(chunks) != len(rng_states):
        raise ValueError(
            f"received {len(chunks)} chunks for {len(rng_states)} virtual RNG streams."
        )

    ambient_rng_state = _clone_rng_state(get_rng_state())
    pending_rng_states = [_clone_rng_state(state) for state in rng_states]
    outputs = []
    try:
        for stream_index, chunk in enumerate(chunks):
            if stream_index > 0 and between_streams is not None:
                between_streams()
            set_rng_state(pending_rng_states[stream_index])
            outputs.append(generate(chunk))
            pending_rng_states[stream_index] = _clone_rng_state(get_rng_state())
    finally:
        set_rng_state(ambient_rng_state)

    rng_states[:] = pending_rng_states
    return outputs
