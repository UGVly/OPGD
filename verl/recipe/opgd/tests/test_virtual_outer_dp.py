"""Contracts for the retained virtual rollout outer-DP utility."""

from types import SimpleNamespace

import pytest
import torch

from verl.workers.virtual_outer_dp import (
    clone_virtual_outer_dp_rng_states,
    load_virtual_outer_dp_rng_checkpoint,
    run_virtual_outer_dp_streams,
    save_virtual_outer_dp_rng_checkpoint,
    split_virtual_outer_dp_batch,
    validate_virtual_outer_dp_sampling_params,
    validate_virtual_outer_dp_topology,
)


class Chunkable(list):
    def chunk(self, chunks: int):
        size = len(self) // chunks
        return [self[index * size : (index + 1) * size] for index in range(chunks)]


def test_topology_and_sampler_are_fail_closed():
    validate_virtual_outer_dp_topology(
        virtual_outer_dp_size=2,
        rollout_name="vllm",
        rollout_mode="sync",
        tensor_model_parallel_size=4,
        rollout_data_parallel_size=1,
        pipeline_model_parallel_size=1,
        actual_outer_dp_size=1,
    )
    with pytest.raises(ValueError, match="only 1 or 2"):
        validate_virtual_outer_dp_topology(
            virtual_outer_dp_size=3,
            rollout_name="vllm",
            rollout_mode="sync",
            tensor_model_parallel_size=4,
            rollout_data_parallel_size=1,
            pipeline_model_parallel_size=1,
        )
    validate_virtual_outer_dp_sampling_params(SimpleNamespace(seed=None))
    with pytest.raises(ValueError, match="seed=None"):
        validate_virtual_outer_dp_sampling_params(SimpleNamespace(seed=42))


def test_split_is_contiguous_and_rejects_uneven_batches():
    batch = Chunkable(range(8))
    assert split_virtual_outer_dp_batch(batch, 2) == [list(range(4)), list(range(4, 8))]
    with pytest.raises(ValueError, match="not divisible"):
        split_virtual_outer_dp_batch(Chunkable(range(7)), 2)


def test_rng_lanes_advance_independently_and_restore_ambient_state():
    ambient = torch.tensor([99])
    current = ambient.clone()
    states = clone_virtual_outer_dp_rng_states(torch.tensor([10]), 2)

    def set_state(state):
        nonlocal current
        current = state.clone()

    def get_state():
        return current.clone()

    def generate(increment):
        nonlocal current
        current += increment
        return int(current.item())

    outputs = run_virtual_outer_dp_streams(
        [1, 3],
        states,
        set_rng_state=set_state,
        get_rng_state=get_state,
        generate=generate,
    )
    assert outputs == [11, 13]
    assert [int(state.item()) for state in states] == [11, 13]
    assert torch.equal(current, ambient)


def test_rng_checkpoint_round_trip(tmp_path):
    states = [torch.tensor([1, 2], dtype=torch.uint8), torch.tensor([3, 4], dtype=torch.uint8)]
    kwargs = {
        "global_step": 10,
        "virtual_outer_dp_size": 2,
        "world_size": 4,
        "rank": 0,
        "rollout_seed": 42,
        "tp_size": 4,
    }
    save_virtual_outer_dp_rng_checkpoint(tmp_path, states=states, **kwargs)
    loaded = load_virtual_outer_dp_rng_checkpoint(
        tmp_path,
        **{f"expected_{name}": value for name, value in kwargs.items()},
    )
    assert all(torch.equal(actual, expected) for actual, expected in zip(loaded, states, strict=True))
