# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import math
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import (
    agg_loss,
    compute_entropy_token_weights,
    get_policy_loss_fn,
    kl_penalty,
)
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _length_denormalize_advantages(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    enabled: bool,
    max_tokens: int,
) -> torch.Tensor:
    """Optionally undo per-sequence token averaging by multiplying by T/T_max."""

    if max_tokens <= 0:
        raise ValueError("length_denormalization_max_tokens must be positive.")
    if advantages.shape != response_mask.shape:
        raise ValueError("advantages and response_mask must have the same shape.")
    if not enabled:
        return advantages
    response_lengths = response_mask.bool().sum(dim=-1).to(advantages.dtype)
    scale = response_lengths / float(max_tokens)
    return advantages * scale.unsqueeze(-1)


class _CorrelationAccumulator:
    """Streaming correlation and least-squares slope over paired quantities.

    Sums are accumulated instead of per-micro-batch correlations because GRPO gives a
    response one advantage shared by all of its tokens: inside a single micro-batch the
    advantage can be near-constant, so the correlation is undefined there and averaging
    those entries would drown the real signal.
    """

    # Summing squares in float64 over float32 inputs leaves a variance residual around 1e-7 of
    # the mean square. Comparing against a relative floor keeps that noise from being read as a
    # real spread and divided into a meaningless ratio.
    _RELATIVE_VARIANCE_FLOOR = 1e-6

    def __init__(self) -> None:
        self._n = 0.0
        self._sx = 0.0
        self._sy = 0.0
        self._sxx = 0.0
        self._syy = 0.0
        self._sxy = 0.0

    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        x = x.detach().float().reshape(-1)
        y = y.detach().float().reshape(-1)
        if x.numel() == 0 or x.numel() != y.numel():
            return
        self._n += float(x.numel())
        self._sx += float(x.sum())
        self._sy += float(y.sum())
        self._sxx += float((x * x).sum())
        self._syy += float((y * y).sum())
        self._sxy += float((x * y).sum())

    def _centred(self) -> tuple[float, float, float, float, float] | None:
        if self._n <= 1:
            return None
        mean_x = self._sx / self._n
        mean_y = self._sy / self._n
        mean_square_x = self._sxx / self._n
        mean_square_y = self._syy / self._n
        var_x = mean_square_x - mean_x * mean_x
        var_y = mean_square_y - mean_y * mean_y
        cov = self._sxy / self._n - mean_x * mean_y
        return var_x, var_y, cov, mean_square_x, mean_square_y

    def corr(self) -> float:
        """Pearson correlation, or 0.0 when either side has no usable variance."""
        centred = self._centred()
        if centred is None:
            return 0.0
        var_x, var_y, cov, mean_square_x, mean_square_y = centred
        floor = self._RELATIVE_VARIANCE_FLOOR
        if var_x <= floor * mean_square_x or var_y <= floor * mean_square_y:
            return 0.0
        return cov / math.sqrt(var_x * var_y)

    def slope(self) -> float:
        """Least-squares slope of y on x: how far y moves per unit of x."""
        centred = self._centred()
        if centred is None:
            return 0.0
        var_x, var_y, cov, mean_square_x, mean_square_y = centred
        floor = self._RELATIVE_VARIANCE_FLOOR
        # A flat y has a flat least-squares fit, and its residual covariance over a genuinely
        # varying x would otherwise come out as a small but nonzero slope.
        if var_x <= floor * mean_square_x or var_y <= floor * mean_square_y:
            return 0.0
        return cov / var_x


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            # Reset model inputs when the reference uses a different tokenization.
            if "ref_input_ids" in micro_batch.keys():
                input_ids = micro_batch["ref_input_ids"]
                attention_mask = micro_batch["ref_attention_mask"]
                position_ids = micro_batch["ref_position_ids"]
                batch_size, seqlen = input_ids.shape

            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        optimizer_stepped = True
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
                optimizer_stepped = False
            else:
                self.actor_optimizer.step()
        return grad_norm, optimizer_stepped

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        # The reference may use a different tokenization from the actor.
        has_ref_input_ids = "ref_input_ids" in data.batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if has_ref_input_ids:
            select_keys.extend(["ref_input_ids", "ref_attention_mask", "ref_position_ids"])
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @staticmethod
    def _opgd_tensor_local(tensor: torch.Tensor) -> torch.Tensor:
        """Return the local shard used for exact OPGD diagnostics."""

        return tensor.to_local() if isinstance(tensor, DTensor) else tensor

    @staticmethod
    def _opgd_replication_factor(tensor: torch.Tensor) -> int:
        """Return DTensor replication multiplicity so global norms are counted once."""

        if not isinstance(tensor, DTensor):
            return 1
        factor = 1
        mesh = tensor.device_mesh.mesh
        for mesh_dim, placement in enumerate(tensor.placements):
            if placement.is_replicate():
                factor *= int(mesh.shape[mesh_dim])
        return factor

    def _opgd_global_l2(self, tensors: list[torch.Tensor | None]) -> torch.Tensor:
        squared = torch.zeros((), dtype=torch.float64, device=get_device_id())
        for tensor in tensors:
            if tensor is None:
                continue
            local = self._opgd_tensor_local(tensor.detach()).double()
            squared += local.square().sum() / self._opgd_replication_factor(tensor)
        torch.distributed.all_reduce(squared, op=torch.distributed.ReduceOp.SUM)
        return squared.sqrt()

    def _compute_opgd_crossfit_teacher_log_prob(self, data: DataProto):
        """Build one leave-one-fold-out virtual teacher for each OPGD expert ID."""

        from recipe.opgd.controller import (
            VirtualTeacherControllerConfig,
            runtime_rescale,
            validate_teacher,
        )

        cfg = data.meta_info["opgd"]
        expert_count = int(cfg["expert_count"])
        expert_ids = data.batch["opgd_expert_id"].long()
        observed_expert_ids = set(expert_ids.detach().cpu().tolist())
        expected_expert_ids = set(range(expert_count))
        if observed_expert_ids != expected_expert_ids:
            raise ValueError(
                "Every actor rank must receive every OPGD expert fold; "
                f"observed {sorted(observed_expert_ids)}, expected "
                f"{sorted(expected_expert_ids)}."
            )

        controller_cfg = VirtualTeacherControllerConfig(
            adaptive_virtual_scale=bool(cfg["adaptive_virtual_scale"]),
            target_logratio_abs_mean=float(cfg["target_logratio_abs_mean"]),
            target_logratio_low=float(cfg["target_logratio_low"]),
            target_logratio_high=float(cfg["target_logratio_high"]),
            max_logratio_abs_mean=float(cfg["max_logratio_abs_mean"]),
            min_runtime_scale_factor=float(cfg["min_runtime_scale_factor"]),
            max_runtime_scale_factor=float(cfg["max_runtime_scale_factor"]),
            max_rescale_passes=int(cfg["max_rescale_passes"]),
            verify_restore=bool(cfg["verify_restore"]),
        )
        virtual_update = str(cfg.get("virtual_update", "normalized_sgd"))
        if virtual_update not in {"normalized_sgd", "sgd"}:
            raise ValueError("OPGD virtual_update must be normalized_sgd or sgd.")
        length_denormalization_rl = bool(cfg["length_denormalization_rl"])
        length_denormalization_max_tokens = int(
            cfg["length_denormalization_max_tokens"]
        )

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
            "opgd_expert_id",
        ]
        if "rollout_is_weights" in data.batch:
            select_keys.append("rollout_is_weights")
        non_tensor_keys = (
            ["multi_modal_inputs"]
            if "multi_modal_inputs" in data.non_tensor_batch
            else []
        )
        selected_batch = data.select(
            batch_keys=select_keys, non_tensor_batch_keys=non_tensor_keys
        )
        trainable_parameters = [
            parameter
            for parameter in self.actor_module.parameters()
            if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise ValueError("OPGD found no trainable actor parameters.")

        student_parameter_snapshots = [
            parameter.detach().clone() for parameter in trainable_parameters
        ]
        snapshot_local_bytes = sum(
            self._opgd_tensor_local(snapshot).numel()
            * self._opgd_tensor_local(snapshot).element_size()
            for snapshot in student_parameter_snapshots
        )
        student_log_probs = selected_batch.batch["old_log_probs"]
        aligned_teacher_log_probs = student_log_probs.detach().clone()

        def restore_student_parameters() -> None:
            with torch.no_grad():
                for parameter, snapshot in zip(
                    trainable_parameters, student_parameter_snapshots, strict=True
                ):
                    parameter.copy_(snapshot)

        expert_metrics = []
        all_sequences = torch.ones_like(expert_ids, dtype=torch.bool)
        for target_expert_id in range(expert_count):
            target_sequence_mask = expert_ids == target_expert_id
            source_sequence_mask = (
                all_sequences if expert_count == 1 else ~target_sequence_mask
            )
            grpo_source_batch = selected_batch[source_sequence_mask]
            teacher_target_batch = selected_batch[target_sequence_mask]
            target_student_log_probs = teacher_target_batch.batch["old_log_probs"]
            target_response_mask = teacher_target_batch.batch["response_mask"].bool()

            virtual_gradient_l2_value = float("nan")
            finite_virtual_gradient = False
            applied_virtual_scale = 0.0
            virtual_parameter_delta_l2 = 0.0
            rescale_passes = 0
            logratio_delta_abs_mean = 0.0
            logratio_delta_mean = 0.0
            logratio_delta_var = 0.0
            logratio_delta_std = 0.0
            teacher_log_probs_for_target = (
                target_student_log_probs.to(get_device_id()).detach().clone()
            )

            try:
                restore_student_parameters()
                self.actor_module.train()
                self.actor_optimizer.zero_grad(set_to_none=True)
                source_micro_batches = grpo_source_batch.split(
                    self.config.ppo_micro_batch_size_per_gpu
                )
                if not source_micro_batches:
                    raise ValueError("A OPGD expert received no GRPO source batch.")
                policy_loss_fn = get_policy_loss_fn(
                    self.config.policy_loss.get("loss_mode", "vanilla")
                )
                for source_micro_batch in source_micro_batches:
                    source_micro_batch = source_micro_batch.to(get_device_id())
                    model_inputs = {
                        **source_micro_batch.batch,
                        **source_micro_batch.non_tensor_batch,
                    }
                    _, current_log_probs = self._forward_micro_batch(
                        model_inputs,
                        temperature=data.meta_info["temperature"],
                        calculate_entropy=False,
                    )
                    virtual_advantages = _length_denormalize_advantages(
                        model_inputs["advantages"],
                        model_inputs["response_mask"],
                        enabled=length_denormalization_rl,
                        max_tokens=length_denormalization_max_tokens,
                    )
                    virtual_loss, _ = policy_loss_fn(
                        old_log_prob=model_inputs["old_log_probs"],
                        log_prob=current_log_probs,
                        advantages=virtual_advantages,
                        response_mask=model_inputs["response_mask"],
                        loss_agg_mode=self.config.loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=model_inputs.get("rollout_is_weights"),
                    )
                    (virtual_loss / len(source_micro_batches)).backward()

                virtual_gradients = [
                    None
                    if parameter.grad is None
                    else parameter.grad.detach().clone()
                    for parameter in trainable_parameters
                ]
                virtual_gradient_l2 = self._opgd_global_l2(virtual_gradients)
                virtual_gradient_l2_value = float(
                    virtual_gradient_l2.detach().cpu()
                )
                finite_virtual_gradient = bool(
                    torch.isfinite(virtual_gradient_l2).item()
                ) and virtual_gradient_l2_value > 0.0

                def apply_virtual_update(
                    virtual_scale: float,
                    *,
                    gradient_l2_value: float = virtual_gradient_l2_value,
                    gradients: list[torch.Tensor | None] = virtual_gradients,
                ) -> float:
                    restore_student_parameters()
                    if virtual_update == "normalized_sgd":
                        coefficient = virtual_scale / max(
                            gradient_l2_value,
                            torch.finfo(torch.float32).eps,
                        )
                    else:
                        coefficient = virtual_scale
                    with torch.no_grad():
                        for parameter, virtual_gradient in zip(
                            trainable_parameters, gradients, strict=True
                        ):
                            if virtual_gradient is not None:
                                parameter.add_(virtual_gradient, alpha=-coefficient)
                    parameter_delta_l2 = self._opgd_global_l2(
                        [
                            parameter.detach() - snapshot
                            for parameter, snapshot in zip(
                                trainable_parameters,
                                student_parameter_snapshots,
                                strict=True,
                            )
                        ]
                    )
                    return float(parameter_delta_l2.cpu())

                def forward_virtual_teacher(
                    *,
                    target_batch: DataProto = teacher_target_batch,
                    response_mask: torch.Tensor = target_response_mask,
                    student_log_probs: torch.Tensor = target_student_log_probs,
                ) -> tuple[torch.Tensor, float]:
                    target_batch.meta_info["micro_batch_size"] = (
                        self.config.ppo_micro_batch_size_per_gpu
                    )
                    target_batch.meta_info["temperature"] = data.meta_info[
                        "temperature"
                    ]
                    target_batch.meta_info["use_dynamic_bsz"] = False
                    teacher_log_probs, _ = self.compute_log_prob(
                        target_batch, calculate_entropy=False
                    )
                    valid_mask = response_mask.to(teacher_log_probs.device)
                    student_values = student_log_probs.to(
                        teacher_log_probs.device
                    )
                    valid_delta = (teacher_log_probs - student_values)[valid_mask].double()
                    absolute_delta_sum = valid_delta.abs().sum()
                    delta_sum = valid_delta.sum()
                    delta_squared_sum = valid_delta.square().sum()
                    valid_token_count = valid_mask.sum().double()
                    torch.distributed.all_reduce(
                        absolute_delta_sum, op=torch.distributed.ReduceOp.SUM
                    )
                    torch.distributed.all_reduce(
                        delta_sum, op=torch.distributed.ReduceOp.SUM
                    )
                    torch.distributed.all_reduce(
                        delta_squared_sum, op=torch.distributed.ReduceOp.SUM
                    )
                    torch.distributed.all_reduce(
                        valid_token_count, op=torch.distributed.ReduceOp.SUM
                    )
                    count = valid_token_count.clamp_min(1.0)
                    delta_mean = delta_sum / count
                    delta_variance = (
                        delta_squared_sum / count - delta_mean.square()
                    ).clamp_min(0.0)
                    delta_abs_mean = absolute_delta_sum / valid_token_count.clamp_min(
                        1.0
                    )
                    return (
                        teacher_log_probs.detach(),
                        float(delta_abs_mean.cpu()),
                        float(delta_mean.cpu()),
                        float(delta_variance.cpu()),
                        float(delta_variance.sqrt().cpu()),
                    )

                if finite_virtual_gradient:
                    applied_virtual_scale = float(
                        cfg["virtual_step_l2"]
                        if virtual_update == "normalized_sgd"
                        else cfg["virtual_sgd_learning_rate"]
                    )
                    virtual_parameter_delta_l2 = apply_virtual_update(
                        applied_virtual_scale
                    )
                    (
                        teacher_log_probs_for_target,
                        logratio_delta_abs_mean,
                        logratio_delta_mean,
                        logratio_delta_var,
                        logratio_delta_std,
                    ) = forward_virtual_teacher()
                    if (
                        math.isfinite(logratio_delta_abs_mean)
                        and logratio_delta_abs_mean > 0.0
                        and not controller_cfg.target_logratio_low
                        <= logratio_delta_abs_mean
                        <= controller_cfg.target_logratio_high
                    ):
                        scale_decision = runtime_rescale(
                            applied_virtual_scale,
                            logratio_delta_abs_mean,
                            controller_cfg,
                        )
                        if scale_decision.rescale_required:
                            applied_virtual_scale = scale_decision.applied_step_l2
                            rescale_passes = 1
                            virtual_parameter_delta_l2 = apply_virtual_update(
                                applied_virtual_scale
                            )
                            (
                                teacher_log_probs_for_target,
                                logratio_delta_abs_mean,
                                logratio_delta_mean,
                                logratio_delta_var,
                                logratio_delta_std,
                            ) = forward_virtual_teacher()
            finally:
                restore_student_parameters()
                self.actor_optimizer.zero_grad(set_to_none=True)

            restore_differences = []
            for parameter, snapshot in zip(
                trainable_parameters, student_parameter_snapshots, strict=True
            ):
                local_parameter = self._opgd_tensor_local(parameter.detach())
                local_snapshot = self._opgd_tensor_local(snapshot.detach())
                if local_parameter.numel():
                    restore_differences.append(
                        (local_parameter - local_snapshot).abs().max()
                    )
            restore_max_abs = (
                torch.stack(restore_differences).max()
                if restore_differences
                else torch.zeros((), device=get_device_id())
            )
            torch.distributed.all_reduce(
                restore_max_abs, op=torch.distributed.ReduceOp.MAX
            )
            restore_max_abs_value = float(restore_max_abs.cpu())
            teacher_valid = float(
                validate_teacher(
                    logratio_delta_abs_mean,
                    finite_nonzero_gradient=finite_virtual_gradient,
                    restore_max_abs=restore_max_abs_value,
                    config=controller_cfg,
                ).valid
            )
            aligned_teacher_log_probs[target_sequence_mask] = (
                teacher_log_probs_for_target.to(aligned_teacher_log_probs.device)
            )
            expert_metrics.append(
                {
                    "virtual_grad_norm": virtual_gradient_l2_value,
                    "virtual_update_scale": applied_virtual_scale,
                    "virtual_weight_delta_l2": virtual_parameter_delta_l2,
                    "teacher_logratio_abs_mean": logratio_delta_abs_mean,
                    "teacher_logratio_delta_mean": logratio_delta_mean,
                    "teacher_logratio_delta_var": logratio_delta_var,
                    "teacher_logratio_delta_std": logratio_delta_std,
                    "rescale_passes": float(rescale_passes),
                    "restore_max_abs": restore_max_abs_value,
                    "teacher_valid": teacher_valid,
                }
            )

        def expert_mean(metric_name: str) -> float:
            return sum(float(item[metric_name]) for item in expert_metrics) / len(
                expert_metrics
            )

        metrics = {
            "virtual_grad_norm": expert_mean("virtual_grad_norm"),
            "virtual_update_scale": expert_mean("virtual_update_scale"),
            "virtual_weight_delta_l2": expert_mean("virtual_weight_delta_l2"),
            "teacher_logratio_abs_mean": expert_mean(
                "teacher_logratio_abs_mean"
            ),
            "teacher_logratio_delta_mean": expert_mean(
                "teacher_logratio_delta_mean"
            ),
            "teacher_logratio_delta_var": expert_mean(
                "teacher_logratio_delta_var"
            ),
            "teacher_logratio_delta_std": expert_mean(
                "teacher_logratio_delta_std"
            ),
            "rescale_passes": sum(
                item["rescale_passes"] for item in expert_metrics
            ),
            "restore_max_abs": max(
                item["restore_max_abs"] for item in expert_metrics
            ),
            "teacher_valid": min(item["teacher_valid"] for item in expert_metrics),
            "snapshot_local_mib": snapshot_local_bytes / (1024**2),
        }
        return aligned_teacher_log_probs, metrics

    def _compute_opgd_teacher_log_prob(self, data: DataProto):
        """Build the configured in-sample or cross-fit virtual teacher."""

        if self.actor_optimizer is None:
            raise ValueError("OPGD virtual teacher requires an actor optimizer.")
        if self.scaler is not None:
            raise ValueError(
                "OPGD supports bf16/fp32 actors only, not an fp16 GradScaler."
            )
        if self.config.ppo_epochs != 1:
            raise ValueError("OPGD requires actor.ppo_epochs=1.")
        if self.config.use_dynamic_bsz:
            raise ValueError("OPGD requires actor.use_dynamic_bsz=false.")
        if "opgd_expert_id" not in data.batch:
            raise ValueError("OPGD batch is missing opgd_expert_id.")
        return self._compute_opgd_crossfit_teacher_log_prob(data)

    def compute_opgd_teacher_log_prob(self, data: DataProto):
        """Public actor method used by the OPGD worker RPC."""

        return self._compute_opgd_teacher_log_prob(data)

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        opgd_enabled = bool(data.meta_info.get("opgd_enabled", False))
        opgd_skip_optimizer = bool(opgd_enabled and data.meta_info.get("opgd_skip_optimizer", False))
        if opgd_skip_optimizer:
            self.actor_optimizer.zero_grad(set_to_none=True)
            return {
                "actor/grad_norm": [0.0],
                "actor/pg_loss": [0.0],
                "actor/opgd_optimizer_steps": [0.0],
                "actor/opgd_optimizer_skipped": [1.0],
            }

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if opgd_enabled:
            if "opgd_update_mask" not in data.batch:
                raise ValueError("OPGD actor update is missing opgd_update_mask.")
            select_keys.append("opgd_update_mask")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        optimizer_steps = 0
        # Pooled across every micro-batch of the update, so these stay comparable between runs
        # regardless of how dynamic batching happened to cut the batch.
        entropy_token_weight_mode = str(self.config.policy_loss.get("entropy_token_weight_mode", "none"))
        token_adv_entropy = _CorrelationAccumulator()
        token_abs_adv_entropy = _CorrelationAccumulator()
        resp_adv_entropy = _CorrelationAccumulator()
        resp_abs_adv_entropy = _CorrelationAccumulator()
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    # The real OPGD update mask can zero out entire responses, so keep the
                    # true response mask for diagnostics that require >=1 token per micro-batch.
                    token_response_mask = model_inputs["response_mask"]
                    response_mask = model_inputs.get("opgd_update_mask", model_inputs["response_mask"])
                    pg_response_mask = response_mask
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    entropy_top_fraction = float(self.config.policy_loss.get("entropy_token_top_fraction", 0.0))
                    if entropy_coeff != 0 or entropy_top_fraction > 0 or entropy_token_weight_mode != "none":
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    if entropy_top_fraction > 0:
                        if not (0.0 < entropy_top_fraction <= 1.0):
                            raise ValueError("entropy_token_top_fraction must be in (0, 1]")
                        valid_counts = response_mask.sum(dim=1).to(torch.long)
                        keep_counts = (
                            torch.ceil(valid_counts.float() * entropy_top_fraction)
                            .to(torch.long)
                            .clamp_min(1)
                        )
                        rank = torch.argsort(
                            torch.argsort(
                                entropy.masked_fill(response_mask <= 0, -torch.inf),
                                dim=1,
                                descending=True,
                            ),
                            dim=1,
                        )
                        pg_response_mask = (rank < keep_counts.unsqueeze(1)).to(response_mask.dtype) * response_mask
                        micro_batch_metrics["actor/entropy_token_fraction"] = (
                            pg_response_mask.sum() / response_mask.sum().clamp_min(1)
                        ).detach().item()

                    if entropy_token_weight_mode != "none":
                        # The pooled diagnostics below need the advantages as GRPO produced them, so keep
                        # an unweighted handle before the weighting overwrites the local name.
                        unweighted_advantages = advantages
                        valid_mask = response_mask.to(torch.bool)
                        entropy_weights = compute_entropy_token_weights(
                            entropy=entropy,
                            response_mask=response_mask,
                            mode=entropy_token_weight_mode,
                            scope=str(self.config.policy_loss.get("entropy_token_weight_scope", "response")),
                            renormalize_mean=bool(
                                self.config.policy_loss.get("entropy_token_weight_renormalize_mean", False)
                            ),
                        )
                        # Seed neutral values so every micro-batch on every rank writes the same key set.
                        # reduce_metrics means over the per-rank lists, and a key that only some
                        # micro-batches produce leaves those lists ragged and makes np.mean raise.
                        micro_batch_metrics.update(
                            {
                                "actor/entropy_weight_mean": 1.0,
                                "actor/entropy_weight_std": 0.0,
                                "actor/entropy_weight_min": 1.0,
                                "actor/entropy_weight_zero_fraction": 0.0,
                                "actor/entropy_weight_advantage_rms": 0.0,
                            }
                        )
                        valid_weights = entropy_weights[valid_mask]
                        if valid_weights.numel() > 0:
                            micro_batch_metrics["actor/entropy_weight_mean"] = valid_weights.mean().detach().item()
                            micro_batch_metrics["actor/entropy_weight_std"] = (
                                valid_weights.std(unbiased=False).detach().item()
                            )
                            micro_batch_metrics["actor/entropy_weight_min"] = valid_weights.min().detach().item()
                            micro_batch_metrics["actor/entropy_weight_zero_fraction"] = (
                                (valid_weights <= 1e-6).float().mean().detach().item()
                            )

                            # Does entropy actually track advantage magnitude? Pooled over the whole
                            # update so the answer does not depend on how micro-batches were cut.
                            entropy_valid = entropy[valid_mask]
                            adv_valid = unweighted_advantages[valid_mask]
                            token_adv_entropy.update(entropy_valid, adv_valid)
                            token_abs_adv_entropy.update(entropy_valid, adv_valid.abs())

                            # A response's advantage is shared by all of its tokens, so the response-level
                            # pairing is the one free of within-response attenuation.
                            lengths = response_mask.sum(dim=-1).clamp_min(1.0)
                            non_empty = response_mask.sum(dim=-1) > 0
                            response_entropy = (entropy * response_mask).sum(dim=-1) / lengths
                            response_advantage = (unweighted_advantages * response_mask).sum(dim=-1) / lengths
                            resp_adv_entropy.update(response_entropy[non_empty], response_advantage[non_empty])
                            resp_abs_adv_entropy.update(
                                response_entropy[non_empty], response_advantage[non_empty].abs()
                            )

                        advantages = unweighted_advantages * entropy_weights
                        valid_adv = advantages[valid_mask]
                        if valid_adv.numel() > 0:
                            micro_batch_metrics["actor/entropy_weight_advantage_rms"] = (
                                valid_adv.square().mean().sqrt().detach().item()
                            )

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=pg_response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using pure rollout correction mode (metrics already in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "rollout_correction" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=token_response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm, optimizer_stepped = self._optimizer_step()
                optimizer_steps += int(optimizer_stepped)
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        if opgd_enabled:
            metrics["actor/opgd_optimizer_steps"] = [float(optimizer_steps)]
            metrics["actor/opgd_optimizer_skipped"] = [float(optimizer_steps == 0)]
        if entropy_token_weight_mode != "none":
            # One value per update, so reduce_metrics averages one entry per rank rather than once
            # per micro-batch. A positive abs-advantage correlation or slope means uncertain tokens
            # carry the larger updates, which is the setting the weighting direction should match.
            metrics["actor/token_entropy_advantage_corr"] = [token_adv_entropy.corr()]
            metrics["actor/token_entropy_abs_advantage_corr"] = [token_abs_adv_entropy.corr()]
            metrics["actor/token_abs_advantage_slope_on_entropy"] = [token_abs_adv_entropy.slope()]
            metrics["actor/resp_entropy_advantage_corr"] = [resp_adv_entropy.corr()]
            metrics["actor/resp_entropy_abs_advantage_corr"] = [resp_abs_adv_entropy.corr()]
            metrics["actor/resp_abs_advantage_slope_on_entropy"] = [resp_abs_adv_entropy.slope()]
        self.actor_optimizer.zero_grad()
        return metrics
