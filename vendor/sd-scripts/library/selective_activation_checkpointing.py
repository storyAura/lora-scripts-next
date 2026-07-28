from __future__ import annotations

from contextlib import AbstractContextManager

import torch
from torch.utils.checkpoint import (
    CheckpointPolicy,
    SelectiveCheckpointContext,
    create_selective_checkpoint_contexts,
)


ANIMA_EXPENSIVE_OPERATIONS: frozenset[torch._ops.OpOverload] = frozenset(
    {
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten._scaled_dot_product_flash_attention.default,
        torch.ops.aten._scaled_dot_product_efficient_attention.default,
        torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    }
)


def _anima_checkpoint_policy(
    context: SelectiveCheckpointContext,
    operation: torch._ops.OpOverload,
    *args: object,
    **kwargs: object,
) -> CheckpointPolicy:
    del context, args, kwargs
    if operation in ANIMA_EXPENSIVE_OPERATIONS:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def create_anima_selective_checkpoint_contexts() -> tuple[
    AbstractContextManager[object],
    AbstractContextManager[object],
]:
    forward_context, recompute_context = create_selective_checkpoint_contexts(
        _anima_checkpoint_policy,
        allow_cache_entry_mutation=False,
    )
    return forward_context, recompute_context


def validate_selective_checkpoint_runtime(
    mode: str,
    gradient_checkpointing: bool,
    cpu_offload_checkpointing: bool,
    unsloth_offload_checkpointing: bool,
    blocks_to_swap: object,
) -> None:
    if mode not in ("standard", "selective"):
        raise ValueError(
            f"unsupported Anima gradient checkpointing mode {mode!r}; expected 'standard' or 'selective'"
        )
    if mode == "standard":
        return
    if not gradient_checkpointing:
        raise ValueError(
            "anima_gradient_checkpointing_mode='selective' requires gradient_checkpointing=true"
        )
    if cpu_offload_checkpointing:
        raise ValueError(
            "selective activation checkpointing cannot be combined with cpu_offload_checkpointing"
        )
    if unsloth_offload_checkpointing:
        raise ValueError(
            "selective activation checkpointing cannot be combined with unsloth_offload_checkpointing"
        )
    if blocks_to_swap not in (None, "", 0, "0"):
        raise ValueError(
            "selective activation checkpointing cannot be combined with blocks_to_swap"
        )
