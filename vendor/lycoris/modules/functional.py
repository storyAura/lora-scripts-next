from __future__ import annotations

from collections.abc import Callable

import torch


def compute_merged_delta(
    base_weight: torch.Tensor,
    diff_weight: torch.Tensor,
    multiplier: float,
    transform: Callable[[torch.Tensor], torch.Tensor] | None,
) -> torch.Tensor:
    base_fp32 = base_weight.float()
    merged_fp32 = base_fp32 + diff_weight.float() * multiplier
    transformed_fp32 = transform(merged_fp32) if transform is not None else merged_fp32
    return (transformed_fp32 - base_fp32).to(dtype=base_weight.dtype)
