from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch.utils.checkpoint import checkpoint


REPO_ROOT = Path(__file__).resolve().parents[1]
SD_SCRIPTS_ROOT = REPO_ROOT / "vendor" / "sd-scripts"
sys.path.insert(0, str(SD_SCRIPTS_ROOT))

from library.anima_models import Block  # noqa: E402
from library.selective_activation_checkpointing import (  # noqa: E402
    create_anima_selective_checkpoint_contexts,
    validate_selective_checkpoint_runtime,
)


def _loss(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.sin(value @ weight).square().mean()


def test_selective_checkpoint_matches_eager_gradients() -> None:
    eager_value = torch.randn(4, 16, dtype=torch.float32, requires_grad=True)
    eager_weight = torch.randn(16, 12, dtype=torch.float32, requires_grad=True)
    checkpoint_value = eager_value.detach().clone().requires_grad_(True)
    checkpoint_weight = eager_weight.detach().clone().requires_grad_(True)

    eager_loss = _loss(eager_value, eager_weight)
    selective_loss = checkpoint(
        _loss,
        checkpoint_value,
        checkpoint_weight,
        use_reentrant=False,
        context_fn=create_anima_selective_checkpoint_contexts,
    )
    eager_loss.backward()
    selective_loss.backward()

    torch.testing.assert_close(selective_loss, eager_loss)
    torch.testing.assert_close(checkpoint_value.grad, eager_value.grad)
    torch.testing.assert_close(checkpoint_weight.grad, eager_weight.grad)


def test_block_checkpoint_modes_are_explicit_and_mutually_exclusive() -> None:
    block = Block(16, 8, 4, 2.0, False, 4)

    block.enable_gradient_checkpointing(cpu_offload=False, unsloth_offload=False)
    assert block.gradient_checkpointing is True
    assert block.selective_activation_checkpointing is False

    block.enable_selective_activation_checkpointing()
    assert block.gradient_checkpointing is True
    assert block.selective_activation_checkpointing is True
    assert block.cpu_offload_checkpointing is False
    assert block.unsloth_offload_checkpointing is False

    block.disable_gradient_checkpointing()
    assert block.gradient_checkpointing is False
    assert block.selective_activation_checkpointing is False


def test_selective_checkpoint_runtime_validation_rejects_conflicts() -> None:
    validate_selective_checkpoint_runtime("standard", True, False, False, None)

    with pytest.raises(ValueError, match="requires gradient_checkpointing"):
        validate_selective_checkpoint_runtime("selective", False, False, False, None)
    with pytest.raises(ValueError, match="cpu_offload_checkpointing"):
        validate_selective_checkpoint_runtime("selective", True, True, False, None)
    with pytest.raises(ValueError, match="unsloth_offload_checkpointing"):
        validate_selective_checkpoint_runtime("selective", True, False, True, None)
    with pytest.raises(ValueError, match="blocks_to_swap"):
        validate_selective_checkpoint_runtime("selective", True, False, False, 2)
    with pytest.raises(ValueError, match="unsupported"):
        validate_selective_checkpoint_runtime("magic", True, False, False, None)
