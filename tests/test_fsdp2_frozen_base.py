from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SD_SCRIPTS_ROOT = REPO_ROOT / "vendor" / "sd-scripts"
sys.path.insert(0, str(SD_SCRIPTS_ROOT))

from library.fsdp2_frozen_base import (  # noqa: E402
    FSDP2FrozenBaseError,
    plan_frozen_base_sharding,
    validate_fsdp2_frozen_base_request,
)


class Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(8, 8, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.proj(value)


class TinyBase(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block(), Block()])
        self.final = torch.nn.Linear(8, 8, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = value
        for block in self.blocks:
            hidden = block(hidden)
        return self.final(hidden)


def test_frozen_base_plan_excludes_external_trainable_adapter() -> None:
    base = TinyBase()
    base.requires_grad_(False)
    adapter = torch.nn.Linear(8, 8, bias=False)

    plan = plan_frozen_base_sharding(base, ("Block",))

    assert plan.transformer_module_names == ("blocks.0", "blocks.1")
    assert plan.base_parameter_count == sum(parameter.numel() for parameter in base.parameters())
    assert all(parameter.requires_grad is False for parameter in base.parameters())
    assert all(parameter.requires_grad is True for parameter in adapter.parameters())


def test_frozen_base_plan_rejects_trainable_or_missing_transformer_modules() -> None:
    trainable_base = TinyBase()
    with pytest.raises(FSDP2FrozenBaseError, match="must be frozen"):
        plan_frozen_base_sharding(trainable_base, ("Block",))

    frozen_base = TinyBase()
    frozen_base.requires_grad_(False)
    with pytest.raises(FSDP2FrozenBaseError, match="matched no transformer modules"):
        plan_frozen_base_sharding(frozen_base, ("MissingBlock",))


def test_fsdp2_runtime_validation_requires_linux_multi_gpu_and_safe_features() -> None:
    validate_fsdp2_frozen_base_request(
        False,
        "win32",
        False,
        1,
        1,
        "0.1.0",
        "1.0.0",
        False,
        False,
        None,
        "none",
        False,
        False,
        True,
        False,
        "networks.lora_anima",
    )
    validate_fsdp2_frozen_base_request(
        True,
        "linux",
        True,
        2,
        2,
        "1.6.0",
        "2.7.0",
        False,
        False,
        None,
        "none",
        False,
        False,
        True,
        False,
        "networks.lora_anima",
    )

    invalid_requests = (
        ("win32", True, 2, 2, "1.6.0", "2.7.0", False, False, None, "none", False, False, True, False, "networks.lora_anima", "Linux"),
        ("linux", False, 2, 2, "1.6.0", "2.7.0", False, False, None, "none", False, False, True, False, "networks.lora_anima", "CUDA"),
        ("linux", True, 1, 2, "1.6.0", "2.7.0", False, False, None, "none", False, False, True, False, "networks.lora_anima", "WORLD_SIZE"),
        ("linux", True, 2, 2, "1.5.2", "2.7.0", False, False, None, "none", False, False, True, False, "networks.lora_anima", "accelerate>=1.6"),
        ("linux", True, 2, 2, "1.6.0", "2.5.0", False, False, None, "none", False, False, True, False, "networks.lora_anima", "torch>=2.5.1"),
        ("linux", True, 2, 2, "1.6.0", "2.7.0", True, False, None, "none", False, False, True, False, "networks.lora_anima", "DeepSpeed"),
        ("linux", True, 2, 2, "1.6.0", "2.7.0", False, True, None, "none", False, False, True, False, "networks.lora_anima", "torch_compile"),
        ("linux", True, 2, 2, "1.6.0", "2.7.0", False, False, 2, "none", False, False, True, False, "networks.lora_anima", "blocks_to_swap"),
        ("linux", True, 2, 2, "1.6.0", "2.7.0", False, False, None, "nf4", False, False, True, False, "networks.lora_anima", "quantization"),
        ("linux", True, 2, 2, "1.6.0", "2.7.0", False, False, None, "none", True, False, True, False, "networks.lora_anima", "regional compile"),
        ("linux", True, 2, 2, "1.6.0", "2.7.0", False, False, None, "none", False, True, True, False, "networks.lora_anima", "text encoder"),
        ("linux", True, 2, 2, "1.6.0", "2.7.0", False, False, None, "none", False, False, False, False, "networks.lora_anima", "DiT"),
        ("linux", True, 2, 2, "1.6.0", "2.7.0", False, False, None, "none", False, False, True, True, "networks.lora_anima", "text_encoder_only"),
        ("linux", True, 2, 2, "1.6.0", "2.7.0", False, False, None, "none", False, False, True, False, "lycoris.kohya", "standard LoRA"),
    )
    for request in invalid_requests:
        with pytest.raises(FSDP2FrozenBaseError, match=request[-1]):
            validate_fsdp2_frozen_base_request(True, *request[:-1])
