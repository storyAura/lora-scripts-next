from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SD_SCRIPTS_ROOT = REPO_ROOT / "vendor" / "sd-scripts"
sys.path.insert(0, str(SD_SCRIPTS_ROOT))

from library.anima_regional_compile import (  # noqa: E402
    AnimaRegionalCompileError,
    compile_anima_blocks,
    validate_anima_regional_compile_request,
)


class TinyBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(8, 8, bias=False)

    def _forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(self.proj(value))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self._forward(value)


class TinyAnima(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([TinyBlock(), TinyBlock()])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = value
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


def test_regional_compile_preserves_module_identity_state_and_gradients() -> None:
    eager_model = TinyAnima()
    compiled_model = TinyAnima()
    compiled_model.load_state_dict(eager_model.state_dict())
    state_keys_before = tuple(compiled_model.state_dict())

    report = compile_anima_blocks(compiled_model, "eager")

    assert report.backend == "eager"
    assert report.compiled_block_count == 2
    assert all(block.__class__.__name__ == "TinyBlock" for block in compiled_model.blocks)
    assert tuple(compiled_model.state_dict()) == state_keys_before

    eager_input = torch.randn(3, 8, requires_grad=True)
    compiled_input = eager_input.detach().clone().requires_grad_(True)
    eager_output = eager_model(eager_input)
    compiled_output = compiled_model(compiled_input)
    eager_output.square().mean().backward()
    compiled_output.square().mean().backward()

    torch.testing.assert_close(compiled_output, eager_output)
    torch.testing.assert_close(compiled_input.grad, eager_input.grad)
    for eager_parameter, compiled_parameter in zip(
        eager_model.parameters(),
        compiled_model.parameters(),
    ):
        torch.testing.assert_close(compiled_parameter.grad, eager_parameter.grad)

    with pytest.raises(AnimaRegionalCompileError, match="already regionally compiled"):
        compile_anima_blocks(compiled_model, "eager")


def test_regional_compile_validation_enforces_supported_environment() -> None:
    validate_anima_regional_compile_request(
        False,
        "win32",
        False,
        "1.0.0",
        False,
        "inductor",
        None,
        "none",
        False,
        False,
    )

    invalid_requests = (
        ("win32", True, "2.7.0", False, "inductor", None, "none", False, False, "Linux"),
        ("linux", False, "2.7.0", False, "inductor", None, "none", False, False, "CUDA"),
        ("linux", True, "2.5.1", False, "inductor", None, "none", False, False, "torch>=2.6"),
        ("linux", True, "2.7.0", True, "inductor", None, "none", False, False, "torch_compile"),
        ("linux", True, "2.7.0", False, "eager", None, "none", False, False, "inductor"),
        ("linux", True, "2.7.0", False, "inductor", 2, "none", False, False, "blocks_to_swap"),
        ("linux", True, "2.7.0", False, "inductor", None, "nf4", False, False, "quantization"),
        ("linux", True, "2.7.0", False, "inductor", None, "none", True, False, "cpu_offload"),
        ("linux", True, "2.7.0", False, "inductor", None, "none", False, True, "unsloth"),
    )
    for (
        platform_name,
        cuda_available,
        torch_version,
        global_compile,
        backend,
        blocks_to_swap,
        quantization_mode,
        cpu_offload,
        unsloth_offload,
        expected_message,
    ) in invalid_requests:
        with pytest.raises(AnimaRegionalCompileError, match=expected_message):
            validate_anima_regional_compile_request(
                True,
                platform_name,
                cuda_available,
                torch_version,
                global_compile,
                backend,
                blocks_to_swap,
                quantization_mode,
                cpu_offload,
                unsloth_offload,
            )
