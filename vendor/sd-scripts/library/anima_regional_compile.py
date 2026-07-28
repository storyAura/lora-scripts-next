from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from packaging.version import InvalidVersion, Version


class AnimaRegionalCompileError(RuntimeError):
    """Raised when regional compilation cannot preserve the training contract."""


@dataclass(frozen=True)
class AnimaRegionalCompileReport:
    backend: str
    compiled_block_count: int
    compiled_block_names: tuple[str, ...]


def validate_anima_regional_compile_request(
    enabled: bool,
    platform_name: str,
    cuda_available: bool,
    torch_version: str,
    global_torch_compile: bool,
    backend: str,
    blocks_to_swap: object,
    base_model_quantization: str,
    cpu_offload_checkpointing: bool,
    unsloth_offload_checkpointing: bool,
) -> None:
    if not enabled:
        return
    if platform_name != "linux":
        raise AnimaRegionalCompileError(
            f"Anima regional torch.compile is supported only on Linux, got platform={platform_name!r}"
        )
    if not cuda_available:
        raise AnimaRegionalCompileError(
            "Anima regional torch.compile requires CUDA, but torch.cuda.is_available() is False"
        )
    try:
        installed_torch = Version(torch_version)
    except InvalidVersion as error:
        raise AnimaRegionalCompileError(
            f"could not parse installed torch version {torch_version!r}"
        ) from error
    if installed_torch < Version("2.6.0"):
        raise AnimaRegionalCompileError(
            f"Anima regional torch.compile requires torch>=2.6, but {torch_version} is installed"
        )
    if global_torch_compile:
        raise AnimaRegionalCompileError(
            "anima_compile_blocks cannot be combined with global torch_compile"
        )
    if backend != "inductor":
        raise AnimaRegionalCompileError(
            f"Anima regional compile supports only backend='inductor', got {backend!r}"
        )
    if blocks_to_swap not in (None, "", 0, "0"):
        raise AnimaRegionalCompileError(
            "Anima regional compile cannot be combined with blocks_to_swap"
        )
    if base_model_quantization != "none":
        raise AnimaRegionalCompileError(
            "Anima regional compile cannot be combined with bitsandbytes base model quantization"
        )
    if cpu_offload_checkpointing:
        raise AnimaRegionalCompileError(
            "Anima regional compile cannot be combined with cpu_offload_checkpointing"
        )
    if unsloth_offload_checkpointing:
        raise AnimaRegionalCompileError(
            "Anima regional compile cannot be combined with unsloth_offload_checkpointing"
        )


def compile_anima_blocks(
    model: torch.nn.Module,
    backend: str,
) -> AnimaRegionalCompileReport:
    blocks = getattr(model, "blocks", None)
    if not isinstance(blocks, torch.nn.ModuleList) or not blocks:
        raise AnimaRegionalCompileError(
            "Anima regional compile requires model.blocks to be a non-empty torch.nn.ModuleList"
        )

    original_forwards: list[Callable[..., torch.Tensor]] = []
    block_names: list[str] = []
    for block_index, block in enumerate(blocks):
        if getattr(block, "_anima_regional_compile_backend", None) is not None:
            raise AnimaRegionalCompileError(
                f"Anima block {block_index} is already regionally compiled"
            )
        original_forward = getattr(block, "_forward", None)
        if not callable(original_forward):
            raise AnimaRegionalCompileError(
                f"Anima block {block_index} does not expose a callable _forward region"
            )
        original_forwards.append(original_forward)
        block_names.append(f"blocks.{block_index}")

    compiled_forwards = tuple(
        torch.compile(
            original_forward,
            backend=backend,
            fullgraph=False,
            dynamic=False,
        )
        for original_forward in original_forwards
    )
    for block, compiled_forward in zip(blocks, compiled_forwards):
        setattr(block, "_forward", compiled_forward)
        setattr(block, "_anima_regional_compile_backend", backend)

    return AnimaRegionalCompileReport(
        backend=backend,
        compiled_block_count=len(compiled_forwards),
        compiled_block_names=tuple(block_names),
    )
