from __future__ import annotations

from dataclasses import dataclass

import torch
from packaging.version import InvalidVersion, Version


SUPPORTED_FSDP2_NETWORK_MODULES = frozenset(
    {
        "networks.lora_anima",
        "networks.lora_flux",
        "networks.lora_sd3",
    }
)

FSDP2_TRANSFORMER_CLASS_NAMES = (
    "Block",
    "DoubleStreamBlock",
    "SingleStreamBlock",
    "MMDiTBlock",
)


class FSDP2FrozenBaseError(RuntimeError):
    """Raised when frozen-base FSDP2 would violate adapter training semantics."""


@dataclass(frozen=True)
class FSDP2FrozenBasePlan:
    transformer_module_names: tuple[str, ...]
    base_parameter_count: int


@dataclass(frozen=True)
class FSDP2FrozenBaseReport:
    transformer_module_names: tuple[str, ...]
    base_parameter_count: int
    cpu_offload: bool


def _parse_version(version_text: str, package_name: str) -> Version:
    try:
        return Version(version_text)
    except InvalidVersion as error:
        raise FSDP2FrozenBaseError(
            f"could not parse installed {package_name} version {version_text!r}"
        ) from error


def validate_fsdp2_frozen_base_request(
    enabled: bool,
    platform_name: str,
    cuda_available: bool,
    world_size: int,
    cuda_device_count: int,
    accelerate_version: str,
    torch_version: str,
    deepspeed_enabled: bool,
    global_torch_compile: bool,
    blocks_to_swap: object,
    base_model_quantization: str,
    anima_regional_compile: bool,
    train_text_encoder: bool,
    train_unet_only: bool,
    train_text_encoder_only: bool,
    network_module: object,
) -> None:
    if not enabled:
        return
    if platform_name != "linux":
        raise FSDP2FrozenBaseError(
            f"FSDP2 frozen-base sharding is supported only on Linux, got platform={platform_name!r}"
        )
    if not cuda_available:
        raise FSDP2FrozenBaseError(
            "FSDP2 frozen-base sharding requires CUDA, but torch.cuda.is_available() is False"
        )
    if world_size < 2:
        raise FSDP2FrozenBaseError(
            f"FSDP2 frozen-base sharding requires WORLD_SIZE>=2, got {world_size}"
        )
    if cuda_device_count < 2:
        raise FSDP2FrozenBaseError(
            f"FSDP2 frozen-base sharding requires at least two visible CUDA devices, got {cuda_device_count}"
        )
    if _parse_version(accelerate_version, "accelerate") < Version("1.6.0"):
        raise FSDP2FrozenBaseError(
            f"FSDP2 frozen-base sharding requires accelerate>=1.6.0, got {accelerate_version}"
        )
    if _parse_version(torch_version, "torch") < Version("2.5.1"):
        raise FSDP2FrozenBaseError(
            f"FSDP2 frozen-base sharding requires torch>=2.5.1, got {torch_version}"
        )
    if deepspeed_enabled:
        raise FSDP2FrozenBaseError(
            "FSDP2 frozen-base sharding cannot be combined with DeepSpeed"
        )
    if global_torch_compile:
        raise FSDP2FrozenBaseError(
            "FSDP2 frozen-base sharding cannot be combined with global torch_compile"
        )
    if blocks_to_swap not in (None, "", 0, "0"):
        raise FSDP2FrozenBaseError(
            "FSDP2 frozen-base sharding cannot be combined with blocks_to_swap"
        )
    if base_model_quantization != "none":
        raise FSDP2FrozenBaseError(
            "FSDP2 frozen-base sharding cannot be combined with base model quantization"
        )
    if anima_regional_compile:
        raise FSDP2FrozenBaseError(
            "FSDP2 frozen-base sharding cannot be combined with Anima regional compile"
        )
    if train_text_encoder:
        raise FSDP2FrozenBaseError(
            "FSDP2 frozen-base sharding currently requires the text encoder to remain frozen"
        )
    if not train_unet_only:
        raise FSDP2FrozenBaseError(
            "FSDP2 frozen-base sharding requires adapter training to target the DiT/U-Net only"
        )
    if train_text_encoder_only:
        raise FSDP2FrozenBaseError(
            "FSDP2 frozen-base sharding cannot be used with network_train_text_encoder_only"
        )
    normalized_network_module = str(network_module or "").strip()
    if normalized_network_module not in SUPPORTED_FSDP2_NETWORK_MODULES:
        raise FSDP2FrozenBaseError(
            "FSDP2 frozen-base sharding currently supports standard LoRA network modules only; "
            f"got network_module={normalized_network_module!r}"
        )


def plan_frozen_base_sharding(
    model: torch.nn.Module,
    transformer_class_names: tuple[str, ...],
) -> FSDP2FrozenBasePlan:
    trainable_parameter_names = tuple(
        parameter_name
        for parameter_name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    if trainable_parameter_names:
        raise FSDP2FrozenBaseError(
            "the FSDP2 base model must be frozen before sharding; "
            f"trainable base parameters include {trainable_parameter_names[:5]!r}"
        )
    transformer_module_names = tuple(
        module_name
        for module_name, module in model.named_modules()
        if module_name and module.__class__.__name__ in transformer_class_names
    )
    if not transformer_module_names:
        raise FSDP2FrozenBaseError(
            "FSDP2 frozen-base sharding matched no transformer modules; "
            f"class names={transformer_class_names!r}"
        )
    return FSDP2FrozenBasePlan(
        transformer_module_names=transformer_module_names,
        base_parameter_count=sum(parameter.numel() for parameter in model.parameters()),
    )


def shard_frozen_base_model_fsdp2(
    model: torch.nn.Module,
    device: torch.device,
    transformer_class_names: tuple[str, ...],
    cpu_offload: bool,
) -> FSDP2FrozenBaseReport:
    if not torch.distributed.is_initialized():
        raise FSDP2FrozenBaseError(
            "torch.distributed must be initialized before applying FSDP2 frozen-base sharding"
        )
    if torch.distributed.get_world_size() < 2:
        raise FSDP2FrozenBaseError(
            f"FSDP2 frozen-base sharding requires at least two ranks, got {torch.distributed.get_world_size()}"
        )

    plan = plan_frozen_base_sharding(model, transformer_class_names)
    model.to(device)

    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        MixedPrecisionPolicy,
        OffloadPolicy,
        fully_shard,
    )

    offload_policy = CPUOffloadPolicy(pin_memory=True) if cpu_offload else OffloadPolicy()
    mixed_precision_policy = MixedPrecisionPolicy(
        param_dtype=None,
        reduce_dtype=None,
        output_dtype=None,
        cast_forward_inputs=True,
    )
    modules_by_name = dict(model.named_modules())
    for module_name in reversed(plan.transformer_module_names):
        fully_shard(
            modules_by_name[module_name],
            mesh=None,
            reshard_after_forward=True,
            shard_placement_fn=None,
            mp_policy=mixed_precision_policy,
            offload_policy=offload_policy,
            ignored_params=None,
        )
    fully_shard(
        model,
        mesh=None,
        reshard_after_forward=True,
        shard_placement_fn=None,
        mp_policy=mixed_precision_policy,
        offload_policy=offload_policy,
        ignored_params=None,
    )
    return FSDP2FrozenBaseReport(
        transformer_module_names=plan.transformer_module_names,
        base_parameter_count=plan.base_parameter_count,
        cpu_offload=cpu_offload,
    )
