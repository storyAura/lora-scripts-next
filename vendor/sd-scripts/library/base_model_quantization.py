from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Literal

import torch
from packaging.version import InvalidVersion, Version


BaseModelQuantizationMode = Literal["int8", "nf4"]
BaseModelQuantizationComputeDtype = Literal["fp16", "bf16"]
BaseModelFamily = Literal["anima", "flux", "sd3"]


MODEL_INCLUDE_PATTERNS: dict[BaseModelFamily, tuple[str, ...]] = {
    "anima": ("blocks.*",),
    "flux": ("double_blocks.*", "single_blocks.*"),
    "sd3": ("joint_blocks.*",),
}

MODEL_SKIP_PATTERNS: dict[BaseModelFamily, tuple[str, ...]] = {
    "anima": (
        "blocks.*.adaln_modulation.*",
        "blocks.*.adaln_modulation_*.*",
    ),
    "flux": (),
    "sd3": (),
}

SUPPORTED_QUANTIZED_NETWORK_MODULES: dict[BaseModelFamily, frozenset[str]] = {
    "anima": frozenset(
        {
            "networks.lora_anima",
            "networks.lora_fa_anima",
            "networks.vera_anima",
        }
    ),
    "flux": frozenset({"networks.lora_flux"}),
    "sd3": frozenset({"networks.lora_sd3"}),
}


class BaseModelQuantizationError(RuntimeError):
    """Raised when frozen-base quantization cannot be applied safely."""


@dataclass(frozen=True)
class BaseModelQuantizationReport:
    mode: BaseModelQuantizationMode
    converted_modules: tuple[str, ...]
    skipped_modules: tuple[str, ...]
    original_weight_bytes: int
    estimated_quantized_weight_bytes: int


def validate_base_model_quantization_runtime(
    mode: str,
    compute_dtype: str,
    cuda_available: bool,
    bitsandbytes_version: str,
) -> None:
    if mode not in ("int8", "nf4"):
        raise BaseModelQuantizationError(
            f"unsupported base model quantization mode {mode!r}; expected 'int8' or 'nf4'"
        )
    if compute_dtype not in ("fp16", "bf16"):
        raise BaseModelQuantizationError(
            f"unsupported base model quantization compute dtype {compute_dtype!r}; expected 'fp16' or 'bf16'"
        )
    if not cuda_available:
        raise BaseModelQuantizationError(
            f"base model quantization mode {mode!r} requires CUDA, but torch.cuda.is_available() is False"
        )
    try:
        installed_version = Version(bitsandbytes_version)
    except InvalidVersion as error:
        raise BaseModelQuantizationError(
            f"could not parse installed bitsandbytes version {bitsandbytes_version!r}"
        ) from error
    if installed_version < Version("0.46.0"):
        raise BaseModelQuantizationError(
            f"base model quantization requires bitsandbytes>=0.46.0, but {bitsandbytes_version} is installed"
        )


def compute_dtype_from_name(compute_dtype: str) -> torch.dtype:
    if compute_dtype == "fp16":
        return torch.float16
    if compute_dtype == "bf16":
        return torch.bfloat16
    raise BaseModelQuantizationError(
        f"unsupported base model quantization compute dtype {compute_dtype!r}; expected 'fp16' or 'bf16'"
    )


def normalize_skip_module_patterns(raw_patterns: object) -> tuple[str, ...]:
    if raw_patterns is None:
        return ()
    if not isinstance(raw_patterns, (list, tuple)):
        raise BaseModelQuantizationError(
            "base_model_quantization_skip_modules must be a list of glob strings"
        )
    normalized: list[str] = []
    for raw_pattern in raw_patterns:
        if not isinstance(raw_pattern, str) or not raw_pattern.strip():
            raise BaseModelQuantizationError(
                "base_model_quantization_skip_modules must contain non-empty glob strings"
            )
        pattern = raw_pattern.strip()
        if pattern not in normalized:
            normalized.append(pattern)
    return tuple(normalized)


def validate_base_model_quantization_training_request(
    mode: str,
    compute_dtype: str,
    model_family: str,
    network_module: object,
    fp8_base: bool,
    fp8_base_unet: bool,
    blocks_to_swap: object,
) -> None:
    if mode == "none":
        return
    if model_family not in MODEL_INCLUDE_PATTERNS:
        raise BaseModelQuantizationError(
            f"unsupported quantized base model family {model_family!r}; expected one of {tuple(MODEL_INCLUDE_PATTERNS)!r}"
        )
    if mode not in ("int8", "nf4"):
        raise BaseModelQuantizationError(
            f"unsupported base model quantization mode {mode!r}; expected 'none', 'int8', or 'nf4'"
        )
    compute_dtype_from_name(compute_dtype)
    if fp8_base or fp8_base_unet:
        raise BaseModelQuantizationError(
            "base_model_quantization cannot be combined with fp8_base or fp8_base_unet"
        )
    if blocks_to_swap not in (None, 0, "0", ""):
        raise BaseModelQuantizationError(
            "base_model_quantization cannot be combined with blocks_to_swap because repeated "
            "bitsandbytes device transfers are not a supported training path"
        )
    normalized_network_module = str(network_module or "").strip()
    supported_modules = SUPPORTED_QUANTIZED_NETWORK_MODULES[model_family]
    if normalized_network_module not in supported_modules:
        raise BaseModelQuantizationError(
            f"base_model_quantization for {model_family} requires one of {tuple(sorted(supported_modules))!r}, "
            f"but network_module={normalized_network_module!r}; PiSSA, DoRA, LyCORIS, and custom modules "
            "must remain full precision unless they declare quantized-weight support"
        )


def _matches_any(module_name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatchcase(module_name, pattern) for pattern in patterns)


def _validate_patterns(patterns: tuple[str, ...], field_name: str) -> None:
    invalid_patterns = tuple(pattern for pattern in patterns if not isinstance(pattern, str) or not pattern.strip())
    if invalid_patterns:
        raise BaseModelQuantizationError(
            f"{field_name} must contain non-empty glob strings; invalid values: {invalid_patterns!r}"
        )


def _replace_child_module(
    root_module: torch.nn.Module,
    module_name: str,
    replacement: torch.nn.Module,
) -> None:
    parent_name, separator, child_name = module_name.rpartition(".")
    parent_module = root_module.get_submodule(parent_name) if separator else root_module
    setattr(parent_module, child_name, replacement)


def _quantize_linear(
    linear: torch.nn.Linear,
    mode: BaseModelQuantizationMode,
    compute_dtype: torch.dtype,
) -> torch.nn.Linear:
    import bitsandbytes as bnb

    if linear.weight.is_meta:
        raise BaseModelQuantizationError("cannot quantize a Linear layer whose weight is still on the meta device")

    weight = linear.weight.detach().clone()
    bias = linear.bias.detach().clone() if linear.bias is not None else None
    device = weight.device

    if mode == "int8":
        replacement = bnb.nn.Linear8bitLt(
            linear.in_features,
            linear.out_features,
            bias=bias is not None,
            has_fp16_weights=False,
            threshold=6.0,
            index=None,
            device=device,
        )
        replacement.weight = bnb.nn.Int8Params(
            weight,
            requires_grad=False,
            has_fp16_weights=False,
        )
    else:
        replacement = bnb.nn.Linear4bit(
            linear.in_features,
            linear.out_features,
            bias=bias is not None,
            compute_dtype=compute_dtype,
            compress_statistics=True,
            quant_type="nf4",
            quant_storage=torch.uint8,
            device=device,
        )
        replacement.weight = bnb.nn.Params4bit(
            weight,
            requires_grad=False,
            quant_state=None,
            blocksize=64,
            compress_statistics=True,
            quant_type="nf4",
            quant_storage=torch.uint8,
            module=replacement,
            bnb_quantized=False,
        )

    if bias is not None:
        replacement.bias = torch.nn.Parameter(bias, requires_grad=False)
    replacement.train(linear.training)
    if device.type == "cuda":
        replacement.to(device=device)
    return replacement


def quantize_frozen_linear_layers(
    model: torch.nn.Module,
    mode: str,
    compute_dtype: torch.dtype,
    include_module_patterns: tuple[str, ...],
    skip_module_patterns: tuple[str, ...],
) -> BaseModelQuantizationReport:
    import bitsandbytes as bnb

    if mode not in ("int8", "nf4"):
        raise BaseModelQuantizationError(
            f"unsupported base model quantization mode {mode!r}; expected 'int8' or 'nf4'"
        )
    if compute_dtype not in (torch.float16, torch.bfloat16):
        raise BaseModelQuantizationError(
            f"base model quantization compute dtype must be torch.float16 or torch.bfloat16, got {compute_dtype}"
        )
    _validate_patterns(include_module_patterns, "include_module_patterns")
    _validate_patterns(skip_module_patterns, "skip_module_patterns")

    converted_modules: list[str] = []
    skipped_modules: list[str] = []
    original_weight_bytes = 0
    estimated_quantized_weight_bytes = 0
    quantized_types = (bnb.nn.Linear8bitLt, bnb.nn.Linear4bit)

    for module_name, module in tuple(model.named_modules()):
        if not module_name or not isinstance(module, torch.nn.Linear):
            continue
        if include_module_patterns and not _matches_any(module_name, include_module_patterns):
            continue
        if _matches_any(module_name, skip_module_patterns):
            skipped_modules.append(module_name)
            continue
        if isinstance(module, quantized_types):
            raise BaseModelQuantizationError(
                f"module {module_name!r} is already quantized as {module.__class__.__name__}; repeated conversion is not allowed"
            )
        if module.weight.requires_grad or (module.bias is not None and module.bias.requires_grad):
            raise BaseModelQuantizationError(
                f"module {module_name!r} must be frozen before base model quantization; "
                "quantized training supports adapter parameters only"
            )

        weight_elements = module.weight.numel()
        original_weight_bytes += weight_elements * module.weight.element_size()
        estimated_quantized_weight_bytes += weight_elements if mode == "int8" else (weight_elements + 1) // 2
        replacement = _quantize_linear(module, mode, compute_dtype)
        _replace_child_module(model, module_name, replacement)
        converted_modules.append(module_name)

    if not converted_modules:
        raise BaseModelQuantizationError(
            "base model quantization matched no frozen Linear modules; "
            f"include patterns={include_module_patterns!r}, skip patterns={skip_module_patterns!r}"
        )

    return BaseModelQuantizationReport(
        mode=mode,
        converted_modules=tuple(converted_modules),
        skipped_modules=tuple(skipped_modules),
        original_weight_bytes=original_weight_bytes,
        estimated_quantized_weight_bytes=estimated_quantized_weight_bytes,
    )


def quantize_model_for_lora(
    model: torch.nn.Module,
    model_family: str,
    mode: str,
    compute_dtype_name: str,
    user_skip_module_patterns: tuple[str, ...],
) -> BaseModelQuantizationReport:
    import bitsandbytes as bnb

    if model_family not in MODEL_INCLUDE_PATTERNS:
        raise BaseModelQuantizationError(
            f"unsupported quantized base model family {model_family!r}; expected one of {tuple(MODEL_INCLUDE_PATTERNS)!r}"
        )
    validate_base_model_quantization_runtime(
        mode,
        compute_dtype_name,
        torch.cuda.is_available(),
        bnb.__version__,
    )
    family = model_family
    include_patterns = MODEL_INCLUDE_PATTERNS[family]
    skip_patterns = MODEL_SKIP_PATTERNS[family] + user_skip_module_patterns
    return quantize_frozen_linear_layers(
        model,
        mode,
        compute_dtype_from_name(compute_dtype_name),
        include_patterns,
        skip_patterns,
    )


def quantize_text_encoder_for_lora(
    text_encoder: torch.nn.Module,
    mode: str,
    compute_dtype_name: str,
    user_skip_module_patterns: tuple[str, ...],
) -> BaseModelQuantizationReport:
    import bitsandbytes as bnb

    validate_base_model_quantization_runtime(
        mode,
        compute_dtype_name,
        torch.cuda.is_available(),
        bnb.__version__,
    )
    default_skip_patterns = (
        "lm_head",
        "*.lm_head",
        "text_projection",
        "*.text_projection",
        "pooler.*",
        "*.pooler.*",
    )
    return quantize_frozen_linear_layers(
        text_encoder,
        mode,
        compute_dtype_from_name(compute_dtype_name),
        ("*",),
        default_skip_patterns + user_skip_module_patterns,
    )
