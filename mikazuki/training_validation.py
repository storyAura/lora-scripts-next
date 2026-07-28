from __future__ import annotations

import math
from collections.abc import Mapping


ANIMA_LORA_TRAIN_TYPES = frozenset({"anima-lora", "sd3-lora"})
# "tglokr" (time-gated GLoKR) was removed; a stale saved config must fail loudly
# instead of silently training plain GLoKR under the old name.
UNIMPLEMENTED_ANIMA_ADAPTER_TYPES: frozenset[str] = frozenset({"tglokr"})


class TrainingConfigurationError(ValueError):
    def __init__(self, field: str, value: object, reason: str) -> None:
        self.field = field
        self.value = value
        self.reason = reason
        super().__init__(f"{field}={value!r} is invalid: {reason}")


def _is_truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _finite_number(
    config: Mapping[str, object],
    field: str,
    default: float,
) -> float:
    raw = config.get(field, default)
    if raw in (None, ""):
        raw = default
    try:
        parsed = float(raw)
    except (TypeError, ValueError) as error:
        raise TrainingConfigurationError(
            field,
            raw,
            "expected a finite number",
        ) from error
    if not math.isfinite(parsed):
        raise TrainingConfigurationError(
            field,
            raw,
            "expected a finite number",
        )
    return parsed


def _integer(
    config: Mapping[str, object],
    field: str,
    default: int,
) -> int:
    raw = config.get(field, default)
    if raw in (None, ""):
        raw = default
    if isinstance(raw, bool):
        raise TrainingConfigurationError(
            field,
            raw,
            "expected an integer",
        )
    try:
        parsed = int(raw)
        numeric = float(raw)
    except (TypeError, ValueError) as error:
        raise TrainingConfigurationError(
            field,
            raw,
            "expected an integer",
        ) from error
    if not math.isfinite(numeric) or numeric != parsed:
        raise TrainingConfigurationError(
            field,
            raw,
            "expected an integer",
        )
    return parsed


def _boolean(
    config: Mapping[str, object],
    field: str,
    default: bool,
) -> bool:
    raw = config.get(field, default)
    if raw in (None, ""):
        return default
    if isinstance(raw, bool):
        return raw
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise TrainingConfigurationError(
        field,
        raw,
        "expected a boolean",
    )


def validate_training_configuration(
    config: Mapping[str, object],
    model_train_type: str,
) -> None:
    attn_mode = str(config.get("attn_mode") or "").strip().lower()
    if attn_mode == "sageattn":
        raise TrainingConfigurationError(
            "attn_mode",
            config.get("attn_mode"),
            "standard SageAttention does not support training backward",
        )

    normalized_train_type = str(model_train_type).strip().lower()
    checkpointing_mode = str(
        config.get("anima_gradient_checkpointing_mode") or "standard"
    ).strip().lower()
    if checkpointing_mode not in {"standard", "selective"}:
        raise TrainingConfigurationError(
            "anima_gradient_checkpointing_mode",
            config.get("anima_gradient_checkpointing_mode"),
            "expected 'standard' or 'selective'",
        )
    if checkpointing_mode == "selective":
        if not _is_truthy(config.get("gradient_checkpointing")):
            raise TrainingConfigurationError(
                "anima_gradient_checkpointing_mode",
                config.get("anima_gradient_checkpointing_mode"),
                "selective mode requires gradient_checkpointing=true",
            )
        conflict_fields = (
            "cpu_offload_checkpointing",
            "unsloth_offload_checkpointing",
        )
        enabled_conflicts = tuple(field for field in conflict_fields if _is_truthy(config.get(field)))
        if enabled_conflicts:
            raise TrainingConfigurationError(
                "anima_gradient_checkpointing_mode",
                config.get("anima_gradient_checkpointing_mode"),
                f"selective mode cannot be combined with {enabled_conflicts!r}",
            )
        if config.get("blocks_to_swap") not in (None, "", 0, "0"):
            raise TrainingConfigurationError(
                "anima_gradient_checkpointing_mode",
                config.get("anima_gradient_checkpointing_mode"),
                "selective mode cannot be combined with blocks_to_swap",
            )

    quantization_mode = str(config.get("base_model_quantization") or "none").strip().lower()
    if quantization_mode not in {"none", "int8", "nf4"}:
        raise TrainingConfigurationError(
            "base_model_quantization",
            config.get("base_model_quantization"),
            "expected one of 'none', 'int8', or 'nf4'",
        )
    quantization_compute_dtype = str(
        config.get("base_model_quantization_compute_dtype") or "bf16"
    ).strip().lower()
    if quantization_compute_dtype not in {"fp16", "bf16"}:
        raise TrainingConfigurationError(
            "base_model_quantization_compute_dtype",
            config.get("base_model_quantization_compute_dtype"),
            "expected 'fp16' or 'bf16'",
        )
    skip_modules = config.get("base_model_quantization_skip_modules")
    if skip_modules is not None and (
        not isinstance(skip_modules, (list, tuple))
        or any(not isinstance(item, str) or not item.strip() for item in skip_modules)
    ):
        raise TrainingConfigurationError(
            "base_model_quantization_skip_modules",
            skip_modules,
            "expected a list of non-empty module-name glob strings",
        )
    if _is_truthy(config.get("quantize_text_encoder")) and quantization_mode == "none":
        raise TrainingConfigurationError(
            "quantize_text_encoder",
            config.get("quantize_text_encoder"),
            "requires base_model_quantization='int8' or 'nf4'",
        )
    if quantization_mode != "none":
        if normalized_train_type == "anima-finetune":
            raise TrainingConfigurationError(
                "base_model_quantization",
                config.get("base_model_quantization"),
                "frozen-base quantization supports adapter training only, not full fine-tuning",
            )
        if _is_truthy(config.get("fp8_base")) or _is_truthy(config.get("fp8_base_unet")):
            raise TrainingConfigurationError(
                "base_model_quantization",
                config.get("base_model_quantization"),
                "cannot be combined with fp8_base or fp8_base_unet",
            )
        if config.get("blocks_to_swap") not in (None, "", 0, "0"):
            raise TrainingConfigurationError(
                "base_model_quantization",
                config.get("base_model_quantization"),
                "cannot be combined with blocks_to_swap",
            )
        lora_type = str(config.get("lora_type") or "").strip().lower()
        quantized_adapter_types = {
            "",
            "lora",
            "lora_plus",
            "lora_fa",
            "vera",
        }
        if lora_type not in quantized_adapter_types or _is_truthy(config.get("pissa_init")):
            raise TrainingConfigurationError(
                "lora_type",
                config.get("lora_type"),
                "the selected adapter has no declared quantized-weight support; "
                "PiSSA, DoRA, LyCORIS, and custom algorithms require a full-precision base",
            )
    if _is_truthy(config.get("anima_compile_blocks")):
        compile_backend = str(config.get("anima_compile_backend") or "inductor").strip().lower()
        if compile_backend != "inductor":
            raise TrainingConfigurationError(
                "anima_compile_backend",
                config.get("anima_compile_backend"),
                "regional Anima compile supports only 'inductor'",
            )
        if _is_truthy(config.get("torch_compile")):
            raise TrainingConfigurationError(
                "anima_compile_blocks",
                config.get("anima_compile_blocks"),
                "regional compilation cannot be combined with global torch_compile",
            )
        if quantization_mode != "none":
            raise TrainingConfigurationError(
                "anima_compile_blocks",
                config.get("anima_compile_blocks"),
                "regional compilation cannot be combined with base model quantization",
            )
        compile_conflicts = (
            "cpu_offload_checkpointing",
            "unsloth_offload_checkpointing",
        )
        enabled_compile_conflicts = tuple(
            field for field in compile_conflicts if _is_truthy(config.get(field))
        )
        if enabled_compile_conflicts:
            raise TrainingConfigurationError(
                "anima_compile_blocks",
                config.get("anima_compile_blocks"),
                f"regional compilation cannot be combined with {enabled_compile_conflicts!r}",
            )
        if config.get("blocks_to_swap") not in (None, "", 0, "0"):
            raise TrainingConfigurationError(
                "anima_compile_blocks",
                config.get("anima_compile_blocks"),
                "regional compilation cannot be combined with blocks_to_swap",
            )
    if _is_truthy(config.get("fsdp2_frozen_base")):
        if _is_truthy(config.get("deepspeed")):
            raise TrainingConfigurationError(
                "fsdp2_frozen_base",
                config.get("fsdp2_frozen_base"),
                "cannot be combined with DeepSpeed",
            )
        if _is_truthy(config.get("torch_compile")):
            raise TrainingConfigurationError(
                "fsdp2_frozen_base",
                config.get("fsdp2_frozen_base"),
                "cannot be combined with global torch_compile",
            )
        if config.get("blocks_to_swap") not in (None, "", 0, "0"):
            raise TrainingConfigurationError(
                "fsdp2_frozen_base",
                config.get("fsdp2_frozen_base"),
                "cannot be combined with blocks_to_swap",
            )
        if quantization_mode != "none":
            raise TrainingConfigurationError(
                "fsdp2_frozen_base",
                config.get("fsdp2_frozen_base"),
                "cannot be combined with base model quantization",
            )
        if _is_truthy(config.get("anima_compile_blocks")):
            raise TrainingConfigurationError(
                "fsdp2_frozen_base",
                config.get("fsdp2_frozen_base"),
                "cannot be combined with Anima regional compile",
            )
        if _is_truthy(config.get("network_train_text_encoder_only")):
            raise TrainingConfigurationError(
                "fsdp2_frozen_base",
                config.get("fsdp2_frozen_base"),
                "requires DiT/U-Net adapter training, not text_encoder_only",
            )
        if config.get("network_train_unet_only") is False:
            raise TrainingConfigurationError(
                "fsdp2_frozen_base",
                config.get("fsdp2_frozen_base"),
                "requires network_train_unet_only=true",
            )
        fsdp_lora_type = str(config.get("lora_type") or "").strip().lower()
        if fsdp_lora_type not in {"", "lora"} or _is_truthy(config.get("pissa_init")):
            raise TrainingConfigurationError(
                "fsdp2_frozen_base",
                config.get("fsdp2_frozen_base"),
                "currently supports standard LoRA only",
            )

    if normalized_train_type not in ANIMA_LORA_TRAIN_TYPES:
        return

    lora_type = str(config.get("lora_type") or "").strip().lower()
    if lora_type in UNIMPLEMENTED_ANIMA_ADAPTER_TYPES:
        raise TrainingConfigurationError(
            "lora_type",
            config.get("lora_type"),
            f"Anima {lora_type} is not available; refusing to silently train a different algorithm",
        )

    if _is_truthy(config.get("pissa_init")) and lora_type not in {"", "lora"}:
        raise TrainingConfigurationError(
            "pissa_init",
            config.get("pissa_init"),
            "Anima PiSSA initialization is supported only when lora_type='lora'",
        )

    nonstandard_norm_types = {
        "delora",
        "waveft",
        "deft",
        "moslora",
    }
    if (
        lora_type in nonstandard_norm_types
        and _finite_number(config, "scale_weight_norms", 0.0) > 0
    ):
        raise TrainingConfigurationError(
            "scale_weight_norms",
            config.get("scale_weight_norms"),
            f"{lora_type} has no mathematically valid independent-factor "
            "max-norm rescaling",
        )

    if lora_type in {"delora", "waveft", "deft"}:
        rank_dropout = _finite_number(config, "rank_dropout", 0.0)
        if rank_dropout != 0:
            raise TrainingConfigurationError(
                "rank_dropout",
                config.get("rank_dropout"),
                f"{lora_type} does not expose LoRA rank components for dropout",
            )

    if lora_type == "delora":
        delora_lambda = _finite_number(config, "delora_lambda", 15.0)
        if delora_lambda <= 0:
            raise TrainingConfigurationError(
                "delora_lambda",
                config.get("delora_lambda"),
                "expected a positive value",
            )

    if lora_type == "waveft":
        n_frequency = _integer(
            config,
            "waveft_n_frequency",
            2592,
        )
        if n_frequency <= 0:
            raise TrainingConfigurationError(
                "waveft_n_frequency",
                config.get("waveft_n_frequency"),
                "expected a positive integer",
            )
        seed = _integer(
            config,
            "waveft_random_loc_seed",
            777,
        )
        if seed < 0:
            raise TrainingConfigurationError(
                "waveft_random_loc_seed",
                config.get("waveft_random_loc_seed"),
                "expected a non-negative integer",
            )
        _finite_number(config, "waveft_scaling", 25.0)
        _boolean(config, "waveft_use_idwt", True)
        family = str(
            config.get("waveft_wavelet_family") or "db1"
        ).strip().lower()
        if family not in {"db1", "haar"}:
            raise TrainingConfigurationError(
                "waveft_wavelet_family",
                config.get("waveft_wavelet_family"),
                "this integration currently supports only db1/haar",
            )

    if lora_type == "deft":
        method = str(
            config.get("deft_decomposition_method") or "qr"
        ).strip().lower()
        if method not in {"qr", "relu"}:
            raise TrainingConfigurationError(
                "deft_decomposition_method",
                config.get("deft_decomposition_method"),
                "expected 'qr' or 'relu'",
            )
        alpha = _finite_number(config, "deft_alpha", 0.0)
        if alpha < 0:
            raise TrainingConfigurationError(
                "deft_alpha",
                config.get("deft_alpha"),
                "expected zero (no scaling) or a positive value",
            )
        init_scale = _finite_number(config, "deft_init_scale", 1.0)
        if init_scale <= 0:
            raise TrainingConfigurationError(
                "deft_init_scale",
                config.get("deft_init_scale"),
                "expected a positive value",
            )
        _boolean(config, "deft_init_weights", True)

    if lora_type == "moslora":
        initializer = str(
            config.get("moslora_mixer_init") or "kaiming"
        ).strip().lower()
        if initializer not in {"kaiming", "identity", "orthogonal"}:
            raise TrainingConfigurationError(
                "moslora_mixer_init",
                config.get("moslora_mixer_init"),
                "expected 'kaiming', 'identity', or 'orthogonal'",
            )

    if lora_type == "tlora":
        if _is_truthy(config.get("network_train_text_encoder_only")):
            raise TrainingConfigurationError(
                "network_train_text_encoder_only",
                config.get("network_train_text_encoder_only"),
                "timestep-dependent T-LoRA supports the diffusion transformer only",
            )
        if config.get("network_train_unet_only") is False:
            raise TrainingConfigurationError(
                "network_train_unet_only",
                config.get("network_train_unet_only"),
                "timestep-dependent T-LoRA requires DiT-only adapter training",
            )
        rank = _integer(config, "network_dim", 4)
        min_rank = _integer(config, "tlora_min_rank", 1)
        if rank <= 0:
            raise TrainingConfigurationError(
                "network_dim",
                config.get("network_dim"),
                "expected a positive integer",
            )
        if not 1 <= min_rank <= rank:
            raise TrainingConfigurationError(
                "tlora_min_rank",
                config.get("tlora_min_rank"),
                f"expected a value between 1 and network_dim ({rank})",
            )
        schedule = str(
            config.get("tlora_rank_schedule") or "cosine"
        ).strip().lower()
        if schedule not in {"cosine", "linear"}:
            raise TrainingConfigurationError(
                "tlora_rank_schedule",
                config.get("tlora_rank_schedule"),
                "expected 'cosine' or 'linear'",
            )
        _boolean(config, "tlora_orthogonal_init", False)
