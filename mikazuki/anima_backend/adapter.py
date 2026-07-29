from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from typing import Any

from mikazuki.training_validation import validate_training_configuration
from mikazuki.optimizer_configuration import normalize_optimizer_configuration
from mikazuki.utils.config_import import ANIMA_LORA_TYPE_BRANCH_CONSTS


SUPPORTED_FIELDS = {
    "pretrained_model_name_or_path",
    "vae",
    "qwen3",
    "llm_adapter_path",
    "t5_tokenizer_path",
    "resume",
    "qwen3_max_token_length",
    "t5_max_token_length",
    "timestep_sampling",
    "sigmoid_scale",
    "discrete_flow_shift",
    "weighting_scheme",
    "logit_mean",
    "logit_std",
    "mode_scale",
    "attn_mode",
    "split_attn",
    "vae_chunk_size",
    "vae_disable_cache",
    "unsloth_offload_checkpointing",
    "anima_gradient_checkpointing_mode",
    "anima_compile_blocks",
    "anima_compile_backend",
    "train_data_dir",
    "reg_data_dir",
    "resolution",
    "enable_bucket",
    "min_bucket_reso",
    "max_bucket_reso",
    "bucket_reso_steps",
    "output_dir",
    "output_name",
    "save_model_as",
    "save_precision",
    "save_every_n_epochs",
    "save_every_n_steps",
    "max_train_epochs",
    "max_train_steps",
    "train_batch_size",
    "gradient_checkpointing",
    "gradient_accumulation_steps",
    "network_train_unet_only",
    "network_train_text_encoder_only",
    "learning_rate",
    "unet_lr",
    "text_encoder_lr",
    "optimizer_type",
    "optimizer_args",
    "lr_scheduler",
    "lr_warmup_steps",
    "network_module",
    "network_weights",
    "network_dim",
    "network_alpha",
    "network_dropout",
    "network_args",
    "loraplus_lr_ratio",
    "loraplus_unet_lr_ratio",
    "loraplus_text_encoder_lr_ratio",
    "vera_projection_seed",
    "vera_save_projection",
    "vera_d_initial",
    "delora_lambda",
    "waveft_n_frequency",
    "waveft_scaling",
    "waveft_random_loc_seed",
    "waveft_use_idwt",
    "waveft_wavelet_family",
    "deft_decomposition_method",
    "deft_alpha",
    "deft_init_scale",
    "deft_init_weights",
    "moslora_mixer_init",
    "dim_from_weights",
    "scale_weight_norms",
    "train_norm",
    "full_matrix",
    "pissa_init",
    "pissa_method",
    "pissa_niter",
    "pissa_oversample",
    "pissa_apply_conv2d",
    "pissa_export_mode",
    "sample_prompts",
    "sample_at_first",
    "sample_every_n_epochs",
    "caption_extension",
    "shuffle_caption",
    "keep_tokens",
    "caption_tag_dropout_rate",
    "prefer_json_caption",
    "noise_offset",
    "multires_noise_iterations",
    "multires_noise_discount",
    "fp8_base",
    "fp8_base_unet",
    "base_model_quantization",
    "base_model_quantization_compute_dtype",
    "base_model_quantization_skip_modules",
    "quantize_text_encoder",
    "cache_latents",
    "cache_latents_to_disk",
    "cache_text_encoder_outputs",
    "cache_text_encoder_outputs_to_disk",
    "persistent_data_loader_workers",
    "max_data_loader_n_workers",
    "text_encoder_batch_size",
    "disable_mmap_load_safetensors",
    "blocks_to_swap",
    "cpu_offload_checkpointing",
    "fsdp2_frozen_base",
    "fsdp2_cpu_offload",
    "mixed_precision",
    "full_fp16",
    "full_bf16",
    "seed",
    "logging_dir",
    "log_with",
    "self_attn_lr",
    "cross_attn_lr",
    "mlp_lr",
    "mod_lr",
    "llm_adapter_lr",
}

NETWORK_ONLY_FIELDS = {
    "network_module",
    "network_weights",
    "network_dim",
    "network_alpha",
    "network_dropout",
    "network_args",
    "loraplus_lr_ratio",
    "loraplus_unet_lr_ratio",
    "loraplus_text_encoder_lr_ratio",
    "vera_projection_seed",
    "vera_save_projection",
    "vera_d_initial",
    "delora_lambda",
    "waveft_n_frequency",
    "waveft_scaling",
    "waveft_random_loc_seed",
    "waveft_use_idwt",
    "waveft_wavelet_family",
    "deft_decomposition_method",
    "deft_alpha",
    "deft_init_scale",
    "deft_init_weights",
    "moslora_mixer_init",
    "dim_from_weights",
    "scale_weight_norms",
    "train_norm",
    "full_matrix",
    "pissa_init",
    "pissa_method",
    "pissa_niter",
    "pissa_oversample",
    "pissa_apply_conv2d",
    "pissa_export_mode",
    "conv_dim",
    "conv_alpha",
    "lycoris_algo",
    "lokr_factor",
    "use_cp",
    "use_scalar",
    "decompose_both",
    "bypass_mode",
    "dora_wd",
    "rs_lora",
    "rank_dropout",
    "module_dropout",
    "rank_dropout_scale",
    "dropout",
    "tlora_min_rank",
    "tlora_rank_schedule",
    "tlora_orthogonal_init",
    "network_train_unet_only",
    "network_train_text_encoder_only",
    "enable_base_weight",
    "base_weights",
    "base_weights_multiplier",
    "use_sora",
    "sora_r",
    "sora_epsilon",
    "boft_constraint",
    "boft_rescaled",
    "cdka_r1",
    "cdka_r2",
    "cdka_r",
    "cdka_alpha",
}

UI_ONLY_FIELDS = {
    "model_train_type",
    "enable_preview",
    "positive_prompts",
    "negative_prompts",
    "sample_width",
    "sample_height",
    "sample_cfg",
    "sample_seed",
    "sample_steps",
    "sample_sampler",
    "sample_scheduler",
    "randomly_choice_prompt",
    "prompt_file",
    "enable_debug_options",
    "json_caption_hint",
    "lycoris_ext_hint",
    "lora_type",
}

# Top-level UI fields that should be injected into network_args for T-LoRA.
TLORA_NETWORK_ARG_FIELDS = {
    "tlora_min_rank",
    "tlora_rank_schedule",
    "tlora_orthogonal_init",
}

STANDARD_LORA_NETWORK_ARG_FIELDS = {
    "loraplus_lr_ratio",
    "loraplus_unet_lr_ratio",
    "loraplus_text_encoder_lr_ratio",
}

PISSA_NETWORK_ARG_FIELDS = {
    "pissa_init",
    "pissa_method",
    "pissa_niter",
    "pissa_oversample",
    "pissa_apply_conv2d",
    "pissa_export_mode",
}

VERA_NETWORK_ARG_FIELDS = {
    "vera_projection_seed",
    "vera_save_projection",
    "vera_d_initial",
}

DELORA_NETWORK_ARG_FIELDS = {
    "delora_lambda",
}

WAVEFT_NETWORK_ARG_FIELDS = {
    "waveft_n_frequency",
    "waveft_scaling",
    "waveft_random_loc_seed",
    "waveft_use_idwt",
    "waveft_wavelet_family",
}

DEFT_NETWORK_ARG_FIELDS = {
    "deft_decomposition_method",
    "deft_alpha",
    "deft_init_scale",
    "deft_init_weights",
}

MOSLORA_NETWORK_ARG_FIELDS = {
    "moslora_mixer_init",
}

# CDKA is a standalone networks.* module (never lycoris — the vendored tree is
# frozen); its branch fields travel as module network_args like every other
# standalone algorithm.
CDKA_NETWORK_ARG_FIELDS = {
    "cdka_r1",
    "cdka_r2",
    "cdka_r",
    "cdka_alpha",
    "bypass_mode",
    "rank_dropout",
    "module_dropout",
    "rank_dropout_scale",
}

ANIMA_NETWORK_MODULE_ARG_FIELDS = {
    "networks.lora_anima": (
        STANDARD_LORA_NETWORK_ARG_FIELDS | PISSA_NETWORK_ARG_FIELDS
    ),
    "networks.vera_anima": VERA_NETWORK_ARG_FIELDS,
    "networks.delora_anima": DELORA_NETWORK_ARG_FIELDS,
    "networks.waveft_anima": WAVEFT_NETWORK_ARG_FIELDS,
    "networks.deft_anima": DEFT_NETWORK_ARG_FIELDS,
    "networks.moslora_anima": MOSLORA_NETWORK_ARG_FIELDS,
    "networks.cdka_anima": CDKA_NETWORK_ARG_FIELDS,
}

EXPLICIT_BOOLEAN_NETWORK_ARG_FIELDS = {
    "vera_save_projection",
    "waveft_use_idwt",
    "deft_init_weights",
}

# LyCORIS UI fields → network_args key names.  sd-scripts only forwards
# network_args items to lycoris.kohya.create_network(**kwargs); top-level
# TOML keys are silently ignored.  Map UI field → LyCORIS kwarg name.
LYCORIS_NETWORK_ARG_MAP: dict[str, str] = {
    "lycoris_algo": "algo",
    "lokr_factor": "factor",
    "conv_dim": "conv_dim",
    "conv_alpha": "conv_alpha",
    "use_cp": "use_cp",
    "use_scalar": "use_scalar",
    "decompose_both": "decompose_both",
    "bypass_mode": "bypass_mode",
    "dora_wd": "dora_wd",
    "rs_lora": "rs_lora",
    "full_matrix": "full_matrix",
    "rank_dropout": "rank_dropout",
    "module_dropout": "module_dropout",
    "rank_dropout_scale": "rank_dropout_scale",
    "train_norm": "train_norm",
    "dropout": "dropout",
    # 本地扩展算法专属字段（gsokr / glora_boft）。
    # 布尔字段的 UI 默认值只允许 lycoris 侧默认为 False 的参数设为 true，
    # 因为 _is_empty_value 会把 False 丢弃（不发送 = lycoris 默认值）。
    "use_sora": "use_sora",
    "sora_r": "sora_r",
    "sora_epsilon": "sora_epsilon",
    "boft_constraint": "constraint",
    "boft_rescaled": "rescaled",
}

LOKR_TRAIN_NORM_WARNING = (
    "LyCORIS train_norm is disabled for Anima LoKr because LyCORIS NormModule "
    "can crash on Anima norm layers without affine weights during preview sampling."
)
LOKR_BF16_DORA_WARNING = (
    "Anima LoKr mixed_precision=bf16 with DoRA/weight_decomposition can be less "
    "stable on some LyCORIS/PyTorch combinations. The trainer keeps your "
    "dora_wd/weight_decomposition settings unchanged."
)
LOKR_FULL_MATRIX_WARNING = (
    "Anima LoKr full_matrix=true is a high-risk stability mode. Consider "
    "disabling full_bf16/full_fp16 and setting scale_weight_norms=1 if the first "
    "epoch becomes unstable. The trainer keeps your parameters unchanged."
)


def _is_empty_value(value: Any) -> bool:
    """Check if a value is empty/invalid (None, NaN, 'undefined', 'null', '')."""
    if value is None or value is False:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "undefined", "null", "nan"}:
        return True
    return False


def _normalize_network_args(values: Any) -> list[str]:
    """
    Normalize network_args from UI payload:
    - keep string items only
    - drop empty / malformed items
    - drop `key=undefined` and `key=null`
    - for duplicate keys, keep the last value (so custom args override earlier defaults)
    """
    if not isinstance(values, list):
        return []

    ordered: list[str] = []
    key_index: dict[str, int] = {}

    for raw in values:
        if not isinstance(raw, str):
            continue
        item = raw.strip()
        if not item or "=" not in item:
            continue

        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value.lower() in {"undefined", "null", "nan"}:
            continue

        normalized = f"{key}={value}"
        if key in key_index:
            ordered[key_index[key]] = normalized
        else:
            key_index[key] = len(ordered)
            ordered.append(normalized)

    return ordered


def _apply_lr_fallback(source: dict[str, Any]) -> None:
    learning_rate = source.get("learning_rate")
    if _is_empty_value(learning_rate):
        return
    for key in ("unet_lr", "text_encoder_lr"):
        if _is_empty_value(source.get(key)):
            source[key] = learning_rate


def _apply_lora_type_overrides(source: dict[str, Any], warnings: list[str]) -> None:
    """Derive network_module / lycoris_algo from lora_type, ignoring stale form values.

    The schema keeps these fields tolerant (a leftover value from another
    lora_type branch must not break the frontend union), so the adapter is the
    authority: whatever the form carried, training always gets the module and
    algo that belong to the selected lora_type.
    """
    lora_type = str(source.get("lora_type") or "").strip().lower()
    consts = ANIMA_LORA_TYPE_BRANCH_CONSTS.get(lora_type)
    if not consts:
        return

    expected_module = consts["network_module"]
    current_module = str(source.get("network_module") or "").strip()
    if current_module and current_module != expected_module:
        warnings.append(
            f"network_module={current_module} 与 lora_type={lora_type} 不符，已改为 {expected_module}"
        )
    source["network_module"] = expected_module

    expected_algo = consts.get("lycoris_algo")
    if expected_algo:
        current_algo = str(source.get("lycoris_algo") or "").strip().lower()
        if current_algo and current_algo != expected_algo:
            warnings.append(
                f"lycoris_algo={current_algo} 与 lora_type={lora_type} 不符，已改为 {expected_algo}"
            )
        source["lycoris_algo"] = expected_algo
    else:
        source.pop("lycoris_algo", None)

    if lora_type == "rslora":
        source["rs_lora"] = True
        source.pop("dora_wd", None)
    elif lora_type == "dora":
        source["dora_wd"] = True
        source.pop("rs_lora", None)
    elif lora_type == "lora_fa":
        requested_optimizer = str(source.get("optimizer_type") or "").strip()
        if requested_optimizer.lower() != "lorafaadamw":
            if requested_optimizer:
                warnings.append(
                    f"LoRA-FA requires LoRAFAAdamW; replaced optimizer_type={requested_optimizer}"
                )
            source["optimizer_type"] = "LoRAFAAdamW"


def _network_args_use_lokr(network_args: list[str]) -> bool:
    for item in network_args:
        if not isinstance(item, str) or "=" not in item:
            continue
        key, value = item.split("=", 1)
        if key.strip().lower() == "algo" and value.strip().lower() == "lokr":
            return True
    return False


def _network_args_has_truthy_arg(network_args: list[str], arg_key: str) -> bool:
    target = arg_key.strip().lower()
    for item in network_args:
        if not isinstance(item, str) or "=" not in item:
            continue
        key, value = item.split("=", 1)
        if key.strip().lower() != target:
            continue
        if str(value).strip().lower() in {"1", "true", "yes", "on"}:
            return True
    return False


def _strip_arg(network_args: list[str], arg_key: str) -> tuple[list[str], bool]:
    stripped: list[str] = []
    removed = False
    target = arg_key.strip().lower()
    for item in network_args:
        if isinstance(item, str) and "=" in item:
            key, _value = item.split("=", 1)
            if key.strip().lower() == target:
                removed = True
                continue
        stripped.append(item)
    return stripped, removed


def adapt_anima_config(
    config: dict[str, Any], *, finetune: bool = False
) -> tuple[dict[str, Any], list[str]]:
    optimizer_config = normalize_optimizer_configuration(deepcopy(config))
    source = optimizer_config.values
    adapted: dict[str, Any] = {}
    warnings = list(optimizer_config.warnings)
    model_train_type = "anima-finetune" if finetune else "anima-lora"
    validate_training_configuration(source, model_train_type)

    # removed T-GLoKR (time-gated GLoKR): stale fields from old autosaves/history
    # are dropped silently instead of leaking into the TOML as unknown keys
    source.pop("train_time_gates", None)
    source.pop("time_gate_dim", None)

    # removed GLoKR experimental fields (multi-term Kronecker sum + BoRA
    # coupling, 2026-07-29): stale fields from old autosaves/history fall back
    # to the vendored defaults (kron_rank=1, use_bora=False, bora_iters=1)
    source.pop("kron_rank", None)
    source.pop("use_bora", None)
    source.pop("bora_iters", None)

    # GLoKR removed from the GUI (2026-07-29; the vendored module remains for
    # CLI / legacy archives): its branch-exclusive fields from old
    # autosaves/history are dropped silently instead of leaking into
    # network_args or "unknown field" warnings
    source.pop("train_gates", None)
    source.pop("init_mode", None)
    source.pop("use_g_out", None)
    source.pop("g_norm_mode", None)

    if finetune:
        for key in list(source):
            if key in NETWORK_ONLY_FIELDS:
                source.pop(key, None)
        source.pop("network_args_custom", None)
        source.pop("lora_type", None)

    custom_network_args = source.pop("network_args_custom", None)
    if not finetune:
        merged_network_args: list[str] = []
        if isinstance(source.get("network_args"), list):
            merged_network_args.extend(source["network_args"])
        if isinstance(custom_network_args, list):
            merged_network_args.extend(custom_network_args)
        normalized_network_args = _normalize_network_args(merged_network_args)
        if normalized_network_args:
            source["network_args"] = normalized_network_args
        elif "network_args" in source:
            source.pop("network_args", None)

        _apply_lr_fallback(source)
        _apply_lora_type_overrides(source, warnings)

    # LyCORIS default preset does not include Anima module class names, which may
    # produce zero trainable modules for LoKr. Inject Anima-specific preset unless
    # user already provided one via network_args.
    if not finetune and source.get("network_module") == "lycoris.kohya":
        network_args = source.get("network_args")
        has_preset = isinstance(network_args, list) and any(
            isinstance(item, str) and item.strip().startswith("preset=")
            for item in network_args
        )
        if not has_preset:
            preset_path = (
                Path(__file__).resolve().parents[2] / "config" / "lycoris_anima_preset.toml"
            )
            source["network_args"] = list(network_args or []) + [
                f"preset={preset_path.as_posix()}"
            ]

    # LyCORIS: convert top-level UI fields into network_args.  sd-scripts only
    # passes network_args items (as **kwargs) to lycoris.kohya.create_network();
    # top-level TOML keys like use_cp, decompose_both, etc. are silently lost.
    if not finetune and source.get("network_module") == "lycoris.kohya":
        network_args = list(source.get("network_args") or [])
        for ui_field, arg_key in LYCORIS_NETWORK_ARG_MAP.items():
            value = source.pop(ui_field, None)
            if _is_empty_value(value):
                continue
            network_args.append(f"{arg_key}={value}")
        if _network_args_use_lokr(network_args):
            network_args, removed_train_norm = _strip_arg(network_args, "train_norm")
            if removed_train_norm:
                warnings.append(LOKR_TRAIN_NORM_WARNING)
            if _network_args_has_truthy_arg(network_args, "full_matrix"):
                warnings.append(LOKR_FULL_MATRIX_WARNING)
            if (
                str(source.get("mixed_precision", "")).strip().lower() == "bf16"
                and (
                    _network_args_has_truthy_arg(network_args, "dora_wd")
                    or _network_args_has_truthy_arg(network_args, "weight_decomposition")
                )
            ):
                warnings.append(LOKR_BF16_DORA_WARNING)
        if network_args:
            source["network_args"] = network_args

    # T-LoRA: convert top-level UI fields into network_args so sd-scripts
    # can forward them to create_network() as **kwargs.
    if not finetune and source.get("network_module") == "networks.tlora_anima":
        network_args = list(source.get("network_args") or [])
        for field in TLORA_NETWORK_ARG_FIELDS:
            value = source.pop(field, None)
            if not _is_empty_value(value):
                network_args.append(f"{field}={value}")
        if network_args:
            source["network_args"] = network_args

    network_module = str(source.get("network_module") or "")
    network_arg_fields = ANIMA_NETWORK_MODULE_ARG_FIELDS.get(network_module)
    if not finetune and network_arg_fields is not None:
        network_args = list(source.get("network_args") or [])
        for field in network_arg_fields:
            value = source.pop(field, None)
            if field in EXPLICIT_BOOLEAN_NETWORK_ARG_FIELDS:
                if value is not None:
                    network_args.append(f"{field}={value}")
            elif not _is_empty_value(value):
                network_args.append(f"{field}={value}")
        if network_args:
            source["network_args"] = network_args

    for key, value in source.items():
        if (
            key in UI_ONLY_FIELDS
            or key in TLORA_NETWORK_ARG_FIELDS
            or key in STANDARD_LORA_NETWORK_ARG_FIELDS
            or key in PISSA_NETWORK_ARG_FIELDS
            or key in VERA_NETWORK_ARG_FIELDS
            or key in DELORA_NETWORK_ARG_FIELDS
            or key in WAVEFT_NETWORK_ARG_FIELDS
            or key in DEFT_NETWORK_ARG_FIELDS
            or key in MOSLORA_NETWORK_ARG_FIELDS
            or key in CDKA_NETWORK_ARG_FIELDS
            or key in LYCORIS_NETWORK_ARG_MAP
        ):
            continue
        if _is_empty_value(value):
            continue
        if key in SUPPORTED_FIELDS:
            if key == "attn_mode" and value in ("", None):
                continue
            adapted[key] = value
            continue
        if key.startswith("anima_"):
            warnings.append(f"Unsupported Anima field ignored: {key}")
            continue
        warnings.append(f"Unknown field passed through to sd-scripts: {key}")
        adapted[key] = value

    return adapted, warnings
