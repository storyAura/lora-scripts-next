# T-LoRA (Anima): Timestep-Dependent Low-Rank Adaptation — Anima model variant
# Based on https://github.com/ControlGenAI/T-LoRA
# Original work: Copyright (c) 2025 AIRI (https://airi.net)
# Licensed under the MIT License — see https://github.com/ControlGenAI/T-LoRA/blob/main/LICENSE
#
# Modifications for Anima/sd-scripts integration by lora-scripts-next contributors.

import ast
import os
from functools import partial
from typing import Dict, Optional

import torch

from library.utils import setup_logging
from networks import lora_anima as anima_lora
from networks.tlora import (
    TLoRAInfModule,
    TLoRAModule,
    _normalize_schedule,
    _parse_bool_arg,
    _parse_positive_int_arg,
)

setup_logging()
import logging

logger = logging.getLogger(__name__)

class TLoRAAnimaNetwork(anima_lora.LoRANetwork):
    TRAIN_NORM_PREFIX_ANIMA = anima_lora.LoRANetwork.LORA_PREFIX_ANIMA
    TRAIN_NORM_PREFIX_TEXT_ENCODER = anima_lora.LoRANetwork.LORA_PREFIX_TEXT_ENCODER

    def __init__(
        self,
        text_encoders,
        unet,
        *args,
        tlora_min_rank: Optional[int] = None,
        tlora_rank_schedule: Optional[str] = None,
        tlora_orthogonal_init: bool = False,
        train_norm: bool = False,
        module_class=None,
        **kwargs,
    ):
        self.current_timestep = None
        self.train_norm = train_norm
        self.tlora_min_rank = _parse_positive_int_arg(
            tlora_min_rank if tlora_min_rank is not None else 1,
            "tlora_min_rank",
        )
        self.tlora_rank_schedule = _normalize_schedule(tlora_rank_schedule)
        self.tlora_orthogonal_init = _parse_bool_arg(tlora_orthogonal_init, default=False)

        base_module_class = TLoRAModule if module_class is None else module_class
        module_class = partial(
            base_module_class,
            tlora_min_rank=self.tlora_min_rank,
            tlora_rank_schedule=self.tlora_rank_schedule,
            tlora_orthogonal_init=self.tlora_orthogonal_init,
        )
        module_class.supports_conv2d = False

        # T-LoRA is defined for the diffusion transformer only. Creating
        # ordinary, non-timestep-aware text-encoder adapters under the same
        # checkpoint would silently mix two algorithms.
        super().__init__(None, unet, *args, module_class=module_class, **kwargs)
        self.adapter_type = "tlora"

        if not hasattr(self, "text_encoder_norms"):
            self.text_encoder_norms = []
        if not hasattr(self, "unet_norms"):
            self.unet_norms = []

    def set_current_timestep(self, timestep):
        self.current_timestep = timestep

    def clear_current_timestep(self):
        self.current_timestep = None

    def is_mergeable(self):
        return False

    def merge_to(self, text_encoders, unet, weights_sd, dtype=None, device=None):
        raise RuntimeError(
            "T-LoRA cannot be merged into one static weight because its "
            "effective rank depends on the current diffusion timestep"
        )

    def save_weights(self, file, dtype, metadata):
        owned_metadata = dict(metadata or {})
        owned_metadata["ss_adapter_algorithm"] = "timestep_lora"
        owned_metadata["ss_tlora_min_rank"] = str(self.tlora_min_rank)
        owned_metadata["ss_tlora_rank_schedule"] = self.tlora_rank_schedule
        owned_metadata["ss_tlora_orthogonal_init"] = str(
            self.tlora_orthogonal_init
        ).lower()
        super().save_weights(file, dtype, owned_metadata)

    def apply_to(self, text_encoders, unet, apply_text_encoder=True, apply_unet=True):
        if apply_text_encoder:
            logger.info(f"enable {self.adapter_type.upper()} for text encoder: {len(self.text_encoder_loras)} modules")
            if self.train_norm and self.text_encoder_norms:
                logger.info(f"enable train_norm for text encoder: {len(self.text_encoder_norms)} modules")
        else:
            self.text_encoder_loras = []
            self.text_encoder_norms = []

        if apply_unet:
            logger.info(f"enable {self.adapter_type.upper()} for DiT: {len(self.unet_loras)} modules")
            if self.train_norm and self.unet_norms:
                logger.info(f"enable train_norm for DiT: {len(self.unet_norms)} modules")
        else:
            self.unet_loras = []
            self.unet_norms = []

        for lora in self.text_encoder_loras + self.unet_loras:
            if hasattr(lora, "set_network"):
                lora.set_network(self)
            lora.apply_to()
            self.add_module(lora.lora_name, lora)


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae,
    text_encoders: list,
    unet,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    del vae
    if network_dim is None:
        network_dim = 4
    if network_alpha is None:
        network_alpha = 1.0

    train_norm = _parse_bool_arg(kwargs.get("train_norm", None), default=False)

    train_llm_adapter = kwargs.get("train_llm_adapter", "false")
    train_llm_adapter = _parse_bool_arg(
        train_llm_adapter,
        default=False,
    )

    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)
        if not isinstance(exclude_patterns, list):
            exclude_patterns = [exclude_patterns]

    if train_norm:
        exclude_patterns.append(r".*(_modulation|_embedder|final_layer).*")
    else:
        exclude_patterns.append(r".*(_modulation|_norm|_embedder|final_layer).*")

    include_patterns = kwargs.get("include_patterns", None)
    if include_patterns is not None:
        include_patterns = ast.literal_eval(include_patterns)
        if not isinstance(include_patterns, list):
            include_patterns = [include_patterns]

    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    tlora_min_rank = _parse_positive_int_arg(
        kwargs.get("tlora_min_rank", 1),
        "tlora_min_rank",
    )
    tlora_rank_schedule = _normalize_schedule(
        kwargs.get("tlora_rank_schedule", "cosine")
    )
    tlora_orthogonal_init = _parse_bool_arg(kwargs.get("tlora_orthogonal_init", False), default=False)

    verbose = _parse_bool_arg(
        kwargs.get("verbose", "false"),
        default=False,
    )

    def parse_kv_pairs(kv_pair_str: str, is_int: bool) -> Dict[str, float]:
        pairs = {}
        for pair in kv_pair_str.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                raise ValueError(
                    f"Invalid network override {pair!r}; expected 'key=value'"
                )
            key, value = pair.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                raise ValueError(
                    f"Invalid network override {pair!r}; key must not be empty"
                )
            try:
                pairs[key] = int(value) if is_int else float(value)
            except ValueError as error:
                raise ValueError(
                    f"Invalid network override value for {key!r}: {value!r}"
                ) from error
        return pairs

    network_reg_lrs = kwargs.get("network_reg_lrs", None)
    reg_lrs = parse_kv_pairs(network_reg_lrs, is_int=False) if network_reg_lrs is not None else None

    network_reg_dims = kwargs.get("network_reg_dims", None)
    reg_dims = parse_kv_pairs(network_reg_dims, is_int=True) if network_reg_dims is not None else None

    network = TLoRAAnimaNetwork(
        text_encoders,
        unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        train_llm_adapter=train_llm_adapter,
        exclude_patterns=exclude_patterns,
        include_patterns=include_patterns,
        reg_dims=reg_dims,
        reg_lrs=reg_lrs,
        verbose=verbose,
        train_norm=train_norm,
        tlora_min_rank=tlora_min_rank,
        tlora_rank_schedule=tlora_rank_schedule,
        tlora_orthogonal_init=tlora_orthogonal_init,
    )

    loraplus_lr_ratio = kwargs.get("loraplus_lr_ratio", None)
    loraplus_unet_lr_ratio = kwargs.get("loraplus_unet_lr_ratio", None)
    loraplus_text_encoder_lr_ratio = kwargs.get("loraplus_text_encoder_lr_ratio", None)
    loraplus_lr_ratio = float(loraplus_lr_ratio) if loraplus_lr_ratio is not None else None
    loraplus_unet_lr_ratio = float(loraplus_unet_lr_ratio) if loraplus_unet_lr_ratio is not None else None
    loraplus_text_encoder_lr_ratio = float(loraplus_text_encoder_lr_ratio) if loraplus_text_encoder_lr_ratio is not None else None
    if loraplus_lr_ratio is not None or loraplus_unet_lr_ratio is not None or loraplus_text_encoder_lr_ratio is not None:
        network.set_loraplus_lr_ratio(loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio)

    return network


def create_network_from_weights(multiplier, file, ae, text_encoders, unet, weights_sd=None, for_inference=False, **kwargs):
    del ae
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    modules_dim = {}
    modules_alpha = {}
    train_llm_adapter = False
    train_norm = False

    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_down" in key:
            modules_dim[lora_name] = value.size()[0]

        if "llm_adapter" in lora_name:
            train_llm_adapter = True

    for key in modules_dim.keys():
        if key not in modules_alpha:
            modules_alpha[key] = modules_dim[key]

    stored_min_rank = next(
        (
            int(value.item())
            for key, value in weights_sd.items()
            if key.endswith(".tlora_min_rank_state")
        ),
        1,
    )
    stored_schedule = next(
        (
            "linear" if int(value.item()) == 0 else "cosine"
            for key, value in weights_sd.items()
            if key.endswith(".tlora_rank_schedule_state")
        ),
        "cosine",
    )
    tlora_min_rank = _parse_positive_int_arg(
        kwargs.get("tlora_min_rank", stored_min_rank),
        "tlora_min_rank",
    )
    tlora_rank_schedule = _normalize_schedule(
        kwargs.get("tlora_rank_schedule", stored_schedule)
    )
    tlora_orthogonal_init = _parse_bool_arg(kwargs.get("tlora_orthogonal_init", False), default=False)

    module_class = TLoRAInfModule if for_inference else None

    network = TLoRAAnimaNetwork(
        text_encoders,
        unet,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        train_llm_adapter=train_llm_adapter,
        train_norm=train_norm,
        tlora_min_rank=tlora_min_rank,
        tlora_rank_schedule=tlora_rank_schedule,
        tlora_orthogonal_init=tlora_orthogonal_init,
    )
    return network, weights_sd
