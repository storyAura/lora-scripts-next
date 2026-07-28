# T-LoRA: Timestep-Dependent Low-Rank Adaptation for Diffusion Models
# Adapted from https://github.com/ControlGenAI/T-LoRA
# Original work: Copyright (c) 2025 AIRI (https://airi.net)
# Licensed under the MIT License — see https://github.com/ControlGenAI/T-LoRA/blob/main/LICENSE
#
# Modifications for Anima/sd-scripts integration by lora-scripts-next contributors.

import math
import os
import weakref
from functools import partial
from typing import Dict, List, Optional, Union

from diffusers import AutoencoderKL
from transformers import CLIPTextModel
import torch

from library.sdxl_original_unet import SdxlUNet2DConditionModel
from library.utils import setup_logging
from networks import lora as lora_network

setup_logging()
import logging

logger = logging.getLogger(__name__)


def _parse_bool_arg(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return value == 1
        raise ValueError(
            f"boolean value must be 0 or 1, received {value!r}"
        )
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value {value!r}")


def _normalize_schedule(value) -> str:
    schedule = str(value or "cosine").strip().lower() or "cosine"
    if schedule not in {"linear", "cosine"}:
        raise ValueError(
            "tlora_rank_schedule must be 'linear' or 'cosine', "
            f"received {value!r}"
        )
    return schedule


def _parse_positive_int_arg(value, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(
            f"{field_name} must be a positive integer, received {value!r}"
        )
    try:
        parsed = int(value)
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field_name} must be a positive integer, received {value!r}"
        ) from error
    if not math.isfinite(numeric) or numeric != parsed or parsed <= 0:
        raise ValueError(
            f"{field_name} must be a positive integer, received {value!r}"
        )
    return parsed


def _parse_probability_arg(value, field_name: str):
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field_name} must be in [0, 1), received {value!r}"
        ) from error
    if not math.isfinite(parsed) or not 0 <= parsed < 1:
        raise ValueError(
            f"{field_name} must be in [0, 1), received {value!r}"
        )
    return parsed


def _broadcast_rank_mask(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if reference.ndim <= 2:
        return values

    rank_dim = int(values.shape[1])

    # For Linear-style tensors, LoRA rank lives on the last axis:
    # e.g. [B, S, R], [B, T, H, W, R].
    if int(reference.shape[-1]) == rank_dim:
        view_shape = [int(values.shape[0])] + [1] * (reference.ndim - 2) + [rank_dim]
        return values.reshape(view_shape)

    # For Conv-style tensors, LoRA rank is channel-first:
    # e.g. [B, R, H, W], [B, R, T, H, W].
    if int(reference.shape[1]) == rank_dim:
        view_shape = [int(values.shape[0]), rank_dim] + [1] * (reference.ndim - 2)
        return values.reshape(view_shape)

    # Fallback for uncommon layouts.
    while values.ndim < reference.ndim:
        values = values.unsqueeze(-1)
    return values


class TLoRAModule(lora_network.LoRAModule):
    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        tlora_min_rank: Optional[int] = None,
        tlora_rank_schedule: Optional[str] = None,
        tlora_orthogonal_init: bool = False,
    ):
        super().__init__(
            lora_name,
            org_module,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
        )
        self._tlora_network_ref = None
        self.tlora_min_rank = _parse_positive_int_arg(
            tlora_min_rank if tlora_min_rank is not None else 1,
            "tlora_min_rank",
        )
        if not 1 <= self.tlora_min_rank <= self.lora_dim:
            raise ValueError(
                "tlora_min_rank must be between 1 and network_dim "
                f"({self.lora_dim}), received {self.tlora_min_rank}"
            )
        self.tlora_rank_schedule = _normalize_schedule(tlora_rank_schedule)
        self.tlora_orthogonal_init = _parse_bool_arg(tlora_orthogonal_init, default=False)
        self.dropout = _parse_probability_arg(
            self.dropout,
            "network_dropout",
        )
        self.rank_dropout = _parse_probability_arg(
            self.rank_dropout,
            "rank_dropout",
        )
        self.module_dropout = _parse_probability_arg(
            self.module_dropout,
            "module_dropout",
        )
        self.register_buffer(
            "tlora_min_rank_state",
            torch.tensor(self.tlora_min_rank, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "tlora_rank_schedule_state",
            torch.tensor(
                0 if self.tlora_rank_schedule == "linear" else 1,
                dtype=torch.int64,
            ),
            persistent=True,
        )
        self.org_module_ref = [org_module]
        self.enabled = True

        if self.tlora_orthogonal_init:
            torch.nn.init.orthogonal_(self.lora_down.weight)

    def set_network(self, network):
        self._tlora_network_ref = weakref.ref(network) if network is not None else None

    def _get_network(self):
        if self._tlora_network_ref is None:
            return None
        return self._tlora_network_ref()

    def _get_current_timesteps(self):
        network = self._get_network()
        if network is None:
            raise RuntimeError(
                f"T-LoRA module {self.lora_name} is not bound to its network"
            )
        if not self.lora_name.startswith(lora_network.LoRANetwork.LORA_PREFIX_UNET):
            return None

        timesteps = getattr(network, "current_timestep", None)
        if timesteps is None:
            raise RuntimeError(
                "T-LoRA current timestep is missing; call "
                "set_current_timestep() before every model forward"
            )
        tensor = (
            timesteps
            if torch.is_tensor(timesteps)
            else torch.as_tensor(timesteps)
        )
        if tensor.numel() == 0:
            raise RuntimeError(
                "T-LoRA current timestep is empty; call "
                "set_current_timestep() with at least one value"
            )
        return tensor

    def _get_tlora_rank_mask_and_scale(self, lx):
        timesteps = self._get_current_timesteps()
        if timesteps is None:
            return None, None

        batch_size = lx.size(0)
        timesteps = timesteps.detach().to(device=lx.device, dtype=torch.float32).reshape(-1)
        if timesteps.numel() == 1:
            timesteps = timesteps.expand(batch_size)
        elif timesteps.numel() != batch_size:
            raise ValueError(
                "T-LoRA timestep batch size mismatch: "
                f"received {timesteps.numel()} values for batch size {batch_size}"
            )

        if not torch.isfinite(timesteps).all():
            raise ValueError(
                "T-LoRA timestep values must all be finite"
            )
        minimum = float(timesteps.min().item())
        maximum = float(timesteps.max().item())
        if minimum < 0.0 or maximum > 1000.0:
            raise ValueError(
                "T-LoRA timestep values must be within [0, 1] or "
                f"[0, 1000], received min={minimum}, max={maximum}"
            )
        if maximum > 1.0:
            timesteps = timesteps / 1000.0

        progress = 1.0 - timesteps
        if self.tlora_rank_schedule == "cosine":
            progress = 0.5 - 0.5 * torch.cos(progress * math.pi)

        active_rank = self.tlora_min_rank + torch.floor(
            (self.lora_dim - self.tlora_min_rank) * progress
        ).to(torch.int64)
        active_rank = active_rank.clamp(min=self.tlora_min_rank, max=self.lora_dim)

        rank_index = torch.arange(self.lora_dim, device=lx.device).unsqueeze(0)
        rank_mask = (rank_index < active_rank.unsqueeze(1)).to(dtype=lx.dtype)
        rank_mask = _broadcast_rank_mask(rank_mask, lx)

        return rank_mask, None

    def forward(self, x):
        org_forwarded = self.org_forward(x)
        if not self.enabled:
            return org_forwarded

        if self.module_dropout is not None and self.training:
            if bool(
                torch.rand((), device=x.device) < self.module_dropout
            ):
                return org_forwarded

        lx = self.lora_down(x)
        tlora_rank_mask, _ = self._get_tlora_rank_mask_and_scale(lx)
        if tlora_rank_mask is not None:
            lx = lx * tlora_rank_mask

        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        if self.rank_dropout is not None and self.training:
            mask = torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
            mask = _broadcast_rank_mask(mask, lx)
            lx = lx * mask
            scale: Union[float, torch.Tensor] = self.scale * (1.0 / (1.0 - self.rank_dropout))
        else:
            scale = self.scale

        lx = self.lora_up(lx)
        return org_forwarded + lx * self.multiplier * scale


class TLoRAInfModule(TLoRAModule):
    pass


class TLoRANetwork(lora_network.LoRANetwork):
    def __init__(
        self,
        text_encoder,
        unet,
        *args,
        tlora_min_rank: Optional[int] = None,
        tlora_rank_schedule: Optional[str] = None,
        tlora_orthogonal_init: bool = False,
        module_class=None,
        **kwargs,
    ):
        self.current_timestep = None
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
        module_class.supports_conv2d = getattr(
            base_module_class,
            "supports_conv2d",
            True,
        )

        super().__init__(text_encoder, unet, *args, module_class=module_class, **kwargs)

    def set_current_timestep(self, timestep):
        self.current_timestep = timestep

    def clear_current_timestep(self):
        self.current_timestep = None

    def apply_to(self, text_encoder, unet, apply_text_encoder=True, apply_unet=True):
        if apply_text_encoder:
            logger.info(f"enable LoRA for text encoder: {len(self.text_encoder_loras)} modules")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info(f"enable LoRA for U-Net: {len(self.unet_loras)} modules")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            if hasattr(lora, "set_network"):
                lora.set_network(self)
            lora.apply_to()
            self.add_module(lora.lora_name, lora)


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: AutoencoderKL,
    text_encoder: Union[CLIPTextModel, List[CLIPTextModel]],
    unet,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    del vae
    is_sdxl = unet is not None and issubclass(unet.__class__, SdxlUNet2DConditionModel)

    if network_dim is None:
        network_dim = 4
    if network_alpha is None:
        network_alpha = 1.0

    conv_dim = kwargs.get("conv_dim", None)
    conv_alpha = kwargs.get("conv_alpha", None)
    if conv_dim is not None:
        conv_dim = int(conv_dim)
        if conv_alpha is None:
            conv_alpha = 1.0
        else:
            conv_alpha = float(conv_alpha)

    block_dims = kwargs.get("block_dims", None)
    block_lr_weight = lora_network.parse_block_lr_kwargs(is_sdxl, kwargs)

    if block_dims is not None or block_lr_weight is not None:
        block_alphas = kwargs.get("block_alphas", None)
        conv_block_dims = kwargs.get("conv_block_dims", None)
        conv_block_alphas = kwargs.get("conv_block_alphas", None)

        block_dims, block_alphas, conv_block_dims, conv_block_alphas = lora_network.get_block_dims_and_alphas(
            is_sdxl,
            block_dims,
            block_alphas,
            network_dim,
            network_alpha,
            conv_block_dims,
            conv_block_alphas,
            conv_dim,
            conv_alpha,
        )
        block_dims, block_alphas, conv_block_dims, conv_block_alphas = lora_network.remove_block_dims_and_alphas(
            is_sdxl, block_dims, block_alphas, conv_block_dims, conv_block_alphas, block_lr_weight
        )
    else:
        block_alphas = None
        conv_block_dims = None
        conv_block_alphas = None

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
    tlora_rank_schedule = _normalize_schedule(kwargs.get("tlora_rank_schedule", "cosine"))
    tlora_orthogonal_init = _parse_bool_arg(kwargs.get("tlora_orthogonal_init", False), default=False)

    network = TLoRANetwork(
        text_encoder,
        unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        conv_lora_dim=conv_dim,
        conv_alpha=conv_alpha,
        block_dims=block_dims,
        block_alphas=block_alphas,
        conv_block_dims=conv_block_dims,
        conv_block_alphas=conv_block_alphas,
        varbose=True,
        is_sdxl=is_sdxl,
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

    if block_lr_weight is not None:
        network.set_block_lr_weight(block_lr_weight)

    return network


def create_network_from_weights(multiplier, file, vae, text_encoder, unet, weights_sd=None, for_inference=False, **kwargs):
    del vae
    is_sdxl = unet is not None and issubclass(unet.__class__, SdxlUNet2DConditionModel)

    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    if is_sdxl:
        lora_network.convert_diffusers_to_sai_if_needed(weights_sd)

    modules_dim = {}
    modules_alpha = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_down" in key:
            modules_dim[lora_name] = value.size()[0]

    for key in modules_dim.keys():
        if key not in modules_alpha:
            modules_alpha[key] = modules_dim[key]

    tlora_min_rank = _parse_positive_int_arg(
        kwargs.get("tlora_min_rank", 1),
        "tlora_min_rank",
    )
    tlora_rank_schedule = _normalize_schedule(kwargs.get("tlora_rank_schedule", "cosine"))
    tlora_orthogonal_init = _parse_bool_arg(kwargs.get("tlora_orthogonal_init", False), default=False)

    module_class = TLoRAInfModule if for_inference else None

    network = TLoRANetwork(
        text_encoder,
        unet,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        is_sdxl=is_sdxl,
        tlora_min_rank=tlora_min_rank,
        tlora_rank_schedule=tlora_rank_schedule,
        tlora_orthogonal_init=tlora_orthogonal_init,
    )

    block_lr_weight = lora_network.parse_block_lr_kwargs(is_sdxl, kwargs)
    if block_lr_weight is not None:
        network.set_block_lr_weight(block_lr_weight)

    return network, weights_sd
