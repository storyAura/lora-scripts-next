from __future__ import annotations

import math
from functools import partial
from typing import Optional

import torch
from torch import nn

from networks import lora_anima


_MIXER_INITIALIZERS = frozenset({"kaiming", "identity", "orthogonal"})


def _mixer_initializer(value: object) -> str:
    normalized = str(value).strip().lower()
    if normalized not in _MIXER_INITIALIZERS:
        raise ValueError(
            "moslora_mixer_init must be one of "
            f"{sorted(_MIXER_INITIALIZERS)}, received {value!r}"
        )
    return normalized


def _initialize_mixer(weight: torch.Tensor, initializer: str) -> None:
    if initializer == "kaiming":
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        return
    if initializer == "identity":
        nn.init.eye_(weight)
        return
    nn.init.orthogonal_(weight)


class MoSLoRAModule(lora_anima.LoRAModule):
    supports_conv2d = False

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier,
        lora_dim,
        alpha,
        dropout,
        rank_dropout,
        module_dropout,
        mixer_initializer: str,
    ) -> None:
        if not isinstance(org_module, nn.Linear):
            raise TypeError(
                "MoSLoRA supports Linear modules only, received "
                f"{org_module.__class__.__name__} for {lora_name}"
            )
        super().__init__(
            lora_name,
            org_module,
            multiplier,
            lora_dim,
            alpha,
            dropout,
            rank_dropout,
            module_dropout,
        )
        self.org_module_ref = [org_module]
        self.lora_mixer = nn.Linear(
            self.lora_dim,
            self.lora_dim,
            bias=False,
        )
        _initialize_mixer(self.lora_mixer.weight, mixer_initializer)
        self.moslora_mixer_init = mixer_initializer
        self.enabled = True

    def _rank_dropout(
        self,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        if self.rank_dropout is None or not self.training:
            return hidden, float(self.scale)
        probability = float(self.rank_dropout)
        if not 0 <= probability < 1:
            raise ValueError(
                f"rank_dropout must be in [0, 1), received {probability}"
            )
        mask_shape = [int(hidden.shape[0])]
        mask_shape.extend([1] * (hidden.ndim - 2))
        mask_shape.append(self.lora_dim)
        mask = (
            torch.rand(mask_shape, device=hidden.device) > probability
        ).to(hidden.dtype)
        return hidden * mask, float(self.scale) / (1 - probability)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original = self.org_forward(inputs)
        if not self.enabled:
            return original
        if self.module_dropout is not None and self.training:
            probability = float(self.module_dropout)
            if not 0 <= probability < 1:
                raise ValueError(
                    f"module_dropout must be in [0, 1), received {probability}"
                )
            if bool(torch.rand((), device=inputs.device) < probability):
                return original

        hidden = self.lora_down(inputs)
        if self.dropout is not None and self.training:
            hidden = torch.nn.functional.dropout(
                hidden,
                p=float(self.dropout),
            )
        hidden = self.lora_mixer(hidden)
        hidden, scale = self._rank_dropout(hidden)
        adapter = self.lora_up(hidden)
        return original + adapter * self.multiplier * scale

    def get_weight(self, multiplier=None) -> torch.Tensor:
        effective_multiplier = (
            self.multiplier if multiplier is None else float(multiplier)
        )
        return (
            self.lora_up.weight.float()
            @ self.lora_mixer.weight.float()
            @ self.lora_down.weight.float()
            * effective_multiplier
            * self.scale
        )

    def merge_to(self, state: dict[str, torch.Tensor], dtype, device) -> None:
        required = {
            "lora_down.weight",
            "lora_up.weight",
            "lora_mixer.weight",
        }
        missing = sorted(required.difference(state))
        if missing:
            raise ValueError(
                f"MoSLoRA checkpoint for {self.lora_name} is missing {missing}"
            )
        self.lora_down.weight.data.copy_(
            state["lora_down.weight"].to(self.lora_down.weight)
        )
        self.lora_up.weight.data.copy_(
            state["lora_up.weight"].to(self.lora_up.weight)
        )
        self.lora_mixer.weight.data.copy_(
            state["lora_mixer.weight"].to(self.lora_mixer.weight)
        )
        original_weight = self.org_module.weight
        target_device = original_weight.device if device is None else device
        target_dtype = original_weight.dtype if dtype is None else dtype
        merged = (
            original_weight.detach().float().to(target_device)
            + self.get_weight().to(target_device)
        )
        self.org_module.weight.data.copy_(
            merged.to(device=original_weight.device, dtype=target_dtype)
        )


class MoSLoRANetwork(lora_anima.LoRANetwork):
    def apply_max_norm_regularization(self, max_norm_value, device):
        raise NotImplementedError(
            "MoSLoRA does not support scale_weight_norms because the "
            "trainable mixer must be included in the effective update"
        )

    def to_standard_lora_state_dict(self) -> dict[str, torch.Tensor]:
        exported: dict[str, torch.Tensor] = {}
        for module in self.text_encoder_loras + self.unet_loras:
            prefix = module.lora_name
            exported[f"{prefix}.lora_down.weight"] = (
                module.lora_down.weight.detach().clone()
            )
            folded_up = (
                module.lora_up.weight.detach().float()
                @ module.lora_mixer.weight.detach().float()
            )
            exported[f"{prefix}.lora_up.weight"] = folded_up.to(
                module.lora_up.weight.dtype
            )
            exported[f"{prefix}.alpha"] = module.alpha.detach().clone()
        return exported

    def save_weights(self, file, dtype, metadata):
        owned_metadata = dict(metadata or {})
        owned_metadata["ss_adapter_algorithm"] = "moslora"
        owned_metadata["ss_moslora_formula"] = "B_M_A"
        super().save_weights(file, dtype, owned_metadata)


def _module_factory(initializer: str):
    factory = partial(
        MoSLoRAModule,
        mixer_initializer=initializer,
    )
    factory.supports_conv2d = False
    return factory


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
    initializer = _mixer_initializer(
        kwargs.pop("moslora_mixer_init", "kaiming")
    )
    kwargs["_network_factory"] = MoSLoRANetwork
    kwargs["_module_class"] = _module_factory(initializer)
    return lora_anima.create_network(
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout,
        **kwargs,
    )


def create_network_from_weights(
    multiplier,
    file,
    ae,
    text_encoders,
    unet,
    weights_sd=None,
    for_inference=False,
    **kwargs,
):
    del for_inference
    initializer = _mixer_initializer(
        kwargs.pop("moslora_mixer_init", "kaiming")
    )
    kwargs["_network_factory"] = MoSLoRANetwork
    kwargs["_module_class"] = _module_factory(initializer)
    return lora_anima.create_network_from_weights(
        multiplier,
        file,
        ae,
        text_encoders,
        unet,
        weights_sd,
        False,
        **kwargs,
    )
