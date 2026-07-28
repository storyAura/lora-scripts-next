from __future__ import annotations

import math
from functools import partial
from typing import Optional

import torch
from torch import nn

from networks import lora_anima


def _positive_finite_float(value: object, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field_name} must be a positive finite number, received {value!r}"
        ) from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(
            f"{field_name} must be a positive finite number, received {parsed}"
        )
    return parsed


def _probability(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 <= parsed < 1:
        raise ValueError(
            f"{field_name} must be in [0, 1), received {parsed}"
        )
    return parsed


class DeLoRAModule(nn.Module):
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
        delora_lambda: float,
    ) -> None:
        super().__init__()
        if not isinstance(org_module, nn.Linear):
            raise TypeError(
                "DeLoRA supports Linear modules only, received "
                f"{org_module.__class__.__name__} for {lora_name}"
            )
        rank = int(lora_dim)
        if rank <= 0:
            raise ValueError(f"network_dim must be positive, received {rank}")
        if rank_dropout is not None:
            raise ValueError(
                "DeLoRA does not support rank_dropout because its normalized "
                "rank components must remain coupled"
            )

        self.lora_name = str(lora_name)
        self.lora_dim = rank
        alpha_value = rank if alpha is None or float(alpha) == 0 else float(alpha)
        self.register_buffer("alpha", torch.tensor(alpha_value))
        self.lora_down = nn.Linear(org_module.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, org_module.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        self.delora_lambda = nn.Parameter(
            torch.tensor(
                _positive_finite_float(delora_lambda, "delora_lambda"),
                dtype=torch.float32,
            )
        )
        self.register_buffer(
            "delora_w_norm",
            torch.linalg.vector_norm(
                org_module.weight.detach().float(),
                dim=0,
            ),
            persistent=True,
        )
        self.multiplier = float(multiplier)
        self.scale = 1.0
        self.dropout = _probability(dropout, "network_dropout")
        self.rank_dropout = None
        self.module_dropout = _probability(
            module_dropout,
            "module_dropout",
        )
        self.org_module = org_module
        self.org_module_ref = [org_module]
        self.enabled = True

    def apply_to(self) -> None:
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def _normalization(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        norm_a = torch.linalg.vector_norm(
            self.lora_down.weight.float(),
            dim=1,
        ).clamp_min(1e-4)
        norm_b = torch.linalg.vector_norm(
            self.lora_up.weight.float(),
            dim=0,
        ).clamp_min(1e-4)
        normalized = self.delora_lambda.float() / self.lora_dim / (norm_a * norm_b)
        return normalized.to(device=device, dtype=dtype)

    def _adapter_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight_norm = self.delora_w_norm.to(
            device=inputs.device,
            dtype=inputs.dtype,
        )
        hidden = self.lora_down(inputs * weight_norm)
        if self.dropout is not None and self.training:
            hidden = torch.nn.functional.dropout(hidden, p=self.dropout)
        hidden = hidden * self._normalization(hidden.dtype, hidden.device)
        return self.lora_up(hidden)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original = self.org_forward(inputs)
        if not self.enabled:
            return original
        if self.module_dropout is not None and self.training:
            if bool(
                torch.rand((), device=inputs.device) < self.module_dropout
            ):
                return original
        return original + self._adapter_forward(inputs) * self.multiplier

    def get_weight(self, multiplier=None) -> torch.Tensor:
        effective_multiplier = (
            self.multiplier if multiplier is None else float(multiplier)
        )
        norm_a = torch.linalg.vector_norm(
            self.lora_down.weight.float(),
            dim=1,
        ).clamp_min(1e-4)
        norm_b = torch.linalg.vector_norm(
            self.lora_up.weight.float(),
            dim=0,
        ).clamp_min(1e-4)
        diagonal = self.delora_lambda.float() / self.lora_dim / (norm_a * norm_b)
        delta = (
            self.lora_up.weight.float() * diagonal.unsqueeze(0)
        ) @ self.lora_down.weight.float()
        delta = delta * self.delora_w_norm.float().unsqueeze(0)
        return delta * effective_multiplier

    def merge_to(self, state: dict[str, torch.Tensor], dtype, device) -> None:
        required = {
            "lora_down.weight",
            "lora_up.weight",
            "delora_lambda",
            "delora_w_norm",
        }
        missing = sorted(required.difference(state))
        if missing:
            raise ValueError(
                f"DeLoRA checkpoint for {self.lora_name} is missing {missing}"
            )
        self.lora_down.weight.data.copy_(
            state["lora_down.weight"].to(self.lora_down.weight)
        )
        self.lora_up.weight.data.copy_(
            state["lora_up.weight"].to(self.lora_up.weight)
        )
        self.delora_lambda.data.copy_(
            state["delora_lambda"].to(self.delora_lambda)
        )
        self.delora_w_norm.copy_(
            state["delora_w_norm"].to(self.delora_w_norm)
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

    @property
    def device(self) -> torch.device:
        return self.lora_down.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.lora_down.weight.dtype


class DeLoRANetwork(lora_anima.LoRANetwork):
    def apply_max_norm_regularization(self, max_norm_value, device):
        raise NotImplementedError(
            "DeLoRA does not support scale_weight_norms because normalizing "
            "its factors independently changes the algorithm"
        )

    def save_weights(self, file, dtype, metadata):
        owned_metadata = dict(metadata or {})
        owned_metadata["ss_adapter_algorithm"] = "delora"
        owned_metadata["ss_delora_normalization"] = "column_weight_norm"
        super().save_weights(file, dtype, owned_metadata)


def _module_factory(delora_lambda: float):
    factory = partial(DeLoRAModule, delora_lambda=delora_lambda)
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
    delora_lambda = _positive_finite_float(
        kwargs.pop("delora_lambda", 15.0),
        "delora_lambda",
    )
    kwargs["_network_factory"] = DeLoRANetwork
    kwargs["_module_class"] = _module_factory(delora_lambda)
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
    delora_lambda = _positive_finite_float(
        kwargs.pop("delora_lambda", 15.0),
        "delora_lambda",
    )
    kwargs["_network_factory"] = DeLoRANetwork
    kwargs["_module_class"] = _module_factory(delora_lambda)
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
