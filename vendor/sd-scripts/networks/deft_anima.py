from __future__ import annotations

import math
import os
from functools import partial
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from networks import lora_anima


_DECOMPOSITION_CODES = {"relu": 0, "qr": 1}
_DECOMPOSITION_NAMES = {
    code: name for name, code in _DECOMPOSITION_CODES.items()
}


def _decomposition_method(value: object) -> str:
    normalized = str(value).strip().lower()
    if normalized not in _DECOMPOSITION_CODES:
        raise ValueError(
            "deft_decomposition_method must be one of "
            f"{sorted(_DECOMPOSITION_CODES)}, received {value!r}"
        )
    return normalized


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


def _optional_alpha(value: object) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if parsed == 0:
        return None
    return _positive_finite_float(parsed, "deft_alpha")


def _parse_bool(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{field_name} must be a boolean, received {value!r}"
    )


def _probability(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 <= parsed < 1:
        raise ValueError(
            f"{field_name} must be in [0, 1), received {parsed}"
        )
    return parsed


def _load_checkpoint(
    file: str | os.PathLike[str] | None,
    weights_sd: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    if weights_sd is not None:
        return dict(weights_sd)
    if file is None:
        raise ValueError(
            "DEFT create_network_from_weights requires file or weights_sd"
        )
    path = Path(file)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return dict(load_file(str(path)))
    loaded = torch.load(str(path), map_location="cpu")
    if not isinstance(loaded, dict):
        raise TypeError(
            "DEFT checkpoint must contain a tensor state dictionary, "
            f"received {type(loaded).__name__}"
        )
    if not all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in loaded.items()
    ):
        raise TypeError(
            "DEFT checkpoint entries must map string keys to tensors"
        )
    return dict(loaded)


def _checkpoint_dimensions(
    state: dict[str, torch.Tensor],
) -> dict[str, int]:
    dimensions: dict[str, int] = {}
    suffix = ".deft_P"
    for key, value in state.items():
        if not key.endswith(suffix):
            continue
        if value.ndim != 2 or value.shape[1] <= 0:
            raise ValueError(
                f"Invalid DEFT projection shape for {key}: {tuple(value.shape)}"
            )
        dimensions[key[: -len(suffix)]] = int(value.shape[1])
    if not dimensions:
        raise ValueError(
            "DEFT checkpoint does not contain any '.deft_P' parameters"
        )
    return dimensions


class DeftModule(nn.Module):
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
        decomposition_method: str,
        deft_alpha: float | None,
        init_scale: float,
        init_weights: bool,
    ) -> None:
        super().__init__()
        if not isinstance(org_module, nn.Linear):
            raise TypeError(
                "DEFT supports Linear modules only, received "
                f"{org_module.__class__.__name__} for {lora_name}"
            )
        if rank_dropout is not None:
            raise ValueError(
                "DEFT does not support rank_dropout because it changes the "
                "projection subspace"
            )
        requested_rank = int(lora_dim)
        if requested_rank <= 0:
            raise ValueError(
                f"network_dim must be positive, received {requested_rank}"
            )
        rank = min(requested_rank, int(org_module.out_features))
        method = _decomposition_method(decomposition_method)
        scaling = 1.0 if deft_alpha is None else deft_alpha / rank

        self.lora_name = str(lora_name)
        self.lora_dim = rank
        self.multiplier = float(multiplier)
        self.scale = 1.0
        self.dropout = _probability(dropout, "network_dropout")
        self.rank_dropout = None
        self.module_dropout = _probability(
            module_dropout,
            "module_dropout",
        )
        self.deft_P = nn.Parameter(
            torch.empty(int(org_module.out_features), rank)
        )
        self.deft_R = nn.Parameter(
            torch.empty(rank, int(org_module.in_features))
        )
        self.register_buffer(
            "deft_decomposition_code",
            torch.tensor(_DECOMPOSITION_CODES[method], dtype=torch.int64),
        )
        self.register_buffer(
            "deft_scaling",
            torch.tensor(scaling, dtype=torch.float32),
        )
        self.register_buffer(
            "deft_init_scale",
            torch.tensor(init_scale, dtype=torch.float32),
        )
        self.register_buffer(
            "deft_init_weights",
            torch.tensor(bool(init_weights), dtype=torch.bool),
        )
        self.register_buffer(
            "alpha",
            torch.tensor(
                rank if alpha is None or float(alpha) == 0 else float(alpha)
            ),
        )
        nn.init.normal_(self.deft_P, mean=0.0, std=0.02)
        self.org_module = org_module
        self.org_module_ref = [org_module]
        self.enabled = True
        if init_weights:
            self._initialize_identity()
        else:
            nn.init.normal_(
                self.deft_R,
                mean=0.0,
                std=0.02 * init_scale,
            )

    def _method(self) -> str:
        code = int(self.deft_decomposition_code.item())
        method = _DECOMPOSITION_NAMES.get(code)
        if method is None:
            raise ValueError(
                f"Unknown DEFT decomposition code {code} in {self.lora_name}"
            )
        return method

    def projector_factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        projection = self.deft_P.float()
        method = self._method()
        if method == "relu":
            return projection, torch.relu(projection)
        q_matrix, _ = torch.linalg.qr(projection, mode="reduced")
        return q_matrix, q_matrix

    def _initialize_identity(self) -> None:
        with torch.no_grad():
            _, right = self.projector_factors()
            weight = self.org_module.weight.detach().float()
            initial = right.transpose(0, 1) @ weight
            initial = initial / self.deft_scaling.float()
            self.deft_R.copy_(initial.to(self.deft_R))

    def apply_to(self) -> None:
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def _adapter_output(
        self,
        inputs: torch.Tensor,
        original: torch.Tensor,
    ) -> torch.Tensor:
        q_matrix, right = self.projector_factors()
        bias = self.org_module_ref[0].bias
        original_float = original.float()
        base_product = (
            original_float
            if bias is None
            else original_float - bias.detach().float()
        )
        correction = (base_product @ right) @ q_matrix.transpose(0, 1)
        dropped = inputs.float()
        if self.dropout is not None and self.training:
            dropped = torch.nn.functional.dropout(
                dropped,
                p=self.dropout,
            )
        injection = torch.nn.functional.linear(
            dropped,
            self.deft_R.float(),
        )
        injection = (
            injection
            @ q_matrix.transpose(0, 1)
            * self.deft_scaling.float()
        )
        return injection - correction

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original = self.org_forward(inputs)
        if not self.enabled:
            return original
        if self.module_dropout is not None and self.training:
            if bool(
                torch.rand((), device=inputs.device) < self.module_dropout
            ):
                return original
        adapted = (
            original.float()
            + self._adapter_output(inputs, original) * self.multiplier
        )
        return adapted.to(original.dtype)

    def get_weight(self, multiplier=None) -> torch.Tensor:
        effective_multiplier = (
            self.multiplier if multiplier is None else float(multiplier)
        )
        q_matrix, right = self.projector_factors()
        weight = self.org_module_ref[0].weight.detach().float()
        factor = (
            self.deft_R.float() * self.deft_scaling.float()
            - right.transpose(0, 1) @ weight
        )
        return q_matrix @ factor * effective_multiplier

    def merge_to(self, state: dict[str, torch.Tensor], dtype, device) -> None:
        required = {
            "deft_P",
            "deft_R",
            "deft_decomposition_code",
            "deft_scaling",
        }
        missing = sorted(required.difference(state))
        if missing:
            raise ValueError(
                f"DEFT checkpoint for {self.lora_name} is missing {missing}"
            )
        self.deft_P.data.copy_(state["deft_P"].to(self.deft_P))
        self.deft_R.data.copy_(state["deft_R"].to(self.deft_R))
        self.deft_decomposition_code.copy_(
            state["deft_decomposition_code"].to(
                self.deft_decomposition_code
            )
        )
        self.deft_scaling.copy_(
            state["deft_scaling"].to(self.deft_scaling)
        )
        if "deft_init_scale" in state:
            self.deft_init_scale.copy_(
                state["deft_init_scale"].to(self.deft_init_scale)
            )
        if "deft_init_weights" in state:
            self.deft_init_weights.copy_(
                state["deft_init_weights"].to(self.deft_init_weights)
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
        return self.deft_P.device

    @property
    def dtype(self) -> torch.dtype:
        return self.deft_P.dtype


class DeftNetwork(lora_anima.LoRANetwork):
    def apply_max_norm_regularization(self, max_norm_value, device):
        raise NotImplementedError(
            "DEFT does not support scale_weight_norms because its projection "
            "and injection matrices cannot be rescaled independently"
        )

    def save_weights(self, file, dtype, metadata):
        owned_metadata = dict(metadata or {})
        owned_metadata["ss_adapter_algorithm"] = "deft"
        owned_metadata["ss_deft_update"] = "(I-P)W+QR"
        super().save_weights(file, dtype, owned_metadata)


def _module_factory(
    decomposition_method: str,
    deft_alpha: float | None,
    init_scale: float,
    init_weights: bool,
):
    factory = partial(
        DeftModule,
        decomposition_method=decomposition_method,
        deft_alpha=deft_alpha,
        init_scale=init_scale,
        init_weights=init_weights,
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
    method = _decomposition_method(
        kwargs.pop("deft_decomposition_method", "qr")
    )
    deft_alpha = _optional_alpha(kwargs.pop("deft_alpha", None))
    init_scale = _positive_finite_float(
        kwargs.pop("deft_init_scale", 1.0),
        "deft_init_scale",
    )
    init_weights = _parse_bool(
        kwargs.pop("deft_init_weights", True),
        "deft_init_weights",
    )
    kwargs["_network_factory"] = DeftNetwork
    kwargs["_module_class"] = _module_factory(
        method,
        deft_alpha,
        init_scale,
        init_weights,
    )
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
    del ae, for_inference, kwargs
    state = _load_checkpoint(file, weights_sd)
    dimensions = _checkpoint_dimensions(state)
    module_factory = _module_factory("qr", None, 1.0, True)
    train_llm_adapter = any(
        "llm_adapter" in name for name in dimensions
    )
    network = DeftNetwork(
        text_encoders,
        unet,
        multiplier=float(multiplier),
        lora_dim=max(dimensions.values()),
        alpha=float(max(dimensions.values())),
        module_class=module_factory,
        modules_dim=dimensions,
        modules_alpha={
            name: rank for name, rank in dimensions.items()
        },
        train_llm_adapter=train_llm_adapter,
        exclude_patterns=[],
        include_patterns=[],
    )
    return network, state
