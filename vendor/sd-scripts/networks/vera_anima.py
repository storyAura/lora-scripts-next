from __future__ import annotations

import ast
import math
import os
from functools import partial
from pathlib import Path
import re
from typing import Optional
import weakref

import torch

from networks import lora_anima


def _parse_bool(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{field_name} must be a boolean value, received {value!r}"
    )


def _parse_positive_int(value: object, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field_name} must be an integer, received {value!r}"
        ) from error
    if parsed <= 0:
        raise ValueError(
            f"{field_name} must be greater than zero, received {parsed}"
        )
    return parsed


def _parse_non_negative_int(value: object, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field_name} must be an integer, received {value!r}"
        ) from error
    if parsed < 0:
        raise ValueError(
            f"{field_name} must be non-negative, received {parsed}"
        )
    return parsed


def _parse_non_negative_float(value: object, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field_name} must be numeric, received {value!r}"
        ) from error
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(
            f"{field_name} must be finite and non-negative, received {parsed}"
        )
    return parsed


def _parse_patterns(value: object, field_name: str) -> tuple[re.Pattern[str], ...]:
    if value is None:
        return ()
    parsed: object
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as error:
            raise ValueError(
                f"{field_name} must be a Python string or list literal, "
                f"received {value!r}"
            ) from error
    else:
        parsed = value
    items = parsed if isinstance(parsed, list) else [parsed]
    patterns: list[re.Pattern[str]] = []
    for item in items:
        if not isinstance(item, str) or not item:
            raise TypeError(
                f"{field_name} entries must be non-empty strings, "
                f"received {item!r}"
            )
        try:
            patterns.append(re.compile(item))
        except re.error as error:
            raise ValueError(
                f"{field_name} contains invalid regex {item!r}: {error}"
            ) from error
    return tuple(patterns)


def _parse_reg_dims(value: object) -> tuple[int, ...]:
    if value is None:
        return ()
    dimensions: list[int] = []
    for raw_pair in str(value).split(","):
        pair = raw_pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(
                "network_reg_dims entries must use regex=rank syntax, "
                f"received {pair!r}"
            )
        _pattern, raw_dimension = pair.split("=", 1)
        dimensions.append(
            _parse_positive_int(raw_dimension.strip(), "network_reg_dims rank")
        )
    return tuple(dimensions)


def _selected_linear_dimensions(
    text_encoders: list[torch.nn.Module],
    unet: torch.nn.Module,
    train_llm_adapter: bool,
    exclude_patterns: tuple[re.Pattern[str], ...],
    include_patterns: tuple[re.Pattern[str], ...],
) -> tuple[int, int]:
    target_roots: list[tuple[torch.nn.Module, tuple[str, ...]]] = []
    for text_encoder in text_encoders:
        if text_encoder is not None:
            target_roots.append(
                (
                    text_encoder,
                    tuple(lora_anima.LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE),
                )
            )
    unet_targets = list(lora_anima.LoRANetwork.ANIMA_TARGET_REPLACE_MODULE)
    if train_llm_adapter:
        unet_targets.extend(
            lora_anima.LoRANetwork.ANIMA_ADAPTER_TARGET_REPLACE_MODULE
        )
    target_roots.append((unet, tuple(unet_targets)))

    dimensions: list[tuple[int, int]] = []
    visited: set[int] = set()
    for root_module, target_names in target_roots:
        for parent_name, parent_module in root_module.named_modules():
            if parent_module.__class__.__name__ not in target_names:
                continue
            for child_name, child_module in parent_module.named_modules():
                if not isinstance(child_module, torch.nn.Linear):
                    continue
                original_name = (
                    (parent_name + "." if parent_name else "") + child_name
                )
                excluded = any(
                    pattern.fullmatch(original_name)
                    for pattern in exclude_patterns
                )
                included = any(
                    pattern.fullmatch(original_name)
                    for pattern in include_patterns
                )
                if excluded and not included:
                    continue
                identity = id(child_module)
                if identity in visited:
                    continue
                visited.add(identity)
                dimensions.append(
                    (int(child_module.in_features), int(child_module.out_features))
                )
    if not dimensions:
        raise ValueError(
            "VeRA did not find any supported Linear target modules after "
            "applying include_patterns and exclude_patterns"
        )
    return (
        max(in_features for in_features, _out_features in dimensions),
        max(out_features for _in_features, out_features in dimensions),
    )


def _create_projection(
    rank: int,
    maximum_input: int,
    maximum_output: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    projection_a = torch.empty(rank, maximum_input, dtype=torch.float32)
    projection_b = torch.empty(maximum_output, rank, dtype=torch.float32)
    torch.nn.init.kaiming_uniform_(
        projection_a,
        a=math.sqrt(5),
        generator=generator,
    )
    torch.nn.init.kaiming_uniform_(
        projection_b,
        a=math.sqrt(5),
        generator=generator,
    )
    return projection_a, projection_b


class VeRAModule(torch.nn.Module):
    supports_conv2d = False

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier,
        lora_dim,
        alpha,
        dropout,
        rank_dropout,
        module_dropout,
        d_initial,
    ):
        super().__init__()
        if not isinstance(org_module, torch.nn.Linear):
            raise TypeError(
                "VeRA supports Linear modules only, received "
                f"{org_module.__class__.__name__} for {lora_name}"
            )
        self.lora_name = str(lora_name)
        self.lora_dim = _parse_positive_int(lora_dim, "network_dim")
        alpha_value = (
            self.lora_dim
            if alpha is None or float(alpha) == 0
            else float(alpha)
        )
        if not math.isfinite(alpha_value):
            raise ValueError(
                f"network_alpha must be finite, received {alpha_value}"
            )
        self.scale = alpha_value / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha_value))
        self.vera_lambda_b = torch.nn.Parameter(
            torch.zeros(int(org_module.out_features), dtype=torch.float32)
        )
        self.vera_lambda_d = torch.nn.Parameter(
            torch.full(
                (self.lora_dim,),
                float(d_initial),
                dtype=torch.float32,
            )
        )
        self.multiplier = float(multiplier)
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.in_features = int(org_module.in_features)
        self.out_features = int(org_module.out_features)
        self.org_module = org_module
        self.enabled = True
        self._network_ref: weakref.ReferenceType[VeRANetwork] | None = None

    def bind_projection(self, network: "VeRANetwork") -> None:
        self._network_ref = weakref.ref(network)

    def _projection(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._network_ref is None:
            raise RuntimeError(
                f"VeRA module {self.lora_name} is not bound to shared projections"
            )
        network = self._network_ref()
        if network is None:
            raise RuntimeError(
                f"VeRA network for module {self.lora_name} has been released"
            )
        projection_a = network.vera_A[: self.lora_dim, : self.in_features]
        projection_b = network.vera_B[: self.out_features, : self.lora_dim]
        return projection_a, projection_b

    def projection_data_ptrs(self) -> tuple[int, int]:
        projection_a, projection_b = self._projection()
        return (
            projection_a.untyped_storage().data_ptr(),
            projection_b.untyped_storage().data_ptr(),
        )

    def apply_to(self) -> None:
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def _adapter_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        projection_a, projection_b = self._projection()
        projection_a = projection_a.to(
            device=inputs.device,
            dtype=inputs.dtype,
        )
        projection_b = projection_b.to(
            device=inputs.device,
            dtype=inputs.dtype,
        )
        lambda_d = self.vera_lambda_d.to(
            device=inputs.device,
            dtype=inputs.dtype,
        )
        lambda_b = self.vera_lambda_b.to(
            device=inputs.device,
            dtype=inputs.dtype,
        )
        hidden = torch.nn.functional.linear(inputs, projection_a)
        if self.dropout is not None and self.training:
            hidden = torch.nn.functional.dropout(hidden, p=float(self.dropout))
        if self.rank_dropout is not None and self.training:
            probability = float(self.rank_dropout)
            if not 0 <= probability < 1:
                raise ValueError(
                    "rank_dropout must be in [0, 1), "
                    f"received {probability}"
                )
            mask = (
                torch.rand(
                    hidden.shape,
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
                > probability
            )
            hidden = hidden * mask / (1 - probability)
        hidden = hidden * lambda_d
        output = torch.nn.functional.linear(hidden, projection_b)
        return output * lambda_b

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original = self.org_forward(inputs)
        if not self.enabled:
            return original
        if self.module_dropout is not None and self.training:
            probability = float(self.module_dropout)
            if not 0 <= probability < 1:
                raise ValueError(
                    "module_dropout must be in [0, 1), "
                    f"received {probability}"
                )
            if bool(torch.rand((), device=inputs.device) < probability):
                return original
        adapter = self._adapter_forward(inputs)
        return original + adapter * self.multiplier * self.scale

    def get_weight(self, multiplier) -> torch.Tensor:
        effective_multiplier = (
            self.multiplier if multiplier is None else float(multiplier)
        )
        projection_a, projection_b = self._projection()
        lambda_d = self.vera_lambda_d.float()
        lambda_b = self.vera_lambda_b.float()
        weighted_a = projection_a.float() * lambda_d.unsqueeze(1)
        weighted_b = projection_b.float() * lambda_b.unsqueeze(1)
        return weighted_b @ weighted_a * effective_multiplier * self.scale

    def merge_to(self, state: dict[str, torch.Tensor], dtype, device) -> None:
        if "vera_lambda_b" in state:
            self.vera_lambda_b.data.copy_(
                state["vera_lambda_b"].to(self.vera_lambda_b)
            )
        if "vera_lambda_d" in state:
            self.vera_lambda_d.data.copy_(
                state["vera_lambda_d"].to(self.vera_lambda_d)
            )
        weight = self.org_module.weight
        target_device = weight.device if device is None else device
        merged = (
            weight.float().to(target_device)
            + self.get_weight(self.multiplier).to(target_device)
        )
        self.org_module.weight.data.copy_(
            merged.to(device=weight.device, dtype=weight.dtype)
        )

    @property
    def device(self) -> torch.device:
        return self.vera_lambda_b.device

    @property
    def dtype(self) -> torch.dtype:
        return self.vera_lambda_b.dtype


class VeRAModuleFactory:
    supports_conv2d = False

    def __init__(self, d_initial: float):
        self._d_initial = d_initial

    def __call__(
        self,
        lora_name,
        org_module,
        multiplier,
        lora_dim,
        alpha,
        dropout,
        rank_dropout,
        module_dropout,
    ) -> VeRAModule:
        return VeRAModule(
            lora_name,
            org_module,
            multiplier,
            lora_dim,
            alpha,
            dropout,
            rank_dropout,
            module_dropout,
            self._d_initial,
        )


class VeRANetwork(lora_anima.LoRANetwork):
    def __init__(
        self,
        *args,
        vera_projection_a: torch.Tensor,
        vera_projection_b: torch.Tensor,
        vera_projection_seed: int,
        vera_save_projection: bool,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "vera_A",
            vera_projection_a,
            persistent=vera_save_projection,
        )
        self.register_buffer(
            "vera_B",
            vera_projection_b,
            persistent=vera_save_projection,
        )
        self.register_buffer(
            "vera_projection_seed",
            torch.tensor(vera_projection_seed, dtype=torch.int64),
        )
        self.vera_save_projection = vera_save_projection
        for module in self.text_encoder_loras + self.unet_loras:
            module.bind_projection(self)

    def save_weights(self, file, dtype, metadata):
        owned_metadata = dict(metadata or {})
        owned_metadata["ss_adapter_algorithm"] = "vera"
        owned_metadata["ss_vera_projection_seed"] = str(
            int(self.vera_projection_seed.item())
        )
        owned_metadata["ss_vera_projection_persisted"] = str(
            self.vera_save_projection
        ).lower()
        owned_metadata["ss_vera_target_kind"] = "linear"
        super().save_weights(file, dtype, owned_metadata)

    def apply_max_norm_regularization(self, max_norm_value, device):
        raise NotImplementedError(
            "VeRA does not support scale_weight_norms because its trainable "
            "vectors cannot be rescaled as independent LoRA factors"
        )

    def merge_to(self, text_encoders, unet, weights_sd, dtype=None, device=None):
        projection_a = weights_sd.get("vera_A")
        projection_b = weights_sd.get("vera_B")
        if projection_a is not None:
            if tuple(projection_a.shape) != tuple(self.vera_A.shape):
                raise ValueError(
                    "VeRA projection A shape mismatch: "
                    f"checkpoint {tuple(projection_a.shape)}, "
                    f"network {tuple(self.vera_A.shape)}"
                )
            self.vera_A.copy_(projection_a.to(self.vera_A))
        if projection_b is not None:
            if tuple(projection_b.shape) != tuple(self.vera_B.shape):
                raise ValueError(
                    "VeRA projection B shape mismatch: "
                    f"checkpoint {tuple(projection_b.shape)}, "
                    f"network {tuple(self.vera_B.shape)}"
                )
            self.vera_B.copy_(projection_b.to(self.vera_B))
        super().merge_to(text_encoders, unet, weights_sd, dtype, device)


def _build_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae,
    text_encoders: list,
    unet,
    neuron_dropout: Optional[float],
    kwargs: dict[str, object],
) -> VeRANetwork:
    rank = _parse_positive_int(
        4 if network_dim is None else network_dim,
        "network_dim",
    )
    seed = _parse_non_negative_int(
        kwargs.pop("vera_projection_seed", 42),
        "vera_projection_seed",
    )
    save_projection = _parse_bool(
        kwargs.pop("vera_save_projection", True),
        "vera_save_projection",
    )
    d_initial = _parse_non_negative_float(
        kwargs.pop("vera_d_initial", 0.1),
        "vera_d_initial",
    )
    train_llm_adapter = _parse_bool(
        kwargs.get("train_llm_adapter", False),
        "train_llm_adapter",
    )
    exclude_patterns = list(
        _parse_patterns(kwargs.get("exclude_patterns"), "exclude_patterns")
    )
    exclude_patterns.append(
        re.compile(r".*(_modulation|_norm|_embedder|final_layer).*")
    )
    include_patterns = _parse_patterns(
        kwargs.get("include_patterns"),
        "include_patterns",
    )
    maximum_input, maximum_output = _selected_linear_dimensions(
        text_encoders,
        unet,
        train_llm_adapter,
        tuple(exclude_patterns),
        include_patterns,
    )
    configured_ranks = _parse_reg_dims(kwargs.get("network_reg_dims"))
    maximum_rank = max((rank, *configured_ranks))
    projection_a, projection_b = _create_projection(
        maximum_rank,
        maximum_input,
        maximum_output,
        seed,
    )
    module_factory = VeRAModuleFactory(d_initial)
    network_factory = partial(
        VeRANetwork,
        vera_projection_a=projection_a,
        vera_projection_b=projection_b,
        vera_projection_seed=seed,
        vera_save_projection=save_projection,
    )
    kwargs["_network_factory"] = network_factory
    kwargs["_module_class"] = module_factory
    return lora_anima.create_network(
        multiplier,
        rank,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout,
        **kwargs,
    )


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
    return _build_network(
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout,
        dict(kwargs),
    )


def _load_checkpoint(
    file: str | os.PathLike[str] | None,
    weights_sd: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    if weights_sd is not None:
        return dict(weights_sd)
    if file is None:
        raise ValueError(
            "VeRA create_network_from_weights requires file or weights_sd"
        )
    checkpoint_path = Path(file)
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return dict(load_file(str(checkpoint_path)))
    loaded = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(loaded, dict):
        raise TypeError(
            "VeRA checkpoint must contain a tensor state dictionary, "
            f"received {type(loaded).__name__}"
        )
    return loaded


def _checkpoint_rank(weights_sd: dict[str, torch.Tensor]) -> int:
    ranks = {
        int(value.numel())
        for key, value in weights_sd.items()
        if key.endswith("vera_lambda_d")
    }
    if not ranks:
        if "vera_A" in weights_sd:
            return int(weights_sd["vera_A"].shape[0])
        raise ValueError(
            "VeRA checkpoint does not contain vera_lambda_d or vera_A"
        )
    if len(ranks) != 1:
        raise ValueError(
            "VeRA checkpoint contains mixed ranks; pass matching "
            "network_reg_dims when loading"
        )
    return ranks.pop()


def _checkpoint_alpha(
    weights_sd: dict[str, torch.Tensor],
    rank: int,
) -> float:
    alpha_values = {
        float(value.detach().float().item())
        for key, value in weights_sd.items()
        if key.endswith(".alpha") and value.numel() == 1
    }
    if not alpha_values:
        return float(rank)
    if len(alpha_values) != 1:
        raise ValueError(
            "VeRA checkpoint contains mixed alpha values; "
            "mixed-rank loading requires explicit network_reg_dims support"
        )
    return alpha_values.pop()


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
    checkpoint = _load_checkpoint(file, weights_sd)
    seed_tensor = checkpoint.get("vera_projection_seed")
    if seed_tensor is not None:
        kwargs["vera_projection_seed"] = int(seed_tensor.item())
    rank = _checkpoint_rank(checkpoint)
    alpha = _checkpoint_alpha(checkpoint, rank)
    kwargs["vera_save_projection"] = (
        "vera_A" in checkpoint and "vera_B" in checkpoint
    )
    network = _build_network(
        multiplier,
        rank,
        alpha,
        ae,
        text_encoders,
        unet,
        None,
        dict(kwargs),
    )
    if "vera_A" in checkpoint:
        network.vera_A.copy_(checkpoint["vera_A"].to(network.vera_A))
    if "vera_B" in checkpoint:
        network.vera_B.copy_(checkpoint["vera_B"].to(network.vera_B))
    return network, checkpoint
