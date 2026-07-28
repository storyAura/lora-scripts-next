from __future__ import annotations

import math
import os
from functools import partial
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from networks import lora_anima


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(
            f"{field_name} must be a positive integer, received {value!r}"
        )
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field_name} must be a positive integer, received {value!r}"
        ) from error
    if str(value).strip() not in {str(parsed), f"+{parsed}"}:
        try:
            if float(value) != parsed:
                raise ValueError
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{field_name} must be a positive integer, received {value!r}"
            ) from error
    if parsed <= 0:
        raise ValueError(
            f"{field_name} must be a positive integer, received {parsed}"
        )
    return parsed


def _non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(
            f"{field_name} must be a non-negative integer, received {value!r}"
        )
    try:
        parsed = int(value)
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field_name} must be a non-negative integer, received {value!r}"
        ) from error
    if not math.isfinite(numeric) or numeric != parsed:
        raise ValueError(
            f"{field_name} must be a non-negative integer, received {value!r}"
        )
    if parsed < 0:
        raise ValueError(
            f"{field_name} must be a non-negative integer, received {parsed}"
        )
    return parsed


def _finite_float(value: object, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{field_name} must be finite, received {value!r}"
        ) from error
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite, received {parsed}")
    return parsed


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


def _wavelet_family(value: object) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"db1", "haar"}:
        return "db1"
    raise ValueError(
        "waveft_wavelet_family currently supports only 'db1'/'haar', "
        f"received {value!r}"
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


def _haar_inverse_2d(
    approximation: torch.Tensor,
    horizontal: torch.Tensor,
    vertical: torch.Tensor,
    diagonal: torch.Tensor,
) -> torch.Tensor:
    top_left = (approximation + horizontal + vertical + diagonal) * 0.5
    top_right = (approximation - horizontal + vertical - diagonal) * 0.5
    bottom_left = (approximation + horizontal - vertical - diagonal) * 0.5
    bottom_right = (approximation - horizontal - vertical + diagonal) * 0.5
    top = torch.stack((top_left, bottom_left), dim=-1).reshape(
        approximation.shape[0],
        approximation.shape[1] * 2,
    )
    bottom = torch.stack((top_right, bottom_right), dim=-1).reshape(
        approximation.shape[0],
        approximation.shape[1] * 2,
    )
    return torch.stack((top, bottom), dim=-2).reshape(
        approximation.shape[0] * 2,
        approximation.shape[1] * 2,
    )


def _load_checkpoint(
    file: str | os.PathLike[str] | None,
    weights_sd: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    if weights_sd is not None:
        return dict(weights_sd)
    if file is None:
        raise ValueError(
            "WaveFT create_network_from_weights requires file or weights_sd"
        )
    path = Path(file)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return dict(load_file(str(path)))
    loaded = torch.load(str(path), map_location="cpu")
    if not isinstance(loaded, dict):
        raise TypeError(
            "WaveFT checkpoint must contain a tensor state dictionary, "
            f"received {type(loaded).__name__}"
        )
    if not all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in loaded.items()
    ):
        raise TypeError(
            "WaveFT checkpoint entries must map string keys to tensors"
        )
    return dict(loaded)


def _checkpoint_frequencies(
    state: dict[str, torch.Tensor],
) -> dict[str, int]:
    frequencies: dict[str, int] = {}
    suffix = ".waveft_spectrum"
    for key, value in state.items():
        if not key.endswith(suffix):
            continue
        if value.ndim != 1 or value.numel() <= 0:
            raise ValueError(
                f"Invalid WaveFT spectrum shape for {key}: {tuple(value.shape)}"
            )
        frequencies[key[: -len(suffix)]] = int(value.numel())
    if not frequencies:
        raise ValueError(
            "WaveFT checkpoint does not contain any '.waveft_spectrum' "
            "parameters"
        )
    return frequencies


class WaveFTModule(nn.Module):
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
        n_frequency_override: int | None,
        scaling: float,
        random_loc_seed: int,
        use_idwt: bool,
        wavelet_family: str,
    ) -> None:
        super().__init__()
        if not isinstance(org_module, nn.Linear):
            raise TypeError(
                "WaveFT supports Linear modules only, received "
                f"{org_module.__class__.__name__} for {lora_name}"
            )
        if rank_dropout is not None:
            raise ValueError(
                "WaveFT does not support rank_dropout because its trainable "
                "parameters are wavelet coefficients rather than ranks"
            )
        n_frequency = _positive_int(
            lora_dim if n_frequency_override is None else n_frequency_override,
            "waveft_n_frequency",
        )
        element_count = int(org_module.in_features) * int(
            org_module.out_features
        )
        if n_frequency > element_count:
            raise ValueError(
                "waveft_n_frequency must not exceed the adapted weight size "
                f"{element_count}, received {n_frequency} for {lora_name}"
            )
        seed = _non_negative_int(
            random_loc_seed,
            "waveft_random_loc_seed",
        )
        family = _wavelet_family(wavelet_family)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        flat_indices = torch.randperm(
            element_count,
            generator=generator,
        )[:n_frequency]
        indices = torch.stack(
            (
                flat_indices // int(org_module.in_features),
                flat_indices % int(org_module.in_features),
            ),
            dim=0,
        )

        self.lora_name = str(lora_name)
        self.lora_dim = n_frequency
        self.in_features = int(org_module.in_features)
        self.out_features = int(org_module.out_features)
        self.waveft_spectrum = nn.Parameter(
            torch.zeros(n_frequency, dtype=torch.float32)
        )
        self.register_buffer(
            "waveft_indices",
            indices.to(torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "waveft_scaling",
            torch.tensor(
                _finite_float(scaling, "waveft_scaling"),
                dtype=torch.float32,
            ),
            persistent=True,
        )
        self.register_buffer(
            "waveft_random_loc_seed",
            torch.tensor(seed, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "waveft_use_idwt",
            torch.tensor(bool(use_idwt), dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer(
            "waveft_family_code",
            torch.tensor(1, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "alpha",
            torch.tensor(
                n_frequency
                if alpha is None or float(alpha) == 0
                else float(alpha)
            ),
        )
        self.multiplier = float(multiplier)
        self.scale = 1.0
        self.dropout = _probability(dropout, "network_dropout")
        self.rank_dropout = None
        self.module_dropout = _probability(
            module_dropout,
            "module_dropout",
        )
        self.wavelet_family = family
        self.org_module = org_module
        self.org_module_ref = [org_module]
        self.enabled = True

    def apply_to(self) -> None:
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def _direct_delta(self) -> torch.Tensor:
        flat = torch.zeros(
            self.out_features * self.in_features,
            device=self.waveft_spectrum.device,
            dtype=self.waveft_spectrum.dtype,
        )
        flat = flat.scatter(
            0,
            (
                self.waveft_indices[0] * self.in_features
                + self.waveft_indices[1]
            ).to(flat.device),
            self.waveft_spectrum,
        )
        return flat.reshape(self.out_features, self.in_features)

    def _idwt_delta(self) -> torch.Tensor:
        padded_rows = self.out_features
        padded_columns = self.in_features
        if padded_rows % 2:
            padded_rows += 1
        if padded_columns % 2:
            padded_columns += 1
        row_offset = (padded_rows - self.out_features) // 2
        column_offset = (padded_columns - self.in_features) // 2
        padded_rows_index = self.waveft_indices[0] + row_offset
        padded_columns_index = self.waveft_indices[1] + column_offset
        flat_indices = (
            padded_rows_index * padded_columns + padded_columns_index
        ).to(self.waveft_spectrum.device)
        dense = torch.zeros(
            padded_rows * padded_columns,
            device=self.waveft_spectrum.device,
            dtype=self.waveft_spectrum.dtype,
        ).scatter(0, flat_indices, self.waveft_spectrum)
        dense = dense.reshape(padded_rows, padded_columns)
        half_rows = padded_rows // 2
        half_columns = padded_columns // 2
        reconstructed = _haar_inverse_2d(
            dense[:half_rows, :half_columns],
            dense[:half_rows, half_columns:],
            dense[half_rows:, :half_columns],
            dense[half_rows:, half_columns:],
        )
        start_row = (reconstructed.shape[0] - self.out_features) // 2
        start_column = (
            reconstructed.shape[1] - self.in_features
        ) // 2
        return reconstructed[
            start_row : start_row + self.out_features,
            start_column : start_column + self.in_features,
        ]

    def get_delta_weight(self) -> torch.Tensor:
        if int(self.waveft_family_code.item()) != 1:
            raise ValueError(
                "WaveFT checkpoint contains an unsupported wavelet family "
                f"code {int(self.waveft_family_code.item())}"
            )
        delta = (
            self._idwt_delta()
            if bool(self.waveft_use_idwt.item())
            else self._direct_delta()
        )
        return delta * self.waveft_scaling

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original = self.org_forward(inputs)
        if not self.enabled:
            return original
        if self.module_dropout is not None and self.training:
            if bool(
                torch.rand((), device=inputs.device) < self.module_dropout
            ):
                return original
        adapted_inputs = inputs
        if self.dropout is not None and self.training:
            adapted_inputs = torch.nn.functional.dropout(
                adapted_inputs,
                p=self.dropout,
            )
        delta = self.get_delta_weight()
        adapter = torch.nn.functional.linear(
            adapted_inputs.to(delta.dtype),
            delta,
        )
        return original + adapter.to(original.dtype) * self.multiplier

    def get_weight(self, multiplier=None) -> torch.Tensor:
        effective_multiplier = (
            self.multiplier if multiplier is None else float(multiplier)
        )
        return self.get_delta_weight().float() * effective_multiplier

    def merge_to(self, state: dict[str, torch.Tensor], dtype, device) -> None:
        required = {
            "waveft_spectrum",
            "waveft_indices",
            "waveft_scaling",
            "waveft_random_loc_seed",
            "waveft_use_idwt",
            "waveft_family_code",
        }
        missing = sorted(required.difference(state))
        if missing:
            raise ValueError(
                f"WaveFT checkpoint for {self.lora_name} is missing {missing}"
            )
        self.waveft_spectrum.data.copy_(
            state["waveft_spectrum"].to(self.waveft_spectrum)
        )
        self.waveft_indices.copy_(
            state["waveft_indices"].to(self.waveft_indices)
        )
        self.waveft_scaling.copy_(
            state["waveft_scaling"].to(self.waveft_scaling)
        )
        self.waveft_random_loc_seed.copy_(
            state["waveft_random_loc_seed"].to(
                self.waveft_random_loc_seed
            )
        )
        self.waveft_use_idwt.copy_(
            state["waveft_use_idwt"].to(self.waveft_use_idwt)
        )
        self.waveft_family_code.copy_(
            state["waveft_family_code"].to(self.waveft_family_code)
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
        return self.waveft_spectrum.device

    @property
    def dtype(self) -> torch.dtype:
        return self.waveft_spectrum.dtype


class WaveFTNetwork(lora_anima.LoRANetwork):
    def apply_max_norm_regularization(self, max_norm_value, device):
        raise NotImplementedError(
            "WaveFT does not support scale_weight_norms because its "
            "coefficients do not form independent LoRA factors"
        )

    def save_weights(self, file, dtype, metadata):
        owned_metadata = dict(metadata or {})
        owned_metadata["ss_adapter_algorithm"] = "waveft"
        owned_metadata["ss_waveft_wavelet_family"] = "db1"
        super().save_weights(file, dtype, owned_metadata)


def _module_factory(
    n_frequency_override: int | None,
    scaling: float,
    random_loc_seed: int,
    use_idwt: bool,
    wavelet_family: str,
):
    factory = partial(
        WaveFTModule,
        n_frequency_override=n_frequency_override,
        scaling=scaling,
        random_loc_seed=random_loc_seed,
        use_idwt=use_idwt,
        wavelet_family=wavelet_family,
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
    n_frequency = _positive_int(
        kwargs.pop("waveft_n_frequency", 2592),
        "waveft_n_frequency",
    )
    scaling = _finite_float(
        kwargs.pop("waveft_scaling", 25.0),
        "waveft_scaling",
    )
    seed = _non_negative_int(
        kwargs.pop("waveft_random_loc_seed", 777),
        "waveft_random_loc_seed",
    )
    use_idwt = _parse_bool(
        kwargs.pop("waveft_use_idwt", True),
        "waveft_use_idwt",
    )
    family = _wavelet_family(
        kwargs.pop("waveft_wavelet_family", "db1")
    )
    kwargs["_network_factory"] = WaveFTNetwork
    kwargs["_module_class"] = _module_factory(
        n_frequency,
        scaling,
        seed,
        use_idwt,
        family,
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
    frequencies = _checkpoint_frequencies(state)
    module_factory = _module_factory(
        None,
        1.0,
        0,
        False,
        "db1",
    )
    train_llm_adapter = any(
        "llm_adapter" in name for name in frequencies
    )
    network = WaveFTNetwork(
        text_encoders,
        unet,
        multiplier=float(multiplier),
        lora_dim=max(frequencies.values()),
        alpha=float(max(frequencies.values())),
        module_class=module_factory,
        modules_dim=frequencies,
        modules_alpha={
            name: frequency for name, frequency in frequencies.items()
        },
        train_llm_adapter=train_llm_adapter,
        exclude_patterns=[],
        include_patterns=[],
    )
    return network, state
