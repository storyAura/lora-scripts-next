from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math
import os
from pathlib import Path
from typing import Optional

import torch

from networks import lora_anima


LOSSLESS_EXPORT = "LoRA无损兼容导出"
FAST_EXPORT = "LoRA快速近似导出"


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


def _parse_method(value: object) -> str:
    method = str(value).strip().lower()
    if method not in {"svd", "rsvd"}:
        raise ValueError(
            "pissa_method must be 'svd' or 'rsvd', "
            f"received {value!r}"
        )
    return method


def _parse_export_mode(value: object) -> str:
    aliases = {
        LOSSLESS_EXPORT: LOSSLESS_EXPORT,
        FAST_EXPORT: FAST_EXPORT,
        "lossless": LOSSLESS_EXPORT,
        "fast": FAST_EXPORT,
    }
    normalized = str(value).strip()
    if normalized not in aliases:
        raise ValueError(
            "pissa_export_mode must be "
            f"{LOSSLESS_EXPORT!r} or {FAST_EXPORT!r}, received {value!r}"
        )
    return aliases[normalized]


def _weight_matrix(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim == 2:
        return weight
    if weight.ndim == 4:
        return weight.reshape(weight.shape[0], -1)
    raise ValueError(
        "PiSSA supports Linear and Conv2d weights only, "
        f"received shape {tuple(weight.shape)}"
    )


def _factor_shapes(
    matrix_down: torch.Tensor,
    matrix_up: torch.Tensor,
    down_template: torch.Tensor,
    up_template: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if down_template.ndim == 2:
        return matrix_down, matrix_up
    return (
        matrix_down.reshape_as(down_template),
        matrix_up.reshape_as(up_template),
    )


def _factor_matrix(
    weight: torch.Tensor,
    rank: int,
    scale: float,
    method: str,
    niter: int,
    oversample: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    matrix = _weight_matrix(weight.detach().float())
    maximum_rank = min(matrix.shape)
    if rank > maximum_rank:
        raise ValueError(
            f"PiSSA rank {rank} exceeds matrix maximum rank {maximum_rank} "
            f"for weight shape {tuple(weight.shape)}"
        )
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError(
            f"PiSSA scaling must be finite and positive, received {scale}"
        )

    if method == "svd":
        left, singular_values, right_h = torch.linalg.svd(
            matrix,
            full_matrices=False,
        )
    else:
        projected_rank = min(maximum_rank, rank + oversample)
        left, singular_values, right = torch.svd_lowrank(
            matrix,
            q=projected_rank,
            niter=niter,
        )
        right_h = right.T

    left = left[:, :rank]
    singular_values = singular_values[:rank]
    right_h = right_h[:rank, :]
    roots = torch.sqrt(singular_values / scale)
    matrix_down = roots.unsqueeze(1) * right_h
    matrix_up = left * roots.unsqueeze(0)
    return matrix_down, matrix_up


def _compose_factors(
    up: torch.Tensor,
    down: torch.Tensor,
    target_shape: torch.Size,
) -> torch.Tensor:
    up_matrix = up.float().reshape(up.shape[0], up.shape[1])
    down_matrix = down.float().reshape(down.shape[0], -1)
    return (up_matrix @ down_matrix).reshape(target_shape)


def _approximate_difference(
    current_down: torch.Tensor,
    current_up: torch.Tensor,
    initial_down: torch.Tensor,
    initial_up: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_shape = torch.Size(
        (
            current_up.shape[0],
            *current_down.shape[1:],
        )
    )
    difference = (
        _compose_factors(current_up, current_down, target_shape)
        - _compose_factors(initial_up, initial_down, target_shape)
    )
    matrix = _weight_matrix(difference)
    rank = current_down.shape[0]
    left, singular_values, right_h = torch.linalg.svd(
        matrix,
        full_matrices=False,
    )
    roots = torch.sqrt(singular_values[:rank])
    matrix_down = roots.unsqueeze(1) * right_h[:rank, :]
    matrix_up = left[:, :rank] * roots.unsqueeze(0)
    return _factor_shapes(
        matrix_down,
        matrix_up,
        current_down,
        current_up,
    )


@dataclass(frozen=True)
class PiSSAResumeState:
    trained_down: torch.Tensor
    trained_up: torch.Tensor
    initial_down: torch.Tensor
    initial_up: torch.Tensor
    original_alpha: float


class PiSSAModule(lora_anima.LoRAModule):
    supports_conv2d = True

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
        method,
        niter,
        oversample,
        resume_state,
    ):
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
        self._target_weight_shape = org_module.weight.shape
        if resume_state is None:
            matrix_down, matrix_up = _factor_matrix(
                org_module.weight,
                self.lora_dim,
                float(self.scale),
                method,
                niter,
                oversample,
            )
            initialized_down, initialized_up = _factor_shapes(
                matrix_down,
                matrix_up,
                self.lora_down.weight,
                self.lora_up.weight,
            )
            self.lora_down.weight.data.copy_(
                initialized_down.to(self.lora_down.weight)
            )
            self.lora_up.weight.data.copy_(
                initialized_up.to(self.lora_up.weight)
            )
            initial_down = initialized_down
            initial_up = initialized_up
            original_alpha = float(self.alpha.item())
        else:
            self._validate_resume_state(resume_state)
            self.lora_down.weight.data.copy_(
                resume_state.trained_down.to(self.lora_down.weight)
            )
            self.lora_up.weight.data.copy_(
                resume_state.trained_up.to(self.lora_up.weight)
            )
            initial_down = resume_state.initial_down
            initial_up = resume_state.initial_up
            original_alpha = float(resume_state.original_alpha)

        self.register_buffer(
            "pissa_initial_down",
            initial_down.detach().clone().to(self.lora_down.weight),
        )
        self.register_buffer(
            "pissa_initial_up",
            initial_up.detach().clone().to(self.lora_up.weight),
        )
        self.register_buffer(
            "pissa_original_alpha",
            torch.tensor(original_alpha, dtype=torch.float32),
        )
        self._pissa_residual_applied = False

    def _validate_resume_state(self, state: PiSSAResumeState) -> None:
        expected_down = tuple(self.lora_down.weight.shape)
        expected_up = tuple(self.lora_up.weight.shape)
        tensors = {
            "trained_down": (state.trained_down, expected_down),
            "trained_up": (state.trained_up, expected_up),
            "initial_down": (state.initial_down, expected_down),
            "initial_up": (state.initial_up, expected_up),
        }
        for field_name, (tensor, expected_shape) in tensors.items():
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"PiSSA {field_name} shape mismatch for {self.lora_name}: "
                    f"expected {expected_shape}, received {tuple(tensor.shape)}"
                )

    def apply_base_residual(self) -> None:
        if self._pissa_residual_applied:
            raise RuntimeError(
                f"PiSSA base residual already applied for {self.lora_name}"
            )
        delta = _compose_factors(
            self.pissa_initial_up,
            self.pissa_initial_down,
            self.org_module.weight.shape,
        )
        self.org_module.weight.data.sub_(
            delta.to(self.org_module.weight) * float(self.scale)
        )
        self._pissa_residual_applied = True

    def get_weight(self, multiplier) -> torch.Tensor:
        effective_multiplier = (
            self.multiplier if multiplier is None else float(multiplier)
        )
        delta = _compose_factors(
            self.lora_up.weight,
            self.lora_down.weight,
            self._target_weight_shape,
        )
        return delta * effective_multiplier * float(self.scale)

    def restore_resume_state(self, state: PiSSAResumeState) -> None:
        self._validate_resume_state(state)
        if hasattr(self, "org_module"):
            base_weight = self.org_module.weight
        elif hasattr(self, "org_forward"):
            base_weight = self.org_forward.__self__.weight
        else:
            raise RuntimeError(
                f"PiSSA module {self.lora_name} is not attached to a base layer"
            )
        current_initial = _compose_factors(
            self.pissa_initial_up,
            self.pissa_initial_down,
            base_weight.shape,
        )
        restored_initial = _compose_factors(
            state.initial_up,
            state.initial_down,
            base_weight.shape,
        )
        base_weight.data.add_(
            (current_initial - restored_initial).to(base_weight)
            * float(self.scale)
        )
        self.lora_down.weight.data.copy_(
            state.trained_down.to(self.lora_down.weight)
        )
        self.lora_up.weight.data.copy_(
            state.trained_up.to(self.lora_up.weight)
        )
        self.pissa_initial_down.copy_(
            state.initial_down.to(self.pissa_initial_down)
        )
        self.pissa_initial_up.copy_(
            state.initial_up.to(self.pissa_initial_up)
        )


class PiSSAModuleFactory:
    def __init__(
        self,
        method: str,
        niter: int,
        oversample: int,
        apply_conv2d: bool,
        resume_states: dict[str, PiSSAResumeState],
    ):
        self._method = method
        self._niter = niter
        self._oversample = oversample
        self.supports_conv2d = apply_conv2d
        self._resume_states = resume_states

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
    ) -> PiSSAModule:
        return PiSSAModule(
            lora_name,
            org_module,
            multiplier,
            lora_dim,
            alpha,
            dropout,
            rank_dropout,
            module_dropout,
            self._method,
            self._niter,
            self._oversample,
            self._resume_states.get(str(lora_name)),
        )


class PiSSANetwork(lora_anima.LoRANetwork):
    def __init__(
        self,
        *args,
        pissa_method: str,
        pissa_niter: int,
        pissa_oversample: int,
        pissa_apply_conv2d: bool,
        pissa_export_mode: str,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        modules = self.text_encoder_loras + self.unet_loras
        if not modules:
            raise ValueError(
                "PiSSA did not find any supported target modules"
            )
        for module in modules:
            module.apply_base_residual()

        self.pissa_method = pissa_method
        self.pissa_niter = pissa_niter
        self.pissa_oversample = pissa_oversample
        self.pissa_apply_conv2d = pissa_apply_conv2d
        self.pissa_export_mode = pissa_export_mode
        self.register_buffer(
            "pissa_method_code",
            torch.tensor(1 if pissa_method == "rsvd" else 0),
        )
        self.register_buffer("pissa_niter_value", torch.tensor(pissa_niter))
        self.register_buffer(
            "pissa_oversample_value",
            torch.tensor(pissa_oversample),
        )
        self.register_buffer(
            "pissa_apply_conv2d_value",
            torch.tensor(int(pissa_apply_conv2d)),
        )
        self.register_buffer(
            "pissa_export_mode_code",
            torch.tensor(1 if pissa_export_mode == FAST_EXPORT else 0),
        )

    def _state_dict_for_save(self):
        state = dict(self.state_dict())
        for module in self.text_encoder_loras + self.unet_loras:
            prefix = module.lora_name
            current_down = module.lora_down.weight.detach()
            current_up = module.lora_up.weight.detach()
            initial_down = module.pissa_initial_down.detach()
            initial_up = module.pissa_initial_up.detach()
            state[f"{prefix}.pissa_trained_down"] = current_down.clone()
            state[f"{prefix}.pissa_trained_up"] = current_up.clone()

            if self.pissa_export_mode == LOSSLESS_EXPORT:
                exported_down = torch.cat(
                    (current_down, initial_down),
                    dim=0,
                )
                exported_up = torch.cat(
                    (current_up, -initial_up),
                    dim=1,
                )
                exported_alpha = module.pissa_original_alpha * 2
            else:
                exported_down, exported_up = _approximate_difference(
                    current_down,
                    current_up,
                    initial_down,
                    initial_up,
                )
                exported_alpha = module.pissa_original_alpha

            state[f"{prefix}.lora_down.weight"] = exported_down.contiguous()
            state[f"{prefix}.lora_up.weight"] = exported_up.contiguous()
            state[f"{prefix}.alpha"] = exported_alpha.detach().clone()
        return state

    def load_weights(self, file):
        checkpoint = _load_checkpoint(file, None)
        restored = 0
        for module in self.text_encoder_loras + self.unet_loras:
            prefix = module.lora_name
            required = {
                "trained_down": f"{prefix}.pissa_trained_down",
                "trained_up": f"{prefix}.pissa_trained_up",
                "initial_down": f"{prefix}.pissa_initial_down",
                "initial_up": f"{prefix}.pissa_initial_up",
            }
            missing = [
                key
                for key in required.values()
                if key not in checkpoint
            ]
            if missing:
                raise ValueError(
                    f"PiSSA resume checkpoint is missing keys for {prefix}: "
                    + ", ".join(missing)
                )
            state = PiSSAResumeState(
                checkpoint[required["trained_down"]],
                checkpoint[required["trained_up"]],
                checkpoint[required["initial_down"]],
                checkpoint[required["initial_up"]],
                float(
                    checkpoint.get(
                        f"{prefix}.pissa_original_alpha",
                        module.pissa_original_alpha,
                    ).item()
                ),
            )
            module.restore_resume_state(state)
            restored += 1
        return {"restored_pissa_modules": restored}

    def save_weights(self, file, dtype, metadata):
        owned_metadata = dict(metadata or {})
        owned_metadata["ss_adapter_algorithm"] = "pissa"
        owned_metadata["ss_pissa_method"] = self.pissa_method
        owned_metadata["ss_pissa_export_mode"] = self.pissa_export_mode
        owned_metadata["ss_pissa_resume_state"] = "embedded"
        super().save_weights(file, dtype, owned_metadata)


def _create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae,
    text_encoders: list,
    unet,
    neuron_dropout: Optional[float],
    kwargs: dict[str, object],
    resume_states: dict[str, PiSSAResumeState],
):
    if not _parse_bool(kwargs.pop("pissa_init", True), "pissa_init"):
        raise ValueError(
            "pissa_anima.create_network requires pissa_init=true"
        )
    method = _parse_method(kwargs.pop("pissa_method", "rsvd"))
    niter = _parse_non_negative_int(
        kwargs.pop("pissa_niter", 2),
        "pissa_niter",
    )
    oversample = _parse_non_negative_int(
        kwargs.pop("pissa_oversample", 8),
        "pissa_oversample",
    )
    apply_conv2d = _parse_bool(
        kwargs.pop("pissa_apply_conv2d", False),
        "pissa_apply_conv2d",
    )
    export_mode = _parse_export_mode(
        kwargs.pop("pissa_export_mode", LOSSLESS_EXPORT)
    )
    module_factory = PiSSAModuleFactory(
        method,
        niter,
        oversample,
        apply_conv2d,
        resume_states,
    )
    network_factory = partial(
        PiSSANetwork,
        pissa_method=method,
        pissa_niter=niter,
        pissa_oversample=oversample,
        pissa_apply_conv2d=apply_conv2d,
        pissa_export_mode=export_mode,
    )
    kwargs["_module_class"] = module_factory
    kwargs["_network_factory"] = network_factory
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
    return _create_network(
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout,
        dict(kwargs),
        {},
    )


def _load_checkpoint(
    file: str | os.PathLike[str] | None,
    weights_sd: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    if weights_sd is not None:
        return dict(weights_sd)
    if file is None:
        raise ValueError(
            "PiSSA create_network_from_weights requires file or weights_sd"
        )
    checkpoint_path = Path(file)
    if checkpoint_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return dict(load_file(str(checkpoint_path)))
    loaded = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(loaded, dict):
        raise TypeError(
            "PiSSA checkpoint must contain a tensor state dictionary, "
            f"received {type(loaded).__name__}"
        )
    return loaded


def _resume_states(
    checkpoint: dict[str, torch.Tensor],
) -> dict[str, PiSSAResumeState]:
    prefixes = {
        key[: -len(".pissa_trained_down")]
        for key in checkpoint
        if key.endswith(".pissa_trained_down")
    }
    if not prefixes:
        raise ValueError(
            "PiSSA checkpoint does not contain embedded resume state"
        )
    states: dict[str, PiSSAResumeState] = {}
    for prefix in prefixes:
        required = (
            f"{prefix}.pissa_trained_down",
            f"{prefix}.pissa_trained_up",
            f"{prefix}.pissa_initial_down",
            f"{prefix}.pissa_initial_up",
            f"{prefix}.pissa_original_alpha",
        )
        missing = [key for key in required if key not in checkpoint]
        if missing:
            raise ValueError(
                f"PiSSA checkpoint is incomplete for {prefix}: "
                + ", ".join(missing)
            )
        states[prefix] = PiSSAResumeState(
            checkpoint[required[0]],
            checkpoint[required[1]],
            checkpoint[required[2]],
            checkpoint[required[3]],
            float(checkpoint[required[4]].item()),
        )
    return states


def _uniform_value(values: set[object], field_name: str) -> object:
    if len(values) != 1:
        raise ValueError(
            f"PiSSA checkpoint contains mixed {field_name}: {sorted(values)}"
        )
    return values.pop()


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
    states = _resume_states(checkpoint)
    ranks = {int(state.trained_down.shape[0]) for state in states.values()}
    alphas = {float(state.original_alpha) for state in states.values()}
    rank = int(_uniform_value(ranks, "rank"))
    alpha = float(_uniform_value(alphas, "alpha"))
    method_code = int(checkpoint.get("pissa_method_code", torch.tensor(1)).item())
    export_code = int(
        checkpoint.get("pissa_export_mode_code", torch.tensor(0)).item()
    )
    kwargs.update(
        {
            "pissa_init": True,
            "pissa_method": "rsvd" if method_code == 1 else "svd",
            "pissa_niter": int(
                checkpoint.get("pissa_niter_value", torch.tensor(2)).item()
            ),
            "pissa_oversample": int(
                checkpoint.get(
                    "pissa_oversample_value",
                    torch.tensor(8),
                ).item()
            ),
            "pissa_apply_conv2d": bool(
                checkpoint.get(
                    "pissa_apply_conv2d_value",
                    torch.tensor(0),
                ).item()
            ),
            "pissa_export_mode": (
                FAST_EXPORT if export_code == 1 else LOSSLESS_EXPORT
            ),
        }
    )
    network = _create_network(
        multiplier,
        rank,
        alpha,
        ae,
        text_encoders,
        unet,
        None,
        dict(kwargs),
        states,
    )
    return network, checkpoint
