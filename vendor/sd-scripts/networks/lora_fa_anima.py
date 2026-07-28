from __future__ import annotations

from collections.abc import Callable, Iterable
import math
import re
from typing import Optional

import torch

from networks import lora_anima


LoRAFAPair = tuple[
    torch.nn.Parameter,
    torch.nn.Parameter,
    float,
    str,
]


def correct_lorafa_gradient(
    matrix_a: torch.Tensor,
    gradient_b: torch.Tensor,
    scaling: float,
    damping: float,
) -> torch.Tensor:
    if matrix_a.ndim != 2:
        raise ValueError(
            f"LoRA-FA matrix A must be two-dimensional, received shape {tuple(matrix_a.shape)}"
        )
    if gradient_b.ndim != 2:
        raise ValueError(
            f"LoRA-FA gradient B must be two-dimensional, received shape {tuple(gradient_b.shape)}"
        )
    if gradient_b.shape[1] != matrix_a.shape[0]:
        raise ValueError(
            "LoRA-FA rank mismatch: "
            f"gradient B shape {tuple(gradient_b.shape)} and "
            f"matrix A shape {tuple(matrix_a.shape)}"
        )
    if scaling == 0:
        raise ValueError("LoRA-FA scaling must be non-zero")
    if damping <= 0:
        raise ValueError(
            f"LoRA-FA damping must be greater than zero, received {damping}"
        )

    matrix_a_fp32 = matrix_a.float()
    gradient_b_fp32 = gradient_b.float()
    gram = matrix_a_fp32 @ matrix_a_fp32.T
    identity = torch.eye(
        matrix_a_fp32.shape[0],
        device=matrix_a_fp32.device,
        dtype=matrix_a_fp32.dtype,
    )
    inverse = torch.linalg.pinv(gram + damping * identity)
    corrected = gradient_b_fp32 @ inverse / scaling**2
    return corrected.to(dtype=gradient_b.dtype)


def _validated_pairs(group: dict[str, object]) -> tuple[LoRAFAPair, ...]:
    raw_pairs = group.get("lorafa_pairs")
    if not isinstance(raw_pairs, (list, tuple)) or not raw_pairs:
        raise ValueError(
            "LoRAFAAdamW requires every parameter group to contain non-empty "
            "'lorafa_pairs' metadata"
        )
    pairs: list[LoRAFAPair] = []
    for raw_pair in raw_pairs:
        if not isinstance(raw_pair, tuple) or len(raw_pair) != 4:
            raise TypeError(
                "Each LoRA-FA pair must be a tuple of "
                "(matrix_a, matrix_b, scaling, module_name)"
            )
        matrix_a, matrix_b, scaling, module_name = raw_pair
        if not isinstance(matrix_a, torch.nn.Parameter):
            raise TypeError("LoRA-FA matrix A must be torch.nn.Parameter")
        if not isinstance(matrix_b, torch.nn.Parameter):
            raise TypeError("LoRA-FA matrix B must be torch.nn.Parameter")
        if not isinstance(scaling, (float, int)):
            raise TypeError("LoRA-FA scaling must be numeric")
        if not isinstance(module_name, str) or not module_name:
            raise TypeError("LoRA-FA module name must be a non-empty string")
        pairs.append((matrix_a, matrix_b, float(scaling), module_name))
    return tuple(pairs)


class LoRAFAAdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[dict[str, object]],
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
        correct_bias: bool,
        damping: float,
    ):
        if lr < 0:
            raise ValueError(f"LoRAFAAdamW lr must be non-negative, received {lr}")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError(
                f"LoRAFAAdamW betas must be in [0, 1), received {betas}"
            )
        if eps < 0:
            raise ValueError(f"LoRAFAAdamW eps must be non-negative, received {eps}")
        if weight_decay < 0:
            raise ValueError(
                "LoRAFAAdamW weight_decay must be non-negative, "
                f"received {weight_decay}"
            )
        if damping <= 0:
            raise ValueError(
                f"LoRAFAAdamW damping must be greater than zero, received {damping}"
            )

        param_groups = list(params)
        for group in param_groups:
            _validated_pairs(group)
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
            "damping": damping,
        }
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable[[], torch.Tensor]] = None,
    ) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            pairs = _validated_pairs(group)
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])
            correct_bias = bool(group["correct_bias"])
            damping = float(group["damping"])

            for matrix_a, matrix_b, scaling, _module_name in pairs:
                if matrix_b.grad is None:
                    continue
                corrected = correct_lorafa_gradient(
                    matrix_a,
                    matrix_b.grad,
                    scaling,
                    damping,
                ).float()
                state = self.state[matrix_b]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(matrix_b, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(
                        matrix_b,
                        dtype=torch.float32,
                    )

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(corrected, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    corrected,
                    corrected,
                    value=1 - beta2,
                )
                denominator = exp_avg_sq.sqrt().add_(eps)
                step_size = lr
                if correct_bias:
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    step_size *= math.sqrt(bias_correction2) / bias_correction1

                updated = matrix_b.float()
                if weight_decay > 0:
                    updated.mul_(1 - lr * weight_decay)
                updated.addcdiv_(exp_avg, denominator, value=-step_size)
                matrix_b.copy_(updated.to(dtype=matrix_b.dtype))

        return loss


class LoRAFAModule(lora_anima.LoRAModule):
    supports_conv2d = False

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
        self.lora_down.weight.requires_grad_(False)


class LoRAFANetwork(lora_anima.LoRANetwork):
    def _freeze_down_matrices(self) -> None:
        self.requires_grad_(True)
        for module in self.text_encoder_loras + self.unet_loras:
            module.lora_down.weight.requires_grad_(False)

    def _module_lr(self, module: LoRAFAModule, base_lr: float) -> float:
        if self.reg_lrs is None:
            return base_lr
        for pattern, learning_rate in self.reg_lrs.items():
            if re.fullmatch(pattern, module.original_name):
                return float(learning_rate)
        return base_lr

    def _groups_for_modules(
        self,
        modules: list[LoRAFAModule],
        base_lr: float,
    ) -> list[dict[str, object]]:
        pairs_by_lr: dict[float, list[LoRAFAPair]] = {}
        for module in modules:
            learning_rate = self._module_lr(module, base_lr)
            if learning_rate <= 0:
                continue
            pair = (
                module.lora_down.weight,
                module.lora_up.weight,
                float(module.scale),
                module.lora_name,
            )
            pairs_by_lr.setdefault(learning_rate, []).append(pair)

        groups: list[dict[str, object]] = []
        for learning_rate, pairs in pairs_by_lr.items():
            parameters = [
                parameter
                for matrix_a, matrix_b, _scaling, _name in pairs
                for parameter in (matrix_a, matrix_b)
            ]
            groups.append(
                {
                    "params": parameters,
                    "lr": learning_rate,
                    "lorafa_pairs": pairs,
                }
            )
        return groups

    def prepare_optimizer_params_with_multiple_te_lrs(
        self,
        text_encoder_lr,
        unet_lr,
        default_lr,
    ):
        self._freeze_down_matrices()
        if default_lr is None:
            raise ValueError("LoRA-FA requires an explicit learning_rate")
        default_learning_rate = float(default_lr)
        if isinstance(text_encoder_lr, list):
            te_learning_rate = (
                float(text_encoder_lr[0])
                if text_encoder_lr
                else default_learning_rate
            )
        elif text_encoder_lr is None:
            te_learning_rate = default_learning_rate
        else:
            te_learning_rate = float(text_encoder_lr)
        unet_learning_rate = (
            float(unet_lr)
            if unet_lr is not None
            else default_learning_rate
        )

        groups: list[dict[str, object]] = []
        descriptions: list[str] = []
        if self.text_encoder_loras:
            text_groups = self._groups_for_modules(
                list(self.text_encoder_loras),
                te_learning_rate,
            )
            groups.extend(text_groups)
            descriptions.extend(["textencoder lorafa"] * len(text_groups))
        if self.unet_loras:
            unet_groups = self._groups_for_modules(
                list(self.unet_loras),
                unet_learning_rate,
            )
            groups.extend(unet_groups)
            descriptions.extend(["unet lorafa"] * len(unet_groups))
        if not groups:
            raise ValueError("LoRA-FA did not find any supported Linear target modules")
        return groups, descriptions

    def prepare_grad_etc(self, text_encoder, unet):
        self._freeze_down_matrices()

    def get_trainable_params(self):
        return (
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def save_weights(self, file, dtype, metadata):
        owned_metadata = dict(metadata or {})
        owned_metadata["ss_adapter_algorithm"] = "lora_fa"
        owned_metadata["ss_lorafa_gradient_correction"] = "aat_pinv"
        owned_metadata["ss_lorafa_trainable_matrix"] = "lora_up"
        super().save_weights(file, dtype, owned_metadata)


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
    kwargs["_network_factory"] = LoRAFANetwork
    kwargs["_module_class"] = LoRAFAModule
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
    kwargs["_network_factory"] = LoRAFANetwork
    kwargs["_module_class"] = LoRAFAModule
    return lora_anima.create_network_from_weights(
        multiplier,
        file,
        ae,
        text_encoders,
        unet,
        weights_sd,
        for_inference,
        **kwargs,
    )
