from __future__ import annotations

import math
from functools import partial
from typing import Optional

import torch
from torch import nn

from networks import lora_anima


def _fit_divisor(dim: int, want: int) -> int:
    """want 若不能整除 dim，退到不超过 want 的最大整除数（最差为 1）。"""
    want = max(1, min(int(want), int(dim)))
    for d in range(want, 0, -1):
        if dim % d == 0:
            return d
    return 1


class CDKAModule(lora_anima.LoRAModule):
    """CDKA (Component Designed Kronecker Adapters), arXiv:2602.01267 (ICML 2026).

    ΔW = λ · Σᵢ₌₁ʳ B⁽ⁱ⁾ ⊗ A⁽ⁱ⁾
        A⁽ⁱ⁾ ∈ R^(r₁ × d_in/r₂)     Kaiming uniform 初始化（论文消融最优）
        B⁽ⁱ⁾ ∈ R^(d_out/r₁ × r₂)    零初始化 → 初始 ΔW = 0
        λ = α / √(r · r₂)            Theorem 3.4 的梯度稳定化缩放

    network_dim / network_alpha 对本算法无效——容量完全由 (r₁, r₂, r) 决定。
    存档键 (cdka_a / cdka_b / alpha) 与旧 vendored-lycoris 实现同构。
    """

    supports_conv2d = False

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier,
        lora_dim,
        alpha,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        cdka_r1=2,
        cdka_r2=8,
        cdka_r=4,
        cdka_alpha=16.0,
        rank_dropout_scale=False,
        bypass_mode=False,
    ) -> None:
        if not isinstance(org_module, nn.Linear):
            raise TypeError(
                "CDKA supports Linear modules only, received "
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
        # CDKA 不使用低秩 down/up 对；从模块树移除避免写入存档
        del self.lora_down
        del self.lora_up

        in_dim = org_module.in_features
        out_dim = org_module.out_features
        self.shape = (out_dim, in_dim)

        # network_args 到模块的值全是字符串，数值参数必须显式转换
        r1 = _fit_divisor(out_dim, int(cdka_r1))
        r2 = _fit_divisor(in_dim, int(cdka_r2))
        r = max(1, int(cdka_r))
        cdka_alpha = float(cdka_alpha)

        self.cdka_r1 = r1
        self.cdka_r2 = r2
        self.cdka_r = r
        self.out_blocks = out_dim // r1
        self.in_blocks = in_dim // r2

        # A: (r, r₁, d_in/r₂)  Kaiming uniform；B: (r, d_out/r₁, r₂) 置零
        self.cdka_a = nn.Parameter(torch.empty(r, r1, self.in_blocks))
        self.cdka_b = nn.Parameter(torch.zeros(r, self.out_blocks, r2))
        nn.init.kaiming_uniform_(self.cdka_a, a=math.sqrt(5))

        # λ = α / √(r · r₂)；覆盖基类按 network_alpha 注册的 buffer
        self.scale = cdka_alpha / math.sqrt(r * r2)
        self.alpha = torch.tensor(cdka_alpha)

        self.rank_dropout_scale = bool(rank_dropout_scale)
        self.bypass_mode = bool(bypass_mode)
        self.org_module_ref = [org_module]
        self.enabled = True

    def _delta_weight(self) -> torch.Tensor:
        # ΔW[u·r₁+q, v·(d_in/r₂)+t] = Σᵢ B[i,u,v] · A[i,q,t]（分块平铺 = Kronecker 积）
        weight = torch.einsum("iuv,iqt->uqvt", self.cdka_b, self.cdka_a)
        weight = weight.reshape(self.shape) * self.scale
        if self.training and self.rank_dropout:
            drop = (
                torch.rand(weight.size(0), device=weight.device)
                > float(self.rank_dropout)
            ).to(weight.dtype)
            drop = drop.view(-1, *[1] * (weight.dim() - 1))
            if self.rank_dropout_scale:
                drop = drop / drop.mean()
            weight = weight * drop
        return weight

    def get_weight(self, multiplier=None) -> torch.Tensor:
        effective = self.multiplier if multiplier is None else float(multiplier)
        return self._delta_weight().float() * effective

    def bypass_forward_diff(self, x: torch.Tensor, scale=1) -> torch.Tensor:
        # (B ⊗ A) x = vec(A X Bᵀ) 的等价 einsum 实现，不显式构造大矩阵
        x_blocks = x.reshape(*x.shape[:-1], self.cdka_r2, self.in_blocks)
        tmp = torch.einsum("...vt,iqt->...ivq", x_blocks, self.cdka_a)
        out = torch.einsum("iuv,...ivq->...uq", self.cdka_b, tmp)
        out = out.reshape(*out.shape[:-2], self.shape[0]) * self.scale
        if self.dropout is not None and self.training:
            out = torch.nn.functional.dropout(out, p=float(self.dropout))
        return out * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original = self.org_forward(x)
        if not self.enabled:
            return original
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < float(self.module_dropout):
                return original

        if self.bypass_mode:
            return original + self.bypass_forward_diff(x, self.multiplier)

        # 合并路径：ΔW 独立走一次 linear（不与 bf16 底模相加再相减），
        # 细小增量不会被底模的 ULP 吸收
        delta_weight = self._delta_weight().to(x.dtype)
        delta = torch.nn.functional.linear(x, delta_weight)
        return original + delta * self.multiplier

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        orig_norm = self._delta_weight().norm()
        norm = torch.clamp(orig_norm, max_norm / 2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = desired.cpu() / norm.cpu()

        scaled = bool(norm != desired)
        if scaled:
            self.cdka_a *= ratio**0.5
            self.cdka_b *= ratio**0.5
        return scaled, orig_norm * ratio


class CDKANetwork(lora_anima.LoRANetwork):
    def apply_max_norm_regularization(self, max_norm_value, device):
        keys_scaled = 0
        norms: list[float] = []
        for lora in self.text_encoder_loras + self.unet_loras:
            scaled, norm = lora.apply_max_norm(max_norm_value, device)
            norms.append(float(norm))
            if scaled:
                keys_scaled += 1
        if not norms:
            return keys_scaled, 0.0, 0.0
        return keys_scaled, sum(norms) / len(norms), max(norms)

    def save_weights(self, file, dtype, metadata):
        owned_metadata = dict(metadata or {})
        owned_metadata["ss_adapter_algorithm"] = "cdka"
        super().save_weights(file, dtype, owned_metadata)


def _module_factory(cdka_r1, cdka_r2, cdka_r, cdka_alpha, rank_dropout_scale, bypass_mode):
    factory = partial(
        CDKAModule,
        cdka_r1=cdka_r1,
        cdka_r2=cdka_r2,
        cdka_r=cdka_r,
        cdka_alpha=cdka_alpha,
        rank_dropout_scale=rank_dropout_scale,
        bypass_mode=bypass_mode,
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
    kwargs["_network_factory"] = CDKANetwork
    kwargs["_module_class"] = _module_factory(
        cdka_r1=kwargs.pop("cdka_r1", 2),
        cdka_r2=kwargs.pop("cdka_r2", 8),
        cdka_r=kwargs.pop("cdka_r", 4),
        cdka_alpha=kwargs.pop("cdka_alpha", 16.0),
        rank_dropout_scale=lora_anima._is_true(kwargs.pop("rank_dropout_scale", False)),
        bypass_mode=lora_anima._is_true(kwargs.pop("bypass_mode", False)),
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
    raise NotImplementedError(
        "CDKA cannot rebuild its (cdka_r1, cdka_r2, cdka_r) geometry from a weights "
        "file alone. Recreate the network with the original cdka_* settings and load "
        "the archive via network_weights instead of dim_from_weights."
    )
