# -*- coding: utf-8 -*-
"""CDKA (Component Designed Kronecker Adapters), arXiv:2602.01267 (ICML 2026).

ΔW = λ · Σᵢ₌₁ʳ B⁽ⁱ⁾ ⊗ A⁽ⁱ⁾
    A⁽ⁱ⁾ ∈ R^(r₁ × d_in/r₂)     Kaiming uniform 初始化（论文消融最优）
    B⁽ⁱ⁾ ∈ R^(d_out/r₁ × r₂)    零初始化 → 初始 ΔW = 0
    λ = α / √(r · r₂)            Theorem 3.4 的梯度稳定化缩放

论文设计原则：r₁ 取小 (2~4)，r₂ 取大，r 中等 (≥ r*，经验 2~8)。
默认 (r₁=2, r₂=8, r=4, α=16) 即论文推荐配置。
network_dim / network_alpha 对本算法无效——容量完全由 (r₁, r₂, r) 决定。
"""
import math

import torch
import torch.nn as nn

from .base import LycorisBaseModule


def _fit_divisor(dim: int, want: int) -> int:
    """want 若不能整除 dim，退到不超过 want 的最大整除数（最差为 1）。"""
    want = max(1, min(int(want), int(dim)))
    for d in range(want, 0, -1):
        if dim % d == 0:
            return d
    return 1


class CDKAModule(LycorisBaseModule):
    name = "cdka"
    # DiT 仅需支持 Linear 层
    support_module = {
        "linear",
    }
    weight_list = [
        "cdka_a",
        "cdka_b",
        "alpha",
    ]
    weight_list_det = ["cdka_a"]

    @classmethod
    def algo_check(cls, state_dict, lora_name):
        return f"{lora_name}.cdka_a" in state_dict

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=0.0,
        rank_dropout=0.0,
        module_dropout=0.0,
        use_tucker=False,   # 第 9 个位置参数必须为 use_tucker，兼容 kohya.py 位置传参（Linear-only 用不到）
        cdka_r1=2,
        cdka_r2=8,
        cdka_r=4,
        cdka_alpha=16.0,
        rank_dropout_scale=False,
        bypass_mode=None,
        **kwargs,
    ):
        super().__init__(
            lora_name,
            org_module,
            multiplier,
            dropout,
            rank_dropout,
            module_dropout,
            rank_dropout_scale,
            bypass_mode,
        )
        if self.module_type not in self.support_module:
            raise ValueError(f"{self.module_type} is not supported in CDKA algo for DiT.")

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
        torch.nn.init.kaiming_uniform_(self.cdka_a, a=math.sqrt(5))

        # λ = α / √(r · r₂)
        self.scale = cdka_alpha / math.sqrt(r * r2)
        self.register_buffer("alpha", torch.tensor(cdka_alpha))

        self.rank_dropout = rank_dropout
        self.rank_dropout_scale = rank_dropout_scale
        self.module_dropout = module_dropout

    @classmethod
    def make_module_from_state_dict(
        cls, lora_name, orig_module, cdka_a, cdka_b, alpha, **kwargs
    ):
        module = cls(
            lora_name,
            orig_module,
            1,
            cdka_r1=cdka_a.size(1),
            cdka_r2=cdka_b.size(2),
            cdka_r=cdka_a.size(0),
            cdka_alpha=float(alpha),
        )
        module.cdka_a.copy_(cdka_a)
        module.cdka_b.copy_(cdka_b)
        return module

    def get_weight(self, shape=None):
        # ΔW[u·r₁+q, v·(d_in/r₂)+t] = Σᵢ B[i,u,v] · A[i,q,t]  （分块平铺 = Kronecker 积）
        weight = torch.einsum("iuv,iqt->uqvt", self.cdka_b, self.cdka_a)
        weight = weight.reshape(self.shape) * self.scale
        if shape is not None:
            weight = weight.view(shape)
        if self.training and self.rank_dropout:
            drop = (
                torch.rand(weight.size(0), device=weight.device) > self.rank_dropout
            ).to(weight.dtype)
            drop = drop.view(-1, *[1] * len(weight.shape[1:]))
            if self.rank_dropout_scale:
                drop /= drop.mean()
            weight *= drop
        return weight

    def get_diff_weight(self, multiplier=1, shape=None, device=None):
        diff = self.get_weight(shape) * multiplier
        if device is not None:
            diff = diff.to(device)
        return diff, None

    def get_merged_weight(self, multiplier=1, shape=None, device=None):
        shape = shape or self.shape
        diff = self.get_diff_weight(multiplier=multiplier, shape=shape, device=device)[0]
        weight = self.org_weight
        if device is not None:
            weight = weight.to(device)
        return weight + diff.to(weight.dtype), None

    def custom_state_dict(self):
        return {
            "alpha": self.alpha,
            "cdka_a": self.cdka_a,
            "cdka_b": self.cdka_b,
        }

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        orig_norm = self.get_weight(self.shape).norm()
        norm = torch.clamp(orig_norm, max_norm / 2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = desired.cpu() / norm.cpu()

        scaled = norm != desired
        if scaled:
            self.cdka_a *= ratio**0.5
            self.cdka_b *= ratio**0.5
        return scaled, orig_norm * ratio

    def bypass_forward_diff(self, h, scale=1):
        # (B ⊗ A) x = vec(A X Bᵀ) 的等价 einsum 实现，不显式构造大矩阵
        x_blocks = h.reshape(*h.shape[:-1], self.cdka_r2, self.in_blocks)
        tmp = torch.einsum("...vt,iqt->...ivq", x_blocks, self.cdka_a)
        out = torch.einsum("iuv,...ivq->...uq", self.cdka_b, tmp)
        out = out.reshape(*out.shape[:-2], self.shape[0]) * self.scale
        return self.drop(out * scale)

    def bypass_forward(self, x, scale=1):
        return self.org_forward(x) + self.bypass_forward_diff(x, scale=scale)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x, *args, **kwargs)

        if self.bypass_mode:
            return self.bypass_forward(x, self.multiplier)

        base = self.org_forward(x, *args, **kwargs)
        base_weight = self._current_weight().to(x.device)

        diff_weight = self.get_weight(self.shape).float()
        if self.multiplier == 1:
            delta_weight = diff_weight.to(base_weight.dtype)
        else:
            delta_weight = (diff_weight * self.multiplier).to(base_weight.dtype)

        delta = self.op(x, delta_weight, None, **self.kw_dict)
        return base + delta
