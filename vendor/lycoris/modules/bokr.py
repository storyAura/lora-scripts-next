import math
from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LycorisBaseModule
from ..functional import factorization
from ..functional.lokr import make_kron
from ..logging import logger


@cache
def logging_force_full_matrix(lora_dim, dim, factor):
    logger.warning(
        f"lora_dim {lora_dim} is too large for"
        f" dim={dim} and {factor=}"
        ", using full matrix mode."
    )


class BokrModule(LycorisBaseModule):
    name = "bokr"
    # DiT 仅需支持 Linear 层
    support_module = {
        "linear",
    }
    weight_list = [
        "lokr_w1",
        "lokr_w1_a",
        "lokr_w1_b",
        "lokr_w2",
        "lokr_w2_a",
        "lokr_w2_b",
        "alpha",
        "bora_r_scale",
        "bora_c_scale",
    ]
    weight_list_det = ["lokr_w1", "lokr_w1_a"]

    @classmethod
    def algo_check(cls, state_dict, lora_name):
        # 键名与 LoKr 相同, 必须靠 BoRA 缩放键区分; 无 BoRA 的存档按 LoKr 加载即可
        has_lokr = any(f"{lora_name}.{k}" in state_dict for k in cls.weight_list_det)
        return has_lokr and f"{lora_name}.bora_r_scale" in state_dict

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
        use_scalar=False,
        decompose_both=False,
        factor: int = -1,  # Kronecker 分解因子
        rank_dropout_scale=False,
        weight_decompose=False,
        full_matrix=False,
        bypass_mode=None,
        rs_lora=False,
        unbalanced_factorization=False,
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
            raise ValueError(f"{self.module_type} is not supported in BoKr algo for DiT.")

        factor = int(factor)
        self.lora_dim = lora_dim
        self.use_w1 = False
        self.use_w2 = False
        self.full_matrix = full_matrix
        self.rs_lora = rs_lora

        # 针对 Transformer/Linear 的维度提取
        in_dim = org_module.in_features
        out_dim = org_module.out_features
        self.shape = (out_dim, in_dim)

        in_m, in_n = factorization(in_dim, factor)
        out_l, out_k = factorization(out_dim, factor)
        if unbalanced_factorization:
            out_l, out_k = out_k, out_l
        shape = ((out_l, out_k), (in_m, in_n))

        # W1 矩阵分解
        if (
            decompose_both
            and lora_dim < max(shape[0][0], shape[1][0]) / 2
            and not self.full_matrix
        ):
            self.lokr_w1_a = nn.Parameter(torch.empty(shape[0][0], lora_dim))
            self.lokr_w1_b = nn.Parameter(torch.empty(lora_dim, shape[1][0]))
        else:
            self.use_w1 = True
            self.lokr_w1 = nn.Parameter(torch.empty(shape[0][0], shape[1][0]))

        # W2 矩阵分解
        if lora_dim < max(shape[0][1], shape[1][1]) / 2 and not self.full_matrix:
            self.lokr_w2_a = nn.Parameter(torch.empty(shape[0][1], lora_dim))
            self.lokr_w2_b = nn.Parameter(torch.empty(lora_dim, shape[1][1]))
        else:
            if not self.full_matrix:
                logging_force_full_matrix(lora_dim, max(in_dim, out_dim), factor)
            self.use_w2 = True
            self.lokr_w2 = nn.Parameter(torch.empty(shape[0][1], shape[1][1]))

        # BoRA 2D 权重分解初始化
        self.wd = weight_decompose
        if self.wd:
            org_weight = org_module.weight.cpu().clone().float()
            # 行向缩放参数 (bora_r_scale): 对应 out_features, 形状为 (out_dim, 1)
            self.bora_r_scale = nn.Parameter(
                org_weight.norm(dim=1, keepdim=True)
            ).float()
            
            # 列向缩放参数 (bora_c_scale): 对应 in_features, 形状为 (1, in_dim)
            self.bora_c_scale = nn.Parameter(
                org_weight.norm(dim=0, keepdim=True)
            ).float()
        else:
            self.bora_r_scale = None
            self.bora_c_scale = None

        self.dropout = dropout
        if dropout:
            print("[WARN] LoKr/BoKr haven't implemented normal dropout yet.")
        self.rank_dropout = rank_dropout
        self.rank_dropout_scale = rank_dropout_scale
        self.module_dropout = module_dropout

        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().float().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        if self.use_w2 and self.use_w1:
            alpha = lora_dim

        r_factor = lora_dim
        if self.rs_lora:
            r_factor = math.sqrt(r_factor)

        self.scale = alpha / r_factor

        self.register_buffer("alpha", torch.tensor(alpha * (lora_dim / r_factor)))

        if use_scalar:
            self.scalar = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("scalar", torch.tensor(1.0), persistent=False)

        # 权重初始化
        if self.use_w2:
            if use_scalar:
                torch.nn.init.kaiming_uniform_(self.lokr_w2, a=math.sqrt(5))
            else:
                torch.nn.init.constant_(self.lokr_w2, 0)
        else:
            torch.nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
            if use_scalar:
                torch.nn.init.kaiming_uniform_(self.lokr_w2_b, a=math.sqrt(5))
            else:
                torch.nn.init.constant_(self.lokr_w2_b, 0)

        if self.use_w1:
            torch.nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
        else:
            torch.nn.init.kaiming_uniform_(self.lokr_w1_a, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.lokr_w1_b, a=math.sqrt(5))

    @classmethod
    def make_module_from_state_dict(
        cls,
        lora_name,
        orig_module,
        w1,
        w1a,
        w1b,
        w2,
        w2a,
        w2b,
        alpha,
        bora_r_scale=None,
        bora_c_scale=None,
        **kwargs,
    ):
        full_matrix = False
        if w1a is not None:
            lora_dim = w1a.size(1)
        elif w2a is not None:
            lora_dim = w2a.size(1)
        else:
            full_matrix = True
            lora_dim = 1

        factor1 = max(w1.shape) if w1 is not None else max(w1a.size(0), w1b.size(1))
        factor2 = max(w2.shape) if w2 is not None else max(w2a.size(0), w2b.size(1))
        factor = min(factor1, factor2)

        module = cls(
            lora_name,
            orig_module,
            1,
            lora_dim,
            float(alpha),
            decompose_both=w1 is None and w2 is None,
            factor=factor,
            weight_decompose=bora_r_scale is not None,
            full_matrix=full_matrix,
        )
        if w1 is not None:
            module.lokr_w1.copy_(w1)
        else:
            module.lokr_w1_a.copy_(w1a)
            module.lokr_w1_b.copy_(w1b)
        if w2 is not None:
            module.lokr_w2.copy_(w2)
        else:
            module.lokr_w2_a.copy_(w2a)
            module.lokr_w2_b.copy_(w2b)
        if bora_r_scale is not None:
            module.bora_r_scale.copy_(bora_r_scale)
        if bora_c_scale is not None:
            module.bora_c_scale.copy_(bora_c_scale)
        return module

    def load_weight_hook(self, module: nn.Module, incompatible_keys):
        missing_keys = incompatible_keys.missing_keys
        for key in missing_keys:
            if "scalar" in key:
                del missing_keys[missing_keys.index(key)]
        if isinstance(self.scalar, nn.Parameter):
            self.scalar.data.copy_(torch.ones_like(self.scalar))
        elif getattr(self, "scalar", None) is not None:
            self.scalar.copy_(torch.ones_like(self.scalar))
        else:
            self.register_buffer(
                "scalar", torch.ones_like(self.scalar), persistent=False
            )

    def get_weight(self, shape=None):
        w1 = self.lokr_w1 if self.use_w1 else self.lokr_w1_a @ self.lokr_w1_b
        w2 = self.lokr_w2 if self.use_w2 else self.lokr_w2_a @ self.lokr_w2_b
        
        weight = make_kron(w1, w2, self.scale)
        dtype = weight.dtype
        if shape is not None:
            weight = weight.view(shape)
            
        if self.training and self.rank_dropout:
            drop = (torch.rand(weight.size(0), device=weight.device) > self.rank_dropout).to(dtype)
            drop = drop.view(-1, *[1] * len(weight.shape[1:]))
            if self.rank_dropout_scale:
                drop /= drop.mean()
            weight *= drop
        return weight

    def get_diff_weight(self, multiplier=1, shape=None, device=None):
        # 修正：get_weight() 已包含 self.scale，此处仅乘 multiplier
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
        if self.wd:
            merged = self.apply_weight_decompose(
                weight.float() + diff.float()
            ).to(weight.dtype)
        else:
            merged = weight + diff
        return merged, None

    def apply_weight_decompose(self, weight):
        """
        专为 Linear 层 2D 权重设计的 BoRA 双向分解 (Row & Column Scaling)
        weight shape: (out_features, in_features)
        """
        target_dtype = weight.dtype
        weight = weight.to(self.bora_r_scale.dtype)

        # 1. 行向归一化 (Row-wise scaling) -> (out_dim, 1)
        weight_norm_r = weight.norm(dim=1, keepdim=True) + torch.finfo(weight.dtype).eps
        scale_r = self.bora_r_scale.to(weight.device) / weight_norm_r
        weight_r = weight * scale_r

        # 2. 列向归一化 (Column-wise scaling) -> (1, in_dim)
        weight_norm_c = weight_r.norm(dim=0, keepdim=True) + torch.finfo(weight_r.dtype).eps
        scale_c = self.bora_c_scale.to(weight.device) / weight_norm_c

        return (weight_r * scale_c).to(target_dtype)

    def custom_state_dict(self):
        destination = {}
        destination["alpha"] = self.alpha
        if self.wd:
            destination["bora_r_scale"] = self.bora_r_scale
            destination["bora_c_scale"] = self.bora_c_scale
        if self.use_w1:
            destination["lokr_w1"] = self.lokr_w1 * self.scalar
        else:
            destination["lokr_w1_a"] = self.lokr_w1_a * self.scalar
            destination["lokr_w1_b"] = self.lokr_w1_b

        if self.use_w2:
            destination["lokr_w2"] = self.lokr_w2
        else:
            destination["lokr_w2_a"] = self.lokr_w2_a
            destination["lokr_w2_b"] = self.lokr_w2_b
        return destination

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        orig_norm = self.get_weight(self.shape).norm()
        norm = torch.clamp(orig_norm, max_norm / 2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = desired.cpu() / norm.cpu()

        scaled = norm != desired
        if scaled:
            modules = 4 - self.use_w1 - self.use_w2
            if self.use_w1:
                self.lokr_w1 *= ratio ** (1 / modules)
            else:
                self.lokr_w1_a *= ratio ** (1 / modules)
                self.lokr_w1_b *= ratio ** (1 / modules)

            if self.use_w2:
                self.lokr_w2 *= ratio ** (1 / modules)
            else:
                self.lokr_w2_a *= ratio ** (1 / modules)
                self.lokr_w2_b *= ratio ** (1 / modules)

        return scaled, orig_norm * ratio

    def bypass_forward_diff(self, h, scale=1):
        if self.wd:
            raise NotImplementedError("Bypass mode does not support weight decomposition (BoRA).")

        w1 = self.lokr_w1 if self.use_w1 else self.lokr_w1_a @ self.lokr_w1_b
        w2 = self.lokr_w2 if self.use_w2 else self.lokr_w2_a @ self.lokr_w2_b
        
        in_m = w1.size(1)
        # h: (..., in_dim) -> (..., in_m, in_n)
        h_in_group = h.reshape(*h.shape[:-1], in_m, -1)

        # 运算 W2
        if self.use_w2:
            hb = F.linear(h_in_group, w2)
        else:
            ha = F.linear(h_in_group, self.lokr_w2_b)
            hb = F.linear(ha, self.lokr_w2_a)

        # 转置做 Kron 交叉维度运算
        h_cross_group = hb.transpose(-1, -2)
        hc = F.linear(h_cross_group, w1)

        # 转置并恢复形状 -> (..., out_dim)
        hc = hc.transpose(-1, -2)
        out = hc.reshape(*hc.shape[:-2], -1)

        return self.drop(out * scale * self.scalar)

    def bypass_forward(self, x, scale=1):
        return self.org_forward(x) + self.bypass_forward_diff(x, scale=scale)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x, *args, **kwargs)

        if self.bypass_mode and not self.wd:
            return self.bypass_forward(x, self.multiplier)

        base = self.org_forward(x, *args, **kwargs)
        base_weight = self._current_weight().to(x.device)
        
        diff_weight = self.get_weight(self.shape).float() * self.scalar.float()

        if self.wd:
            base_weight_f32 = base_weight.float()
            scaled_diff = diff_weight * self.multiplier
            new_weight_f32 = self.apply_weight_decompose(base_weight_f32 + scaled_diff)
            delta_weight = (new_weight_f32 - base_weight_f32).to(base_weight.dtype)
        elif self.multiplier == 1:
            delta_weight = diff_weight.to(base_weight.dtype)
        else:
            delta_weight = (diff_weight * self.multiplier).to(base_weight.dtype)

        delta = self.op(x, delta_weight, None, **self.kw_dict)
        return base + delta