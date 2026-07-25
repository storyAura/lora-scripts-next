import math
from functools import cache
from math import log2

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LycorisBaseModule
from ..functional import power2factorization, tucker_weight_from_conv
from ..logging import logger


@cache
def log_butterfly_factorize(dim, factor, result):
    logger.info(
        f"Use BOFT({int(log2(result[1]))}, {result[0]//2})"
        f" (equivalent to factor={result[0]}) "
        f"for {dim=} and {factor=}"
    )


def butterfly_factor(dimension: int, factor: int = -1) -> tuple[int, int]:
    m, n = power2factorization(dimension, factor)
    if n == 0:
        raise ValueError(
            f"It is impossible to decompose {dimension} with factor {factor} under BOFT constraints."
        )
    log_butterfly_factorize(dimension, factor, (m, n))
    return m, n


class GLoRABOFTModule(LycorisBaseModule):
    name = "glora_boft"
    support_module = {
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
    }

    weight_list = [
        "oft_blocks",
        "rescale",
        "boft_alpha",
        "a1.weight",
        "a2.weight",
        "b1.weight",
        "b2.weight",
        "bm.weight",
        "alpha",
    ]
    weight_list_det = ["oft_blocks", "a1.weight"]

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
        use_tucker=False,
        use_scalar=False,
        rank_dropout_scale=False,
        constraint=0,
        rescaled=False,
        bypass_mode=None,
        rs_lora=False,
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
            raise ValueError(f"{self.module_type} is not supported in GLoRA-BOFT algo.")

        # =====================================================================
        # 1. 初始化 BOFT 结构
        # =====================================================================
        out_dim = self.dim
        b, m_exp = butterfly_factor(out_dim, lora_dim)
        self.block_size = b
        self.block_num = m_exp
        
        self.boft_b = b
        self.boft_m = sum(int(i) for i in f"{m_exp-1:b}") + 1
        self.rescaled = rescaled
        
        self.register_buffer("boft_alpha", torch.tensor(constraint * out_dim))
        
        self.oft_blocks = nn.Parameter(
            torch.zeros(self.boft_m, self.block_num, self.block_size, self.block_size)
        )
        if rescaled:
            self.rescale = nn.Parameter(
                torch.ones(out_dim, *(1 for _ in range(org_module.weight.dim() - 1)))
            )

        # =====================================================================
        # 2. 初始化 GLoRA 结构
        # =====================================================================
        self.lora_dim = lora_dim
        self.tucker = False
        self.rs_lora = rs_lora

        if self.module_type.startswith("conv"):
            self.isconv = True
            in_dim = org_module.in_channels
            k_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            out_dim = org_module.out_channels
            
            # Tucker 分解适用于卷积核尺寸大于 1 的情况
            self.tucker = use_tucker and any(i != 1 for i in k_size)
            self.down_op = self.op
            self.up_op = self.op

            self.a2 = self.module(in_dim, lora_dim, 1, bias=False)
            self.a1 = self.module(lora_dim, in_dim, 1, bias=False)

            if self.tucker:
                self.b2 = self.module(in_dim, lora_dim, 1, bias=False)
                self.bm = self.module(
                    lora_dim, lora_dim, k_size, stride, padding, bias=False
                )
            else:
                self.b2 = self.module(
                    in_dim, lora_dim, k_size, stride, padding, bias=False
                )
            self.b1 = self.module(lora_dim, out_dim, 1, bias=False)
        else:
            self.isconv = False
            self.down_op = F.linear
            self.up_op = F.linear
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            
            self.a2 = nn.Linear(in_dim, lora_dim, bias=False)
            self.a1 = nn.Linear(lora_dim, in_dim, bias=False)
            self.b2 = nn.Linear(in_dim, lora_dim, bias=False)
            self.b1 = nn.Linear(lora_dim, out_dim, bias=False)

        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().float().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha

        r_factor = lora_dim
        if self.rs_lora:
            r_factor = math.sqrt(r_factor)

        self.scale = alpha / r_factor
        self.register_buffer("alpha", torch.tensor(alpha))

        if use_scalar:
            self.scalar = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("scalar", torch.tensor(1.0), persistent=False)

        # 参数初始化
        torch.nn.init.kaiming_uniform_(self.a1.weight, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.b1.weight, a=math.sqrt(5))
        if use_scalar:
            torch.nn.init.kaiming_uniform_(self.a2.weight, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.b2.weight, a=math.sqrt(5))
        else:
            torch.nn.init.zeros_(self.a2.weight)
            torch.nn.init.zeros_(self.b2.weight)

    @classmethod
    def algo_check(cls, state_dict, lora_name):
        return f"{lora_name}.oft_blocks" in state_dict and f"{lora_name}.a1.weight" in state_dict

    @classmethod
    def make_module_from_state_dict(
        cls,
        lora_name,
        orig_module,
        oft_blocks,
        boft_alpha,
        a1,
        a2,
        b1,
        b2,
        alpha,
        rescale=None,
        bm=None,
        **kwargs,
    ):
        m, n, s, _ = oft_blocks.shape
        alpha_val = float(alpha.item()) if isinstance(alpha, torch.Tensor) else float(alpha)
        boft_alpha_val = float(boft_alpha.item()) if isinstance(boft_alpha, torch.Tensor) else float(boft_alpha)
        
        module = cls(
            lora_name=lora_name,
            org_module=orig_module,
            multiplier=1.0,
            lora_dim=s,
            alpha=alpha_val,
            constraint=boft_alpha_val / orig_module.weight.shape[0],
            rescaled=rescale is not None,
            use_tucker=bm is not None,
        )
        with torch.no_grad():
            module.oft_blocks.copy_(oft_blocks)
            if rescale is not None:
                module.rescale.copy_(rescale)
            
            module.a1.weight.copy_(a1)
            module.a2.weight.copy_(a2)
            module.b1.weight.copy_(b1)
            module.b2.weight.copy_(b2)
            if bm is not None and hasattr(module, "bm"):
                module.bm.weight.copy_(bm)
            
            module.boft_alpha.copy_(boft_alpha)
            module.alpha.copy_(alpha)
        return module

    def custom_state_dict(self):
        destination = {}
        destination["alpha"] = self.alpha
        destination["boft_alpha"] = self.boft_alpha
        destination["oft_blocks"] = self.oft_blocks
        if self.rescaled:
            destination["rescale"] = self.rescale
        destination["a1.weight"] = self.a1.weight
        destination["a2.weight"] = self.a2.weight * self.scalar
        destination["b1.weight"] = self.b1.weight
        destination["b2.weight"] = self.b2.weight * self.scalar
        if self.tucker:
            destination["bm.weight"] = self.bm.weight
        return destination

    def load_weight_hook(self, module: nn.Module, incompatible_keys):
        missing_keys = incompatible_keys.missing_keys
        keys_to_remove = [key for key in missing_keys if "scalar" in key]
        for key in keys_to_remove:
            missing_keys.remove(key)
            
        if isinstance(self.scalar, nn.Parameter):
            self.scalar.data.copy_(torch.ones_like(self.scalar))
        elif getattr(self, "scalar", None) is not None:
            self.scalar.copy_(torch.ones_like(self.scalar))
        else:
            self.register_buffer(
                "scalar", torch.ones_like(self.scalar), persistent=False
            )

    def get_r(self):
        # 统一使用 float32 进行高精度的 Cayley 变换求解
        q = self.oft_blocks.float() - self.oft_blocks.float().transpose(-1, -2)
        normed_q = q
        
        constraint = self.boft_alpha.item()
        if constraint > 0:
            q_norm = torch.norm(q) + 1e-8
            if q_norm > constraint:
                normed_q = q * constraint / q_norm

        I = torch.eye(self.block_size, device=self.oft_blocks.device, dtype=torch.float32)
        # 优化点：利用 linalg.solve 求解 (I - Q) R = I + Q 线性方程组，避免显式求逆
        r = torch.linalg.solve(I - normed_q, I + normed_q)
        return r

    def make_boft_weight(self, scale=1, device=None):
        m = self.boft_m
        b = self.boft_b
        r_b = b // 2
        r = self.get_r()
        
        if device is None:
            device = self.oft_blocks.device
        inp = self.org_weight.to(device, dtype=r.dtype).contiguous()

        for i in range(m):
            bi = r[i]
            g = 2
            k = 2**i * r_b
            if scale != 1:
                I = torch.eye(self.block_size, device=bi.device, dtype=bi.dtype)
                bi = bi * scale + (1 - scale) * I
            inp = (
                inp.unflatten(0, (-1, g, k))
                .transpose(1, 2)
                .flatten(0, 2)
                .unflatten(0, (-1, b))
            )
            inp = torch.einsum("b i j, b j ...-> b i ...", bi, inp)
            inp = (
                inp.flatten(0, 1).unflatten(0, (-1, k, g)).transpose(1, 2).flatten(0, 2)
            )

        if self.rescaled:
            inp = inp * self.rescale.to(device=inp.device, dtype=inp.dtype)

        return inp

    def make_weight(self, scale=1, device=None, diff=False):
        if device is None:
            device = self.oft_blocks.device
            
        w_org = self.org_weight.to(device).contiguous()
        w_boft = self.make_boft_weight(scale=scale, device=device).contiguous()

        wa1 = self.a1.weight.view(self.a1.weight.size(0), -1)
        wa2 = self.a2.weight.view(self.a2.weight.size(0), -1)

        if self.tucker:
            wb = tucker_weight_from_conv(self.b1.weight, self.b2.weight, self.bm.weight)
        else:
            wb1 = self.b1.weight.view(self.b1.weight.size(0), -1)
            wb2 = self.b2.weight.view(self.b2.weight.size(0), -1)
            wb = wb1 @ wb2
            wb = wb.view(*w_org.shape)

        wa1 = wa1.to(device=w_boft.device, dtype=w_boft.dtype)
        wa2 = wa2.to(device=w_boft.device, dtype=w_boft.dtype)
        wb = wb.to(device=w_boft.device, dtype=w_boft.dtype)

        # 优化点：GLoRA 自适应基于已经经过 BOFT 旋转的权重基底 w_boft，确保前向传播与权重合并的高度一致性
        if w_boft.dim() > 2:
            w_wa1 = torch.einsum("o i ..., i j -> o j ...", w_boft, wa1)
            w_wa2 = torch.einsum("o i ..., i j -> o j ...", w_wa1, wa2)
        else:
            w_wa2 = (w_boft @ wa1) @ wa2

        w_glora_delta = (wb + w_wa2) * (self.scale * self.scalar * scale)

        if diff:
            out = (w_boft - w_org) + w_glora_delta
        else:
            out = w_boft + w_glora_delta
            
        return out.to(self.oft_blocks.dtype)

    def get_diff_weight(self, multiplier=1, shape=None, device=None):
        diff = self.make_weight(scale=multiplier, device=device, diff=True)
        if shape is not None:
            diff = diff.view(shape)
        return diff, None

    def get_merged_weight(self, multiplier=1, shape=None, device=None):
        diff = self.make_weight(scale=multiplier, device=device, diff=False)
        if shape is not None:
            diff = diff.view(shape)
        return diff, None

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        orig_norm = self.oft_blocks.to(device).norm()
        norm = torch.clamp(orig_norm, max_norm / 2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = desired / norm

        scaled = norm != desired
        if scaled:
            self.oft_blocks *= ratio.to(self.oft_blocks.device)

        return scaled, orig_norm * ratio

    def boft_forward(self, x, scale=1, custom_input=None):
        """
        根据自定义输入(如 x + A(x))计算 BOFT 变换后的前向输出
        """
        m = self.boft_m
        b = self.boft_b
        r_b = b // 2
        r = self.get_r()
        
        inp = custom_input if custom_input is not None else self.org_forward(x)
        
        org_dtype = inp.dtype
        inp = inp.to(dtype=r.dtype)

        is_conv = self.op in {F.conv2d, F.conv1d, F.conv3d}
        if is_conv:
            inp = inp.transpose(1, -1)

        for i in range(m):
            bi = r[i]
            g = 2
            k = 2**i * r_b
            if scale != 1:
                I = torch.eye(self.block_size, device=bi.device, dtype=bi.dtype)
                bi = bi * scale + (1 - scale) * I
            
            # 优化点：移除多余的 .contiguous() 调用，充分利用 einsum 内部的非连续计算
            inp = (
                inp.unflatten(-1, (-1, g, k))
                .transpose(-2, -1)
                .flatten(-3)
                .unflatten(-1, (-1, b))
            )
            inp = torch.einsum("b i j, ... b j -> ... b i", bi, inp)
            inp = (
                inp.flatten(-2)
                .unflatten(-1, (-1, k, g))
                .transpose(-2, -1)
                .flatten(-3)
            )

        if self.rescaled:
            rescale_val = self.rescale.to(device=inp.device, dtype=inp.dtype)
            if is_conv:
                rescale_val = rescale_val.transpose(0, -1)
            inp = inp * rescale_val

        if is_conv:
            inp = inp.transpose(1, -1)
            
        return inp.to(dtype=org_dtype)

    def _bypass_forward(self, x, scale=1, diff=False):
        glora_scale = self.scale * self.scalar * scale
        
        # 1. 基础分流 - 计算 A 分支
        ax_mid = self.a2(x)
        if self.rank_dropout and self.training:
            # 修复点：将 `<` 修复为 `>=`，纠正 rank_dropout 的物理意义
            drop_a = (
                torch.rand(self.lora_dim, device=ax_mid.device) >= self.rank_dropout
            ).to(ax_mid.dtype)
            if self.rank_dropout_scale:
                drop_a /= (drop_a.mean() + 1e-8)
            
            drop_shape = [1, self.lora_dim] + [1] * (len(x.shape) - 2) if self.isconv else [1] * (len(x.shape) - 1) + [self.lora_dim]
            ax_mid = ax_mid * drop_a.view(*drop_shape)
            
        a_out = self.drop(self.a1(ax_mid)) * glora_scale

        # 2. 基础分流 - 计算 B 分支
        bx_mid = self.b2(x)
        if self.rank_dropout and self.training:
            drop_b = (
                torch.rand(self.lora_dim, device=bx_mid.device) >= self.rank_dropout
            ).to(bx_mid.dtype)
            if self.rank_dropout_scale:
                drop_b /= (drop_b.mean() + 1e-8)
                
            drop_shape = [1, self.lora_dim] + [1] * (len(x.shape) - 2) if self.isconv else [1] * (len(x.shape) - 1) + [self.lora_dim]
            bx_mid = bx_mid * drop_b.view(*drop_shape)

        if self.tucker:
            bx_mid = self.bm(bx_mid)
        b_out = self.drop(self.b1(bx_mid)) * glora_scale

        # 3. 统一融合前向：只需一次 base layer 前向调用
        # 数学等价于：f(x) = W_boft(X + A(X)) + B(X)
        merged_input = x + a_out
        
        if diff:
            # diff 前向：(W_boft - W)(X + A(X)) + B(X)
            org_out = self.org_forward(merged_input)
            boft_out = self.boft_forward(None, scale=scale, custom_input=org_out)
            out = (boft_out - org_out) + b_out
        else:
            # 正常前向
            out = self.boft_forward(merged_input, scale=scale) + b_out

        return out

    def bypass_forward_diff(self, x, scale=1):
        return self._bypass_forward(x, scale, diff=True)

    def bypass_forward(self, x, scale=1):
        return self._bypass_forward(x, scale, diff=False)

    def forward(self, x, *args, **kwargs):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x, *args, **kwargs)
        scale = self.multiplier

        if self.bypass_mode:
            return self.bypass_forward(x, scale)
        else:
            base = self.org_forward(x, *args, **kwargs)
            diff_weight, _ = self.get_diff_weight(multiplier=scale, device=x.device)
            diff_weight = diff_weight.to(device=x.device, dtype=self.org_weight.dtype)
            delta = self.op(x, weight=diff_weight, bias=None, **self.kw_dict)
            return base + delta