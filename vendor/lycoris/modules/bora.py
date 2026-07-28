import math
from functools import cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LycorisBaseModule
from .functional import compute_merged_delta
from ..functional.general import rebuild_tucker
from ..logging import logger


@cache
def log_wd():
    return logger.warning(
        "Using weight_decompose=True with LoRA (DoRA/BoRA) will ignore network_dropout. "
        "Only rank dropout and module dropout will be applied."
    )


# =====================================================================
# 1. 修复后的 LoConModule (支持常规 LoRA 与标准 DoRA)
# =====================================================================
class LoConModule(LycorisBaseModule):
    name = "locon"
    support_module = {
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
    }
    weight_list = [
        "lora_up.weight",
        "lora_down.weight",
        "lora_mid.weight",
        "alpha",
        "dora_scale",
    ]
    weight_list_det = ["lora_up.weight"]

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
        weight_decompose=False,
        wd_on_out=True,
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
            raise ValueError(f"{self.module_type} is not supported in LoRA/LoCon algo.")
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
            use_tucker = use_tucker and any(i != 1 for i in k_size)
            self.down_op = self.op
            self.up_op = self.op
            if use_tucker and any(i != 1 for i in k_size):
                self.lora_down = self.module(in_dim, lora_dim, 1, bias=False)
                self.lora_mid = self.module(
                    lora_dim, lora_dim, k_size, stride, padding, bias=False
                )
                self.tucker = True
            else:
                self.lora_down = self.module(
                    in_dim, lora_dim, k_size, stride, padding, bias=False
                )
            self.lora_up = self.module(lora_dim, out_dim, 1, bias=False)
        elif isinstance(org_module, nn.Linear):
            self.isconv = False
            self.down_op = F.linear
            self.up_op = F.linear
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)
        else:
            raise NotImplementedError

        self.wd = weight_decompose
        self.wd_on_out = wd_on_out
        if self.wd:
            org_weight = org_module.weight.cpu().clone().float()
            self.dora_norm_dims = org_weight.dim() - 1
            if self.wd_on_out:
                self.dora_scale = nn.Parameter(
                    torch.norm(
                        org_weight.reshape(org_weight.shape[0], -1),
                        dim=1,
                        keepdim=True,
                    ).reshape(org_weight.shape[0], *[1] * self.dora_norm_dims)
                ).float()
            else:
                self.dora_scale = nn.Parameter(
                    torch.norm(
                        org_weight.transpose(1, 0).reshape(org_weight.shape[1], -1),
                        dim=1,
                        keepdim=True,
                    )
                    .reshape(org_weight.shape[1], *[1] * self.dora_norm_dims)
                    .transpose(1, 0)
                ).float()

        # 解决权重分解与 bypass_mode 产生的数学逻辑冲突
        if self.wd and self.bypass_mode:
            logger.warning(
                f"Using bypass_mode with weight_decompose=True in {self.name} is not supported. "
                "bypass_mode has been disabled to ensure correct decomposition math."
            )
            self.bypass_mode = None

        if dropout:
            self.dropout = nn.Dropout(dropout)
            if self.wd:
                log_wd()
        else:
            self.dropout = nn.Identity()

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha

        r_factor = lora_dim
        if self.rs_lora:
            r_factor = math.sqrt(r_factor)

        self.scale = alpha / r_factor

        self.register_buffer("alpha", torch.tensor(alpha * (lora_dim / r_factor)))

        if use_scalar:
            self.scalar = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("scalar", torch.tensor(1.0), persistent=False)

        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        if use_scalar:
            torch.nn.init.kaiming_uniform_(self.lora_up.weight, a=math.sqrt(5))
        else:
            torch.nn.init.constant_(self.lora_up.weight, 0)
        if self.tucker:
            torch.nn.init.kaiming_uniform_(self.lora_mid.weight, a=math.sqrt(5))

    @classmethod
    def make_module_from_state_dict(
        cls, lora_name, orig_module, up, down, mid, alpha, dora_scale
    ):
        module = cls(
            lora_name,
            orig_module,
            1,
            down.size(0),
            float(alpha),
            use_tucker=mid is not None,
            weight_decompose=dora_scale is not None,
        )
        module.lora_up.weight.data.copy_(up)
        module.lora_down.weight.data.copy_(down)
        if mid is not None:
            module.lora_mid.weight.data.copy_(mid)
        if dora_scale is not None:
            module.dora_scale.copy_(dora_scale)
        return module

    def load_weight_hook(self, module: nn.Module, incompatible_keys):
        missing_keys = incompatible_keys.missing_keys
        is_missing = False
        for key in list(missing_keys):
            if "scalar" in key:
                missing_keys.remove(key)
                is_missing = True
        
        # 仅当检测到 checkpoint 缺失 scalar 时才执行默认初始化，防止覆盖载入的值
        if is_missing:
            if isinstance(self.scalar, nn.Parameter):
                self.scalar.data.copy_(torch.ones_like(self.scalar))
            elif getattr(self, "scalar", None) is not None:
                self.scalar.copy_(torch.ones_like(self.scalar))
            else:
                self.register_buffer(
                    "scalar", torch.ones_like(self.scalar), persistent=False
                )

    def make_weight(self, device=None):
        wa = self.lora_up.weight.to(device)
        wb = self.lora_down.weight.to(device)
        if self.tucker:
            t = self.lora_mid.weight.to(device)  # 修复: 确保 t 移动到指定设备
            wa = wa.view(wa.size(0), -1).transpose(0, 1)
            wb = wb.view(wb.size(0), -1)
            weight = rebuild_tucker(t, wa, wb)
        else:
            weight = wa.view(wa.size(0), -1) @ wb.view(wb.size(0), -1)

        weight = weight.view(self.shape)
        if self.training and self.rank_dropout:
            drop = (torch.rand(weight.size(0), device=device) > self.rank_dropout).to(
                weight.dtype
            )
            drop = drop.view(-1, *[1] * len(weight.shape[1:]))
            if self.rank_dropout_scale:
                drop /= drop.mean()
            weight *= drop

        return weight * self.scalar.to(device)

    def get_diff_weight(self, multiplier=1, shape=None, device=None):
        scale = self.scale * multiplier
        diff = self.make_weight(device=device) * scale
        if shape is not None:
            diff = diff.view(shape)
        if device is not None:
            diff = diff.to(device)
        return diff, None

    def get_merged_weight(self, multiplier=1, shape=None, device=None):
        # 先应用 multiplier 计算增量
        diff = self.get_diff_weight(multiplier=multiplier, shape=shape, device=device)[0]
        weight = self.org_weight
        if self.wd:
            merged = self.apply_weight_decompose(weight + diff)
        else:
            merged = weight + diff
        return merged, None

    def apply_weight_decompose(self, weight):
        # 移除内部 multiplier 插值，避免 multiplier=0 时的失效问题
        weight = weight.to(self.dora_scale.dtype)
        if self.wd_on_out:
            weight_norm = (
                weight.reshape(weight.shape[0], -1)
                .norm(dim=1)
                .reshape(weight.shape[0], *[1] * self.dora_norm_dims)
            ) + torch.finfo(weight.dtype).eps
        else:
            weight_norm = (
                weight.transpose(0, 1)
                .reshape(weight.shape[1], -1)
                .norm(dim=1, keepdim=True)
                .reshape(weight.shape[1], *[1] * self.dora_norm_dims)
                .transpose(0, 1)
            ) + torch.finfo(weight.dtype).eps

        scale = self.dora_scale.to(weight.device) / weight_norm
        return weight * scale

    def custom_state_dict(self):
        destination = {}
        if self.wd:
            destination["dora_scale"] = self.dora_scale
        destination["alpha"] = self.alpha
        destination["lora_up.weight"] = self.lora_up.weight * self.scalar
        destination["lora_down.weight"] = self.lora_down.weight
        if self.tucker:
            destination["lora_mid.weight"] = self.lora_mid.weight
        return destination

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        orig_norm = self.make_weight(device).norm() * self.scale
        norm = torch.clamp(orig_norm, max_norm / 2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = desired.cpu() / norm.cpu()

        scaled = norm != desired
        if scaled:
            self.scalar *= ratio

        return scaled, orig_norm * ratio

    def bypass_forward_diff(self, x, scale=1):
        if self.tucker:
            mid = self.lora_mid(self.lora_down(x))
        else:
            mid = self.lora_down(x)

        if self.rank_dropout and self.training:
            drop = (
                torch.rand(self.lora_dim, device=mid.device) > self.rank_dropout
            ).to(mid.dtype)
            if self.rank_dropout_scale:
                drop /= drop.mean()
            if (dims := len(x.shape)) == 4:
                drop = drop.view(1, -1, 1, 1)
            else:
                drop = drop.view(*[1] * (dims - 1), -1)
            mid = mid * drop

        return self.dropout(self.lora_up(mid) * self.scalar * self.scale * scale)

    def bypass_forward(self, x, scale=1):
        return self.org_forward(x) + self.bypass_forward_diff(x, scale=scale)

    def forward(self, x, *args, **kwargs):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x, *args, **kwargs)

        if self.bypass_mode:
            return self.bypass_forward(x, scale=self.multiplier)

        base = self.org_forward(x, *args, **kwargs)
        scale = self.scale
        device = x.device

        base_weight = self._current_weight().to(device)
        diff_weight = self.make_weight(device) * scale
        transform = (
            (lambda weight: self.apply_weight_decompose(weight))
            if self.wd
            else None
        )
        delta_weight = compute_merged_delta(
            base_weight,
            diff_weight,
            self.multiplier,
            transform,
        )
        delta = self.op(x, delta_weight, None, **self.kw_dict)
        return base + delta


# =====================================================================
# 2. 修复后的 BoRAModule (严格契合 BoRA 理论与双维度幅值分解公式 3)
# =====================================================================
class BoRAModule(LycorisBaseModule):
    name = "bora"
    support_module = {
        "linear",
    }
    weight_list = [
        "lora_up.weight",
        "lora_down.weight",
        "alpha",
        "bora_scale_r",
        "bora_scale_c",
    ]
    weight_list_det = ["lora_up.weight"]

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        *args,
        **kwargs,
    ):
        def get_arg(name, index, default):
            if len(args) > index:
                return args[index]
            return kwargs.get(name, default)

        dropout = get_arg("dropout", 0, 0.0)
        rank_dropout = get_arg("rank_dropout", 1, 0.0)
        module_dropout = get_arg("module_dropout", 2, 0.0)
        use_tucker = get_arg("use_tucker", 3, False)
        use_scalar = get_arg("use_scalar", 4, False)
        rank_dropout_scale = get_arg("rank_dropout_scale", 5, False)
        weight_decompose = get_arg("weight_decompose", 6, False)
        wd_on_out = get_arg("wd_on_out", 7, True)
        bypass_mode = get_arg("bypass_mode", 8, None)
        rs_lora = get_arg("rs_lora", 9, False)

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
            raise ValueError(f"{self.module_type} is not supported in BoRA.")
            
        self.lora_dim = lora_dim
        self.rs_lora = rs_lora
        self.tucker = False

        self.isconv = False
        self.down_op = F.linear
        self.up_op = F.linear
        in_dim = org_module.in_features
        out_dim = org_module.out_features
        self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
        self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)

        self.wd = weight_decompose
        if self.wd:
            org_weight = org_module.weight.cpu().clone().float()
            # 对齐理论公式:
            # m^r (bora_scale_r): 行幅值矩阵，计算 weight 的行(dim=1)范数，大小为 (out_features, 1)
            self.bora_scale_r = nn.Parameter(
                org_weight.norm(dim=1, keepdim=True)
            ).float()
            # m^c (bora_scale_c): 列幅值矩阵，计算 weight 的列(dim=0)范数，大小为 (1, in_features)
            self.bora_scale_c = nn.Parameter(
                org_weight.norm(dim=0, keepdim=True)
            ).float()

        # 解决双向归一化与 bypass 快速分支前向计算的冲突
        if self.wd and self.bypass_mode:
            logger.warning(
                f"Using bypass_mode with weight_decompose=True in {self.name} is not supported. "
                "bypass_mode has been disabled to ensure correct decomposition math."
            )
            self.bypass_mode = None

        if dropout:
            self.dropout = nn.Dropout(dropout)
            if self.wd:
                log_wd()
        else:
            self.dropout = nn.Identity()

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha

        r_factor = lora_dim
        if self.rs_lora:
            r_factor = math.sqrt(r_factor)

        self.scale = alpha / r_factor

        self.register_buffer("alpha", torch.tensor(alpha * (lora_dim / r_factor)))

        if use_scalar:
            self.scalar = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("scalar", torch.tensor(1.0), persistent=False)
            
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        if use_scalar:
            torch.nn.init.kaiming_uniform_(self.lora_up.weight, a=math.sqrt(5))
        else:
            torch.nn.init.constant_(self.lora_up.weight, 0)

    @classmethod
    def make_module_from_state_dict(
        cls, lora_name, orig_module, up, down, alpha, bora_scale_r, bora_scale_c
    ):
        module = cls(
            lora_name,
            orig_module,
            1,
            down.size(0),
            float(alpha),
            weight_decompose=bora_scale_r is not None,
        )
        module.lora_up.weight.data.copy_(up)
        module.lora_down.weight.data.copy_(down)
        if bora_scale_r is not None:
            module.bora_scale_r.copy_(bora_scale_r)
        if bora_scale_c is not None:
            module.bora_scale_c.copy_(bora_scale_c)
        return module

    def load_weight_hook(self, module: nn.Module, incompatible_keys):
        missing_keys = incompatible_keys.missing_keys
        is_missing = False
        for key in list(missing_keys):
            if "scalar" in key:
                missing_keys.remove(key)
                is_missing = True
        
        if is_missing:
            if isinstance(self.scalar, nn.Parameter):
                self.scalar.data.copy_(torch.ones_like(self.scalar))
            elif getattr(self, "scalar", None) is not None:
                self.scalar.copy_(torch.ones_like(self.scalar))
            else:
                self.register_buffer(
                    "scalar", torch.ones_like(self.scalar), persistent=False
                )

    def make_weight(self, device=None):
        wa = self.lora_up.weight.to(device)
        wb = self.lora_down.weight.to(device)
        weight = wa.view(wa.size(0), -1) @ wb.view(wb.size(0), -1)
        weight = weight.view(self.shape)
        
        if self.training and self.rank_dropout:
            drop = (torch.rand(weight.size(0), device=device) > self.rank_dropout).to(
                weight.dtype
            )
            drop = drop.view(-1, *[1] * len(weight.shape[1:]))
            if self.rank_dropout_scale:
                drop /= drop.mean()
            weight *= drop

        return weight * self.scalar.to(device)

    def get_diff_weight(self, multiplier=1, shape=None, device=None):
        scale = self.scale * multiplier
        diff = self.make_weight(device=device) * scale
        if shape is not None:
            diff = diff.view(shape)
        if device is not None:
            diff = diff.to(device)
        return diff, None

    def get_merged_weight(self, multiplier=1, shape=None, device=None):
        diff = self.get_diff_weight(multiplier=multiplier, shape=shape, device=device)[0]
        weight = self.org_weight
        if self.wd:
            merged = self.apply_weight_decompose(weight + diff)
        else:
            merged = weight + diff
        return merged, None

    def apply_weight_decompose(self, weight):
        """
        严格实现公式 3 双维度归一化与调整:
        W = m^c * ( (V^r * m^r) / ||V^r * m^r||_c )
        """
        weight = weight.to(self.bora_scale_r.dtype)
        eps = torch.finfo(weight.dtype).eps

        # 1. 行归一化 (Row Normalization -> V^r)
        row_norm = weight.norm(dim=1, keepdim=True) + eps
        v_r = weight / row_norm

        # 2. 乘以行幅值 m^r (u = m^r * V^r)
        scale_r = self.bora_scale_r.to(weight.device)
        u = v_r * scale_r

        # 3. 列归一化 (Column Normalization -> H^c)
        col_norm = u.norm(dim=0, keepdim=True) + eps
        h_c = u / col_norm

        # 4. 乘以列幅值 m^c (W = m^c * H^c)
        scale_c = self.bora_scale_c.to(weight.device)
        return h_c * scale_c

    def custom_state_dict(self):
        destination = {}
        if self.wd:
            destination["bora_scale_r"] = self.bora_scale_r
            destination["bora_scale_c"] = self.bora_scale_c
        destination["alpha"] = self.alpha
        destination["lora_up.weight"] = self.lora_up.weight * self.scalar
        destination["lora_down.weight"] = self.lora_down.weight
        return destination

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        orig_norm = self.make_weight(device).norm() * self.scale
        norm = torch.clamp(orig_norm, max_norm / 2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = desired.cpu() / norm.cpu()

        scaled = norm != desired
        if scaled:
            self.scalar *= ratio

        return scaled, orig_norm * ratio

    def bypass_forward_diff(self, x, scale=1):
        mid = self.lora_down(x)

        if self.rank_dropout and self.training:
            drop = (
                torch.rand(self.lora_dim, device=mid.device) > self.rank_dropout
            ).to(mid.dtype)
            if self.rank_dropout_scale:
                drop /= drop.mean()
            if (dims := len(x.shape)) == 4:
                drop = drop.view(1, -1, 1, 1)
            else:
                drop = drop.view(*[1] * (dims - 1), -1)
            mid = mid * drop

        return self.dropout(self.lora_up(mid) * self.scalar * self.scale * scale)

    def bypass_forward(self, x, scale=1):
        return self.org_forward(x) + self.bypass_forward_diff(x, scale=scale)

    def forward(self, x, *args, **kwargs):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x, *args, **kwargs)

        if self.bypass_mode:
            return self.bypass_forward(x, scale=self.multiplier)

        base = self.org_forward(x, *args, **kwargs)
        scale = self.scale
        device = x.device

        base_weight = self._current_weight().to(device)
        diff_weight = self.make_weight(device) * scale
        transform = (
            (lambda weight: self.apply_weight_decompose(weight))
            if self.wd
            else None
        )
        delta_weight = compute_merged_delta(
            base_weight,
            diff_weight,
            self.multiplier,
            transform,
        )
        delta = self.op(x, delta_weight, None, **self.kw_dict)
        return base + delta
