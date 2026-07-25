import torch
import torch.nn as nn

from .base import LycorisBaseModule
from ..logging import warning_once


class NormModule(LycorisBaseModule):
    name = "norm"
    support_module = {
        "layernorm",
        "groupnorm",
    }
    weight_list = ["w_norm", "b_norm"]
    weight_list_det = ["w_norm"]

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        rank_dropout=0.0,
        module_dropout=0.0,
        rank_dropout_scale=False,
        **kwargs,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__(
            lora_name=lora_name,
            org_module=org_module,
            multiplier=multiplier,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
            rank_dropout_scale=rank_dropout_scale,
            **kwargs,
        )
        if self.module_type == "unknown":
            if not hasattr(org_module, "weight") or not hasattr(org_module, "_norm"):
                warning_once(f"{type(org_module)} is not supported in Norm algo.")
                self.not_supported = True
                return
            else:
                self.dim = org_module.weight.numel()
                self.not_supported = False
        elif self.module_type not in self.support_module:
            warning_once(f"{self.module_type} is not supported in Norm algo.")
            self.not_supported = True
            return

        # elementwise_affine=False / affine=False norms carry weight=None: there is no
        # scale to offset, and make_weight() would crash on `None.to(...)` at the first
        # forward. Anima's DiT LayerNorms are all affine-free, so train_norm=True would
        # otherwise build hundreds of modules that blow up during training/sampling.
        if getattr(org_module, "weight", None) is None:
            warning_once(f"{self.module_type} without affine weight is not supported in Norm algo.")
            self.not_supported = True
            return

        self.w_norm = nn.Parameter(torch.zeros(self.dim))
        if getattr(org_module, "bias", None) is not None:
            self.b_norm = nn.Parameter(torch.zeros(self.dim))
        if hasattr(org_module, "_norm"):
            self.org_norm = org_module._norm
        else:
            self.org_norm = None

    @classmethod
    def make_module_from_state_dict(cls, lora_name, orig_module, w_norm, b_norm):
        module = cls(
            lora_name,
            orig_module,
            1,
        )
        module.w_norm.copy_(w_norm)
        if b_norm is not None:
            module.b_norm.copy_(b_norm)
        return module

    def make_weight(self, scale=1, device=None):
        org_weight = self.org_module[0].weight.to(device, dtype=self.w_norm.dtype)
        # `hasattr` is true even when a norm carries bias=None (bias=False layers),
        # so test the value and require the matching b_norm parameter to exist.
        org_bias_src = getattr(self.org_module[0], "bias", None)
        if org_bias_src is not None and hasattr(self, "b_norm"):
            org_bias = org_bias_src.to(device, dtype=self.b_norm.dtype)
        else:
            org_bias = None
        if self.rank_dropout and self.training:
            drop = (torch.rand(self.dim, device=device) < self.rank_dropout).to(
                self.w_norm.device
            )
            if self.rank_dropout_scale:
                drop /= drop.mean()
        else:
            drop = 1
        drop = (
            torch.rand(self.dim, device=device) < self.rank_dropout
            if self.rank_dropout and self.training
            else 1
        )
        weight = self.w_norm.to(device) * drop * scale
        if org_bias is not None:
            bias = self.b_norm.to(device) * drop * scale
        return org_weight + weight, org_bias + bias if org_bias is not None else None

    def get_diff_weight(self, multiplier=1, shape=None, device=None):
        if self.not_supported:
            return 0, 0
        w = self.w_norm * multiplier
        if device is not None:
            w = w.to(device)
        if shape is not None:
            w = w.view(shape)
        if self.b_norm is not None:
            b = self.b_norm * multiplier
            if device is not None:
                b = b.to(device)
            if shape is not None:
                b = b.view(shape)
        else:
            b = None
        return w, b

    def get_merged_weight(self, multiplier=1, shape=None, device=None):
        if self.not_supported:
            return None, None
        diff_w, diff_b = self.get_diff_weight(multiplier, shape, device)
        org_w = self.org_module[0].weight.to(device, dtype=self.w_norm.dtype)
        weight = org_w + diff_w
        if diff_b is not None:
            org_b = self.org_module[0].bias.to(device, dtype=self.b_norm.dtype)
            bias = org_b + diff_b
        else:
            bias = None
        return weight, bias

    def forward(self, x, *args, **kwargs):
        if self.not_supported or (
            self.module_dropout
            and self.training
            and torch.rand(1) < self.module_dropout
        ):
            return self.org_forward(x, *args, **kwargs)

        base = self.org_forward(x, *args, **kwargs)

        weight, bias = self.make_weight(self.multiplier, x.device)
        org_weight = self._current_weight().to(weight.device, dtype=weight.dtype)
        delta_w = weight - org_weight

        delta_b = None
        if bias is not None:
            bias = bias.to(x.device)
            org_bias = self._current_bias()
            if org_bias is not None:
                delta_b = bias - org_bias.to(bias.device)
            else:
                delta_b = bias

        if self.org_norm is not None:
            normed = self.org_norm(x)
            delta = normed * delta_w
            if delta_b is not None:
                delta = delta + delta_b
            return base + delta

        kw_dict = self.kw_dict | {"weight": delta_w, "bias": delta_b}
        delta = self.op(x, **kw_dict)
        return base + delta


if __name__ == "__main__":
    base = nn.LayerNorm(128).cuda()
    norm = NormModule("test", base, 1).cuda()
    print(norm)
    test_input = torch.randn(1, 128).cuda()
    test_output = norm(test_input)
    torch.sum(test_output).backward()
    print(test_output.shape)

    base = nn.GroupNorm(4, 128).cuda()
    norm = NormModule("test", base, 1).cuda()
    print(norm)
    test_input = torch.randn(1, 128, 3, 3).cuda()
    test_output = norm(test_input)
    torch.sum(test_output).backward()
    print(test_output.shape)
