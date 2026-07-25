import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LycorisBaseModule

try:
    from ..functional import factorization
except ImportError:
    def factorization(dimension: int, factor: int = -1) -> tuple[int, int]:
        if factor > 0 and dimension % factor == 0:
            return factor, dimension // factor
        for i in range(int(math.sqrt(dimension)), 0, -1):
            if dimension % i == 0:
                return i, dimension // i
        return 1, dimension


def kron_matmul(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """
    高效计算 x @ (w1 ⊗ w2)^T，完全避免在内存中重建高维 Kronecker 大矩阵。
    """
    a, c = w1.shape
    b, d = w2.shape
    orig_shape = x.shape
    x_reshaped = x.reshape(*orig_shape[:-1], c, d)
    t1 = F.linear(x_reshaped, w2)
    t1 = t1.transpose(-1, -2)
    t2 = F.linear(t1, w1)
    t2 = t2.transpose(-1, -2)
    return t2.reshape(*orig_shape[:-1], a * b)


class GloKrSoraModule(LycorisBaseModule):
    name = "glora_lokr"
    support_module = {"linear"}

    weight_list = [
        "a_w1", "a_w1_a", "a_w1_b", "a_w2", "a_w2_a", "a_w2_b",
        "b_w1", "b_w1_a", "b_w1_b", "b_w2", "b_w2_a", "b_w2_b",
        "alpha", "sora_cp", "sora_dp"
    ]

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
        use_tucker=False,       # 恢复被删除的参数，确保严格的 Positional Align 顺序
        use_scalar=False,       # use_scalar 恢复至第 10 位，消除 multiple values 冲突
        decompose_both=False,
        factor: int = -1,
        rank_dropout_scale=False,
        bypass_mode=None,
        rs_lora=False,
        use_sora=False,
        sora_r=4,
        sora_epsilon=1e-5,
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
            raise ValueError(f"{self.module_type} is not supported in GloKrSoraModule.")

        in_dim = org_module.in_features
        out_dim = org_module.out_features
        self.shape = (out_dim, in_dim)
        self.sora_in_dim = in_dim

        self.lora_dim = lora_dim
        self.rs_lora = rs_lora
        self.use_sora = use_sora
        self.sora_r = int(sora_r)
        self.sora_epsilon = float(sora_epsilon)

        # 维度因子分解（network_args 传入的 factor 是字符串，必须先转 int）
        factor = int(factor)
        in_m, in_n = factorization(in_dim, factor)
        out_l, out_k = factorization(out_dim, factor)

        # ----------------- B 增量通路参数化 -----------------
        self.use_b_w1 = True
        self.use_b_w2 = True

        if decompose_both and lora_dim < max(out_l, in_m) / 2:
            self.b_w1_a = nn.Parameter(torch.empty(out_l, lora_dim))
            self.b_w1_b = nn.Parameter(torch.empty(lora_dim, in_m))
            self.use_b_w1 = False
        else:
            self.b_w1 = nn.Parameter(torch.empty(out_l, in_m))

        if lora_dim < max(out_k, in_n) / 2:
            self.b_w2_a = nn.Parameter(torch.empty(out_k, lora_dim))
            self.b_w2_b = nn.Parameter(torch.empty(lora_dim, in_n))
            self.use_b_w2 = False
        else:
            self.b_w2 = nn.Parameter(torch.empty(out_k, in_n))

        # ----------------- A 重参数通路参数化 -----------------
        self.use_a_w1 = True
        self.use_a_w2 = True

        if decompose_both and lora_dim < in_m / 2:
            self.a_w1_a = nn.Parameter(torch.empty(in_m, lora_dim))
            self.a_w1_b = nn.Parameter(torch.empty(lora_dim, in_m))
            self.use_a_w1 = False
        else:
            self.a_w1 = nn.Parameter(torch.empty(in_m, in_m))

        if lora_dim < in_n / 2:
            self.a_w2_a = nn.Parameter(torch.empty(in_n, lora_dim))
            self.a_w2_b = nn.Parameter(torch.empty(lora_dim, in_n))
            self.use_a_w2 = False
        else:
            self.a_w2 = nn.Parameter(torch.empty(in_n, in_n))

        # ----------------- SORA 旋转参数 -----------------
        if self.use_sora:
            self.sora_cp = nn.Parameter(torch.empty(self.sora_in_dim, self.sora_r))
            self.sora_dp = nn.Parameter(torch.empty(self.sora_in_dim, self.sora_r))
            nn.init.kaiming_uniform_(self.sora_cp, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.sora_dp, a=math.sqrt(5))

        # ----------------- 缩放因子设置与全矩阵安全锁 -----------------
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().float().item()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha

        # 全矩阵安全锁：当 dim 极大触发全矩阵 Kronecker 微调时，强制重置 alpha
        # 避免缩放比 alpha / dim 趋于 0 导致的网络梯度冻结
        if (self.use_b_w1 and self.use_b_w2) and (self.use_a_w1 and self.use_a_w2):
            alpha = lora_dim

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
        self.reset_parameters(use_scalar)

    def reset_parameters(self, use_scalar: bool):
        # 初始化 B
        if self.use_b_w1:
            nn.init.kaiming_uniform_(self.b_w1, a=math.sqrt(5))
        else:
            nn.init.kaiming_uniform_(self.b_w1_a, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.b_w1_b, a=math.sqrt(5))

        if self.use_b_w2:
            if use_scalar:
                nn.init.kaiming_uniform_(self.b_w2, a=math.sqrt(5))
            else:
                nn.init.constant_(self.b_w2, 0.0)
        else:
            nn.init.kaiming_uniform_(self.b_w2_a, a=math.sqrt(5))
            if use_scalar:
                nn.init.kaiming_uniform_(self.b_w2_b, a=math.sqrt(5))
            else:
                nn.init.constant_(self.b_w2_b, 0.0)

        # 初始化 A
        if self.use_a_w1:
            nn.init.kaiming_uniform_(self.a_w1, a=math.sqrt(5))
        else:
            nn.init.kaiming_uniform_(self.a_w1_a, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.a_w1_b, a=math.sqrt(5))

        if self.use_a_w2:
            if use_scalar:
                nn.init.kaiming_uniform_(self.a_w2, a=math.sqrt(5))
            else:
                nn.init.constant_(self.a_w2, 0.0)
        else:
            nn.init.kaiming_uniform_(self.a_w2_a, a=math.sqrt(5))
            if use_scalar:
                nn.init.kaiming_uniform_(self.a_w2_b, a=math.sqrt(5))
            else:
                nn.init.constant_(self.a_w2_b, 0.0)

    def apply_sora(self, weight_2d):
        device = weight_2d.device
        dtype = weight_2d.dtype

        cp = self.sora_cp.to(device=device, dtype=dtype)
        dp = self.sora_dp.to(device=device, dtype=dtype)

        # 求解旋转步长因子
        cp_norm = torch.norm(cp, p="fro")
        dp_norm = torch.norm(dp, p="fro")
        s_P = self.sora_epsilon / (2.0 * cp_norm * dp_norm + self.sora_epsilon)

        w_dp = torch.matmul(weight_2d, dp)
        w_cp = torch.matmul(weight_2d, cp)

        term1 = torch.matmul(w_dp, cp.t())
        term2 = torch.matmul(w_cp, dp.t())

        return weight_2d + s_P * (term1 - term2)

    def get_weight(self, shape=None):
        # 1. 组装 A (Kronecker 表示)
        a1 = self.a_w1 if self.use_a_w1 else self.a_w1_a @ self.a_w1_b
        a2 = self.a_w2 if self.use_a_w2 else self.a_w2_a @ self.a_w2_b
        A = torch.kron(a1, a2) * self.scale

        # 2. 组装 B (Kronecker 表示)
        b1 = self.b_w1 if self.use_b_w1 else self.b_w1_a @ self.b_w1_b
        b2 = self.b_w2 if self.use_b_w2 else self.b_w2_a @ self.b_w2_b
        B = torch.kron(b1, b2) * self.scale

        # 3. 按照 GLoRA 的公式融合：W_diff = W_base @ A + B
        base_weight = self.org_weight.to(device=A.device, dtype=A.dtype)
        weight = base_weight @ A + B

        # 4. 实施 SORA 正交旋转映射
        if self.use_sora:
            weight_2d = weight.view(weight.size(0), -1)
            weight_rotated = self.apply_sora(weight_2d)
            weight = weight_rotated.view_as(weight)

        if shape is not None:
            weight = weight.view(shape)

        # 5. Rank Dropout
        if self.training and self.rank_dropout:
            dtype = weight.dtype
            drop = (torch.rand(weight.size(0)) > self.rank_dropout).to(dtype)
            drop = drop.view(-1, *[1] * len(weight.shape[1:]))
            if self.rank_dropout_scale:
                drop /= drop.mean()
            weight *= drop

        return weight

    def get_diff_weight(self, multiplier=1.0, shape=None, device=None):
        # 修复点 2：避免重复应用 self.scale。调用 get_weight() 时内部已乘以 self.scale
        diff = self.get_weight(shape) * multiplier * self.scalar
        if device is not None:
            diff = diff.to(device)
        return diff, None

    def get_merged_weight(self, multiplier=1, shape=None, device=None):
        diff = self.get_diff_weight(multiplier=multiplier, shape=shape, device=device)[0]
        merged = self.org_weight + diff
        return merged, None

    def bypass_forward_diff(self, x, scale=1.0):
        # 1. 隐式计算 A(x) = x @ A^T
        a1 = self.a_w1 if self.use_a_w1 else self.a_w1_a @ self.a_w1_b
        a2 = self.a_w2 if self.use_a_w2 else self.a_w2_a @ self.a_w2_b
        ax = kron_matmul(x, a1, a2) * self.scale

        # 2. 隐式计算 B(x) = x @ B^T
        b1 = self.b_w1 if self.use_b_w1 else self.b_w1_a @ self.b_w1_b
        b2 = self.b_w2 if self.use_b_w2 else self.b_w2_a @ self.b_w2_b
        bx = kron_matmul(x, b1, b2) * self.scale

        # 3. 将 A(x) 映射过原线性层的权重矩阵，加上加性分量 B(x)
        base_weight = self.org_weight.to(device=ax.device, dtype=ax.dtype)
        diff_out = F.linear(ax, base_weight) + bx

        return self.drop(diff_out * scale * self.scalar)

    def bypass_forward(self, x, scale=1.0):
        return self.org_forward(x) + self.bypass_forward_diff(x, scale=scale)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x, *args, **kwargs)

        if self.bypass_mode and not self.use_sora:
            return self.bypass_forward(x, self.multiplier)

        base = self.org_forward(x, *args, **kwargs)
        base_weight = self._current_weight().to(x.device)
        diff_weight = self.get_weight(self.shape).to(base_weight.dtype) * self.scalar

        if self.multiplier == 1:
            new_weight = base_weight + diff_weight
        else:
            new_weight = base_weight + diff_weight * self.multiplier

        delta_weight = new_weight - base_weight
        delta = self.op(x, delta_weight, None, **self.kw_dict)
        return base + delta

    def custom_state_dict(self):
        destination = {}
        destination["alpha"] = self.alpha

        if self.use_sora:
            destination["sora_cp"] = self.sora_cp
            destination["sora_dp"] = self.sora_dp

        # B state dict (烘焙 scalar)
        if self.use_b_w1:
            destination["b_w1"] = self.b_w1 * self.scalar
        else:
            destination["b_w1_a"] = self.b_w1_a * self.scalar
            destination["b_w1_b"] = self.b_w1_b

        if self.use_b_w2:
            destination["b_w2"] = self.b_w2
        else:
            destination["b_w2_a"] = self.b_w2_a
            destination["b_w2_b"] = self.b_w2_b

        # A state dict (修复点 3：A 通路烘焙 scalar，确保模型加载后与保存前数学等价)
        if self.use_a_w1:
            destination["a_w1"] = self.a_w1 * self.scalar
        else:
            destination["a_w1_a"] = self.a_w1_a * self.scalar
            destination["a_w1_b"] = self.a_w1_b

        if self.use_a_w2:
            destination["a_w2"] = self.a_w2
        else:
            destination["a_w2_a"] = self.a_w2_a
            destination["a_w2_b"] = self.a_w2_b  # 修复点 1：修正 typo 拼写错误 (此前误写为 self.b_w2_b)

        return destination

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
            self.register_buffer("scalar", torch.ones_like(self.scalar), persistent=False)

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        orig_norm = self.get_weight(self.shape).norm()
        norm = torch.clamp(orig_norm, max_norm / 2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = desired.cpu() / norm.cpu()

        scaled = norm != desired
        if scaled:
            # 准确计算乘性阶数 (Degree)
            deg_a = (1 if self.use_a_w1 else 2) + (1 if self.use_a_w2 else 2)
            deg_b = (1 if self.use_b_w1 else 2) + (1 if self.use_b_w2 else 2)
            deg_sora = 2 if self.use_sora else 0
            modules = deg_a + deg_b + deg_sora

            factor = ratio ** (1 / modules)

            if self.use_a_w1:
                self.a_w1 *= factor
            else:
                self.a_w1_a *= factor
                self.a_w1_b *= factor

            if self.use_a_w2:
                self.a_w2 *= factor
            else:
                self.a_w2_a *= factor
                self.a_w2_b *= factor

            if self.use_b_w1:
                self.b_w1 *= factor
            else:
                self.b_w1_a *= factor
                self.b_w1_b *= factor

            if self.use_b_w2:
                self.b_w2 *= factor
            else:
                self.b_w2_a *= factor
                self.b_w2_b *= factor

            if self.use_sora:
                self.sora_cp *= factor
                self.sora_dp *= factor

        return scaled, orig_norm * ratio

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x, *args, **kwargs)

        if self.bypass_mode and not self.use_sora:
            return self.bypass_forward(x, self.multiplier)

        base = self.org_forward(x, *args, **kwargs)
        base_weight = self._current_weight().to(x.device)
        diff_weight = self.get_weight(self.shape).to(base_weight.dtype) * self.scalar

        delta_weight = diff_weight if self.multiplier == 1 else diff_weight * self.multiplier

        delta = self.op(x, delta_weight, None, **self.kw_dict)
        return base + delta