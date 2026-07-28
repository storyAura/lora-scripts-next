import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LycorisBaseModule
from ..functional import factorization
from ..functional.lokr import make_kron


def _kaiming_fro(shape):
    # kaiming_uniform(a=sqrt(5)) 的期望 Frobenius 范数: sqrt(rows/3)
    return math.sqrt(shape[0] / 3)


def _norm_like_kaiming(t):
    return t * (_kaiming_fro(t.shape) / (t.norm() + 1e-12))


def _nkp_w1_factors(weight, out_l, out_k, in_m, in_n, terms):
    """Van Loan 最近 Kronecker 积分解: 取 W₀ 前 terms 个主成分的 w1 因子。"""
    try:
        rearranged = (
            weight.reshape(out_l, out_k, in_m, in_n)
            .permute(0, 2, 1, 3)
            .reshape(out_l * in_m, out_k * in_n)
        )
        q = min(terms + 2, *rearranged.shape)
        u, _, _ = torch.svd_lowrank(rearranged, q=q)
    except RuntimeError:
        return None
    return [u[:, i].reshape(out_l, in_m).contiguous() for i in range(min(terms, u.size(1)))]


def _split_lowrank(target, rank):
    u, s, v = torch.svd_lowrank(target, q=rank)
    s_sqrt = s.clamp_min(0).sqrt()
    return u * s_sqrt, (v * s_sqrt).transpose(0, 1)


class GLoKRModule(LycorisBaseModule):
    name = "glokr"
    # 专为 DiT 优化：仅支持 Linear 层
    support_module = {
        "linear",
    }
    # 参数属性名 == 存档键名 (kron_rank>1 时带 _{i} 后缀)
    weight_list = [
        "b_w1", "b_w1_a", "b_w1_b", "b_w2", "b_w2_a", "b_w2_b",
        "a_w1", "a_w1_a", "a_w1_b", "a_w2", "a_w2_a", "a_w2_b",
        "c_w1", "c_w1_a", "c_w1_b", "c_w2", "c_w2_a", "c_w2_b",
        "alpha", "dora_scale", "bora_scale_r", "bora_scale_c",
        "g_norm", "gate_b", "gate_a", "gate_c", "kron_mix", "bora_iters",
    ]
    weight_list_det = ["b_w1", "b_w1_a"]  # 仅作参考, 实际检测走 algo_check 前缀匹配

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
        use_tucker=False,       # 第 9 个位置参数必须为 use_tucker，兼容 kohya.py 签名绑定
        use_scalar=False,
        decompose_both=False,
        factor=-1,
        rank_dropout_scale=False,
        weight_decompose=False,
        use_bora=False,         # 是否使用 BoRA 双向权重分解
        wd_on_out=True,
        full_matrix=False,
        bypass_mode=None,
        rs_lora=False,
        unbalanced_factorization=False,
        kron_rank=1,
        use_g=True,             # 是否启用输入侧 G 融合路径 (W₀·A)
        use_g_out=False,        # 是否启用输出侧 G 路径 (C·W₀)
        train_gates=False,      # 可学习的 G/B 门控 + kron 项混合系数
        init_mode="kaiming",    # "kaiming" | "nkp" (Van Loan 主成分对齐初始化)
        g_norm_mode="frobenius",  # "frobenius" | "spectral" (G 路径尺度对齐所用范数)
        bora_iters=1,           # BoRA 行/列交替平衡迭代次数
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
            raise ValueError(f"{self.module_type} is not supported in DiT GLoKR algo.")

        # BoRA 本质上是更高级的权重解耦
        self.wd = weight_decompose or use_bora
        self.use_bora = use_bora and self.wd
        self.wd_on_out = wd_on_out
        self.use_g = use_g
        self.use_g_out = use_g_out
        self.train_gates = train_gates

        if self.wd and bypass_mode:
            raise ValueError("GLoKR does not support both `weight_decompose` / `use_bora` and `bypass_mode` active at the same time.")

        if isinstance(factor, (list, tuple)):
            in_factor, out_factor = int(factor[0]), int(factor[1])
        else:
            in_factor = out_factor = int(factor)

        self.lora_dim = lora_dim
        self.full_matrix = full_matrix
        self.rs_lora = rs_lora
        self.kron_rank = kron_rank

        in_dim = org_module.in_features
        out_dim = org_module.out_features
        self.shape = (out_dim, in_dim)

        in_m, in_n = factorization(in_dim, in_factor)
        out_l, out_k = factorization(out_dim, out_factor)
        if unbalanced_factorization:
            out_l, out_k = out_k, out_l
        self._factors = (out_l, out_k, in_m, in_n)

        # 各路径的 Kron 形状: ((w1_rows, w2_rows), (w1_cols, w2_cols))
        self.paths = ["b"] + (["a"] if use_g else []) + (["c"] if use_g_out else [])
        self.path_shapes = {
            "b": ((out_l, out_k), (in_m, in_n)),
            "a": ((in_m, in_n), (in_m, in_n)),
            "c": ((out_l, out_k), (out_l, out_k)),
        }

        # 各路径 w1/w2 是否用完整矩阵 (阈值逻辑与 LoKr 一致)
        self.use_w1 = {}
        self.use_w2 = {}
        for p in self.paths:
            (o1, o2), (i1, i2) = self.path_shapes[p]
            self.use_w1[p] = not (
                decompose_both and lora_dim < max(o1, i1) / 2 and not full_matrix
            )
            self.use_w2[p] = lora_dim >= max(o2, i2) / 2 or full_matrix
            for i in range(kron_rank):
                if self.use_w1[p]:
                    setattr(self, self._pname(p, "w1", i), nn.Parameter(torch.empty(o1, i1)))
                else:
                    setattr(self, self._pname(p, "w1_a", i), nn.Parameter(torch.empty(o1, lora_dim)))
                    setattr(self, self._pname(p, "w1_b", i), nn.Parameter(torch.empty(lora_dim, i1)))
                if self.use_w2[p]:
                    setattr(self, self._pname(p, "w2", i), nn.Parameter(torch.empty(o2, i2)))
                else:
                    setattr(self, self._pname(p, "w2_a", i), nn.Parameter(torch.empty(o2, lora_dim)))
                    setattr(self, self._pname(p, "w2_b", i), nn.Parameter(torch.empty(lora_dim, i2)))

        # 权重解耦 (DoRA / BoRA)
        if self.wd:
            if hasattr(org_module, "weight") and org_module.weight is not None and org_module.weight.device.type != "meta":
                org_weight = org_module.weight.cpu().clone().float()
            else:
                org_weight = torch.ones(self.shape, dtype=torch.float32)

            if self.use_bora:
                self.bora_scale_r = nn.Parameter(torch.norm(org_weight, dim=1, keepdim=True))
                self.bora_scale_c = nn.Parameter(torch.norm(org_weight, dim=0, keepdim=True))
                self.register_buffer("bora_iters", torch.tensor(int(bora_iters)))
            else:
                dim = 1 if self.wd_on_out else 0
                self.dora_scale = nn.Parameter(torch.norm(org_weight, dim=dim, keepdim=True))

        # 缩放配置
        if isinstance(alpha, torch.Tensor):
            alpha = alpha.detach().cpu().item()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        r_factor = math.sqrt(lora_dim) if self.rs_lora else lora_dim

        stabilizer = math.sqrt(self.kron_rank) if self.kron_rank > 1 else 1.0
        self.path_scale = {
            p: (1.0 if (self.use_w1[p] and self.use_w2[p]) else alpha / r_factor) / stabilizer
            for p in self.paths
        }
        self.scale = self.path_scale["b"]

        # G 路径的尺度对齐分母 ‖W₀‖ — 存成 buffer, 随存档持久化, 换底模不漂移
        if self.use_g or self.use_g_out:
            w = getattr(org_module, "weight", None)
            if w is None or w.device.type == "meta":
                norm_val = 1.0
            elif g_norm_mode == "spectral":
                try:
                    q = min(6, min(w.shape))
                    norm_val = float(torch.svd_lowrank(w.detach().float(), q=q)[1][0])
                except RuntimeError:
                    norm_val = float(w.detach().float().norm())
            else:
                norm_val = float(w.detach().float().norm())
            self.register_buffer("g_norm", torch.tensor(norm_val))

        # 可学习门控: 每路径一个标量 + kron 项混合系数
        if train_gates:
            for p in self.paths:
                setattr(self, f"gate_{p}", nn.Parameter(torch.tensor(1.0)))
            if kron_rank > 1:
                self.kron_mix = nn.Parameter(torch.ones(kron_rank))

        self.register_buffer("alpha", torch.tensor(alpha * (lora_dim / r_factor)))

        if use_scalar:
            self.scalar = nn.Parameter(torch.tensor(0.01))
        else:
            self.register_buffer("scalar", torch.tensor(1.0), persistent=False)

        # apply_max_norm 的范数缓存 (省掉每步的全量重建)
        self._cached_diff_norm = None
        self._max_norm_seen = False

        self.init_weights(use_scalar, init_mode)

    # ------------------------------------------------------------------
    # 参数访问辅助
    # ------------------------------------------------------------------
    def _pname(self, path, part, i):
        return f"{path}_{part}" + (f"_{i}" if self.kron_rank > 1 else "")

    def _p(self, path, part, i):
        return getattr(self, self._pname(path, part, i), None)

    def _w1(self, path, i, dtype=None):
        w = self._p(path, "w1", i)
        if w is not None:
            return w if dtype is None else w.to(dtype)
        a, b = self._p(path, "w1_a", i), self._p(path, "w1_b", i)
        if dtype is not None:
            a, b = a.to(dtype), b.to(dtype)
        return a @ b

    def _w2(self, path, i, dtype=None):
        w = self._p(path, "w2", i)
        if w is not None:
            return w if dtype is None else w.to(dtype)
        a, b = self._p(path, "w2_a", i), self._p(path, "w2_b", i)
        if dtype is not None:
            a, b = a.to(dtype), b.to(dtype)
        return a @ b

    def _eff_scale(self, path):
        scale = self.path_scale[path]
        if path in ("a", "c"):
            scale = scale / (self.g_norm + 1e-8)
        gate = getattr(self, f"gate_{path}", None)
        if gate is not None:
            scale = scale * gate
        return scale

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def init_weights(self, use_scalar=False, init_mode="kaiming"):
        out_l, out_k, in_m, in_n = self._factors
        nkp_w1 = None
        if init_mode == "nkp":
            w0 = getattr(self.org_module[0], "weight", None)
            if w0 is not None and w0.device.type != "meta":
                nkp_w1 = _nkp_w1_factors(
                    w0.detach().float(), out_l, out_k, in_m, in_n, self.kron_rank
                )

        for p in self.paths:
            for i in range(self.kron_rank):
                # w2 侧置零 → 初始增量为 0, 不扰动底模
                w2 = self._p(p, "w2", i)
                if w2 is not None:
                    if use_scalar:
                        torch.nn.init.kaiming_uniform_(w2, a=math.sqrt(5))
                    else:
                        torch.nn.init.constant_(w2, 0)
                else:
                    torch.nn.init.kaiming_uniform_(self._p(p, "w2_a", i), a=math.sqrt(5))
                    if use_scalar:
                        torch.nn.init.kaiming_uniform_(self._p(p, "w2_b", i), a=math.sqrt(5))
                    else:
                        torch.nn.init.constant_(self._p(p, "w2_b", i), 0)

                # w1 侧: kaiming / NKP 主成分 / 单位阵
                target = None
                if p == "b" and nkp_w1 is not None and i < len(nkp_w1):
                    target = _norm_like_kaiming(nkp_w1[i])
                w1 = self._p(p, "w1", i)
                if w1 is not None:
                    if target is not None:
                        w1.data.copy_(target)
                    elif p == "a" and init_mode == "nkp":
                        # 让 A 路的早期梯度从"组内重混"出发
                        w1.data.copy_(torch.eye(w1.size(0)) / math.sqrt(3.0))
                    else:
                        torch.nn.init.kaiming_uniform_(w1, a=math.sqrt(5))
                else:
                    w1a, w1b = self._p(p, "w1_a", i), self._p(p, "w1_b", i)
                    rank = w1a.size(1)
                    if target is not None and rank <= min(target.shape):
                        a_init, b_init = _split_lowrank(target, rank)
                        w1a.data.copy_(a_init)
                        w1b.data.copy_(b_init)
                    else:
                        torch.nn.init.kaiming_uniform_(w1a, a=math.sqrt(5))
                        torch.nn.init.kaiming_uniform_(w1b, a=math.sqrt(5))

    # ------------------------------------------------------------------
    # 权重重建 (合并路径)
    # ------------------------------------------------------------------
    def get_weight(self):
        out = {}
        for p in self.paths:
            total = None
            for i in range(self.kron_rank):
                w = make_kron(self._w1(p, i), self._w2(p, i), 1.0)
                if hasattr(self, "kron_mix"):
                    w = w * self.kron_mix[i]
                total = w if total is None else total + w
            out[p] = total

        if self.training and self.rank_dropout:
            for p, w in out.items():
                drop = (torch.rand(w.size(0), device=w.device) > self.rank_dropout).to(w.dtype)
                drop = drop.view(-1, *[1] * (w.dim() - 1))
                if self.rank_dropout_scale:
                    drop = drop / (drop.mean() + 1e-8)
                out[p] = w * drop
        return out

    def get_diff_weight(self, multiplier=1.0, shape=None, device=None):
        base_weight = self._current_weight()
        device = device or base_weight.device
        dtype = base_weight.dtype

        ws = self.get_weight()
        scalar = self.scalar.to(device=device, dtype=dtype)

        diff = ws["b"].to(device=device, dtype=dtype) * self._eff_scale("b")
        if self.use_g or self.use_g_out:
            orig = base_weight.to(device=device, dtype=dtype)
        if self.use_g:
            diff = diff + (orig @ ws["a"].to(device=device, dtype=dtype)) * self._eff_scale("a")
        if self.use_g_out:
            diff = diff + (ws["c"].to(device=device, dtype=dtype) @ orig) * self._eff_scale("c")

        diff = diff * scalar * multiplier
        return diff, None

    def get_merged_weight(self, multiplier=1, shape=None, device=None):
        diff = self.get_diff_weight(multiplier=1, shape=shape, device=device)[0]
        weight = self._current_weight().to(device=diff.device, dtype=diff.dtype)
        if self.wd:
            merged = self.apply_weight_decompose(weight + diff, multiplier)
        else:
            merged = weight + diff * multiplier
        return merged, None

    def apply_weight_decompose(self, weight, multiplier=1):
        orig_dtype = weight.dtype
        w = weight.float()
        eps = torch.finfo(torch.float32).eps

        if self.use_bora:
            r = self.bora_scale_r.to(w.device).float()
            c = self.bora_scale_c.to(w.device).float()
            # bora_iters>1: Sinkhorn 式行/列交替平衡, 缓解单次行→列的顺序偏置
            for _ in range(int(self.bora_iters)):
                rs = r / (w.norm(dim=1, keepdim=True) + eps)
                if multiplier != 1:
                    rs = multiplier * (rs - 1) + 1
                w = w * rs
                cs = c / (w.norm(dim=0, keepdim=True) + eps)
                if multiplier != 1:
                    cs = multiplier * (cs - 1) + 1
                w = w * cs
        else:
            dim = 1 if self.wd_on_out else 0
            scale = self.dora_scale.to(w.device).float() / (
                w.norm(dim=dim, keepdim=True) + eps
            )
            if multiplier != 1:
                scale = multiplier * (scale - 1) + 1
            w = w * scale
        return w.to(orig_dtype)

    # ------------------------------------------------------------------
    # bypass 路径 (分组 Kron, 不重建大矩阵)
    # ------------------------------------------------------------------
    def _bypass_diff_path(self, path, h, scale=1.0):
        out = 0.0
        dtype = h.dtype
        for i in range(self.kron_rank):
            ba = self._w2(path, i, dtype)
            c = self._w1(path, i, dtype)
            uq = c.size(1)

            h_in_group = h.reshape(*h.shape[:-1], uq, -1)
            hb = F.linear(h_in_group, ba)
            hc = F.linear(hb.transpose(-1, -2), c).transpose(-1, -2)
            h_out = hc.reshape(*hc.shape[:-2], -1)

            if hasattr(self, "kron_mix"):
                h_out = h_out * self.kron_mix[i]

            if self.training and self.rank_dropout:
                mask_shape = [1] * (h_out.dim() - 1) + [h_out.size(-1)]
                drop = (torch.rand(mask_shape, device=h_out.device) > self.rank_dropout).to(dtype)
                if self.rank_dropout_scale:
                    drop = drop / (drop.mean() + 1e-8)
                h_out = h_out * drop

            out = out + self.drop(h_out * scale * self.scalar)
        return out

    def bypass_forward_diff(self, x, scale=1.0):
        out = self._bypass_diff_path("b", x, self._eff_scale("b") * scale)
        if self.use_g or self.use_g_out:
            base_weight = self._current_weight().to(device=x.device, dtype=x.dtype)
        if self.use_g:
            ax = self._bypass_diff_path("a", x, self._eff_scale("a") * scale)
            out = out + F.linear(ax, base_weight)
        if self.use_g_out:
            wx = F.linear(x, base_weight)
            out = out + self._bypass_diff_path("c", wx, self._eff_scale("c") * scale)
        return out

    def bypass_forward(self, x, scale=1.0, *args, **kwargs):
        bx = self._bypass_diff_path("b", x, self._eff_scale("b") * scale)
        x_in = x
        if self.use_g:
            x_in = x + self._bypass_diff_path("a", x, self._eff_scale("a") * scale)
        out = self.org_forward(x_in, *args, **kwargs) + bx
        if self.use_g_out:
            wx = F.linear(x, self._current_weight().to(device=x.device, dtype=x.dtype))
            out = out + self._bypass_diff_path("c", wx, self._eff_scale("c") * scale)
        return out

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x, *args, **kwargs):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.org_forward(x, *args, **kwargs)
        if self.bypass_mode:
            return self.bypass_forward(x, self.multiplier, *args, **kwargs)

        base = self.org_forward(x, *args, **kwargs)
        base_weight = self._current_weight().to(x.device)
        diff_weight = self.get_diff_weight(multiplier=1.0, device=base_weight.device)[0]

        if self._max_norm_seen and self.training:
            with torch.no_grad():
                self._cached_diff_norm = diff_weight.detach().float().norm()

        if self.wd:
            # 减法在 fp32 中完成, 只把小差值降回低精度, 避免 bf16 吸收掉细小增量
            w32 = base_weight.float()
            new32 = self.apply_weight_decompose(
                w32 + diff_weight.float() * self.multiplier, self.multiplier
            )
            delta_weight = (new32 - w32).to(dtype=base_weight.dtype)
        else:
            delta_weight = diff_weight.to(dtype=base_weight.dtype)
            if self.multiplier != 1:
                delta_weight = delta_weight * self.multiplier

        return base + F.linear(x, delta_weight)

    # ------------------------------------------------------------------
    # 存档 / 读档
    # ------------------------------------------------------------------
    def custom_state_dict(self):
        destination = {"alpha": self.alpha}
        if self.wd:
            if self.use_bora:
                destination["bora_scale_r"] = self.bora_scale_r
                destination["bora_scale_c"] = self.bora_scale_c
                destination["bora_iters"] = self.bora_iters
            else:
                destination["dora_scale"] = self.dora_scale
        if hasattr(self, "g_norm"):
            destination["g_norm"] = self.g_norm
        if hasattr(self, "kron_mix"):
            destination["kron_mix"] = self.kron_mix
        for p in self.paths:
            gate = getattr(self, f"gate_{p}", None)
            if gate is not None:
                destination[f"gate_{p}"] = gate
            for i in range(self.kron_rank):
                for part in ("w1", "w1_a", "w1_b", "w2", "w2_a", "w2_b"):
                    t = self._p(p, part, i)
                    if t is None:
                        continue
                    if part in ("w1", "w1_a"):
                        t = t * self.scalar
                    destination[self._pname(p, part, i)] = t
        return destination

    def load_weight_hook(self, module: nn.Module, incompatible_keys):
        missing_keys = list(incompatible_keys.missing_keys)
        handled = ("scalar", "g_norm", "bora_iters")
        incompatible_keys.missing_keys[:] = [
            k for k in missing_keys if not any(h in k for h in handled)
        ]

        if isinstance(self.scalar, nn.Parameter):
            self.scalar.data.copy_(torch.ones_like(self.scalar))
        elif getattr(self, "scalar", None) is not None:
            self.scalar.copy_(torch.ones_like(self.scalar))
        else:
            self.register_buffer("scalar", torch.tensor(1.0), persistent=False)

        if self.wd:
            base_weight = self._current_weight().detach().float()
            if base_weight.device.type != "meta":
                if self.use_bora:
                    if any("bora_scale_r" in k or "bora_scale_c" in k for k in missing_keys):
                        self.bora_scale_r.data.copy_(torch.norm(base_weight, dim=1, keepdim=True))
                        self.bora_scale_c.data.copy_(torch.norm(base_weight, dim=0, keepdim=True))
                elif hasattr(self, "dora_scale"):
                    if any("dora_scale" in k for k in missing_keys):
                        dim = 1 if self.wd_on_out else 0
                        self.dora_scale.data.copy_(torch.norm(base_weight, dim=dim, keepdim=True))

        # 旧存档没有 g_norm: 退回"用当前底模的范数"
        if hasattr(self, "g_norm") and any("g_norm" in k for k in missing_keys):
            w = self._current_weight().detach().float()
            if w.device.type != "meta":
                self.g_norm.copy_(w.norm())

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        # ponytail: 复用 forward 缓存的范数 (滞后一步), 避免每步重建 kron + W₀@A
        if self._max_norm_seen and self._cached_diff_norm is not None:
            orig_norm = self._cached_diff_norm.cpu()
        else:
            self._max_norm_seen = True
            orig_norm = self.get_diff_weight(multiplier=1.0, device=device)[0].float().norm().cpu()

        if orig_norm < 1e-8:
            return False, orig_norm

        desired = torch.clamp(orig_norm, max=max_norm)
        ratio = (desired / orig_norm).item()
        scaled = bool(orig_norm > max_norm)
        if scaled:
            for p in self.paths:
                for i in range(self.kron_rank):
                    w1 = self._p(p, "w1", i)
                    (w1 if w1 is not None else self._p(p, "w1_a", i)).data.mul_(ratio)
            if self._cached_diff_norm is not None:
                self._cached_diff_norm = self._cached_diff_norm * ratio

        return scaled, orig_norm * ratio

    # ------------------------------------------------------------------
    # 从 state dict 检测 / 重建
    # ------------------------------------------------------------------
    @classmethod
    def algo_check(cls, state_dict, lora_name):
        # gsokr 与 glokr 共用 b_w1 命名, 用其独有的 sora 键区分
        if (
            f"{lora_name}.sora_cp" in state_dict
            or f"{lora_name}.sora_dp" in state_dict
        ):
            return False
        prefix = f"{lora_name}.b_w1"
        return any(k.startswith(prefix) for k in state_dict)

    @classmethod
    def extract_state_dict(cls, state_dict, lora_name):
        prefix = f"{lora_name}."
        return [{k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}]

    @classmethod
    @torch.no_grad()
    def make_module_from_state_dict(cls, lora_name, orig_module, weights=None, **extra):
        weights = {**(weights or {}), **extra}
        alpha = weights.get("alpha", None)
        use_bora = "bora_scale_r" in weights and "bora_scale_c" in weights
        weight_decompose = ("dora_scale" in weights) or use_bora

        indices = set()
        for key in weights:
            head, _, tail = key.rpartition("_")
            if head and tail.isdigit():
                indices.add(int(tail))
        kron_rank = max(indices) + 1 if indices else 1
        s = "_0" if kron_rank > 1 else ""

        get = weights.get
        use_g = any(k.startswith("a_w") for k in weights)
        use_g_out = any(k.startswith("c_w") for k in weights)
        train_gates = "gate_b" in weights

        lora_dim = None
        for name in ("b_w1_a", "b_w2_a", "a_w1_a", "a_w2_a", "c_w1_a", "c_w2_a"):
            t = get(f"{name}{s}")
            if t is not None:
                lora_dim = t.size(1)
                break
        full_matrix = lora_dim is None
        if full_matrix:
            lora_dim = 1

        b_w1 = get(f"b_w1{s}")
        if b_w1 is not None:
            out_factor, in_factor = b_w1.shape
        else:
            out_factor, in_factor = get(f"b_w1_a{s}").size(0), get(f"b_w1_b{s}").size(1)

        decompose_both = (
            b_w1 is None
            or (use_g and get(f"a_w1{s}") is None)
            or (use_g_out and get(f"c_w1{s}") is None)
        )

        module = cls(
            lora_name,
            orig_module,
            multiplier=1,
            lora_dim=lora_dim,
            alpha=float(alpha) if alpha is not None else 1.0,
            decompose_both=decompose_both,
            factor=(in_factor, out_factor),
            weight_decompose=weight_decompose,
            use_bora=use_bora,
            full_matrix=full_matrix,
            kron_rank=kron_rank,
            use_g=use_g,
            use_g_out=use_g_out,
            train_gates=train_gates,
            bora_iters=int(weights["bora_iters"]) if "bora_iters" in weights else 1,
        )
        module.load_state_dict(weights, strict=False)
        return module
