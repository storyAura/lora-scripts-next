# GLoKR 优化后完整验证套件
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from lycoris.modules.glokr import GLoKRModule, _nkp_w1_factors
from lycoris.modules.lokr import LokrModule
from lycoris.modules.bokr import BokrModule
from lycoris.modules import get_module, make_module

torch.manual_seed(0)
IN, OUT = 32, 16  # factorization: 32->(4,8), 16->(4,4)


def fresh_base(dtype=torch.float32):
    torch.manual_seed(42)
    return nn.Linear(IN, OUT, bias=True).to(dtype)


def make_m(base=None, **kw):
    if base is None:
        base = fresh_base()
    torch.manual_seed(1)
    args = dict(multiplier=1.0, lora_dim=1, alpha=1, use_scalar=True, use_g=True)
    args.update(kw)
    return base, GLoKRModule("t", base, **args)


def report(name, fn):
    try:
        fn()
        print(f"[PASS] {name}")
    except AssertionError as e:
        print(f"[FAIL] {name}: {e}")
    except Exception as e:
        print(f"[ERROR] {name}: {type(e).__name__}: {e}")


def merged_ref(m, base, x):
    diff = m.get_diff_weight()[0].detach()
    return F.linear(x, m._current_weight() + diff, base.bias.detach())


# ---------- 1. forward == 合并数学 ----------
def t1():
    base, m = make_m()
    m.apply_to()
    x = torch.randn(5, IN)
    assert torch.allclose(base(x), merged_ref(m, base, x), atol=1e-5)


# ---------- 2. bypass == 合并数学 ----------
def t2():
    base, m = make_m(bypass_mode=True)
    m.apply_to()
    x = torch.randn(5, IN)
    assert torch.allclose(base(x), merged_ref(m, base, x), atol=1e-5)


# ---------- 3. kron_rank=2 + 门控/混合系数扰动后 bypass == 合并 ----------
def t3():
    base, m = make_m(kron_rank=2, train_gates=True, bypass_mode=True)
    with torch.no_grad():
        m.gate_b.mul_(1.3)
        m.gate_a.mul_(0.7)
        m.kron_mix.copy_(torch.tensor([1.5, 0.5]))
    m.apply_to()
    x = torch.randn(5, IN)
    assert torch.allclose(base(x), merged_ref(m, base, x), atol=1e-5)


# ---------- 4. use_g_out: forward 与 bypass 都与合并数学一致 ----------
def t4():
    for bypass in (None, True):
        base, m = make_m(use_g_out=True, bypass_mode=bypass)
        m.apply_to()
        x = torch.randn(5, IN)
        y, y_ref = base(x), merged_ref(m, base, x)
        assert torch.allclose(y, y_ref, atol=1e-5), f"bypass={bypass}: {(y - y_ref).abs().max().item():.2e}"


# ---------- 5. wd 扰动缩放后 multiplier=0 仍还原底模 (插值修复) ----------
def t5():
    for kw in ({"weight_decompose": True}, {"use_bora": True, "bora_iters": 3}):
        base, m = make_m(multiplier=0.0, **kw)
        with torch.no_grad():
            if m.use_bora:
                m.bora_scale_r.mul_(1.7)
                m.bora_scale_c.mul_(0.6)
            else:
                m.dora_scale.mul_(1.7)
        m.apply_to()
        x = torch.randn(5, IN)
        y_base = F.linear(x, m._current_weight(), base.bias.detach())
        assert torch.allclose(base(x), y_base, atol=1e-5), f"{kw}"


# ---------- 6. wrapper 加载路径 (kron_rank=1) ----------
def t6():
    base, m = make_m()
    d0 = m.get_diff_weight()[0].detach().clone()
    full_sd = {f"lycoris_t.{k}": v.detach().clone() for k, v in m.state_dict().items()}
    lyco_type, params = get_module(full_sd, "lycoris_t")
    assert lyco_type is GLoKRModule, f"detected {lyco_type}"
    mod = make_module(lyco_type, params, "lycoris_t", fresh_base())
    assert mod is not None
    assert torch.allclose(d0, mod.get_diff_weight()[0], atol=1e-5)


# ---------- 7. wrapper 加载路径 (kron_rank=2 + 门控 + bora) ----------
def t7():
    base, m = make_m(kron_rank=2, train_gates=True, use_bora=True, bora_iters=2)
    with torch.no_grad():
        m.kron_mix.copy_(torch.tensor([1.4, 0.6]))
        m.gate_a.mul_(0.5)
    d0 = m.get_diff_weight()[0].detach().clone()
    full_sd = {f"lycoris_t.{k}": v.detach().clone() for k, v in m.state_dict().items()}
    lyco_type, params = get_module(full_sd, "lycoris_t")
    assert lyco_type is GLoKRModule, f"kron_rank=2 detected as {lyco_type}"
    mod = make_module(lyco_type, params, "lycoris_t", fresh_base())
    assert mod is not None
    assert int(mod.bora_iters) == 2
    assert torch.allclose(d0, mod.get_diff_weight()[0], atol=1e-5)


# ---------- 8. 续训路径: 同名属性 load_state_dict ----------
def t8():
    base, m = make_m(kron_rank=2, train_gates=True)
    d0 = m.get_diff_weight()[0].detach().clone()
    _, m2 = make_m(fresh_base(), kron_rank=2, train_gates=True, use_scalar=False)
    with torch.no_grad():
        for p in m2.parameters():
            p.mul_(0.123)  # 打乱, 确认 load 真正覆盖
    m2.load_state_dict({k: v.detach().clone() for k, v in m.state_dict().items()}, strict=False)
    assert torch.allclose(d0, m2.get_diff_weight()[0], atol=1e-5)


# ---------- 9. merge_to: 非 wd 精确合并; wd 合并 == forward ----------
def t9():
    base, m = make_m()
    w0 = base.weight.detach().clone()
    diff = m.get_diff_weight()[0].detach().clone()
    m.merge_to(1.0)
    assert torch.allclose(base.weight.detach(), w0 + diff, atol=1e-5)

    base2, m2 = make_m(use_bora=True)
    m2.apply_to()
    x = torch.randn(5, IN)
    y = base2(x)
    merged = m2.get_merged_weight(1.0)[0]
    y_ref = F.linear(x, merged, base2.bias.detach())
    assert torch.allclose(y, y_ref, atol=1e-5), (y - y_ref).abs().max().item()


# ---------- 10. 梯度流到所有参数 (含门控/混合/BoRA/双 G 路) ----------
def t10():
    base, m = make_m(kron_rank=2, train_gates=True, use_bora=True, use_g_out=True)
    m.apply_to()
    base(torch.randn(5, IN)).sum().backward()
    no_grad = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not no_grad, f"no grad: {no_grad}"


# ---------- 11. bf16 吸收修复: wd 的 delta 精度显著优于旧算法 ----------
def t11():
    base = fresh_base(torch.bfloat16)
    torch.manual_seed(1)
    m = GLoKRModule("t", base, 1.0, lora_dim=1, alpha=1, use_scalar=True,
                    use_g=True, weight_decompose=True)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "w2" in n:
                p.mul_(0.1)  # 让 diff ≈ 底模的 1e-3 量级
    W = m._current_weight()
    diff = m.get_diff_weight()[0].detach()
    # fp32 真值
    w32 = W.float()
    new32 = m.apply_weight_decompose(w32 + diff.float(), 1)
    ref = new32 - w32
    # 新算法: fp32 相减后降精度; 旧算法: bf16 域内相减
    delta_new = ref.to(torch.bfloat16).float()
    delta_old = (m.apply_weight_decompose((W + diff).to(W.dtype), 1) - W).float()
    err_new = (delta_new - ref).norm()
    err_old = (delta_old - ref).norm()
    assert err_new < 0.3 * err_old, f"err_new={err_new:.2e} err_old={err_old:.2e}"


# ---------- 12. g_norm 持久化: 换底模不漂移 ----------
def t12():
    base, m = make_m()
    saved_norm = float(m.g_norm)
    sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
    base_b = fresh_base()
    with torch.no_grad():
        base_b.weight.mul_(3.0)
    m2 = GLoKRModule.make_module_from_state_dict("t", base_b, sd)
    assert abs(float(m2.g_norm) - saved_norm) < 1e-4, \
        f"saved={saved_norm:.4f} loaded={float(m2.g_norm):.4f}"
    # 旧存档 (无 g_norm) 回退到当前底模范数
    sd2 = {k: v for k, v in sd.items() if k != "g_norm"}
    m3 = GLoKRModule.make_module_from_state_dict("t", base_b, sd2)
    expect = float(base_b.weight.detach().float().norm())
    assert abs(float(m3.g_norm) - expect) < 1e-3


# ---------- 13. NKP 初始化: 初始为恒等, w1 对齐主成分方向 ----------
def t13():
    base, m = make_m(init_mode="nkp", kron_rank=2, use_scalar=False)
    m.apply_to()
    x = torch.randn(5, IN)
    y_base = F.linear(x, m._current_weight(), base.bias.detach())
    assert torch.allclose(base(x), y_base, atol=1e-6), "nkp init must start as identity"
    facs = _nkp_w1_factors(base.weight.detach().float(), 4, 4, 4, 8, 2)
    w1 = m._p("b", "w1", 0).detach().flatten()
    ref = facs[0].flatten()
    cos = (w1 @ ref).abs() / (w1.norm() * ref.norm())
    assert cos > 0.9, f"|cos|={cos:.3f}"


# ---------- 14. apply_max_norm: 缓存路径 + 实际裁剪 ----------
def t14():
    base, m = make_m()
    m.apply_to()
    m.train()
    scaled, n0 = m.apply_max_norm(1e9)  # 首次: 全量重建, 打开缓存开关
    assert not scaled and n0 > 0
    base(torch.randn(5, IN))  # forward 填充缓存
    assert m._cached_diff_norm is not None
    scaled, _ = m.apply_max_norm(float(n0) * 0.5)
    assert scaled
    fresh = m.get_diff_weight()[0].detach().norm()
    assert fresh <= float(n0) * 0.5 * 1.01, f"{fresh} vs {float(n0) * 0.5}"


# ---------- 15. Bokr/Lokr 检测消歧 ----------
def t15():
    z = torch.zeros(2, 2)
    sd_bokr = {"p.lokr_w1": z, "p.lokr_w2": z, "p.alpha": torch.tensor(1.0),
               "p.bora_r_scale": z, "p.bora_c_scale": z}
    assert get_module(sd_bokr, "p")[0] is BokrModule
    sd_lokr = {"p.lokr_w1": z, "p.lokr_w2": z, "p.alpha": torch.tensor(1.0)}
    assert get_module(sd_lokr, "p")[0] is LokrModule


# ---------- 16. gsokr 存档不会被 glokr 误认 ----------
def t16():
    z = torch.zeros(2, 2)
    sd = {"p.b_w1": z, "p.a_w1": z, "p.sora_cp": z}
    assert get_module(sd, "p")[0] is not GLoKRModule


# ---------- 17. 兼容"旧版 glokr"存档 (无新键) ----------
def t17():
    base, m = make_m(use_bora=True)
    d0 = m.get_diff_weight()[0].detach().clone()
    sd = {k: v.detach().clone() for k, v in m.state_dict().items()
          if k not in ("g_norm", "bora_iters")}
    m2 = GLoKRModule.make_module_from_state_dict("t", fresh_base(), sd)
    assert torch.allclose(d0, m2.get_diff_weight()[0], atol=1e-5)


for name, fn in [
    ("forward == merged math", t1),
    ("bypass == merged math", t2),
    ("kron_rank=2 + gates/mix, bypass == merged", t3),
    ("use_g_out forward/bypass == merged", t4),
    ("wd perturbed, multiplier=0 recovers base", t5),
    ("wrapper loading (kron_rank=1)", t6),
    ("wrapper loading (kron_rank=2 + gates + bora)", t7),
    ("resume via load_state_dict", t8),
    ("merge_to (plain + wd equivalence)", t9),
    ("gradients flow everywhere", t10),
    ("bf16 absorption fix (wd delta)", t11),
    ("g_norm persistence across bases", t12),
    ("nkp init identity + principal alignment", t13),
    ("apply_max_norm cache + clipping", t14),
    ("bokr/lokr detection disambiguation", t15),
    ("gsokr not claimed by glokr", t16),
    ("legacy glokr checkpoint compat", t17),
]:
    report(name, fn)
