# 端到端: 建网 -> 训一步 -> 保存 -> 从权重重建 -> 输出一致; preset 解析
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

from lycoris.wrapper import create_lycoris, create_lycoris_from_weights, LycorisNetwork
from lycoris.utils.preset import read_preset


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 32)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def report(name, fn):
    try:
        fn()
        print(f"[PASS] {name}")
    except AssertionError as e:
        print(f"[FAIL] {name}: {e}")
    except Exception as e:
        print(f"[ERROR] {name}: {type(e).__name__}: {e}")


def t_e2e():
    torch.manual_seed(0)
    toy = Toy()
    LycorisNetwork.apply_preset({"target_module": ["Linear"]})
    net = create_lycoris(
        toy, 1.0, linear_dim=4, linear_alpha=1,
        algo="glokr", use_g=True, use_bora=True, bora_iters=2,
        kron_rank=2, train_gates=True, init_mode="nkp",
    )
    net.apply_to()
    assert len(net.loras) == 2, f"{len(net.loras)} modules"

    x = torch.randn(4, 32)
    opt = torch.optim.SGD(net.parameters(), lr=1e-2)
    loss = toy(x).square().mean()
    loss.backward()
    opt.step()  # 训一步, 让适配器非零

    y_trained = toy(x).detach()
    y_gap = (y_trained - Toy.forward(toy, x)).abs().max()  # 同一 hook 下应相同

    sd = {k: v.detach().clone() for k, v in net.state_dict().items()}

    # 全新模型 + 从权重重建
    torch.manual_seed(0)
    toy2 = Toy()
    net2, _ = create_lycoris_from_weights(1.0, None, toy2, weights_sd=sd)
    net2.apply_to()
    y_rebuilt = toy2(x).detach()
    assert torch.allclose(y_trained, y_rebuilt, atol=1e-5), \
        f"max diff {(y_trained - y_rebuilt).abs().max().item():.2e}"

    # merge_to 后输出仍一致 (适配器并入底模权重)
    net2.restore()
    net2.merge_to(1.0)
    y_merged = toy2(x).detach()
    assert torch.allclose(y_trained, y_merged, atol=1e-4), \
        f"merged max diff {(y_trained - y_merged).abs().max().item():.2e}"


def t_preset():
    cfg = read_preset(r"D:\桌面\lycoris\presets\flux_style_glokr.toml")
    assert cfg is not None, "preset failed to parse"
    assert "name_algo_map" in cfg and len(cfg["name_algo_map"]) == 5
    entry = cfg["name_algo_map"]["single_blocks.*.linear*"]
    assert entry["algo"] == "glokr" and entry["dim"] == 64 and entry["init_mode"] == "nkp"


report("e2e: create -> train -> save -> rebuild -> merge", t_e2e)
report("preset toml parses", t_preset)
