"""exclude_name 必须能剪掉类圈选(target_module)递归出的子层。

2026-07-31 定案回归:lycoris_anima_preset 用类名圈 Block,旧 wrapper 的
create_modules_ 内层递归不查 TARGET_EXCLUDE_NAME,导致 adaln_modulation
调制层被 LoKr 接管,训练全局色彩崩坏。此测试守住"排除名单对类圈选生效"。
"""
import unittest

import torch.nn as nn

from lycoris.wrapper import LycorisNetwork

_PRESET_ATTRS = (
    "ENABLE_CONV",
    "TARGET_REPLACE_MODULE",
    "TARGET_REPLACE_NAME",
    "MODULE_ALGO_MAP",
    "NAME_ALGO_MAP",
    "USE_FNMATCH",
    "TARGET_EXCLUDE_NAME",
)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(8, 8)
        self.mlp = nn.Linear(8, 8)
        self.adaln_modulation_self_attn = nn.Sequential(
            nn.SiLU(), nn.Linear(8, 24, bias=False)
        )


class TinyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block()])


class LycorisPresetExclusionTests(unittest.TestCase):
    def setUp(self):
        self._snapshot = {k: getattr(LycorisNetwork, k) for k in _PRESET_ATTRS}

    def tearDown(self):
        for k, v in self._snapshot.items():
            setattr(LycorisNetwork, k, v)

    def test_exclude_name_prunes_class_swept_children(self):
        LycorisNetwork.apply_preset(
            {
                "enable_conv": False,
                "target_module": ["Block"],
                "target_name": [],
                "exclude_name": ["*adaln_modulation*"],
                "use_fnmatch": True,
            }
        )
        net = LycorisNetwork(
            TinyDiT(),
            lora_dim=4,
            alpha=4,
            network_module="lokr",
            warn_on_unmatched=False,
        )
        names = sorted(lora.lora_name for lora in net.loras)
        adaln = [n for n in names if "adaln_modulation" in n]
        self.assertFalse(adaln, f"调制层必须被排除,实际建了: {adaln}")
        self.assertEqual(
            len(names), 4, f"两个 Block 各留 self_attn+mlp 共 4 个,实际: {names}"
        )
