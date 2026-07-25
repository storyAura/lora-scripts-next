# -*- coding: utf-8 -*-
"""End-to-end pipeline tests: UI lora_type config -> adapter -> real network build -> fwd/bwd.

Covers every lora_type offered by the Anima training page (mikazuki/schema/sd3-lora.ts).
LyCORIS algos build against the patched local lycoris package in the venv; the
networks.* algos build against vendor/sd-scripts. A tiny fake DiT whose block class
is named ``Block`` matches both target-module lists.
"""
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "sd-scripts"))

import torch
import torch.nn as nn

from mikazuki.anima_backend.adapter import adapt_anima_config

DIM = 16


class Block(nn.Module):
    def __init__(self, dim=DIM):
        super().__init__()
        self.attn = nn.Linear(dim, dim)
        self.mlp = nn.Linear(dim, dim)

    def forward(self, x):
        return self.mlp(torch.relu(self.attn(x)))


class FakeDiT(nn.Module):
    def __init__(self, dim=DIM, n=2):
        super().__init__()
        self.blocks = nn.ModuleList(Block(dim) for _ in range(n))

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


BASE = {
    "pretrained_model_name_or_path": "x.safetensors",
    "learning_rate": "1e-4",
    "network_dim": 8,
    "network_alpha": 8,
}


def _args_to_kwargs(network_args):
    kwargs = {}
    for item in network_args or []:
        key, value = item.split("=", 1)
        kwargs[key] = value
    return kwargs


def _build(ui_fields):
    adapted, _ = adapt_anima_config({**BASE, **ui_fields})
    module_name = adapted["network_module"]
    kwargs = _args_to_kwargs(adapted.get("network_args"))
    te = nn.Module()
    dit = FakeDiT()
    if module_name == "lycoris.kohya":
        from lycoris.kohya import create_network

        net = create_network(
            1.0, adapted["network_dim"], adapted["network_alpha"], None, te, dit,
            warn_on_unmatched=False, **kwargs,
        )
        net.apply_to(te, dit, False, True)
        loras = list(net.unet_loras)
    else:
        import importlib

        mod = importlib.import_module(module_name)
        net = mod.create_network(
            1.0, adapted["network_dim"], adapted["network_alpha"], None, [te], dit, **kwargs
        )
        net.apply_to([te], dit, False, True)
        loras = list(net.unet_loras)
    return net, dit, loras


def _smoke_backward(net, dit):
    x = torch.randn(2, DIM)
    dit(x).sum().backward()
    return sum(1 for p in net.parameters() if p.grad is not None)


class LoraTypePipelineTests(unittest.TestCase):
    def _assert_pipeline(self, ui_fields, expect_cls):
        net, dit, loras = _build(ui_fields)
        self.assertEqual(len(loras), 4, f"expected 4 modules, got {len(loras)}")
        self.assertEqual(type(loras[0]).__name__, expect_cls)
        self.assertGreater(_smoke_backward(net, dit), 0, "no gradients reached network params")
        return loras[0]

    def test_lora(self):
        self._assert_pipeline(
            {"network_module": "networks.lora_anima", "network_dropout": 0}, "LoRAModule"
        )

    def test_tlora_dynamic_rank_fields_reach_module(self):
        m0 = self._assert_pipeline(
            {
                "network_module": "networks.tlora_anima", "network_dropout": 0,
                "tlora_min_rank": 4, "tlora_rank_schedule": "cosine",
                "tlora_orthogonal_init": True,
            },
            "TLoRAModule",
        )
        self.assertEqual(int(m0.tlora_min_rank), 4)

    def test_loha(self):
        self._assert_pipeline(
            {"network_module": "networks.loha", "network_dropout": 0}, "LoHaModule"
        )

    def test_lokr(self):
        self._assert_pipeline(
            {"network_module": "lycoris.kohya", "lycoris_algo": "lokr", "lokr_factor": -1},
            "LokrModule",
        )

    def test_glokr_recommended_defaults_reach_module(self):
        m0 = self._assert_pipeline(
            {
                "network_module": "lycoris.kohya", "lycoris_algo": "glokr",
                "kron_rank": 2, "use_bora": True, "bora_iters": 2, "train_gates": True,
                "init_mode": "nkp", "g_norm_mode": "frobenius", "lokr_factor": -1,
            },
            "GLoKRModule",
        )
        self.assertTrue(m0.wd, "use_bora should imply weight decomposition")
        self.assertEqual(int(m0.kron_rank), 2)
        self.assertTrue(hasattr(m0, "gate_b"), "train_gates should create gate params")
        self.assertTrue(hasattr(m0, "bora_scale_r"), "BoRA scales should exist")

    def test_bokr_positional_signature_and_bora(self):
        # Regression: BokrModule.__init__ must keep use_tucker at positional slot 9,
        # otherwise kohya's positional call collides with use_scalar kwargs.
        m0 = self._assert_pipeline(
            {
                "network_module": "lycoris.kohya", "lycoris_algo": "bokr",
                "dora_wd": True, "lokr_factor": -1,
            },
            "BokrModule",
        )
        self.assertTrue(m0.wd, "dora_wd should enable BoRA decomposition in BoKR")

    def test_bora(self):
        m0 = self._assert_pipeline(
            {"network_module": "lycoris.kohya", "lycoris_algo": "bora", "dora_wd": True},
            "BoRAModule",
        )
        self.assertTrue(m0.wd)
        self.assertTrue(hasattr(m0, "bora_scale_r"))

    def test_gsokr_string_factor_and_sora(self):
        # Regression: gsokr must int()/float() network_args strings (factor, sora_r, ...).
        m0 = self._assert_pipeline(
            {
                "network_module": "lycoris.kohya", "lycoris_algo": "gsokr",
                "use_sora": True, "sora_r": 4, "sora_epsilon": 0.00001, "lokr_factor": -1,
            },
            "GloKrSoraModule",
        )
        self.assertTrue(m0.use_sora)
        self.assertEqual(m0.sora_r, 4)

    def test_glora_boft(self):
        self._assert_pipeline(
            {
                "network_module": "lycoris.kohya", "lycoris_algo": "glora_boft",
                "boft_constraint": 0, "boft_rescaled": False,
            },
            "GLoRABOFTModule",
        )

    def test_vera_and_lora_fa_are_plain_lora_for_now(self):
        # Known limitation: networks/lora_anima.py has no VeRA / LoRA-FA variant logic,
        # so both currently train as a standard LoRA. This test documents the fact and
        # will fail (guiding an update) once real variants are implemented.
        for ui in (
            {"network_module": "networks.lora_anima", "network_dropout": 0},  # vera branch
            {"network_module": "networks.lora_anima", "network_dropout": 0},  # lora_fa branch
        ):
            _, _, loras = _build(ui)
            self.assertEqual(type(loras[0]).__name__, "LoRAModule")


if __name__ == "__main__":
    unittest.main()
