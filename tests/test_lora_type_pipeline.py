# -*- coding: utf-8 -*-
"""End-to-end pipeline tests: UI lora_type config -> adapter -> real network build -> fwd/bwd.

Covers every lora_type offered by the Anima training page (mikazuki/schema/sd3-lora.ts).
LyCORIS algos build against the patched local lycoris package in the venv; the
networks.* algos build against vendor/sd-scripts. A tiny fake DiT whose block class
is named ``Block`` matches both target-module lists.
"""
import sys
import math
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "sd-scripts"))

import torch
import torch.nn as nn

from mikazuki.anima_backend.adapter import adapt_anima_config
from mikazuki.training_validation import TrainingConfigurationError

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
    set_timestep = getattr(net, "set_current_timestep", None)
    if callable(set_timestep):
        set_timestep(torch.tensor([500.0, 500.0]))
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

    def test_rslora_uses_rank_stabilized_scaling(self):
        module = self._assert_pipeline(
            {"lora_type": "rslora"},
            "LoConModule",
        )

        self.assertTrue(module.rs_lora)
        self.assertAlmostEqual(module.scale, 8 / math.sqrt(8))

    def test_dora_uses_weight_decomposed_lora(self):
        module = self._assert_pipeline(
            {"lora_type": "dora"},
            "LoConModule",
        )

        self.assertTrue(module.wd)
        self.assertTrue(hasattr(module, "dora_scale"))

    def test_lora_plus_creates_distinct_up_matrix_lr_group(self):
        net, dit, loras = _build(
            {
                "lora_type": "lora_plus",
                "loraplus_lr_ratio": 16,
            }
        )
        groups, _ = net.prepare_optimizer_params_with_multiple_te_lrs(
            None,
            1e-4,
            1e-4,
        )
        group_lrs = sorted(group["lr"] for group in groups)

        self.assertEqual(type(loras[0]).__name__, "LoRAModule")
        self.assertEqual(group_lrs, [1e-4, 1.6e-3])
        self.assertGreater(_smoke_backward(net, dit), 0)

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

    def test_delora(self):
        module = self._assert_pipeline(
            {
                "lora_type": "delora",
                "network_dropout": 0,
                "delora_lambda": 15,
            },
            "DeLoRAModule",
        )
        self.assertAlmostEqual(float(module.delora_lambda.item()), 15.0)

    def test_waveft(self):
        module = self._assert_pipeline(
            {
                "lora_type": "waveft",
                "network_dropout": 0,
                "waveft_n_frequency": 32,
                "waveft_scaling": 25,
                "waveft_random_loc_seed": 777,
                "waveft_use_idwt": True,
                "waveft_wavelet_family": "db1",
            },
            "WaveFTModule",
        )
        self.assertEqual(module.waveft_spectrum.numel(), 32)

    def test_deft(self):
        module = self._assert_pipeline(
            {
                "lora_type": "deft",
                "network_dropout": 0,
                "deft_decomposition_method": "qr",
                "deft_alpha": 0,
                "deft_init_scale": 1,
            },
            "DeftModule",
        )
        self.assertEqual(int(module.deft_decomposition_code.item()), 1)

    def test_moslora(self):
        module = self._assert_pipeline(
            {
                "lora_type": "moslora",
                "network_dropout": 0,
                "moslora_mixer_init": "identity",
            },
            "MoSLoRAModule",
        )
        torch.testing.assert_close(
            module.lora_mixer.weight,
            torch.eye(module.lora_dim),
        )

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

    def test_lorafa_freezes_down_matrix_and_trains_up_matrix(self):
        net, dit, loras = _build({"lora_type": "lora_fa"})
        module = loras[0]
        dit(torch.randn(2, DIM)).sum().backward()

        self.assertEqual(type(module).__name__, "LoRAFAModule")
        self.assertFalse(module.lora_down.weight.requires_grad)
        self.assertIsNone(module.lora_down.weight.grad)
        self.assertIsNotNone(module.lora_up.weight.grad)
        groups, _ = net.prepare_optimizer_params_with_multiple_te_lrs(
            None,
            1e-4,
            1e-4,
        )
        self.assertTrue(all("lorafa_pairs" in group for group in groups))

    def test_vera_uses_shared_projection_and_trainable_scaling_vectors(self):
        net, dit, loras = _build(
            {
                "lora_type": "vera",
                "vera_projection_seed": 42,
                "vera_d_initial": 0.1,
            }
        )
        sample = torch.randn(2, DIM)
        net.set_enabled(False)
        baseline = dit(sample).detach()
        net.set_enabled(True)
        adapted = dit(sample)
        adapted.sum().backward()

        self.assertEqual(type(loras[0]).__name__, "VeRAModule")
        torch.testing.assert_close(adapted, baseline)
        self.assertIsNotNone(loras[0].vera_lambda_b.grad)
        self.assertIsNotNone(loras[0].vera_lambda_d.grad)
        self.assertEqual(loras[0].projection_data_ptrs(), loras[1].projection_data_ptrs())
        state = net.state_dict()
        self.assertIn("vera_A", state)
        self.assertIn("vera_B", state)
        self.assertIn("vera_projection_seed", state)
        self.assertEqual(
            sum(key.endswith("vera_lambda_b") for key in state),
            len(loras),
        )

    def test_pissa_uses_svd_initialization_and_preserves_gradients(self):
        net, dit, loras = _build(
            {
                "lora_type": "lora",
                "pissa_init": True,
                "pissa_method": "svd",
                "pissa_export_mode": "LoRA无损兼容导出",
            }
        )
        dit(torch.randn(2, DIM)).sum().backward()

        self.assertEqual(type(loras[0]).__name__, "PiSSAModule")
        self.assertGreater(torch.count_nonzero(loras[0].lora_down.weight), 0)
        self.assertGreater(torch.count_nonzero(loras[0].lora_up.weight), 0)
        self.assertIsNotNone(loras[0].lora_down.weight.grad)
        self.assertIsNotNone(loras[0].lora_up.weight.grad)


if __name__ == "__main__":
    unittest.main()
