# -*- coding: utf-8 -*-
"""Timestep supply pipeline + LyCORIS adapter plumbing tests.

Covers:
- the network-level set_current_timestep broadcast (lycoris side)
- T-LoRA rank masking actually engages once a timestep is supplied (pipeline revival)
- gradient checkpointing recomputation must see the same t as the first forward
- trainer-side timestep lifecycle (keep t through backward, clear before sampling)
- UI field -> adapter -> network_args -> real module construction
- adapter forward map and import reverse map stay symmetric
"""
import ast
import sys
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "sd-scripts"))

import torch
import torch.nn as nn
import torch.utils.checkpoint as torch_ckpt

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


def _build_glokr(extra_args=None):
    from lycoris.kohya import create_network

    torch.manual_seed(0)
    te = nn.Module()
    dit = FakeDiT()
    kwargs = {
        "algo": "glokr", "factor": "-1", "kron_rank": "2", "train_gates": "True",
        "init_mode": "nkp",
    }
    kwargs.update(extra_args or {})
    net = create_network(1.0, 8, 8, None, te, dit, warn_on_unmatched=False, **kwargs)
    net.apply_to(te, dit, False, True)
    return net, dit


def _build_tlora():
    import importlib

    mod = importlib.import_module("networks.tlora_anima")
    torch.manual_seed(0)
    te = nn.Module()
    dit = FakeDiT()
    net = mod.create_network(
        1.0, 8, 8, None, [te], dit,
        tlora_min_rank="1", tlora_rank_schedule="cosine", tlora_orthogonal_init="True",
    )
    net.apply_to([te], dit, False, True)
    return net, dit


class TimestepBroadcastTests(unittest.TestCase):
    def test_lycoris_network_broadcasts_timestep(self):
        net, _ = _build_glokr()
        ts = torch.tensor([500.0])
        net.set_current_timestep(ts)
        self.assertTrue(all(m.current_timestep is ts for m in net.unet_loras))
        net.clear_current_timestep()
        self.assertTrue(all(m.current_timestep is None for m in net.unet_loras))


class TLoRAPipelineRevivalTests(unittest.TestCase):
    def test_rank_mask_engages_once_timestep_supplied(self):
        # Regression: nothing ever called set_current_timestep, so T-LoRA's dynamic
        # rank masking silently never ran. With the training hook in place the mask
        # must actually change the module output at high noise.
        net, dit = _build_tlora()
        # give the adapters non-zero output so masking is observable
        with torch.no_grad():
            for m in net.unet_loras:
                for p in m.parameters():
                    p.add_(torch.randn_like(p) * 0.05)

        x = torch.randn(2, DIM)
        net.set_current_timestep(None)
        with self.assertRaisesRegex(RuntimeError, "set_current_timestep"):
            dit(x)
        net.set_current_timestep(torch.tensor([0.0, 0.0]))
        out_low_noise = dit(x)
        net.set_current_timestep(torch.tensor([999.0, 999.0]))  # high noise -> min rank
        out_high_noise = dit(x)
        net.clear_current_timestep()

        self.assertFalse(
            torch.allclose(out_low_noise, out_high_noise, atol=1e-7),
            "high-noise rank mask must alter the forward output (pipeline is alive)",
        )


class LycorisMappingSymmetryTests(unittest.TestCase):
    def test_import_reverse_map_covers_every_ui_field(self):
        # The import table lagged behind the adapter's forward map, so imported
        # configs silently dropped every extension-algo field. Keep them in sync.
        from mikazuki.anima_backend.adapter import LYCORIS_NETWORK_ARG_MAP
        from mikazuki.utils.config_import import _LYCORIS_NETWORK_ARG_TO_UI

        missing = {
            arg_key: ui_field
            for ui_field, arg_key in LYCORIS_NETWORK_ARG_MAP.items()
            if _LYCORIS_NETWORK_ARG_TO_UI.get(arg_key) != ui_field
        }
        self.assertEqual(missing, {}, f"import map missing/mismatched entries: {missing}")


class AdapterForwardingTests(unittest.TestCase):
    def test_ui_fields_reach_glokr_module(self):
        adapted, _ = adapt_anima_config({
            "pretrained_model_name_or_path": "x.safetensors",
            "network_module": "lycoris.kohya",
            "lycoris_algo": "glokr",
            "kron_rank": 3,
            "g_norm_mode": "spectral",
            "lokr_factor": -1,
        })
        args = {item.split("=", 1)[0]: item.split("=", 1)[1] for item in adapted["network_args"]}
        self.assertEqual(args.get("kron_rank"), "3")
        self.assertEqual(args.get("g_norm_mode"), "spectral")

        net, _ = _build_glokr({"kron_rank": args["kron_rank"]})
        m0 = net.unet_loras[0]
        self.assertEqual(m0.kron_rank, 3)


class GradientCheckpointRecomputeTests(unittest.TestCase):
    """CheckpointError 回归：backward 里的重算前向必须看到与首次前向相同的 t。

    旧训练循环在 forward 一结束就清空 current_timestep；gradient checkpointing
    在 backward 中重演前向时 timestep-aware 适配器看到 t=None，重算前向与首次
    前向不一致（T-LoRA 直接拒绝 t=None）→ backward 失败。
    """

    def _one_step(self, clear_before_backward):
        net, dit = _build_tlora()
        x = torch.randn(2, DIM)
        net.set_current_timestep(torch.tensor([500.0, 500.0]))
        h = x
        for block in dit.blocks:
            h = torch_ckpt.checkpoint(block, h, use_reentrant=False)
        loss = h.sum()
        if clear_before_backward:
            net.clear_current_timestep()
        loss.backward()
        net.clear_current_timestep()

    def test_old_behavior_clearing_t_breaks_recomputation(self):
        # 守住"测试抓得到原 bug"：提前清空必须让 backward 的重算前向失败
        with self.assertRaises((torch_ckpt.CheckpointError, RuntimeError)):
            self._one_step(clear_before_backward=True)

    def test_keeping_t_alive_through_backward_passes(self):
        self._one_step(clear_before_backward=False)  # must not raise


class _FakeTimestepNetwork:
    def __init__(self):
        self.current_timestep = "leftover"
        self.cleared = 0

    def set_current_timestep(self, t):
        self.current_timestep = t

    def clear_current_timestep(self):
        self.cleared += 1
        self.current_timestep = None


class TrainerTimestepLifecycleTests(unittest.TestCase):
    """训练器侧：训练路径保留注入的 t 到 backward 之后，采样入口清残留。"""

    @staticmethod
    def _import_trainer_module():
        import anima_train_network  # vendor/sd-scripts is on sys.path

        return anima_train_network

    def test_clear_helper_prefers_clear_hook_then_setter(self):
        atn = self._import_trainer_module()
        clear = atn.AnimaNetworkTrainer._clear_network_timestep

        both = _FakeTimestepNetwork()
        clear(both)
        self.assertEqual(both.cleared, 1)
        self.assertIsNone(both.current_timestep)

        class OnlySetter:
            def __init__(self):
                self.current_timestep = "leftover"

            def set_current_timestep(self, t):
                self.current_timestep = t

        setter_only = OnlySetter()
        clear(setter_only)
        self.assertIsNone(setter_only.current_timestep)

        clear(None)  # must not raise
        clear(object())  # hookless network: must not raise

    def test_sample_images_clears_leftover_timestep(self):
        atn = self._import_trainer_module()
        trainer = atn.AnimaNetworkTrainer()
        fake = _FakeTimestepNetwork()
        trainer._timestep_network = fake

        with mock.patch.object(atn.anima_train_utils, "sample_images") as sampler, \
                mock.patch.object(atn.strategy_base.TextEncodingStrategy, "get_strategy", return_value=None), \
                mock.patch.object(atn.strategy_base.TokenizeStrategy, "get_strategy", return_value=None), \
                mock.patch.object(trainer, "get_models_for_text_encoding", return_value=None):
            trainer.sample_images(None, None, 0, 0, None, None, None, [], None)

        self.assertIsNone(fake.current_timestep, "preview must run with the t=None fallback")
        self.assertEqual(fake.cleared, 1)
        sampler.assert_called_once()

    def test_finally_cleanup_is_guarded_by_is_train(self):
        src = (PROJECT_ROOT / "vendor" / "sd-scripts" / "anima_train_network.py").read_text(encoding="utf-8")
        fn = next(
            node
            for node in ast.walk(ast.parse(src))
            if isinstance(node, ast.FunctionDef) and node.name == "get_noise_pred_and_target"
        )
        tries = [t for t in ast.walk(fn) if isinstance(t, ast.Try) and t.finalbody]
        self.assertTrue(tries, "get_noise_pred_and_target must keep its try/finally")
        for t in tries:
            dump = " ".join(ast.dump(stmt) for stmt in t.finalbody)
            self.assertIn(
                "'is_train'",
                dump,
                "the finally cleanup must stay gated on is_train, or gradient "
                "checkpointing recomputation diverges (CheckpointError)",
            )


if __name__ == "__main__":
    unittest.main()
