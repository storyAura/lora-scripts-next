# -*- coding: utf-8 -*-
"""T-GLoKR (timestep-gated GLoKR) + timestep supply pipeline tests.

Covers:
- the network-level set_current_timestep broadcast (lycoris side)
- T-LoRA rank masking actually engages once a timestep is supplied (pipeline revival)
- T-GLoKR identity-at-init: zero-initialized gates reproduce plain GLoKR bit-for-bit
- gates respond to their parameters and receive gradients
- t=None falls back to g=1 (never crashes)
- state_dict round-trip carries time_gate keys; old archives load fine without them
- UI field -> adapter -> network_args -> real module construction
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


def _is_truthy_flag(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _perturb(net, scale=0.05):
    """Deterministically perturb adapter weights so ΔW != 0 (gates become observable).

    Noise is seeded per parameter NAME, so two structurally different networks
    (with/without time_gate params) still receive identical noise on shared params.
    """
    import zlib

    with torch.no_grad():
        for name, p in sorted(net.named_parameters()):
            if "time_gate" in name:
                continue
            gen = torch.Generator().manual_seed(zlib.crc32(name.encode()) & 0x7FFFFFFF)
            p.add_(torch.randn(p.shape, generator=gen) * scale)


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
        # give the adapters non-zero output so masking is observable
        with torch.no_grad():
            for m in net.unet_loras:
                for p in m.parameters():
                    p.add_(torch.randn_like(p) * 0.05)

        x = torch.randn(2, DIM)
        net.set_current_timestep(None)
        out_no_ts = dit(x)
        net.set_current_timestep(torch.tensor([999.0, 999.0]))  # high noise -> min rank
        out_high_noise = dit(x)
        net.clear_current_timestep()

        self.assertFalse(
            torch.allclose(out_no_ts, out_high_noise, atol=1e-7),
            "high-noise rank mask must alter the forward output (pipeline is alive)",
        )


class TGLoKRTests(unittest.TestCase):
    def test_identity_at_init_with_timestep_set(self):
        # Perturb weights so ΔW != 0 — otherwise 0 == 0 would pass trivially.
        x = torch.randn(2, DIM)
        net_plain, dit_plain = _build_glokr({"train_time_gates": "False"})
        _perturb(net_plain)
        out_plain = dit_plain(x)

        net_tg, dit_tg = _build_glokr({"train_time_gates": "True", "time_gate_dim": "4"})
        _perturb(net_tg)
        net_tg.set_current_timestep(torch.tensor([700.0, 300.0]))
        out_tg = dit_tg(x)
        net_tg.clear_current_timestep()

        self.assertFalse(torch.allclose(out_plain, torch.zeros_like(out_plain)), "sanity: ΔW must be non-zero")
        self.assertTrue(
            torch.allclose(out_plain, out_tg, atol=1e-6),
            "zero-initialized time gates must reproduce plain GLoKR (identity start)",
        )

    def test_gate_bias_changes_output_and_gets_gradients(self):
        net, dit = _build_glokr({"train_time_gates": "True"})
        _perturb(net)  # gates multiply ΔW; with ΔW = 0 they would be unobservable
        m0 = net.unet_loras[0]
        self.assertTrue(hasattr(m0, "time_gate_w"))

        x = torch.randn(2, DIM)
        net.set_current_timestep(torch.tensor([500.0]))
        out_base = dit(x).detach()

        with torch.no_grad():
            for m in net.unet_loras:
                m.time_gate_b.fill_(2.0)  # g = 2*sigmoid(2) != 1
        out_gated = dit(x)
        self.assertFalse(torch.allclose(out_base, out_gated, atol=1e-7), "gates must act on the paths")

        out_gated.sum().backward()
        self.assertIsNotNone(m0.time_gate_w.grad, "gate parameters must receive gradients")
        net.clear_current_timestep()

    def test_none_timestep_falls_back_to_identity(self):
        x = torch.randn(2, DIM)
        net_plain, dit_plain = _build_glokr({"train_time_gates": "False"})
        _perturb(net_plain)
        net_tg, dit_tg = _build_glokr({"train_time_gates": "True"})
        _perturb(net_tg)
        # no set_current_timestep at all -> g must default to 1
        self.assertTrue(torch.allclose(dit_plain(x), dit_tg(x), atol=1e-6))

    def test_state_dict_round_trip_and_legacy_load(self):
        net, _ = _build_glokr({"train_time_gates": "True"})
        m0 = net.unet_loras[0]
        sd = m0.custom_state_dict()
        self.assertIn("time_gate_w", sd)
        self.assertIn("time_gate_b", sd)

        # round-trip into a fresh module of the same shape
        net2, _ = _build_glokr({"train_time_gates": "True"})
        m2 = net2.unet_loras[0]
        missing, unexpected = m2.load_state_dict({k: v.detach().clone() for k, v in sd.items()}, strict=False)
        self.assertFalse(any("time_gate" in k for k in missing))
        self.assertFalse(unexpected)

        # legacy archive without time_gate keys must still load into a gated module
        legacy = {k: v for k, v in sd.items() if "time_gate" not in k}
        net3, _ = _build_glokr({"train_time_gates": "True"})
        m3 = net3.unet_loras[0]
        m3.load_state_dict(legacy, strict=False)  # must not raise


class TGLoKRLoraTypeBranchTests(unittest.TestCase):
    """tglokr is its own lora_type in the UI while running as algo=glokr."""

    def test_schema_exposes_tglokr_branch(self):
        schema = (PROJECT_ROOT / "mikazuki" / "schema" / "sd3-lora.ts").read_text(encoding="utf-8")
        self.assertIn('"tglokr"', schema.split("lora_type: Schema.union(", 1)[1][:400])
        self.assertIn('lora_type: Schema.const("tglokr").required()', schema)

    def test_tglokr_ui_config_builds_gated_module(self):
        adapted, _ = adapt_anima_config({
            "pretrained_model_name_or_path": "x.safetensors",
            "lora_type": "tglokr",
            "network_module": "lycoris.kohya",
            "lycoris_algo": "glokr",
            "train_time_gates": True,
            "time_gate_dim": 4,
            "kron_rank": 2,
            "lokr_factor": -1,
        })
        args = {i.split("=", 1)[0]: i.split("=", 1)[1] for i in adapted["network_args"]}
        self.assertEqual(args.get("algo"), "glokr")
        self.assertEqual(args.get("train_time_gates"), "True")
        self.assertNotIn("lora_type", adapted, "lora_type is UI-only, must not reach sd-scripts")

        net, _ = _build_glokr({"train_time_gates": args["train_time_gates"],
                               "time_gate_dim": args["time_gate_dim"]})
        self.assertTrue(all(hasattr(m, "time_gate_w") for m in net.unet_loras))

    def test_config_round_trip_keeps_tglokr_branch(self):
        # export/import infer lora_type from algo; algo=glokr alone would land on
        # plain GLoKR, so the time-gate flag must disambiguate in both directions.
        from mikazuki.utils.config_export import _ensure_gui_identity_fields
        from mikazuki.utils.config_import import _hydrate_lycoris_ui_fields_from_network_args

        exported = {"lycoris_algo": "glokr", "train_time_gates": True}
        _ensure_gui_identity_fields(exported, page_train_type="anima-lora")
        self.assertEqual(exported["lora_type"], "tglokr")

        plain = {"lycoris_algo": "glokr"}
        _ensure_gui_identity_fields(plain, page_train_type="anima-lora")
        self.assertEqual(plain["lora_type"], "glokr")

        from_args = {"network_args": ["algo=glokr", "train_time_gates=True"]}
        _ensure_gui_identity_fields(from_args, page_train_type="anima-lora")
        self.assertEqual(from_args["lora_type"], "tglokr")

        imported = {
            "network_module": "lycoris.kohya",
            "network_args": ["algo=glokr", "train_time_gates=True"],
        }
        _hydrate_lycoris_ui_fields_from_network_args(imported)
        self.assertEqual(imported.get("lora_type"), "tglokr")
        self.assertTrue(_is_truthy_flag(imported.get("train_time_gates")))

        imported_plain = {
            "network_module": "lycoris.kohya",
            "network_args": ["algo=glokr"],
        }
        _hydrate_lycoris_ui_fields_from_network_args(imported_plain)
        self.assertEqual(imported_plain.get("lora_type"), "glokr")


class Bf16EndToEndTests(unittest.TestCase):
    """A bf16 model with T-GLoKR must survive an fp32 timestep.

    Regression: the time-gate buffer was cast to bf16 along with the module, while
    the gate weights were explicitly .float() -> torch.dot raised on mixed dtypes.
    All other tglokr tests run in fp32, so only a bf16 end-to-end pass catches it.
    """

    def test_bf16_block_with_time_gates_and_fp32_timestep(self):
        from library import anima_models, attention
        from lycoris.kohya import create_network

        dt = torch.bfloat16
        torch.manual_seed(0)
        t_embedder = torch.nn.Sequential(
            anima_models.Timesteps(64),
            anima_models.TimestepEmbedding(64, 64, use_adaln_lora=True),
        ).to(dt)
        block = anima_models.Block(x_dim=64, context_dim=64, num_heads=4, use_adaln_lora=True).to(dt)

        dit = nn.Module()
        dit.blocks = nn.ModuleList([block])
        net = create_network(
            1.0, 8, 8, None, nn.Module(), dit, warn_on_unmatched=False,
            algo="glokr", factor="-1", kron_rank="2", train_gates="True",
            init_mode="nkp", train_time_gates="True", time_gate_dim="4",
        )
        net.apply_to(nn.Module(), dit, False, True)
        net.to(dt)
        self.assertTrue(net.unet_loras, "adapter should attach to the block")

        x = torch.randn(1, 1, 2, 2, 64, dtype=dt)
        ctx = torch.randn(1, 4, 64, dtype=dt)
        attn_params = attention.AttentionParams.create_attention_params("torch", False)

        for timesteps in (torch.tensor([[0.5]], dtype=torch.float32), torch.tensor([[0.5]], dtype=dt)):
            net.set_current_timestep(timesteps)
            emb, adaln = t_embedder(timesteps)
            self.assertEqual(emb.dtype, dt)
            # blocks run the AdaLN modulation with autocast disabled on the bf16 path
            with torch.autocast(device_type="cpu", dtype=torch.float32, enabled=False):
                out = block(x, emb, ctx, attn_params, False, adaln_lora_B_T_3D=adaln)
            self.assertEqual(out.dtype, dt)
            net.clear_current_timestep()

    def test_time_gate_freqs_not_persisted_in_state_dict(self):
        net, _ = _build_glokr({"train_time_gates": "True"})
        self.assertNotIn("time_gate_freqs", net.unet_loras[0].custom_state_dict())


class StaleAutosaveTests(unittest.TestCase):
    """A stale train_time_gates value in the browser autosave must never break submit.

    Regression: the tglokr branch declared train_time_gates as Schema.const(true).
    Browsers keep a raw (unvalidated) autosave of the form, so a leftover `false`
    from the earlier build made the whole union fail to match -> the schema threw
    inside the submit handler -> the request was never sent and the UI reported a
    misleading "network error".
    """

    def test_schema_uses_tolerant_boolean_not_const(self):
        schema = (PROJECT_ROOT / "mikazuki" / "schema" / "sd3-lora.ts").read_text(encoding="utf-8")
        tglokr_branch = schema.split('lora_type: Schema.const("tglokr").required()', 1)[1][:900]
        self.assertIn("train_time_gates: Schema.boolean()", tglokr_branch)
        self.assertNotIn("train_time_gates: Schema.const", tglokr_branch)

    def test_adapter_forces_flag_from_lora_type(self):
        base = {
            "pretrained_model_name_or_path": "x.safetensors",
            "network_module": "lycoris.kohya",
            "lycoris_algo": "glokr",
        }

        def gates_of(cfg):
            adapted, _ = adapt_anima_config(cfg)
            args = {i.split("=", 1)[0]: i.split("=", 1)[1] for i in adapted.get("network_args", [])}
            return args.get("train_time_gates")

        # stale False must not silently disable T-GLoKR
        self.assertEqual(gates_of({**base, "lora_type": "tglokr", "train_time_gates": False}), "True")
        self.assertEqual(gates_of({**base, "lora_type": "tglokr"}), "True")
        # stale True must not silently turn plain GLoKR into T-GLoKR
        self.assertIsNone(gates_of({**base, "lora_type": "glokr", "train_time_gates": True}))


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
            "train_time_gates": True,
            "time_gate_dim": 6,
            "lokr_factor": -1,
        })
        args = {item.split("=", 1)[0]: item.split("=", 1)[1] for item in adapted["network_args"]}
        self.assertEqual(args.get("train_time_gates"), "True")
        self.assertEqual(args.get("time_gate_dim"), "6")

        net, _ = _build_glokr({"train_time_gates": args["train_time_gates"], "time_gate_dim": args["time_gate_dim"]})
        m0 = net.unet_loras[0]
        self.assertEqual(m0.time_gate_w.shape, (3, 12))  # 2K with K=6


class GradientCheckpointRecomputeTests(unittest.TestCase):
    """CheckpointError 回归：backward 里的重算前向必须看到与首次前向相同的 t。

    旧训练循环在 forward 一结束就清空 current_timestep；gradient checkpointing
    在 backward 中重演前向时 _time_gate 走 t=None (g≡1) 回退分支，保存张量数
    少于首次前向 → torch.utils.checkpoint.CheckpointError（线上 1262 vs 1134）。
    """

    def _one_step(self, clear_before_backward):
        net, dit = _build_glokr({"train_time_gates": "True", "time_gate_dim": "4"})
        _perturb(net)
        x = torch.randn(2, DIM)
        net.set_current_timestep(torch.tensor([500.0, 500.0]))
        h = x
        for block in dit.blocks:
            h = torch_ckpt.checkpoint(block, h, use_reentrant=False)
        loss = h.sum()
        if clear_before_backward:
            net.clear_current_timestep()
        loss.backward()

    def test_old_behavior_clearing_t_breaks_recomputation(self):
        # 守住"测试抓得到原 bug"：提前清空必须触发 CheckpointError
        with self.assertRaises(torch_ckpt.CheckpointError):
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

        self.assertIsNone(fake.current_timestep, "preview must run with the t=None gate fallback")
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
