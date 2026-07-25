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


if __name__ == "__main__":
    unittest.main()
