# -*- coding: utf-8 -*-
"""Preview sampling must feed per-step sigmas to timestep-aware adapters (T-LoRA / T-GLoKR)."""
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "sd-scripts"))

import torch  # noqa: E402

from library.anima_train_utils import do_sample, get_sample_sigmas  # noqa: E402
from networks.tlora import TLoRAModule  # noqa: E402


class _FakeDiT(torch.nn.Module):
    """Minimal stand-in: records the t it was called with, predicts zeros."""

    def __init__(self):
        super().__init__()
        self.seen_t = []

    def forward(self, x, t, crossattn_emb, padding_mask=None):
        self.seen_t.append(float(t.reshape(-1)[0]))
        return torch.zeros_like(x)


class DoSampleTimestepCallbackTests(unittest.TestCase):
    def _run(self, steps=5, guidance_scale=1.0, neg=None):
        dit = _FakeDiT()
        seen_cb = []
        do_sample(
            64,
            64,
            1,
            dit,
            torch.zeros(1, 4, 8),
            steps,
            torch.float32,
            torch.device("cpu"),
            guidance_scale,
            1.0,
            neg,
            scheduler="simple",
            timestep_callback=lambda s: seen_cb.append(float(s)),
        )
        return dit, seen_cb

    def test_callback_receives_every_step_sigma(self):
        steps = 5
        dit, seen_cb = self._run(steps=steps)
        expected = [float(s) for s in get_sample_sigmas(steps, 1.0, "simple")[:-1]]
        self.assertEqual(len(seen_cb), steps)
        for got, want in zip(seen_cb, expected):
            self.assertAlmostEqual(got, want, places=6)
        # the model must see the same t that the adapters were fed
        for got, want in zip(dit.seen_t, seen_cb):
            self.assertAlmostEqual(got, want, places=6)

    def test_cfg_fires_callback_once_per_step(self):
        steps = 4
        dit, seen_cb = self._run(steps=steps, guidance_scale=4.5, neg=torch.zeros(1, 4, 8))
        self.assertEqual(len(seen_cb), steps)
        # two model passes (pos + neg) per step, one injection per step
        self.assertEqual(len(dit.seen_t), steps * 2)

    def test_no_callback_keeps_previous_behavior(self):
        dit = _FakeDiT()
        do_sample(
            64, 64, 1, dit, torch.zeros(1, 4, 8), 3, torch.float32,
            torch.device("cpu"), 1.0, 1.0, None, scheduler="simple",
        )
        self.assertEqual(len(dit.seen_t), 3)


class _FakeNetwork:
    """weakref-able holder mimicking TLoRAAnimaNetwork's timestep attribute."""

    def __init__(self):
        self.current_timestep = None


class TLoraMaskRespondsToInjectedSigmaTests(unittest.TestCase):
    def _module(self):
        module = TLoRAModule(
            "lora_unet_test",
            torch.nn.Linear(8, 8),
            multiplier=1.0,
            lora_dim=8,
            alpha=8,
            tlora_min_rank=2,
        )
        network = _FakeNetwork()
        module.set_network(network)
        return module, network

    def _active_rank(self, module, network, sigma):
        network.current_timestep = torch.tensor([sigma], dtype=torch.float32)
        mask, _ = module._get_tlora_rank_mask_and_scale(torch.zeros(1, 8))
        return int(mask.reshape(-1).sum()) if mask is not None else None

    def test_pure_noise_sigma_masks_down_to_min_rank(self):
        module, network = self._module()
        self.assertEqual(self._active_rank(module, network, 1.0), 2)

    def test_clean_sigma_uses_full_rank(self):
        module, network = self._module()
        self.assertEqual(self._active_rank(module, network, 0.0), 8)

    def test_without_timestep_mask_is_disabled(self):
        module, network = self._module()
        network.current_timestep = None
        mask, scale = module._get_tlora_rank_mask_and_scale(torch.zeros(1, 8))
        self.assertIsNone(mask)
        self.assertIsNone(scale)


if __name__ == "__main__":
    unittest.main()
