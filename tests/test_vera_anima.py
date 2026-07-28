from __future__ import annotations

import importlib
from pathlib import Path
import sys
import unittest

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "sd-scripts"))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(8, 12)
        self.second = nn.Linear(12, 8)


class TinyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block()


class VeRAAnimaTests(unittest.TestCase):
    def _build(self, seed: int, save_projection: bool):
        module = importlib.import_module("networks.vera_anima")
        text_encoder = nn.Module()
        dit = TinyDiT()
        network = module.create_network(
            1.0,
            4,
            4,
            None,
            [text_encoder],
            dit,
            vera_projection_seed=str(seed),
            vera_save_projection=str(save_projection),
            vera_d_initial="0.1",
        )
        network.apply_to([text_encoder], dit, False, True)
        return network

    def test_projection_seed_is_deterministic(self):
        first = self._build(123, True)
        second = self._build(123, True)
        third = self._build(124, True)

        torch.testing.assert_close(first.vera_A, second.vera_A)
        torch.testing.assert_close(first.vera_B, second.vera_B)
        self.assertFalse(torch.equal(first.vera_A, third.vera_A))

    def test_seed_only_checkpoint_regenerates_projection(self):
        network = self._build(123, False)
        state = network.state_dict()

        self.assertNotIn("vera_A", state)
        self.assertNotIn("vera_B", state)
        self.assertIn("vera_projection_seed", state)

    def test_merge_delta_matches_forward_delta(self):
        network = self._build(123, True)
        module = network.unet_loras[0]
        with torch.no_grad():
            module.vera_lambda_b.fill_(0.25)
            module.vera_lambda_d.fill_(0.5)
        weight_delta = module.get_weight(1.0)
        sample = torch.randn(3, 8)
        direct_delta = torch.nn.functional.linear(sample, weight_delta)

        network.set_enabled(False)
        base = module.org_forward(sample)
        network.set_enabled(True)
        forward_delta = module(sample) - base

        torch.testing.assert_close(forward_delta, direct_delta)

    def test_seed_only_checkpoint_round_trip_restores_trainable_vectors(self):
        module = importlib.import_module("networks.vera_anima")
        original = self._build(123, False)
        with torch.no_grad():
            original.unet_loras[0].vera_lambda_b.fill_(0.25)
            original.unet_loras[0].vera_lambda_d.fill_(0.5)
        checkpoint = original.state_dict()

        restored_dit = TinyDiT()
        restored, returned_checkpoint = module.create_network_from_weights(
            1.0,
            None,
            None,
            [nn.Module()],
            restored_dit,
            weights_sd=checkpoint,
        )
        restored.apply_to([nn.Module()], restored_dit, False, True)
        load_result = restored.load_state_dict(returned_checkpoint, strict=False)

        self.assertEqual(load_result.unexpected_keys, [])
        torch.testing.assert_close(restored.vera_A, original.vera_A)
        torch.testing.assert_close(
            restored.unet_loras[0].vera_lambda_b,
            original.unet_loras[0].vera_lambda_b,
        )


if __name__ == "__main__":
    unittest.main()
