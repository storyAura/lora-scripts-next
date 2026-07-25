# -*- coding: utf-8 -*-
"""Regression tests for training-loop fixes found in the 2026-07 training audit.

Covers:
- caption dropout must not produce an all-zero attention mask (SDPA NaN poison)
- text-encoder cache stores fp16 and loads back as fp32 (old fp32 caches still load)
- huber_schedule='snr' is rewritten to 'exponential' for the FlowMatch scheduler
- get_noisy_model_input_and_timesteps returns fp32 timesteps (no bf16 quantization)
"""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "sd-scripts"))

import torch

from library import strategy_anima


class CaptionDropoutMaskTests(unittest.TestCase):
    def test_dropped_sample_keeps_one_valid_mask_position(self):
        strategy = strategy_anima.AnimaTextEncodingStrategy()
        bsz, seq, dim = 4, 8, 16
        prompt_embeds = torch.randn(bsz, seq, dim)
        attn_mask = torch.ones(bsz, seq, dtype=torch.long)
        t5_ids = torch.ones(bsz, seq, dtype=torch.long)
        t5_mask = torch.ones(bsz, seq, dtype=torch.long)
        rates = torch.full((bsz,), 1.0)  # always drop

        embeds, mask, _, t5m = strategy.drop_cached_text_encoder_outputs(
            prompt_embeds, attn_mask, t5_ids, t5_mask, caption_dropout_rates=rates
        )

        self.assertTrue(torch.all(embeds == 0), "dropped embeds must be zeroed")
        row_sums = mask.sum(dim=1)
        self.assertTrue(
            torch.all(row_sums >= 1),
            "every dropped sample must keep >=1 valid mask position (all-zero mask => SDPA NaN)",
        )
        self.assertTrue(torch.all(t5m.sum(dim=1) >= 1))


class TextEncoderCacheDtypeTests(unittest.TestCase):
    def test_load_outputs_npz_upcasts_fp16_and_accepts_legacy_fp32(self):
        strategy = strategy_anima.AnimaTextEncoderOutputsCachingStrategy(
            cache_to_disk=True, batch_size=1, skip_disk_cache_validity_check=True
        )
        with tempfile.TemporaryDirectory() as td:
            for src_dtype in (np.float16, np.float32):
                npz_path = str(Path(td) / f"cache_{np.dtype(src_dtype).name}.npz")
                np.savez(
                    npz_path,
                    prompt_embeds=np.random.rand(4, 8).astype(src_dtype),
                    attn_mask=np.ones((4,), dtype=np.int32),
                    t5_input_ids=np.ones((4,), dtype=np.int32),
                    t5_attn_mask=np.ones((4,), dtype=np.int32),
                    caption_dropout_rate=np.float32(0.0),
                )
                loaded = strategy.load_outputs_npz(npz_path)
                self.assertEqual(loaded[0].dtype, np.float32, f"src={src_dtype}")


class HuberScheduleGuardTests(unittest.TestCase):
    def _args(self, **overrides):
        base = dict(
            fp8_base=False, fp8_base_unet=False, fp8_scaled=False,
            cache_text_encoder_outputs=False, cache_text_encoder_outputs_to_disk=False,
            network_train_unet_only=True, blocks_to_swap=None,
            cpu_offload_checkpointing=False, unsloth_offload_checkpointing=False,
            gradient_checkpointing=True, loss_type="l2", huber_schedule="snr",
            timestep_sampling="shift", discrete_flow_shift=3.0,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def _trainer(self):
        import anima_train_network

        return anima_train_network.AnimaNetworkTrainer()

    @staticmethod
    def _dataset_group():
        return SimpleNamespace(verify_bucket_reso_steps=lambda _steps: None)

    def test_huber_snr_falls_back_to_exponential(self):
        args = self._args(loss_type="huber", huber_schedule="snr")
        self._trainer().assert_extra_args(args, self._dataset_group(), None)
        self.assertEqual(args.huber_schedule, "exponential")

    def test_l2_keeps_huber_schedule_untouched(self):
        args = self._args(loss_type="l2", huber_schedule="snr")
        self._trainer().assert_extra_args(args, self._dataset_group(), None)
        self.assertEqual(args.huber_schedule, "snr")


class TimestepDtypeTests(unittest.TestCase):
    def test_noisy_input_timesteps_stay_fp32(self):
        from diffusers import FlowMatchEulerDiscreteScheduler
        from library import flux_train_utils

        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
        args = SimpleNamespace(
            timestep_sampling="shift", sigmoid_scale=1.0, discrete_flow_shift=3.0,
            weighting_scheme="uniform", logit_mean=None, logit_std=None, mode_scale=None,
            min_timestep=None, max_timestep=None, ip_noise_gamma=None,
            ip_noise_gamma_random_strength=False,
        )
        latents = torch.randn(2, 16, 8, 8)
        noise = torch.randn_like(latents)
        noisy, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            args, scheduler, latents, noise, torch.device("cpu"), torch.bfloat16
        )
        self.assertEqual(noisy.dtype, torch.bfloat16)
        self.assertEqual(timesteps.dtype, torch.float32, "timesteps must not be bf16-quantized")


if __name__ == "__main__":
    unittest.main()
