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

    def test_sigma_with_shift_warns_but_does_not_mutate(self):
        args = self._args(timestep_sampling="sigma", discrete_flow_shift=3.0)
        with self.assertLogs(level="WARNING") as captured:
            self._trainer().assert_extra_args(args, self._dataset_group(), None)
        self.assertTrue(any("discrete_flow_shift" in msg for msg in captured.output))
        self.assertEqual(args.timestep_sampling, "sigma")
        self.assertEqual(args.discrete_flow_shift, 3.0)


class TimestepEmbeddingDtypeTests(unittest.TestCase):
    """fp32 timesteps must not poison the AdaLN path of a bf16 model.

    Regression: TimestepEmbedding returned the raw sinusoidal features
    (`emb_B_T_D = sample`) with the caller's dtype. Blocks consume that inside
    `torch.autocast(..., enabled=use_fp32)`, which DISABLES autocast on the bf16
    path, so an fp32 tensor met bf16 weights and training/sampling crashed with
    "expected mat1 and mat2 to have the same dtype".
    """

    def _run(self, t_dtype, weight_dtype):
        from library import anima_models

        torch.manual_seed(0)
        embedder = torch.nn.Sequential(
            anima_models.Timesteps(32),
            anima_models.TimestepEmbedding(32, 32, use_adaln_lora=True),
        ).to(weight_dtype)
        timesteps = torch.tensor([[0.5]], dtype=t_dtype)
        return embedder(timesteps)

    def test_fp32_timestep_yields_weight_dtype_embedding(self):
        emb, adaln = self._run(torch.float32, torch.bfloat16)
        self.assertEqual(emb.dtype, torch.bfloat16, "emb must follow the projection dtype")
        self.assertEqual(adaln.dtype, torch.bfloat16)

    def test_adaln_linear_accepts_the_embedding_without_autocast(self):
        # This is the exact call that crashed: Block._forward runs the AdaLN
        # modulation with autocast disabled on the bf16 path.
        emb, _ = self._run(torch.float32, torch.bfloat16)
        adaln_modulation = torch.nn.Sequential(
            torch.nn.SiLU(), torch.nn.Linear(32, 96, bias=False)
        ).to(torch.bfloat16)
        with torch.autocast(device_type="cpu", dtype=torch.float32, enabled=False):
            out = adaln_modulation(emb)  # must not raise
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_bf16_timestep_still_works(self):
        emb, _ = self._run(torch.bfloat16, torch.bfloat16)
        self.assertEqual(emb.dtype, torch.bfloat16)


class RotaryEmbeddingShortcutTests(unittest.TestCase):
    def test_full_rot_dim_matches_manual_rotation(self):
        # Anima's config has rot_dim == head_dim, so t_pass is empty and the cat is skipped.
        from library import anima_models

        S, B, H, D = 6, 2, 2, 8
        t = torch.randn(S, B, H, D)
        freqs = torch.randn(S, 1, 1, D)
        out = anima_models._apply_rotary_pos_emb_base(t, freqs)
        cos_, sin_ = torch.cos(freqs).to(t.dtype), torch.sin(freqs).to(t.dtype)
        expected = (t * cos_) + (anima_models._rotate_half(t, False) * sin_)
        self.assertEqual(out.shape, t.shape)
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_partial_rot_dim_keeps_pass_through_channels(self):
        from library import anima_models

        S, B, H, D, ROT = 6, 2, 2, 8, 4
        t = torch.randn(S, B, H, D)
        freqs = torch.randn(S, 1, 1, ROT)
        out = anima_models._apply_rotary_pos_emb_base(t, freqs)
        self.assertEqual(out.shape, t.shape)
        self.assertTrue(torch.equal(out[..., ROT:], t[..., ROT:]), "t_pass channels must pass through untouched")


@unittest.skipUnless(torch.cuda.is_available(), "cpu_offload checkpointing needs CUDA")
class CpuOffloadCheckpointingTests(unittest.TestCase):
    def test_multi_block_forward_backward_with_cpu_offload(self):
        # Regression: device inference from inputs made block 2+ crash (inputs already on CPU).
        from library import anima_models, attention

        device = torch.device("cuda")
        blocks = [anima_models.Block(x_dim=32, context_dim=16, num_heads=2).to(device) for _ in range(3)]
        for b in blocks:
            b.enable_gradient_checkpointing(cpu_offload=True)
            b.train()

        x = torch.randn(1, 2, 2, 2, 32, device=device)  # [B, T, H, W, D]
        emb = torch.randn(1, 2, 32, device=device)
        ctx = torch.randn(1, 4, 16, device=device)
        attn_params = attention.AttentionParams.create_attention_params("torch", False)

        for b in blocks:
            x = b(x, emb, ctx, attn_params, False)

        # cpu_offload leaves the last block's output on CPU — mirror the
        # forward_mini_train_dit recovery before the "final layer" stage.
        x = x.to(device)
        x.float().sum().backward()
        self.assertTrue(any(p.grad is not None for p in blocks[0].parameters()))
        self.assertTrue(any(p.grad is not None for p in blocks[-1].parameters()))


class AllocatorEnvInjectionTests(unittest.TestCase):
    def test_train_subprocess_gets_fragmentation_config(self):
        import os
        import sys as _sys

        sys.path.insert(0, str(PROJECT_ROOT))
        from mikazuki.process import build_accelerate_train_command

        saved = os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        try:
            _, env, _ = build_accelerate_train_command(trainer_file="x.py", toml_path="missing.toml")
            expected = (
                "garbage_collection_threshold:0.8,max_split_size_mb:512"
                if _sys.platform == "win32"
                else "expandable_segments:True"
            )
            self.assertEqual(env.get("PYTORCH_CUDA_ALLOC_CONF"), expected)

            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "user-custom"
            _, env, _ = build_accelerate_train_command(trainer_file="x.py", toml_path="missing.toml")
            self.assertEqual(env.get("PYTORCH_CUDA_ALLOC_CONF"), "user-custom", "user setting must win")
        finally:
            if saved is None:
                os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            else:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = saved


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
