# -*- coding: utf-8 -*-
"""Tests for the mem_efficient (PyTorch Efficient SDPA) attention backend.

mem_efficient forces SDPBackend.EFFICIENT_ATTENTION inside a local context:
no silent fallback to flash/math kernels, fail-fast on unsupported setups.
"""
import argparse
import io
import sys
import unittest
from contextlib import nullcontext, redirect_stderr
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "sd-scripts"))

import torch

from library import anima_train_utils
from library.attention import AttentionParams, _sdpa_backend_context, attention


class MemEfficientBackendTests(unittest.TestCase):
    def test_cli_rejects_sageattention_for_training(self):
        parser = argparse.ArgumentParser()
        anima_train_utils.add_anima_training_arguments(parser)

        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit):
            parser.parse_args(["--attn_mode", "sageattn"])

        self.assertIn("invalid choice", stderr.getvalue())

    def test_dispatcher_rejects_sageattention_before_kernel_call(self):
        called = False

        def inference_only_kernel(q, k, v):
            nonlocal called
            called = True
            return q

        q = torch.randn(1, 4, 1, 4, requires_grad=True)
        with mock.patch("library.attention.sageattn", inference_only_kernel):
            with self.assertRaisesRegex(RuntimeError, "does not support training backward"):
                attention(q, q, q, AttentionParams("sageattn", False))

        self.assertFalse(called)

    def test_context_is_null_for_other_modes(self):
        for mode in (None, "torch", "xformers", "flash", "sageattn"):
            self.assertIsInstance(_sdpa_backend_context(mode), nullcontext)

    def test_supports_fp32_flags(self):
        self.assertFalse(AttentionParams("mem_efficient").supports_fp32)
        self.assertFalse(AttentionParams("flash").supports_fp32)
        self.assertTrue(AttentionParams("torch").supports_fp32)

    def test_cli_choices_include_mem_efficient(self):
        source = (PROJECT_ROOT / "vendor" / "sd-scripts" / "library" / "anima_train_utils.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"mem_efficient"', source.split('"--attn_mode"', 1)[1][:300])

    def test_torch_baseline_cpu(self):
        q = torch.randn(2, 8, 2, 4)
        k = torch.randn(2, 8, 2, 4)
        v = torch.randn(2, 8, 2, 4)
        out = attention(q, k, v, AttentionParams("torch", False))
        self.assertEqual(tuple(out.shape), (2, 8, 8))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for the efficient kernel")
    def test_mem_efficient_forward_backward_matches_torch(self):
        device = "cuda"
        torch.manual_seed(0)
        q = torch.randn(2, 8, 2, 8, device=device, dtype=torch.float16, requires_grad=True)
        k = torch.randn(2, 8, 2, 8, device=device, dtype=torch.float16)
        v = torch.randn(2, 8, 2, 8, device=device, dtype=torch.float16)

        out_me = attention(q, k, v, AttentionParams("mem_efficient", False))
        out_me.float().sum().backward()
        self.assertIsNotNone(q.grad)
        self.assertTrue(torch.isfinite(out_me).all())
        self.assertTrue(torch.isfinite(q.grad).all())

        with torch.no_grad():
            out_torch = attention(
                q.detach().clone(), k.clone(), v.clone(), AttentionParams("torch", False)
            )
        self.assertTrue(
            torch.allclose(out_me.detach().float(), out_torch.float(), atol=2e-2),
            "mem_efficient output diverged from default SDPA",
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA for the efficient kernel")
    def test_mem_efficient_split_attn_gpu(self):
        device = "cuda"
        q = torch.randn(2, 8, 2, 8, device=device, dtype=torch.float16)
        k = torch.randn(2, 8, 2, 8, device=device, dtype=torch.float16)
        v = torch.randn(2, 8, 2, 8, device=device, dtype=torch.float16)
        out = attention(q, k, v, AttentionParams("mem_efficient", True))
        self.assertEqual(tuple(out.shape), (2, 8, 16))

    @unittest.skipIf(torch.cuda.is_available(), "CPU-only fail-fast semantics check")
    def test_mem_efficient_fails_fast_on_cpu(self):
        q = torch.randn(1, 4, 1, 4)
        with self.assertRaises(RuntimeError):
            attention(q, q.clone(), q.clone(), AttentionParams("mem_efficient", False))


if __name__ == "__main__":
    unittest.main()
