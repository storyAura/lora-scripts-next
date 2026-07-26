# -*- coding: utf-8 -*-
"""Tests for the Anima preview beta scheduler (--sch flag + sigma schedule)."""
import sys
import unittest
from pathlib import Path

import numpy as np

from mikazuki.utils import train_utils

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "sd-scripts"))

import torch  # noqa: E402

from library.train_util import line_to_prompt_dict  # noqa: E402
from library.anima_train_utils import _beta_ppf, get_sample_sigmas  # noqa: E402


class PromptLineSchedulerTests(unittest.TestCase):
    def test_build_line_emits_sch_flag_and_parses_back(self):
        line = train_utils.build_sample_prompt_line(
            "1girl, solo",
            "lowres",
            width=1024,
            height=1024,
            cfg=4.5,
            steps=40,
            seed=42,
            sampler="euler",
            scheduler="beta",
        )
        self.assertIn(" --sch beta", line)
        parsed = line_to_prompt_dict(line)
        self.assertEqual(parsed["scheduler"], "beta")
        self.assertEqual(parsed["sample_sampler"], "euler")
        self.assertEqual(parsed["sample_steps"], 40)

    def test_no_scheduler_keeps_line_unchanged(self):
        line = train_utils.build_sample_prompt_line("1girl", "lowres")
        self.assertNotIn("--sch", line)


class BetaSigmaScheduleTests(unittest.TestCase):
    def test_simple_matches_legacy_linspace_shift(self):
        steps, shift = 40, 3.0
        sigmas = get_sample_sigmas(steps, shift, "simple")
        legacy = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)
        legacy = (legacy * shift) / (1 + (shift - 1) * legacy)
        self.assertEqual(sigmas.shape, (steps + 1,))
        self.assertTrue(torch.allclose(sigmas, legacy))

    def test_beta_endpoints_and_monotonic(self):
        steps = 40
        sigmas = get_sample_sigmas(steps, 1.0, "beta")
        self.assertEqual(sigmas.shape, (steps + 1,))
        self.assertAlmostEqual(float(sigmas[0]), 1.0, places=5)
        self.assertAlmostEqual(float(sigmas[-1]), 0.0, places=6)
        diffs = sigmas[:-1] - sigmas[1:]
        self.assertTrue(bool(torch.all(diffs > 0)), "sigmas must be strictly decreasing")

    def test_beta_concentrates_steps_at_both_ends(self):
        steps = 40
        sigmas = get_sample_sigmas(steps, 1.0, "beta")
        diffs = (sigmas[:-1] - sigmas[1:]).numpy()
        mid = diffs[len(diffs) // 2]
        self.assertLess(diffs[0], mid, "steps near sigma=1.0 must be denser than mid-schedule")
        self.assertLess(diffs[-1], mid, "steps near sigma=0.0 must be denser than mid-schedule")

    def test_beta_ppf_symmetry(self):
        # Beta(a, a) is symmetric: ppf(0.5) = 0.5 and ppf(1-q) = 1 - ppf(q).
        self.assertAlmostEqual(float(_beta_ppf(np.array([0.5]), 0.6, 0.6)[0]), 0.5, places=3)
        q = np.linspace(0.01, 0.99, 33)
        np.testing.assert_allclose(_beta_ppf(1.0 - q, 0.6, 0.6), 1.0 - _beta_ppf(q, 0.6, 0.6), atol=2e-3)

    def test_beta_composes_with_flow_shift(self):
        steps, shift = 30, 3.0
        base = get_sample_sigmas(steps, 1.0, "beta")
        shifted = get_sample_sigmas(steps, shift, "beta")
        expected = (base * shift) / (1 + (shift - 1) * base)
        self.assertTrue(torch.allclose(shifted, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
