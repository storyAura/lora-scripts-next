from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "sd-scripts"))

from networks.lora_fa_anima import (
    LoRAFAAdamW,
    correct_lorafa_gradient,
)


class LoRAFAOptimizerTests(unittest.TestCase):
    def test_gradient_correction_matches_closed_form(self):
        matrix_a = torch.tensor(
            [
                [1.0, 0.0, 1.0],
                [0.0, 2.0, 1.0],
            ]
        )
        gradient_b = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )
        scaling = 0.5
        damping = 1e-8

        corrected = correct_lorafa_gradient(
            matrix_a,
            gradient_b,
            scaling,
            damping,
        )
        gram = matrix_a @ matrix_a.T
        expected = (
            gradient_b
            @ torch.linalg.pinv(gram + damping * torch.eye(matrix_a.shape[0]))
            / scaling**2
        )

        torch.testing.assert_close(corrected, expected)

    def test_optimizer_updates_only_b_with_corrected_gradient(self):
        matrix_a = torch.nn.Parameter(
            torch.tensor(
                [
                    [1.0, 0.0, 1.0],
                    [0.0, 2.0, 1.0],
                ]
            ),
            requires_grad=False,
        )
        matrix_b = torch.nn.Parameter(torch.zeros(3, 2))
        matrix_b.grad = torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )
        original_a = matrix_a.detach().clone()
        optimizer = LoRAFAAdamW(
            [
                {
                    "params": [matrix_a, matrix_b],
                    "lorafa_pairs": [
                        (matrix_a, matrix_b, 0.5, "test"),
                    ],
                }
            ],
            lr=0.1,
            betas=(0.0, 0.0),
            eps=1.0,
            weight_decay=0.0,
            correct_bias=False,
            damping=1e-8,
        )

        optimizer.step()

        torch.testing.assert_close(matrix_a, original_a)
        self.assertFalse(torch.equal(matrix_b, torch.zeros_like(matrix_b)))

    def test_optimizer_rejects_groups_without_pair_metadata(self):
        parameter = torch.nn.Parameter(torch.ones(2, 2))
        with self.assertRaises(ValueError):
            LoRAFAAdamW(
                [{"params": [parameter]}],
                lr=0.1,
                betas=(0.9, 0.999),
                eps=1e-6,
                weight_decay=0.0,
                correct_bias=True,
                damping=1e-8,
            )


if __name__ == "__main__":
    unittest.main()
