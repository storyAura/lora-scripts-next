from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import torch

from mikazuki.optimizer_configuration import (
    OptimizerConfigurationError,
    normalize_optimizer_configuration,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "sd-scripts"))


def optimizer_args(
    optimizer_type: str,
    values: list[str],
    learning_rate: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        use_8bit_adam=False,
        use_lion_optimizer=False,
        optimizer_type=optimizer_type,
        fused_backward_pass=False,
        gradient_accumulation_steps=1,
        optimizer_args=values,
        learning_rate=learning_rate,
        optimizer_schedulefree_wrapper=False,
    )


class OptimizerConfigurationTests(unittest.TestCase):
    def test_ademamix_alias_maps_to_full_path_and_structured_args(self):
        result = normalize_optimizer_configuration(
            {
                "optimizer_type": "AdEMAMix8bit",
                "ademamix_beta1": 0.9,
                "ademamix_beta2": 0.999,
                "ademamix_beta3": 0.9999,
                "ademamix_alpha": 5,
                "ademamix_t_alpha": 100,
                "ademamix_t_beta3": 200,
            }
        )

        self.assertEqual(
            result.values["optimizer_type"],
            "bitsandbytes.optim.AdEMAMix8bit",
        )
        self.assertIn(
            "betas=(0.9, 0.999, 0.9999)",
            result.values["optimizer_args"],
        )
        self.assertIn("alpha=5.0", result.values["optimizer_args"])
        self.assertIn("t_alpha=100", result.values["optimizer_args"])
        self.assertIn("t_beta3=200", result.values["optimizer_args"])
        self.assertNotIn("ademamix_beta1", result.values)

    def test_paged_ademamix_alias_maps_to_full_path(self):
        result = normalize_optimizer_configuration(
            {"optimizer_type": "PagedAdEMAMix8bit"}
        )

        self.assertEqual(
            result.values["optimizer_type"],
            "bitsandbytes.optim.PagedAdEMAMix8bit",
        )

    def test_fused_adamw_is_a_strict_adamw_preset(self):
        result = normalize_optimizer_configuration(
            {"optimizer_type": "AdamWFused"}
        )

        self.assertEqual(result.values["optimizer_type"], "AdamW")
        self.assertIn("fused=True", result.values["optimizer_args"])

    def test_conflicting_structured_and_custom_args_fail(self):
        with self.assertRaisesRegex(
            OptimizerConfigurationError,
            "two sources",
        ):
            normalize_optimizer_configuration(
                {
                    "optimizer_type": "AdEMAMix8bit",
                    "ademamix_alpha": 5,
                    "optimizer_args": ["alpha=7"],
                }
            )

    def test_invalid_ademamix_beta_fails(self):
        with self.assertRaisesRegex(
            OptimizerConfigurationError,
            "ademamix_beta3",
        ):
            normalize_optimizer_configuration(
                {
                    "optimizer_type": "AdEMAMix8bit",
                    "ademamix_beta1": 0.9,
                    "ademamix_beta2": 0.999,
                    "ademamix_beta3": 1.0,
                }
            )

    def test_lora_plus_rejects_single_global_rate_optimizers(self):
        with self.assertRaisesRegex(
            OptimizerConfigurationError,
            "LoRA\\+",
        ):
            normalize_optimizer_configuration(
                {
                    "lora_type": "lora_plus",
                    "optimizer_type": "Prodigy",
                }
            )

    def test_schedule_free_optimizer_state_round_trip(self):
        from library import train_util

        parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
        _name, _arguments, optimizer = train_util.get_optimizer(
            optimizer_args("AdamWScheduleFree", [], 1e-3),
            [parameter],
        )
        parameter.grad = torch.tensor([0.2, -0.3])
        optimizer.step()
        state = optimizer.state_dict()
        optimizer.eval()

        restored_parameter = torch.nn.Parameter(parameter.detach().clone())
        _name, _arguments, restored = train_util.get_optimizer(
            optimizer_args("AdamWScheduleFree", [], 1e-3),
            [restored_parameter],
        )
        restored.load_state_dict(state)
        restored.train()
        restored_parameter.grad = torch.tensor([0.1, -0.1])
        restored.step()

        self.assertTrue(torch.isfinite(restored_parameter).all())

    def test_schema_exposes_supported_optimizer_presets(self):
        schema = (
            PROJECT_ROOT / "mikazuki" / "schema" / "shared.ts"
        ).read_text(encoding="utf-8")

        for optimizer_name in (
            "AdamWFused",
            "AdamWScheduleFree",
            "SGDScheduleFree",
            "PagedAdamW",
            "PagedAdamW32bit",
            "AdEMAMix8bit",
            "PagedAdEMAMix8bit",
        ):
            self.assertIn(f'"{optimizer_name}"', schema)
        self.assertIn("ademamix_beta3", schema)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_short_ademamix_alias_performs_a_real_cuda_step(self):
        from library import train_util

        parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0], device="cuda"))
        _name, _arguments, optimizer = train_util.get_optimizer(
            optimizer_args(
                "AdEMAMix8bit",
                [
                    "betas=(0.9,0.999,0.9999)",
                    "alpha=5.0",
                    "min_8bit_size=1",
                ],
                1e-3,
            ),
            [parameter],
        )
        before = parameter.detach().clone()
        for _step in range(3):
            optimizer.zero_grad()
            parameter.square().sum().backward()
            optimizer.step()

        self.assertFalse(torch.equal(parameter, before))
        self.assertTrue(torch.isfinite(parameter).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_paged_ademamix_alias_performs_a_real_cuda_step(self):
        from library import train_util

        parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0], device="cuda"))
        _name, _arguments, optimizer = train_util.get_optimizer(
            optimizer_args(
                "PagedAdEMAMix8bit",
                [
                    "betas=(0.9,0.999,0.9999)",
                    "alpha=5.0",
                    "min_8bit_size=1",
                ],
                1e-3,
            ),
            [parameter],
        )
        before = parameter.detach().clone()
        for _step in range(3):
            optimizer.zero_grad()
            parameter.square().sum().backward()
            optimizer.step()

        self.assertFalse(torch.equal(parameter, before))
        self.assertTrue(torch.isfinite(parameter).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_paged_adamw32bit_performs_a_real_bf16_cuda_step(self):
        from library import train_util

        parameter = torch.nn.Parameter(
            torch.tensor([1.0, -1.0], device="cuda", dtype=torch.bfloat16)
        )
        _name, _arguments, optimizer = train_util.get_optimizer(
            optimizer_args(
                "PagedAdamW32bit",
                ["min_8bit_size=1"],
                0.1,
            ),
            [parameter],
        )
        before = parameter.detach().clone()
        for _step in range(3):
            optimizer.zero_grad()
            parameter.float().square().sum().backward()
            optimizer.step()

        self.assertFalse(torch.equal(parameter, before))
        self.assertTrue(torch.isfinite(parameter).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_fused_adamw_performs_a_real_cuda_step(self):
        from library import train_util

        parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0], device="cuda"))
        _name, _arguments, optimizer = train_util.get_optimizer(
            optimizer_args("AdamW", ["fused=True"], 1e-3),
            [parameter],
        )
        before = parameter.detach().clone()
        parameter.square().sum().backward()
        optimizer.step()

        self.assertFalse(torch.equal(parameter, before))
        self.assertTrue(torch.isfinite(parameter).all())


if __name__ == "__main__":
    unittest.main()
