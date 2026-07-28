from __future__ import annotations

import importlib
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "sd-scripts"))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)
        self.conv = nn.Conv2d(4, 4, 1, bias=False)


class TinyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block()


class PiSSAAnimaTests(unittest.TestCase):
    def _build(
        self,
        method: str,
        export_mode: str,
        apply_conv2d: bool,
    ):
        module = importlib.import_module("networks.lora_anima")
        dit = TinyDiT()
        original_linear = dit.block.linear.weight.detach().clone()
        original_conv = dit.block.conv.weight.detach().clone()
        network = module.create_network(
            1.0,
            2,
            2,
            None,
            [nn.Module()],
            dit,
            pissa_init="true",
            pissa_method=method,
            pissa_niter="2",
            pissa_oversample="2",
            pissa_apply_conv2d=str(apply_conv2d),
            pissa_export_mode=export_mode,
        )
        network.apply_to([nn.Module()], dit, False, True)
        return network, dit, original_linear, original_conv

    def test_initial_adapter_preserves_original_linear_function(self):
        network, dit, original_weight, _original_conv = self._build(
            "svd",
            "LoRA无损兼容导出",
            False,
        )
        module = network.unet_loras[0]
        sample = torch.randn(3, 4)

        actual = dit.block.linear(sample)
        expected = torch.nn.functional.linear(sample, original_weight)

        self.assertEqual(type(module).__name__, "PiSSAModule")
        self.assertFalse(torch.equal(dit.block.linear.weight, original_weight))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_conv2d_flag_controls_one_by_one_convolution_initialization(self):
        without_conv, _dit, _linear, _conv = self._build(
            "svd",
            "LoRA无损兼容导出",
            False,
        )
        with_conv, dit, _linear, original_conv = self._build(
            "svd",
            "LoRA无损兼容导出",
            True,
        )

        self.assertEqual(len(without_conv.unet_loras), 1)
        self.assertEqual(len(with_conv.unet_loras), 2)
        sample = torch.randn(2, 4, 3, 3)
        expected = torch.nn.functional.conv2d(sample, original_conv)
        actual = dit.block.conv(sample)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_lossless_export_is_standard_lora_difference_with_resume_state(self):
        network, _dit, original_weight, _original_conv = self._build(
            "svd",
            "LoRA无损兼容导出",
            False,
        )
        module = network.unet_loras[0]
        with torch.no_grad():
            module.lora_up.weight.add_(0.25)
            module.lora_down.weight.mul_(1.1)
        exported = network._state_dict_for_save()
        prefix = module.lora_name
        down = exported[f"{prefix}.lora_down.weight"]
        up = exported[f"{prefix}.lora_up.weight"]
        alpha = float(exported[f"{prefix}.alpha"].item())
        export_scale = alpha / down.shape[0]
        exported_delta = up @ down * export_scale
        effective_delta = (
            dit_weight(module) - original_weight
            + module.get_weight(1.0)
        )

        self.assertEqual(down.shape[0], 4)
        self.assertIn(f"{prefix}.pissa_trained_down", exported)
        self.assertIn(f"{prefix}.pissa_initial_down", exported)
        torch.testing.assert_close(
            exported_delta,
            effective_delta,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_fast_export_keeps_original_rank(self):
        network, _dit, _original_weight, _original_conv = self._build(
            "svd",
            "LoRA快速近似导出",
            False,
        )
        module = network.unet_loras[0]
        with torch.no_grad():
            module.lora_up.weight.add_(0.25)
        exported = network._state_dict_for_save()

        self.assertEqual(
            exported[f"{module.lora_name}.lora_down.weight"].shape[0],
            2,
        )

    def test_invalid_randomized_svd_iteration_count_fails_explicitly(self):
        module = importlib.import_module("networks.lora_anima")
        with self.assertRaisesRegex(ValueError, "pissa_niter"):
            module.create_network(
                1.0,
                2,
                2,
                None,
                [nn.Module()],
                TinyDiT(),
                pissa_init="true",
                pissa_method="rsvd",
                pissa_niter="-1",
                pissa_oversample="2",
                pissa_apply_conv2d="false",
                pissa_export_mode="LoRA无损兼容导出",
            )

    def test_randomized_svd_resume_restores_effective_weight(self):
        module = importlib.import_module("networks.lora_anima")
        original_dit = TinyDiT()
        original_state = {
            key: value.detach().clone()
            for key, value in original_dit.state_dict().items()
        }
        torch.manual_seed(11)
        original_network = module.create_network(
            1.0,
            2,
            2,
            None,
            [nn.Module()],
            original_dit,
            pissa_init="true",
            pissa_method="rsvd",
            pissa_niter="1",
            pissa_oversample="0",
            pissa_apply_conv2d="false",
            pissa_export_mode="LoRA无损兼容导出",
        )
        original_network.apply_to([nn.Module()], original_dit, False, True)
        with torch.no_grad():
            original_network.unet_loras[0].lora_up.weight.add_(0.2)
        checkpoint = original_network._state_dict_for_save()
        sample = torch.randn(3, 4)
        expected = original_dit.block.linear(sample).detach()

        restored_dit = TinyDiT()
        restored_dit.load_state_dict(original_state)
        torch.manual_seed(999)
        restored_network = module.create_network(
            1.0,
            2,
            2,
            None,
            [nn.Module()],
            restored_dit,
            pissa_init="true",
            pissa_method="rsvd",
            pissa_niter="1",
            pissa_oversample="0",
            pissa_apply_conv2d="false",
            pissa_export_mode="LoRA无损兼容导出",
        )
        restored_network.apply_to([nn.Module()], restored_dit, False, True)
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "pissa.pt"
            torch.save(checkpoint, checkpoint_path)
            restored_network.load_weights(checkpoint_path)
        actual = restored_dit.block.linear(sample).detach()

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def dit_weight(module) -> torch.Tensor:
    return module.org_forward.__self__.weight.detach().float()


if __name__ == "__main__":
    unittest.main()
