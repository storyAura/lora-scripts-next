from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "vendor"))


def load_compute_merged_delta():
    try:
        functional = importlib.import_module("lycoris.modules.functional")
    except ModuleNotFoundError:
        return None
    return getattr(functional, "compute_merged_delta", None)


def build_merged_module(
    module_name: str,
    weight_decompose: bool,
    multiplier: float,
):
    base = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        base.weight.fill_(1.0)

    if module_name == "lokr":
        from lycoris.modules.lokr import LokrModule

        module = LokrModule(
            "test_lokr",
            base,
            multiplier=multiplier,
            lora_dim=1,
            alpha=1,
            factor=-1,
            full_matrix=True,
            weight_decompose=weight_decompose,
        )
    elif module_name == "loha":
        from lycoris.modules.loha import LohaModule

        module = LohaModule(
            "test_loha",
            base,
            multiplier=multiplier,
            lora_dim=1,
            alpha=1,
            weight_decompose=weight_decompose,
        )
    elif module_name == "locon":
        from lycoris.modules.locon import LoConModule

        module = LoConModule(
            "test_locon",
            base,
            multiplier=multiplier,
            lora_dim=1,
            alpha=1,
            weight_decompose=weight_decompose,
        )
    elif module_name == "bora_locon":
        from lycoris.modules.bora import LoConModule

        module = LoConModule(
            "test_bora_locon",
            base,
            multiplier=multiplier,
            lora_dim=1,
            alpha=1,
            weight_decompose=weight_decompose,
        )
    elif module_name == "bora":
        from lycoris.modules.bora import BoRAModule

        module = BoRAModule(
            "test_bora",
            base,
            multiplier=multiplier,
            lora_dim=1,
            alpha=1,
            weight_decompose=weight_decompose,
        )
    elif module_name == "tlora":
        from lycoris.modules.tlora import TLoraModule

        module = TLoraModule(
            "test_tlora",
            base,
            multiplier=multiplier,
            lora_dim=1,
            alpha=1,
            use_data_init=True,
        )
    else:
        raise ValueError(f"Unsupported test module: {module_name}")

    base.to(dtype=torch.bfloat16)
    return module, base


def capture_forward_delta(
    module,
    module_name: str,
    diff_value: float,
) -> torch.Tensor:
    controlled_diff = torch.full((4, 4), diff_value, dtype=torch.float32)
    captured: list[torch.Tensor] = []

    if module_name in {"lokr", "loha"}:
        module.get_weight = lambda shape: controlled_diff
    elif module_name in {"locon", "bora_locon", "bora"}:
        module.make_weight = lambda device: controlled_diff.to(device)
    elif module_name == "tlora":
        module.get_diff_weight = lambda **values: (
            controlled_diff.to(values["device"]),
            None,
        )
    else:
        raise ValueError(f"Unsupported test module: {module_name}")

    def capture_op(
        value: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        **values: object,
    ) -> torch.Tensor:
        captured.append(weight)
        return torch.zeros(
            (*value.shape[:-1], weight.shape[0]),
            dtype=value.dtype,
            device=value.device,
        )

    module.op = capture_op
    module(torch.zeros((1, 4), dtype=torch.bfloat16))
    if len(captured) != 1:
        raise AssertionError(f"Expected one delta for {module_name}, got {len(captured)}")
    return captured[0]


class MergedDeltaPrimitiveTests(unittest.TestCase):
    def test_small_bf16_update_survives_merge_subtraction(self):
        compute_merged_delta = load_compute_merged_delta()
        self.assertIsNotNone(compute_merged_delta, "compute_merged_delta is missing")

        base = torch.ones(4096, dtype=torch.bfloat16)
        diff = torch.full((4096,), 0.001, dtype=torch.float32, requires_grad=True)

        delta = compute_merged_delta(base, diff, 1.0, None)

        self.assertEqual(delta.dtype, torch.bfloat16)
        self.assertEqual(torch.count_nonzero(delta).item(), 4096)
        delta.float().sum().backward()
        self.assertIsNotNone(diff.grad)
        self.assertTrue(torch.isfinite(diff.grad).all())

    def test_transform_receives_fp32_weight(self):
        compute_merged_delta = load_compute_merged_delta()
        self.assertIsNotNone(compute_merged_delta, "compute_merged_delta is missing")
        observed_dtypes: list[torch.dtype] = []

        def transform(weight: torch.Tensor) -> torch.Tensor:
            observed_dtypes.append(weight.dtype)
            return weight

        delta = compute_merged_delta(
            torch.ones(4, dtype=torch.bfloat16),
            torch.full((4,), 0.001, dtype=torch.float32),
            0.5,
            transform,
        )

        self.assertEqual(observed_dtypes, [torch.float32])
        self.assertTrue(
            torch.allclose(
                delta.float(),
                torch.full((4,), 0.0005, dtype=torch.float32),
                atol=1e-5,
                rtol=0,
            )
        )


class MergedModulePrecisionTests(unittest.TestCase):
    def test_audited_modules_preserve_small_bf16_delta(self):
        cases = (
            ("lokr", False),
            ("lokr", True),
            ("loha", False),
            ("loha", True),
            ("locon", False),
            ("locon", True),
            ("bora_locon", False),
            ("bora_locon", True),
            ("bora", False),
            ("bora", True),
            ("tlora", False),
        )

        for module_name, weight_decompose in cases:
            with self.subTest(
                module=module_name,
                weight_decompose=weight_decompose,
            ):
                module, _base = build_merged_module(
                    module_name,
                    weight_decompose,
                    1.0,
                )
                try:
                    delta = capture_forward_delta(module, module_name, 0.001)
                except Exception as exc:
                    self.fail(f"{module_name} merged forward raised {type(exc).__name__}: {exc}")

                self.assertGreater(torch.count_nonzero(delta).item(), 0)
                self.assertTrue(torch.isfinite(delta).all())

    def test_zero_multiplier_disables_merged_delta(self):
        cases = (
            ("lokr", False),
            ("lokr", True),
            ("loha", False),
            ("loha", True),
            ("locon", False),
            ("locon", True),
            ("bora_locon", False),
            ("bora_locon", True),
            ("bora", False),
            ("bora", True),
            ("tlora", False),
        )

        for module_name, weight_decompose in cases:
            with self.subTest(
                module=module_name,
                weight_decompose=weight_decompose,
            ):
                module, _base = build_merged_module(
                    module_name,
                    weight_decompose,
                    0.0,
                )
                try:
                    delta = capture_forward_delta(module, module_name, 0.01)
                except Exception as exc:
                    self.fail(f"{module_name} merged forward raised {type(exc).__name__}: {exc}")

                self.assertEqual(torch.count_nonzero(delta).item(), 0)


class LycorisRankDropoutTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_lokr_family_rank_dropout_runs_on_cuda(self):
        device = torch.device("cuda")
        dtype = torch.bfloat16

        for module_name in ("lokr", "gsokr"):
            with self.subTest(module=module_name):
                base = nn.Linear(16, 16, bias=False)
                if module_name == "lokr":
                    from lycoris.modules.lokr import LokrModule

                    module = LokrModule(
                        "test_lokr_dropout",
                        base,
                        multiplier=1.0,
                        lora_dim=2,
                        alpha=2,
                        factor=-1,
                        rank_dropout=0.5,
                    )
                else:
                    from lycoris.modules.gsokr import GloKrSoraModule

                    module = GloKrSoraModule(
                        "test_gsokr_dropout",
                        base,
                        multiplier=1.0,
                        lora_dim=2,
                        alpha=2,
                        factor=-1,
                        rank_dropout=0.5,
                    )

                base.to(device=device, dtype=dtype)
                module.to(device=device, dtype=dtype)
                value = torch.randn(
                    (2, 16),
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )

                output = module(value)
                output.float().sum().backward()

                self.assertEqual(output.device.type, "cuda")
                self.assertTrue(torch.isfinite(output).all())
                self.assertIsNotNone(value.grad)
                self.assertTrue(torch.isfinite(value.grad).all())


if __name__ == "__main__":
    unittest.main()
