from __future__ import annotations

import pytest
import torch

from mikazuki.hardware_capabilities import (
    HardwareCapabilityError,
    evaluate_flash_attention_hardware,
    evaluate_fp8_training_hardware,
    probe_fp8_frozen_linear_training,
)


def test_flash_attention_gate_checks_architecture_and_software_stack() -> None:
    supported = evaluate_flash_attention_hardware(
        True,
        9,
        0,
        "12.4",
        "2.7.0",
        "2.8.3",
    )
    assert supported.usable is True

    cases = (
        (False, 0, 0, "none", "2.7.0", "2.8.3", "CUDA"),
        (True, 7, 5, "12.4", "2.7.0", "2.8.3", "compute capability"),
        (True, 8, 0, "11.8", "2.7.0", "2.8.3", "CUDA>=12"),
        (True, 8, 0, "12.4", "2.1.0", "2.8.3", "torch>=2.2"),
        (True, 8, 0, "12.4", "2.7.0", "not-installed", "flash-attn"),
    )
    for cuda_available, major, minor, cuda_version, torch_version, package_version, message in cases:
        decision = evaluate_flash_attention_hardware(
            cuda_available,
            major,
            minor,
            cuda_version,
            torch_version,
            package_version,
        )
        assert decision.usable is False
        assert message in decision.reason


def test_fp8_gate_requires_native_fp8_generation() -> None:
    assert evaluate_fp8_training_hardware(
        True,
        8,
        9,
        "12.4",
        "2.7.0",
    ).usable

    ampere = evaluate_fp8_training_hardware(
        True,
        8,
        0,
        "12.4",
        "2.7.0",
    )
    assert ampere.usable is False
    assert "8.9" in ampere.reason

    old_cuda = evaluate_fp8_training_hardware(
        True,
        9,
        0,
        "11.8",
        "2.7.0",
    )
    assert old_cuda.usable is False
    assert "CUDA>=12" in old_cuda.reason


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (8, 9),
    reason="Native FP8 CUDA hardware is required",
)
def test_fp8_probe_runs_frozen_weight_forward_and_input_backward() -> None:
    result = probe_fp8_frozen_linear_training()
    assert result.usable, result.reason


def test_fp8_probe_failure_raises_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mikazuki.hardware_capabilities.probe_fp8_frozen_linear_training",
        lambda: type("Result", (), {"usable": False, "reason": "kernel failed"})(),
    )
    from mikazuki.hardware_capabilities import require_fp8_frozen_base_training

    with pytest.raises(HardwareCapabilityError, match="kernel failed"):
        require_fp8_frozen_base_training()
