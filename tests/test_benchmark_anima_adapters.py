from __future__ import annotations

import math

import pytest
import torch

from scripts import benchmark_anima_adapters


def test_parse_algorithms_is_strict_and_ordered() -> None:
    assert benchmark_anima_adapters.parse_algorithms(
        "lora,delora,waveft"
    ) == ("lora", "delora", "waveft")
    with pytest.raises(ValueError, match="unknown algorithms"):
        benchmark_anima_adapters.parse_algorithms("lora,imaginary")
    with pytest.raises(ValueError, match="duplicate algorithm"):
        benchmark_anima_adapters.parse_algorithms("lora,lora")


def test_cpu_microbenchmark_reports_comparable_metrics() -> None:
    results = benchmark_anima_adapters.run_benchmarks(
        algorithms=("lora", "moslora"),
        device=torch.device("cpu"),
        dtype=torch.float32,
        features=8,
        blocks=1,
        batch_size=2,
        tokens=3,
        rank=2,
        warmup_iterations=1,
        measured_iterations=2,
        seed=123,
    )

    assert [result["algorithm"] for result in results] == [
        "lora",
        "moslora",
    ]
    for result in results:
        assert result["device"] == "cpu"
        assert result["dtype"] == "float32"
        assert result["trainable_parameters"] > 0
        assert result["checkpoint_bytes"] > 0
        assert result["initialization_ms"] >= 0
        assert result["forward_backward_ms"] > 0
        assert result["samples_per_second"] > 0
        assert math.isfinite(result["forward_backward_ms"])
        assert math.isfinite(result["samples_per_second"])
        assert result["peak_allocated_bytes"] is None
        assert result["peak_reserved_bytes"] is None
        assert result["baseline_reserved_bytes"] is None
        assert result["peak_incremental_allocated_bytes"] is None
        assert result["peak_incremental_reserved_bytes"] is None


def test_invalid_benchmark_shape_fails_before_execution() -> None:
    with pytest.raises(ValueError, match="features"):
        benchmark_anima_adapters.run_benchmarks(
            algorithms=("lora",),
            device=torch.device("cpu"),
            dtype=torch.float32,
            features=0,
            blocks=1,
            batch_size=1,
            tokens=1,
            rank=1,
            warmup_iterations=0,
            measured_iterations=1,
            seed=1,
        )
