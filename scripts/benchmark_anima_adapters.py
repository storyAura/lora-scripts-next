from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
from pathlib import Path
import sys
import time
from typing import TypedDict

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SD_SCRIPTS_ROOT = PROJECT_ROOT / "vendor" / "sd-scripts"
if str(SD_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SD_SCRIPTS_ROOT))

ALGORITHM_MODULES = {
    "lora": "lora_anima",
    "lora_fa": "lora_fa_anima",
    "vera": "vera_anima",
    "pissa": "pissa_anima",
    "delora": "delora_anima",
    "waveft": "waveft_anima",
    "deft": "deft_anima",
    "moslora": "moslora_anima",
    "tlora": "tlora_anima",
}
DEFAULT_ALGORITHMS = tuple(ALGORITHM_MODULES)


class BenchmarkResult(TypedDict):
    algorithm: str
    device: str
    dtype: str
    trainable_parameters: int
    checkpoint_bytes: int
    initialization_ms: float
    forward_backward_ms: float
    samples_per_second: float
    baseline_allocated_bytes: int | None
    baseline_reserved_bytes: int | None
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    peak_incremental_allocated_bytes: int | None
    peak_incremental_reserved_bytes: int | None


class Block(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.proj_in = nn.Linear(features, features, bias=False)
        self.activation = nn.GELU()
        self.proj_out = nn.Linear(features, features, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.proj_in(inputs))
        return inputs + self.proj_out(hidden)


class BenchmarkDiT(nn.Module):
    def __init__(self, features: int, block_count: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(features) for _ in range(block_count)]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


def parse_algorithms(value: str) -> tuple[str, ...]:
    parsed = tuple(
        item.strip().lower()
        for item in value.split(",")
        if item.strip()
    )
    if not parsed:
        raise ValueError("algorithms must contain at least one name")
    duplicates = tuple(
        name
        for index, name in enumerate(parsed)
        if name in parsed[:index]
    )
    if duplicates:
        raise ValueError(
            f"duplicate algorithm entries are not allowed: {duplicates!r}"
        )
    unknown = tuple(
        name for name in parsed if name not in ALGORITHM_MODULES
    )
    if unknown:
        raise ValueError(
            f"unknown algorithms {unknown!r}; expected a subset of "
            f"{tuple(ALGORITHM_MODULES)!r}"
        )
    return parsed


def _positive_int(value: int, field_name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(
            f"{field_name} must be greater than zero, received {parsed}"
        )
    return parsed


def _non_negative_int(value: int, field_name: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(
            f"{field_name} must be non-negative, received {parsed}"
        )
    return parsed


def _validate_shape(
    features: int,
    blocks: int,
    batch_size: int,
    tokens: int,
    rank: int,
    warmup_iterations: int,
    measured_iterations: int,
) -> None:
    validated_features = _positive_int(features, "features")
    _positive_int(blocks, "blocks")
    _positive_int(batch_size, "batch_size")
    _positive_int(tokens, "tokens")
    validated_rank = _positive_int(rank, "rank")
    _non_negative_int(warmup_iterations, "warmup_iterations")
    _positive_int(measured_iterations, "measured_iterations")
    if validated_rank > validated_features:
        raise ValueError(
            f"rank must not exceed features, received rank={validated_rank} "
            f"and features={validated_features}"
        )


def _algorithm_arguments(
    algorithm: str,
    features: int,
    rank: int,
) -> dict[str, object]:
    arguments: dict[str, dict[str, object]] = {
        "lora": {},
        "lora_fa": {},
        "vera": {
            "vera_projection_seed": 42,
            "vera_save_projection": True,
            "vera_d_initial": 0.1,
        },
        "pissa": {
            "pissa_init": True,
            "pissa_method": "rsvd",
            "pissa_niter": 2,
            "pissa_oversample": 8,
            "pissa_apply_conv2d": False,
            "pissa_export_mode": "lossless",
        },
        "delora": {"delora_lambda": 15.0},
        "waveft": {
            "waveft_n_frequency": min(2592, features * features),
            "waveft_scaling": 25.0,
            "waveft_random_loc_seed": 777,
            "waveft_use_idwt": True,
            "waveft_wavelet_family": "db1",
        },
        "deft": {
            "deft_decomposition_method": "qr",
            "deft_alpha": 0,
            "deft_init_scale": 1.0,
            "deft_init_weights": True,
        },
        "moslora": {"moslora_mixer_init": "kaiming"},
        "tlora": {
            "tlora_min_rank": max(1, rank // 8),
            "tlora_rank_schedule": "cosine",
            "tlora_orthogonal_init": True,
        },
    }
    try:
        return dict(arguments[algorithm])
    except KeyError as error:
        raise ValueError(f"unsupported algorithm {algorithm!r}") from error


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _state_size_bytes(module: nn.Module) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in module.state_dict().values()
    )


def _set_timestep(
    network: nn.Module,
    device: torch.device,
    batch_size: int,
) -> None:
    setter = getattr(network, "set_current_timestep", None)
    if callable(setter):
        setter(
            torch.linspace(
                1000.0,
                0.0,
                batch_size,
                device=device,
                dtype=torch.float32,
            )
        )


def _one_iteration(
    model: nn.Module,
    network: nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> None:
    network.zero_grad(set_to_none=True)
    _set_timestep(network, device, batch_size)
    output = model(inputs)
    loss = output.float().square().mean()
    loss.backward()
    gradients = tuple(
        parameter.grad
        for parameter in network.parameters()
        if parameter.requires_grad and parameter.grad is not None
    )
    if not gradients:
        raise RuntimeError("adapter backward produced no trainable gradients")
    if not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise FloatingPointError(
            "adapter backward produced a non-finite gradient"
        )


def _run_one(
    algorithm: str,
    device: torch.device,
    dtype: torch.dtype,
    features: int,
    blocks: int,
    batch_size: int,
    tokens: int,
    rank: int,
    warmup_iterations: int,
    measured_iterations: int,
    seed: int,
) -> BenchmarkResult:
    module = importlib.import_module(
        f"networks.{ALGORITHM_MODULES[algorithm]}"
    )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    initialization_started = time.perf_counter()
    model = BenchmarkDiT(features, blocks)
    model.requires_grad_(False)
    network = module.create_network(
        1.0,
        rank,
        rank,
        None,
        [],
        model,
        None,
        **_algorithm_arguments(algorithm, features, rank),
    )
    network.apply_to([], model, False, True)
    model.to(device=device, dtype=dtype)
    network.to(device=device, dtype=dtype)
    inputs = torch.randn(
        batch_size,
        tokens,
        features,
        device=device,
        dtype=dtype,
    )
    _synchronize(device)
    initialization_ms = (
        time.perf_counter() - initialization_started
    ) * 1000.0

    trainable_parameters = sum(
        parameter.numel()
        for parameter in network.parameters()
        if parameter.requires_grad
    )
    if trainable_parameters <= 0:
        raise RuntimeError(
            f"{algorithm} created no trainable adapter parameters"
        )
    checkpoint_bytes = _state_size_bytes(network)
    baseline_allocated = (
        torch.cuda.memory_allocated(device)
        if device.type == "cuda"
        else None
    )
    baseline_reserved = (
        torch.cuda.memory_reserved(device)
        if device.type == "cuda"
        else None
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(warmup_iterations):
        _one_iteration(
            model,
            network,
            inputs,
            device,
            batch_size,
        )
    _synchronize(device)

    measured_started = time.perf_counter()
    for _ in range(measured_iterations):
        _one_iteration(
            model,
            network,
            inputs,
            device,
            batch_size,
        )
    _synchronize(device)
    measured_seconds = time.perf_counter() - measured_started
    average_seconds = measured_seconds / measured_iterations
    if not math.isfinite(average_seconds) or average_seconds <= 0:
        raise RuntimeError(
            f"invalid measured duration for {algorithm}: "
            f"{average_seconds!r}"
        )

    peak_allocated = (
        torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(device)
        if device.type == "cuda"
        else None
    )
    peak_incremental_allocated = (
        max(0, peak_allocated - baseline_allocated)
        if peak_allocated is not None and baseline_allocated is not None
        else None
    )
    peak_incremental_reserved = (
        max(0, peak_reserved - baseline_reserved)
        if peak_reserved is not None and baseline_reserved is not None
        else None
    )
    return {
        "algorithm": algorithm,
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "trainable_parameters": trainable_parameters,
        "checkpoint_bytes": checkpoint_bytes,
        "initialization_ms": initialization_ms,
        "forward_backward_ms": average_seconds * 1000.0,
        "samples_per_second": batch_size / average_seconds,
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_incremental_allocated_bytes": peak_incremental_allocated,
        "peak_incremental_reserved_bytes": peak_incremental_reserved,
    }


def run_benchmarks(
    algorithms: tuple[str, ...],
    device: torch.device,
    dtype: torch.dtype,
    features: int,
    blocks: int,
    batch_size: int,
    tokens: int,
    rank: int,
    warmup_iterations: int,
    measured_iterations: int,
    seed: int,
) -> list[BenchmarkResult]:
    _validate_shape(
        features,
        blocks,
        batch_size,
        tokens,
        rank,
        warmup_iterations,
        measured_iterations,
    )
    unknown = tuple(
        algorithm
        for algorithm in algorithms
        if algorithm not in ALGORITHM_MODULES
    )
    if unknown:
        raise ValueError(f"unknown algorithms {unknown!r}")
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 benchmark is not supported on CPU")
    results: list[BenchmarkResult] = []
    for algorithm in algorithms:
        results.append(
            _run_one(
                algorithm,
                device,
                dtype,
                features,
                blocks,
                batch_size,
                tokens,
                rank,
                warmup_iterations,
                measured_iterations,
                seed,
            )
        )
    return results


def _resolve_device(value: str) -> torch.device:
    normalized = value.strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {value!r} was requested but CUDA is unavailable"
        )
    return device


def _resolve_dtype(value: str, device: torch.device) -> torch.dtype:
    choices = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    try:
        dtype = choices[value.strip().lower()]
    except KeyError as error:
        raise ValueError(
            f"dtype must be one of {tuple(choices)!r}, received {value!r}"
        ) from error
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 benchmark is not supported on CPU")
    if (
        device.type == "cuda"
        and dtype == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError(
            "BF16 benchmark was requested but the CUDA device does not "
            "support BF16"
        )
    return dtype


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark real Anima adapter modules on a deterministic "
            "synthetic DiT workload."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algorithms",
        default=",".join(DEFAULT_ALGORITHMS),
        help="comma-separated algorithm names",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("fp32", "bf16", "fp16"),
        default="bf16",
    )
    parser.add_argument("--features", type=int, default=512)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--warmup-iterations", type=int, default=3)
    parser.add_argument("--measured-iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON result path",
    )
    return parser


def main() -> int:
    arguments = _argument_parser().parse_args()
    device = _resolve_device(arguments.device)
    dtype = _resolve_dtype(arguments.dtype, device)
    algorithms = parse_algorithms(arguments.algorithms)
    results = run_benchmarks(
        algorithms,
        device,
        dtype,
        arguments.features,
        arguments.blocks,
        arguments.batch_size,
        arguments.tokens,
        arguments.rank,
        arguments.warmup_iterations,
        arguments.measured_iterations,
        arguments.seed,
    )
    payload = {
        "environment": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "CPU"
            ),
            "dtype": str(dtype).removeprefix("torch."),
        },
        "workload": {
            "features": arguments.features,
            "blocks": arguments.blocks,
            "batch_size": arguments.batch_size,
            "tokens": arguments.tokens,
            "rank": arguments.rank,
            "warmup_iterations": arguments.warmup_iterations,
            "measured_iterations": arguments.measured_iterations,
            "seed": arguments.seed,
        },
        "results": results,
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    if arguments.output is None:
        print(serialized)
    else:
        output_path = arguments.output.resolve()
        if not output_path.parent.is_dir():
            raise FileNotFoundError(
                f"output parent directory does not exist: "
                f"{output_path.parent}"
            )
        output_path.write_text(serialized + "\n", encoding="utf-8")
        print(f"Wrote benchmark results to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
