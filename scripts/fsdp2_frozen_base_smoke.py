from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from accelerate import Accelerator
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[1]
SD_SCRIPTS_ROOT = REPO_ROOT / "vendor" / "sd-scripts"
sys.path.insert(0, str(SD_SCRIPTS_ROOT))

from library.fsdp2_frozen_base import shard_frozen_base_model_fsdp2  # noqa: E402


class Block(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(width, width, bias=False)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(width, width * 2, bias=False),
            torch.nn.SiLU(),
            torch.nn.Linear(width * 2, width, bias=False),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.mlp(self.proj(value))


class FrozenBase(torch.nn.Module):
    def __init__(self, width: int, depth: int) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block(width) for _ in range(depth)])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = value
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


class Adapter(torch.nn.Module):
    def __init__(self, width: int, rank: int) -> None:
        super().__init__()
        self.down = torch.nn.Linear(width, rank, bias=False)
        self.up = torch.nn.Linear(rank, width, bias=False)
        torch.nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.up.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--cpu-offload", action="store_true")
    return parser.parse_args()


def _clone_adapter_state(adapter: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in adapter.state_dict().items()
    }


def main() -> None:
    args = parse_args()
    accelerator = Accelerator()
    if accelerator.num_processes < 2:
        raise RuntimeError(
            f"FSDP2 smoke test requires at least two processes, got {accelerator.num_processes}"
        )
    if accelerator.device.type != "cuda":
        raise RuntimeError(
            f"FSDP2 smoke test requires CUDA devices, got {accelerator.device}"
        )
    if args.width < 8 or args.depth < 2 or args.rank < 1 or args.steps < 1:
        raise ValueError(
            f"invalid smoke dimensions: width={args.width}, depth={args.depth}, "
            f"rank={args.rank}, steps={args.steps}"
        )

    torch.manual_seed(20260728)
    base = FrozenBase(args.width, args.depth)
    base.requires_grad_(False)
    sharding_report = shard_frozen_base_model_fsdp2(
        base,
        accelerator.device,
        ("Block",),
        bool(args.cpu_offload),
    )

    adapter = Adapter(args.width, args.rank).to(accelerator.device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
        foreach=None,
        maximize=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )
    adapter, optimizer = accelerator.prepare(adapter, optimizer)
    torch.cuda.reset_peak_memory_stats(accelerator.device)

    for step in range(args.steps):
        generator = torch.Generator(device=accelerator.device)
        generator.manual_seed(20260728 + step)
        value = torch.randn(
            4,
            args.width,
            generator=generator,
            device=accelerator.device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            base_output = base(value)
        output = base_output + adapter(value)
        loss = output.square().mean()
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    accelerator.wait_for_everyone()
    unwrapped_adapter = accelerator.unwrap_model(adapter)
    reference_state = _clone_adapter_state(unwrapped_adapter)
    state_dir = args.output_dir / "accelerate-state"
    state_dir.mkdir(parents=True, exist_ok=True)
    accelerator.save_state(str(state_dir))
    accelerator.wait_for_everyone()

    with torch.no_grad():
        for parameter in unwrapped_adapter.parameters():
            parameter.add_(1.0)
    accelerator.load_state(str(state_dir))
    accelerator.wait_for_everyone()

    restored_state = _clone_adapter_state(accelerator.unwrap_model(adapter))
    for key, reference_value in reference_state.items():
        torch.testing.assert_close(restored_state[key], reference_value)

    gathered_up = accelerator.gather(
        accelerator.unwrap_model(adapter).up.weight.detach().reshape(1, -1)
    )
    reference_up = gathered_up[:1]
    for rank_index in range(accelerator.num_processes):
        rank_up = gathered_up[rank_index : rank_index + 1]
        torch.testing.assert_close(rank_up, reference_up)

    peak_memory_bytes = torch.cuda.max_memory_allocated(accelerator.device)
    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        adapter_path = args.output_dir / "adapter.safetensors"
        save_file(reference_state, str(adapter_path))
        result = {
            "status": "passed",
            "world_size": accelerator.num_processes,
            "sharded_transformer_modules": len(
                sharding_report.transformer_module_names
            ),
            "base_parameter_count": sharding_report.base_parameter_count,
            "adapter_parameter_count": sum(
                parameter.numel() for parameter in unwrapped_adapter.parameters()
            ),
            "cpu_offload": sharding_report.cpu_offload,
            "peak_cuda_memory_bytes_rank0": peak_memory_bytes,
            "state_directory": str(state_dir),
            "adapter_checkpoint": str(adapter_path),
        }
        result_path = args.output_dir / "result.json"
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
