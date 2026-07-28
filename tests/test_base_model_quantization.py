from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SD_SCRIPTS_ROOT = REPO_ROOT / "vendor" / "sd-scripts"
sys.path.insert(0, str(SD_SCRIPTS_ROOT))

from library.base_model_quantization import (  # noqa: E402
    BaseModelQuantizationError,
    normalize_skip_module_patterns,
    quantize_frozen_linear_layers,
    validate_base_model_quantization_runtime,
    validate_base_model_quantization_training_request,
)
from networks.lora_anima import LoRAModule, LoRANetwork  # noqa: E402


class Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(32, 24, bias=True, dtype=torch.bfloat16)
        self.norm = torch.nn.LayerNorm(24, dtype=torch.bfloat16)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(value))


class TinyAnima(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block(), Block()])
        self.final_layer = torch.nn.Linear(24, 4, bias=False, dtype=torch.bfloat16)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = value
        for block in self.blocks:
            hidden = block(hidden)
        return self.final_layer(hidden)


def _frozen_model() -> TinyAnima:
    model = TinyAnima()
    model.requires_grad_(False)
    return model


def test_runtime_validation_rejects_invalid_or_unavailable_requests() -> None:
    with pytest.raises(BaseModelQuantizationError, match="unsupported base model quantization mode"):
        validate_base_model_quantization_runtime("fp4", "bf16", True, "0.46.0")

    with pytest.raises(BaseModelQuantizationError, match="requires CUDA"):
        validate_base_model_quantization_runtime("nf4", "bf16", False, "0.46.0")

    with pytest.raises(BaseModelQuantizationError, match="bitsandbytes>=0.46.0"):
        validate_base_model_quantization_runtime("int8", "bf16", True, "0.45.5")

    with pytest.raises(BaseModelQuantizationError, match="compute dtype"):
        validate_base_model_quantization_runtime("nf4", "float32", True, "0.46.0")


def test_training_request_validation_rejects_unsafe_combinations() -> None:
    with pytest.raises(BaseModelQuantizationError, match="cannot be combined with fp8_base"):
        validate_base_model_quantization_training_request(
            "int8",
            "bf16",
            "anima",
            "networks.lora_anima",
            True,
            False,
            None,
        )

    with pytest.raises(BaseModelQuantizationError, match="blocks_to_swap"):
        validate_base_model_quantization_training_request(
            "nf4",
            "bf16",
            "flux",
            "networks.lora_flux",
            False,
            False,
            4,
        )

    with pytest.raises(BaseModelQuantizationError, match="PiSSA, DoRA, LyCORIS"):
        validate_base_model_quantization_training_request(
            "int8",
            "bf16",
            "anima",
            "lycoris.kohya",
            False,
            False,
            None,
        )


def test_skip_pattern_normalization_is_strict_and_stable() -> None:
    assert normalize_skip_module_patterns([" blocks.0.* ", "blocks.0.*", "blocks.1.*"]) == (
        "blocks.0.*",
        "blocks.1.*",
    )
    with pytest.raises(BaseModelQuantizationError, match="list of glob strings"):
        normalize_skip_module_patterns("blocks.0.*")
    with pytest.raises(BaseModelQuantizationError, match="non-empty glob strings"):
        normalize_skip_module_patterns([""])


def test_quantization_converts_only_included_frozen_linear_layers() -> None:
    import bitsandbytes as bnb

    model = _frozen_model()
    report = quantize_frozen_linear_layers(
        model,
        "int8",
        torch.bfloat16,
        ("blocks.*",),
        ("blocks.1.*",),
    )

    assert report.mode == "int8"
    assert report.converted_modules == ("blocks.0.proj",)
    assert report.skipped_modules == ("blocks.1.proj",)
    assert isinstance(model.blocks[0].proj, bnb.nn.Linear8bitLt)
    assert isinstance(model.blocks[1].proj, torch.nn.Linear)
    assert not isinstance(model.blocks[1].proj, bnb.nn.Linear8bitLt)
    assert isinstance(model.final_layer, torch.nn.Linear)
    assert model.blocks[0].proj.weight.requires_grad is False


def test_quantization_rejects_trainable_base_weight_and_zero_matches() -> None:
    model = TinyAnima()
    with pytest.raises(BaseModelQuantizationError, match="must be frozen"):
        quantize_frozen_linear_layers(
            model,
            "int8",
            torch.bfloat16,
            ("blocks.*",),
            (),
        )

    frozen_model = _frozen_model()
    with pytest.raises(BaseModelQuantizationError, match="matched no frozen Linear modules"):
        quantize_frozen_linear_layers(
            frozen_model,
            "nf4",
            torch.bfloat16,
            ("missing.*",),
            (),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for bitsandbytes quantization")
@pytest.mark.parametrize(
    ("mode", "expected_class_name"),
    (("int8", "Linear8bitLt"), ("nf4", "Linear4bit")),
)
def test_quantized_anima_linear_is_discovered_and_trains_lora(
    mode: str,
    expected_class_name: str,
) -> None:
    model = _frozen_model()
    quantize_frozen_linear_layers(
        model,
        mode,
        torch.bfloat16,
        ("blocks.*",),
        (),
    )
    assert model.blocks[0].proj.__class__.__name__ == expected_class_name

    network = LoRANetwork(
        [],
        model,
        1.0,
        8,
        8.0,
        None,
        None,
        None,
        module_class=LoRAModule,
        modules_dim=None,
        modules_alpha=None,
        train_llm_adapter=False,
        exclude_patterns=None,
        include_patterns=None,
        reg_dims=None,
        reg_lrs=None,
        verbose=False,
    )
    assert len(network.unet_loras) == 2
    network.apply_to([], model, False, True)

    model = model.cuda()
    network = network.cuda().to(torch.bfloat16)
    value = torch.randn(3, 32, device="cuda", dtype=torch.bfloat16)
    output = model.blocks[0](value)
    output.float().square().mean().backward()

    gradients = [parameter.grad for parameter in network.parameters() if parameter.requires_grad]
    assert gradients
    assert any(gradient is not None and torch.count_nonzero(gradient).item() > 0 for gradient in gradients)
    assert all(gradient is None or torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for bitsandbytes quantization")
@pytest.mark.parametrize("mode", ("int8", "nf4"))
@pytest.mark.parametrize(
    ("module_name", "network_args"),
    (
        ("lora_fa_anima", {}),
        (
            "vera_anima",
            {
                "vera_projection_seed": 42,
                "vera_save_projection": True,
                "vera_d_initial": 0.1,
            },
        ),
    ),
)
def test_declared_quantized_anima_adapters_train_on_real_bnb_linear(
    mode: str,
    module_name: str,
    network_args: dict[str, object],
) -> None:
    model = _frozen_model()
    quantize_frozen_linear_layers(
        model,
        mode,
        torch.bfloat16,
        ("blocks.*",),
        (),
    )
    module = importlib.import_module(f"networks.{module_name}")
    network = module.create_network(
        1.0,
        8,
        8.0,
        None,
        [],
        model,
        None,
        **network_args,
    )
    network.apply_to([], model, False, True)
    model = model.cuda()
    network = network.cuda().to(torch.bfloat16)

    value = torch.randn(
        3,
        32,
        device="cuda",
        dtype=torch.bfloat16,
    )
    output = model.blocks[0](value)
    output.float().square().mean().backward()
    gradients = tuple(
        parameter.grad
        for parameter in network.parameters()
        if parameter.requires_grad
    )
    assert gradients
    assert any(
        gradient is not None
        and torch.count_nonzero(gradient).item() > 0
        for gradient in gradients
    )
    assert all(
        gradient is None or torch.isfinite(gradient).all()
        for gradient in gradients
    )
