from __future__ import annotations

import pytest

from mikazuki.anima_backend.adapter import adapt_anima_config
from mikazuki.training_validation import TrainingConfigurationError, validate_training_configuration


def test_quantized_base_configuration_is_forwarded_to_sd_scripts() -> None:
    adapted, warnings = adapt_anima_config(
        {
            "lora_type": "lora",
            "base_model_quantization": "nf4",
            "base_model_quantization_compute_dtype": "bf16",
            "base_model_quantization_skip_modules": ["blocks.0.*"],
            "quantize_text_encoder": True,
        }
    )

    assert warnings == []
    assert adapted["base_model_quantization"] == "nf4"
    assert adapted["base_model_quantization_compute_dtype"] == "bf16"
    assert adapted["base_model_quantization_skip_modules"] == ["blocks.0.*"]
    assert adapted["quantize_text_encoder"] is True


@pytest.mark.parametrize(
    "lora_type",
    ("rslora", "dora", "lokr", "tlora"),
)
def test_quantized_base_rejects_algorithms_without_quantized_weight_contract(
    lora_type: str,
) -> None:
    with pytest.raises(TrainingConfigurationError, match="quantized-weight support"):
        adapt_anima_config(
            {
                "lora_type": lora_type,
                "base_model_quantization": "int8",
                "base_model_quantization_compute_dtype": "bf16",
            }
        )


def test_quantized_base_rejects_pissa_fp8_and_block_swap() -> None:
    invalid_configs = (
        {
            "lora_type": "lora",
            "pissa_init": True,
            "base_model_quantization": "int8",
        },
        {
            "lora_type": "lora",
            "fp8_base": True,
            "base_model_quantization": "nf4",
        },
        {
            "lora_type": "lora",
            "blocks_to_swap": 4,
            "base_model_quantization": "int8",
        },
    )

    for config in invalid_configs:
        with pytest.raises(TrainingConfigurationError):
            validate_training_configuration(config, "anima-lora")


def test_text_encoder_quantization_requires_a_quantized_base_mode() -> None:
    with pytest.raises(TrainingConfigurationError, match="requires base_model_quantization"):
        validate_training_configuration(
            {
                "lora_type": "lora",
                "base_model_quantization": "none",
                "quantize_text_encoder": True,
            },
            "anima-lora",
        )
