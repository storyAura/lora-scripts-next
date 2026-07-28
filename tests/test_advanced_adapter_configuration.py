from __future__ import annotations

from pathlib import Path

import pytest

from mikazuki.training_validation import (
    TrainingConfigurationError,
    validate_training_configuration,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("lora_type", "extra"),
    (
        ("delora", {"delora_lambda": 15}),
        (
            "waveft",
            {
                "waveft_n_frequency": 128,
                "waveft_scaling": 25,
                "waveft_random_loc_seed": 777,
                "waveft_wavelet_family": "db1",
            },
        ),
        (
            "deft",
            {
                "deft_decomposition_method": "qr",
                "deft_alpha": 0,
                "deft_init_scale": 1,
                "deft_init_weights": True,
            },
        ),
        ("moslora", {"moslora_mixer_init": "kaiming"}),
        (
            "tlora",
            {
                "network_dim": 16,
                "tlora_min_rank": 4,
                "tlora_rank_schedule": "cosine",
                "network_train_unet_only": True,
            },
        ),
    ),
)
def test_new_adapter_valid_configuration_is_accepted(
    lora_type: str,
    extra: dict[str, object],
) -> None:
    validate_training_configuration(
        {
            "lora_type": lora_type,
            "base_model_quantization": "none",
            **extra,
        },
        "anima-lora",
    )


@pytest.mark.parametrize(
    ("config", "field"),
    (
        ({"lora_type": "delora", "delora_lambda": 0}, "delora_lambda"),
        (
            {"lora_type": "waveft", "waveft_n_frequency": 0},
            "waveft_n_frequency",
        ),
        (
            {"lora_type": "waveft", "waveft_random_loc_seed": -1},
            "waveft_random_loc_seed",
        ),
        (
            {"lora_type": "waveft", "waveft_wavelet_family": "sym2"},
            "waveft_wavelet_family",
        ),
        (
            {"lora_type": "deft", "deft_decomposition_method": "svd"},
            "deft_decomposition_method",
        ),
        ({"lora_type": "deft", "deft_alpha": -1}, "deft_alpha"),
        (
            {"lora_type": "moslora", "moslora_mixer_init": "zeros"},
            "moslora_mixer_init",
        ),
        (
            {
                "lora_type": "tlora",
                "network_dim": 8,
                "tlora_min_rank": 9,
            },
            "tlora_min_rank",
        ),
        (
            {
                "lora_type": "tlora",
                "network_dim": 8,
                "tlora_min_rank": 2,
                "network_train_text_encoder_only": True,
            },
            "network_train_text_encoder_only",
        ),
        (
            {
                "lora_type": "moslora",
                "scale_weight_norms": 1,
            },
            "scale_weight_norms",
        ),
    ),
)
def test_new_adapter_invalid_configuration_fails_early(
    config: dict[str, object],
    field: str,
) -> None:
    with pytest.raises(TrainingConfigurationError) as error:
        validate_training_configuration(config, "anima-lora")
    assert error.value.field == field


@pytest.mark.parametrize(
    "lora_type",
    ("delora", "waveft", "deft", "moslora", "tlora"),
)
def test_new_adapters_reject_unverified_frozen_base_quantization(
    lora_type: str,
) -> None:
    with pytest.raises(
        TrainingConfigurationError,
        match="quantized-weight support",
    ):
        validate_training_configuration(
            {
                "lora_type": lora_type,
                "base_model_quantization": "nf4",
                "network_dim": 16,
                "tlora_min_rank": 4,
            },
            "anima-lora",
        )


def test_anima_schema_offers_all_new_adapter_types_and_fields() -> None:
    schema = (
        PROJECT_ROOT / "mikazuki" / "schema" / "sd3-lora.ts"
    ).read_text(encoding="utf-8")
    for adapter_type in (
        "delora",
        "waveft",
        "deft",
        "moslora",
        "tlora",
    ):
        assert f'Schema.const("{adapter_type}")' in schema
    for field in (
        "delora_lambda",
        "waveft_n_frequency",
        "waveft_use_idwt",
        "deft_decomposition_method",
        "deft_alpha",
        "deft_init_weights",
        "moslora_mixer_init",
        "tlora_min_rank",
    ):
        assert field in schema
