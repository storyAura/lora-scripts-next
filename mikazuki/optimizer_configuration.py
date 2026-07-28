from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import math

from mikazuki.utils.config_args import normalize_kv_arg_list


ADEMAMIX_ALIASES: dict[str, str] = {
    "ademamix8bit": "bitsandbytes.optim.AdEMAMix8bit",
    "pagedademamix8bit": "bitsandbytes.optim.PagedAdEMAMix8bit",
}
FUSED_ADAMW_ALIAS = "adamwfused"
ADEMAMIX_FIELDS = frozenset(
    {
        "ademamix_beta1",
        "ademamix_beta2",
        "ademamix_beta3",
        "ademamix_alpha",
        "ademamix_t_alpha",
        "ademamix_t_beta3",
    }
)


class OptimizerConfigurationError(ValueError):
    def __init__(self, field: str, value: object, reason: str) -> None:
        self.field = field
        self.value = value
        self.reason = reason
        super().__init__(f"{field}={value!r} is invalid: {reason}")


@dataclass(frozen=True)
class OptimizerConfiguration:
    values: dict[str, object]
    warnings: tuple[str, ...]


def _optimizer_arg_keys(values: object) -> frozenset[str]:
    if values is None:
        return frozenset()
    if not isinstance(values, list):
        raise OptimizerConfigurationError(
            "optimizer_args",
            values,
            "optimizer_args must be a list of key=value strings",
        )
    keys: set[str] = set()
    for item in values:
        if not isinstance(item, str) or "=" not in item:
            raise OptimizerConfigurationError(
                "optimizer_args",
                item,
                "every optimizer argument must use key=value syntax",
            )
        key, _value = item.split("=", 1)
        normalized_key = key.strip()
        if not normalized_key:
            raise OptimizerConfigurationError(
                "optimizer_args",
                item,
                "optimizer argument keys must be non-empty",
            )
        keys.add(normalized_key)
    return frozenset(keys)


def _finite_float(value: object, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise OptimizerConfigurationError(
            field_name,
            value,
            "value must be numeric",
        ) from error
    if not math.isfinite(parsed):
        raise OptimizerConfigurationError(
            field_name,
            value,
            "value must be finite",
        )
    return parsed


def _beta(value: object, field_name: str) -> float:
    parsed = _finite_float(value, field_name)
    if not 0 <= parsed < 1:
        raise OptimizerConfigurationError(
            field_name,
            value,
            "AdEMAMix beta values must be in [0, 1)",
        )
    return parsed


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise OptimizerConfigurationError(
            field_name,
            value,
            "value must be a positive integer",
        )
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise OptimizerConfigurationError(
            field_name,
            value,
            "value must be a positive integer",
        ) from error
    if parsed <= 0 or str(parsed) != str(value).strip():
        raise OptimizerConfigurationError(
            field_name,
            value,
            "value must be a positive integer",
        )
    return parsed


def _add_unique_argument(
    arguments: list[str],
    existing_keys: frozenset[str],
    key: str,
    value: object,
    source_field: str,
) -> None:
    if key in existing_keys:
        raise OptimizerConfigurationError(
            source_field,
            value,
            f"{key} has two sources; remove it from optimizer_args or clear "
            f"{source_field}",
        )
    arguments.append(f"{key}={value}")


def _normalize_ademamix(
    values: dict[str, object],
    arguments: list[str],
    existing_keys: frozenset[str],
) -> None:
    beta_fields = (
        "ademamix_beta1",
        "ademamix_beta2",
        "ademamix_beta3",
    )
    present_betas = tuple(
        field_name
        for field_name in beta_fields
        if values.get(field_name) not in (None, "")
    )
    if present_betas and len(present_betas) != len(beta_fields):
        raise OptimizerConfigurationError(
            "ademamix_betas",
            present_betas,
            "all of ademamix_beta1, ademamix_beta2, and ademamix_beta3 "
            "must be provided together",
        )
    if present_betas:
        betas = tuple(
            _beta(values[field_name], field_name)
            for field_name in beta_fields
        )
        _add_unique_argument(
            arguments,
            existing_keys,
            "betas",
            repr(betas),
            "ademamix_beta1",
        )

    alpha_value = values.get("ademamix_alpha")
    if alpha_value not in (None, ""):
        alpha = _finite_float(alpha_value, "ademamix_alpha")
        if alpha < 0:
            raise OptimizerConfigurationError(
                "ademamix_alpha",
                alpha_value,
                "AdEMAMix alpha must be non-negative",
            )
        _add_unique_argument(
            arguments,
            existing_keys,
            "alpha",
            alpha,
            "ademamix_alpha",
        )

    for field_name, argument_name in (
        ("ademamix_t_alpha", "t_alpha"),
        ("ademamix_t_beta3", "t_beta3"),
    ):
        raw_value = values.get(field_name)
        if raw_value in (None, ""):
            continue
        parsed = _positive_int(raw_value, field_name)
        _add_unique_argument(
            arguments,
            existing_keys,
            argument_name,
            parsed,
            field_name,
        )


def normalize_optimizer_configuration(
    config: Mapping[str, object],
) -> OptimizerConfiguration:
    values = deepcopy(dict(config))
    raw_type = str(values.get("optimizer_type") or "").strip()
    normalized_type = raw_type.lower()
    arguments = normalize_kv_arg_list(values.get("optimizer_args"))
    existing_keys = _optimizer_arg_keys(arguments)

    is_ademamix = normalized_type in ADEMAMIX_ALIASES
    if is_ademamix:
        values["optimizer_type"] = ADEMAMIX_ALIASES[normalized_type]
        _normalize_ademamix(values, arguments, existing_keys)
    elif any(values.get(field_name) not in (None, "") for field_name in ADEMAMIX_FIELDS):
        present = [
            field_name
            for field_name in ADEMAMIX_FIELDS
            if values.get(field_name) not in (None, "")
        ]
        raise OptimizerConfigurationError(
            present[0],
            values[present[0]],
            "AdEMAMix structured fields require optimizer_type "
            "AdEMAMix8bit or PagedAdEMAMix8bit",
        )

    if normalized_type == FUSED_ADAMW_ALIAS:
        values["optimizer_type"] = "AdamW"
        _add_unique_argument(
            arguments,
            existing_keys,
            "fused",
            True,
            "optimizer_type",
        )
    elif "fused" in existing_keys and normalized_type != "adamw":
        raise OptimizerConfigurationError(
            "optimizer_args",
            values.get("optimizer_args"),
            "fused is supported only by the AdamW optimizer",
        )

    lora_type = str(values.get("lora_type") or "").strip().lower()
    effective_type = str(values.get("optimizer_type") or raw_type).strip().lower()
    if lora_type == "lora_plus" and (
        effective_type.startswith("dadapt")
        or "prodigy" in effective_type
    ):
        raise OptimizerConfigurationError(
            "optimizer_type",
            values.get("optimizer_type"),
            "LoRA+ requires independent parameter-group learning rates and "
            "is incompatible with D-Adaptation and Prodigy optimizers",
        )

    for field_name in ADEMAMIX_FIELDS:
        values.pop(field_name, None)
    if arguments:
        values["optimizer_args"] = arguments
    else:
        values.pop("optimizer_args", None)
    return OptimizerConfiguration(values, ())
