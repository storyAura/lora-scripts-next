"""GUI-side bridge to the vendored ``multires_training`` package.

``multires_per_image`` trains every source image once per selected free-fit
tier **inside the same epoch**. The tier math lives in
``vendor/multires_training`` so the GUI, the trainer and the package tests all
agree on one definition of the allowed tiers.

The heavy import (numpy / Pillow through the package ``__init__``) is deferred
until a config actually asks for multi-resolution training, so importing this
module stays cheap for every other code path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable

VENDOR_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "multires_training"

MULTIRES_FIELDS = ("multires_per_image", "target_res")

# Kept in sync with multires_training.tiers.EDGE_TOKEN_BANDS; used for error
# messages and schema copy without paying for the package import.
DOCUMENTED_ALLOWED_TARGET_RES = (512, 768, 896, 1024, 1280, 1536)


class MultiresUnavailableError(RuntimeError):
    """The vendored multires_training package could not be imported."""


def ensure_importable() -> None:
    path = str(VENDOR_ROOT)
    if VENDOR_ROOT.is_dir() and path not in sys.path:
        sys.path.insert(0, path)


def load_tiers():
    """Return the ``multires_training.tiers`` module."""
    ensure_importable()
    try:
        from multires_training import tiers
    except ImportError as error:  # pragma: no cover - packaging accident
        raise MultiresUnavailableError(
            f"multires_training is not importable from {VENDOR_ROOT}"
        ) from error
    return tiers


def allowed_target_res() -> tuple[int, ...]:
    return tuple(load_tiers().ALLOWED_TARGET_RES)


def is_multires_enabled(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def normalize_target_res(value: Any) -> list[int]:
    """Parse a UI ``target_res`` value (list, int or "512,768") into ints."""
    if isinstance(value, (list, tuple)):
        parts: list[int] = []
        for item in value:
            text = str(item).strip()
            if not text:
                continue
            parts.append(_parse_edge(text))
        return parts
    if isinstance(value, bool):
        raise ValueError("target_res expects resolutions, not a boolean")
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [_parse_edge(part) for part in text.replace("，", ",").split(",") if part.strip()]


def _parse_edge(text: str) -> int:
    try:
        return int(str(text).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(f"target_res item {text!r} is not an integer") from error


def validate_target_res(value: Any) -> tuple[int, ...]:
    """Validate tiers for ``multires_per_image`` (≥2 tiers, all allowed)."""
    tiers = load_tiers()
    return tiers.validate_multires_target_res(normalize_target_res(value))


def format_target_res(tiers: Iterable[int]) -> str:
    """Serialize tiers for the TOML (sd-scripts reads a comma-separated str)."""
    return ",".join(str(int(edge)) for edge in tiers)
