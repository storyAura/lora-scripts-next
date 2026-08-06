"""Same-epoch multi-resolution expansion (``multires_per_image``) for Anima.

Bridges the portable ``vendor/multires_training`` package into the sd-scripts
dataset: one source image becomes one training sample per selected free-fit
tier, each carrying its own ``bucket_reso``. Per-tier latents share a single
npz file per image because the Anima caching strategy keys latents by latent
resolution (``latents_{H/8}x{W/8}``) and ``save_latents_to_disk`` merges keys.

Pure Python — no torch import — so the planning logic stays unit-testable
without a training environment.
"""

from __future__ import annotations

import copy
import os
import sys
import zlib
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_LIBRARY_DIR = os.path.dirname(os.path.abspath(__file__))
_VENDOR_DIR = os.path.dirname(os.path.dirname(_LIBRARY_DIR))
VENDOR_PACKAGE_ROOT = os.path.join(_VENDOR_DIR, "multires_training")

IMAGE_KEY_MARKER = "::anima-multires="


class MultiresUnavailableError(RuntimeError):
    """The vendored multires_training package could not be imported."""


def ensure_importable() -> None:
    if os.path.isdir(VENDOR_PACKAGE_ROOT) and VENDOR_PACKAGE_ROOT not in sys.path:
        sys.path.insert(0, VENDOR_PACKAGE_ROOT)


def _tiers_module():
    ensure_importable()
    try:
        from multires_training import tiers
    except ImportError as error:  # pragma: no cover - packaging accident
        raise MultiresUnavailableError(
            f"multires_training is not importable from {VENDOR_PACKAGE_ROOT}"
        ) from error
    return tiers


def normalize_target_res(value: Any) -> List[int]:
    """Parse ``target_res`` from TOML/argparse (str, int or list) into ints."""
    if value is None:
        return []
    if isinstance(value, bool):
        raise ValueError("target_res expects resolutions, not a boolean")
    if isinstance(value, (list, tuple)):
        edges: List[int] = []
        for item in value:
            text = str(item).strip()
            if text:
                edges.append(_parse_edge(text))
        return edges
    text = str(value).strip()
    if not text:
        return []
    return [_parse_edge(part) for part in text.replace("，", ",").split(",") if part.strip()]


def _parse_edge(text: str) -> int:
    try:
        return int(str(text).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(f"target_res item {text!r} is not an integer") from error


def validate_target_res(value: Any) -> Tuple[int, ...]:
    """Validate tiers for ``multires_per_image`` (≥2 tiers, all allowed)."""
    return _tiers_module().validate_multires_target_res(normalize_target_res(value))


def cover_resized_size(
    image_size: Tuple[int, int], bucket_reso: Tuple[int, int]
) -> Tuple[int, int]:
    """Pre-crop resize target: scale so the image covers the bucket.

    Mirrors ``BucketManager.select_bucket``'s upscale branch, which
    ``trim_and_resize_if_required`` relies on (resize to ``resized_size`` then
    center-crop to ``bucket_reso``).
    """
    width, height = int(image_size[0]), int(image_size[1])
    bucket_width, bucket_height = int(bucket_reso[0]), int(bucket_reso[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image size {image_size}")
    scale = max(bucket_width / width, bucket_height / height)
    resized = (int(width * scale + 0.5), int(height * scale + 0.5))
    return (max(bucket_width, resized[0]), max(bucket_height, resized[1]))


def plan_image_tiers(
    image_size: Tuple[int, int], target_res: Sequence[int]
) -> List[Tuple[int, Tuple[int, int], Tuple[int, int]]]:
    """Return ``(edge, bucket_reso, resized_size)`` for every selected tier."""
    tiers = _tiers_module()
    edges = tiers.validate_multires_target_res(target_res)
    width, height = int(image_size[0]), int(image_size[1])
    plan: List[Tuple[int, Tuple[int, int], Tuple[int, int]]] = []
    for edge in edges:
        bucket = tiers.freefit_bucket(width, height, tiers.freefit_band_for_edge(edge))
        plan.append((edge, bucket, cover_resized_size((width, height), bucket)))
    return plan


def multires_image_key(image_key: str, bucket_reso: Tuple[int, int]) -> str:
    return f"{image_key}{IMAGE_KEY_MARKER}{bucket_reso[0]}x{bucket_reso[1]}"


def source_image_key(image_key: str) -> str:
    """Strip the tier marker, e.g. for grouping samples back per source image."""
    return image_key.split(IMAGE_KEY_MARKER, 1)[0]


def expand_image_data(
    image_data: Dict[str, Any],
    image_to_subset: Dict[str, Any],
    target_res: Sequence[int],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[float]]:
    """Expand every image info into one info per tier.

    Returns ``(image_data, image_to_subset, ar_errors)``. Each produced info
    keeps the original ``image_size`` (so the latent npz path stays shared) and
    gets its own ``bucket_reso`` / ``resized_size``.
    """
    edges = validate_target_res(target_res)

    expanded: Dict[str, Any] = {}
    expanded_subsets: Dict[str, Any] = {}
    ar_errors: List[float] = []

    for image_key, info in image_data.items():
        if getattr(info, "latents_npz", None):
            raise ValueError(
                "multires_per_image cannot use a metadata dataset with pre-baked "
                f"latents ({info.latents_npz}); per-tier latents must be cached by "
                "the trainer"
            )
        if info.image_size is None:
            raise ValueError(f"image size unknown for {image_key}; cannot plan tiers")

        subset = image_to_subset.get(image_key)
        image_ar = info.image_size[0] / info.image_size[1]
        for _edge, bucket, resized in plan_image_tiers(info.image_size, edges):
            tier_info = copy.copy(info)
            tier_info.image_key = multires_image_key(image_key, bucket)
            tier_info.bucket_reso = bucket
            tier_info.resized_size = resized
            expanded[tier_info.image_key] = tier_info
            expanded_subsets[tier_info.image_key] = subset
            ar_errors.append((bucket[0] / bucket[1]) - image_ar)

    return expanded, expanded_subsets, ar_errors


def shard_index(absolute_path: str, num_processes: int) -> int:
    """Deterministic process shard for one source image.

    All tiers of an image must be cached by the same process because they share
    one npz file, which is rewritten (read-modify-write) per tier.
    """
    if num_processes <= 1:
        return 0
    key = os.path.normcase(os.path.abspath(absolute_path)).encode("utf-8")
    return zlib.crc32(key) % num_processes


def bucket_resolutions(image_data: Dict[str, Any]) -> List[Tuple[int, int]]:
    return sorted({tuple(info.bucket_reso) for info in image_data.values()})


def derive_token_budget(
    resos: Iterable[Tuple[int, int]],
    sample_prompt_sizes: Optional[Iterable[Tuple[int, int]]] = None,
) -> Tuple[int, int, set]:
    """``(min_tokens, max_tokens, distinct_counts)`` from the real bucket shapes."""
    ensure_importable()
    try:
        from multires_training import budget
    except ImportError as error:  # pragma: no cover - packaging accident
        raise MultiresUnavailableError(
            f"multires_training is not importable from {VENDOR_PACKAGE_ROOT}"
        ) from error
    return budget.derive_token_budget(resos, sample_prompt_sizes=sample_prompt_sizes)
