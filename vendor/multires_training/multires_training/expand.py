"""Expand one physical image into one training sample per selected tier."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Sequence

from .cache import (
    DEFAULT_CONVENTION,
    LatentCacheConvention,
    LatentCacheFile,
    index_latents_in_dir,
    validate_latent_npz,
)
from .tiers import cache_matches_edge, validate_multires_target_res


@dataclass(frozen=True)
class MultiresSample:
    """One (image × tier) training row after expansion."""

    source_path: str
    image_key: str
    width: int
    height: int
    latents_npz: str
    edge: int
    stem: str
    num_repeats: int = 1
    caption: str | None = None


def resolve_cache_root_for_image(
    image_path: str,
    image_dir: str | None,
    cache_dir: str | None,
) -> str:
    """Mirror Anima subset layout: nest under ``cache_dir`` when present."""
    if cache_dir:
        if image_dir:
            try:
                rel_dir = os.path.relpath(os.path.dirname(image_path), image_dir)
            except ValueError:
                rel_dir = ""
            if rel_dir not in {"", "."}:
                return os.path.join(cache_dir, rel_dir)
        return cache_dir
    return os.path.dirname(image_path)


def select_tier_caches(
    candidates: Sequence[LatentCacheFile],
    target_res: Sequence[int],
    *,
    convention: LatentCacheConvention = DEFAULT_CONVENTION,
) -> list[tuple[int, LatentCacheFile]]:
    """Pick exactly one usable cache per ``target_res`` edge."""
    edges = validate_multires_target_res(target_res)
    selected: list[tuple[int, LatentCacheFile]] = []
    missing: list[int] = []
    invalid: dict[int, list[str]] = {}

    for edge in edges:
        matches = [
            item
            for item in candidates
            if cache_matches_edge(item.width, item.height, edge)
        ]
        if not matches:
            missing.append(edge)
            continue
        valid_matches: list[LatentCacheFile] = []
        errors: list[str] = []
        for item in matches:
            err = validate_latent_npz(
                item.path, item.width, item.height, convention
            )
            if err is None:
                valid_matches.append(item)
            else:
                errors.append(f"{item.path}: {err}")
        if not valid_matches:
            invalid[edge] = errors
            continue
        if len(valid_matches) > 1:
            paths = ", ".join(str(item.path) for item in valid_matches)
            raise ValueError(
                f"multiple usable VAE caches for tier {edge}: {paths}. "
                "Remove stale same-tier caches; do not pick by mtime."
            )
        selected.append((edge, valid_matches[0]))

    if missing:
        raise FileNotFoundError(
            f"missing VAE cache tier(s) {missing}; expected target_res={list(edges)}"
        )
    if invalid:
        details = "; ".join(
            f"tier {edge}: {', '.join(errs)}" for edge, errs in invalid.items()
        )
        raise ValueError(f"no usable VAE cache for tier(s): {details}")
    return selected


def expand_image_to_samples(
    image_path: str,
    *,
    target_res: Sequence[int],
    cache_dir: str | None = None,
    image_dir: str | None = None,
    num_repeats: int = 1,
    caption: str | None = None,
    convention: LatentCacheConvention = DEFAULT_CONVENTION,
    cache_index: dict[str, dict[str, list[LatentCacheFile]]] | None = None,
) -> list[MultiresSample]:
    """Expand one source image into one ``MultiresSample`` per selected tier.

    Call **after** train/val split on source paths so all tiers of one image
    stay on the same split side.
    """
    edges = validate_multires_target_res(target_res)
    root = resolve_cache_root_for_image(image_path, image_dir, cache_dir)
    stem = os.path.splitext(os.path.basename(image_path))[0]
    root_key = os.path.normcase(os.path.abspath(root))

    if cache_index is not None:
        by_stem = cache_index.get(root_key)
        if by_stem is None:
            by_stem = index_latents_in_dir(root, convention=convention)
            cache_index[root_key] = by_stem
    else:
        by_stem = index_latents_in_dir(root, convention=convention)

    candidates = by_stem.get(stem, [])
    if not candidates:
        raise FileNotFoundError(
            f"no VAE caches for {image_path!r} under {root!r}; "
            "run multi-resolution resize + VAE caching first"
        )

    selected = select_tier_caches(candidates, edges, convention=convention)
    samples: list[MultiresSample] = []
    for edge, cache in selected:
        samples.append(
            MultiresSample(
                source_path=image_path,
                image_key=f"{image_path}::anima-multires={cache.width}x{cache.height}",
                width=cache.width,
                height=cache.height,
                latents_npz=str(cache.path),
                edge=edge,
                stem=stem,
                num_repeats=num_repeats,
                caption=caption,
            )
        )
    return samples


def expand_dataset(
    image_paths: Iterable[str],
    *,
    target_res: Sequence[int],
    cache_dir: str | None = None,
    image_dir: str | None = None,
    num_repeats: int = 1,
    captions: dict[str, str] | None = None,
    convention: LatentCacheConvention = DEFAULT_CONVENTION,
) -> list[MultiresSample]:
    """Expand many source paths (already split) into an epoch sample list."""
    validate_multires_target_res(target_res)
    index: dict[str, dict[str, list[LatentCacheFile]]] = {}
    out: list[MultiresSample] = []
    for path in image_paths:
        cap = None if captions is None else captions.get(path)
        out.extend(
            expand_image_to_samples(
                path,
                target_res=target_res,
                cache_dir=cache_dir,
                image_dir=image_dir,
                num_repeats=num_repeats,
                caption=cap,
                convention=convention,
                cache_index=index,
            )
        )
    return out
