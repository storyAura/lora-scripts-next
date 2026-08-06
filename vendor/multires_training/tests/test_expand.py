from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from multires_training import (
    expand_dataset,
    expand_image_to_samples,
    freefit_band_for_edge,
    freefit_bucket,
    write_stub_latent_npz,
)


def _tier_size(edge: int, native: tuple[int, int] = (640, 640)) -> tuple[int, int]:
    """A (W,H) that lands inside the free-fit band for ``edge``."""
    return freefit_bucket(*native, freefit_band_for_edge(edge))


def _setup_caches(cache_dir: Path, stem: str, edges: list[int]) -> list[tuple[int, int]]:
    sizes = []
    for edge in edges:
        w, h = _tier_size(edge)
        write_stub_latent_npz(cache_dir / f"{stem}_{w:04d}x{h:04d}_anima.npz", w, h)
        sizes.append((w, h))
    return sizes


def test_one_image_expands_to_all_tiers(tmp_path: Path):
    image_dir = tmp_path / "resized"
    cache_dir = tmp_path / "cache"
    image_dir.mkdir()
    cache_dir.mkdir()
    img = image_dir / "sample.png"
    Image.new("RGB", (512, 512)).save(img)
    sizes = _setup_caches(cache_dir, "sample", [512, 1024])

    samples = expand_image_to_samples(
        str(img),
        target_res=[512, 1024],
        cache_dir=str(cache_dir),
        image_dir=str(image_dir),
    )
    assert len(samples) == 2
    assert {(s.width, s.height) for s in samples} == set(sizes)
    assert {s.source_path for s in samples} == {str(img)}
    assert all("::anima-multires=" in s.image_key for s in samples)


def test_missing_tier_fails(tmp_path: Path):
    image_dir = tmp_path / "resized"
    cache_dir = tmp_path / "cache"
    image_dir.mkdir()
    cache_dir.mkdir()
    img = image_dir / "sample.png"
    Image.new("RGB", (512, 512)).save(img)
    _setup_caches(cache_dir, "sample", [512])

    with pytest.raises(FileNotFoundError, match="missing VAE cache tier"):
        expand_image_to_samples(
            str(img),
            target_res=[512, 1024],
            cache_dir=str(cache_dir),
            image_dir=str(image_dir),
        )


def test_malformed_npz_fails(tmp_path: Path):
    image_dir = tmp_path / "resized"
    cache_dir = tmp_path / "cache"
    image_dir.mkdir()
    cache_dir.mkdir()
    img = image_dir / "sample.png"
    Image.new("RGB", (512, 512)).save(img)
    _setup_caches(cache_dir, "sample", [512])
    w, h = _tier_size(1024)
    np.savez(cache_dir / f"sample_{w:04d}x{h:04d}_anima.npz", unrelated=np.array([1]))

    with pytest.raises(ValueError, match="no usable VAE cache"):
        expand_image_to_samples(
            str(img),
            target_res=[512, 1024],
            cache_dir=str(cache_dir),
            image_dir=str(image_dir),
        )


def test_expand_dataset_multiplies_by_tiers(tmp_path: Path):
    image_dir = tmp_path / "resized"
    cache_dir = tmp_path / "cache"
    image_dir.mkdir()
    cache_dir.mkdir()
    paths = []
    for name in ("a", "b"):
        p = image_dir / f"{name}.png"
        Image.new("RGB", (640, 640)).save(p)
        paths.append(str(p))
        _setup_caches(cache_dir, name, [512, 768])

    samples = expand_dataset(
        paths,
        target_res=[512, 768],
        cache_dir=str(cache_dir),
        image_dir=str(image_dir),
    )
    assert len(samples) == 4
