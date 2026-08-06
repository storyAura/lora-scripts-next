"""Latent cache naming + NPZ key validation for multires expansion."""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

# Default matches Anima / MonadForge. Override via ``LatentCacheConvention``.
DEFAULT_LATENT_SUFFIX = "_anima.npz"


@dataclass(frozen=True)
class LatentCacheConvention:
    """Filename + NPZ key conventions for bucketed VAE caches."""

    suffix: str = DEFAULT_LATENT_SUFFIX
    # Inside NPZ: latents_{H/8}x{W/8}, original_size_*, crop_ltrb_*
    spatial_div: int = 8  # VAE downsample

    def filename(self, stem: str, width: int, height: int) -> str:
        return f"{stem}_{width:04d}x{height:04d}{self.suffix}"

    def spatial_suffix(self, width: int, height: int) -> str:
        return f"_{height // self.spatial_div}x{width // self.spatial_div}"

    def required_keys(self, width: int, height: int) -> set[str]:
        s = self.spatial_suffix(width, height)
        return {f"latents{s}", f"original_size{s}", f"crop_ltrb{s}"}

    @property
    def name_re(self) -> re.Pattern[str]:
        return re.compile(
            r"^(?P<stem>.+)_(?P<w>\d{3,5})x(?P<h>\d{3,5})"
            + re.escape(self.suffix)
            + r"$"
        )


DEFAULT_CONVENTION = LatentCacheConvention()


@dataclass(frozen=True)
class LatentCacheFile:
    path: Path
    stem: str
    width: int
    height: int


def parse_latent_cache_name(
    path: str | os.PathLike,
    convention: LatentCacheConvention = DEFAULT_CONVENTION,
) -> LatentCacheFile | None:
    p = Path(path)
    m = convention.name_re.match(p.name)
    if m is None:
        return None
    return LatentCacheFile(p, m.group("stem"), int(m.group("w")), int(m.group("h")))


def discover_latents_by_stem(
    cache_dir: str | os.PathLike,
    *,
    recursive: bool = True,
    convention: LatentCacheConvention = DEFAULT_CONVENTION,
) -> dict[str, list[LatentCacheFile]]:
    root = Path(cache_dir)
    pattern = "*" + convention.suffix
    paths: Iterable[Path] = root.rglob(pattern) if recursive else root.glob(pattern)
    out: dict[str, list[LatentCacheFile]] = {}
    for parsed in (parse_latent_cache_name(p, convention) for p in sorted(paths)):
        if parsed is not None:
            out.setdefault(parsed.stem, []).append(parsed)
    return out


def index_latents_in_dir(
    cache_root: str | os.PathLike,
    *,
    convention: LatentCacheConvention = DEFAULT_CONVENTION,
) -> dict[str, list[LatentCacheFile]]:
    """Non-recursive index of ``*_anima.npz`` under one cache root."""
    by_stem: dict[str, list[LatentCacheFile]] = {}
    for path in glob.glob(os.path.join(str(cache_root), "*" + convention.suffix)):
        parsed = parse_latent_cache_name(path, convention)
        if parsed is not None:
            by_stem.setdefault(parsed.stem, []).append(parsed)
    return by_stem


def validate_latent_npz(
    path: str | os.PathLike,
    width: int,
    height: int,
    convention: LatentCacheConvention = DEFAULT_CONVENTION,
) -> str | None:
    """Return an error string if NPZ is unusable, else ``None``."""
    required = convention.required_keys(width, height)
    try:
        with np.load(path, allow_pickle=False) as npz:
            missing = sorted(required.difference(npz.files))
    except Exception as exc:  # noqa: BLE001 — surface any load failure
        return f"{type(exc).__name__}: {exc}"
    if missing:
        return f"missing NPZ key(s) {missing}"
    return None


def write_stub_latent_npz(
    path: str | os.PathLike,
    width: int,
    height: int,
    *,
    channels: int = 2,
    convention: LatentCacheConvention = DEFAULT_CONVENTION,
) -> Path:
    """Write a minimal valid latent NPZ (for tests / dry pipelines)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    s = convention.spatial_suffix(width, height)
    np.savez(
        out,
        **{
            f"latents{s}": np.zeros(
                (channels, height // convention.spatial_div, width // convention.spatial_div),
                dtype=np.float32,
            ),
            f"original_size{s}": np.array([width, height]),
            f"crop_ltrb{s}": np.array([0, 0, width, height]),
        },
    )
    return out
