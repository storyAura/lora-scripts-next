"""Preprocess staging plan + minimal free-fit resize for multires tiers."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageOps

from .tiers import (
    DEFAULT_FREEFIT_MAX_RATIO,
    DEFAULT_TARGET_RES,
    choose_edge,
    freefit_band_for_edge,
    freefit_bucket,
    normalize_target_res,
    validate_multires_target_res,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(frozen=True)
class StagingPlan:
    """Where resize / VAE should read and write for one preprocess run."""

    source_dir: Path
    resized_dir: Path
    multires_dir: Path
    cache_dir: Path
    target_res: tuple[int, ...]
    multires_per_image: bool

    @property
    def vae_input_dirs(self) -> list[Path]:
        """Directories the VAE cache step should scan."""
        if self.multires_per_image:
            return [self.multires_dir / str(edge) for edge in self.target_res]
        return [self.resized_dir]

    @property
    def sidecar_image_dir(self) -> Path:
        """Nearest-tier tree used by TE / PE / mask (always ``resized_dir``)."""
        return self.resized_dir


def make_staging_plan(
    *,
    source_dir: str | Path,
    resized_dir: str | Path,
    cache_dir: str | Path,
    target_res: Sequence[int] | None = None,
    multires_per_image: bool = False,
    multires_dir: str | Path | None = None,
) -> StagingPlan:
    tiers = tuple(normalize_target_res(target_res))
    if multires_per_image:
        tiers = validate_multires_target_res(tiers)
    resized = Path(resized_dir)
    multi = Path(multires_dir) if multires_dir else resized.parent / "multires"
    return StagingPlan(
        source_dir=Path(source_dir),
        resized_dir=resized,
        multires_dir=multi,
        cache_dir=Path(cache_dir),
        target_res=tiers,
        multires_per_image=bool(multires_per_image),
    )


def _cover_crop(img: Image.Image, bucket: tuple[int, int]) -> Image.Image:
    bw, bh = bucket
    w, h = img.size
    ar_img = w / h
    ar_bucket = bw / bh
    if ar_img > ar_bucket:
        new_h = bh
        new_w = round(bh * ar_img)
    else:
        new_w = bw
        new_h = round(bw / ar_img)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (new_w - bw) // 2
    top = (new_h - bh) // 2
    return img.crop((left, top, left + bw, top + bh))


def resize_one(
    src: Path,
    dst: Path,
    *,
    target_res: Sequence[int],
    max_ratio: float = DEFAULT_FREEFIT_MAX_RATIO,
    copy_caption: bool = True,
    overwrite: bool = False,
) -> tuple[int, int]:
    """Resize one image into ``dst`` at free-fit bucket for ``target_res``."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as raw:
        img = ImageOps.exif_transpose(raw).convert("RGB")
        w, h = img.size
        edge = choose_edge(w, h, list(target_res) or list(DEFAULT_TARGET_RES))
        bucket = freefit_bucket(w, h, freefit_band_for_edge(edge), max_ratio=max_ratio)
        if not overwrite and dst.exists():
            with Image.open(dst) as existing:
                if existing.size == bucket:
                    return bucket
        out = _cover_crop(img, bucket)
        out.save(dst, format="PNG")
    if copy_caption:
        for ext in (".txt", ".caption"):
            cap = src.with_suffix(ext)
            if cap.exists():
                shutil.copy2(cap, dst.with_suffix(ext))
    return bucket


def stage_multires_images(
    plan: StagingPlan,
    *,
    max_ratio: float = DEFAULT_FREEFIT_MAX_RATIO,
    overwrite: bool = False,
) -> dict[str, list[tuple[int, tuple[int, int]]]]:
    """Write nearest-tier ``resized/`` plus optional ``multires/<edge>/``.

    Returns ``{stem: [(edge, (W,H)), ...]}`` for staged tiers (multires mode
    lists every selected edge; nearest-only lists the single chosen edge).
    """
    if not plan.source_dir.is_dir():
        raise FileNotFoundError(f"source_dir not found: {plan.source_dir}")

    sources = sorted(
        p
        for p in plan.source_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    report: dict[str, list[tuple[int, tuple[int, int]]]] = {}

    for src in sources:
        rel = src.relative_to(plan.source_dir)
        with Image.open(src) as raw:
            src_w, src_h = ImageOps.exif_transpose(raw).size

        # Nearest-tier primary (sidecar path for TE / PE / mask / captions)
        nearest_edge = choose_edge(src_w, src_h, list(plan.target_res))
        primary_dst = (plan.resized_dir / rel).with_suffix(".png")
        bucket = resize_one(
            src,
            primary_dst,
            target_res=list(plan.target_res),
            max_ratio=max_ratio,
            copy_caption=True,
            overwrite=overwrite,
        )
        entries = [(nearest_edge, bucket)]

        if plan.multires_per_image:
            entries = []
            for edge in plan.target_res:
                tier_dst = (plan.multires_dir / str(edge) / rel).with_suffix(".png")
                tier_bucket = resize_one(
                    src,
                    tier_dst,
                    target_res=[edge],
                    max_ratio=max_ratio,
                    copy_caption=False,
                    overwrite=overwrite,
                )
                entries.append((edge, tier_bucket))

        report[src.stem] = entries
    return report
