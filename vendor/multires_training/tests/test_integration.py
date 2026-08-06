"""End-to-end接入测试：resize staging → stub VAE → expand → epoch buckets."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from multires_training import (
    build_shape_buckets,
    derive_token_budget,
    expand_dataset,
    make_staging_plan,
    samples_to_bucket_items,
    stage_multires_images,
    write_stub_latent_npz,
)


def test_integration_multires_pipeline(tmp_path: Path):
    source = tmp_path / "src"
    resized = tmp_path / "resized"
    multires = tmp_path / "multires"
    cache = tmp_path / "lora"
    source.mkdir()
    Image.new("RGB", (640, 768), "white").save(source / "sample.png")
    (source / "sample.txt").write_text("a caption", encoding="utf-8")

    plan = make_staging_plan(
        source_dir=source,
        resized_dir=resized,
        cache_dir=cache,
        target_res=[512, 768],
        multires_per_image=True,
        multires_dir=multires,
    )
    assert [p.name for p in plan.vae_input_dirs] == ["512", "768"]
    assert plan.sidecar_image_dir == resized

    report = stage_multires_images(plan, overwrite=True)
    assert "sample" in report
    assert len(report["sample"]) == 2
    assert (resized / "sample.png").exists()
    assert (resized / "sample.txt").exists()
    for edge in (512, 768):
        assert (multires / str(edge) / "sample.png").exists()

    # Simulate VAE: one NPZ per staged tier (trainer would encode real latents).
    for edge, (w, h) in report["sample"]:
        write_stub_latent_npz(cache / f"sample_{w:04d}x{h:04d}_anima.npz", w, h)

    image_paths = [str(resized / "sample.png")]
    samples = expand_dataset(
        image_paths,
        target_res=[512, 768],
        cache_dir=str(cache),
        image_dir=str(resized),
    )
    assert len(samples) == 2
    assert {s.edge for s in samples} == {512, 768}
    assert all(Path(s.latents_npz).exists() for s in samples)

    # batch_size larger than each single-item shape bucket must not drop a tier.
    epoch = build_shape_buckets(
        samples_to_bucket_items(samples),
        batch_size=2,
        keep_incomplete_batches=True,
    )
    assert epoch.all_keys_in_epoch() == {s.image_key for s in samples}
    lo, hi, counts = derive_token_budget(epoch.resos)
    assert lo <= hi
    assert counts


def test_integration_split_before_expand_keeps_image_together(tmp_path: Path):
    """Train/val split is on source paths; expansion happens after."""
    from multires_training import freefit_band_for_edge, freefit_bucket

    image_dir = tmp_path / "resized"
    cache = tmp_path / "cache"
    image_dir.mkdir()
    cache.mkdir()

    paths = []
    for name in ("keep", "holdout"):
        p = image_dir / f"{name}.png"
        Image.new("RGB", (512, 512)).save(p)
        paths.append(str(p))
        for edge in (512, 768):
            w, h = freefit_bucket(512, 512, freefit_band_for_edge(edge))
            write_stub_latent_npz(cache / f"{name}_{w:04d}x{h:04d}_anima.npz", w, h)

    train_paths = [paths[0]]  # only "keep"
    samples = expand_dataset(
        train_paths,
        target_res=[512, 768],
        cache_dir=str(cache),
        image_dir=str(image_dir),
    )
    assert len(samples) == 2
    assert all(s.stem == "keep" for s in samples)
    assert all("holdout" not in s.source_path for s in samples)


def test_integration_fails_fast_on_missing_tier_after_staging(tmp_path: Path):
    source = tmp_path / "src"
    resized = tmp_path / "resized"
    multires = tmp_path / "multires"
    cache = tmp_path / "lora"
    source.mkdir()
    Image.new("RGB", (640, 640)).save(source / "x.png")

    plan = make_staging_plan(
        source_dir=source,
        resized_dir=resized,
        cache_dir=cache,
        target_res=[512, 768],
        multires_per_image=True,
        multires_dir=multires,
    )
    report = stage_multires_images(plan)
    # Only write one tier's cache on purpose.
    edge, (w, h) = report["x"][0]
    write_stub_latent_npz(cache / f"x_{w:04d}x{h:04d}_anima.npz", w, h)

    with pytest.raises(FileNotFoundError, match="missing VAE cache tier"):
        expand_dataset(
            [str(resized / "x.png")],
            target_res=[512, 768],
            cache_dir=str(cache),
            image_dir=str(resized),
        )
