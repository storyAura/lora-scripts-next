from __future__ import annotations

from pathlib import Path

import numpy as np

from multires_training import (
    parse_latent_cache_name,
    validate_latent_npz,
    write_stub_latent_npz,
)


def test_parse_latent_cache_name(tmp_path: Path):
    path = tmp_path / "cat_0512x0768_anima.npz"
    path.write_bytes(b"")
    parsed = parse_latent_cache_name(path)
    assert parsed is not None
    assert parsed.stem == "cat"
    assert parsed.width == 512
    assert parsed.height == 768
    assert parse_latent_cache_name(tmp_path / "cat_anima_te.safetensors") is None


def test_validate_and_stub(tmp_path: Path):
    path = write_stub_latent_npz(tmp_path / "a_0512x0512_anima.npz", 512, 512)
    assert validate_latent_npz(path, 512, 512) is None

    bad = tmp_path / "b_0512x0512_anima.npz"
    np.savez(bad, unrelated=np.array([1]))
    err = validate_latent_npz(bad, 512, 512)
    assert err is not None
    assert "missing NPZ key" in err
