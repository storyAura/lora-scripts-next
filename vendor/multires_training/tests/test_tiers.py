from __future__ import annotations

import pytest

from multires_training import (
    ALLOWED_TARGET_RES,
    EDGE_TOKEN_BANDS,
    choose_edge,
    freefit_band_for_edge,
    freefit_bucket,
    patch_token_count,
    token_count_families,
    token_count_range,
    validate_multires_target_res,
)


def test_band_family_counts():
    expected = {512: 2, 768: 1, 896: 2, 1024: 2, 1280: 1, 1536: 1}
    for edge, (lo, hi) in EDGE_TOKEN_BANDS.items():
        n = 1 if lo == hi else 2
        assert n == expected[edge]


def test_1024_band_frozen():
    assert EDGE_TOKEN_BANDS[1024] == (4032, 4200)
    assert freefit_band_for_edge(1024) == (4032, 4200)


def test_token_count_helpers():
    assert token_count_families([1024]) == 2
    assert token_count_range([768, 1280]) == (2160, 6300)
    assert set(ALLOWED_TARGET_RES) == set(EDGE_TOKEN_BANDS)


@pytest.mark.parametrize(
    "w,h,target_res,expected",
    [
        (1500, 1500, [512, 768, 1024, 1280, 1536], 1536),
        (1024, 1024, [768, 1024, 1536], 1024),
        (1000, 950, [768, 1024], 1024),
        (800, 800, [512, 768, 1024], 768),
    ],
)
def test_choose_edge(w, h, target_res, expected):
    assert choose_edge(w, h, target_res) == expected


def test_freefit_bucket_lands_in_band():
    w, h = 1920, 1080
    for edge in (512, 768, 1024):
        band = freefit_band_for_edge(edge)
        bw, bh = freefit_bucket(w, h, band)
        tok = patch_token_count(bw, bh)
        assert band[0] <= tok <= band[1]
        assert bw % 16 == 0 and bh % 16 == 0


def test_validate_multires_requires_two_tiers():
    with pytest.raises(ValueError, match="at least two"):
        validate_multires_target_res([1024])
    assert validate_multires_target_res([512, 1024, 512]) == (512, 1024)
