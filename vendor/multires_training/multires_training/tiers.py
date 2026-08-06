"""Free-fit multi-scale tier math — pure, deterministic, numpy-free."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

EDGE_TOKEN_BANDS: dict[int, tuple[int, int]] = {
    512: (1008, 1024),
    768: (2160, 2160),
    896: (3000, 3024),
    1024: (4032, 4200),
    1280: (6300, 6300),
    1536: (8640, 8640),
}
ALLOWED_TARGET_RES: tuple[int, ...] = tuple(sorted(EDGE_TOKEN_BANDS))
DEFAULT_TARGET_RES: tuple[int, ...] = (1024,)

DEFAULT_FREEFIT_MAX_RATIO = 4.0
FREEFIT_BAND_TOLERANCE = 0.025
FREEFIT_FROZEN_EDGES: tuple[int, ...] = (1024,)
FREEFIT_BAND_VERSION = 2
PATCH = 16
ROPE_CAP = 256


def _band(edge: int) -> tuple[int, int]:
    try:
        return EDGE_TOKEN_BANDS[edge]
    except KeyError as exc:
        raise ValueError(
            f"target_res {edge} not in allowed tiers {ALLOWED_TARGET_RES}"
        ) from exc


def normalize_target_res(target_res: Iterable[int] | int | str | None) -> list[int]:
    """Normalize config-style ``target_res`` into a non-empty int list."""
    if target_res is None:
        return list(DEFAULT_TARGET_RES)
    if isinstance(target_res, int):
        return [int(target_res)]
    if isinstance(target_res, str):
        raw = target_res.strip()
        if not raw:
            return list(DEFAULT_TARGET_RES)
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    values = [int(v) for v in target_res]
    return values or list(DEFAULT_TARGET_RES)


def validate_multires_target_res(target_res: Sequence[int]) -> tuple[int, ...]:
    """Dedup + validate tiers for ``multires_per_image`` (requires ≥2)."""
    edges = tuple(dict.fromkeys(int(e) for e in target_res))
    unknown = [e for e in edges if e not in EDGE_TOKEN_BANDS]
    if unknown:
        raise ValueError(
            f"target_res {unknown} not in allowed tiers {list(ALLOWED_TARGET_RES)}"
        )
    if len(edges) < 2:
        raise ValueError(
            f"multires_per_image requires at least two target_res tiers; got {list(edges)}"
        )
    return edges


def token_count_families(target_res: Sequence[int]) -> int:
    counts: set[int] = set()
    for edge in target_res:
        lo, hi = _band(edge)
        counts.add(lo)
        counts.add(hi)
    return len(counts)


def token_count_range(target_res: Sequence[int]) -> tuple[int, int]:
    los = [_band(edge)[0] for edge in target_res]
    his = [_band(edge)[1] for edge in target_res]
    if not los:
        raise ValueError("token_count_range requires at least one tier")
    return min(los), max(his)


def token_counts_for_resos(resos: Iterable[tuple[int, int]]) -> set[int]:
    return {(w // PATCH) * (h // PATCH) for w, h in resos}


def patch_token_count(width: int, height: int, patch: int = PATCH) -> int:
    return (width // patch) * (height // patch)


def choose_edge(width: int, height: int, target_res: Sequence[int]) -> int:
    """Assign an image to the tier that resizes it the least (area log-distance)."""
    if len(target_res) == 1:
        return int(target_res[0])
    native_tokens = (width / float(PATCH)) * (height / float(PATCH))
    best_edge: int | None = None
    best_cost = float("inf")
    for edge in target_res:
        lo, hi = _band(int(edge))
        nominal = (lo + hi) / 2.0
        cost = abs(math.log(nominal / native_tokens))
        if cost < best_cost:
            best_cost, best_edge = cost, int(edge)
    assert best_edge is not None
    return best_edge


def freefit_band_for_edge(
    edge: int, tol: float = FREEFIT_BAND_TOLERANCE
) -> tuple[int, int]:
    lo, hi = token_count_range((edge,))
    if edge in FREEFIT_FROZEN_EDGES:
        return lo, hi
    return round(lo * (1.0 - tol)), round(hi * (1.0 + tol))


def freefit_bucket(
    width: int,
    height: int,
    band: tuple[int, int],
    max_ratio: float = DEFAULT_FREEFIT_MAX_RATIO,
    patch: int = PATCH,
    rope_cap: int = ROPE_CAP,
) -> tuple[int, int]:
    """Native-aspect resize target whose patch grid fills ``band``."""
    lo, hi = int(band[0]), int(band[1])
    if lo <= 0 or hi < lo:
        raise ValueError(f"invalid free-fit band {band}")
    a = width / height
    a_clamped = min(max(a, 1.0 / max_ratio), float(max_ratio))

    best: tuple | None = None
    hp_max = min(rope_cap, hi)
    for hp in range(1, hp_max + 1):
        wp_lo = max(1, -(-lo // hp))
        wp_hi = min(rope_cap, hi // hp)
        for wp in range(wp_lo, wp_hi + 1):
            aspect_err = abs(wp / hp - a_clamped)
            cover_scale = max(wp * patch / width, hp * patch / height)
            key = (aspect_err, abs(math.log(cover_scale)), hp, wp)
            if best is None or key < best:
                best = key
    if best is None:
        raise ValueError(
            f"free-fit band {band} admits no grid under rope_cap={rope_cap}"
        )
    _, _, hp, wp = best
    return wp * patch, hp * patch


def select_bucket(
    width: int,
    height: int,
    target_res: Sequence[int] | None = None,
    *,
    max_ratio: float = DEFAULT_FREEFIT_MAX_RATIO,
) -> tuple[int, tuple[int, int]]:
    """``choose_edge`` + ``freefit_bucket`` for one image / one active tier set."""
    tiers = normalize_target_res(target_res)
    edge = choose_edge(width, height, tiers)
    bucket = freefit_bucket(
        width, height, freefit_band_for_edge(edge), max_ratio=max_ratio
    )
    return edge, bucket


def cache_matches_edge(width: int, height: int, edge: int) -> bool:
    """True when ``(W,H)`` token count lies in the free-fit band for ``edge``."""
    lo, hi = freefit_band_for_edge(edge)
    tok = patch_token_count(width, height)
    return lo <= tok <= hi
