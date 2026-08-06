"""Compile / dynamic-seq token budget helpers from populated shapes."""

from __future__ import annotations

from typing import Iterable, Sequence

from .tiers import (
    PATCH,
    token_count_families,
    token_count_range,
    token_counts_for_resos,
)


def derive_token_budget(
    resos: Iterable[tuple[int, int]],
    *,
    sample_prompt_sizes: Iterable[tuple[int, int]] | None = None,
) -> tuple[int, int, set[int]]:
    """Return ``(min_tokens, max_tokens, distinct_counts)`` from real shapes.

    Prefer this over trusting ``target_res`` alone — on-disk caches are truth.
    """
    all_resos = list(resos)
    if sample_prompt_sizes:
        all_resos.extend(
            (max(64, w - w % PATCH), max(64, h - h % PATCH))
            for w, h in sample_prompt_sizes
        )
    counts = token_counts_for_resos(all_resos)
    if not counts:
        raise ValueError("derive_token_budget requires at least one resolution")
    return min(counts), max(counts), counts


def tier_list_budget(target_res: Sequence[int]) -> tuple[int, int, int]:
    """Upper-bound budget from tier list: ``(lo, hi, n_families)``."""
    lo, hi = token_count_range(target_res)
    return lo, hi, token_count_families(target_res)
