"""Shared SPA dist cache-bust key for in-place patched frontend bundles."""

from __future__ import annotations

# Bump this whenever frontend/dist assets are patched in place (same filename hash).
SPA_ASSET_CACHE_KEY = "20260804-v2.9.2-changelog"

# Previous keys replaced by scripts/bump_spa_asset_cache_key.py when bumping.
LEGACY_SPA_ASSET_CACHE_KEYS = (
    "20260605-routefix2",
    "20260627-config-import",
    "20260723-v2.9.0-lokr-preview",
    "20260723-v2.9.0-fast-submit-feedback",
    "20260725-help-pages",
    "20260725-submit-error",
    "20260725-submit-const-fix",
    "20260727-changelog-v2.9.1",
    "20260801-story-brand",
    "20260803-zh-training-controls",
)

IN_PLACE_PATCHED_DIST_ASSETS = (
    "/assets/app.547295de.js",
    "/assets/layout.96d49288.js",
)
