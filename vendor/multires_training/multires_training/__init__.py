"""Portable multi-resolution-per-image training helpers.

Same-epoch expansion of every source image across free-fit tiers
(``multires_per_image``). Independent of Anima/MonadForge trainers — copy or
``pip install -e packages/multires_training`` into a new trainer.
"""

from .batching import (
    BucketBatchIndex,
    ShapeBucketEpoch,
    build_shape_buckets,
    samples_to_bucket_items,
)
from .budget import derive_token_budget, tier_list_budget
from .cache import (
    DEFAULT_CONVENTION,
    DEFAULT_LATENT_SUFFIX,
    LatentCacheConvention,
    LatentCacheFile,
    discover_latents_by_stem,
    parse_latent_cache_name,
    validate_latent_npz,
    write_stub_latent_npz,
)
from .expand import (
    MultiresSample,
    expand_dataset,
    expand_image_to_samples,
    resolve_cache_root_for_image,
    select_tier_caches,
)
from .staging import StagingPlan, make_staging_plan, resize_one, stage_multires_images
from .tiers import (
    ALLOWED_TARGET_RES,
    DEFAULT_FREEFIT_MAX_RATIO,
    DEFAULT_TARGET_RES,
    EDGE_TOKEN_BANDS,
    FREEFIT_BAND_TOLERANCE,
    FREEFIT_BAND_VERSION,
    FREEFIT_FROZEN_EDGES,
    PATCH,
    ROPE_CAP,
    cache_matches_edge,
    choose_edge,
    freefit_band_for_edge,
    freefit_bucket,
    normalize_target_res,
    patch_token_count,
    select_bucket,
    token_count_families,
    token_count_range,
    token_counts_for_resos,
    validate_multires_target_res,
)

__all__ = [
    "ALLOWED_TARGET_RES",
    "BucketBatchIndex",
    "DEFAULT_CONVENTION",
    "DEFAULT_FREEFIT_MAX_RATIO",
    "DEFAULT_LATENT_SUFFIX",
    "DEFAULT_TARGET_RES",
    "EDGE_TOKEN_BANDS",
    "FREEFIT_BAND_TOLERANCE",
    "FREEFIT_BAND_VERSION",
    "FREEFIT_FROZEN_EDGES",
    "LatentCacheConvention",
    "LatentCacheFile",
    "MultiresSample",
    "PATCH",
    "ROPE_CAP",
    "ShapeBucketEpoch",
    "StagingPlan",
    "build_shape_buckets",
    "cache_matches_edge",
    "choose_edge",
    "derive_token_budget",
    "discover_latents_by_stem",
    "expand_dataset",
    "expand_image_to_samples",
    "freefit_band_for_edge",
    "freefit_bucket",
    "make_staging_plan",
    "normalize_target_res",
    "parse_latent_cache_name",
    "patch_token_count",
    "resize_one",
    "resolve_cache_root_for_image",
    "samples_to_bucket_items",
    "select_bucket",
    "select_tier_caches",
    "stage_multires_images",
    "tier_list_budget",
    "token_count_families",
    "token_count_range",
    "token_counts_for_resos",
    "validate_latent_npz",
    "validate_multires_target_res",
    "write_stub_latent_npz",
]

__version__ = "0.1.0"
