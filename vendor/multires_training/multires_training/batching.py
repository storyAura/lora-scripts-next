"""Shape-homogeneous batching with incomplete-tail keep for multires epochs."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence


@dataclass(frozen=True)
class BucketBatchIndex:
    bucket_index: int
    bucket_batch_size: int
    batch_index: int


@dataclass
class ShapeBucketEpoch:
    """One epoch of shape-homogeneous batches over expanded samples."""

    resos: list[tuple[int, int]]
    buckets: list[list[Hashable]]
    indices: list[BucketBatchIndex]
    batch_size: int
    keep_incomplete_batches: bool

    def __len__(self) -> int:
        return len(self.indices)

    def batch_keys(self, index: BucketBatchIndex) -> list[Hashable]:
        bucket = self.buckets[index.bucket_index]
        start = index.batch_index * index.bucket_batch_size
        return bucket[start : start + index.bucket_batch_size]

    def all_keys_in_epoch(self) -> set[Hashable]:
        keys: set[Hashable] = set()
        for idx in self.indices:
            keys.update(self.batch_keys(idx))
        return keys

    def shuffle(self, rng: random.Random | None = None) -> None:
        rng = rng or random.Random()
        for bucket in self.buckets:
            rng.shuffle(bucket)
        rng.shuffle(self.indices)


def build_shape_buckets(
    items: Iterable[tuple[Hashable, tuple[int, int]]],
    *,
    batch_size: int,
    keep_incomplete_batches: bool = True,
    repeats: int = 1,
) -> ShapeBucketEpoch:
    """Group ``(key, (W,H))`` into shape buckets and emit batch indices.

    For ``multires_per_image``, pass ``keep_incomplete_batches=True`` so a
    resolution tier with ``len(bucket) % batch_size != 0`` is not silently
    dropped from the epoch.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    grouped: dict[tuple[int, int], list[Hashable]] = defaultdict(list)
    for key, size in items:
        for _ in range(max(1, repeats)):
            grouped[(int(size[0]), int(size[1]))].append(key)

    resos = sorted(grouped)
    buckets = [grouped[r] for r in resos]
    indices: list[BucketBatchIndex] = []
    for bucket_index, bucket in enumerate(buckets):
        if keep_incomplete_batches:
            batch_count = int(math.ceil(len(bucket) / batch_size))
        else:
            batch_count = len(bucket) // batch_size
        for batch_index in range(batch_count):
            indices.append(BucketBatchIndex(bucket_index, batch_size, batch_index))

    return ShapeBucketEpoch(
        resos=resos,
        buckets=buckets,
        indices=indices,
        batch_size=batch_size,
        keep_incomplete_batches=keep_incomplete_batches,
    )


def samples_to_bucket_items(
    samples: Sequence,  # MultiresSample-like: image_key, width, height
) -> list[tuple[str, tuple[int, int]]]:
    return [(s.image_key, (s.width, s.height)) for s in samples]
