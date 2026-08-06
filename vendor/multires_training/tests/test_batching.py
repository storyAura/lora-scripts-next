from __future__ import annotations

from multires_training import build_shape_buckets, derive_token_budget


def test_incomplete_batches_kept():
    items = [
        ("low", (512, 512)),
        ("high", (768, 768)),
    ]
    epoch = build_shape_buckets(items, batch_size=2, keep_incomplete_batches=True)
    assert len(epoch) == 2
    assert epoch.all_keys_in_epoch() == {"low", "high"}


def test_incomplete_batches_dropped_when_disabled():
    items = [("only", (512, 512))]
    epoch = build_shape_buckets(items, batch_size=2, keep_incomplete_batches=False)
    assert len(epoch) == 0


def test_derive_token_budget():
    lo, hi, counts = derive_token_budget([(512, 512), (1024, 1024)])
    assert lo == (512 // 16) * (512 // 16)
    assert hi == (1024 // 16) * (1024 // 16)
    assert counts == {lo, hi}
