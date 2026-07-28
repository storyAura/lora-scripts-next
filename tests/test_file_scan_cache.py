from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from mikazuki.file_scan_cache import DirectoryScanCache


class MutableClock:
    def __init__(self, value: float):
        self.value = value

    def __call__(self) -> float:
        return self.value


class DirectoryScanCacheTests(unittest.TestCase):
    def test_scan_reuses_snapshot_until_ttl_expires(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.safetensors"
            first.write_bytes(b"first")
            clock = MutableClock(100.0)
            cache = DirectoryScanCache(2.0, clock)

            initial = cache.scan(root)
            second = root / "second.safetensors"
            second.write_bytes(b"second")
            cached = cache.scan(root)
            clock.value = 102.1
            refreshed = cache.scan(root)

        self.assertEqual([item.path.name for item in initial], ["first.safetensors"])
        self.assertEqual(cached, initial)
        self.assertEqual(
            sorted(item.path.name for item in refreshed),
            ["first.safetensors", "second.safetensors"],
        )

    def test_child_scan_reuses_valid_ancestor_snapshot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            child = root / "run"
            child.mkdir()
            (child / "model.safetensors").write_bytes(b"model")
            clock = MutableClock(100.0)
            cache = DirectoryScanCache(2.0, clock)

            parent_snapshot = cache.scan(root)
            (child / "late.safetensors").write_bytes(b"late")
            child_snapshot = cache.scan(child)

        self.assertEqual(len(parent_snapshot), 1)
        self.assertEqual(
            [item.path.name for item in child_snapshot],
            ["model.safetensors"],
        )

    def test_invalidate_forces_rescan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "first.json").write_text("{}", encoding="utf-8")
            cache = DirectoryScanCache(60.0, MutableClock(100.0))
            cache.scan(root)
            (root / "second.json").write_text("{}", encoding="utf-8")

            cache.invalidate(root)
            refreshed = cache.scan(root)

        self.assertEqual(
            sorted(item.path.name for item in refreshed),
            ["first.json", "second.json"],
        )


if __name__ == "__main__":
    unittest.main()
