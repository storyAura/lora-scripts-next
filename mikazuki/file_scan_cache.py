from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import stat
from threading import Lock
from typing import Callable


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    path: Path
    size: int
    mtime: float
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    expires_at: float
    snapshots: tuple[FileSnapshot, ...]


def _contains(parent: Path, child: Path) -> bool:
    return child == parent or parent in child.parents


def _scan_files(root: Path) -> tuple[FileSnapshot, ...]:
    if not root.exists():
        return ()
    snapshots: list[FileSnapshot] = []
    for path in root.rglob("*"):
        try:
            file_stat = path.stat()
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(file_stat.st_mode):
            continue
        snapshots.append(
            FileSnapshot(
                path=path,
                size=file_stat.st_size,
                mtime=file_stat.st_mtime,
                mtime_ns=file_stat.st_mtime_ns,
            )
        )
    return tuple(sorted(snapshots, key=lambda item: str(item.path)))


class DirectoryScanCache:
    """Thread-safe connector that shares bounded-lived recursive scans."""

    def __init__(self, ttl_seconds: float, clock: Callable[[], float]):
        if ttl_seconds <= 0:
            raise ValueError(
                f"ttl_seconds must be greater than zero, received {ttl_seconds}"
            )
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries: dict[Path, _CacheEntry] = {}
        self._lock = Lock()

    def scan(self, root: Path) -> tuple[FileSnapshot, ...]:
        resolved = root.resolve()
        with self._lock:
            now = self._clock()
            direct = self._entries.get(resolved)
            if direct is not None and direct.expires_at > now:
                return direct.snapshots

            ancestors = [
                (cached_root, entry)
                for cached_root, entry in self._entries.items()
                if entry.expires_at > now and _contains(cached_root, resolved)
            ]
            if ancestors:
                cached_root, entry = max(
                    ancestors,
                    key=lambda item: len(item[0].parts),
                )
                del cached_root
                return tuple(
                    snapshot
                    for snapshot in entry.snapshots
                    if _contains(resolved, snapshot.path)
                )

            snapshots = _scan_files(resolved)
            self._entries[resolved] = _CacheEntry(
                expires_at=now + self._ttl_seconds,
                snapshots=snapshots,
            )
            return snapshots

    def invalidate(self, root: Path) -> None:
        resolved = root.resolve()
        with self._lock:
            overlapping = [
                cached_root
                for cached_root in self._entries
                if _contains(cached_root, resolved)
                or _contains(resolved, cached_root)
            ]
            for cached_root in overlapping:
                del self._entries[cached_root]
