"""
Buffers training subprocess stdout per task_id for SSE streaming and optional UI.
"""

from __future__ import annotations

import copy
import threading
import re
from collections import deque
from itertools import islice
from typing import Any, Deque, Dict, List, Tuple

_MAX_LINES = 15000
_ANSI_ESCAPE_RE = re.compile(
    r"\x1B\][^\x07]*?(?:\x07|\x1B\\)|"
    r"\x1B\[[0-?]*[ -/]*[@-~]|"
    r"\x1B[@-Z\\-_]"
)


def strip_ansi(text: str) -> str:
    """Remove terminal color/control codes before streaming logs to browsers."""
    return _ANSI_ESCAPE_RE.sub("", text)


class TrainLogHub:
    """Thread-safe line ring buffer per training task.

    Cursors handed to ``snapshot_from`` / ``snapshot_events_from`` are absolute
    counters (total items ever appended), so SSE clients keep receiving new
    lines after the ring wraps past ``_MAX_LINES`` instead of freezing at the
    buffer length.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._lines: Dict[str, Deque[str]] = {}
        self._events: Dict[str, Deque[dict[str, Any]]] = {}
        self._done: Dict[str, bool] = {}
        self._line_totals: Dict[str, int] = {}
        self._event_totals: Dict[str, int] = {}

    def start_task(self, task_id: str) -> None:
        with self._lock:
            self._lines[task_id] = deque(maxlen=_MAX_LINES)
            self._events[task_id] = deque(maxlen=_MAX_LINES)
            self._done[task_id] = False
            self._line_totals[task_id] = 0
            self._event_totals[task_id] = 0

    def drop_task(self, task_id: str) -> None:
        """Free all buffers for a pruned task so long-lived servers stay bounded."""
        with self._lock:
            self._lines.pop(task_id, None)
            self._events.pop(task_id, None)
            self._done.pop(task_id, None)
            self._line_totals.pop(task_id, None)
            self._event_totals.pop(task_id, None)

    def append_line(self, task_id: str, line: str) -> None:
        text = strip_ansi(line.rstrip("\r\n"))
        if not text and line == "":
            return
        with self._lock:
            dq = self._lines.get(task_id)
            if dq is None:
                dq = deque(maxlen=_MAX_LINES)
                self._lines[task_id] = dq
            dq.append(text)
            self._line_totals[task_id] = self._line_totals.get(task_id, 0) + 1

    def append_event(self, task_id: str, event: dict[str, Any]) -> None:
        payload = copy.deepcopy(event)
        payload.setdefault("type", "progress")
        with self._lock:
            dq = self._events.get(task_id)
            if dq is None:
                dq = deque(maxlen=_MAX_LINES)
                self._events[task_id] = dq
            dq.append(payload)
            self._event_totals[task_id] = self._event_totals.get(task_id, 0) + 1

    def mark_done(self, task_id: str) -> None:
        with self._lock:
            self._done[task_id] = True

    def is_done(self, task_id: str) -> bool:
        with self._lock:
            return self._done.get(task_id, False)

    def snapshot_from(self, task_id: str, start_idx: int) -> Tuple[List[str], int, bool]:
        """Return lines since absolute index start_idx, total appended count, done flag."""
        with self._lock:
            dq = self._lines.get(task_id)
            done = self._done.get(task_id, False)
            if dq is None:
                return [], 0, done
            total = self._line_totals.get(task_id, len(dq))
            # Lines evicted from the ring are gone; clamp the cursor to what remains.
            rel_start = max(0, start_idx - (total - len(dq)))
            chunk = list(islice(dq, rel_start, None))
        return chunk, total, done

    def snapshot_events_from(self, task_id: str, start_idx: int) -> Tuple[List[dict[str, Any]], int, bool]:
        """Return structured progress events since absolute index start_idx."""
        with self._lock:
            dq = self._events.get(task_id)
            done = self._done.get(task_id, False)
            if dq is None:
                return [], 0, done
            total = self._event_totals.get(task_id, len(dq))
            rel_start = max(0, start_idx - (total - len(dq)))
            chunk = [copy.deepcopy(event) for event in islice(dq, rel_start, None)]
        return chunk, total, done

    def tail(self, task_id: str, limit: int = 80) -> List[str]:
        """Return the most recent sanitized log lines for diagnostics."""
        limit = max(1, min(int(limit or 1), _MAX_LINES))
        with self._lock:
            dq = self._lines.get(task_id)
            if dq is None:
                return []
            return list(dq)[-limit:]


hub = TrainLogHub()
