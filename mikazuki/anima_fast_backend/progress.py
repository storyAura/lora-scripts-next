from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import json
import logging
import os
import re
from threading import Lock
from typing import Any


logger = logging.getLogger(__name__)

_LOSS_KEYS = (
    "loss/average",
    "loss/current",
    "loss",
    "train_loss",
    "avr_loss",
)


@dataclass
class _JsonlReadState:
    identity: tuple[int, int, int]
    offset: int
    partial: bytes
    run_start: dict[str, Any] | None
    recent: deque[dict[str, Any]]


def _file_identity(file_stat: os.stat_result) -> tuple[int, int, int]:
    device = file_stat.st_dev
    inode = file_stat.st_ino
    creation_time = file_stat.st_ctime_ns
    fallback_creation_time = creation_time if inode == 0 else 0
    return (device, inode, fallback_creation_time)


def _read_from_offset(path: Path, offset: int) -> tuple[bytes, int]:
    with path.open("rb") as handle:
        handle.seek(offset)
        payload = handle.read()
        return payload, handle.tell()


def _decode_jsonl_line(path: Path, line: bytes) -> dict[str, Any] | None:
    if not line.strip():
        return None
    try:
        decoded = line.rstrip(b"\r").decode("utf-8")
    except UnicodeDecodeError as exc:
        logger.warning(
            "Ignoring invalid UTF-8 JSONL event",
            extra={
                "path": str(path),
                "byte_start": exc.start,
                "byte_end": exc.end,
            },
        )
        return None
    try:
        value = json.loads(decoded)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Ignoring malformed JSONL event",
            extra={
                "path": str(path),
                "line": exc.lineno,
                "column": exc.colno,
                "reason": exc.msg,
            },
        )
        return None
    if not isinstance(value, dict):
        logger.warning(
            "Ignoring non-object JSONL event",
            extra={
                "path": str(path),
                "value_type": type(value).__name__,
            },
        )
        return None
    return value


def _decode_complete_partial(line: bytes) -> dict[str, Any] | None:
    if not line.strip():
        return None
    try:
        value = json.loads(line.rstrip(b"\r").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


class JsonlEventReader:
    """Incremental connector for append-only JSONL progress files."""

    def __init__(self, max_events: int):
        if max_events < 2:
            raise ValueError(f"max_events must be at least 2, received {max_events}")
        self._max_events = max_events
        self._states: dict[Path, _JsonlReadState] = {}
        self._lock = Lock()

    def _new_state(self, identity: tuple[int, int, int]) -> _JsonlReadState:
        return _JsonlReadState(
            identity=identity,
            offset=0,
            partial=b"",
            run_start=None,
            recent=deque(maxlen=self._max_events),
        )

    def _snapshot(self, state: _JsonlReadState) -> list[dict[str, Any]]:
        partial_event = _decode_complete_partial(state.partial)
        partial_kind = (
            partial_event.get("ev") or partial_event.get("event")
            if partial_event is not None
            else None
        )
        if partial_kind == "run_start":
            return [partial_event]

        recent_limit = self._max_events - 1 if state.run_start is not None else self._max_events
        recent = list(state.recent)[-recent_limit:]
        if state.run_start is None:
            snapshot = recent
        else:
            snapshot = [state.run_start, *recent]
        if partial_event is None:
            return snapshot
        if state.run_start is not None:
            recent_slots = self._max_events - 2
            retained = snapshot[1:][-recent_slots:] if recent_slots > 0 else []
            return [state.run_start, *retained, partial_event]
        return [*snapshot[-(self._max_events - 1):], partial_event]

    def read(self, path: Path) -> list[dict[str, Any]]:
        resolved = path.resolve()
        with self._lock:
            if not resolved.is_file():
                self._states.pop(resolved, None)
                return []

            stat = resolved.stat()
            identity = _file_identity(stat)
            state = self._states.get(resolved)
            if (
                state is None
                or state.identity != identity
                or stat.st_size < state.offset
            ):
                state = self._new_state(identity)
            elif stat.st_size == state.offset:
                return self._snapshot(state)

            payload, new_offset = _read_from_offset(resolved, state.offset)
            if not payload:
                self._states[resolved] = state
                return self._snapshot(state)

            parts = (state.partial + payload).split(b"\n")
            state.partial = parts.pop()
            state.offset = new_offset
            for raw_line in parts:
                event = _decode_jsonl_line(resolved, raw_line)
                if event is None:
                    continue
                kind = event.get("ev") or event.get("event")
                if kind == "run_start":
                    state.run_start = event
                    state.recent.clear()
                    continue
                state.recent.append(event)

            self._states[resolved] = state
            return self._snapshot(state)


_DEFAULT_JSONL_EVENT_READER = JsonlEventReader(4096)


def read_jsonl_events(path: Path) -> list[dict[str, Any]]:
    return _DEFAULT_JSONL_EVENT_READER.read(path)


def _pick_loss(event: dict[str, Any]) -> Any:
    for key in _LOSS_KEYS:
        if key in event and event[key] is not None:
            return event[key]
    return None


def _format_seconds(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}小时{minutes:02d}分{secs:02d}秒"
    if minutes:
        return f"{minutes}分{secs:02d}秒"
    return f"{secs}秒"


def _format_epoch(current: int, total_epochs: int) -> str:
    if total_epochs > 0:
        return f"{current}/{total_epochs}"
    return str(current)


def metrics_from_anima_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    total_steps = 0
    total_epochs = 0
    loss_points: list[dict[str, float | int]] = []
    last_ts = 0.0
    last_step = 0

    for event in events:
        kind = event.get("ev") or event.get("event")
        ts = float(event.get("ts") or 0)
        if ts > 0:
            last_ts = ts

        if kind == "run_start":
            total_steps = int(event.get("total_steps") or total_steps or 0)
            total_epochs = int(event.get("total_epochs") or total_epochs or 0)
            metrics["total_steps"] = total_steps
            metrics["started"] = True
        elif kind == "step":
            step = int(event.get("global_step") or event.get("step") or 0)
            total = int(event.get("total_steps") or total_steps or 0)
            loss = _pick_loss(event)
            last_step = step
            metrics.update({
                "step": step,
                "total_steps": total,
                "percent": round(step * 100 / total, 2) if total else 0,
            })
            epoch_raw = event.get("epoch")
            if epoch_raw is not None:
                try:
                    metrics["epoch"] = _format_epoch(int(epoch_raw), total_epochs)
                except (TypeError, ValueError):
                    pass
            if loss is not None:
                try:
                    loss_float = float(loss)
                    metrics["loss"] = f"{loss_float:.4g}"
                    loss_points.append({"step": step, "loss": loss_float})
                except (TypeError, ValueError):
                    metrics["loss"] = str(loss)
        elif kind == "val":
            if "cmmd" in event:
                metrics["cmmd"] = event.get("cmmd")
            epoch_raw = event.get("epoch")
            if epoch_raw is not None and total_epochs > 0:
                try:
                    metrics["epoch"] = _format_epoch(int(epoch_raw), total_epochs)
                except (TypeError, ValueError):
                    pass
        elif kind == "ckpt":
            metrics["last_checkpoint"] = event.get("path")
        elif kind == "run_end":
            metrics["completed"] = event.get("status") == "ok"
            metrics["run_status"] = event.get("status")
            metrics["step"] = int(event.get("final_step") or metrics.get("step") or 0)
            if event.get("error"):
                metrics["has_error"] = True
                metrics["strong_error"] = event.get("error")

    if last_ts > 0:
        metrics["elapsed"] = _format_seconds(last_ts)
        if last_step > 0 and total_steps > last_step:
            rate = last_ts / last_step
            metrics["eta"] = _format_seconds((total_steps - last_step) * rate)

    if loss_points:
        metrics["loss_points"] = loss_points[-240:]
    return metrics


def merge_anima_training_metrics(
    stdout_metrics: dict[str, Any],
    anima_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Merge stdout-parsed Kohya/tqdm metrics with anima progress.jsonl metrics."""
    if not anima_metrics:
        return dict(stdout_metrics)

    merged = dict(stdout_metrics)
    jsonl_step = anima_metrics.get("step")
    stdout_step = merged.get("step")

    jsonl_has_progress = isinstance(jsonl_step, int) and jsonl_step > 0
    stdout_has_progress = isinstance(stdout_step, int) and stdout_step > 0
    prefer_jsonl_step = jsonl_has_progress and (
        not stdout_has_progress or jsonl_step >= stdout_step
    )

    progress_keys = ("step", "total_steps", "percent")
    for key in progress_keys:
        if prefer_jsonl_step and key in anima_metrics:
            merged[key] = anima_metrics[key]
        elif key not in merged and key in anima_metrics:
            merged[key] = anima_metrics[key]

    fill_keys = (
        "loss",
        "loss_points",
        "epoch",
        "elapsed",
        "eta",
        "cmmd",
        "last_checkpoint",
        "completed",
        "run_status",
        "started",
        "progress_source",
    )
    for key in fill_keys:
        value = anima_metrics.get(key)
        if value in (None, ""):
            continue
        if key == "eta" and str(merged.get("eta", "")).strip() not in ("", "?", "-"):
            continue
        if key in merged and merged.get(key) not in (None, "", "-", "?"):
            if key != "loss":
                continue
        merged[key] = value

    if anima_metrics.get("progress_source"):
        merged["progress_source"] = anima_metrics["progress_source"]
    return merged


def fallback_metrics_from_stdout(lines: list[str]) -> dict[str, Any]:
    text = "\n".join(lines[-1000:])
    matches = list(re.finditer(
        r"steps:\s*\d+%\|.*?\|\s*(?P<step>\d+)\s*/\s*(?P<total>\d+)"
        r".*?(?:loss|avr_loss|train_loss|loss/average|loss/current)[=:]\s*(?P<loss>[0-9.eE+-]+)",
        text,
    ))
    if not matches:
        return {}
    m = matches[-1]
    step = int(m.group("step"))
    total = int(m.group("total"))
    return {
        "step": step,
        "total_steps": total,
        "percent": round(step * 100 / total, 2) if total else 0,
        "loss": float(m.group("loss")),
    }
