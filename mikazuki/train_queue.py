"""Sequential training queue / 顺序训练队列.

State machine behind the 「训练队列」 sidebar panel:

- ``intercept_run`` sits at the top of ``POST /api/run``: while a task is
  running (or 排队模式 is on) new submits become queue entries instead of
  hitting the trainer; while an entry is being edited the submit *saves* into
  that entry and the conveyor stays halted until the user starts it manually.
- ``runner`` is an asyncio loop started at app lifespan: when the current
  task reaches a terminal state it records the achieved it/s (fed back into
  per-entry ETA estimates), marks the entry done/failed, and launches the
  next queued entry. Failures (OOM etc.) are surfaced on the entry and the
  queue continues.

State persists to ``config/train_queue.json``; ``active`` is always False
after a restart so a reboot never silently starts the GPU.
"""

import asyncio
import copy
import json
import math
import os
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from mikazuki.log import log

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".avif", ".jfif"}
_REPEAT_DIR_RE = re.compile(r"^(\d+)_")
_SPEED_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(it/s|s/it)", re.IGNORECASE)
_SPEED_TAIL_LINES = 300

# Terminal entry states never re-enter the conveyor on their own.
EDITABLE_STATUSES = {"queued", "paused", "failed", "done"}


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _to_pos_int(value: Any, default: int = 0) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def count_dataset_images(train_data_dir: Any) -> Optional[int]:
    """Repeat-weighted image count over kohya-style ``<repeat>_<name>`` subdirs."""
    if not isinstance(train_data_dir, str) or not train_data_dir.strip():
        return None
    root = Path(train_data_dir.strip())
    if not root.is_dir():
        return None
    total = 0
    try:
        for sub in root.iterdir():
            if not sub.is_dir():
                continue
            match = _REPEAT_DIR_RE.match(sub.name)
            if not match:
                continue
            repeats = int(match.group(1))
            images = sum(1 for f in sub.iterdir() if f.is_file() and f.suffix.lower() in _IMAGE_EXTS)
            total += repeats * images
    except OSError:
        return None
    return total or None


def estimate_total_steps(config: Dict[str, Any]) -> tuple:
    """Best-effort (images, steps) from the raw GUI config. Either may be None."""
    images = count_dataset_images(config.get("train_data_dir"))
    batch_size = _to_pos_int(config.get("train_batch_size"), 1)
    grad_acc = _to_pos_int(config.get("gradient_accumulation_steps"), 1)
    epochs = _to_pos_int(config.get("max_train_epochs"), 0)
    max_steps = _to_pos_int(config.get("max_train_steps"), 0)
    steps = None
    if images and epochs:
        steps = math.ceil(math.ceil(images / batch_size) / grad_acc) * epochs
    if max_steps:
        steps = min(steps, max_steps) if steps else max_steps
    return images, steps


def parse_speed_from_lines(lines: List[str]) -> Optional[float]:
    """Last tqdm-style speed in the log tail, normalized to it/s."""
    for line in reversed(lines or []):
        match = None
        for match in _SPEED_RE.finditer(line):
            pass
        if match:
            value = float(match.group(1))
            if value <= 0:
                continue
            return value if match.group(2).lower() == "it/s" else 1.0 / value
    return None


def _default_tasks() -> Dict[str, Any]:
    from mikazuki.tasks import tm
    return tm.tasks


def _default_tail(task_id: str, limit: int) -> List[str]:
    from mikazuki.train_log_hub import hub
    return hub.tail(task_id, limit)


def _success(message: str, data: Optional[dict] = None):
    from mikazuki.app.models import APIResponseSuccess
    return APIResponseSuccess(message=message, data=data)


def _fail(message: str, data: Optional[dict] = None):
    from mikazuki.app.models import APIResponseFail
    return APIResponseFail(message=message, data=data)


class TrainQueue:
    def __init__(
        self,
        path: Optional[Path] = None,
        tasks_source: Callable[[], Dict[str, Any]] = _default_tasks,
        log_tail: Callable[[str, int], List[str]] = _default_tail,
    ) -> None:
        self._explicit_path = Path(path) if path else None
        self._lock = threading.RLock()
        self._tasks_source = tasks_source
        self._log_tail = log_tail
        self._submit = None  # async callable(config) -> APIResponse, wired by api.py
        # /api/run interception stays dormant until the runner starts (app lifespan),
        # so direct create_toml_file() calls in tests/scripts keep stock behavior.
        self._armed = False
        self._loaded = False
        self._last_speed_task_id: Optional[str] = None
        self.entries: List[Dict[str, Any]] = []
        self.active = False
        # user's explicit pause switch: while True, new submits still enqueue
        # but the conveyor will not start until 「开始队列」
        self.user_paused = False
        self.halt_reason = ""
        self.last_speed: Optional[Dict[str, Any]] = None

    # ---------------------------------------------------------------- plumbing

    @property
    def _path(self) -> Path:
        return self._explicit_path or (Path.cwd() / "config" / "train_queue.json")

    def set_submit(self, submit) -> None:
        self._submit = submit

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError) as exc:
            log.warning(f"train queue state unreadable, starting empty: {exc}")
            return
        entries = raw.get("entries")
        self.entries = entries if isinstance(entries, list) else []
        self.user_paused = bool(raw.get("user_paused"))
        self.last_speed = raw.get("last_speed") if isinstance(raw.get("last_speed"), dict) else None
        interrupted = False
        for entry in self.entries:
            if entry.get("status") in ("running", "editing"):
                # a restart kills training children and orphans browser edits
                if entry.get("status") == "running":
                    entry["status"] = "failed"
                    entry["error"] = "服务重启，训练已中断"
                    entry["finished_at"] = _now_iso()
                else:
                    entry["status"] = "queued"
                interrupted = True
        # never auto-start the GPU on boot
        self.active = False
        if interrupted:
            self._save()

    def _save(self) -> None:
        payload = {
            "entries": self.entries,
            "user_paused": self.user_paused,
            "last_speed": self.last_speed,
        }
        try:
            path = self._path
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, path)
        except OSError as exc:
            log.warning(f"train queue state save failed: {exc}")

    def _find(self, entry_id: str) -> Optional[Dict[str, Any]]:
        return next((e for e in self.entries if e.get("id") == entry_id), None)

    @staticmethod
    def _task_status_name(task: Any) -> str:
        status = getattr(task, "status", None)
        return getattr(status, "name", str(status))

    def _busy_locked(self, tasks: Dict[str, Any]) -> bool:
        if any(e.get("status") == "running" for e in self.entries):
            return True
        return any(self._task_status_name(t) in ("CREATED", "RUNNING") for t in tasks.values())

    def busy(self) -> bool:
        with self._lock:
            self._ensure_loaded()
            try:
                tasks = self._tasks_source()
            except Exception:
                tasks = {}
            return self._busy_locked(tasks)

    # ------------------------------------------------------------------ state

    def _new_entry(self, config: Dict[str, Any]) -> Dict[str, Any]:
        name = str(config.get("output_name") or "").strip() or str(config.get("model_train_type") or "未命名任务")
        images, steps = estimate_total_steps(config)
        now = _now_iso()
        return {
            "id": uuid.uuid4().hex[:12],
            "name": name,
            "config": copy.deepcopy(config),
            "status": "queued",
            "error": None,
            "task_id": None,
            "train_type": config.get("model_train_type"),
            "lora_type": config.get("lora_type"),
            "images": images,
            "steps": steps,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
        }

    @staticmethod
    def _duration_seconds(entry: Dict[str, Any]) -> Optional[int]:
        started, finished = entry.get("started_at"), entry.get("finished_at")
        if not started or not finished:
            return None
        try:
            seconds = (datetime.fromisoformat(finished) - datetime.fromisoformat(started)).total_seconds()
        except (TypeError, ValueError):
            return None
        return round(seconds) if seconds >= 0 else None

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            self._ensure_loaded()
            it_s = (self.last_speed or {}).get("it_s")
            entries = []
            for entry in self.entries:
                view = {k: v for k, v in entry.items() if k != "config"}
                steps = entry.get("steps")
                view["eta_seconds"] = round(steps / it_s) if steps and it_s else None
                view["duration_seconds"] = self._duration_seconds(entry)
                entries.append(view)
            running = next((e for e in self.entries if e.get("status") == "running"), None)
            editing = next((e for e in self.entries if e.get("status") == "editing"), None)
            return {
                "active": self.active,
                "user_paused": self.user_paused,
                "busy": self.busy(),
                "halt_reason": self.halt_reason,
                "last_speed": self.last_speed,
                "running_entry_id": running["id"] if running else None,
                "running_task_id": running.get("task_id") if running else None,
                "editing_entry_id": editing["id"] if editing else None,
                "entries": entries,
            }

    def entry_config(self, entry_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            self._ensure_loaded()
            entry = self._find(entry_id)
            return copy.deepcopy(entry["config"]) if entry else None

    # ------------------------------------------------------------- run intercept

    def intercept_run(self, config: Dict[str, Any]):
        """Queue-aware routing for POST /api/run. None → run immediately."""
        if not self._armed or not isinstance(config, dict):
            return None
        with self._lock:
            self._ensure_loaded()
            editing = next((e for e in self.entries if e.get("status") == "editing"), None)
            if editing is not None:
                editing["config"] = copy.deepcopy(config)
                editing["name"] = str(config.get("output_name") or "").strip() or editing["name"]
                editing["train_type"] = config.get("model_train_type")
                editing["lora_type"] = config.get("lora_type")
                editing["images"], editing["steps"] = estimate_total_steps(config)
                editing["status"] = "queued"
                editing["error"] = None
                editing["updated_at"] = _now_iso()
                self._save()
                return _success(
                    f"已保存修改到队列任务「{editing['name']}」",
                    data={
                        "queued": True,
                        "entry_id": editing["id"],
                        "queue_message": f"已保存修改到队列任务「{editing['name']}」，队列保持暂停，请到训练队列手动开始",
                    },
                )

            # every submit goes through the queue; unless the user paused it,
            # the conveyor (re)starts so an idle submit begins within seconds
            try:
                tasks = self._tasks_source()
            except Exception:
                tasks = {}
            running_busy = self._busy_locked(tasks)
            entry = self._new_entry(config)
            self.entries.append(entry)
            position = sum(1 for e in self.entries if e.get("status") == "queued")
            if self.user_paused:
                suffix = "。队列处于暂停中，请到训练队列点「开始队列」"
            else:
                self.active = True
                self.halt_reason = ""
                suffix = "，当前任务完成后自动开始" if running_busy else "，即将自动开始"
            self._save()
            message = f"已加入训练队列（第 {position} 位）{suffix}"
            return _success(message, data={"queued": True, "entry_id": entry["id"], "queue_message": message})

    # ------------------------------------------------------------- entry actions

    def reorder(self, ids: List[str]):
        with self._lock:
            self._ensure_loaded()
            by_id = {e["id"]: e for e in self.entries}
            picked = [by_id[i] for i in ids if i in by_id]
            rest = [e for e in self.entries if e["id"] not in set(ids)]
            self.entries = picked + rest
            self._save()
            return _success("已更新队列顺序", data=self.snapshot())

    def remove(self, entry_id: str):
        with self._lock:
            self._ensure_loaded()
            entry = self._find(entry_id)
            if entry is None:
                return _fail("任务不存在或已被移除")
            if entry.get("status") == "running":
                return _fail("正在训练的任务不能删除，请先在训练页终止")
            self.entries.remove(entry)
            self._save()
            return _success(f"已删除「{entry['name']}」", data=self.snapshot())

    def _set_status(self, entry_id: str, wanted: str, allowed_from: set, verb: str):
        with self._lock:
            self._ensure_loaded()
            entry = self._find(entry_id)
            if entry is None:
                return _fail("任务不存在或已被移除")
            if entry.get("status") not in allowed_from:
                return _fail(f"当前状态（{entry.get('status')}）不能{verb}")
            entry["status"] = wanted
            entry["updated_at"] = _now_iso()
            if wanted == "queued":
                entry["error"] = None
            self._save()
            return _success(f"已{verb}「{entry['name']}」", data=self.snapshot())

    def pause(self, entry_id: str):
        return self._set_status(entry_id, "paused", {"queued"}, "暂停")

    def resume(self, entry_id: str):
        return self._set_status(entry_id, "queued", {"paused"}, "恢复")

    def requeue(self, entry_id: str):
        return self._set_status(entry_id, "queued", {"failed", "done"}, "重新排队")

    def set_editing(self, entry_id: str, editing: bool):
        with self._lock:
            self._ensure_loaded()
            entry = self._find(entry_id)
            if entry is None:
                return _fail("任务不存在或已被移除")
            if editing:
                other = next((e for e in self.entries if e.get("status") == "editing"), None)
                if other is not None and other is not entry:
                    return _fail(f"「{other['name']}」正在编辑中，请先保存或取消")
                if entry.get("status") not in EDITABLE_STATUSES:
                    return _fail("正在训练或排队启动中的任务不能编辑")
                entry["status"] = "editing"
                # per spec: while editing, a finishing task must NOT trigger the
                # next one — and new submits must not resume the conveyor either
                self.active = False
                self.user_paused = True
                self.halt_reason = "有任务处于编辑状态，队列已暂停；保存后请手动开始"
            else:
                if entry.get("status") != "editing":
                    return _fail("该任务不在编辑状态")
                entry["status"] = "queued"
            entry["updated_at"] = _now_iso()
            self._save()
            return _success("已更新编辑状态", data=self.snapshot())

    def start_entry(self, entry_id: str):
        """立即开始：move to front and let the conveyor pick it up."""
        with self._lock:
            self._ensure_loaded()
            entry = self._find(entry_id)
            if entry is None:
                return _fail("任务不存在或已被移除")
            if entry.get("status") not in ("queued", "paused", "failed", "done"):
                return _fail(f"当前状态（{entry.get('status')}）不能开始")
            try:
                tasks = self._tasks_source()
            except Exception:
                tasks = {}
            if self._busy_locked(tasks):
                return _fail("已有任务正在训练，它完成后队列会继续；如需插队请先终止当前训练")
            entry["status"] = "queued"
            entry["error"] = None
            entry["updated_at"] = _now_iso()
            self.entries.remove(entry)
            self.entries.insert(0, entry)
            self.user_paused = False
            self.active = True
            self.halt_reason = ""
            self._save()
            return _success(f"「{entry['name']}」将立即开始", data=self.snapshot())

    # ------------------------------------------------------------- queue actions

    def start(self, include_paused: bool = False):
        with self._lock:
            self._ensure_loaded()
            if any(e.get("status") == "editing" for e in self.entries):
                return _fail("有任务处于编辑状态，请先保存或取消编辑再开始队列")
            if include_paused:
                for entry in self.entries:
                    if entry.get("status") == "paused":
                        entry["status"] = "queued"
            self.user_paused = False
            self.active = True
            self.halt_reason = ""
            self._save()
            if any(e.get("status") in ("queued", "running") for e in self.entries):
                message = "队列已开始，按顺序自动训练"
            else:
                message = "队列已就绪：提交训练任务后自动按序开始"
            return _success(message, data=self.snapshot())

    def stop(self):
        with self._lock:
            self._ensure_loaded()
            self.user_paused = True
            self.active = False
            self.halt_reason = "已手动暂停队列（正在训练的任务不受影响）"
            self._save()
            return _success("队列已暂停，当前训练继续，不再自动开始下一个", data=self.snapshot())

    def clear_finished(self):
        with self._lock:
            self._ensure_loaded()
            before = len(self.entries)
            self.entries = [e for e in self.entries if e.get("status") not in ("done", "failed")]
            self._save()
            return _success(f"已清除 {before - len(self.entries)} 个已结束任务", data=self.snapshot())

    # ---------------------------------------------------------------- conveyor

    def _record_speed(self, entry: Optional[Dict[str, Any]], task_id: str,
                      name: str, lora_type: Optional[str], train_type: Optional[str]) -> None:
        if not task_id or task_id == self._last_speed_task_id:
            return
        try:
            lines = self._log_tail(task_id, _SPEED_TAIL_LINES)
        except Exception:
            return
        it_s = parse_speed_from_lines(lines)
        self._last_speed_task_id = task_id
        if not it_s:
            return
        self.last_speed = {
            "it_s": round(it_s, 3),
            "name": name,
            "lora_type": lora_type,
            "train_type": train_type,
            "recorded_at": _now_iso(),
        }

    def _capture_manual_speed(self, tasks: Dict[str, Any]) -> bool:
        """Record it/s from the newest terminal task even if it was started manually."""
        queue_task_ids = {e.get("task_id") for e in self.entries}
        newest = None
        for task_id, task in tasks.items():
            if self._task_status_name(task) in ("FINISHED", "FAILED", "TERMINATED"):
                newest = (task_id, task)
        if newest is None:
            return False
        task_id, task = newest
        if task_id in queue_task_ids or task_id == self._last_speed_task_id:
            return False
        metadata = getattr(task, "metadata", None) or {}
        config_path = str(metadata.get("config_path") or "")
        name = Path(config_path).stem if config_path else "手动任务"
        self._record_speed(None, task_id, name, None, None)
        return True

    def tick(self) -> Optional[Dict[str, Any]]:
        """One conveyor step. Returns a deep copy of the entry to launch, or None."""
        with self._lock:
            self._ensure_loaded()
            changed = False
            try:
                tasks = self._tasks_source()
            except Exception:
                tasks = {}

            for entry in self.entries:
                if entry.get("status") != "running" or not entry.get("task_id"):
                    continue
                task = tasks.get(entry["task_id"])
                if task is None:
                    entry["status"] = "failed"
                    entry["error"] = "训练任务记录丢失（服务可能重启过）"
                    entry["finished_at"] = _now_iso()
                    changed = True
                    continue
                status = self._task_status_name(task)
                if status not in ("FINISHED", "FAILED", "TERMINATED"):
                    continue
                entry["finished_at"] = _now_iso()
                changed = True
                self._record_speed(entry, entry["task_id"], entry["name"],
                                   entry.get("lora_type"), entry.get("train_type"))
                if status == "FINISHED":
                    entry["status"] = "done"
                    entry["error"] = None
                elif status == "TERMINATED":
                    entry["status"] = "failed"
                    entry["error"] = "训练被手动终止"
                else:
                    metadata = getattr(task, "metadata", None) or {}
                    entry["status"] = "failed"
                    entry["error"] = str(metadata.get("error") or "训练进程异常退出（可查看日志）")

            if self._capture_manual_speed(tasks):
                changed = True

            if self.active and any(e.get("status") == "editing" for e in self.entries):
                self.active = False
                self.user_paused = True
                self.halt_reason = "有任务处于编辑状态，队列已暂停；保存后请手动开始"
                changed = True

            launch = None
            if self.active and not self._busy_locked(tasks):
                launch = next((e for e in self.entries if e.get("status") == "queued"), None)
                if launch is None:
                    self.active = False
                    self.halt_reason = "队列已全部执行完毕，提交新任务会自动开始"
                    changed = True
                else:
                    launch["status"] = "running"
                    launch["task_id"] = None
                    launch["error"] = None
                    launch["started_at"] = _now_iso()
                    launch["updated_at"] = _now_iso()
                    changed = True

            if changed:
                self._save()
            return copy.deepcopy(launch) if launch else None

    def mark_launch_result(self, entry_id: str, ok: bool,
                           task_id: Optional[str] = None, error: Optional[str] = None) -> None:
        with self._lock:
            self._ensure_loaded()
            entry = self._find(entry_id)
            if entry is None or entry.get("status") != "running":
                return
            if ok:
                entry["task_id"] = task_id
            else:
                entry["status"] = "failed"
                entry["error"] = str(error or "启动失败")
                entry["finished_at"] = _now_iso()
                # queue keeps going: next tick launches the following entry
            self._save()

    async def _launch(self, entry: Dict[str, Any]) -> None:
        submit = self._submit
        if submit is None:
            self.mark_launch_result(entry["id"], False, error="队列后端未接线（内部错误）")
            return
        log.info(f"train queue: launching entry {entry['id']} 「{entry['name']}」")
        try:
            response = await submit(copy.deepcopy(entry["config"]))
        except Exception as exc:  # noqa: BLE001 - entry-level failure must not kill the conveyor
            log.exception("train queue: launch crashed")
            self.mark_launch_result(entry["id"], False, error=f"启动异常: {exc}")
            return
        status = getattr(response, "status", None)
        data = getattr(response, "data", None) or {}
        if status == "success" and data.get("task_id"):
            self.mark_launch_result(entry["id"], True, task_id=data["task_id"])
        else:
            message = getattr(response, "message", None) or "启动失败"
            self.mark_launch_result(entry["id"], False, error=str(message))

    async def runner(self, interval: float = 2.0) -> None:
        self._armed = True
        log.info("train queue runner started")
        while True:
            try:
                launch = self.tick()
                if launch is not None:
                    await self._launch(launch)
            except Exception:  # noqa: BLE001 - the conveyor must survive anything
                log.exception("train queue runner tick failed")
            await asyncio.sleep(interval)


train_queue = TrainQueue()
