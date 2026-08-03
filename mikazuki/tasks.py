import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from subprocess import CompletedProcess, Popen, TimeoutExpired
from typing import Dict, List, Optional

import psutil

from mikazuki.log import log
from mikazuki.train_log_hub import hub

_FAILURE_LOG_TAIL_LINES = 80
_MAX_RETAINED_TERMINAL_TASKS = 16
_KILL_TIMEOUT_SECONDS = 8.0

try:
    import msvcrt
    import _winapi
    _mswindows = True
except ModuleNotFoundError:
    _mswindows = False


def _pid_alive(pid: int) -> bool:
    try:
        proc = psutil.Process(pid)
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False


def _collect_tree(pid: int) -> list[psutil.Process]:
    try:
        parent = psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return []
    procs = [parent]
    try:
        procs.extend(parent.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return procs


def _kill_psutil_tree(pid: int) -> None:
    procs = _collect_tree(pid)
    for proc in reversed(procs):
        try:
            proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    if procs:
        psutil.wait_procs(procs, timeout=2)


def _windows_taskkill_tree(pid: int) -> None:
    try:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning(f"taskkill fallback failed for pid={pid}: {exc}")


def ensure_proc_tree_dead(pid: int, timeout: float = _KILL_TIMEOUT_SECONDS) -> bool:
    """Force-kill a process and all descendants. Returns True when none remain."""
    if pid <= 0:
        return True

    deadline = time.monotonic() + max(1.0, timeout)
    _kill_psutil_tree(pid)

    if _mswindows and _pid_alive(pid):
        _windows_taskkill_tree(pid)
    elif not _mswindows:
        try:
            os.killpg(pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            try:
                os.kill(pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass

    while time.monotonic() < deadline:
        if not _pid_alive(pid) and not _collect_tree(pid):
            return True
        _kill_psutil_tree(pid)
        if _mswindows and _pid_alive(pid):
            _windows_taskkill_tree(pid)
        time.sleep(0.2)

    survivors = [proc.pid for proc in _collect_tree(pid)]
    if survivors:
        log.error(f"Process tree still alive after kill attempts: pid={pid} survivors={survivors}")
        return False
    return not _pid_alive(pid)


def kill_proc_tree(pid, including_parent=True):
    """Backwards-compatible wrapper; always aims to clear the whole tree."""
    if including_parent:
        ensure_proc_tree_dead(pid)
        return
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return
    for child in children:
        try:
            child.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    if children:
        gone, still_alive = psutil.wait_procs(children, timeout=5)
        for child in still_alive:
            try:
                child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass


class TaskStatus(Enum):
    CREATED = 0
    RUNNING = 1
    FINISHED = 2
    TERMINATED = 3
    FAILED = 4
    TERMINATING = 5


class Task:
    _OCCUPYING_STATUSES = (
        TaskStatus.CREATED,
        TaskStatus.RUNNING,
        TaskStatus.TERMINATING,
    )

    def __init__(self, task_id, command, environ=None, metadata=None, cwd=None):
        self.task_id = task_id
        self.lock = threading.Lock()
        self.command = command
        self.status = TaskStatus.CREATED
        self.cancel_requested = False
        self.environ = environ or os.environ
        self.metadata = metadata or {}
        self.cwd = cwd
        self.returncode = None
        self.process: Optional[Popen] = None
        self.log_file = self.metadata.get("log_file")
        self._stdout_thread = None

    def _append_disk_log(self, text: str):
        if not self.log_file:
            return
        try:
            path = Path(self.log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8", errors="replace") as f:
                f.write(text)
                if text and not text.endswith("\n"):
                    f.write("\n")
        except Exception:
            pass

    def occupies_slot(self) -> bool:
        return self.status in self._OCCUPYING_STATUSES and not (
            self.status == TaskStatus.CREATED and self.cancel_requested
        )

    def is_process_alive(self) -> bool:
        proc = self.process
        if proc is None:
            return False
        return proc.poll() is None

    def start_log_only(self):
        with self.lock:
            if self.cancel_requested:
                self.status = TaskStatus.TERMINATED
                return
            self.status = TaskStatus.RUNNING
            self.returncode = None
            self.metadata.pop("returncode", None)
        hub.start_task(self.task_id)

    def finish_log_only(self, returncode=0, error=None):
        self.returncode = returncode
        if error:
            self.metadata["error"] = str(error)
            self._append_disk_log(f"[error] {error}")
        self.status = TaskStatus.FINISHED if returncode == 0 else TaskStatus.FAILED
        self._append_disk_log(f"[task finished] returncode={returncode}")
        hub.mark_done(self.task_id)
        self._record_completion(returncode)

    def _join_stdout_pump(self):
        thread = self._stdout_thread
        if thread and thread.is_alive():
            thread.join(timeout=2)

    def _record_completion(self, returncode):
        self.metadata["returncode"] = returncode
        if returncode == 0:
            self.metadata.pop("last_log_lines", None)
            if self.metadata.get("error") == "Training process exited with code 0":
                self.metadata.pop("error", None)
            return
        message = f"Training process exited with code {returncode}"
        self.metadata.setdefault("error", message)
        self.metadata["last_log_lines"] = hub.tail(self.task_id, _FAILURE_LOG_TAIL_LINES)

    def communicate(self, input=None, timeout=None):
        try:
            stdout, stderr = self.process.communicate(input, timeout=timeout)
        except TimeoutExpired as exc:
            self.process.kill()
            if _mswindows:
                exc.stdout, exc.stderr = self.process.communicate()
            else:
                self.process.wait()
            raise
        except Exception:
            self.process.kill()
            raise
        retcode = self.process.poll()
        self.returncode = retcode
        self.status = TaskStatus.FINISHED if retcode == 0 else TaskStatus.FAILED
        self._append_disk_log(f"[task communicate finished] returncode={retcode}")
        self._record_completion(retcode)
        return CompletedProcess(self.process.args, retcode, stdout, stderr)

    def wait(self):
        if self.process is None:
            self._join_stdout_pump()
            with self.lock:
                if self.status not in (
                    TaskStatus.TERMINATED,
                    TaskStatus.TERMINATING,
                    TaskStatus.FAILED,
                    TaskStatus.FINISHED,
                ):
                    self.status = TaskStatus.FAILED
                    self.returncode = -1
            return

        retcode = self.process.wait()
        self._join_stdout_pump()
        self.returncode = retcode
        with self.lock:
            if self.status in (TaskStatus.TERMINATED, TaskStatus.TERMINATING):
                self.status = TaskStatus.TERMINATED
            else:
                self.status = TaskStatus.FINISHED if retcode == 0 else TaskStatus.FAILED
        self._append_disk_log(f"[task wait finished] returncode={retcode}")
        self._record_completion(retcode)

    def _stdout_pump(self):
        """Drain child stdout into TrainLogHub AND echo to parent console."""
        try:
            if not self.process or self.process.stdout is None:
                return
            for line in iter(self.process.stdout.readline, ""):
                hub.append_line(self.task_id, line)
                self._append_disk_log(line)
                try:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                except Exception:
                    pass
        except Exception as e:
            hub.append_line(self.task_id, f"[stdout pump] {e}")
            self._append_disk_log(f"[stdout pump] {e}")
        finally:
            try:
                if self.process and self.process.stdout:
                    self.process.stdout.close()
            except Exception:
                pass
            hub.mark_done(self.task_id)

    def execute(self):
        with self.lock:
            if self.cancel_requested or self.status in (
                TaskStatus.TERMINATED,
                TaskStatus.TERMINATING,
            ):
                self.status = TaskStatus.TERMINATED
                self.returncode = -1
                self.metadata["error"] = "训练在启动前被终止"
                self._append_disk_log("[task cancelled before start]")
                return
            self.status = TaskStatus.RUNNING
            self.returncode = None
            self.metadata.pop("returncode", None)

        hub.start_task(self.task_id)
        self._append_disk_log(
            "\n"
            f"[task start] {datetime.now().isoformat(timespec='seconds')}\n"
            f"task_id={self.task_id}\n"
            f"cwd={self.cwd or os.getcwd()}\n"
            f"command={' '.join(map(str, self.command))}\n"
        )

        popen_kwargs = {
            "env": self.environ,
            "cwd": self.cwd,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "universal_newlines": True,
            "bufsize": 1,
            "encoding": "utf-8",
            "errors": "replace",
        }
        if _mswindows:
            # Own a process group so taskkill /T can tear down Accelerate children.
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        try:
            process = subprocess.Popen(self.command, **popen_kwargs)
        except Exception as e:
            hub.append_line(self.task_id, f"[error] Failed to start training process: {e}")
            self._append_disk_log(f"[error] Failed to start training process: {e}")
            hub.mark_done(self.task_id)
            with self.lock:
                self.status = TaskStatus.FAILED
                self.returncode = -1
                self.metadata["returncode"] = -1
                self.metadata["error"] = str(e)
            raise

        with self.lock:
            if self.cancel_requested:
                self.process = process
                self.status = TaskStatus.TERMINATING
            else:
                self.process = process

        if self.cancel_requested:
            self._force_stop_process(process)
            with self.lock:
                self.status = TaskStatus.TERMINATED
                self.returncode = process.poll()
                self.metadata["error"] = "训练在启动前被终止"
            self._append_disk_log("[task terminated immediately after start]")
            hub.mark_done(self.task_id)
            return

        self._stdout_thread = threading.Thread(target=self._stdout_pump, daemon=True)
        self._stdout_thread.start()

    def _force_stop_process(self, process: Optional[Popen] = None) -> bool:
        target = process if process is not None else self.process
        if target is None:
            return True
        pid = target.pid
        ok = ensure_proc_tree_dead(pid)
        try:
            if target.poll() is None:
                target.kill()
            target.wait(timeout=5)
        except Exception:
            pass
        return ok and target.poll() is not None

    def terminate(self):
        with self.lock:
            self.cancel_requested = True
            if self.status in (TaskStatus.FINISHED, TaskStatus.FAILED, TaskStatus.TERMINATED):
                return
            process = self.process
            if process is None:
                self.status = TaskStatus.TERMINATED
                self.returncode = -1
                self.metadata.setdefault("error", "训练在启动前被终止")
                self._append_disk_log("[task terminated before start]")
                return
            self.status = TaskStatus.TERMINATING

        try:
            ok = self._force_stop_process(process)
            if not ok:
                self.metadata["error"] = "训练终止超时：仍有子进程存活，请检查 GPU 占用"
                log.error(
                    f"Task {self.task_id} terminate incomplete; refusing to report clean stop"
                )
        except Exception as e:
            log.error(f"Error when killing process: {e}")
            self.metadata["error"] = f"终止训练失败: {e}"
        finally:
            with self.lock:
                self.status = TaskStatus.TERMINATED
                if self.process is not None:
                    self.returncode = self.process.poll()
                elif self.returncode is None:
                    self.returncode = -1
                self._append_disk_log("[task terminated]")


class TaskManager:
    _TERMINAL_STATUSES = (TaskStatus.FINISHED, TaskStatus.TERMINATED, TaskStatus.FAILED)
    _OCCUPYING_STATUSES = Task._OCCUPYING_STATUSES

    def __init__(self, max_concurrent=1) -> None:
        self.max_concurrent = max_concurrent
        self.tasks: Dict[str, Task] = {}

    def _prune_terminal_tasks(self):
        """Drop oldest finished tasks and their log buffers so long sessions stay bounded."""
        terminal_ids = [tid for tid, t in self.tasks.items() if t.status in self._TERMINAL_STATUSES]
        excess = len(terminal_ids) - _MAX_RETAINED_TERMINAL_TASKS
        if excess <= 0:
            return
        for tid in terminal_ids[:excess]:
            self.tasks.pop(tid, None)
            hub.drop_task(tid)

    def create_task(self, command: List[str], environ, metadata=None, cwd=None, task_id=None):
        self._prune_terminal_tasks()
        occupying = [t for t in self.tasks.values() if t.occupies_slot() or t.is_process_alive()]
        if len(occupying) >= self.max_concurrent:
            log.error(
                f"Unable to create a task because there are already {len(occupying)} tasks running, reaching the maximum concurrent limit. / 无法创建任务，因为已经有 {len(occupying)} 个任务正在运行，已达到最大并发限制。")
            return None
        task_id = task_id or str(uuid.uuid4())
        task = Task(task_id=task_id, command=command, environ=environ, metadata=metadata, cwd=cwd)
        self.tasks[task_id] = task
        log.info(f"Task {task_id} created")
        return task

    def add_task(self, task_id: str, task: Task):
        self._prune_terminal_tasks()
        self.tasks[task_id] = task

    def terminate_task(self, task_id: str):
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.terminate()

    def wait_for_process(self, task_id: str):
        if task_id in self.tasks:
            task: Task = self.tasks[task_id]
            task.wait()

    def dump(self) -> List[Dict]:
        return [
            {
                "id": task.task_id,
                "status": task.status.name,
                "metadata": task.metadata,
                "returncode": task.returncode,
            }
            for task in self.tasks.values()
        ]


tm = TaskManager()
