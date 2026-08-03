from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from mikazuki.tasks import (
    Task,
    TaskManager,
    TaskStatus,
    ensure_proc_tree_dead,
    tm as global_tm,
)
from mikazuki.train_log_hub import hub
from mikazuki.train_queue import TrainQueue


def _sleep_command(seconds: float) -> list[str]:
    return [
        sys.executable,
        "-c",
        (
            "import os, time, sys\n"
            "print('child-start', os.getpid(), flush=True)\n"
            f"time.sleep({seconds})\n"
            "print('child-end', flush=True)\n"
        ),
    ]


def _nested_sleep_command(seconds: float) -> list[str]:
    """Parent python that spawns a grandchild sleeper (Accelerate-like tree)."""
    inner = (
        "import os, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import os,time; print(\"grand-start\", os.getpid(), flush=True); "
        f"time.sleep({seconds})'])\n"
        "print('parent-start', os.getpid(), 'child', child.pid, flush=True)\n"
        "child.wait()\n"
    )
    return [sys.executable, "-c", inner]


class TaskTerminateLifecycleTests(unittest.TestCase):
    def tearDown(self):
        for task_id in list(global_tm.tasks):
            hub.drop_task(task_id)
        global_tm.tasks.clear()

    def test_terminate_before_execute_blocks_launch(self):
        task = Task("pre-start", _sleep_command(30), environ=os.environ.copy())
        task.terminate()
        self.assertEqual(task.status, TaskStatus.TERMINATED)
        self.assertTrue(task.cancel_requested)

        task.execute()
        self.assertEqual(task.status, TaskStatus.TERMINATED)
        self.assertIsNone(task.process)
        task.wait()
        self.assertEqual(task.status, TaskStatus.TERMINATED)

    def test_terminate_kills_running_process_tree(self):
        task = Task("tree-kill", _nested_sleep_command(60), environ=os.environ.copy())
        task.execute()
        self.assertEqual(task.status, TaskStatus.RUNNING)
        self.assertIsNotNone(task.process)
        time.sleep(0.4)
        root_pid = task.process.pid

        task.terminate()
        self.assertEqual(task.status, TaskStatus.TERMINATED)
        self.assertIsNotNone(task.process.poll())
        self.assertFalse(task.is_process_alive())
        # Give Windows a beat to reap grandchildren reported by taskkill.
        time.sleep(0.5)
        self.assertTrue(ensure_proc_tree_dead(root_pid, timeout=2.0))

    def test_create_task_refuses_while_terminating_or_alive(self):
        manager = TaskManager(max_concurrent=1)
        first = manager.create_task(_sleep_command(30), environ=os.environ.copy(), task_id="slot-1")
        self.assertIsNotNone(first)
        first.status = TaskStatus.TERMINATING
        second = manager.create_task(_sleep_command(1), environ=os.environ.copy(), task_id="slot-2")
        self.assertIsNone(second)

    def test_queue_stays_busy_while_terminating(self):
        with tempfile.TemporaryDirectory() as td:
            tasks = {
                "t1": SimpleNamespace(
                    status=SimpleNamespace(name="TERMINATING"),
                    metadata={},
                    is_process_alive=lambda: True,
                )
            }
            queue = TrainQueue(
                path=Path(td) / "queue.json",
                tasks_source=lambda: tasks,
                log_tail=lambda task_id, limit: [],
            )
            queue._armed = True
            queue.stop()
            queue.intercept_run({"output_name": "a", "model_train_type": "anima-lora"})
            queue.intercept_run({"output_name": "b", "model_train_type": "anima-lora"})
            entry = queue.entries[0]
            entry["status"] = "running"
            entry["task_id"] = "t1"
            queue.start()

            self.assertIsNone(queue.tick())
            self.assertEqual(queue.entries[0]["status"], "running")

            tasks["t1"] = SimpleNamespace(
                status=SimpleNamespace(name="TERMINATED"),
                metadata={},
                is_process_alive=lambda: False,
            )
            launch = queue.tick()
            self.assertIsNotNone(launch)
            self.assertEqual(launch["name"], "b")
            self.assertEqual(queue.entries[0]["status"], "failed")
            self.assertEqual(queue.entries[0]["error"], "训练被手动终止")


if __name__ == "__main__":
    unittest.main()
