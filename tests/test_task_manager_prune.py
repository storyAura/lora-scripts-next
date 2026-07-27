from __future__ import annotations

import unittest

from mikazuki.tasks import TaskManager, TaskStatus, _MAX_RETAINED_TERMINAL_TASKS
from mikazuki.train_log_hub import hub


class TaskManagerPruneTests(unittest.TestCase):
    """Finished tasks (and their log buffers) must stay bounded across long sessions."""

    def test_terminal_tasks_and_log_buffers_are_bounded(self):
        tm = TaskManager()
        total = _MAX_RETAINED_TERMINAL_TASKS + 5
        try:
            for i in range(total):
                task = tm.create_task(["echo", str(i)], environ={}, task_id=f"prune-{i}")
                self.assertIsNotNone(task)
                hub.start_task(task.task_id)
                hub.append_line(task.task_id, f"log-{i}\n")
                hub.mark_done(task.task_id)
                task.status = TaskStatus.FINISHED

            newest = tm.create_task(["echo", "new"], environ={}, task_id="prune-live")
            self.assertIsNotNone(newest)

            terminal = [t for t in tm.tasks.values() if t.status == TaskStatus.FINISHED]
            self.assertLessEqual(len(terminal), _MAX_RETAINED_TERMINAL_TASKS)

            self.assertNotIn("prune-0", tm.tasks)
            self.assertEqual(hub.tail("prune-0"), [])

            last = f"prune-{total - 1}"
            self.assertIn(last, tm.tasks)
            self.assertEqual(hub.tail(last), [f"log-{total - 1}"])
        finally:
            for task_id in list(tm.tasks):
                hub.drop_task(task_id)

    def test_running_tasks_are_never_pruned(self):
        tm = TaskManager()
        try:
            running = tm.create_task(["echo", "run"], environ={}, task_id="keep-running")
            self.assertIsNotNone(running)
            running.status = TaskStatus.RUNNING

            tm.max_concurrent = 2
            for i in range(_MAX_RETAINED_TERMINAL_TASKS + 3):
                task = tm.create_task(["echo", str(i)], environ={}, task_id=f"fin-{i}")
                self.assertIsNotNone(task)
                task.status = TaskStatus.FINISHED

            self.assertIn("keep-running", tm.tasks)
            self.assertEqual(tm.tasks["keep-running"].status, TaskStatus.RUNNING)
        finally:
            for task_id in list(tm.tasks):
                hub.drop_task(task_id)


if __name__ == "__main__":
    unittest.main()
