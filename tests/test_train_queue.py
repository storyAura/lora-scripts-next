"""Sequential training queue (mikazuki/train_queue.py): conveyor, intercept, estimates."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from mikazuki.train_queue import (
    TrainQueue,
    count_dataset_images,
    estimate_total_steps,
    parse_speed_from_lines,
)


def fake_task(status_name: str, metadata=None):
    return SimpleNamespace(status=SimpleNamespace(name=status_name), metadata=metadata or {})


def make_queue(tmp: Path, tasks=None, tail_lines=None, armed=True) -> TrainQueue:
    tasks = tasks if tasks is not None else {}
    queue = TrainQueue(
        path=tmp / "queue.json",
        tasks_source=lambda: tasks,
        log_tail=lambda task_id, limit: list(tail_lines or []),
    )
    queue._armed = armed
    queue._test_tasks = tasks  # keep a handle for mutation in tests
    return queue


BASE_CONFIG = {
    "model_train_type": "anima-lora",
    "lora_type": "lokr",
    "output_name": "my-char",
    "train_batch_size": 2,
    "max_train_epochs": 10,
}


class EstimateTests(unittest.TestCase):
    def test_count_dataset_images_weights_repeats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sub = root / "5_char"
            sub.mkdir()
            for i in range(3):
                (sub / f"{i}.png").write_bytes(b"x")
            (sub / "caption.txt").write_text("tag")
            (root / "no_repeat_dir").mkdir()
            (root / "no_repeat_dir" / "a.png").write_bytes(b"x")
            self.assertEqual(count_dataset_images(str(root)), 15)

    def test_count_dataset_images_missing_dir(self):
        self.assertIsNone(count_dataset_images("Z:/does/not/exist"))
        self.assertIsNone(count_dataset_images(None))

    def test_estimate_total_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            sub = Path(tmp) / "4_c"
            sub.mkdir()
            for i in range(10):
                (sub / f"{i}.jpg").write_bytes(b"x")
            config = {
                "train_data_dir": tmp,
                "train_batch_size": 4,
                "max_train_epochs": 8,
            }
            images, steps = estimate_total_steps(config)
            self.assertEqual(images, 40)
            self.assertEqual(steps, 80)  # ceil(40/4)=10 per epoch * 8
            config["max_train_steps"] = 50
            self.assertEqual(estimate_total_steps(config)[1], 50)

    def test_parse_speed(self):
        lines = [
            "steps:  10%| 10/100 [00:10<01:30, 1.05s/it]",
            "steps:  50%| 50/100 [00:30<00:25, 2.00it/s, avr_loss=0.1]",
        ]
        self.assertAlmostEqual(parse_speed_from_lines(lines), 2.0)
        self.assertAlmostEqual(parse_speed_from_lines(lines[:1]), 1 / 1.05, places=3)
        self.assertIsNone(parse_speed_from_lines(["no speed here"]))


class InterceptTests(unittest.TestCase):
    def test_disarmed_and_idle_pass_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue = make_queue(Path(tmp), armed=False)
            self.assertIsNone(queue.intercept_run(dict(BASE_CONFIG)))
            queue._armed = True
            self.assertIsNone(queue.intercept_run(dict(BASE_CONFIG)))
            self.assertEqual(queue.entries, [])

    def test_busy_enqueues_and_activates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks = {"t0": fake_task("RUNNING")}
            queue = make_queue(Path(tmp), tasks=tasks)
            response = queue.intercept_run(dict(BASE_CONFIG))
            self.assertEqual(response.status, "success")
            self.assertTrue(response.data["queued"])
            self.assertIn("加入训练队列", response.data["queue_message"])
            self.assertTrue(queue.active)
            self.assertEqual(queue.entries[0]["status"], "queued")
            self.assertEqual(queue.entries[0]["name"], "my-char")

    def test_queue_mode_enqueues_while_idle_without_activating(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue = make_queue(Path(tmp))
            queue.set_mode(True)
            response = queue.intercept_run(dict(BASE_CONFIG))
            self.assertEqual(response.status, "success")
            self.assertFalse(queue.active)

    def test_editing_entry_receives_the_submit(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks = {"t0": fake_task("RUNNING")}
            queue = make_queue(Path(tmp), tasks=tasks)
            queue.intercept_run(dict(BASE_CONFIG))
            entry_id = queue.entries[0]["id"]
            tasks.clear()  # queue must halt for editing even when idle
            self.assertEqual(queue.set_editing(entry_id, True).status, "success")
            self.assertFalse(queue.active)

            updated = dict(BASE_CONFIG, output_name="my-char-v2")
            response = queue.intercept_run(updated)
            self.assertIn("保存修改", response.message)
            entry = queue.entries[0]
            self.assertEqual(entry["status"], "queued")
            self.assertEqual(entry["name"], "my-char-v2")
            self.assertEqual(entry["config"]["output_name"], "my-char-v2")
            self.assertFalse(queue.active, "saved edit must not auto-start the queue")

    def test_editing_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue = make_queue(Path(tmp))
            queue.set_mode(True)
            queue.intercept_run(dict(BASE_CONFIG))
            queue.intercept_run(dict(BASE_CONFIG, output_name="second"))
            first, second = queue.entries
            self.assertEqual(queue.set_editing(first["id"], True).status, "success")
            self.assertEqual(queue.set_editing(second["id"], True).status, "fail")


class ConveyorTests(unittest.TestCase):
    def _queued_queue(self, tmp, tasks, names=("a", "b")):
        queue = make_queue(Path(tmp), tasks=tasks, tail_lines=["50/50 [00:25<00:00, 2.00it/s]"])
        queue.set_mode(True)
        for name in names:
            queue.intercept_run(dict(BASE_CONFIG, output_name=name))
        queue.set_mode(False)
        return queue

    def test_tick_launches_next_and_marks_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks = {}
            queue = self._queued_queue(tmp, tasks)
            self.assertIsNone(queue.tick(), "inactive queue must not launch")
            queue.start()
            launch = queue.tick()
            self.assertEqual(launch["name"], "a")
            self.assertEqual(queue.entries[0]["status"], "running")
            self.assertIsNone(queue.tick(), "launch in flight counts as busy")
            queue.mark_launch_result(launch["id"], True, task_id="t1")
            self.assertEqual(queue.entries[0]["task_id"], "t1")

    def test_finished_task_records_speed_and_advances(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks = {}
            queue = self._queued_queue(tmp, tasks)
            queue.start()
            launch = queue.tick()
            queue.mark_launch_result(launch["id"], True, task_id="t1")
            tasks["t1"] = fake_task("FINISHED")
            next_launch = queue.tick()
            self.assertEqual(queue.entries[0]["status"], "done")
            self.assertEqual(next_launch["name"], "b")
            self.assertAlmostEqual(queue.last_speed["it_s"], 2.0)
            self.assertEqual(queue.last_speed["lora_type"], "lokr")

    def test_failed_task_reports_error_and_queue_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks = {}
            queue = self._queued_queue(tmp, tasks)
            queue.start()
            launch = queue.tick()
            queue.mark_launch_result(launch["id"], True, task_id="t1")
            tasks["t1"] = fake_task("FAILED", metadata={"error": "CUDA out of memory"})
            next_launch = queue.tick()
            self.assertEqual(queue.entries[0]["status"], "failed")
            self.assertIn("CUDA out of memory", queue.entries[0]["error"])
            self.assertEqual(next_launch["name"], "b", "failure must not stop the conveyor")
            self.assertTrue(queue.active)

    def test_editing_halts_auto_advance(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks = {}
            queue = self._queued_queue(tmp, tasks)
            queue.start()
            launch = queue.tick()
            queue.mark_launch_result(launch["id"], True, task_id="t1")
            other = queue.entries[1]
            self.assertEqual(queue.set_editing(other["id"], True).status, "success")
            tasks["t1"] = fake_task("FINISHED")
            self.assertIsNone(queue.tick(), "editing must block the next launch")
            self.assertFalse(queue.active)
            self.assertEqual(queue.entries[0]["status"], "done")

    def test_drained_queue_deactivates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks = {}
            queue = self._queued_queue(tmp, tasks, names=("only",))
            queue.start()
            launch = queue.tick()
            queue.mark_launch_result(launch["id"], True, task_id="t1")
            tasks["t1"] = fake_task("FINISHED")
            self.assertIsNone(queue.tick())
            self.assertFalse(queue.active)

    def test_launch_failure_marks_entry_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks = {}
            queue = self._queued_queue(tmp, tasks, names=("only",))
            queue.start()
            launch = queue.tick()

            async def failing_submit(config):
                return SimpleNamespace(status="fail", message="不支持的训练类型", data=None)

            queue.set_submit(failing_submit)
            asyncio.run(queue._launch(launch))
            self.assertEqual(queue.entries[0]["status"], "failed")
            self.assertIn("不支持", queue.entries[0]["error"])

    def test_paused_entries_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks = {}
            queue = self._queued_queue(tmp, tasks)
            queue.pause(queue.entries[0]["id"])
            queue.start()
            launch = queue.tick()
            self.assertEqual(launch["name"], "b")

    def test_start_with_include_paused_resumes_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue = self._queued_queue(tmp, {})
            queue.pause(queue.entries[0]["id"])
            queue.start(include_paused=True)
            self.assertEqual(queue.entries[0]["status"], "queued")


class EntryOpTests(unittest.TestCase):
    def _one_entry_queue(self, tmp, tasks=None):
        queue = make_queue(Path(tmp), tasks=tasks if tasks is not None else {})
        queue.set_mode(True)
        queue.intercept_run(dict(BASE_CONFIG))
        queue.set_mode(False)
        return queue

    def test_running_entry_cannot_be_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue = self._one_entry_queue(tmp)
            queue.start()
            queue.tick()
            self.assertEqual(queue.remove(queue.entries[0]["id"]).status, "fail")

    def test_reorder_keeps_unknown_ids_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue = make_queue(Path(tmp))
            queue.set_mode(True)
            for name in ("a", "b", "c"):
                queue.intercept_run(dict(BASE_CONFIG, output_name=name))
            ids = [e["id"] for e in queue.entries]
            queue.reorder([ids[2], "bogus", ids[0]])
            self.assertEqual([e["name"] for e in queue.entries], ["c", "a", "b"])

    def test_start_entry_moves_to_front(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue = make_queue(Path(tmp))
            queue.set_mode(True)
            for name in ("a", "b"):
                queue.intercept_run(dict(BASE_CONFIG, output_name=name))
            response = queue.start_entry(queue.entries[1]["id"])
            self.assertEqual(response.status, "success")
            self.assertEqual(queue.entries[0]["name"], "b")
            self.assertTrue(queue.active)

    def test_start_entry_rejected_while_busy(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks = {"t0": fake_task("RUNNING")}
            queue = self._one_entry_queue(tmp, tasks=tasks)
            self.assertEqual(queue.start_entry(queue.entries[0]["id"]).status, "fail")

    def test_clear_finished(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue = self._one_entry_queue(tmp)
            queue.entries[0]["status"] = "failed"
            queue.clear_finished()
            self.assertEqual(queue.entries, [])


class PersistenceTests(unittest.TestCase):
    def test_restart_marks_running_as_failed_and_deactivates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "queue.json"
            first = TrainQueue(path=path, tasks_source=dict, log_tail=lambda *a: [])
            first._armed = True
            first.set_mode(True)
            first.intercept_run(dict(BASE_CONFIG))
            first.start()
            first.tick()  # entry now running
            self.assertEqual(first.entries[0]["status"], "running")

            second = TrainQueue(path=path, tasks_source=dict, log_tail=lambda *a: [])
            snapshot = second.snapshot()
            self.assertFalse(snapshot["active"])
            self.assertEqual(snapshot["entries"][0]["status"], "failed")
            self.assertIn("服务重启", snapshot["entries"][0]["error"])
            self.assertTrue(snapshot["queue_mode"], "queue_mode survives restart")

    def test_corrupt_state_starts_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "queue.json"
            path.write_text("{not json", encoding="utf-8")
            queue = TrainQueue(path=path, tasks_source=dict, log_tail=lambda *a: [])
            self.assertEqual(queue.snapshot()["entries"], [])

    def test_snapshot_hides_config_and_computes_eta(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue = make_queue(Path(tmp))
            queue.set_mode(True)
            queue.intercept_run(dict(BASE_CONFIG))
            queue.entries[0]["steps"] = 600
            queue.last_speed = {"it_s": 2.0, "name": "prev", "lora_type": "lora"}
            view = queue.snapshot()["entries"][0]
            self.assertNotIn("config", view)
            self.assertEqual(view["eta_seconds"], 300)

    def test_state_file_is_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            queue = make_queue(Path(tmp))
            queue.set_mode(True)
            queue.intercept_run(dict(BASE_CONFIG))
            raw = json.loads((Path(tmp) / "queue.json").read_text(encoding="utf-8"))
            self.assertEqual(raw["entries"][0]["name"], "my-char")


if __name__ == "__main__":
    unittest.main()
