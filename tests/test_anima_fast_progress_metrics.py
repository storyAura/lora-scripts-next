from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from mikazuki.anima_fast_backend import progress as progress_module
from mikazuki.anima_fast_backend.progress import (
    JsonlEventReader,
    merge_anima_training_metrics,
    metrics_from_anima_events,
)


class AnimaFastProgressMetricsTests(unittest.TestCase):
    def test_incremental_reader_reads_only_appended_complete_lines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "progress.jsonl"
            path.write_text(
                json.dumps({"ev": "run_start", "total_steps": 10}) + "\n",
                encoding="utf-8",
            )
            reader = JsonlEventReader(8)

            with mock.patch.object(
                progress_module,
                "_read_from_offset",
                wraps=progress_module._read_from_offset,
            ) as read_from_offset:
                first = reader.read(path)
                initial_size = path.stat().st_size
                path.write_bytes(
                    path.read_bytes()
                    + b'{"ev":"step","global_step":1'
                )
                second = reader.read(path)
                path.write_bytes(path.read_bytes() + b"}\n")
                third = reader.read(path)

            self.assertEqual([event["ev"] for event in first], ["run_start"])
            self.assertEqual(second, first)
            self.assertEqual(third[-1]["global_step"], 1)
            self.assertEqual(read_from_offset.call_args_list[1].args[1], initial_size)

    def test_incremental_reader_resets_on_truncate_and_new_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "progress.jsonl"
            reader = JsonlEventReader(4)
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"ev": "run_start", "run_id": "first"}),
                        json.dumps({"ev": "step", "global_step": 1}),
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            self.assertEqual(reader.read(path)[-1]["global_step"], 1)

            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"ev": "run_start", "run_id": "second"}) + "\n")
                handle.write(json.dumps({"ev": "step", "global_step": 2}) + "\n")
            appended_run = reader.read(path)
            self.assertEqual(appended_run[0]["run_id"], "second")
            self.assertEqual(appended_run[-1]["global_step"], 2)

            path.write_text(
                json.dumps({"ev": "run_start", "run_id": "third"}) + "\n",
                encoding="utf-8",
            )
            truncated_run = reader.read(path)
            self.assertEqual(truncated_run, [{"ev": "run_start", "run_id": "third"}])

    def test_incremental_reader_bounds_retained_events(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "progress.jsonl"
            path.write_text(
                "".join(
                    json.dumps({"ev": "step", "global_step": step}) + "\n"
                    for step in range(10)
                ),
                encoding="utf-8",
            )
            events = JsonlEventReader(3).read(path)

        self.assertEqual(
            [event["global_step"] for event in events],
            [7, 8, 9],
        )

    def test_step_event_reads_avr_loss_and_epoch(self):
        events = [
            {"ev": "run_start", "total_steps": 20, "total_epochs": 2, "ts": 0},
            {"ev": "step", "global_step": 5, "epoch": 1, "avr_loss": 0.0886, "ts": 50},
        ]
        metrics = metrics_from_anima_events(events)
        self.assertEqual(metrics["step"], 5)
        self.assertEqual(metrics["total_steps"], 20)
        self.assertEqual(metrics["epoch"], "1/2")
        self.assertEqual(metrics["loss"], "0.0886")
        self.assertEqual(metrics["elapsed"], "50秒")
        self.assertIn("eta", metrics)

    def test_step_event_reads_loss_average_key(self):
        events = [
            {"ev": "run_start", "total_steps": 10, "total_epochs": 1},
            {"ev": "step", "global_step": 2, "epoch": 1, "loss/average": 0.12, "ts": 10},
        ]
        metrics = metrics_from_anima_events(events)
        self.assertEqual(metrics["loss"], "0.12")

    def test_merge_prefers_jsonl_loss_when_stdout_missing(self):
        stdout = {"step": 0, "total_steps": 20, "epoch": "1/2", "eta": "?"}
        anima = {
            "step": 5,
            "total_steps": 20,
            "loss": "0.09",
            "epoch": "1/2",
            "elapsed": "1分00秒",
            "eta": "3分00秒",
            "progress_source": "anima_progress_jsonl",
        }
        merged = merge_anima_training_metrics(stdout, anima)
        self.assertEqual(merged["step"], 5)
        self.assertEqual(merged["loss"], "0.09")
        self.assertEqual(merged["eta"], "3分00秒")

    def test_merge_keeps_stdout_eta_when_jsonl_missing(self):
        stdout = {"step": 3, "total_steps": 20, "eta": "02:30", "loss": "0.1"}
        anima = {"step": 3, "total_steps": 20, "loss": "0.11", "progress_source": "anima_progress_jsonl"}
        merged = merge_anima_training_metrics(stdout, anima)
        self.assertEqual(merged["eta"], "02:30")
        self.assertEqual(merged["loss"], "0.11")


if __name__ == "__main__":
    unittest.main()
