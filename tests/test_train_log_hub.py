from __future__ import annotations

import unittest

import mikazuki.train_log_hub as train_log_hub_module
from mikazuki.train_log_hub import TrainLogHub, strip_ansi


class TrainLogHubAnsiTests(unittest.TestCase):
    def test_strip_ansi_removes_colors_progress_and_hyperlinks(self):
        raw = (
            "\x1b[2;36m2026-06-05 15:21:06\x1b[0m "
            "\x1b[34mINFO\x1b[0m "
            "\x1b]8;;file:///tmp/train.py\x1b\\train.py\x1b]8;;\x1b\\ "
            "steps: 0%|\x1b[34m \x1b[0m| 0/10"
        )

        self.assertEqual(
            strip_ansi(raw),
            "2026-06-05 15:21:06 INFO train.py steps: 0%| | 0/10",
        )

    def test_append_line_buffers_sanitized_text(self):
        hub = TrainLogHub()
        hub.start_task("task-ansi")

        hub.append_line("task-ansi", "\x1b[34mINFO\x1b[0m running training\r\n")
        hub.append_line("task-ansi", "\x1b]8;;file:///tmp/a.py\x1b\\a.py\x1b]8;;\x1b\\\n")

        lines, total, done = hub.snapshot_from("task-ansi", 0)

        self.assertEqual(lines, ["INFO running training", "a.py"])
        self.assertEqual(total, 2)
        self.assertFalse(done)

    def test_tail_returns_recent_sanitized_lines(self):
        hub = TrainLogHub()
        hub.start_task("task-tail")

        hub.append_line("task-tail", "\x1b[31mfirst\x1b[0m\n")
        hub.append_line("task-tail", "second\n")
        hub.append_line("task-tail", "third\n")

        self.assertEqual(hub.tail("task-tail", 2), ["second", "third"])
        self.assertEqual(hub.tail("missing", 2), [])


class TrainLogHubRingBufferTests(unittest.TestCase):
    """The ring buffer must keep streaming (and stay bounded) after it wraps."""

    def setUp(self):
        self._old_max = train_log_hub_module._MAX_LINES
        train_log_hub_module._MAX_LINES = 5
        self.addCleanup(setattr, train_log_hub_module, "_MAX_LINES", self._old_max)

    def test_snapshot_keeps_streaming_after_ring_wrap(self):
        hub = TrainLogHub()
        hub.start_task("wrap")
        for i in range(12):
            hub.append_line("wrap", f"line-{i}\n")

        lines, total, done = hub.snapshot_from("wrap", 0)
        self.assertEqual(total, 12)
        self.assertEqual(lines, [f"line-{i}" for i in range(7, 12)])
        self.assertFalse(done)

        # A caught-up cursor (start_idx == total) must yield fresh lines after
        # the wrap instead of returning [] forever (the old SSE freeze bug).
        hub.append_line("wrap", "line-12\n")
        lines, total, _ = hub.snapshot_from("wrap", 12)
        self.assertEqual(lines, ["line-12"])
        self.assertEqual(total, 13)

        lines, _, _ = hub.snapshot_from("wrap", 13)
        self.assertEqual(lines, [])

    def test_snapshot_events_keep_streaming_after_ring_wrap(self):
        hub = TrainLogHub()
        hub.start_task("wrap-ev")
        for i in range(8):
            hub.append_event("wrap-ev", {"step": i})

        events, total, _ = hub.snapshot_events_from("wrap-ev", 8)
        self.assertEqual(events, [])
        self.assertEqual(total, 8)

        hub.append_event("wrap-ev", {"step": 8})
        events, total, _ = hub.snapshot_events_from("wrap-ev", 8)
        self.assertEqual([event["step"] for event in events], [8])
        self.assertEqual(total, 9)

    def test_drop_task_frees_buffers(self):
        hub = TrainLogHub()
        hub.start_task("gone")
        hub.append_line("gone", "x\n")
        hub.append_event("gone", {"step": 0})
        hub.mark_done("gone")

        hub.drop_task("gone")

        self.assertEqual(hub.snapshot_from("gone", 0), ([], 0, False))
        self.assertEqual(hub.snapshot_events_from("gone", 0), ([], 0, False))
        self.assertEqual(hub.tail("gone"), [])


if __name__ == "__main__":
    unittest.main()
