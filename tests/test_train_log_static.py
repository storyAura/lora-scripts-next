"""Static guards for the /train-log viewer page (mikazuki/static/train_log.html).

2026-08-02 regressions this pins down:
- tqdm stats only accepted ``it/s``; sub-1it/s training prints ``s/it`` so the
  training bar never parsed and the *Sampling* bar (fast, it/s) hijacked the
  进度/ETA/速度 tiles.
- ``avr_loss`` sits behind extra postfix segments (``Average key norm=…``) and
  the old regex only looked in the first segment → Loss tile stayed "—".
- the LR tile was dead weight: kohya progress postfix never contains lr.
- the page shipped a violet dark theme + old "Next Trainer" brand instead of
  the site-wide cream-coffee palette.
"""

import unittest
from pathlib import Path

PAGE = Path("mikazuki/static/train_log.html")


class TrainLogViewerStaticTests(unittest.TestCase):
    def setUp(self):
        self.html = PAGE.read_text(encoding="utf-8")

    def test_parser_accepts_both_tqdm_speed_units(self):
        self.assertIn("(it\\/s|s\\/it)", self.html)
        # the old it/s-only stat regex must be gone
        self.assertNotIn("it\\/s(?:[^,]*,\\s*avr_loss=", self.html)

    def test_avr_loss_parsed_anywhere_in_line(self):
        self.assertIn("RX_LOSS", self.html)
        self.assertIn("avr_loss\\s*=", self.html)

    def test_sampling_bar_cannot_hijack_training_stats(self):
        # bar attribution state machine: stats tiles only update for steps bars
        self.assertIn('barKind === "steps"', self.html)
        self.assertIn('barKind = "sampling"', self.html)
        # unknown bare bars (cache latents etc.) are ignored until a steps bar
        self.assertIn("let barKind = null", self.html)

    def test_dead_lr_tile_removed(self):
        self.assertNotIn('id="s-lr"', self.html)
        self.assertNotIn("RX_LR", self.html)

    def test_speed_unit_follows_trainer_output(self):
        self.assertIn('id="s-speed-unit"', self.html)

    def test_epoch_incremented_matches_kohya_key_value_format(self):
        # kohya logs "epoch is incremented. current_epoch: 2, epoch: 3"
        self.assertIn("RX_EPOCH_INC_KV", self.html)

    def test_brand_palette_and_name(self):
        self.assertIn("Next Story Trainer", self.html)
        self.assertNotIn("— Next Trainer", self.html)
        # cream-coffee palette; violet dark theme removed
        self.assertIn("#faf8f1", self.html)
        self.assertIn("#574d38", self.html)
        self.assertNotIn("#0b0d12", self.html)
        self.assertNotIn("#a78bfa", self.html)
        self.assertNotIn("--pink", self.html)


if __name__ == "__main__":
    unittest.main()
