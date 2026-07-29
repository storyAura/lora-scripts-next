from __future__ import annotations

import os
import unittest
from unittest import mock

import torch
import torch.nn as nn

from mikazuki.anima_backend.upstream import verify_vendored_lycoris


class VendoredLycorisGuardTests(unittest.TestCase):
    def test_synced_venv_passes(self):
        # The venv must carry the vendored copy (sync-script contract);
        # the guard must accept it silently.
        verify_vendored_lycoris()

    def test_upstream_copy_fails_with_sync_hint(self):
        fake_upstream_lokr = object()  # no compute_merged_delta attribute
        with mock.patch("importlib.import_module", return_value=fake_upstream_lokr):
            with self.assertRaisesRegex(RuntimeError, "sync_vendored_lycoris"):
                verify_vendored_lycoris()

    def test_drift_escape_hatch_skips_check(self):
        fake_upstream_lokr = object()
        with mock.patch.dict(os.environ, {"ANIMA_ALLOW_LYCORIS_DRIFT": "1"}), \
                mock.patch("importlib.import_module", return_value=fake_upstream_lokr):
            verify_vendored_lycoris()


class InstalledLokrForwardNumericTests(unittest.TestCase):
    """Regression for the lycoris_patch clobber (removed 2026-07-29).

    The forward that training actually runs — imported from site-packages,
    without vendor/ sys.path injection and without runtime monkeypatches —
    must keep small bf16 updates alive in merged mode. The old
    patch_lokr_dora_bf16_forward() replaced LokrModule.forward with a bf16
    base+diff/subtract implementation that absorbed every update below the
    base weight's ULP; this test fails if any such patching comes back.
    """

    def test_small_bf16_delta_survives_real_import_path(self):
        from lycoris.modules.lokr import LokrModule

        base = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            base.weight.fill_(1.0)
        module = LokrModule(
            "guard_lokr",
            base,
            multiplier=1.0,
            lora_dim=1,
            alpha=1,
            factor=-1,
            full_matrix=True,
        )
        base.to(dtype=torch.bfloat16)

        module.get_weight = lambda shape: torch.full(
            (4, 4), 0.001, dtype=torch.float32
        )
        captured: list[torch.Tensor] = []

        def capture_op(value, weight, bias, **kwargs):
            captured.append(weight)
            return torch.zeros(
                (*value.shape[:-1], weight.shape[0]), dtype=value.dtype
            )

        module.op = capture_op
        module(torch.zeros((1, 4), dtype=torch.bfloat16))

        self.assertEqual(len(captured), 1)
        # 0.001 against a bf16 base of 1.0 is far below the base ULP (0.0078):
        # the absorbing implementation returns an all-zero delta here.
        self.assertEqual(torch.count_nonzero(captured[0]).item(), 16)


if __name__ == "__main__":
    unittest.main()
