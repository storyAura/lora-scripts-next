# -*- coding: utf-8 -*-
"""The vendored LyCORIS must stay in sync with the installed one.

`pip install lycoris-lora` provides upstream LyCORIS, which has none of the local
extension algos (glokr / tglokr / bokr / bora / gsokr / glora_boft) nor the
Anima-specific fixes. vendor/lycoris holds the patched package so it travels with
the repo; scripts/sync_vendored_lycoris.py copies it over the installed one.

Editing only the venv copy used to silently lose the change on the next install,
so this test fails when the two drift apart.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sync_vendored_lycoris as sync


class VendoredLycorisTests(unittest.TestCase):
    def test_vendored_package_is_present_and_complete(self):
        self.assertTrue(sync.VENDORED.is_dir(), "vendor/lycoris is missing")
        files = {p.relative_to(sync.VENDORED).as_posix() for p in sync.vendored_files()}
        for required in (
            "__init__.py",
            "kohya.py",
            "wrapper.py",
            "modules/__init__.py",
            "modules/glokr.py",
            "modules/bokr.py",
            "modules/bora.py",
            "modules/gsokr.py",
            "modules/glora_boft.py",
            "modules/norms.py",
        ):
            self.assertIn(required, files, f"vendored lycoris is missing {required}")

    def test_vendored_package_carries_the_local_extensions(self):
        source = (sync.VENDORED / "kohya.py").read_text(encoding="utf-8")
        self.assertIn("extra_algo_kwargs", source, "kohya.py lost the extension kwargs forwarding")
        self.assertIn("set_current_timestep", source, "kohya.py lost the timestep hook")

        glokr = (sync.VENDORED / "modules" / "glokr.py").read_text(encoding="utf-8")
        self.assertIn("train_time_gates", glokr, "glokr.py lost the T-GLoKR time gates")

    def test_installed_lycoris_matches_the_vendored_copy(self):
        target = sync.installed_lycoris_dir()
        if target is None:
            self.skipTest("lycoris is not installed in this interpreter")
        drifted = sync.compare(target)
        self.assertEqual(
            drifted,
            [],
            "installed lycoris differs from vendor/lycoris — run "
            "`python scripts/sync_vendored_lycoris.py` (and commit the vendored change "
            "if you edited the venv copy)",
        )


if __name__ == "__main__":
    unittest.main()
