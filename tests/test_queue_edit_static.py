"""Static guards for the queue 「编辑」 config handover (sd-trainer-queue.js).

The entry config stored by the queue is the flat POSTed /api/run body
(string LRs already parseFloat'ed to numbers, branch fields folded into
network_args). Writing it verbatim into ``configs-*-autosave`` blanked the
form on every page load (2026-08-02 incident). The only correct handover is
``sessionStorage["mikazuki-pending-import"]`` — the page layout runs it
through /api/config/validate-import (branch consts, network_args hydration)
and merges schema defaults on mount.
"""

import unittest
from pathlib import Path

QUEUE_JS = Path("frontend/dist/assets/sd-trainer-queue.js")


class QueueEditHandoverStaticTests(unittest.TestCase):
    def setUp(self):
        self.js = QUEUE_JS.read_text(encoding="utf-8")

    def test_edit_hands_config_over_via_pending_import(self):
        self.assertIn('sessionStorage.setItem("mikazuki-pending-import"', self.js)

    def test_edit_never_writes_the_autosave_key(self):
        # the autosave key holds raw form snapshots only; a flat POSTed config
        # written there breaks the form (string vs number fields, deleted
        # branch fields) and gets re-applied on every page load
        self.assertNotIn("localStorage.setItem(`configs-", self.js)
        self.assertNotIn('localStorage.setItem("configs-', self.js)

    def test_dreambooth_page_path_is_the_dreambooth_route(self):
        self.assertIn('"sd-dreambooth": { path: "/dreambooth/" }', self.js)


if __name__ == "__main__":
    unittest.main()
