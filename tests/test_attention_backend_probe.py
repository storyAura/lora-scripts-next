from __future__ import annotations

import unittest
from unittest import mock

from mikazuki.attention_probe import (
    AttentionEnvironmentKey,
    clear_attention_probe_cache,
    detect_best_training_attention,
    probe_training_attention_backend,
)


def environment_key(torch_version: str) -> AttentionEnvironmentKey:
    return AttentionEnvironmentKey(
        device_name="test-gpu",
        compute_capability="9.0",
        driver_version="test-driver",
        torch_version=torch_version,
        cuda_version="13.0",
        flash_attn_version="2.8.3",
        xformers_version="0.0.33",
    )


class AttentionBackendProbeTests(unittest.TestCase):
    def tearDown(self):
        clear_attention_probe_cache()

    def test_detector_uses_forward_backward_result_and_caches_by_environment(self):
        results = {
            "flash": (False, "backward failed"),
            "xformers": (True, ""),
            "torch": (True, ""),
        }

        with mock.patch(
            "mikazuki.attention_probe._environment_key",
            return_value=environment_key("2.11.0"),
        ), mock.patch(
            "mikazuki.attention_probe._probe_backend_uncached",
            side_effect=lambda backend: results[backend],
        ) as probe:
            first = detect_best_training_attention()
            second = detect_best_training_attention()

        self.assertEqual(first, "xformers")
        self.assertEqual(second, "xformers")
        self.assertEqual(
            [call.args[0] for call in probe.call_args_list],
            ["flash", "xformers"],
        )

    def test_environment_change_invalidates_cached_result(self):
        keys = [
            environment_key("2.11.0"),
            environment_key("2.12.0"),
        ]
        with mock.patch(
            "mikazuki.attention_probe._environment_key",
            side_effect=keys,
        ), mock.patch(
            "mikazuki.attention_probe._probe_backend_uncached",
            return_value=(True, ""),
        ) as probe:
            self.assertEqual(detect_best_training_attention(), "flash")
            self.assertEqual(detect_best_training_attention(), "flash")

        self.assertEqual(probe.call_count, 2)

    def test_torch_probe_executes_real_forward_and_backward(self):
        result = probe_training_attention_backend("torch")

        self.assertTrue(result.usable, result.reason)
        self.assertEqual(result.backend, "torch")

    def test_unknown_backend_fails_explicitly(self):
        with self.assertRaises(ValueError):
            probe_training_attention_backend("unknown")


if __name__ == "__main__":
    unittest.main()
