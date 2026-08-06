# -*- coding: utf-8 -*-
"""Same-epoch multi-resolution training (``multires_per_image``).

Covers the whole knob path:
- GUI parsing / validation of ``target_res`` (mikazuki.multires)
- pre-launch validation refusals (training_validation)
- adapter normalization into the sd-scripts TOML
- trainer-side tier planning and ImageInfo expansion (library.anima_multires)
- the epoch keeps every tier of every image (no silently dropped tier)
"""
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "sd-scripts"))

from library import anima_multires

from mikazuki import multires
from mikazuki.anima_backend.adapter import (
    MULTIRES_ARB_OVERRIDDEN_WARNING,
    adapt_anima_config,
)
from mikazuki.training_validation import (
    TrainingConfigurationError,
    validate_training_configuration,
)


class FakeImageInfo:
    """Minimal stand-in for train_util.ImageInfo (no torch import needed)."""

    def __init__(self, image_key, absolute_path, image_size, num_repeats=1, is_reg=False):
        self.image_key = image_key
        self.absolute_path = absolute_path
        self.image_size = image_size
        self.num_repeats = num_repeats
        self.is_reg = is_reg
        self.bucket_reso = None
        self.resized_size = None
        self.latents_npz = None


def base_lora_config(**overrides):
    config = {
        "model_train_type": "anima-lora",
        "lora_type": "lora",
        "train_data_dir": "./train/aki",
        "resolution": "1024,1024",
        "enable_bucket": True,
        "network_module": "networks.lora_anima",
        "learning_rate": "1e-4",
    }
    config.update(overrides)
    return config


class TargetResParsingTests(unittest.TestCase):
    def test_accepts_comma_string_list_and_fullwidth_comma(self):
        self.assertEqual(multires.normalize_target_res("512,1024"), [512, 1024])
        self.assertEqual(multires.normalize_target_res(" 512 , 1024 "), [512, 1024])
        self.assertEqual(multires.normalize_target_res("512，1024"), [512, 1024])
        self.assertEqual(multires.normalize_target_res([512, "1024"]), [512, 1024])
        self.assertEqual(multires.normalize_target_res(""), [])
        self.assertEqual(multires.normalize_target_res(None), [])

    def test_rejects_non_integer_items(self):
        with self.assertRaises(ValueError):
            multires.normalize_target_res("512,large")

    def test_validate_requires_two_allowed_tiers(self):
        self.assertEqual(multires.validate_target_res("512,1024"), (512, 1024))
        # duplicates collapse, and a single distinct tier is not multi-resolution
        with self.assertRaises(ValueError):
            multires.validate_target_res("1024,1024")
        with self.assertRaises(ValueError):
            multires.validate_target_res("1024")
        with self.assertRaises(ValueError):
            multires.validate_target_res("512,999")

    def test_allowed_tiers_match_the_documented_list(self):
        self.assertEqual(
            multires.allowed_target_res(), multires.DOCUMENTED_ALLOWED_TARGET_RES
        )

    def test_format_target_res(self):
        self.assertEqual(multires.format_target_res((512, 1024)), "512,1024")


class ValidationTests(unittest.TestCase):
    def test_valid_multires_config_passes(self):
        validate_training_configuration(
            base_lora_config(multires_per_image=True, target_res="512,1024"),
            "anima-lora",
        )

    def test_single_tier_is_refused(self):
        with self.assertRaises(TrainingConfigurationError) as ctx:
            validate_training_configuration(
                base_lora_config(multires_per_image=True, target_res="1024"),
                "anima-lora",
            )
        self.assertEqual(ctx.exception.field, "target_res")

    def test_unknown_tier_is_refused(self):
        with self.assertRaises(TrainingConfigurationError) as ctx:
            validate_training_configuration(
                base_lora_config(multires_per_image=True, target_res="640,1024"),
                "anima-lora",
            )
        self.assertEqual(ctx.exception.field, "target_res")

    def test_non_anima_train_type_is_refused(self):
        with self.assertRaises(TrainingConfigurationError) as ctx:
            validate_training_configuration(
                base_lora_config(multires_per_image=True, target_res="512,1024"),
                "sdxl-lora",
            )
        self.assertEqual(ctx.exception.field, "multires_per_image")

    def test_random_crop_is_refused(self):
        with self.assertRaises(TrainingConfigurationError) as ctx:
            validate_training_configuration(
                base_lora_config(
                    multires_per_image=True, target_res="512,1024", random_crop=True
                ),
                "anima-lora",
            )
        self.assertEqual(ctx.exception.field, "random_crop")

    def test_disabled_switch_ignores_target_res(self):
        validate_training_configuration(
            base_lora_config(multires_per_image=False, target_res="1024"),
            "anima-lora",
        )


class AdapterTests(unittest.TestCase):
    def test_enabled_config_stamps_normalized_tier_string(self):
        adapted, warnings = adapt_anima_config(
            base_lora_config(multires_per_image=True, target_res=[1024, 512, 1024])
        )
        self.assertIs(adapted["multires_per_image"], True)
        self.assertEqual(adapted["target_res"], "1024,512")
        self.assertIn(MULTIRES_ARB_OVERRIDDEN_WARNING, warnings)

    def test_checkbox_string_tiers_are_accepted(self):
        adapted, _warnings = adapt_anima_config(
            base_lora_config(
                multires_per_image=True, target_res=["512", "1024", "768"]
            )
        )
        self.assertEqual(adapted["target_res"], "512,1024,768")

    def test_disabled_config_drops_both_knobs(self):
        adapted, warnings = adapt_anima_config(
            base_lora_config(multires_per_image=False, target_res="512,1024")
        )
        self.assertNotIn("multires_per_image", adapted)
        self.assertNotIn("target_res", adapted)
        self.assertNotIn(MULTIRES_ARB_OVERRIDDEN_WARNING, warnings)

    def test_finetune_path_supports_multires(self):
        adapted, _warnings = adapt_anima_config(
            {
                "model_train_type": "anima-finetune",
                "train_data_dir": "./train/aki",
                "resolution": "1024,1024",
                "enable_bucket": True,
                "learning_rate": "1e-5",
                "multires_per_image": True,
                "target_res": "512,1024",
            },
            finetune=True,
        )
        self.assertEqual(adapted["target_res"], "512,1024")


class TierPlanningTests(unittest.TestCase):
    def test_tier_buckets_are_patch_aligned_and_inside_the_band(self):
        from multires_training import tiers

        plan = anima_multires.plan_image_tiers((1280, 960), (512, 1024))
        self.assertEqual([edge for edge, _bucket, _resized in plan], [512, 1024])
        for edge, bucket, resized in plan:
            self.assertEqual(bucket[0] % 16, 0)
            self.assertEqual(bucket[1] % 16, 0)
            low, high = tiers.freefit_band_for_edge(edge)
            tokens = tiers.patch_token_count(*bucket)
            self.assertTrue(low <= tokens <= high, f"{edge}: {tokens} not in [{low},{high}]")
            # resized_size must cover the bucket so the center crop has pixels
            self.assertGreaterEqual(resized[0], bucket[0])
            self.assertGreaterEqual(resized[1], bucket[1])

    def test_larger_tier_yields_more_tokens(self):
        from multires_training import tiers

        plan = dict(
            (edge, bucket) for edge, bucket, _resized in
            anima_multires.plan_image_tiers((1024, 1024), (512, 1024))
        )
        self.assertLess(
            tiers.patch_token_count(*plan[512]), tiers.patch_token_count(*plan[1024])
        )

    def test_cover_resized_size_keeps_aspect_and_covers_bucket(self):
        resized = anima_multires.cover_resized_size((1000, 500), (512, 256))
        self.assertGreaterEqual(resized[0], 512)
        self.assertGreaterEqual(resized[1], 256)

    def test_shard_index_is_stable_per_source_image(self):
        first = anima_multires.shard_index("/data/a.png", 4)
        second = anima_multires.shard_index("/data/a.png", 4)
        self.assertEqual(first, second)
        self.assertEqual(anima_multires.shard_index("/data/a.png", 1), 0)
        self.assertIn(anima_multires.shard_index("/data/b.png", 4), range(4))


class ExpansionTests(unittest.TestCase):
    def _expand(self, sizes, target_res=(512, 1024)):
        image_data = {}
        image_to_subset = {}
        for index, size in enumerate(sizes):
            key = f"/data/img{index}.png"
            image_data[key] = FakeImageInfo(key, key, size)
            image_to_subset[key] = object()
        return anima_multires.expand_image_data(image_data, image_to_subset, target_res)

    def test_one_sample_per_tier_sharing_the_source_image(self):
        expanded, subsets, ar_errors = self._expand([(1024, 768)])
        self.assertEqual(len(expanded), 2)
        self.assertEqual(len(ar_errors), 2)
        self.assertEqual(set(subsets), set(expanded))
        self.assertEqual({info.absolute_path for info in expanded.values()}, {"/data/img0.png"})
        # original image size is preserved so all tiers share one latent npz path
        self.assertEqual({info.image_size for info in expanded.values()}, {(1024, 768)})
        self.assertEqual(len({info.bucket_reso for info in expanded.values()}), 2)
        for key, info in expanded.items():
            self.assertEqual(key, info.image_key)
            self.assertEqual(anima_multires.source_image_key(key), "/data/img0.png")

    def test_three_tiers_expand_every_image(self):
        expanded, _subsets, _errors = self._expand(
            [(1024, 1024), (800, 1200)], target_res=(512, 768, 1024)
        )
        self.assertEqual(len(expanded), 6)

    def test_prebaked_latents_are_refused(self):
        image_data = {"/data/a.png": FakeImageInfo("/data/a.png", "/data/a.png", (512, 512))}
        image_data["/data/a.png"].latents_npz = "/data/a_0512x0512_anima.npz"
        with self.assertRaises(ValueError):
            anima_multires.expand_image_data(image_data, {}, (512, 1024))

    def test_unknown_image_size_is_refused(self):
        info = FakeImageInfo("/data/a.png", "/data/a.png", None)
        with self.assertRaises(ValueError):
            anima_multires.expand_image_data({"/data/a.png": info}, {}, (512, 1024))

    def test_epoch_batches_keep_every_tier(self):
        """batch_size larger than a tier bucket must not drop that tier."""
        from multires_training import build_shape_buckets

        expanded, _subsets, _errors = self._expand([(1024, 768)])
        epoch = build_shape_buckets(
            [(key, tuple(info.bucket_reso)) for key, info in expanded.items()],
            batch_size=4,
            keep_incomplete_batches=True,
        )
        self.assertEqual(epoch.all_keys_in_epoch(), set(expanded))

    def test_token_budget_comes_from_real_bucket_shapes(self):
        expanded, _subsets, _errors = self._expand([(1024, 768), (768, 1024)])
        resos = anima_multires.bucket_resolutions(expanded)
        low, high, counts = anima_multires.derive_token_budget(resos)
        self.assertLessEqual(low, high)
        self.assertTrue(counts)


if __name__ == "__main__":
    unittest.main()
