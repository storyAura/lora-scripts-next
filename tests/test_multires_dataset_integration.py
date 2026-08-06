# -*- coding: utf-8 -*-
"""``multires_per_image`` inside the real sd-scripts dataset.

Builds a DreamBoothDataset the way ``generate_dataset_group_by_blueprint``
does (keyword args from the blueprint params) and runs the real
``make_buckets()``, asserting:

- every source image produces one sample per tier, in one epoch
- tier samples keep the original ``image_size`` so all tiers share one latent
  npz path (the Anima strategy keys latents by latent resolution)
- the epoch's batch indices cover every sample even when ``batch_size`` is
  larger than a tier bucket
- the config plumbing (blueprint params + user-config schema) accepts the knobs
"""
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "sd-scripts"))

from PIL import Image

from library import config_util, train_util
from library.strategy_anima import AnimaLatentsCachingStrategy

TIERS = "512,1024"
IMAGES = {"a": (1024, 768), "b": (768, 1024)}


def build_dataset(image_dir: Path, *, batch_size: int, multires: bool, target_res=TIERS):
    subset_params = config_util.DreamBoothSubsetParams(
        image_dir=str(image_dir),
        num_repeats=1,
        caption_extension=".txt",
        caption_separator=",",
        keep_tokens_separator=None,
    )
    subset = train_util.DreamBoothSubset(**asdict(subset_params))
    dataset_params = config_util.DreamBoothDatasetParams(
        batch_size=batch_size,
        resolution=(1024, 1024),
        enable_bucket=True,
        min_bucket_reso=256,
        max_bucket_reso=2048,
        bucket_reso_steps=64,
        bucket_no_upscale=False,
        multires_per_image=multires,
        target_res=target_res,
    )
    return train_util.DreamBoothDataset(
        subsets=[subset], is_training_dataset=True, **asdict(dataset_params)
    )


class MultiresDatasetTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.image_dir = Path(self._tmp.name) / "1_aki"
        self.image_dir.mkdir(parents=True)
        for name, size in IMAGES.items():
            Image.new("RGB", size, "white").save(self.image_dir / f"{name}.png")
            (self.image_dir / f"{name}.txt").write_text("aki", encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def test_make_buckets_expands_every_image_across_tiers(self):
        dataset = build_dataset(self.image_dir, batch_size=1, multires=True)
        self.assertEqual(len(dataset.image_data), len(IMAGES))

        dataset.make_buckets()

        self.assertEqual(len(dataset.image_data), len(IMAGES) * 2)
        self.assertEqual(dataset.num_train_images, len(IMAGES) * 2)
        # two tiers x two aspect ratios -> four distinct shape buckets
        self.assertEqual(len(dataset.bucket_manager.resos), 4)
        for reso in dataset.bucket_manager.resos:
            self.assertEqual(reso[0] % 16, 0)
            self.assertEqual(reso[1] % 16, 0)

        per_source = {}
        for info in dataset.image_data.values():
            per_source.setdefault(info.absolute_path, []).append(info)
        self.assertEqual(len(per_source), len(IMAGES))
        for infos in per_source.values():
            self.assertEqual(len(infos), 2)
            self.assertEqual(len({tuple(i.bucket_reso) for i in infos}), 2)
            # one npz per source image: the strategy keys latents by resolution
            npz_paths = {
                AnimaLatentsCachingStrategy(True, 1, False).get_latents_npz_path(
                    info.absolute_path, info.image_size
                )
                for info in infos
            }
            self.assertEqual(len(npz_paths), 1)
            for info in infos:
                self.assertGreaterEqual(info.resized_size[0], info.bucket_reso[0])
                self.assertGreaterEqual(info.resized_size[1], info.bucket_reso[1])

    def test_epoch_covers_every_sample_with_large_batch_size(self):
        dataset = build_dataset(self.image_dir, batch_size=4, multires=True)
        dataset.make_buckets()

        covered = set()
        for index in dataset.buckets_indices:
            bucket = dataset.bucket_manager.buckets[index.bucket_index]
            start = index.batch_index * index.bucket_batch_size
            covered.update(bucket[start : start + index.bucket_batch_size])
        self.assertEqual(covered, set(dataset.image_data))

    def test_disabled_switch_keeps_arb_bucketing(self):
        dataset = build_dataset(self.image_dir, batch_size=1, multires=False, target_res=None)
        dataset.make_buckets()

        self.assertEqual(len(dataset.image_data), len(IMAGES))
        self.assertFalse(dataset.multires_per_image)
        self.assertIsNone(dataset.multires_target_res)

    def test_single_tier_is_refused_at_dataset_construction(self):
        with self.assertRaises(ValueError):
            build_dataset(self.image_dir, batch_size=1, multires=True, target_res="1024")


class ConfigPlumbingTests(unittest.TestCase):
    def test_blueprint_params_carry_the_knobs(self):
        params = asdict(config_util.DreamBoothDatasetParams())
        self.assertIn("multires_per_image", params)
        self.assertIn("target_res", params)

    def test_user_config_schema_accepts_the_knobs(self):
        sanitizer = config_util.ConfigSanitizer(True, True, False, True)
        sanitized = sanitizer.sanitize_user_config(
            {
                "datasets": [
                    {
                        "multires_per_image": True,
                        "target_res": TIERS,
                        "subsets": [{"image_dir": "train/aki"}],
                    }
                ]
            }
        )
        dataset = sanitized["datasets"][0]
        self.assertIs(dataset["multires_per_image"], True)
        self.assertEqual(dataset["target_res"], TIERS)

    def test_dataset_arguments_expose_the_cli_flags(self):
        import argparse

        parser = argparse.ArgumentParser()
        train_util.add_dataset_arguments(parser, True, True, True)
        args = parser.parse_args(["--multires_per_image", "--target_res", TIERS])
        self.assertTrue(args.multires_per_image)
        self.assertEqual(args.target_res, TIERS)


if __name__ == "__main__":
    unittest.main()
