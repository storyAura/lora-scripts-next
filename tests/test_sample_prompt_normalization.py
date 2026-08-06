import sys
import tempfile
import unittest
from pathlib import Path

from mikazuki.utils import train_utils

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "sd-scripts"))
from library.train_util import load_prompts  # noqa: E402


class SamplePromptNormalizationTests(unittest.TestCase):
    def test_build_sample_prompt_line_is_single_line(self):
        line = train_utils.build_sample_prompt_line(
            "@71style, 1boy silhouette\n",
            "nsfw, explicit\n",
            width=1024,
            height=1024,
            cfg=4.5,
            steps=40,
            seed=42,
        )
        self.assertNotIn("\n", line)
        self.assertIn("--n nsfw, explicit", line)
        self.assertIn("@71style, 1boy silhouette", line)

    def test_build_sample_prompt_file_content_keeps_multiline_prompts(self):
        content = train_utils.build_sample_prompt_file_content(
            "prompt one\nprompt two\n# ignored\n",
            "bad quality",
            width=832,
            height=1024,
            cfg=4,
            steps=20,
            seed=42,
            sampler="euler",
            scheduler="simple",
        )
        lines = content.split("\n")
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("prompt one "))
        self.assertTrue(lines[1].startswith("prompt two "))
        self.assertIn("--d 42", lines[0])
        self.assertIn("--d 43", lines[1])
        prompts = []
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
            handle.write(content + "\n")
            path = handle.name
        prompts = load_prompts(path)
        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts[0]["prompt"], "prompt one")
        self.assertEqual(prompts[1]["prompt"], "prompt two")

    def test_normalize_sample_prompt_file_content_merges_broken_multiline(self):
        content = (
            "@71style, 1boy silhouette, flying on a sword\n"
            " --n nsfw, explicit, sexual content --w 1024 --h 1024 --l 4.5 --s 40 --d 42\n"
        )
        normalized = train_utils.normalize_sample_prompt_file_content(content)
        self.assertNotIn("\n", normalized)
        self.assertTrue(normalized.startswith("@71style"))
        self.assertIn("--n nsfw, explicit, sexual content", normalized)

    def test_load_prompts_sees_one_prompt_after_normalization(self):
        content = (
            "@71style, 1boy silhouette, flying on a sword\n"
            " --n nsfw, explicit, sexual content --w 1024 --h 1024 --l 4.5 --s 40 --d 42\n"
        )
        normalized = train_utils.normalize_sample_prompt_file_content(content)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
            handle.write(normalized + "\n")
            path = handle.name
        prompts = load_prompts(path)
        self.assertEqual(len(prompts), 1)
        self.assertIn("silhouette", prompts[0]["prompt"])
        self.assertIn("nsfw", prompts[0]["negative_prompt"])

    def test_intentional_multi_prompt_file_is_preserved(self):
        content = "\n".join([
            "prompt one --n bad --w 512 --h 512 --l 4 --s 20 --d 1",
            "prompt two --n bad --w 512 --h 512 --l 4 --s 20 --d 2",
        ])
        normalized = train_utils.normalize_sample_prompt_file_content(content)
        self.assertEqual(normalized.count("\n"), 1)


if __name__ == "__main__":
    unittest.main()
