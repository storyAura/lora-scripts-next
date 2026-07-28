from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TrainingSchemaSafetyTests(unittest.TestCase):
    def test_training_schemas_do_not_offer_sageattention(self):
        schema_paths = (
            PROJECT_ROOT / "mikazuki" / "schema" / "sd3-lora.ts",
            PROJECT_ROOT / "mikazuki" / "schema" / "anima-lora-fast.ts",
            PROJECT_ROOT / "mikazuki" / "schema" / "anima-finetune.ts",
        )

        for schema_path in schema_paths:
            with self.subTest(schema=schema_path.name):
                schema = schema_path.read_text(encoding="utf-8")
                self.assertNotIn('"sageattn"', schema)


if __name__ == "__main__":
    unittest.main()
