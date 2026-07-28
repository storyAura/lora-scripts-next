# -*- coding: utf-8 -*-
"""Static guards for schema field constraints that break union matching.

The browser-side schemastery validates ``step`` as "(value - min) must be an
exact multiple of step" (decimal-shifted, no epsilon).  A field whose default
or min is off the step grid rejects its own default, the whole lora_type
branch fails to match, and the branch section silently disappears from the
form (found live: delora_lambda / deft_init_scale with min(0.000001)).

Branch marker fields must also stay tolerant: a required const on anything
except the ``lora_type`` discriminator un-matches the union whenever the form
model carries values from another branch (found live: rs_lora / dora_wd).
"""
import re
import unittest
from decimal import Decimal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = PROJECT_ROOT / "mikazuki" / "schema"

NUMBER_CHAIN = re.compile(r"(\w+):\s*Schema\.number\(\)((?:\.\w+\([^()]*\))*)")
PART = re.compile(r"\.(min|max|step|default)\(([-0-9.]+)\)")


def _iter_number_fields():
    for schema_path in sorted(SCHEMA_DIR.glob("*.ts")):
        for line_no, line in enumerate(
            schema_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for field, chain in NUMBER_CHAIN.findall(line):
                parts = {name: Decimal(value) for name, value in PART.findall(chain)}
                yield f"{schema_path.name}:{line_no}", field, parts


class SchemaStepGridTests(unittest.TestCase):
    def test_decimal_step_fields_keep_min_and_default_on_the_grid(self):
        violations = []
        for location, field, parts in _iter_number_fields():
            step = parts.get("step")
            if not step:
                continue
            minimum = parts.get("min", Decimal(0))
            if minimum % step != 0:
                violations.append(
                    f"{location} {field}: min={minimum} is not a multiple of step={step}"
                )
            default = parts.get("default")
            if default is not None and (default - minimum) % step != 0:
                violations.append(
                    f"{location} {field}: default={default} unreachable from "
                    f"min={minimum} with step={step}"
                )
        self.assertEqual(violations, [], "\n".join(violations))


class LoraTypeBranchMarkerTests(unittest.TestCase):
    """Branch-stamped fields must be tolerant; the adapter is the authority."""

    def test_rslora_and_dora_markers_are_tolerant_booleans(self):
        schema = (SCHEMA_DIR / "sd3-lora.ts").read_text(encoding="utf-8")
        self.assertIn("rs_lora: Schema.boolean().default(true)", schema)
        self.assertIn("dora_wd: Schema.boolean().default(true).hidden()", schema)
        self.assertNotIn("rs_lora: Schema.const", schema)
        for match in re.finditer(r"dora_wd: Schema\.const", schema):
            self.fail(f"dora_wd must not be a Schema.const marker: {match.group(0)}")


if __name__ == "__main__":
    unittest.main()
