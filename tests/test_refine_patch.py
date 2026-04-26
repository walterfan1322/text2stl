"""Unit tests for the refine diff-patch helpers.

Covers _apply_patch_edits and _parse_patch_response — the parts that decide
whether to take the small-diff fast path or fall back to a full rewrite.

Run with:  python tests/test_refine_patch.py
     or:  python -m pytest tests/test_refine_patch.py -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from refine_patch import (
    apply_patch_edits as _apply_patch_edits,
    parse_patch_response as _parse_patch_response,
)


SAMPLE = """import cadquery as cq
result = cq.Workplane("XY").box(10, 10, 5)
result = result.translate((0, 0, 2.5))
"""


class ApplyPatchEditsTest(unittest.TestCase):
    def test_single_unique_edit(self):
        edits = [{"find": "box(10, 10, 5)", "replace": "box(20, 20, 5)"}]
        out, err = _apply_patch_edits(SAMPLE, edits)
        self.assertIsNone(err)
        self.assertIn("box(20, 20, 5)", out)
        self.assertNotIn("box(10, 10, 5)", out)

    def test_multiple_edits_applied_in_order(self):
        edits = [
            {"find": "10, 10, 5", "replace": "20, 20, 10"},
            {"find": "0, 0, 2.5", "replace": "0, 0, 5"},
        ]
        out, err = _apply_patch_edits(SAMPLE, edits)
        self.assertIsNone(err)
        self.assertIn("20, 20, 10", out)
        self.assertIn("0, 0, 5", out)

    def test_find_not_found(self):
        edits = [{"find": "nonexistent_substring", "replace": "x"}]
        _, err = _apply_patch_edits(SAMPLE, edits)
        self.assertIsNotNone(err)
        self.assertIn("not present", err)

    def test_find_not_unique(self):
        # "result" appears 3 times in SAMPLE
        edits = [{"find": "result", "replace": "shape"}]
        _, err = _apply_patch_edits(SAMPLE, edits)
        self.assertIsNotNone(err)
        self.assertIn("must be unique", err)

    def test_full_rewrite_sentinel(self):
        edits = [{"find": "FULL_REWRITE", "replace": ""}]
        _, err = _apply_patch_edits(SAMPLE, edits)
        self.assertIsNotNone(err)
        self.assertIn("FULL_REWRITE", err)

    def test_empty_edits(self):
        _, err = _apply_patch_edits(SAMPLE, [])
        self.assertIsNotNone(err)

    def test_malformed_edit(self):
        _, err = _apply_patch_edits(SAMPLE, [{"find": "x"}])  # missing replace
        self.assertIsNotNone(err)


class ParsePatchResponseTest(unittest.TestCase):
    def test_raw_json_array(self):
        raw = '[{"find":"a","replace":"b"}]'
        edits, err = _parse_patch_response(raw)
        self.assertIsNone(err)
        self.assertEqual(edits, [{"find": "a", "replace": "b"}])

    def test_strips_json_fence(self):
        raw = '```json\n[{"find":"a","replace":"b"}]\n```'
        edits, err = _parse_patch_response(raw)
        self.assertIsNone(err)
        self.assertEqual(len(edits), 1)

    def test_strips_bare_fence(self):
        raw = '```\n[]\n```'
        edits, err = _parse_patch_response(raw)
        self.assertIsNone(err)
        self.assertEqual(edits, [])

    def test_invalid_json(self):
        edits, err = _parse_patch_response('not json at all')
        self.assertIsNone(edits)
        self.assertIsNotNone(err)

    def test_top_level_not_array(self):
        edits, err = _parse_patch_response('{"find":"a","replace":"b"}')
        self.assertIsNone(edits)
        self.assertIn("array", err)


if __name__ == "__main__":
    unittest.main()
