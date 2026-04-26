"""Unit tests for pattern_cache.

Run with:  python3 -m pytest tests/test_pattern_cache.py -v
     or:  python3 tests/test_pattern_cache.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pattern_cache import (
    PatternCache,
    MAX_PER_CATEGORY,
    MIN_SCORE_TO_CACHE,
    format_examples_block,
    infer_category,
)


class InferCategoryTest(unittest.TestCase):
    def test_english_keywords(self):
        self.assertEqual(infer_category("a blue mug with handle"), "mug")
        self.assertEqual(infer_category("simple vase"), "vase")
        self.assertEqual(infer_category("bowl for soup"), "bowl")
        self.assertEqual(infer_category("Chair with four legs"), "chair")

    def test_chinese_keywords(self):
        self.assertEqual(infer_category("一個馬克杯"), "mug")
        self.assertEqual(infer_category("一個花瓶"), "vase")
        self.assertEqual(infer_category("一張桌子"), "table")
        self.assertEqual(infer_category("雪人"), "figurine")

    def test_misc_fallback(self):
        self.assertEqual(infer_category(""), "misc")
        self.assertEqual(infer_category("quasar"), "misc")
        self.assertEqual(infer_category(None or ""), "misc")

    def test_case_insensitive(self):
        self.assertEqual(infer_category("MUG"), "mug")
        self.assertEqual(infer_category("Mug"), "mug")


class PatternCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        self.tmp.close()
        self.path = Path(self.tmp.name)
        self.path.unlink()  # start fresh, no file
        self.cache = PatternCache(self.path)

    def tearDown(self):
        if self.path.exists():
            self.path.unlink()

    def test_low_score_not_cached(self):
        self.cache.record_success("a mug", "code=1", MIN_SCORE_TO_CACHE - 1)
        self.assertEqual(self.cache.examples_for("a mug"), [])

    def test_none_score_not_cached(self):
        self.cache.record_success("a mug", "code=1", None)
        self.assertEqual(self.cache.examples_for("a mug"), [])

    def test_high_score_cached_and_retrieved(self):
        self.cache.record_success("a mug", "CODE_A", 9)
        ex = self.cache.examples_for("a coffee mug")
        self.assertEqual(len(ex), 1)
        self.assertEqual(ex[0]["code"], "CODE_A")
        self.assertEqual(ex[0]["score"], 9)

    def test_misc_not_cached(self):
        self.cache.record_success("quasar", "CODE_Q", 10)
        # quasar→misc, and we explicitly skip misc bucket
        self.assertEqual(self.cache.examples_for("quasar"), [])

    def test_max_per_category_eviction(self):
        for i in range(MAX_PER_CATEGORY + 2):
            self.cache.record_success(f"mug {i}", f"CODE_{i}", 8 + (i % 3))
        ex = self.cache.examples_for("a mug", k=10)
        self.assertLessEqual(len(ex), MAX_PER_CATEGORY)

    def test_persistence_across_instances(self):
        self.cache.record_success("a vase", "VASE_1", 10)
        c2 = PatternCache(self.path)
        ex = c2.examples_for("a vase")
        self.assertEqual(len(ex), 1)
        self.assertEqual(ex[0]["code"], "VASE_1")

    def test_cross_pollination_same_category(self):
        # Chinese prompt on record, English on lookup → should still match via category
        self.cache.record_success("馬克杯", "MUG_ZH", 9)
        ex = self.cache.examples_for("simple coffee mug")
        self.assertEqual(len(ex), 1)
        self.assertEqual(ex[0]["code"], "MUG_ZH")

    def test_sort_by_score_desc(self):
        self.cache.record_success("mug a", "LOW", 8)
        self.cache.record_success("mug b", "HIGH", 10)
        ex = self.cache.examples_for("mug")
        self.assertEqual(ex[0]["score"], 10)
        self.assertEqual(ex[0]["code"], "HIGH")

    def test_corrupt_file_recovers(self):
        self.path.write_text("{not valid json}")
        c = PatternCache(self.path)
        # Should not raise; just start empty
        self.assertEqual(c.examples_for("mug"), [])

    def test_json_is_utf8(self):
        self.cache.record_success("咖啡杯", "CODE_ZH", 9)
        raw = self.path.read_text("utf-8")
        self.assertIn("咖啡杯", raw)
        data = json.loads(raw)
        self.assertIn("mug", data["categories"])


class FormatExamplesBlockTest(unittest.TestCase):
    def test_empty_returns_empty_string(self):
        self.assertEqual(format_examples_block([]), "")

    def test_single_example_formatting(self):
        block = format_examples_block([
            {"prompt": "a mug", "code": "import cq", "score": 9}
        ])
        self.assertIn("RECENT SUCCESSFUL EXAMPLES", block)
        self.assertIn("'a mug'", block)
        self.assertIn("9/10", block)
        self.assertIn("import cq", block)

    def test_multiple_examples(self):
        block = format_examples_block([
            {"prompt": "a mug", "code": "c1", "score": 9},
            {"prompt": "a vase", "code": "c2", "score": 10},
        ])
        self.assertIn("Example A1", block)
        self.assertIn("Example A2", block)


if __name__ == "__main__":
    unittest.main()
