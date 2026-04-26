"""Unit tests for S5.1 output cache.

Run: python3 tests/test_output_cache.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from output_cache import OutputCache  # noqa: E402


class OutputCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="oc_test_"))
        self.outputs = self.tmp / "outputs"
        self.outputs.mkdir()
        self.db = self.tmp / "cache.sqlite"
        self.cache = OutputCache(self.db, self.outputs)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_job(self, job_id: str) -> Path:
        d = self.outputs / job_id
        d.mkdir()
        (d / "model.stl").write_text("dummy")
        return d

    def test_miss_returns_none(self):
        self.assertIsNone(self.cache.lookup("a vase", "MiniMax-M2.7", "sys"))

    def test_store_then_hit(self):
        self._make_job("job1")
        self.cache.store("a vase", "MiniMax-M2.7", "sys", "job1", 9)
        hit = self.cache.lookup("a vase", "MiniMax-M2.7", "sys")
        self.assertIsNotNone(hit)
        self.assertEqual(hit["job_id"], "job1")
        self.assertEqual(hit["judge_score"], 9)

    def test_different_model_no_pollution(self):
        self._make_job("job1")
        self.cache.store("a vase", "MiniMax-M2.7", "sys", "job1", 9)
        self.assertIsNone(self.cache.lookup("a vase", "deepseek-chat", "sys"))

    def test_different_sys_prompt_no_pollution(self):
        self._make_job("job1")
        self.cache.store("a vase", "MiniMax-M2.7", "sys-A", "job1", 9)
        self.assertIsNone(self.cache.lookup("a vase", "MiniMax-M2.7", "sys-B"))

    def test_stale_folder_returns_none_and_purges(self):
        self._make_job("job1")
        self.cache.store("a vase", "MiniMax-M2.7", "sys", "job1", 9)
        # delete folder
        shutil.rmtree(self.outputs / "job1")
        self.assertIsNone(self.cache.lookup("a vase", "MiniMax-M2.7", "sys"))
        s = self.cache.stats()
        self.assertGreaterEqual(s["stales"], 1)

    def test_stats_track_hits_and_misses(self):
        self._make_job("job1")
        self.cache.store("p", "m", "s", "job1", 8)
        self.cache.lookup("p", "m", "s")          # hit
        self.cache.lookup("p", "m", "s")          # hit
        self.cache.lookup("other", "m", "s")      # miss
        s = self.cache.stats()
        self.assertEqual(s["hits"], 2)
        self.assertEqual(s["misses"], 1)
        self.assertAlmostEqual(s["hit_rate"], 2 / 3, places=2)

    def test_purge_stale(self):
        self._make_job("a")
        self._make_job("b")
        self.cache.store("p1", "m", "s", "a", 8)
        self.cache.store("p2", "m", "s", "b", 8)
        shutil.rmtree(self.outputs / "a")
        purged = self.cache.purge_stale()
        self.assertEqual(purged, 1)
        s = self.cache.stats()
        self.assertEqual(s["total_entries"], 1)


if __name__ == "__main__":
    unittest.main()
