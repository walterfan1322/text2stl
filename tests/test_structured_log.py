"""Unit tests for S7.2 structured logging.

Run: python3 tests/test_structured_log.py
"""
from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from structured_log import StructuredLog  # noqa: E402


class StructuredLogTest(unittest.TestCase):
    def setUp(self):
        self.fd = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        self.fd.close()
        self.log = StructuredLog(Path(self.fd.name))

    def tearDown(self):
        try:
            Path(self.fd.name).unlink()
        except FileNotFoundError:
            pass

    def test_empty_file_returns_empty_tail(self):
        self.assertEqual(self.log.tail(10), [])

    def test_emit_and_tail(self):
        self.log.emit("generate_done", job_id="j1", exec_ok=True,
                      judge_score=8, latency_ms=1500)
        records = self.log.tail(10)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["job_id"], "j1")
        self.assertEqual(records[0]["judge_score"], 8)
        self.assertTrue(records[0]["exec_ok"])

    def test_prompt_is_hashed(self):
        self.log.emit("generate_start", prompt="一個花瓶")
        records = self.log.tail(10)
        self.assertNotIn("prompt", records[0])
        self.assertIn("prompt_hash", records[0])
        self.assertEqual(len(records[0]["prompt_hash"]), 12)

    def test_aggregate_pass_rate(self):
        for i in range(5):
            self.log.emit("generate_done", exec_ok=True, judge_score=8,
                          latency_ms=1000)
        for i in range(2):
            self.log.emit("generate_done", exec_ok=False, latency_ms=2000)
        agg = self.log.aggregate()
        self.assertEqual(agg["n"], 7)
        self.assertAlmostEqual(agg["pass_rate"], 5 / 7, places=2)
        self.assertGreater(agg["avg_latency_ms"], 0)

    def test_aggregate_filters_event(self):
        self.log.emit("generate_done", exec_ok=True, judge_score=8)
        self.log.emit("some_other_event", x=1)
        agg = self.log.aggregate()
        self.assertEqual(agg["n"], 1)

    def test_aggregate_since_ts(self):
        self.log.emit("generate_done", exec_ok=True, judge_score=8)
        cutoff = int(time.time()) + 100  # in the future
        agg = self.log.aggregate(since_ts=cutoff)
        self.assertEqual(agg["n"], 0)


if __name__ == "__main__":
    unittest.main()
