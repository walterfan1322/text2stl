"""Unit tests for S7.1 best-of-N helper.

Run: python3 tests/test_best_of_n.py
"""
from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from best_of_n import (  # noqa: E402
    Candidate, n_for_category, pick_best, run_best_of_n, score_candidate,
    temps_for,
)


class TempsTest(unittest.TestCase):
    def test_n1(self):
        self.assertEqual(temps_for(1), [0.5])

    def test_n3(self):
        self.assertEqual(temps_for(3), [0.3, 0.6, 0.9])

    def test_unknown_n_spread(self):
        ts = temps_for(5)
        self.assertEqual(len(ts), 5)
        self.assertAlmostEqual(ts[0], 0.3, places=2)
        self.assertAlmostEqual(ts[-1], 1.0, places=2)


class PickBestTest(unittest.TestCase):
    def test_exec_ok_beats_higher_score_failed(self):
        c1 = Candidate(idx=0, temperature=0.5, exec_ok=True,
                       judge_score=5, watertight=True, geom_passed=True)
        c2 = Candidate(idx=1, temperature=0.7, exec_ok=False,
                       judge_score=10, watertight=False, geom_passed=False)
        self.assertEqual(pick_best([c1, c2]).idx, 0)

    def test_higher_score_wins_when_both_pass(self):
        c1 = Candidate(idx=0, temperature=0.5, exec_ok=True,
                       judge_score=7, watertight=True, geom_passed=True)
        c2 = Candidate(idx=1, temperature=0.7, exec_ok=True,
                       judge_score=9, watertight=True, geom_passed=True)
        self.assertEqual(pick_best([c1, c2]).idx, 1)

    def test_watertight_breaks_tied_score(self):
        c1 = Candidate(idx=0, temperature=0.5, exec_ok=True,
                       judge_score=8, watertight=False, geom_passed=True)
        c2 = Candidate(idx=1, temperature=0.7, exec_ok=True,
                       judge_score=8, watertight=True, geom_passed=True)
        self.assertEqual(pick_best([c1, c2]).idx, 1)

    def test_fast_breaks_full_tie(self):
        c1 = Candidate(idx=0, temperature=0.5, exec_ok=True,
                       judge_score=8, watertight=True, geom_passed=True,
                       elapsed_s=10.0)
        c2 = Candidate(idx=1, temperature=0.7, exec_ok=True,
                       judge_score=8, watertight=True, geom_passed=True,
                       elapsed_s=5.0)
        self.assertEqual(pick_best([c1, c2]).idx, 1)


class CategoryTest(unittest.TestCase):
    def test_unstable_categories_get_higher_n(self):
        self.assertGreaterEqual(n_for_category("figurine"), 2)
        self.assertGreaterEqual(n_for_category("bottle"), 2)

    def test_stable_categories_default_n1(self):
        self.assertEqual(n_for_category("vase"), 1)
        self.assertEqual(n_for_category("chair"), 1)

    def test_override_table(self):
        self.assertEqual(n_for_category("chair", {"chair": 5}), 5)


class RunBestOfNTest(unittest.TestCase):
    def test_runs_concurrently_and_picks_best(self):
        async def runner(t: float, idx: int) -> Candidate:
            await asyncio.sleep(0.01 * idx)  # different latencies
            return Candidate(
                idx=idx, temperature=t, exec_ok=True,
                judge_score=int(t * 10),  # higher temp → higher score (mock)
                watertight=True, geom_passed=True, elapsed_s=0.01 * idx,
            )
        best, all_ = asyncio.run(run_best_of_n(3, runner))
        self.assertEqual(len(all_), 3)
        # Highest temp (0.9) should yield highest score
        self.assertEqual(best.judge_score, 9)


if __name__ == "__main__":
    unittest.main()
