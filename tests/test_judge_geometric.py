"""Unit tests for S6.1 programmatic geometric judge.

Run: python3 tests/test_judge_geometric.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import trimesh  # noqa: E402

from judge_geometric import check  # noqa: E402


def _save(mesh: trimesh.Trimesh) -> Path:
    fd = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
    fd.close()
    mesh.export(fd.name)
    return Path(fd.name)


class GeometricJudgeTest(unittest.TestCase):
    def test_unknown_category_passes(self):
        m = trimesh.creation.box((10, 10, 10))
        r = check(_save(m), "asteroid")
        self.assertTrue(r.passed)
        self.assertEqual(r.score, 10)

    def test_chair_block_fails(self):
        # 10x10x10 brick claiming to be a chair → too dense + too short
        m = trimesh.creation.box((20, 20, 10))
        r = check(_save(m), "chair")
        self.assertFalse(r.passed)
        self.assertIn("chair", r.method)
        self.assertTrue(any("flat" in i or "dense" in i for i in r.issues))

    def test_table_legs_only_passes_better_than_block(self):
        # Plausible table: thin top + 4 thin legs
        top = trimesh.creation.box((100, 80, 4))
        top.apply_translation((0, 0, 50))
        leg = trimesh.creation.box((4, 4, 50))
        legs = [
            leg.copy().apply_translation((45, 35, 25)),
            leg.copy().apply_translation((-45, 35, 25)),
            leg.copy().apply_translation((45, -35, 25)),
            leg.copy().apply_translation((-45, -35, 25)),
        ]
        from trimesh.boolean import union  # may need manifold/blender
        try:
            tbl = union([top] + legs)
            if not tbl.is_volume:
                self.skipTest("trimesh union backend not available")
        except Exception:
            self.skipTest("trimesh boolean unavailable")
        r = check(_save(tbl), "table")
        # Should not flag "too flat" (h=52 > max(w,d)*0.4=40)
        self.assertNotIn("too flat", r.fail_reason)

    def test_keychain_thick_fails(self):
        m = trimesh.creation.box((30, 20, 25))   # 25mm thick = thick
        r = check(_save(m), "keychain")
        self.assertFalse(r.passed)
        self.assertTrue(any("thick" in i for i in r.issues))

    def test_phone_stand_flat_fails(self):
        m = trimesh.creation.box((100, 100, 5))
        r = check(_save(m), "phone_stand")
        self.assertFalse(r.passed)
        self.assertTrue(any("flat" in i for i in r.issues))

    def test_shoe_round_blob_fails(self):
        m = trimesh.creation.icosphere(radius=30)
        r = check(_save(m), "shoe")
        self.assertFalse(r.passed)
        self.assertTrue(any("elongated" in i or "tall" in i
                            for i in r.issues))

    def test_bottle_short_fails(self):
        m = trimesh.creation.cylinder(radius=30, height=10)
        r = check(_save(m), "bottle")
        self.assertFalse(r.passed)
        self.assertTrue(any("short" in i for i in r.issues))

    def test_fix_suggestion_present_on_fail(self):
        m = trimesh.creation.box((20, 20, 10))   # bad chair
        r = check(_save(m), "chair")
        self.assertFalse(r.passed)
        self.assertTrue(len(r.fix_suggestion) > 10)

    def test_load_failure_returns_score_1(self):
        bogus = Path(tempfile.NamedTemporaryFile(suffix=".stl", delete=False).name)
        bogus.write_text("not actually an STL")
        r = check(bogus, "vase")
        self.assertFalse(r.passed)
        self.assertEqual(r.score, 1)


if __name__ == "__main__":
    unittest.main()
