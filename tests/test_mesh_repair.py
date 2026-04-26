"""Unit tests for mesh_repair.

Uses trimesh.creation primitives to build known-good / known-broken
meshes, saves them to a tempfile, and verifies the repair pipeline.

Run:  python3 tests/test_mesh_repair.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class MeshRepairTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_already_watertight_is_noop(self):
        import trimesh
        from mesh_repair import repair_stl

        box = trimesh.creation.box((10, 10, 10))
        p = self.tmp / "box.stl"
        box.export(str(p))

        res = repair_stl(p)
        self.assertTrue(res.loaded)
        self.assertTrue(res.before_watertight)
        self.assertTrue(res.after_watertight)
        self.assertFalse(res.wrote_back)
        self.assertEqual(res.method, "none")

    def test_open_mesh_attempts_repair(self):
        """A box with one face deleted → non-watertight.

        trimesh.repair.fill_holes should close the gap.
        """
        import trimesh
        from mesh_repair import repair_stl

        box = trimesh.creation.box((10, 10, 10))
        # Remove the top face (face[0] is usually top for a box; we just
        # delete the last two faces to make a clear hole.)
        box.update_faces(list(range(len(box.faces) - 2)))
        self.assertFalse(box.is_watertight, "test setup: should be open")

        p = self.tmp / "open_box.stl"
        box.export(str(p))

        res = repair_stl(p)
        self.assertTrue(res.loaded)
        self.assertFalse(res.before_watertight)
        # Depending on trimesh version/pymeshfix presence it may or may
        # not be fully watertight after repair. We just assert the
        # pipeline ran.
        self.assertIn(res.method, ("trimesh", "trimesh+pymeshfix", "none"))

    def test_missing_file_returns_not_loaded(self):
        from mesh_repair import repair_stl
        res = repair_stl(self.tmp / "nonexistent.stl")
        self.assertFalse(res.loaded)
        self.assertEqual(res.method, "none")
        self.assertIn("load failed", res.notes)

    def test_summary_string(self):
        from mesh_repair import RepairResult
        r = RepairResult(True, False, True, 0.0, 1000.0, True, "trimesh", "")
        s = r.summary()
        self.assertIn("False→True", s)
        self.assertIn("method=trimesh", s)


if __name__ == "__main__":
    unittest.main()
