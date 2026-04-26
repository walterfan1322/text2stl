"""Unit tests for error→hint mapping.

Covers _cadquery_fix_hint (loft, fillet, selector, BRep, workplane) and
_trimesh_fix_hint. These hints nudge the LLM toward the correct repair
instead of flailing (adding more points, nesting more fillets).

Run:  python3 tests/test_fix_hints.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class CadqueryHintTest(unittest.TestCase):
    def setUp(self):
        from app import _cadquery_fix_hint  # type: ignore
        self.hint = _cadquery_fix_hint

    def test_loft_failure_suggests_winding(self):
        h = self.hint("BRepFill_Filling failed")
        self.assertIn("wind", h.lower())
        self.assertIn("same direction", h.lower())

    def test_fillet_failure_lists_valid_selectors(self):
        h = self.hint("fillets requires edges")
        for sel in ('">Z"', '"<Z"', '"|Z"', '"|X"'):
            self.assertIn(sel, h)

    def test_invented_percent_selector(self):
        h = self.hint("BRep_API: command not done with %Circle selector")
        # Either branch (brep_api or %circle) should fire — both give advice
        self.assertTrue(len(h) > 0)

    def test_degenerate_profile_gets_simplify_advice(self):
        h = self.hint("BRep_API: command not done")
        self.assertIn("FEWER points", h)
        self.assertIn("SIMPLE", h)

    def test_unknown_error_returns_empty(self):
        self.assertEqual(self.hint("completely unrelated error"), "")

    def test_ncollection_suggests_primitives(self):
        h = self.hint("NCollection_Sequence out of range")
        self.assertIn("primitives", h.lower())


class TrimeshHintTest(unittest.TestCase):
    def setUp(self):
        from app import _trimesh_fix_hint  # type: ignore
        self.hint = _trimesh_fix_hint

    def test_revolve_suggests_helper(self):
        h = self.hint("trimesh.creation.revolve missing")
        self.assertIn("make_solid_revolution", h)

    def test_boolean_suggests_concatenate(self):
        h = self.hint("not all meshes are volumes")
        self.assertIn("concatenate", h)

    def test_unknown_returns_empty(self):
        self.assertEqual(self.hint("oom"), "")


if __name__ == "__main__":
    unittest.main()
