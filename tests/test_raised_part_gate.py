"""Unit tests for raised_part_gate — the AST/bbox gate that catches
sub-parts buried inside the root volume.

Run: python tests/test_raised_part_gate.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from raised_part_gate import check, extract_primitives  # noqa: E402


# ---------------------------------------------------------------------------
# Test corpus — small CadQuery snippets
# ---------------------------------------------------------------------------

CHESSBOARD_GOOD = """
import cadquery as cq

base = cq.Workplane("XY").box(400, 400, 8, centered=(True, True, False))
dark = None
for i in range(8):
    for j in range(8):
        x = -175 + i * 50
        y = -175 + j * 50
        sq = (cq.Workplane("XY")
              .center(x, y)
              .workplane(offset=7.9)
              .box(50, 50, 2.1, centered=(True, True, False)))
        dark = sq if dark is None else dark.union(sq)

result = base.union(dark)
"""

# Same shape but `.workplane(offset=8)` exactly — touches at z=8.
# Should be flagged as touching-face (OCC silent-drop risk).
CHESSBOARD_TOUCHING = """
import cadquery as cq

base = cq.Workplane("XY").box(400, 400, 8, centered=(True, True, False))
dark = None
for i in range(8):
    for j in range(8):
        x = -175 + i * 50
        y = -175 + j * 50
        sq = (cq.Workplane("XY")
              .center(x, y)
              .workplane(offset=8)
              .box(50, 50, 2, centered=(True, True, False)))
        dark = sq if dark is None else dark.union(sq)

result = base.union(dark)
"""

CHESSBOARD_BURIED = """
import cadquery as cq

base = cq.Workplane("XY").box(400, 400, 8, centered=(True, True, False))
dark = None
for i in range(8):
    for j in range(8):
        x = -175 + i * 50
        y = -175 + j * 50
        sq = (cq.Workplane("XY")
              .center(x, y)
              .box(50, 50, 2, centered=(True, True, False)))
        dark = sq if dark is None else dark.union(sq)

result = base.union(dark)
"""

# Mug: revolve body + sweep handle. Neither chain ends in box/cylinder/sphere
# at top-level so the gate skips them — must pass.
MUG = """
import cadquery as cq

profile = cq.Workplane("XZ").polyline([(0,0),(40,0),(40,80),(35,80),(35,5),(0,5)]).close()
body = profile.revolve(360)

handle = (cq.Workplane("YZ").center(0, 40)
          .ellipse(15, 25)
          .sweep(cq.Workplane("XY").circle(3)))

result = body.union(handle)
"""

SNOWMAN = """
import cadquery as cq

# Each sphere overlaps the one below it by 0.1mm to avoid OCC's
# touching-face silent-drop.
bottom = cq.Workplane("XY").workplane(offset=60).sphere(60)
middle = cq.Workplane("XY").workplane(offset=159.9).sphere(40)
head   = cq.Workplane("XY").workplane(offset=239.8).sphere(30)

result = bottom.union(middle).union(head)
"""

TABLE_GOOD = """
import cadquery as cq

top = (cq.Workplane("XY")
       .workplane(offset=720)
       .box(800, 500, 30, centered=(True, True, False)))
leg1 = (cq.Workplane("XY")
        .center(-380, -230)
        .box(40, 40, 720, centered=(True, True, False)))
leg2 = (cq.Workplane("XY")
        .center( 380, -230)
        .box(40, 40, 720, centered=(True, True, False)))
leg3 = (cq.Workplane("XY")
        .center(-380,  230)
        .box(40, 40, 720, centered=(True, True, False)))
leg4 = (cq.Workplane("XY")
        .center( 380,  230)
        .box(40, 40, 720, centered=(True, True, False)))

result = top.union(leg1).union(leg2).union(leg3).union(leg4)
"""

# Pathological table: top is a huge thick slab, legs are short and embedded
# inside the slab — should fail.
TABLE_BURIED_LEGS = """
import cadquery as cq

# Wide+thick top from 0..40 covering the whole footprint
top = (cq.Workplane("XY")
       .box(800, 500, 40, centered=(True, True, False)))
# Legs sit inside the top: 30mm tall starting at 0 (Z 0..30 < 40)
leg1 = (cq.Workplane("XY")
        .center(-200, -100)
        .box(40, 40, 30, centered=(True, True, False)))
leg2 = (cq.Workplane("XY")
        .center( 200, -100)
        .box(40, 40, 30, centered=(True, True, False)))

result = top.union(leg1).union(leg2)
"""

VASE_SOLO = """
import cadquery as cq

profile = cq.Workplane("XZ").polyline([(0,0),(40,0),(35,30),(50,80),(40,120),(0,120)]).close()
result = profile.revolve(360)
"""

# Recess: cut a 50x50x2 cavity from the top of a slab. Cut child IS buried
# (sits in z=6..8 inside z=0..8 base) but role is `cut` so it's allowed.
RECESS_OK = """
import cadquery as cq

base = cq.Workplane("XY").box(100, 100, 8, centered=(True, True, False))
cavity = (cq.Workplane("XY")
          .center(0, 0)
          .workplane(offset=6)
          .box(50, 50, 2.5, centered=(True, True, False)))

result = base.cut(cavity)
"""


class RaisedPartGateTest(unittest.TestCase):
    # --- happy paths ---------------------------------------------------

    def test_chessboard_with_offset_passes(self):
        r = check(CHESSBOARD_GOOD)
        self.assertTrue(r.passed,
                        f"expected pass, got: {r.fail_reason}; issues={r.issues}")
        self.assertEqual(r.buried_vars, [])

    def test_mug_revolve_plus_sweep_passes(self):
        # Neither chain ends in a recognised XY primitive — the gate
        # silently skips both, which is the conservative behaviour.
        r = check(MUG)
        self.assertTrue(r.passed)

    def test_snowman_three_spheres_passes(self):
        r = check(SNOWMAN)
        self.assertTrue(r.passed,
                        f"snowman should pass, got: {r.fail_reason}; "
                        f"issues={r.issues}")
        # 3 spheres extracted
        self.assertEqual(len(r.primitives), 3)

    def test_table_with_legs_below_passes(self):
        r = check(TABLE_GOOD)
        self.assertTrue(r.passed,
                        f"table-good should pass, got: {r.fail_reason}; "
                        f"issues={r.issues}")
        # 5 primitives: top + 4 legs
        self.assertEqual(len(r.primitives), 5)

    def test_single_revolve_vase_passes(self):
        r = check(VASE_SOLO)
        # No primitives at all — single revolve, gate skips.
        self.assertTrue(r.passed)
        self.assertEqual(len(r.primitives), 0)

    def test_recess_via_cut_passes(self):
        r = check(RECESS_OK)
        self.assertTrue(r.passed,
                        f"recess via .cut() should pass, got: "
                        f"{r.fail_reason}; issues={r.issues}")
        # Cavity should be tagged role=cut
        cavity = next((p for p in r.primitives
                       if p.var_name == "cavity"), None)
        self.assertIsNotNone(cavity)
        self.assertEqual(cavity.role, "cut")

    # --- failure paths -------------------------------------------------

    def test_chessboard_touching_face_fails(self):
        # `.workplane(offset=8)` puts squares Z=8..10, base is Z=0..8.
        # They share the z=8 plane exactly — OCC may silently drop.
        r = check(CHESSBOARD_TOUCHING)
        self.assertFalse(r.passed,
                         "touching-face should be caught; "
                         f"primitives={[(p.var_name, p.bbox.zmin, p.bbox.zmax) for p in r.primitives]}")
        self.assertIn("sq", r.touching_vars)
        self.assertEqual(r.buried_vars, [])
        self.assertEqual(len(r.issues), 1)
        self.assertIn("EXACTLY", r.issues[0])
        self.assertIn("0.1mm", r.fix_suggestion)

    def test_chessboard_buried_squares_fails(self):
        r = check(CHESSBOARD_BURIED)
        self.assertFalse(r.passed,
                         "buried squares should be caught; "
                         f"primitives={[p.var_name for p in r.primitives]}")
        # The primitive var name is `sq` (assigned inside the loop body)
        self.assertIn("sq", r.buried_vars)
        # Issues list collapses the 64 unrolled iterations into one entry
        self.assertEqual(len(r.issues), 1)
        self.assertIn("union has no visible effect", r.issues[0])
        # Fix suggestion mentions workplane(offset=...) with 0.1mm overlap
        self.assertIn("workplane(offset=7.9)", r.fix_suggestion)
        self.assertIn("0.1mm", r.fix_suggestion)

    def test_table_with_buried_legs_fails(self):
        r = check(TABLE_BURIED_LEGS)
        self.assertFalse(r.passed,
                         "legs Z=0..30 inside top Z=0..40 should be buried")
        self.assertIn("leg1", r.buried_vars)
        self.assertIn("leg2", r.buried_vars)


class ExtractPrimitivesTest(unittest.TestCase):
    """Lower-level checks on the AST → bbox layer."""

    def test_basic_box_centered(self):
        code = 'import cadquery as cq\nb = cq.Workplane("XY").box(10, 20, 30)\n'
        prims = extract_primitives(code)
        self.assertEqual(len(prims), 1)
        bb = prims[0].bbox
        self.assertAlmostEqual(bb.xmin, -5);   self.assertAlmostEqual(bb.xmax, 5)
        self.assertAlmostEqual(bb.ymin, -10);  self.assertAlmostEqual(bb.ymax, 10)
        self.assertAlmostEqual(bb.zmin, -15);  self.assertAlmostEqual(bb.zmax, 15)

    def test_box_centered_false_z(self):
        code = ('import cadquery as cq\n'
                'b = cq.Workplane("XY").box(10, 20, 8, centered=(True, True, False))\n')
        prims = extract_primitives(code)
        self.assertEqual(len(prims), 1)
        bb = prims[0].bbox
        self.assertAlmostEqual(bb.zmin, 0)
        self.assertAlmostEqual(bb.zmax, 8)

    def test_workplane_offset_lifts_z(self):
        code = ('import cadquery as cq\n'
                's = cq.Workplane("XY").workplane(offset=8).box(50, 50, 2, '
                'centered=(True, True, False))\n')
        prims = extract_primitives(code)
        self.assertEqual(len(prims), 1)
        bb = prims[0].bbox
        self.assertAlmostEqual(bb.zmin, 8)
        self.assertAlmostEqual(bb.zmax, 10)

    def test_center_translates_xy(self):
        code = ('import cadquery as cq\n'
                's = cq.Workplane("XY").center(100, -50).box(10, 10, 10, '
                'centered=(True, True, False))\n')
        prims = extract_primitives(code)
        self.assertEqual(len(prims), 1)
        bb = prims[0].bbox
        self.assertAlmostEqual(bb.xmin, 95);  self.assertAlmostEqual(bb.xmax, 105)
        self.assertAlmostEqual(bb.ymin, -55); self.assertAlmostEqual(bb.ymax, -45)

    def test_for_loop_unrolls(self):
        code = ('import cadquery as cq\n'
                'for i in range(3):\n'
                '    s = cq.Workplane("XY").center(i * 50, 0).box(10, 10, 10, '
                'centered=(True, True, False))\n')
        prims = extract_primitives(code)
        self.assertEqual(len(prims), 3)
        xs = sorted(p.bbox.xmin for p in prims)
        # Three iterations at i = 0, 1, 2 → xmin = -5, 45, 95
        self.assertAlmostEqual(xs[0], -5)
        self.assertAlmostEqual(xs[1], 45)
        self.assertAlmostEqual(xs[2], 95)

    def test_append_inline_chain(self):
        # LLMs sometimes write `parts.append(cq.Workplane("XY")...)`
        # instead of assigning to a temp var first. The walker should
        # treat the inline chain as a synthetic primitive named after
        # the list.
        code = (
            'import cadquery as cq\n'
            'parts = []\n'
            'for i in range(3):\n'
            '    parts.append(cq.Workplane("XY").center(i*50, 0).box(10, 10, 10, '
            'centered=(True, True, False)))\n'
        )
        prims = extract_primitives(code)
        self.assertEqual(len(prims), 3)
        # All three should be named after the list (with iter suffixes)
        for p in prims:
            self.assertTrue(p.var_name.startswith("parts"),
                            f"unexpected var_name: {p.var_name}")

    def test_xz_plane_skipped(self):
        # XZ-plane primitives (typically revolve profiles) are
        # conservatively skipped.
        code = ('import cadquery as cq\n'
                'p = cq.Workplane("XZ").box(10, 10, 10)\n')
        prims = extract_primitives(code)
        self.assertEqual(len(prims), 0)


if __name__ == "__main__":
    unittest.main()
