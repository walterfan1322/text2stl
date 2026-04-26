"""Unit tests for the loft-topology pre-check.

Catches mismatched-point-count loft chains before they reach the BRep kernel
and produce a flat-blob shoe. Run:
    python tests/test_loft_topology.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validators import check_loft_topology, validate_cadquery


GOOD_2_SECTION = '''
import cadquery as cq
sole_pts = [(0,30),(15,10),(50,0),(120,0),(200,0),(250,10),(275,30),
            (280,55),(270,80),(240,95),(180,100),(100,100),(40,95),(10,80),(0,55)]
upper_pts = [(20,40),(40,25),(80,20),(140,20),(200,20),(240,25),(255,40),
             (255,65),(240,80),(200,90),(140,90),(80,90),(40,85),(20,70),(20,55)]
result = (cq.Workplane("XY")
          .polyline(sole_pts).close()
          .workplane(offset=70)
          .polyline(upper_pts).close()
          .loft(combine=True))
export_stl(result, OUTPUT_PATH)
'''


BAD_5_SECTION = '''
import cadquery as cq
toe_pts = [(0,32),(20,18),(50,12),(90,10),(140,8),(170,10),(200,15),(230,28),
           (250,42),(265,52),(270,60),(250,70),(220,75),(180,78),(140,80),(90,76),
           (50,68),(20,52),(0,40)]
ball_pts = [(0,28),(18,14),(45,8),(85,6),(135,5),(165,7),(195,12),(225,25),
            (245,38),(258,48),(262,55),(245,64),(215,68),(175,72),(135,74),(85,70),
            (45,62),(18,46),(0,35)]
instep_pts = [(20,26),(40,12),(70,6),(110,4),(150,5),(180,8),(210,20),(235,32),
              (250,44),(258,52),(245,60),(220,64),(185,68),(150,70),(110,68),(70,62),
              (40,48),(20,38)]
heel_pts = [(80,24),(105,12),(135,8),(165,8),(195,12),(220,24),(235,35),(242,45),
            (235,54),(215,60),(180,64),(150,66),(120,64),(100,56),(80,44)]
collar_pts = [(100,22),(120,12),(145,8),(165,8),(185,12),(200,24),(208,34),
              (202,44),(185,52),(160,56),(140,56),(120,52),(105,44),(100,36)]
result = (cq.Workplane("XY")
          .polyline(toe_pts).close()
          .workplane(offset=20)
          .polyline(ball_pts).close()
          .workplane(offset=20)
          .polyline(instep_pts).close()
          .workplane(offset=20)
          .polyline(heel_pts).close()
          .workplane(offset=15)
          .polyline(collar_pts).close()
          .loft(combine=True))
export_stl(result, OUTPUT_PATH)
'''


INLINE_MISMATCH = '''
import cadquery as cq
result = (cq.Workplane("XY")
          .polyline([(0,0),(10,0),(10,10),(0,10)]).close()
          .workplane(offset=20)
          .polyline([(0,0),(5,0),(5,5)]).close()
          .loft(combine=True))
export_stl(result, OUTPUT_PATH)
'''


NO_LOFT = '''
import cadquery as cq
result = cq.Workplane("XY").box(10, 10, 5)
export_stl(result, OUTPUT_PATH)
'''


class CheckLoftTopologyTest(unittest.TestCase):
    def test_two_matched_sections_ok(self):
        errs = check_loft_topology(GOOD_2_SECTION)
        self.assertEqual(errs, [])

    def test_five_mismatched_sections_flagged(self):
        errs = check_loft_topology(BAD_5_SECTION)
        self.assertEqual(len(errs), 1)
        msg = errs[0]
        # Should mention all the offending names with their counts
        self.assertIn("toe_pts=19", msg)
        self.assertIn("ball_pts=19", msg)
        self.assertIn("instep_pts=18", msg)
        self.assertIn("heel_pts=15", msg)
        self.assertIn("collar_pts=14", msg)
        self.assertIn("IDENTICAL point counts", msg)

    def test_inline_polyline_mismatch_flagged(self):
        errs = check_loft_topology(INLINE_MISMATCH)
        self.assertEqual(len(errs), 1)
        # Inline literals get an <inline-LINE> label
        self.assertIn("<inline-", errs[0])

    def test_no_loft_no_errors(self):
        self.assertEqual(check_loft_topology(NO_LOFT), [])

    def test_validate_cadquery_propagates_loft_error(self):
        res = validate_cadquery(BAD_5_SECTION)
        self.assertFalse(res.ok)
        self.assertTrue(any("loft topology mismatch" in e for e in res.errors))

    def test_validate_cadquery_passes_good_shoe(self):
        res = validate_cadquery(GOOD_2_SECTION)
        # API allowlist may emit other errors but loft check should not.
        self.assertFalse(any("loft topology" in e for e in res.errors))

    def test_syntax_error_returns_empty_no_crash(self):
        self.assertEqual(check_loft_topology("def(:"), [])


if __name__ == "__main__":
    unittest.main()
