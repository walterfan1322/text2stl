"""Unit tests for S6.2 print-readiness analyser.

Run: python3 tests/test_print_readiness.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import trimesh  # noqa: E402

from print_readiness import analyse, summary_line  # noqa: E402


def _save(mesh) -> Path:
    fd = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
    fd.close()
    mesh.export(fd.name)
    return Path(fd.name)


class PrintReadinessTest(unittest.TestCase):
    def test_clean_box(self):
        m = trimesh.creation.box((20, 20, 20))
        warnings = analyse(_save(m))
        # A solid 20mm cube should produce essentially no warnings
        codes = {w["code"] for w in warnings}
        self.assertNotIn("tiny", codes)
        self.assertNotIn("multi_body", codes)
        self.assertNotIn("thin_walls", codes)

    def test_tiny_dimension(self):
        m = trimesh.creation.box((20, 20, 0.5))   # 0.5mm < 1mm
        warnings = analyse(_save(m))
        codes = [w["code"] for w in warnings]
        self.assertIn("tiny", codes)

    def test_multi_body(self):
        a = trimesh.creation.box((10, 10, 10))
        b = trimesh.creation.box((10, 10, 10))
        b.apply_translation((50, 0, 0))   # disjoint
        combined = trimesh.util.concatenate([a, b])
        warnings = analyse(_save(combined))
        codes = [w["code"] for w in warnings]
        self.assertIn("multi_body", codes)

    def test_summary_line_clean(self):
        self.assertIn("clean", summary_line([]))

    def test_summary_line_with_issues(self):
        s = summary_line([
            {"code": "tiny", "severity": "warn", "message": "x"},
            {"code": "multi_body", "severity": "info", "message": "y"},
        ])
        self.assertIn("tiny", s)
        self.assertIn("multi_body", s)


if __name__ == "__main__":
    unittest.main()
