"""Unit tests for S6.3 slicer integration.

We don't require a slicer to actually be installed; the tests just
verify that the module degrades gracefully when one isn't available.

Run: python3 tests/test_slicer_check.py
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slicer_check import find_slicer, slice_stl, SlicerResult  # noqa: E402


class SlicerCheckTest(unittest.TestCase):
    def test_find_slicer_returns_path_or_none(self):
        # Whichever it returns is fine — the type contract matters.
        result = find_slicer()
        self.assertTrue(result is None or isinstance(result, Path))

    def test_slice_unavailable_returns_unavailable_result(self):
        # Force unavailability with bogus path
        fake_stl = Path(tempfile.NamedTemporaryFile(suffix=".stl", delete=False).name)
        fake_stl.write_text("dummy")
        # If no slicer installed, find_slicer returns None and slice_stl
        # returns available=False. If one IS installed, the test still
        # passes — we just check that the SlicerResult contract holds.
        r = slice_stl(fake_stl, slicer_path="/nonexistent/path/to/slicer")
        self.assertIsInstance(r, SlicerResult)
        # In the "no slicer" case we expect available=False.
        # If a slicer IS found through PATH (despite our explicit_path
        # being bogus), the dummy STL won't slice → sliced=False.
        # Either way, printable should be False.
        self.assertFalse(r.printable)

    def test_slicer_result_dataclass_defaults(self):
        r = SlicerResult(available=False, sliced=False)
        self.assertEqual(r.warnings, [])
        self.assertEqual(r.errors, [])
        self.assertIsNone(r.print_time_s)
        self.assertFalse(r.printable)


if __name__ == "__main__":
    unittest.main()
