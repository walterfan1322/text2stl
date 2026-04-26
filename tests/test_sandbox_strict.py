"""Unit tests for S7.3 RestrictedPython sandbox layer.

Run: python3 tests/test_sandbox_strict.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sandbox_strict  # noqa: E402


@unittest.skipUnless(sandbox_strict.is_available(),
                     "RestrictedPython not installed — skipping")
class SandboxStrictTest(unittest.TestCase):

    def test_simple_code_runs(self):
        code = "x = 1 + 2\nresult = x * 10"
        g = sandbox_strict.exec_strict(code, helper_globals={},
                                       output_path="/tmp/x.stl")
        self.assertEqual(g.get("result"), 30)

    def test_helper_globals_visible(self):
        called = []

        def my_helper(x):
            called.append(x)
            return x + 1

        code = "result = my_helper(5)"
        g = sandbox_strict.exec_strict(code,
                                       helper_globals={"my_helper": my_helper},
                                       output_path="/tmp/x.stl")
        self.assertEqual(g.get("result"), 6)
        self.assertEqual(called, [5])

    def test_import_blocked(self):
        code = "import os\nresult = os.listdir('.')"
        with self.assertRaises(Exception):
            sandbox_strict.exec_strict(code, helper_globals={},
                                       output_path="/tmp/x.stl")

    def test_import_via_dunder_blocked(self):
        # Either compile fails (RestrictedPython refuses dunder names) or
        # exec fails because __import__ isn't in safe_builtins.
        code = "__import__('os').system('ls')"
        with self.assertRaises(Exception):
            sandbox_strict.exec_strict(code, helper_globals={},
                                       output_path="/tmp/x.stl")

    def test_open_blocked(self):
        code = "open('/etc/passwd', 'r').read()"
        with self.assertRaises(Exception):
            sandbox_strict.exec_strict(code, helper_globals={},
                                       output_path="/tmp/x.stl")

    def test_loop_unpack_works(self):
        code = (
            "items = [(1,2), (3,4), (5,6)]\n"
            "out = []\n"
            "for a, b in items:\n"
            "    out.append(a + b)\n"
            "result = out"
        )
        g = sandbox_strict.exec_strict(code, helper_globals={},
                                       output_path="/tmp/x.stl")
        self.assertEqual(g.get("result"), [3, 7, 11])


class SandboxAvailabilityTest(unittest.TestCase):
    def test_is_available_returns_bool(self):
        self.assertIsInstance(sandbox_strict.is_available(), bool)


if __name__ == "__main__":
    unittest.main()
