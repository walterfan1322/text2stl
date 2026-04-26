"""Unit tests for P3 per-shape routing.

Tests the _route_model_for_prompt helper that decides whether to override
the user's model choice based on inferred shape category.

Run: python3 tests/test_shape_routing.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class ShapeRoutingTest(unittest.TestCase):
    def setUp(self):
        import app
        self.app = app
        # Snapshot original config so tests can mutate freely
        self._orig_enabled = app.SHAPE_ROUTING_ENABLED
        self._orig_routing = dict(app.SHAPE_ROUTING)

    def tearDown(self):
        self.app.SHAPE_ROUTING_ENABLED = self._orig_enabled
        self.app.SHAPE_ROUTING.clear()
        self.app.SHAPE_ROUTING.update(self._orig_routing)

    def _route(self, prompt, user_model):
        return self.app._route_model_for_prompt(prompt, user_model)

    def test_disabled_routing_honors_user_choice(self):
        self.app.SHAPE_ROUTING_ENABLED = False
        self.app.SHAPE_ROUTING.clear()
        self.app.SHAPE_ROUTING["bottle"] = "deepseek-chat"
        self.assertEqual(self._route("a bottle", "MiniMax-M2.7"), "MiniMax-M2.7")

    def test_routing_overrides_for_matched_category(self):
        self.app.SHAPE_ROUTING_ENABLED = True
        self.app.SHAPE_ROUTING.clear()
        self.app.SHAPE_ROUTING["bottle"] = "deepseek-chat"
        self.assertEqual(self._route("a bottle", "MiniMax-M2.7"), "deepseek-chat")

    def test_routing_no_match_keeps_user_choice(self):
        self.app.SHAPE_ROUTING_ENABLED = True
        self.app.SHAPE_ROUTING.clear()
        self.app.SHAPE_ROUTING["bottle"] = "deepseek-chat"
        self.assertEqual(self._route("a vase", "MiniMax-M2.7"), "MiniMax-M2.7")

    def test_chinese_prompt_routes(self):
        self.app.SHAPE_ROUTING_ENABLED = True
        self.app.SHAPE_ROUTING.clear()
        self.app.SHAPE_ROUTING["bottle"] = "deepseek-chat"
        self.assertEqual(self._route("一個瓶子", "MiniMax-M2.7"), "deepseek-chat")

    def test_ollama_user_choice_not_hijacked(self):
        """If user picks a local Ollama model, don't silently push to cloud."""
        self.app.SHAPE_ROUTING_ENABLED = True
        self.app.SHAPE_ROUTING.clear()
        self.app.SHAPE_ROUTING["bottle"] = "deepseek-chat"
        # 'qwen3.5:35b-a3b' is not in cloud_models
        self.assertEqual(self._route("a bottle", "qwen3.5:35b-a3b"),
                         "qwen3.5:35b-a3b")

    def test_empty_user_model_still_routes(self):
        self.app.SHAPE_ROUTING_ENABLED = True
        self.app.SHAPE_ROUTING.clear()
        self.app.SHAPE_ROUTING["bottle"] = "deepseek-chat"
        self.assertEqual(self._route("a bottle", None), "deepseek-chat")

    def test_misc_prompt_no_override(self):
        self.app.SHAPE_ROUTING_ENABLED = True
        self.app.SHAPE_ROUTING.clear()
        self.app.SHAPE_ROUTING["bottle"] = "deepseek-chat"
        # 'quasar' → misc → no override
        self.assertEqual(self._route("a quasar", "MiniMax-M2.7"), "MiniMax-M2.7")


if __name__ == "__main__":
    unittest.main()
