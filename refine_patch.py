"""Diff-patch helpers for the /api/refine endpoint.

The refine flow tries to get the LLM to emit a small JSON of string-replacement
edits instead of the full rewritten file. That saves ~80% of refine output
tokens and 5–10s of latency for narrow feedback. If parsing or applying the
patch fails, the endpoint falls back to a full-code rewrite.

Kept in its own module (rather than inline in app.py) so unit tests can import
without pulling in fastapi / cadquery / the rest of the runtime."""
from __future__ import annotations

import json


REFINE_PATCH_PROMPT = """You are editing an existing 3D-modelling Python script. The user has given feedback. Output a JSON array of small string-replacement edits — NOT the full code.

Format (output ONLY this JSON, no markdown, no prose):
[
  {"find": "<exact existing substring>", "replace": "<new substring>"},
  ...
]

Rules:
- Each `find` must appear EXACTLY ONCE in the current code (include enough surrounding context to be unique).
- Keep edits minimal — change only what the feedback requires.
- Preserve indentation and surrounding whitespace inside `find` / `replace`.
- 1–6 edits is typical. If the change is too sweeping for small edits, output the single-element array [{"find":"FULL_REWRITE","replace":""}] and the system will fall back to a full-code rewrite.
- Do NOT wrap in ```json fences. Output raw JSON only.
"""


def apply_patch_edits(code: str, edits: list) -> tuple[str, str | None]:
    """Apply a list of {find, replace} edits to code. Returns (new_code, error).
    error is None on success. On any failure (find not found / not unique /
    sentinel FULL_REWRITE), returns (code, error_message)."""
    if not isinstance(edits, list) or not edits:
        return code, "edits is empty or not a list"
    for i, e in enumerate(edits):
        if not isinstance(e, dict) or "find" not in e or "replace" not in e:
            return code, f"edit #{i} missing find/replace"
        find_s = e["find"]
        if find_s == "FULL_REWRITE":
            return code, "FULL_REWRITE sentinel — fall back to full-code refine"
        if not isinstance(find_s, str) or not isinstance(e["replace"], str):
            return code, f"edit #{i} find/replace not strings"
        count = code.count(find_s)
        if count == 0:
            return code, f"edit #{i} find string not present in code"
        if count > 1:
            return code, f"edit #{i} find string appears {count} times (must be unique)"
        code = code.replace(find_s, e["replace"], 1)
    return code, None


def parse_patch_response(raw: str) -> tuple[list | None, str | None]:
    """Strip ```json fences and parse. Returns (edits, error)."""
    s = raw.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    try:
        parsed = json.loads(s)
    except Exception as e:
        return None, f"json parse failed: {e}"
    if not isinstance(parsed, list):
        return None, "top-level must be a JSON array"
    return parsed, None
