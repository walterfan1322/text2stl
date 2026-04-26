"""Pattern cache — retrieval-augmented few-shot for code generation.

After each successful, high-scoring generation, save the (category, prompt,
code) tuple keyed by shape category. On a new generation, look up the
latest 1-2 cached successes that match the user's shape category and
inject them as additional few-shot examples in the system prompt.

Storage: simple JSON file at BASE_DIR/pattern_cache.json. Keyed by a
canonical shape category (mug, vase, shoe, chair, ...), each entry stores
up to MAX_PER_CATEGORY recent successes with rolling eviction.

Category inference is done by keyword matching on the user prompt. It's
intentionally fuzzy — we'd rather cache under "mug" for "咖啡杯" and get
cross-pollination than have 40 empty categories.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Iterable

log = logging.getLogger("text2stl.pattern_cache")

MAX_PER_CATEGORY = 3        # keep this many best samples per category
MIN_SCORE_TO_CACHE = 8      # only cache judge ≥ this
MAX_EXAMPLES_INJECTED = 2   # never inject more than this

# Keyword → canonical category. First match wins.
CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("mug",        ("mug", "cup", "coffee", "tea", "teapot", "pitcher",
                    "馬克杯", "杯子", "咖啡杯", "茶杯", "水杯", "茶壺")),
    ("vase",       ("vase", "花瓶")),
    ("bowl",       ("bowl", "碗")),
    ("bottle",     ("bottle", "瓶子", "瓶")),
    ("pen_holder", ("pen holder", "penholder", "筆筒", "pencil holder")),
    ("phone_stand",("phone stand", "phone holder", "手機支架", "手機立架")),
    ("shoe",       ("shoe", "sneaker", "boot", "sandal",
                    "鞋", "鞋子", "靴子", "拖鞋")),
    ("chair",      ("chair", "stool", "椅子", "凳子", "板凳")),
    ("table",      ("table", "desk", "桌子", "書桌")),
    ("figurine",   ("figurine", "snowman", "雪人", "公仔", "娃娃")),
    ("keychain",   ("keychain", "pendant", "鑰匙圈", "吊飾", "項鍊")),
]


def infer_category(prompt: str) -> str:
    p = (prompt or "").lower()
    for cat, kws in CATEGORY_KEYWORDS:
        for kw in kws:
            if kw.lower() in p:
                return cat
    return "misc"


class PatternCache:
    """Thread-safe JSON-backed cache of (category → [successful samples])."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data: dict = {"version": 1, "categories": {}}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text("utf-8"))
                if "categories" not in self._data:
                    self._data["categories"] = {}
            except Exception as e:
                log.warning(f"pattern_cache: failed to load {self.path}: {e}")

    def _save(self) -> None:
        try:
            self.path.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            log.warning(f"pattern_cache: save failed: {e}")

    def record_success(self, prompt: str, code: str, score: int | None,
                       category: str | None = None) -> None:
        """Insert if score is high enough. Oldest entry is evicted."""
        if score is None or score < MIN_SCORE_TO_CACHE:
            return
        cat = category or infer_category(prompt)
        if cat == "misc":
            return  # don't pollute misc bucket
        with self._lock:
            bucket = self._data["categories"].setdefault(cat, [])
            bucket.append({
                "prompt": prompt[:200],
                "code": code,
                "score": int(score),
                "ts": int(time.time()),
            })
            # Keep only the most recent MAX_PER_CATEGORY, sorted by score desc
            bucket.sort(key=lambda x: (-x["score"], -x["ts"]))
            del bucket[MAX_PER_CATEGORY:]
            self._save()
            log.info(f"pattern_cache: saved {cat!r} sample "
                     f"(score={score}, bucket={len(bucket)})")

    def examples_for(self, prompt: str, k: int = MAX_EXAMPLES_INJECTED
                     ) -> list[dict]:
        """Return up to k cached examples matching the prompt's category."""
        cat = infer_category(prompt)
        if cat == "misc":
            return []
        with self._lock:
            bucket = list(self._data["categories"].get(cat, []))
        return bucket[:k]


def format_examples_block(examples: Iterable[dict]) -> str:
    """Format cached examples into a system-prompt appendix."""
    examples = list(examples)
    if not examples:
        return ""
    parts = ["\n\nRECENT SUCCESSFUL EXAMPLES (these produced high-scoring "
             "STL outputs — prefer this style):\n"]
    for i, e in enumerate(examples, 1):
        parts.append(f"\nExample A{i} — prompt: {e['prompt']!r} "
                     f"(judge score {e['score']}/10)")
        parts.append("```python")
        parts.append(e["code"].rstrip())
        parts.append("```")
    return "\n".join(parts)
