"""Exact-match output cache (S5.1).

Skips the entire LLM/exec/judge pipeline when an identical
(prompt, model, system_prompt) tuple was successfully generated before.
Storage: SQLite at BASE_DIR/output_cache.sqlite.

The cache stores a pointer to the historical job's outputs/<job_id>/
folder, so a hit just returns those existing files. We require the
referenced folder to still exist on disk; if it was cleaned up, we
treat it as a miss and overwrite.

Design notes:
- prompt_hash is sha256 of the *raw* user prompt — we don't normalise
  whitespace because "a vase" vs "a   vase" might intentionally cue the
  LLM differently.
- sys_hash is sha256 of the system_prompt actually sent (which already
  includes pattern_cache injections for that category, so prompt-cache
  hits stay invalidated when pattern_cache evolves).
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("text2stl.output_cache")


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]


class OutputCache:
    """SQLite-backed cache of (prompt_hash, model, sys_hash) -> job_id."""

    def __init__(self, db_path: Path, outputs_root: Path):
        self.db_path = Path(db_path)
        self.outputs_root = Path(outputs_root)
        self._lock = threading.Lock()
        self._stats = {"hits": 0, "misses": 0, "stales": 0, "stores": 0}
        self._init_db()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as cx:
            cx.execute("""
                CREATE TABLE IF NOT EXISTS output_cache (
                    prompt_hash TEXT NOT NULL,
                    model       TEXT NOT NULL,
                    sys_hash    TEXT NOT NULL,
                    job_id      TEXT NOT NULL,
                    judge_score INTEGER,
                    created_ts  INTEGER NOT NULL,
                    PRIMARY KEY (prompt_hash, model, sys_hash)
                )
            """)
            cx.commit()

    def lookup(self, prompt: str, model: str, system_prompt: str
               ) -> Optional[dict]:
        """Return {job_id, judge_score, created_ts} on hit, or None."""
        ph = _hash(prompt)
        sh = _hash(system_prompt or "")
        with self._lock, sqlite3.connect(self.db_path) as cx:
            row = cx.execute(
                "SELECT job_id, judge_score, created_ts FROM output_cache "
                "WHERE prompt_hash=? AND model=? AND sys_hash=?",
                (ph, model, sh),
            ).fetchone()
        if row is None:
            self._stats["misses"] += 1
            return None
        job_id, score, ts = row
        # Verify the referenced folder still exists
        folder = self.outputs_root / job_id
        if not folder.exists() or not (folder / "model.stl").exists():
            self._stats["stales"] += 1
            with self._lock, sqlite3.connect(self.db_path) as cx:
                cx.execute(
                    "DELETE FROM output_cache WHERE prompt_hash=? AND "
                    "model=? AND sys_hash=?", (ph, model, sh),
                )
                cx.commit()
            return None
        self._stats["hits"] += 1
        return {"job_id": job_id, "judge_score": score, "created_ts": ts}

    def store(self, prompt: str, model: str, system_prompt: str,
              job_id: str, judge_score: Optional[int]) -> None:
        ph = _hash(prompt)
        sh = _hash(system_prompt or "")
        with self._lock, sqlite3.connect(self.db_path) as cx:
            cx.execute(
                "INSERT OR REPLACE INTO output_cache "
                "(prompt_hash, model, sys_hash, job_id, judge_score, "
                " created_ts) VALUES (?, ?, ?, ?, ?, ?)",
                (ph, model, sh, job_id, judge_score, int(time.time())),
            )
            cx.commit()
        self._stats["stores"] += 1
        log.info(f"output_cache: stored {model}/{ph[:8]} -> {job_id}")

    def stats(self) -> dict:
        with self._lock, sqlite3.connect(self.db_path) as cx:
            n = cx.execute("SELECT COUNT(*) FROM output_cache").fetchone()[0]
        s = dict(self._stats)
        s["total_entries"] = int(n)
        total_lookups = s["hits"] + s["misses"]
        s["hit_rate"] = (s["hits"] / total_lookups) if total_lookups else 0.0
        return s

    def purge_stale(self) -> int:
        """Drop rows whose job folder no longer exists. Returns count."""
        n = 0
        with self._lock, sqlite3.connect(self.db_path) as cx:
            rows = cx.execute(
                "SELECT prompt_hash, model, sys_hash, job_id FROM output_cache"
            ).fetchall()
            for ph, model, sh, job_id in rows:
                if not (self.outputs_root / job_id / "model.stl").exists():
                    cx.execute(
                        "DELETE FROM output_cache WHERE prompt_hash=? "
                        "AND model=? AND sys_hash=?", (ph, model, sh),
                    )
                    n += 1
            cx.commit()
        return n
