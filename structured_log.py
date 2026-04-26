"""Structured logging (S7.2 — local replacement for Langfuse).

Append-only JSONL log of every generation request, capturing the
fields you'd otherwise look up in a Langfuse dashboard:

    timestamp, job_id, prompt, model, system_prompt_hash,
    attempts, exec_ok, judge_score, judge_category,
    geom_passed, watertight, mesh_repair_method,
    cache_hit, latency_ms_breakdown, failovers, error

One line per completed generation. Easy to tail / grep / pipe to
duckdb / pandas later for ad-hoc analysis. No 3rd-party deps.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("text2stl.structured_log")

_LOCK = threading.Lock()


def _hash_short(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:12]


class StructuredLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **fields: Any) -> None:
        """Append one structured event. `event` is the kind tag."""
        rec = {"ts": int(time.time()), "event": event}
        # Hash the prompt + system_prompt rather than storing them in full —
        # the cache pointer (job_id) gives traceability if needed.
        if "prompt" in fields:
            rec["prompt_hash"] = _hash_short(fields.pop("prompt"))
        if "system_prompt" in fields:
            rec["sys_hash"] = _hash_short(fields.pop("system_prompt"))
        rec.update(fields)
        try:
            line = json.dumps(rec, ensure_ascii=False, default=str)
        except Exception as e:
            line = json.dumps({"ts": rec["ts"], "event": "log_error",
                               "msg": str(e)})
        with _LOCK:
            try:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception as e:
                log.warning(f"structured_log write failed: {e}")

    def tail(self, n: int = 100) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            lines = self.path.read_text("utf-8").splitlines()[-n:]
        except Exception:
            return []
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out

    def aggregate(self, since_ts: int = 0) -> dict:
        """Quick aggregate over events since `since_ts`."""
        if not self.path.exists():
            return {"n": 0}
        n = 0
        n_pass = 0
        n_cache_hit = 0
        score_sum = 0
        score_n = 0
        latency_sum = 0
        latency_n = 0
        errors = 0
        try:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if rec.get("ts", 0) < since_ts:
                        continue
                    if rec.get("event") != "generate_done":
                        continue
                    n += 1
                    if rec.get("exec_ok"):
                        n_pass += 1
                    if rec.get("cache_hit"):
                        n_cache_hit += 1
                    if rec.get("error"):
                        errors += 1
                    sc = rec.get("judge_score")
                    if isinstance(sc, (int, float)):
                        score_sum += sc
                        score_n += 1
                    lt = rec.get("latency_ms")
                    if isinstance(lt, (int, float)):
                        latency_sum += lt
                        latency_n += 1
        except Exception:
            pass
        return {
            "n": n,
            "pass_rate": (n_pass / n) if n else 0.0,
            "cache_hit_rate": (n_cache_hit / n) if n else 0.0,
            "avg_score": (score_sum / score_n) if score_n else 0.0,
            "avg_latency_ms": (latency_sum / latency_n) if latency_n else 0.0,
            "error_rate": (errors / n) if n else 0.0,
        }
