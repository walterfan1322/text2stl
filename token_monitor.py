"""Token / cost monitor — append per-LLM-call usage to a JSONL log.

Each call_cloud_llm that completes successfully invokes
record_llm_usage(model, usage_dict). The usage dict is whatever the
provider's `usage` field contained (OpenAI-compat: prompt_tokens,
completion_tokens, total_tokens).

We persist to `token_usage.jsonl` (one line per call). A separate
summariser (`tests/cost_summary.py`) aggregates it.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

log = logging.getLogger("text2stl.tokens")

# Approximate $ per 1K tokens (as of 2026-04). These are rough and only
# for trend-tracking; do not use for billing. Numbers are blended
# (input+output average) to keep the estimate simple.
APPROX_USD_PER_1K: dict[str, float] = {
    "MiniMax-M2.7":         0.0030,
    "MiniMax-M2.5":         0.0020,
    "deepseek-chat":        0.0007,
    "deepseek-reasoner":    0.0020,
    "gemini-2.5-flash":     0.0015,
    "gemini-2.5-flash-lite":0.0005,
    "gemini-2.0-flash":     0.0010,
}


class TokenMonitor:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def record(self, model: str, usage: dict | None) -> None:
        if not usage:
            return
        prompt_t = int(usage.get("prompt_tokens", 0) or 0)
        completion_t = int(usage.get("completion_tokens", 0) or 0)
        total_t = int(usage.get("total_tokens", prompt_t + completion_t) or 0)
        if total_t <= 0:
            return
        rate = APPROX_USD_PER_1K.get(model, 0.001)
        cost_usd = round(total_t / 1000.0 * rate, 6)
        entry = {
            "ts": int(time.time()),
            "model": model,
            "prompt_tokens": prompt_t,
            "completion_tokens": completion_t,
            "total_tokens": total_t,
            "est_usd": cost_usd,
        }
        with self._lock:
            try:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception as e:
                log.debug(f"token_monitor: write failed: {e}")
