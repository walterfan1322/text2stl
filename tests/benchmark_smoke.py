"""Smoke benchmark for CI gate — fast, minimal coverage.

3 shapes × 1 model × 1 trial = 3 runs, ~3 min. Uses a REPRESENTATIVE
subset covering revolve (vase), extrude (phone_stand), and composite
(chair). Exits non-zero if pass-rate < 2/3.

Usage:
    python3 tests/benchmark_smoke.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_v2 import run_one, format_summary  # noqa: E402

SMOKE_SHAPES = {
    "vase":        "一個花瓶",
    "phone_stand": "一個簡單的手機立架",
    "chair":       "一張有四條腿和靠背的椅子",
}
MODEL = "MiniMax-M2.7"
MIN_PASS = 2  # require 2/3 pass to consider the change safe


def main() -> int:
    t0 = time.time()
    results = []
    for i, (key, prompt) in enumerate(SMOKE_SHAPES.items(), 1):
        print(f"[{i}/{len(SMOKE_SHAPES)}] {MODEL} × {key} ... ", end="", flush=True)
        r = run_one(MODEL, key, prompt, trial=1)
        results.append(r)
        tag = "PASS" if r["exec_ok"] else "FAIL"
        sc = r["judge_score"]
        sc_s = f"{sc}/10" if sc is not None else "  -"
        print(f"{tag}  score={sc_s}  t={r['elapsed_s']:>5.1f}s")
        time.sleep(2)
    elapsed = time.time() - t0

    n_pass = sum(1 for r in results if r["exec_ok"])
    print()
    print(f"Smoke pass: {n_pass}/{len(results)} (gate: >={MIN_PASS})  "
          f"wall={elapsed/60:.1f}min")
    if n_pass < MIN_PASS:
        print("SMOKE FAILED", file=sys.stderr)
        return 1
    print("smoke ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
