"""Regression diff — compares the latest benchmark_v2 run to baseline.json.

Usage:
    # run benchmark first: python3 tests/benchmark_v2.py
    python3 tests/regression_diff.py                     # diff to baseline
    python3 tests/regression_diff.py --gate 0.2          # CI gate: non-zero exit
                                                         # if any (model,shape)
                                                         # exec_pass drops > 0.2

Exits 0 if no regression, 1 if any model/shape pair drops beyond the gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
BASELINE = ROOT / "baseline.json"
CURRENT = ROOT / "benchmark_v2_results.json"


def aggregate(rows: list[dict]) -> dict:
    from benchmark_v2 import aggregate as agg
    return agg(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default=str(BASELINE),
                   help="path to baseline.json")
    p.add_argument("--current", default=str(CURRENT),
                   help="path to current benchmark_v2_results.json")
    p.add_argument("--gate", type=float, default=0.2,
                   help="regression threshold on exec_pass_rate (default 0.2 = 20pp)")
    p.add_argument("--score-gate", type=float, default=1.5,
                   help="regression threshold on avg_score (default 1.5 pts)")
    args = p.parse_args()

    if not Path(args.baseline).exists():
        print(f"No baseline at {args.baseline}. Run `benchmark_v2.py --baseline` first.",
              file=sys.stderr)
        return 2
    if not Path(args.current).exists():
        print(f"No current results at {args.current}. Run benchmark_v2.py first.",
              file=sys.stderr)
        return 2

    base = json.loads(Path(args.baseline).read_text())["baseline"]
    cur = json.loads(Path(args.current).read_text())["results"]

    # Group current by (model, shape)
    cur_agg: dict[tuple[str, str], dict] = {}
    model_shape_set = set()
    for r in cur:
        key = (r["model"], r["shape"])
        model_shape_set.add(key)
    for (m, s) in model_shape_set:
        rows = [r for r in cur if r["model"] == m and r["shape"] == s]
        cur_agg[(m, s)] = aggregate(rows)

    regressions: list[str] = []
    improvements: list[str] = []
    unchanged = 0

    print(f"{'model':22} {'shape':14} {'exec Δ':>10} {'score Δ':>10}  note")
    print("-" * 80)
    for (m, s), a_cur in sorted(cur_agg.items()):
        a_base = base.get(m, {}).get(s)
        if not a_base:
            print(f"{m:22} {s:14} {'  new':>10} {'':>10}  (no baseline)")
            continue
        d_exec = a_cur["exec_pass_rate"] - a_base["exec_pass_rate"]
        sc_base = a_base.get("avg_score") or 0.0
        sc_cur = a_cur.get("avg_score") or 0.0
        d_score = sc_cur - sc_base
        note = ""
        if d_exec <= -args.gate:
            note = f"REGRESSION exec drop {d_exec*100:+.0f}pp"
            regressions.append(f"{m}/{s}: exec {d_exec*100:+.0f}pp")
        elif d_score <= -args.score_gate:
            note = f"REGRESSION score drop {d_score:+.1f}"
            regressions.append(f"{m}/{s}: score {d_score:+.1f}")
        elif d_exec >= args.gate or d_score >= args.score_gate:
            note = "improvement"
            improvements.append(f"{m}/{s}")
        else:
            unchanged += 1
        print(f"{m:22} {s:14} {d_exec*100:+9.0f}% {d_score:+9.1f}  {note}")

    print()
    print(f"Summary: {len(regressions)} regressions, {len(improvements)} improvements, "
          f"{unchanged} unchanged")
    if regressions:
        print("\nRegressions:")
        for r in regressions:
            print(f"  - {r}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
