"""Aggregate token_usage.jsonl into a weekly cost summary.

Reads BASE_DIR/token_usage.jsonl, groups by (date, model), prints
table of prompt/completion/total tokens and estimated USD per day.

Usage:
    python3 tests/cost_summary.py             # all time
    python3 tests/cost_summary.py --days 7    # last 7 days
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent / "token_usage.jsonl"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--path", default=str(LOG))
    args = p.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"No log at {path}. Run some generations first.")
        return 1

    cutoff_ts = int(time.time()) - args.days * 86400

    # (date_str, model) → aggregate
    buckets: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"prompt": 0, "completion": 0, "total": 0, "usd": 0.0, "calls": 0}
    )

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("ts", 0) < cutoff_ts:
                continue
            day = datetime.fromtimestamp(e["ts"]).strftime("%Y-%m-%d")
            b = buckets[(day, e["model"])]
            b["prompt"]     += e.get("prompt_tokens", 0)
            b["completion"] += e.get("completion_tokens", 0)
            b["total"]      += e.get("total_tokens", 0)
            b["usd"]        += e.get("est_usd", 0.0)
            b["calls"]      += 1

    print(f"{'date':12} {'model':22} {'calls':>6} {'prompt':>10} "
          f"{'completion':>11} {'total':>10} {'est USD':>10}")
    print("-" * 86)
    total_usd = 0.0
    for (day, model), b in sorted(buckets.items()):
        print(f"{day:12} {model:22} {b['calls']:>6} {b['prompt']:>10} "
              f"{b['completion']:>11} {b['total']:>10} {b['usd']:>10.4f}")
        total_usd += b["usd"]
    print("-" * 86)
    print(f"Total estimated USD over last {args.days} day(s): {total_usd:.4f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
