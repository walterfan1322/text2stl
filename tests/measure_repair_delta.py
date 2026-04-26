"""Measure mesh_repair lift: how many historically-generated STLs become
watertight after a repair pass?

Reads every .stl under outputs/ that was generated BEFORE the
mesh_repair hook went live (i.e., any file the server hasn't already
rewritten). Runs repair_stl against a COPY of each so originals are
preserved, then reports:

    - baseline watertight rate (as stored on disk)
    - post-repair watertight rate
    - delta (absolute and relative)
    - break-out by repair method that succeeded

Run: python3 tests/measure_repair_delta.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mesh_repair import repair_stl  # noqa: E402


def main() -> int:
    root = Path(__file__).resolve().parent.parent / "outputs"
    stls = sorted(root.glob("*/model.stl"))
    if not stls:
        print(f"No STLs found under {root}")
        return 1

    tmp_root = Path(tempfile.mkdtemp(prefix="repair_delta_"))
    baseline_wt = 0
    post_wt = 0
    flipped = 0
    already_wt = 0
    methods: Counter = Counter()
    failures = 0
    total = len(stls)

    print(f"Measuring repair delta over {total} STLs...\n")

    for i, src in enumerate(stls, 1):
        dst = tmp_root / f"{i:03d}_{src.parent.name}.stl"
        shutil.copy2(src, dst)
        try:
            r = repair_stl(dst)
        except Exception as e:
            failures += 1
            print(f"  [{i:3d}] {src.parent.name}: ERROR {e}")
            continue

        if not r.loaded:
            failures += 1
            continue

        if r.before_watertight:
            baseline_wt += 1
            already_wt += 1
        if r.after_watertight:
            post_wt += 1
        if not r.before_watertight and r.after_watertight:
            flipped += 1
            methods[r.method] += 1

        if i % 25 == 0 or i == total:
            print(f"  [{i:3d}/{total}] baseline_wt={baseline_wt}  "
                  f"post_wt={post_wt}  flipped={flipped}")

    print()
    print("=== Mesh Repair Delta ===")
    print(f"  Total STLs measured     : {total}")
    print(f"  Load failures           : {failures}")
    usable = total - failures
    if usable == 0:
        print("  Nothing usable.")
        return 1
    print(f"  Baseline watertight     : {baseline_wt}/{usable} "
          f"({baseline_wt/usable*100:.1f}%)")
    print(f"  Post-repair watertight  : {post_wt}/{usable} "
          f"({post_wt/usable*100:.1f}%)")
    delta_abs = post_wt - baseline_wt
    delta_pct = (post_wt - baseline_wt) / usable * 100
    print(f"  Delta (absolute)        : +{delta_abs} "
          f"({delta_pct:+.1f} pp)")
    print(f"  Flipped False→True      : {flipped}")
    print(f"  Already watertight      : {already_wt}")
    if methods:
        print("  Methods that succeeded  :")
        for method, count in methods.most_common():
            print(f"    {method:30s} {count}")

    shutil.rmtree(tmp_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
