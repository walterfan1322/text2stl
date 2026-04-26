"""Age-based cleanup of outputs/ directory.

Each /api/generate call creates a job folder under outputs/. They
accumulate forever unless pruned. This script removes folders older
than N days.

Usage:
    python3 scripts/cleanup_outputs.py                # dry-run, 7 days
    python3 scripts/cleanup_outputs.py --days 3       # 3 days
    python3 scripts/cleanup_outputs.py --execute      # actually delete
    python3 scripts/cleanup_outputs.py --days 7 --execute
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"


def _folder_size_mb(d: Path) -> float:
    try:
        total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    except OSError:
        return 0.0
    return total / (1024 * 1024)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--days", type=int, default=7,
                   help="Remove folders older than this many days (default: 7)")
    p.add_argument("--execute", action="store_true",
                   help="Actually delete (default: dry-run)")
    p.add_argument("--dir", default=str(OUTPUT_DIR))
    args = p.parse_args()

    output_dir = Path(args.dir)
    if not output_dir.exists():
        print(f"No outputs dir at {output_dir}")
        return 0

    cutoff = time.time() - args.days * 86400
    total = 0
    to_remove: list[tuple[Path, float, float]] = []
    for d in output_dir.iterdir():
        if not d.is_dir():
            continue
        total += 1
        try:
            mtime = d.stat().st_mtime
        except OSError:
            continue
        if mtime < cutoff:
            to_remove.append((d, _folder_size_mb(d), mtime))

    to_remove.sort(key=lambda x: x[2])
    total_mb = sum(x[1] for x in to_remove)
    print(f"{len(to_remove)} folder(s) older than {args.days}d "
          f"(of {total} total)  —  reclaimable: {total_mb:.1f} MB")

    if not to_remove:
        return 0

    if not args.execute:
        print("DRY-RUN — pass --execute to delete")
        for d, mb, mt in to_remove[:10]:
            age_d = (time.time() - mt) / 86400
            print(f"  {d.name:12}  {mb:>6.1f} MB   {age_d:>5.1f}d old")
        if len(to_remove) > 10:
            print(f"  ... and {len(to_remove)-10} more")
        return 0

    removed, reclaimed = 0, 0.0
    for d, mb, _ in to_remove:
        try:
            shutil.rmtree(d)
            removed += 1
            reclaimed += mb
        except Exception as e:
            print(f"  failed: {d.name}: {e}")
    print(f"Removed {removed} folder(s), reclaimed {reclaimed:.1f} MB")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
