"""Benchmark v2 — 12 shapes × N trials × M models.

Richer than v1:
- Aggregates N trials per (model, shape) and reports mean / variance
- Records watertight flag, volume, attempts, judge_score, judge_category
- Distinguishes judge-unscored (API error) vs judge-scored-low
- Writes raw JSON + a human-readable summary

Usage:
    python3 tests/benchmark_v2.py                    # default: 3 trials
    python3 tests/benchmark_v2.py --trials 1         # smoke
    python3 tests/benchmark_v2.py --models MiniMax-M2.7
    python3 tests/benchmark_v2.py --shapes mug,shoe
    python3 tests/benchmark_v2.py --baseline         # write baseline.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

API = "http://127.0.0.1:8765"
TIMEOUT = 300  # judge loop can take a while on hard shapes

DEFAULT_MODELS = ["MiniMax-M2.7", "deepseek-chat"]

# 12 shapes organised by difficulty tier (tier comment is informational)
DEFAULT_PROMPTS: dict[str, str] = {
    # Tier 1 — revolve / extrude (easy)
    "vase":         "一個花瓶",
    "pen_holder":   "一個圓筒筆筒",
    "bowl":         "一個碗",
    "phone_stand":  "一個簡單的手機立架",
    # Tier 2 — revolve + sweep / loft (medium)
    "mug":          "一個有把手的馬克杯",
    "bottle":       "一個瓶子",
    "teapot":       "一個有把手和壺嘴的茶壺",
    "shoe":         "一隻鞋子的形狀",
    # Tier 3 — composite / primitives (hard)
    "chair":        "一張有四條腿和靠背的椅子",
    "table":        "一張有四條腿的桌子",
    "keychain":     "一個愛心形的鑰匙圈",
    "figurine":     "一個簡單的雪人，三個球疊起來",
}


def post(path: str, body: dict, timeout: int = TIMEOUT) -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{API}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"detail": body[:300]}
    except Exception as e:
        return -1, {"detail": f"request-error: {e}"}


def download_and_check(stl_url: str) -> tuple[bool, dict]:
    """Download STL, load with trimesh, return stats (incl. watertight)."""
    import trimesh  # lazy
    try:
        with urllib.request.urlopen(f"{API}{stl_url}", timeout=30) as r:
            blob = r.read()
        tmp = Path(f"/tmp/bench_{int(time.time()*1000)}.stl")
        tmp.write_bytes(blob)
        m = trimesh.load(str(tmp))
        stats = {
            "size_kb": len(blob) // 1024,
            "vertices": len(m.vertices),
            "faces": len(m.faces),
            "watertight": bool(m.is_watertight),
            "volume_cm3": float(m.volume) / 1000.0 if m.is_volume else None,
        }
        tmp.unlink(missing_ok=True)
        return len(m.vertices) > 10 and len(m.faces) > 10, stats
    except Exception as e:
        return False, {"error": str(e)[:100]}


def run_one(model: str, shape: str, prompt: str, trial: int) -> dict:
    t0 = time.time()
    # ?no_cache=1 bypasses output_cache so trial 2/3 actually re-generate
    # rather than returning the cached trial-1 result instantly. We want
    # to measure variance across trials for evaluation.
    status, resp = post("/api/generate?no_cache=1",
                        {"prompt": prompt, "model": model})
    elapsed = time.time() - t0
    out = {
        "model": model, "shape": shape, "trial": trial,
        "elapsed_s": round(elapsed, 1),
        "exec_ok": False, "attempts": 0,
        "judge_score": None, "judge_category": None, "judge_issues": [],
        "watertight": None,
    }
    if status != 200:
        out["error"] = f"http {status}: {resp.get('detail','?')[:140]}"
        return out

    out["attempts"] = resp.get("attempts", 1)
    judge = resp.get("judge") or {}
    out["judge_score"] = judge.get("match_score")
    out["judge_category"] = judge.get("category")
    out["judge_issues"] = judge.get("geometry_issues", []) or []

    ok, stl_stats = download_and_check(resp.get("stl_url", ""))
    out["exec_ok"] = ok
    out["stl"] = stl_stats
    if isinstance(stl_stats, dict) and "watertight" in stl_stats:
        out["watertight"] = stl_stats["watertight"]
    return out


def aggregate(rows: list[dict]) -> dict:
    """Reduce a list of per-trial rows into mean/variance/pass-rate."""
    n = len(rows)
    exec_ok = sum(1 for r in rows if r["exec_ok"])
    wt = sum(1 for r in rows if r.get("watertight"))
    judged = [r["judge_score"] for r in rows if isinstance(r.get("judge_score"), int)]
    pass_judge = sum(1 for s in judged if s >= 6)
    return {
        "n": n,
        "exec_pass_rate": round(exec_ok / n, 2) if n else 0.0,
        "watertight_rate": round(wt / n, 2) if n else 0.0,
        "judge_scored_n": len(judged),
        "judge_pass_rate": round(pass_judge / n, 2) if n else 0.0,
        "avg_score": round(sum(judged) / len(judged), 1) if judged else None,
        "min_score": min(judged) if judged else None,
        "max_score": max(judged) if judged else None,
        "avg_attempts": round(sum(r["attempts"] for r in rows) / n, 2) if n else 0.0,
        "avg_time_s": round(sum(r["elapsed_s"] for r in rows) / n, 1) if n else 0.0,
    }


def format_summary(results: list[dict], models: list[str], shapes: list[str]) -> str:
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("Overall (all shapes per model):")
    lines.append("-" * 100)
    lines.append(f"{'model':22} {'exec':>6} {'wt':>6} {'judge≥6':>8} "
                 f"{'avgSc':>6} {'avgAtt':>7} {'avgT':>7}")
    for model in models:
        rows = [r for r in results if r["model"] == model]
        agg = aggregate(rows)
        score_s = f"{agg['avg_score']:.1f}" if agg['avg_score'] is not None else "-"
        lines.append(
            f"{model:22} {agg['exec_pass_rate']*100:>5.0f}% "
            f"{agg['watertight_rate']*100:>5.0f}% "
            f"{agg['judge_pass_rate']*100:>7.0f}% "
            f"{score_s:>6} {agg['avg_attempts']:>7.2f} "
            f"{agg['avg_time_s']:>6.1f}s"
        )

    lines.append("")
    lines.append("Per-shape exec pass-rate (model × shape):")
    lines.append("-" * 100)
    header = f"{'shape':14}" + "".join(f"{m:>22}" for m in models)
    lines.append(header)
    for shape in shapes:
        row = f"{shape:14}"
        for model in models:
            rs = [r for r in results if r["model"] == model and r["shape"] == shape]
            if not rs:
                row += f"{'?':>22}"; continue
            agg = aggregate(rs)
            sc = f"{agg['avg_score']:.1f}" if agg['avg_score'] is not None else "-"
            cell = f"{agg['exec_pass_rate']*100:.0f}% ({sc})"
            row += f"{cell:>22}"
        lines.append(row)

    lines.append("")
    lines.append("Notes: exec = STL produced & >10 verts; wt = watertight STL; "
                 "judge≥6 counts only SCORED trials ≥6; avgSc over scored only.")
    lines.append("=" * 100)
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=",".join(DEFAULT_MODELS),
                   help="comma-separated model names")
    p.add_argument("--shapes", default="", help="comma-separated shape keys (default: all)")
    p.add_argument("--trials", type=int, default=3, help="trials per (model, shape)")
    p.add_argument("--sleep", type=float, default=3.0,
                   help="seconds between calls (rate-limit cushion)")
    p.add_argument("--baseline", action="store_true",
                   help="also overwrite tests/baseline.json after a successful run")
    p.add_argument("--out", default="tests/benchmark_v2_results.json",
                   help="raw results JSON output path")
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    shape_keys = [s.strip() for s in args.shapes.split(",") if s.strip()] \
                 or list(DEFAULT_PROMPTS.keys())
    shapes = {k: DEFAULT_PROMPTS[k] for k in shape_keys if k in DEFAULT_PROMPTS}

    total = len(models) * len(shapes) * args.trials
    print(f"Benchmark v2: {len(models)} models × {len(shapes)} shapes × "
          f"{args.trials} trials = {total} runs\n")

    results: list[dict] = []
    i = 0
    t_start = time.time()
    for model in models:
        for key, prompt in shapes.items():
            for trial in range(1, args.trials + 1):
                i += 1
                print(f"[{i:>3}/{total}] {model:22} × {key:12} t{trial} ... ",
                      end="", flush=True)
                r = run_one(model, key, prompt, trial)
                results.append(r)
                tag = "PASS" if r["exec_ok"] else "FAIL"
                sc = r["judge_score"]
                sc_s = f"{sc}/10" if sc is not None else "  -"
                wt = "wt" if r.get("watertight") else "  "
                extra = (r.get("error") or r.get("judge_category") or "")[:30]
                print(f"{tag} {sc_s} {wt} t={r['elapsed_s']:>5.1f}s "
                      f"att={r['attempts']} {extra}")
                time.sleep(args.sleep)

    elapsed_total = time.time() - t_start
    print()
    summary = format_summary(results, models, list(shapes.keys()))
    print(summary)
    print(f"\nTotal wall-clock: {elapsed_total/60:.1f} min")

    # Write raw results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "meta": {
            "models": models,
            "shapes": list(shapes.keys()),
            "trials": args.trials,
            "timestamp": int(time.time()),
        },
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"Raw results: {out_path}")

    # Optionally overwrite baseline
    if args.baseline:
        baseline = {
            model: {
                shape: aggregate([r for r in results
                                  if r["model"] == model and r["shape"] == shape])
                for shape in shapes
            }
            for model in models
        }
        base_path = Path(__file__).parent / "baseline.json"
        base_path.write_text(json.dumps({
            "meta": {"models": models, "shapes": list(shapes.keys()),
                     "trials": args.trials, "timestamp": int(time.time())},
            "baseline": baseline,
        }, indent=2), encoding="utf-8")
        print(f"Baseline written: {base_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
