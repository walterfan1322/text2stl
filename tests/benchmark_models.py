"""Head-to-head benchmark: 3 LLMs × 5 shapes through the full pipeline.

For each (model, prompt) pair: hit /api/generate, then download the STL,
validate it, and record the Gemini judge's score. Summarise at the end.

Run: python3 tests/benchmark_models.py
Requires: server listening on http://127.0.0.1:8765
"""
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

API = "http://127.0.0.1:8765"

MODELS = ["MiniMax-M2.7", "deepseek-chat", "gemini-2.5-flash"]
PROMPTS = {
    "vase":        "一個花瓶",
    "pen_holder":  "一個圓筒筆筒",
    "phone_stand": "一個簡單的手機立架",
    "mug":         "一個有把手的馬克杯",
    "shoe":        "一隻鞋子的形狀",
}

TIMEOUT = 240  # generous; judge + 3 retry slots


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
    """Download STL, load with trimesh, return basic stats."""
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


def run_one(model: str, shape: str, prompt: str) -> dict:
    t0 = time.time()
    status, resp = post("/api/generate", {"prompt": prompt, "model": model})
    elapsed = time.time() - t0
    out = {
        "model": model, "shape": shape, "elapsed_s": round(elapsed, 1),
        "ok": False, "attempts": 0, "judge_score": None, "judge_category": None,
    }
    if status != 200:
        out["error"] = f"http {status}: {resp.get('detail','?')[:120]}"
        return out

    out["attempts"] = resp.get("attempts", 1)
    judge = resp.get("judge") or {}
    out["judge_score"] = judge.get("match_score")
    out["judge_category"] = judge.get("category")

    ok, stl_stats = download_and_check(resp.get("stl_url", ""))
    out["ok"] = ok
    out["stl"] = stl_stats
    return out


def main() -> int:
    results = []
    total = len(MODELS) * len(PROMPTS)
    i = 0
    print(f"Benchmarking {len(MODELS)} models × {len(PROMPTS)} shapes = {total} runs\n")
    for model in MODELS:
        for shape, prompt in PROMPTS.items():
            i += 1
            print(f"[{i:>2}/{total}] {model:22} × {shape:12} ... ", end="", flush=True)
            r = run_one(model, shape, prompt)
            results.append(r)
            tag = "PASS" if r["ok"] else "FAIL"
            score = r["judge_score"]
            score_s = f"{score}/10" if score is not None else "  -"
            extra = (r.get("error") or r.get("judge_category") or "")[:40]
            print(f"{tag}  score={score_s}  t={r['elapsed_s']:>5.1f}s  {extra}")

    # Summarise
    print("\n" + "=" * 72)
    print(f"{'Model':22}  {'Pass':>6}  {'AvgScore':>10}  {'AvgTime':>9}  {'AvgAtt':>7}")
    print("-" * 72)
    for model in MODELS:
        rows = [r for r in results if r["model"] == model]
        passed = sum(1 for r in rows if r["ok"])
        scores = [r["judge_score"] for r in rows if r["judge_score"] is not None]
        times = [r["elapsed_s"] for r in rows]
        atts = [r["attempts"] for r in rows]
        avg_s = f"{sum(scores)/len(scores):.1f}" if scores else "-"
        avg_t = f"{sum(times)/len(times):.1f}s" if times else "-"
        avg_a = f"{sum(atts)/len(atts):.2f}" if atts else "-"
        print(f"{model:22}  {passed:>3}/{len(rows):<2}  {avg_s:>10}  "
              f"{avg_t:>9}  {avg_a:>7}")
    print("=" * 72)

    # Per-shape breakdown
    print("\nPer-shape pass-rate (model x shape):\n")
    header = f"{'shape':14}" + "".join(f"{m:>24}" for m in MODELS)
    print(header)
    print("-" * len(header))
    for shape in PROMPTS:
        row = f"{shape:14}"
        for model in MODELS:
            r = next((x for x in results if x["model"] == model and x["shape"] == shape), None)
            if not r:
                row += f"{'?':>24}"; continue
            mark = "PASS" if r["ok"] else "FAIL"
            sc = r["judge_score"]
            row += f"{f'{mark} ({sc}/10)' if sc is not None else mark:>24}"
        print(row)

    # Dump raw JSON
    out_path = Path(__file__).parent / "benchmark_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nRaw results: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
