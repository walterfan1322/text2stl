"""Retry the 4 Gemini-as-generator runs that hit 429 earlier.
Wait 25s between requests (Gemini free tier: 15 RPM, each generate = 3 calls).
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from benchmark_models import run_one, PROMPTS  # reuse

MODEL = "gemini-2.5-flash-lite"  # 2.5-flash daily quota exhausted
SHAPES = ["pen_holder", "phone_stand", "mug", "shoe", "vase"]

print(f"Retrying {len(SHAPES)} Gemini shapes with 25s spacing...\n")
results = []
for i, shape in enumerate(SHAPES, 1):
    if i > 1:
        print(f"     (sleeping 25s for RPM budget)"); time.sleep(25)
    print(f"[{i}/{len(SHAPES)}] {MODEL} × {shape:12} ... ", end="", flush=True)
    r = run_one(MODEL, shape, PROMPTS[shape])
    results.append(r)
    tag = "PASS" if r["ok"] else "FAIL"
    sc = r["judge_score"]
    sc_s = f"{sc}/10" if sc is not None else "  -"
    extra = (r.get("error") or r.get("judge_category") or "")[:40]
    print(f"{tag}  score={sc_s}  t={r['elapsed_s']:>5.1f}s  {extra}")

n_pass = sum(1 for r in results if r["ok"])
scores = [r["judge_score"] for r in results if r["judge_score"] is not None]
print(f"\nGemini retry: {n_pass}/{len(results)} passed, avg score="
      f"{sum(scores)/len(scores):.1f}" if scores else "Gemini retry: 0 passed")
