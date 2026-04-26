"""Mug-only mini-benchmark to verify new few-shot example works across models.
Runs 2 trials per model, 25s spacing to respect Gemini RPM.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from benchmark_models import run_one

MODELS = ["MiniMax-M2.7", "deepseek-chat", "gemini-2.5-flash-lite"]
TRIALS = 2
PROMPT = "一個有把手的馬克杯"

print(f"Mug fix verification: {len(MODELS)} models × {TRIALS} trials = {len(MODELS)*TRIALS} runs\n")

results = []
for i, model in enumerate(MODELS):
    for t in range(TRIALS):
        idx = i * TRIALS + t
        if idx > 0:
            time.sleep(25)  # Gemini RPM cool-off
        print(f"[{idx+1}] {model:22} trial {t+1}: ", end="", flush=True)
        r = run_one(model, f"mug_t{t+1}", PROMPT)
        results.append(r)
        tag = "PASS" if r["ok"] else "FAIL"
        sc = r["judge_score"]; sc_s = f"{sc}/10" if sc is not None else "  -"
        extra = (r.get("error") or r.get("judge_category") or "")[:50]
        print(f"{tag}  score={sc_s}  t={r['elapsed_s']:>5.1f}s  {extra}")

# Summary
print("\n" + "=" * 60)
for model in MODELS:
    rows = [r for r in results if r["model"] == model]
    passed = sum(1 for r in rows if r["ok"])
    scores = [r["judge_score"] for r in rows if r["judge_score"] is not None]
    avg_s = f"{sum(scores)/len(scores):.1f}" if scores else "-"
    print(f"{model:24}  {passed}/{len(rows)} pass, avg judge score: {avg_s}")
print("=" * 60)

# Dump URLs of passing runs so we can visually verify
print("\nDownload URLs for visual verification:")
for r in results:
    if r.get("ok"):
        jid = ""
        # the run_one doesn't surface id directly; find via stl_url later
        jid = r.get("stl", {}).get("id", "?")
        print(f"  {r['model']:24} {r['shape']}  score={r['judge_score']}")
