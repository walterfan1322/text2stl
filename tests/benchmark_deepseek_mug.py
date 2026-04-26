"""5-trial mug stress test for DeepSeek with the new retry hints + prompts.
Goal: show the pass rate is now meaningfully above the previous 1/2 = 50%.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from benchmark_models import run_one

MODEL = "deepseek-chat"
TRIALS = 5
PROMPT = "一個有把手的馬克杯"

print(f"DeepSeek mug stress test: {TRIALS} trials\n")
results = []
for t in range(TRIALS):
    if t > 0:
        time.sleep(5)  # DeepSeek is not rate-limited strictly
    print(f"[{t+1}/{TRIALS}] {MODEL} mug: ", end="", flush=True)
    r = run_one(MODEL, f"mug_t{t+1}", PROMPT)
    results.append(r)
    tag = "PASS" if r["ok"] else "FAIL"
    sc = r["judge_score"]; sc_s = f"{sc}/10" if sc is not None else "  -"
    extra = (r.get("error") or r.get("judge_category") or "")[:60]
    print(f"{tag}  score={sc_s}  t={r['elapsed_s']:>5.1f}s  attempts={r.get('attempts','?')}  {extra}")

n_pass = sum(1 for r in results if r["ok"])
scores = [r["judge_score"] for r in results if r["judge_score"] is not None]
print(f"\n{'='*60}")
print(f"DeepSeek mug pass rate: {n_pass}/{TRIALS} ({100*n_pass/TRIALS:.0f}%)")
if scores:
    print(f"Avg judge score (passed): {sum(scores)/len(scores):.1f}/10")
print(f"{'='*60}")
