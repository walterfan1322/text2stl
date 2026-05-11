# Quality Fix Plan — 8-category survey 2026-04-28

## Survey results (8 prompts)

| Category | judge score | attempts | Real visual quality |
|---|---|---|---|
| 鑰匙 key | 8 | 1 | ✅ Genuinely looks like a key |
| 椅子 chair | 8 | 2 | ❌ Legs float 150mm below seat; backrest is just 3 sticks |
| 馬克杯 cup | 8 | 1 | ❌ Handle ring detached, passes through body centre |
| 房子 house | 8 | 1 | ⚠️ Stacked boxes, no roof slope, no door/window |
| 汽車 car | 7 | 1 | ⚠️ Wheels don't touch body; cabin floats above |
| 狗 dog | 1 | 3 (retry exhausted) | ❌ Giant sphere + random parts |
| 錘子 hammer | 1 | 3 (retry exhausted) | ❌ Vertical handle, head pierces middle |
| 樹 tree | — | — | 💥 HTTP 500 — `.circle().close().extrude()` ValueError |

→ 5 nominally pass / 2 retry-exhausted / 1 server crash. Of the 5 "passing", only **key** is actually good.

---

## Seven systemic problems

### P1. Judge too lenient (false positives)
Judge LLM lists real issues but still scores 8/10:
- chair: `"legs are not attached to the seat"` + `"backrest lacks a solid surface"` → 8
- cup: `"handle is not fully attached"` → 8
- house: `"roof is flat and lacks pitch"`, `"no doors or windows"` → 8

Score decoupled from listed issues. `JUDGE_MIN_SCORE=6` effectively useless.

### P2. No "parts touching" check
- chair: legs at z=-150..150, seat at z=300+ → 150mm air gap (verified in generated code)
- cup: handle in XZ plane sweep, body revolved → no shared volume
- car: wheels at z=15..27, body at z=0..40 → wheels embedded in body, not below

Validators only check AST/syntax. STL with N disconnected components passes.

### P3. Object orientation not constrained
- hammer: `cq.Workplane("XY").cylinder(130, 12)` → handle along Z-axis (should be horizontal)
- dog: revolve around Y-axis body + XY-axis head/legs → coordinate frame mismatch

`prompts/system_cadquery.md` has no per-category default orientation.

### P4. Organic shapes have no canonical recipe
Working patterns are all rigid composites (chair=box+legs, cup=revolve, key=extrusion). Animals/plants/irregular natural forms have no Route-A-style template. LLM ad-hoc-composes primitives → spatial chaos.

### P5. Recurring CadQuery API misuse: `.circle(N).close().extrude(...)`
Tree failed all 3 attempts with `ValueError: Cannot convert object type Wire to vector`. `.circle()` already returns a closed wire; `.close()` then `.extrude()` corrupts the operation. Validator doesn't catch.

### P6. HTTP 500 on retry exhaustion
When all 3 attempts fail at exec stage, `app.py:1849-1852` raises `HTTPException(500)` instead of returning structured `{success: false, last_error, attempts, code}`.

### P7. Reference-image vision feedback too generic
For tree, vision said `"main stem (cylinder) with conical top"` — no proportions, no anchor points, no attachment rules. Doesn't constrain bad LLM recipes.

---

## Confirmed code locations

| Concern | File:line |
|---|---|
| Judge system prompt | `judge.py:26-57` (`JUDGE_SYSTEM_PROMPT`) |
| Judge response parser | `judge.py:74-111` (`JudgeResult.from_response`) |
| Retry loop | `app.py:1797-2039` (`_generate_impl`) |
| Retry-exhaustion 500 | `app.py:1849-1852` |
| Pre-judge geom gate hook | `app.py:1908-1946` and `app.py:1481-1492` |
| Watertight gate (template) | `app.py:1295-1308`, `app.py:1875-1906` |
| Vision-analysis prompt | `app.py:619-650` (lines 622-628) |
| AST validators | `validators.py:380-509`; loft check at 512-589 (template) |
| System prompt patterns | `prompts/system_cadquery.md:68-281` (8 patterns) |
| `JUDGE_MIN_SCORE` | `app.py:133`; `config.json:67` |
| Category inference | `pattern_cache.py:32-55` (`CATEGORY_KEYWORDS`) |

---

## Priority order (revised from initial proposal)

**P5 → P6 → P1+P2 → P3 → P4 → P7**

Rationale:
1. **P5 first** — cheap, isolated, immediately unblocks tree
2. **P6 second** — needed before P2 because P2's new fail mode (multi-component) will trigger retry-exhaustion. Without P6, P2 makes the user-visible 500 problem worse
3. **P1+P2 together** — share the same retry loop at `app.py:1797`; doing them together avoids touching same code twice; P1 must merge first because P2's connectivity-fail emits `geometry_issues` that P1's clamp consumes
4. **P3 before P4** — P4's recipes need a fixed coordinate convention from P3
5. **P4** is heaviest (pure prompt, can iterate without Python)
6. **P7 last** — small reach into LLM output, pipeline works without it

---

## Per-fix sequencing

### P5 — AST closed-primitive misuse check (~30 lines, `validators.py`)
- New `check_closed_primitive_misuse(code) -> list[str]` next to `check_loft_topology` (line 512)
- Walk AST. For each `ast.Call` where `func.attr == "close"`, inspect receiver: if it's a `Call` with `func.attr in {"circle","rect","ellipse"}` immediately preceding (no `.lineTo`/`.threePointArc` between), emit error pointing at fix
- Wire into `validate_cadquery` at lines 493-509 like `check_loft_topology`
- **Risk**: false positive if `.circle()` is part of a larger sketch with subsequent segments. Mitigate by checking the chain is direct (receiver check)

### P6 — Structured 200 on retry exhaustion (~10 lines, `app.py`)
- At `app.py:1849-1852`: replace `raise HTTPException(500)` with `return GenerateResponse(success=False, id=job_id, code=last_code, attempts=N, judge={"category":"exec_failed","match_score":1,"geometry_issues":[last_error]}, ...)`
- Add `success: bool = True` to `GenerateResponse` model (line 743)
- SSE path at `app.py:2078-2089` already handles errors structured — only legacy blocking endpoint affected
- **Risk**: benchmark scripts (`tests/benchmark_*.py`) check `status_code == 200` only — must update to read `success`

### P1 — Strict judge prompt + server-side clamp (`judge.py`)
- Edit `JUDGE_SYSTEM_PROMPT` (lines 26-57): add scoring rubric paragraph between schema (41) and examples (46):
  > Scoring rule (mandatory): If `geometry_issues` is non-empty, `match_score` MUST be ≤ 5. If any issue describes missing/disconnected/wrong-orientation parts, `match_score` MUST be ≤ 4. Only score 7+ when issues empty AND silhouette unambiguously the requested object.
- Server-side safety net at `judge.py:104-111` (`JudgeResult.from_response`): after parsing, if `geometry_issues` non-empty AND `score > 5`, clamp to 5 and log. Add `clamped_from` field for telemetry
- **Risk**: VLM hallucinates trivial issues (color, surface) → punishes good outputs. Mitigate by prompt wording "geometry-only issues" + telemetry to monitor clamp rate

### P2 — Connected-component check (`app.py`)
- New `_check_connected(stl_path) -> tuple[int, str]` next to `_check_watertight` at line 1295:
  ```
  m = trimesh.load_mesh(stl_path); parts = m.split(only_watertight=False)
  return len(parts), f"{len(parts)} disconnected component(s)"
  ```
- New `FEATURE_CONNECTED_GATE` near `app.py:147-149`; `connected_gate_enabled: true` in `config.json:73`
- Hook in retry loop **before** watertight gate (line 1875). On `parts > 1`:
  - Build `judge={"category":"disconnected_parts","match_score":2,"geometry_issues":[...]}` 
  - Append category-specific retry hint (chair: "legs MUST overlap seat ≥5mm"; mug: "handle endpoints MUST land in body wall")
  - `continue` to next round (mirror watertight pattern at 1901-1906)
- **Risk**: legitimate compound scenes — but `system_cadquery.md` already enforces "single-piece" convention via shoe pattern

### P3 — Per-category orientation in `system_cadquery.md`
- Insert section between line 41 (after 2D→3D ops) and line 43 (Boolean ops). Title: **"DEFAULT ORIENTATION (per category)"**
- Tabular mapping:
  - hammer/screwdriver/wrench: handle along **+X**, head at **+X end**, struck face on **-Z**
  - vehicle: length **X**, height **Z**, wheels protrude **-Z**
  - quadruped: body length **+X**, head at **+X end**, legs extend **-Z**, tail at **-X end**
  - tree/plant: trunk along **+Z**, foliage at **+Z end**
  - house/building: longest wall **+X**, roof apex at **+Z end**, door on **-Y face**
- Pure prompt content — safe wrt locked patterns
- **Risk**: LLM inconsistent application — mitigate with one explicit hammer example after pattern #6

### P4 — Canonical recipes for organic shapes (`system_cadquery.md`)
- Append two patterns after pattern #8 (line 281):
  - **#9 Quadruped (dog) — two-silhouette intersect** (mirror shoe Route-A): side XZ × top XY intersect for body shell, then validated leg cylinders + ear/tail anchors
  - **#10 Tree — revolved trunk + scaled foliage** at z-anchor: trunk z=0..200; foliage `sphere(80)` at offset=180 (MANDATORY 20mm overlap, explicitly note for P2)
- Extend `pattern_cache.py:32-46` `CATEGORY_KEYWORDS`: add `dog/狗`, `tree/樹`, `hammer/錘子`, `car/汽車`, `house/房子`
- **Risk**: new patterns need empirical iteration like shoe did — out of scope this round, ship as v1

### P7 — Tighten vision-analysis prompt (`app.py:622-628`)
- Replace prose prompt with structured template demanding ratios + attachment points:
  > This is a {object_name}. Fill in:
  > • Main parts (2-5): _____
  > • Primitive shapes per part: _____
  > • Bounding box ratio (L:W:H): _____
  > • For each part, attachment point (which other part it shares volume with): _____
  > • For elongated parts, axis (X/Y/Z): _____
  > Skip color, texture, surface detail.
- Bump `num_predict` 200 → 300 at line 633 to fit
- **Risk**: structured prompts may degrade smaller VLMs (qwen2.5vl:7b). Watch chair/mug regression; gate behind feature flag if needed

---

## Out of scope this round

- Not migrating off Gemini judge chain
- Not adding categories beyond dog/tree/hammer/car/house
- Not modifying `judge_geometric.py` rules (P2 lives in `app.py` as universal gate)
- Not touching `refine_patch.py` / `/api/refine` (same exhaustion bug, lower impact)
- No tests for new patterns (manual benchmark cycle validates)
- No `index.html` UI work for `success=False` beyond minimum

---

## Risks summary

1. **P1 clamp aggression** — trivial-issue false positives. Mitigate via prompt wording + `clamped_from` telemetry
2. **P2 false positive on intentional pairs** — `system_cadquery.md` already enforces single-piece convention; document in new gate
3. **P5 misses dynamic builders** — e.g. `wp = .circle(...); wp.close().extrude(...)` across statements. Acceptable; failing tree case is single-chain
4. **P6 silently swallows 500s** — benchmark consumers must read `success` field. Update at least `tests/benchmark_smoke.py` and `tests/benchmark_v2.py`
5. **`MIN_SCORE_TO_CACHE=8` legacy entries** — current `pattern_cache` entries scored under lenient regime. Consider one-time invalidation, or live with stale-but-working cache until natural eviction

---

## Critical files

- `E:\github-projects\text2stl\app.py`
- `E:\github-projects\text2stl\judge.py`
- `E:\github-projects\text2stl\validators.py`
- `E:\github-projects\text2stl\prompts\system_cadquery.md`
- `E:\github-projects\text2stl\pattern_cache.py`
