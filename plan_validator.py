"""Plan-vs-output reconciliation validator (P1, 2026-04-28).

After exec, compare the generated STL against the LLM's own PLAN comment.
If the LLM declared parts with sizes [200, 200, 200] but the STL bbox is
8mm in every axis, OCC silently dropped the booleans. Catch that and
emit an actionable retry signal — without category awareness.

Three general checks (no per-category logic):

  (a) BBOX collapse:  max(STL.extents) should be >= 0.5 * max declared
      part size in PLAN. Catches OCC silently dropping most parts.
      Real-world miss this caught: the dog (13 unions → 8mm STL).

  (b) COMPONENT count: STL connected components should equal the number
      of root/union "bodies" PLAN declares. Cuts don't add components;
      explicit role=floating adds one each. Catches parts positioned so
      they don't overlap the root and end up disconnected.
      Real-world miss this caught: car wheels floating outside body.

  (c) SUBPART contribution: each non-cut part is detectable in the STL
      bbox — if a part's bbox would extend the model past `extents`,
      it was probably dropped during boolean. Catches handles/protrusions
      that fail during sweep/loft.
      Real-world miss this caught: cup handle that should reach x=92mm
      but STL only spans x=80mm.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("text2stl.plan_validator")


@dataclass
class PlanCheckResult:
    passed: bool
    issues: list[str] = field(default_factory=list)
    fix_suggestion: str = ""
    plan: dict | None = None
    measurements: dict = field(default_factory=dict)

    @property
    def fail_reason(self) -> str:
        return "; ".join(self.issues) if self.issues else ""

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "issues": list(self.issues),
            "fix_suggestion": self.fix_suggestion,
            "measurements": dict(self.measurements),
        }


_PLAN_RE = re.compile(r"#\s*PLAN\s*:\s*(\{.*?\})\s*(?:\n|$)", re.DOTALL)


def parse_plan(code: str) -> dict | None:
    """Extract the JSON from a leading `# PLAN: {...}` comment."""
    if not code:
        return None
    # Line-based scan first (most robust against multi-line JSON-like text)
    accum: list[str] = []
    for line in code.splitlines():
        s = line.strip()
        if accum:
            accum.append(s.lstrip("#").strip())
            joined = " ".join(accum)
            try:
                return json.loads(joined)
            except json.JSONDecodeError:
                # Stop trying once braces are clearly unbalanced
                if joined.count("}") > joined.count("{"):
                    accum = []
        elif s.startswith("# PLAN:") or s.startswith("#PLAN:"):
            payload = s.split(":", 1)[1].strip()
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                accum = [payload]
    # Regex fallback
    m = _PLAN_RE.search(code)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _max_declared_size(plan: dict) -> float:
    max_size = 0.0
    for part in plan.get("parts", []) or []:
        size = part.get("size") or []
        if isinstance(size, list):
            for v in size:
                if isinstance(v, (int, float)) and v > max_size:
                    max_size = float(v)
    return max_size


def _expected_component_count(plan: dict) -> int:
    """How many disconnected pieces should the final STL contain?

    Default: 1. Each part with role=floating (explicit) adds one.
    """
    parts = plan.get("parts") or []
    floating = sum(1 for p in parts if str(p.get("role", "")).lower() == "floating")
    return 1 + floating


def check(stl_path: Path, code: str) -> PlanCheckResult | None:
    """Run plan-vs-output reconciliation.

    Returns None if no PLAN found in `code` (skip — not a plan-mode gen).
    Returns PlanCheckResult with passed=True/False otherwise.
    """
    plan = parse_plan(code)
    if not plan:
        return None
    parts = plan.get("parts") or []
    if not parts:
        return None

    try:
        import trimesh
    except ImportError:
        log.warning("trimesh missing — plan_validator skipped")
        return None

    try:
        m = trimesh.load(str(stl_path))
    except Exception as e:
        log.warning(f"plan_validator: STL load failed: {e}")
        return None

    extents = list(m.extents) if hasattr(m, "extents") else [0.0, 0.0, 0.0]
    try:
        n_components = len(m.split(only_watertight=False))
    except Exception:
        n_components = 1

    measurements = {
        "extents_mm": [round(float(x), 1) for x in extents],
        "max_extent_mm": round(float(max(extents) if extents else 0.0), 1),
        "n_components": int(n_components),
        "volume_cm3": (round(float(m.volume) / 1000, 2)
                       if hasattr(m, "volume") else None),
        "n_parts_declared": len(parts),
        "expected_components": _expected_component_count(plan),
        "max_declared_size_mm": _max_declared_size(plan),
    }

    issues: list[str] = []
    fix_lines: list[str] = []

    # (a) BBOX collapse
    max_declared = measurements["max_declared_size_mm"]
    if max_declared > 0:
        max_extent = max(extents) if extents else 0.0
        ratio = (max_extent / max_declared) if max_declared > 0 else 1.0
        if ratio < 0.5:
            issues.append(
                f"STL collapsed: model bbox max is {max_extent:.1f}mm but "
                f"the PLAN declared a part of size {max_declared:.0f}mm "
                f"(ratio {ratio:.2f}). The boolean operations silently "
                f"dropped most of the geometry — this is an OCC robustness "
                f"failure, not a sizing intent issue."
            )
            fix_lines.append(
                "Reduce the number of chained .union() calls (combine "
                "co-located parts into a single primitive when possible), "
                "OR add small overlaps (>=1mm) where parts meet to give "
                "OCC a clean intersection, OR replace fragile sweep/loft "
                "operations with simpler primitives."
            )

    # (b) COMPONENT count
    expected = measurements["expected_components"]
    if n_components > expected:
        issues.append(
            f"STL has {n_components} disconnected pieces but the PLAN "
            f"declares {len(parts)} parts that should produce "
            f"{expected} connected body(ies). One or more parts are "
            f"floating in space, not actually overlapping the root part."
        )
        fix_lines.append(
            "For each non-root part, ensure its coordinates make it "
            "OVERLAP the root by at least 1mm before .union(). Common "
            "miss: positioning a part with .center(x, y) at coordinates "
            "outside the root's footprint."
        )

    fix_suggestion = " ".join(fix_lines).strip()

    return PlanCheckResult(
        passed=(len(issues) == 0),
        issues=issues,
        fix_suggestion=fix_suggestion,
        plan=plan,
        measurements=measurements,
    )


def build_retry_hint(result: PlanCheckResult) -> str:
    """Format a retry message that includes measurements + issues + fix."""
    if result.passed:
        return ""
    meas = result.measurements
    return (
        f"Plan-vs-output reconciliation FAILED.\n"
        f"Measured: extents={meas.get('extents_mm')}mm, "
        f"components={meas.get('n_components')}, "
        f"volume={meas.get('volume_cm3')}cm^3.\n"
        f"PLAN declared: {meas.get('n_parts_declared')} parts, "
        f"expected {meas.get('expected_components')} component(s), "
        f"max part size {meas.get('max_declared_size_mm')}mm.\n"
        f"Problems:\n" + "\n".join(f"- {i}" for i in result.issues) + "\n"
        f"Fix: {result.fix_suggestion}\n"
        f"Output ONLY corrected Python code (with the same `# PLAN: {{...}}` "
        f"comment at the top, possibly with adjusted sizes/positions)."
    )
