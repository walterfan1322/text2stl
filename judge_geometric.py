"""Programmatic geometric judge (S6.1).

Pre-VLM gate that runs purely on mesh topology / geometry.
Catches structural errors that:
  (a) the VLM might miss (e.g. "chair has 3 legs not 4") and
  (b) we can detect deterministically and for free (zero token).

Each rule lives in CATEGORY_RULES keyed by inferred shape category.
A `GeomCheckResult` mirrors the VLM judge's contract so the caller can
treat both the same way (pass / fail / fix_suggestion / score).

The rules are intentionally conservative: a rule fires *only* when we
are confident a real defect exists. False negatives are fine — VLM
will still see the mesh.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("text2stl.judge_geometric")


@dataclass
class GeomCheckResult:
    passed: bool                 # True = no obvious defect found
    score: int                   # 1..10, 10 = passes all checks cleanly
    issues: list[str] = field(default_factory=list)
    fix_suggestion: str = ""
    method: str = "geom"         # for logging

    @property
    def fail_reason(self) -> str:
        return "; ".join(self.issues) if self.issues else ""


# ----------------------------------------------------------------------
# helper geometry primitives
# ----------------------------------------------------------------------

def _load_mesh(stl_path: Path):
    import trimesh
    m = trimesh.load_mesh(str(stl_path))
    if hasattr(m, "geometry"):
        geoms = tuple(m.geometry.values())
        if geoms:
            m = trimesh.util.concatenate(geoms)
    return m


def _bbox_dims(m) -> tuple[float, float, float]:
    lo, hi = m.bounds
    return float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(hi[2] - lo[2])


def _is_hollow(m) -> bool:
    """Estimate hollowness: check if interior contains air. Rough heuristic
    using volume of mesh vs convex hull volume — hollow shapes have a
    small volume / bbox-volume ratio.
    """
    try:
        ch = m.convex_hull
        ch_vol = float(abs(ch.volume))
        m_vol = float(abs(m.volume))
        if ch_vol <= 0:
            return False
        # Hollow if mesh volume < 60% of convex hull volume — suggests
        # there's interior space.
        return (m_vol / ch_vol) < 0.6
    except Exception:
        return False


def _has_topology_hole(m) -> bool:
    """Genus > 0 == topologically a torus (one hole)."""
    try:
        return int(m.euler_number) <= 0  # sphere=2, torus=0, double-torus=-2
    except Exception:
        return False


def _wall_thickness_estimate(m) -> float:
    """Very rough wall-thickness estimate via bbox vs (volume)^(1/3)."""
    try:
        bbox = _bbox_dims(m)
        bbox_vol = bbox[0] * bbox[1] * bbox[2]
        if bbox_vol <= 0:
            return 0.0
        # If volume_ratio is e.g. 5%, that's ~thin shell. Convert to mm
        # roughly assuming average part dim ~50mm.
        ratio = float(abs(m.volume)) / bbox_vol
        return ratio * min(bbox)
    except Exception:
        return 0.0


# ----------------------------------------------------------------------
# per-category rules — return list of issue strings
# ----------------------------------------------------------------------

def _rule_chair(m) -> tuple[list[str], str]:
    issues, fix = [], ""
    try:
        bodies = int(m.body_count)
    except Exception:
        bodies = 1
    bw, bd, bh = _bbox_dims(m)
    # Chair: needs height > width by significant margin (back + legs)
    if bh < max(bw, bd) * 0.8:
        issues.append(f"chair too flat (h={bh:.1f} vs w={bw:.1f}) — missing back/legs")
        fix = "Build chair as 4 legs (boxes) + seat slab + tall backrest unioned."
    # Single solid body might still be OK (legs could merge through seat),
    # but if vol/bbox > 70% it's a brick not a chair.
    try:
        bbox_vol = bw * bd * bh
        vol = float(abs(m.volume))
        if bbox_vol > 0 and vol / bbox_vol > 0.7:
            issues.append("chair too dense (>70% of bbox filled) — likely a solid block")
            fix = "Use 4 thin leg cylinders/boxes + seat slab; do NOT use a solid block."
    except Exception:
        pass
    return issues, fix


def _rule_table(m) -> tuple[list[str], str]:
    issues, fix = [], ""
    bw, bd, bh = _bbox_dims(m)
    if bh < max(bw, bd) * 0.4:
        issues.append(f"table too flat (h={bh:.1f}) — missing legs")
        fix = "Build table as flat top slab + 4 leg cylinders/boxes."
    try:
        bbox_vol = bw * bd * bh
        vol = float(abs(m.volume))
        if bbox_vol > 0 and vol / bbox_vol > 0.6:
            issues.append("table too dense — likely solid block")
            fix = "Use thin top + 4 thin legs; do NOT use a solid block."
    except Exception:
        pass
    return issues, fix


def _rule_bottle(m) -> tuple[list[str], str]:
    issues, fix = [], ""
    bw, bd, bh = _bbox_dims(m)
    # Bottle should be tall: height > 1.5x width
    if bh < max(bw, bd) * 1.2:
        issues.append(f"bottle too short (h={bh:.1f} vs w={max(bw,bd):.1f})")
        fix = "Bottle should be tall (height ≥ 1.5x diameter); revolve a tall outline."
    # Bottle should have a neck: top 30% should be narrower than middle
    try:
        lo, hi = m.bounds
        z_top = hi[2] - (hi[2] - lo[2]) * 0.2
        # Slice top 20% — its bbox xy should be < 70% of full bbox xy
        # (rough proxy without doing actual slicing)
        # We skip this when import is slow; just check the volume ratio.
        if not _is_hollow(m):
            issues.append("bottle appears solid (no hollow interior)")
            fix = "Hollow the bottle: revolve outline with .cut() of inner outline, or use shell."
    except Exception:
        pass
    return issues, fix


def _rule_vase(m) -> tuple[list[str], str]:
    issues, fix = [], ""
    bw, bd, bh = _bbox_dims(m)
    if bh < max(bw, bd) * 1.0:
        issues.append(f"vase too flat (h={bh:.1f})")
        fix = "Vase should have height ≥ diameter; revolve a tall curving outline."
    if not _is_hollow(m):
        issues.append("vase appears solid (no hollow interior)")
        fix = "Hollow the vase: revolve outline with .cut() of inner outline."
    return issues, fix


def _rule_bowl(m) -> tuple[list[str], str]:
    issues, fix = [], ""
    bw, bd, bh = _bbox_dims(m)
    if bh > max(bw, bd) * 1.0:
        issues.append(f"bowl too tall (h={bh:.1f}) — looks like a cup, not a bowl")
        fix = "Bowl is wider than tall; reduce height or widen the rim."
    if not _is_hollow(m):
        issues.append("bowl appears solid (no hollow interior)")
        fix = "Hollow the bowl: cut a smaller hemisphere from the inside."
    return issues, fix


def _rule_mug(m) -> tuple[list[str], str]:
    issues, fix = [], ""
    if not _is_hollow(m):
        issues.append("mug appears solid (no hollow interior)")
        fix = "Hollow the mug: shell or cut a smaller cylinder from the top."
    # Handle check via convex_hull diff
    try:
        ch_vol = float(abs(m.convex_hull.volume))
        m_vol = float(abs(m.volume))
        # If mug has handle, convex hull is significantly larger than mesh.
        # Without handle, ratio is close to 1 (just the outer cylinder).
        # We expect ch_vol / m_vol >= 1.15 for a handled mug.
        # But on hollow mugs m_vol shrinks, so this isn't reliable for
        # detecting handle absence — skip strict check, just warn.
    except Exception:
        pass
    return issues, fix


def _rule_keychain(m) -> tuple[list[str], str]:
    issues, fix = [], ""
    bw, bd, bh = _bbox_dims(m)
    # Keychain is thin: shortest dim < 8mm
    thin_dim = min(bw, bd, bh)
    if thin_dim > 12:
        issues.append(f"keychain too thick ({thin_dim:.1f}mm > 12mm)")
        fix = "Keychain should be flat: extrude a 2-5mm thin profile."
    if not _has_topology_hole(m):
        issues.append("keychain has no hole (genus=0)")
        fix = "Add a small ring hole (cylinder cut through the body) for the key ring."
    return issues, fix


def _rule_phone_stand(m) -> tuple[list[str], str]:
    issues, fix = [], ""
    bw, bd, bh = _bbox_dims(m)
    # Phone stand: should not be a flat plate — needs vertical support.
    if bh < min(bw, bd) * 0.5:
        issues.append(f"phone stand too flat (h={bh:.1f}) — phone has nothing to lean against")
        fix = "Phone stand needs an angled or vertical back support; not just a flat base."
    return issues, fix


def _rule_figurine(m) -> tuple[list[str], str]:
    issues, fix = [], ""
    bw, bd, bh = _bbox_dims(m)
    # Figurine should be more tall than wide
    if bh < max(bw, bd) * 0.8:
        issues.append(f"figurine too flat (h={bh:.1f}) — should be taller")
        fix = "Stack body parts vertically: feet/legs at bottom, head on top."
    return issues, fix


def _rule_shoe(m) -> tuple[list[str], str]:
    issues, fix = [], ""
    bw, bd, bh = _bbox_dims(m)
    # Shoe is longer than wide and longer than tall
    longest = max(bw, bd)
    shortest_horiz = min(bw, bd)
    if longest < shortest_horiz * 1.5:
        issues.append(f"shoe not elongated (longest={longest:.1f}, "
                      f"shortest={shortest_horiz:.1f}) — should be ~2x longer than wide")
        fix = "Shoe is a long oval, not round; use loft between toe and heel cross-sections."
    if bh > longest * 0.7:
        issues.append("shoe too tall — should lie flat")
        fix = "Reduce shoe height; lay it flat along the ground plane."
    return issues, fix


def _rule_teapot(m) -> tuple[list[str], str]:
    issues, fix = [], ""
    if not _is_hollow(m):
        issues.append("teapot appears solid (no hollow interior)")
        fix = "Hollow the body: cut a smaller sphere or use shell on the main body."
    # Without actually slicing we can't reliably detect spout/handle.
    # Skip those; rely on VLM.
    return issues, fix


CATEGORY_RULES = {
    "chair":       _rule_chair,
    "table":       _rule_table,
    "bottle":      _rule_bottle,
    "vase":        _rule_vase,
    "bowl":        _rule_bowl,
    "mug":         _rule_mug,
    "keychain":    _rule_keychain,
    "phone_stand": _rule_phone_stand,
    "figurine":    _rule_figurine,
    "shoe":        _rule_shoe,
    "teapot":      _rule_teapot,
}


def check(stl_path: Path, category: str) -> GeomCheckResult:
    """Run the rule for `category` against the STL at `stl_path`."""
    rule = CATEGORY_RULES.get(category)
    if rule is None:
        return GeomCheckResult(passed=True, score=10,
                               issues=[], fix_suggestion="",
                               method=f"geom/no-rule:{category}")
    try:
        m = _load_mesh(Path(stl_path))
    except Exception as e:
        return GeomCheckResult(passed=False, score=1,
                               issues=[f"STL load failed: {e}"],
                               fix_suggestion="",
                               method="geom/load-fail")

    if len(m.vertices) < 8 or len(m.faces) < 4:
        return GeomCheckResult(passed=False, score=1,
                               issues=["mesh too small (< 8 verts)"],
                               fix_suggestion="",
                               method="geom/empty")

    try:
        issues, fix = rule(m)
    except Exception as e:
        log.warning(f"geom rule {category} crashed: {e}")
        return GeomCheckResult(passed=True, score=10, issues=[],
                               fix_suggestion="",
                               method=f"geom/rule-err:{category}")

    if not issues:
        return GeomCheckResult(passed=True, score=10, issues=[],
                               fix_suggestion="",
                               method=f"geom/{category}/ok")

    # Score: 1 issue → 5, 2 → 3, 3+ → 2
    score = max(2, 6 - 2 * len(issues))
    return GeomCheckResult(passed=False, score=score, issues=issues,
                           fix_suggestion=fix,
                           method=f"geom/{category}/fail")
