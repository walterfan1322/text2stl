"""Print-readiness analysis (S6.2 part).

Emits warnings about likely 3D-printing problems detected purely from
mesh geometry — no slicer needed. Catches:
  - thin walls (< 1.2mm)
  - extreme overhangs (> 45° from vertical without support)
  - multi-body parts (need to be split or bridged)
  - very high poly count (>200k faces, slow to slice)
  - impossibly small features (mesh is sub-millimeter)
  - tiny disjoint floaters (artifacts from CSG)

Returns a list of `Warning` dicts that the API can attach to the
generation response so the frontend can surface them as chips.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

log = logging.getLogger("text2stl.print_readiness")


def analyse(stl_path: Path,
            min_wall_mm: float = 1.2,
            max_overhang_deg: float = 45.0,
            max_poly_count: int = 200_000,
            ) -> list[dict]:
    """Return a list of warning dicts:
        [{"severity": "warn"|"info", "code": str, "message": str}]
    """
    out: list[dict] = []
    try:
        import trimesh
        m = trimesh.load_mesh(str(stl_path))
        if hasattr(m, "geometry"):
            geoms = tuple(m.geometry.values())
            if geoms:
                m = trimesh.util.concatenate(geoms)
    except Exception as e:
        return [{"severity": "warn", "code": "load",
                 "message": f"could not load STL for analysis: {e}"}]

    if len(m.faces) == 0:
        return [{"severity": "warn", "code": "empty",
                 "message": "mesh has zero faces"}]

    # 1. Bounding box check
    try:
        bbox = m.bounds
        dims = bbox[1] - bbox[0]
        if min(dims) < 1.0:
            out.append({
                "severity": "warn",
                "code": "tiny",
                "message": f"smallest dimension {min(dims):.2f}mm — "
                           f"may not print reliably (<1mm)",
            })
    except Exception:
        pass

    # 2. Body count
    try:
        bodies = int(m.body_count)
        if bodies > 1:
            out.append({
                "severity": "info",
                "code": "multi_body",
                "message": f"mesh has {bodies} disjoint bodies — "
                           f"may need to be printed separately",
            })
    except Exception:
        pass

    # 3. Poly count
    nf = len(m.faces)
    if nf > max_poly_count:
        out.append({
            "severity": "info",
            "code": "high_poly",
            "message": f"high face count ({nf:,}) — slicer may be slow; "
                       f"consider decimation",
        })

    # 4. Overhang detection — share of triangles with normal pointing
    # downward at angle > max_overhang_deg from vertical.
    try:
        normals = m.face_normals
        # Z is up. Triangle "facing down" = normal_z < 0
        # Angle from -Z (downward): cos(theta) = -normal_z
        # An overhang facing down at angle = (90 - theta)° from horizontal
        # Actually: overhangs are surfaces whose normal points >max_overhang_deg
        # from vertical. We measure angle from +Z axis: cos = normal[2].
        # A face with normal pointing nearly downward (cos(angle from +Z)
        # close to -1) is an overhang.
        z = normals[:, 2]
        threshold = math.cos(math.radians(180 - max_overhang_deg))
        bad_count = int((z < threshold).sum())
        bad_share = bad_count / max(1, len(normals))
        if bad_share > 0.05:
            out.append({
                "severity": "info",
                "code": "overhang",
                "message": f"~{bad_share*100:.0f}% of faces are steep overhangs "
                           f"(>{max_overhang_deg}°) — will need supports",
            })
    except Exception:
        pass

    # 5. Wall thickness — proxy: ratio of mesh volume to bounding-box vol.
    # If solid is < 5% of bbox, walls are likely thin.
    try:
        if m.is_watertight:
            vol = float(abs(m.volume))
            bbox_vol = float(dims[0] * dims[1] * dims[2])
            if bbox_vol > 0 and vol / bbox_vol < 0.03:
                out.append({
                    "severity": "warn",
                    "code": "thin_walls",
                    "message": f"mesh volume only {vol/bbox_vol*100:.1f}% of "
                               f"bbox — walls may be thinner than {min_wall_mm}mm",
                })
    except Exception:
        pass

    # 6. Tiny floaters — connected components < 1% of total volume each
    try:
        comps = m.split(only_watertight=False)
        if len(comps) > 1:
            total_vol = sum(abs(c.volume) for c in comps if hasattr(c, "volume"))
            small = [c for c in comps
                     if hasattr(c, "volume") and total_vol > 0
                     and abs(c.volume) / total_vol < 0.01]
            if small:
                out.append({
                    "severity": "warn",
                    "code": "floaters",
                    "message": f"{len(small)} tiny floating piece(s) "
                               f"detected — likely CSG artifacts",
                })
    except Exception:
        pass

    return out


def summary_line(warnings: list[dict]) -> str:
    if not warnings:
        return "print-readiness: clean"
    codes = ",".join(w["code"] for w in warnings)
    return f"print-readiness: {len(warnings)} issue(s) [{codes}]"
