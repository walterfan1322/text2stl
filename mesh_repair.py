"""Mesh repair — try to make non-watertight STLs printable.

Pipeline:
  1. Load STL with trimesh (always present).
  2. If already watertight → no-op.
  3. Pass 1: trimesh built-in (fix_normals / fill_holes / fix_winding /
     fix_inversion). Cheap, safe.
  4. Pass 2 (optional): pymeshfix if installed — aggressive hole-closing
     that often fixes meshes trimesh can't.
  5. Write repaired mesh back ONLY if it became watertight (to avoid
     silently replacing a 'mostly right' STL with a reconstructed hull
     that might look different).

The hook returns a `RepairResult` so the caller can log before/after
stats and feed 'still not watertight' into the retry loop (see S4.2).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("text2stl.mesh_repair")


@dataclass
class RepairResult:
    loaded: bool                  # True if STL loaded at all
    before_watertight: bool
    after_watertight: bool
    before_volume: float
    after_volume: float
    wrote_back: bool              # True if we overwrote the STL
    method: str                   # "none" | "trimesh" | "trimesh+pymeshfix"
    notes: str = ""

    def summary(self) -> str:
        return (
            f"wt {self.before_watertight}→{self.after_watertight} "
            f"vol {self.before_volume:.1f}→{self.after_volume:.1f} "
            f"method={self.method} wrote_back={self.wrote_back}"
        )


def repair_stl(path: Path) -> RepairResult:
    path = Path(path)
    try:
        import trimesh
    except ImportError:
        return RepairResult(False, False, False, 0.0, 0.0, False, "none",
                            "trimesh not installed")
    try:
        m = trimesh.load_mesh(str(path))
    except Exception as e:
        return RepairResult(False, False, False, 0.0, 0.0, False, "none",
                            f"load failed: {e}")

    # Handle Scene (multi-geometry) by concatenating — the judge / STL
    # consumer expects a single mesh.
    if not hasattr(m, "is_watertight"):
        try:
            geoms = tuple(m.geometry.values()) if hasattr(m, "geometry") else ()
            if not geoms:
                return RepairResult(True, False, False, 0.0, 0.0, False,
                                    "none", "scene with no mesh")
            m = trimesh.util.concatenate(geoms)
        except Exception as e:
            return RepairResult(True, False, False, 0.0, 0.0, False, "none",
                                f"scene flatten failed: {e}")

    before_wt = bool(m.is_watertight)
    before_vol = float(abs(m.volume)) if before_wt else 0.0

    if before_wt:
        return RepairResult(True, True, True, before_vol, before_vol,
                            False, "none", "already watertight")

    methods: list[str] = []

    # ---- Pass 1: trimesh built-in ----
    try:
        trimesh.repair.fix_normals(m)
        trimesh.repair.fix_winding(m)
        trimesh.repair.fix_inversion(m)
        trimesh.repair.fill_holes(m)
        methods.append("trimesh")
    except Exception as e:
        log.debug(f"trimesh repair raised: {e}")

    after_wt = bool(m.is_watertight)

    # ---- Pass 2: pymeshfix (optional, more aggressive) ----
    if not after_wt:
        try:
            import pymeshfix
            mf = pymeshfix.MeshFix(m.vertices, m.faces)
            # pymeshfix ≥0.18 dropped `verbose`; older versions accept it.
            # Try both so we work across installed versions.
            try:
                mf.repair(verbose=False)
            except TypeError:
                mf.repair()
            # API also renamed v/f → points/faces in ≥0.18.
            mf_v = getattr(mf, "points", None)
            mf_f = getattr(mf, "faces", None)
            if mf_v is None or mf_f is None:
                mf_v = getattr(mf, "v", None)
                mf_f = getattr(mf, "f", None)
            m2 = trimesh.Trimesh(vertices=mf_v, faces=mf_f, process=False)
            # Accept only if it actually improved something.
            if bool(m2.is_watertight):
                m = m2
                methods.append("pymeshfix")
                after_wt = True
        except ImportError:
            pass
        except Exception as e:
            log.debug(f"pymeshfix repair raised: {e}")

    # ---- Pass 3 (S6.2): aggressive cleanup if still not watertight ----
    # PyMeshLab catches some cases pymeshfix misses (e.g. duplicated faces
    # with reversed winding). Skip if pymeshfix already succeeded.
    if not after_wt:
        try:
            import pymeshlab
            ms = pymeshlab.MeshSet()
            current = pymeshlab.Mesh(
                vertex_matrix=m.vertices.astype("float64"),
                face_matrix=m.faces.astype("int32"),
            )
            ms.add_mesh(current)
            try:
                ms.meshing_remove_duplicate_vertices()
            except Exception:
                pass
            try:
                ms.meshing_remove_duplicate_faces()
            except Exception:
                pass
            try:
                ms.meshing_remove_unreferenced_vertices()
            except Exception:
                pass
            try:
                ms.meshing_repair_non_manifold_edges()
            except Exception:
                pass
            try:
                ms.meshing_close_holes(maxholesize=300, refinehole=False)
            except Exception:
                pass
            cur = ms.current_mesh()
            v3 = cur.vertex_matrix()
            f3 = cur.face_matrix()
            if v3 is not None and f3 is not None and len(v3) and len(f3):
                m3 = trimesh.Trimesh(vertices=v3, faces=f3, process=False)
                if bool(m3.is_watertight):
                    m = m3
                    methods.append("pymeshlab")
                    after_wt = True
        except ImportError:
            pass
        except Exception as e:
            log.debug(f"pymeshlab repair raised: {e}")

    after_vol = float(abs(m.volume)) if after_wt else 0.0

    # Sanity: if pymeshfix produced a volume wildly different from before
    # (>3x or <0.3x), it may have reconstructed the convex hull instead of
    # the original shape. Keep the repaired one — non-watertight is worse
    # than 'slightly different' — but note it.
    notes = ""
    if after_wt and before_vol > 0 and not (0.3 <= after_vol / max(before_vol, 1e-9) <= 3.0):
        notes = f"volume changed {before_vol:.1f}→{after_vol:.1f}"

    wrote_back = False
    if after_wt and methods:
        try:
            m.export(str(path))
            wrote_back = True
        except Exception as e:
            notes = f"export failed: {e}"

    return RepairResult(
        loaded=True,
        before_watertight=before_wt,
        after_watertight=after_wt,
        before_volume=before_vol,
        after_volume=after_vol,
        wrote_back=wrote_back,
        method="+".join(methods) if methods else "none",
        notes=notes,
    )
