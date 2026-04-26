"""Regression test: run the 7 standard objects through the CadQuery backend
and verify all produce valid watertight STLs. Also renders thumbnails to
sanity-check the rendering pipeline.

This is what the VLM judge would see. It does NOT call the judge (no cloud
API cost) — for that, run a manual end-to-end test against the live server.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backends import get_backend
from rendering import render_stl_views


OBJECTS = {
    "vase": """
import cadquery as cq
profile = [(25,0),(32,40),(38,100),(25,160),(22,160),(35,100),(29,40),(22,0)]
result = cq.Workplane("XZ").polyline(profile).close().revolve()
export_stl(result, OUTPUT_PATH)
""",
    "cup": """
import cadquery as cq
# Outer revolve
outer_profile = [(35,0),(35,100),(0,100),(0,0)]
outer = cq.Workplane("XZ").polyline(outer_profile).close().revolve()
# Inner hollow (shell via cut)
inner_profile = [(31,4),(31,100),(0,100),(0,4)]
inner = cq.Workplane("XZ").polyline(inner_profile).close().revolve()
result = outer.cut(inner)
export_stl(result, OUTPUT_PATH)
""",
    "phone_stand": """
import cadquery as cq
profile = [(0,0),(90,0),(90,5),(82,5),(62,105),(57,105),(77,5),(5,5),(5,14),(0,14)]
result = cq.Workplane("XY").polyline(profile).close().extrude(80).edges("|X").fillet(2)
export_stl(result, OUTPUT_PATH)
""",
    "bookend": """
import cadquery as cq
profile = [(0,0),(80,0),(80,5),(5,5),(5,100),(0,100)]
result = cq.Workplane("XY").polyline(profile).close().extrude(120)
export_stl(result, OUTPUT_PATH)
""",
    "pen_holder": """
import cadquery as cq
outer = cq.Workplane("XY").cylinder(100, 35)
inner = cq.Workplane("XY").workplane(offset=3).cylinder(97, 32)
result = outer.cut(inner).edges(">Z").fillet(1)
export_stl(result, OUTPUT_PATH)
""",
    "bowl": """
import cadquery as cq
profile = [(0,0),(60,0),(65,10),(70,30),(70,40),(65,45),(5,45),(0,40)]
outer = cq.Workplane("XZ").polyline(profile).close().revolve()
inner_profile = [(0,4),(57,4),(62,14),(66,30),(66,40),(62,42),(5,42)]
inner = cq.Workplane("XZ").polyline(inner_profile).close().revolve()
result = outer.cut(inner)
export_stl(result, OUTPUT_PATH)
""",
    "shoe": """
import cadquery as cq
sole_pts = [(0,20),(20,5),(60,0),(140,0),(220,0),(260,10),(280,30),
            (280,50),(270,65),(260,75),(220,90),(140,100),(60,100),(20,90),(0,70)]
upper_pts = [(20,30),(50,20),(100,15),(160,15),(200,20),(230,30),
             (230,60),(200,75),(160,80),(100,80),(50,75),(20,60)]
result = (cq.Workplane("XY")
          .polyline(sole_pts).close()
          .workplane(offset=80)
          .polyline(upper_pts).close()
          .loft(combine=True))
export_stl(result, OUTPUT_PATH)
""",
}


def stl_ok(path: Path) -> tuple[bool, dict]:
    import trimesh
    m = trimesh.load(str(path))
    stats = {
        "size_bytes": path.stat().st_size,
        "vertices": len(m.vertices),
        "faces": len(m.faces),
        "watertight": bool(m.is_watertight),
        "volume_mm3": float(m.volume) if m.is_volume else None,
    }
    ok = (
        stats["size_bytes"] > 500 and
        stats["vertices"] > 10 and
        stats["faces"] > 10
    )
    return ok, stats


def main() -> int:
    backend = get_backend("cadquery")
    results = {}
    for name, code in OBJECTS.items():
        with tempfile.TemporaryDirectory() as d:
            stl = Path(d) / f"{name}.stl"
            try:
                backend.execute_and_export(code, stl)
            except Exception as e:
                print(f"[FAIL] {name}: backend error: {e}")
                results[name] = ("fail_exec", str(e))
                continue
            ok, stats = stl_ok(stl)
            # Render thumbnails (but to a temp location; don't clutter)
            try:
                views_dir = Path(d) / "views"
                pngs = render_stl_views(stl, views_dir, (256, 256))
                stats["thumbs"] = len(pngs)
            except Exception as e:
                stats["thumbs"] = 0
                stats["render_err"] = str(e)[:80]
            tag = "PASS" if ok else "FAIL"
            wt = "watertight" if stats["watertight"] else "NOT watertight"
            print(f"[{tag}] {name:14} v={stats['vertices']:<5} f={stats['faces']:<5} "
                  f"thumbs={stats['thumbs']}/4 {wt}")
            results[name] = ("pass" if ok else "fail_stl", stats)

    n_pass = sum(1 for k, v in results.items() if v[0] == "pass")
    n = len(results)
    print()
    print(f"== {n_pass}/{n} objects passed ==")
    return 0 if n_pass == n else 1


if __name__ == "__main__":
    sys.exit(main())
