"""End-to-end backend tests: run sample code through each backend and check
that a valid STL is produced.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backends import get_backend


def run(label: str, ok: bool):
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    return ok


def stl_ok(path: Path) -> bool:
    """Basic STL sanity: file exists, non-trivial size, parseable."""
    if not path.exists() or path.stat().st_size < 200:
        return False
    try:
        import trimesh
        m = trimesh.load(str(path))
        return len(m.vertices) > 4 and len(m.faces) > 4
    except Exception as e:
        print(f"   load failure: {e}")
        return False


all_pass = True

# =============================================================================
# Trimesh backend tests
# =============================================================================
tb = get_backend("trimesh")
print(f"\n=== Trimesh backend ===")

with tempfile.TemporaryDirectory() as d:
    out = Path(d) / "box.stl"
    tb.execute_and_export(
        """
import trimesh
m = trimesh.creation.box(extents=[10, 20, 30])
m.apply_translation(-m.centroid)
m.export(OUTPUT_PATH)
""",
        out,
    )
    all_pass &= run("trimesh box", stl_ok(out))

with tempfile.TemporaryDirectory() as d:
    out = Path(d) / "vase.stl"
    tb.execute_and_export(
        """
profile = [(25,0), (30,40), (35,90), (22,140), (18,160),
           (15,160), (19,140), (31,90), (26,40), (21,0)]
vase = make_solid_revolution(profile)
vase.export(OUTPUT_PATH)
""",
        out,
    )
    all_pass &= run("trimesh make_solid_revolution", stl_ok(out))


# =============================================================================
# CadQuery backend tests
# =============================================================================
cq = get_backend("cadquery")
print(f"\n=== CadQuery backend ===")

with tempfile.TemporaryDirectory() as d:
    out = Path(d) / "box.stl"
    cq.execute_and_export(
        """
import cadquery as cq
result = cq.Workplane("XY").box(10, 20, 30)
export_stl(result, OUTPUT_PATH)
""",
        out,
    )
    all_pass &= run("cadquery box", stl_ok(out))

with tempfile.TemporaryDirectory() as d:
    out = Path(d) / "vase.stl"
    cq.execute_and_export(
        """
import cadquery as cq
profile = [(25,0), (32,40), (38,100), (25,160), (22,160), (35,100), (29,40), (22,0)]
result = cq.Workplane("XZ").polyline(profile).close().revolve()
export_stl(result, OUTPUT_PATH)
""",
        out,
    )
    all_pass &= run("cadquery revolve (vase)", stl_ok(out))

with tempfile.TemporaryDirectory() as d:
    out = Path(d) / "pen.stl"
    cq.execute_and_export(
        """
import cadquery as cq
outer = cq.Workplane("XY").cylinder(100, 35)
inner = cq.Workplane("XY").workplane(offset=3).cylinder(97, 32)
result = outer.cut(inner)
export_stl(result, OUTPUT_PATH)
""",
        out,
    )
    all_pass &= run("cadquery pen-holder (cylinder + cut)", stl_ok(out))

with tempfile.TemporaryDirectory() as d:
    out = Path(d) / "stand.stl"
    cq.execute_and_export(
        """
import cadquery as cq
profile = [(0,0),(90,0),(90,5),(82,5),(62,105),(57,105),(77,5),(5,5),(5,14),(0,14)]
result = cq.Workplane("XY").polyline(profile).close().extrude(80).edges("|X").fillet(2)
export_stl(result, OUTPUT_PATH)
""",
        out,
    )
    all_pass &= run("cadquery phone-stand (extrude + fillet)", stl_ok(out))

with tempfile.TemporaryDirectory() as d:
    out = Path(d) / "shoe.stl"
    cq.execute_and_export(
        """
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
        out,
    )
    all_pass &= run("cadquery shoe (loft)", stl_ok(out))

# Test auto-export fallback (no export_stl call, just `result`)
with tempfile.TemporaryDirectory() as d:
    out = Path(d) / "auto.stl"
    cq.execute_and_export(
        """
import cadquery as cq
result = cq.Workplane("XY").box(10, 10, 10)
""",
        out,
    )
    all_pass &= run("cadquery auto-export via `result`", stl_ok(out))

print()
print("=" * 50)
print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
print("=" * 50)
sys.exit(0 if all_pass else 1)
