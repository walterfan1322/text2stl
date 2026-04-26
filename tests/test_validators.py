"""Unit tests for validators.py — AST allowlist validator."""
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from validators import (
    validate_trimesh,
    validate_cadquery,
    validate_code,
    format_errors_for_llm,
)


def run(label: str, cond: bool):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}")
    return cond


all_pass = True


# -------------------------------------------------------------------
# Trimesh backend tests
# -------------------------------------------------------------------
valid_box = """
import trimesh
mesh = trimesh.creation.box(extents=[10, 20, 30])
mesh.apply_translation(-mesh.centroid)
mesh.export(OUTPUT_PATH)
"""
r = validate_trimesh(valid_box)
all_pass &= run("valid box passes", r.ok)

valid_composite = """
import trimesh
import numpy as np
seat = trimesh.creation.box(extents=[450, 400, 25])
seat.apply_translation([0, 0, 437.5])
leg = trimesh.creation.cylinder(radius=25, height=425)
chair = trimesh.util.concatenate([seat, leg])
chair.export(OUTPUT_PATH)
"""
r = validate_trimesh(valid_composite)
all_pass &= run("valid composite passes", r.ok)

valid_revolution = """
import trimesh
vase = make_solid_revolution([(25,0), (30,40), (35,90)])
vase.export(OUTPUT_PATH)
"""
r = validate_trimesh(valid_revolution)
all_pass &= run("valid make_solid_revolution passes", r.ok)

# Hallucinated APIs
bad_frustum = """
import trimesh
m = trimesh.creation.frustum(r1=10, r2=20, h=100)
m.export(OUTPUT_PATH)
"""
r = validate_trimesh(bad_frustum)
all_pass &= run("rejects trimesh.creation.frustum", not r.ok and any("frustum" in e for e in r.errors))

bad_sphere = """
import trimesh
m = trimesh.creation.sphere(radius=10)
m.export(OUTPUT_PATH)
"""
r = validate_trimesh(bad_sphere)
all_pass &= run("rejects trimesh.creation.sphere", not r.ok and any("sphere" in e for e in r.errors))

bad_revolve = """
import trimesh
m = trimesh.creation.revolve(profile)
m.export(OUTPUT_PATH)
"""
r = validate_trimesh(bad_revolve)
all_pass &= run("rejects trimesh.creation.revolve", not r.ok and any("revolve" in e for e in r.errors))

# Security: forbidden builtins
malicious_eval = """
import trimesh
eval("print('pwn')")
m = trimesh.creation.box(extents=[1,1,1])
m.export(OUTPUT_PATH)
"""
r = validate_trimesh(malicious_eval)
all_pass &= run("rejects eval()", not r.ok and any("eval" in e for e in r.errors))

malicious_import = """
import os
os.system("rm -rf /")
"""
r = validate_trimesh(malicious_import)
all_pass &= run("rejects import os", not r.ok and any("os" in e for e in r.errors))

malicious_dunder = """
__import__('os').system('pwn')
"""
r = validate_trimesh(malicious_dunder)
all_pass &= run("rejects __import__", not r.ok)

malicious_open = """
import trimesh
open('/etc/passwd').read()
m = trimesh.creation.box(extents=[1,1,1])
m.export(OUTPUT_PATH)
"""
r = validate_trimesh(malicious_open)
all_pass &= run("rejects open()", not r.ok and any("open" in e for e in r.errors))

# Syntax error
bad_syntax = """
import trimesh
mesh = trimesh.creation.box(
"""
r = validate_trimesh(bad_syntax)
all_pass &= run("catches syntax error", not r.ok and any("SyntaxError" in e for e in r.errors))


# -------------------------------------------------------------------
# CadQuery backend tests
# -------------------------------------------------------------------
valid_cq_box = """
import cadquery as cq
result = cq.Workplane("XY").box(10, 20, 30)
export_stl(result, OUTPUT_PATH)
"""
r = validate_cadquery(valid_cq_box)
all_pass &= run("valid cadquery box passes", r.ok)

valid_cq_complex = """
import cadquery as cq
base = cq.Workplane("XY").box(80, 60, 10)
back = base.faces(">Y").workplane().box(80, 60, 10, combine=True)
result = base.union(back).edges("|Y").fillet(3)
export_stl(result, OUTPUT_PATH)
"""
r = validate_cadquery(valid_cq_complex)
all_pass &= run("valid cadquery composite passes", r.ok)

valid_cq_revolve = """
import cadquery as cq
profile = [(0,0),(30,0),(35,20),(25,50),(0,80)]
result = cq.Workplane("XZ").polyline(profile).close().revolve()
export_stl(result, OUTPUT_PATH)
"""
r = validate_cadquery(valid_cq_revolve)
all_pass &= run("valid cadquery revolve passes", r.ok)

# CadQuery hallucination: non-existent method
bad_cq = """
import cadquery as cq
result = cq.Workplane("XY").magicShape(10)
export_stl(result, OUTPUT_PATH)
"""
r = validate_cadquery(bad_cq)
all_pass &= run("rejects unknown cadquery method", not r.ok)


# -------------------------------------------------------------------
# format_errors_for_llm
# -------------------------------------------------------------------
msg = format_errors_for_llm(["Forbidden name: open", "Unknown API call: x.y.z()"])
all_pass &= run("format_errors produces retry message", "regenerate" in msg.lower())


print()
print("=" * 50)
print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
print("=" * 50)
sys.exit(0 if all_pass else 1)
