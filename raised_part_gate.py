"""raised_part_gate — pre-judge gate that catches sub-parts buried inside
the root via AST + bounding-box analysis.

Motivating bug
--------------
LLM writes:
    base = cq.Workplane("XY").box(400, 400, 8, centered=(True, True, False))
    sq   = cq.Workplane("XY").center(x, y).box(50, 50, 2, centered=(True, True, False))
    result = base.union(sq)

`sq`'s Z range is [0, 2] — fully inside `base`'s [0, 8]. The union has no
visible effect, the render is a featureless slab, and the VLM judge often
hallucinates the missing pattern. The fix is `.workplane(offset=8)` after
`.center(x, y)` — but pitfall #12 in the system prompt isn't strong
enough on its own.

This gate catches the failure deterministically. Pure AST + bbox math —
no category logic, no per-shape recipes.

Algorithm
---------
1. Parse the user's code with `ast`.
2. For each top-level assignment whose RHS is a CadQuery chain ending in
   a primitive (`box` / `cylinder` / `sphere`), simulate the chain
   forward and compute its world-frame AABB.
3. Walk the rest of the AST to discover the "role" of each named
   primitive — cut targets are not flagged (a recess INSIDE the base is
   the whole point), union targets are flagged.
4. Pick the largest (by Z extent) primitive as the ROOT; the rest are
   children.
5. A child is BURIED if its AABB is fully contained inside the root's
   AABB AND its role is `union` (not `cut`).
6. If any union-child is buried, gate fails with an actionable message.

Limits
------
- Only XY-plane primitives are analysed (XZ / YZ are conservatively
  skipped — those are usually revolve profiles, not stacked parts).
- Loops (`for ... in range`) ARE handled: every iteration's primitive is
  computed using the loop range and the parametric center expression
  (when it reduces to `start + step * loop_var`). When parameters can't
  be statically simplified, the gate gives up on that primitive (return
  None bbox) — i.e. silent skip rather than false-positive.
- Only `int` / `float` literal arithmetic is followed; symbolic
  expressions return None and skip.
- This is intentionally conservative: false-positive rate must be near
  zero, since the gate forces a retry. False-negatives (missed buried
  parts) are acceptable — the VLM judge picks them up.
"""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("raised_part_gate")


@dataclass
class BBox:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    @property
    def zspan(self) -> float:
        return self.zmax - self.zmin

    @property
    def vol(self) -> float:
        return ((self.xmax - self.xmin)
                * (self.ymax - self.ymin)
                * (self.zmax - self.zmin))

    def contains(self, other: "BBox", tol: float = 0.5) -> bool:
        """True iff `other` is fully inside `self` (with tolerance for
        the small overlap LLMs intentionally add to ensure clean union)."""
        return (other.xmin >= self.xmin - tol
                and other.xmax <= self.xmax + tol
                and other.ymin >= self.ymin - tol
                and other.ymax <= self.ymax + tol
                and other.zmin >= self.zmin - tol
                and other.zmax <= self.zmax + tol)

    def xy_overlaps(self, other: "BBox", tol: float = 0.0) -> bool:
        """True iff the XY footprints of `self` and `other` have a
        non-empty intersection (allowing `tol` slack)."""
        return not (other.xmax < self.xmin - tol
                    or other.xmin > self.xmax + tol
                    or other.ymax < self.ymin - tol
                    or other.ymin > self.ymax + tol)


@dataclass
class Primitive:
    var_name: str       # the LHS name in the assignment (or first assignment)
    bbox: BBox
    role: str = "?"     # 'union' / 'cut' / 'root' / 'unknown'
    line: int = 0


@dataclass
class RaisedPartResult:
    passed: bool
    fail_reason: str = ""
    issues: list[str] = field(default_factory=list)
    fix_suggestion: str = ""
    method: str = "raised_part_gate.v2"
    score: int | None = None
    # Diagnostics for debugging / telemetry
    primitives: list[Primitive] = field(default_factory=list)
    root_var: str | None = None
    buried_vars: list[str] = field(default_factory=list)
    # v2: touching-face issues (separate from buried). Same retry path
    # but a different remediation (overlap by 0.1mm vs raise by H).
    touching_vars: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------
def _eval_const(node: ast.AST, env: dict[str, float]) -> float | None:
    """Constant-fold a numeric expression. Returns None if it depends on
    any variable not in `env`. Supports +, -, *, /, unary -, int/float
    literals, and `Name` lookups in `env`."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _eval_const(node.operand, env)
        return None if v is None else -v
    if isinstance(node, ast.BinOp):
        l = _eval_const(node.left, env)
        r = _eval_const(node.right, env)
        if l is None or r is None:
            return None
        if isinstance(node.op, ast.Add):  return l + r
        if isinstance(node.op, ast.Sub):  return l - r
        if isinstance(node.op, ast.Mult): return l * r
        if isinstance(node.op, ast.Div):
            return None if r == 0 else l / r
        if isinstance(node.op, ast.FloorDiv):
            return None if r == 0 else float(int(l // r))
        return None
    if isinstance(node, ast.Name):
        return env.get(node.id)
    return None


def _arg_or_kw(call: ast.Call, idx: int, kw: str,
               env: dict[str, float]) -> float | None:
    """Pull the `idx`-th positional or keyword `kw` numeric arg of `call`."""
    if idx < len(call.args):
        return _eval_const(call.args[idx], env)
    for k in call.keywords:
        if k.arg == kw:
            return _eval_const(k.value, env)
    return None


def _kw_centered(call: ast.Call) -> tuple[bool, bool, bool]:
    """Pull `centered=(cx, cy, cz)` from a box call. Default is
    (True, True, True)."""
    for k in call.keywords:
        if k.arg == "centered" and isinstance(k.value, ast.Tuple):
            elts = k.value.elts
            if len(elts) == 3 and all(isinstance(e, ast.Constant) for e in elts):
                return tuple(bool(e.value) for e in elts)  # type: ignore
    return (True, True, True)


def _flatten_chain(node: ast.AST) -> list[ast.Call]:
    """Walk a CadQuery method chain and return the calls in order from
    outermost to innermost (so the FIRST element is the root call like
    `cq.Workplane("XY")` and the LAST is the final method)."""
    calls: list[ast.Call] = []
    cur = node
    while isinstance(cur, ast.Call):
        calls.append(cur)
        # Recurse into the .func — which is an Attribute on the previous
        # Call's value, OR a Name like `cq.Workplane`.
        if isinstance(cur.func, ast.Attribute):
            cur = cur.func.value
        else:
            break
    return list(reversed(calls))


def _method_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _is_workplane_root(call: ast.Call) -> str | None:
    """Return the plane string ('XY' / 'XZ' / 'YZ') if this is
    `cq.Workplane("...")`, else None."""
    func = call.func
    if isinstance(func, ast.Attribute) and func.attr == "Workplane":
        if isinstance(func.value, ast.Name) and func.value.id == "cq":
            if call.args and isinstance(call.args[0], ast.Constant):
                return str(call.args[0].value)
    return None


# ---------------------------------------------------------------------------
# Chain → BBox
# ---------------------------------------------------------------------------
def chain_to_bbox(chain_root: ast.AST,
                  env: dict[str, float]) -> Optional[BBox]:
    """Simulate a CadQuery chain and return its world-AABB.

    Returns None if:
    - The chain doesn't end in a recognised primitive (box/cylinder/sphere)
    - The plane is not XY (we don't analyse XZ/YZ — too rare for this bug)
    - Any numeric arg can't be constant-folded
    """
    calls = _flatten_chain(chain_root)
    if not calls:
        return None

    plane = _is_workplane_root(calls[0])
    if plane != "XY":
        return None  # only handle XY for now; XZ/YZ usually = revolve profiles

    cx, cy, cz = 0.0, 0.0, 0.0   # workplane origin (current frame)
    final = calls[-1]
    name = _method_name(final)

    # Walk the intermediate methods (between Workplane root and primitive)
    for c in calls[1:-1]:
        m = _method_name(c)
        if m == "center":
            x = _arg_or_kw(c, 0, "x", env)
            y = _arg_or_kw(c, 1, "y", env)
            if x is None or y is None:
                return None
            cx += x
            cy += y
        elif m == "workplane":
            off = _arg_or_kw(c, 0, "offset", env)
            if off is None:
                # workplane() with no numeric offset (e.g. .faces(...)
                # workplane()) — too complex; bail
                return None
            cz += off
        elif m in ("translate", "moveTo"):
            # Skip — moveTo is for sketches; translate is post-shape and
            # rarely seen on raised parts. Bail out.
            return None

    # Now interpret the final primitive
    if name == "box":
        W = _arg_or_kw(final, 0, "length", env)
        D = _arg_or_kw(final, 1, "width",  env)
        H = _arg_or_kw(final, 2, "height", env)
        if W is None or D is None or H is None:
            return None
        ccx, ccy, ccz = _kw_centered(final)
        x0 = cx - W / 2 if ccx else cx
        x1 = x0 + W
        y0 = cy - D / 2 if ccy else cy
        y1 = y0 + D
        z0 = cz - H / 2 if ccz else cz
        z1 = z0 + H
        return BBox(x0, x1, y0, y1, z0, z1)

    if name == "cylinder":
        H = _arg_or_kw(final, 0, "height", env)
        R = _arg_or_kw(final, 1, "radius", env)
        if H is None or R is None:
            return None
        # Cylinder default: centered=(True, True, True) like box
        ccx, ccy, ccz = _kw_centered(final)
        x0 = cx - R if ccx else cx - R   # cylinder always XY-centered
        x1 = cx + R
        y0 = cy - R
        y1 = cy + R
        z0 = cz - H / 2 if ccz else cz
        z1 = z0 + H
        return BBox(x0, x1, y0, y1, z0, z1)

    if name == "sphere":
        R = _arg_or_kw(final, 0, "radius", env)
        if R is None:
            return None
        return BBox(cx - R, cx + R, cy - R, cy + R, cz - R, cz + R)

    # Other terminal methods (revolve, extrude, loft, sweep, ...) —
    # we don't analyse them. Return None so they don't enter the
    # buried-vs-root comparison.
    return None


# ---------------------------------------------------------------------------
# Top-level: extract primitives + roles
# ---------------------------------------------------------------------------
def _walk_assignments(tree: ast.Module
                      ) -> list[tuple[str, ast.AST, int, dict[str, float]]]:
    """Yield (var_name, rhs_ast, lineno, env) for every Assign reachable
    by unrolling top-level for-loops over `range(...)`.

    `env` accumulates every PRIOR numeric-constant assignment in scope
    (including loop variables) so that `chain_to_bbox` can resolve
    `Name` references inside the chain's args (e.g. `.center(x, y)`
    where `x`, `y` were assigned earlier in the same loop body).
    """
    out: list[tuple[str, ast.AST, int, dict[str, float]]] = []

    def walk_body(stmts: list[ast.stmt], env: dict[str, float],
                  iter_tag: str) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                v = _eval_const(stmt.value, env)
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Name):
                        if v is not None:
                            env[tgt.id] = v
                        # Tag with the current iteration suffix so
                        # loop-unrolled assignments to the same name
                        # don't collide.
                        var = f"{tgt.id}{iter_tag}" if iter_tag else tgt.id
                        out.append((var, stmt.value, stmt.lineno, dict(env)))
            elif (isinstance(stmt, ast.Expr)
                  and isinstance(stmt.value, ast.Call)
                  and isinstance(stmt.value.func, ast.Attribute)
                  and stmt.value.func.attr == "append"
                  and isinstance(stmt.value.func.value, ast.Name)
                  and stmt.value.args
                  and isinstance(stmt.value.args[0], ast.Call)):
                # Pattern: `<list_name>.append(cq.Workplane(...).box(...))`
                # — inline chain in an append call. Treat the chain as
                # a synthetic primitive named after the list.
                list_name = stmt.value.func.value.id
                chain = stmt.value.args[0]
                synthetic = f"{list_name}{iter_tag}" if iter_tag else list_name
                out.append((synthetic, chain, stmt.lineno, dict(env)))
            elif isinstance(stmt, ast.For):
                if not isinstance(stmt.target, ast.Name):
                    continue
                loopvar = stmt.target.id
                if not (isinstance(stmt.iter, ast.Call)
                        and isinstance(stmt.iter.func, ast.Name)
                        and stmt.iter.func.id == "range"):
                    continue
                consts = []
                ok = True
                for a in stmt.iter.args:
                    cv = _eval_const(a, env)
                    if cv is None:
                        ok = False
                        break
                    consts.append(int(cv))
                if not ok:
                    continue
                if len(consts) == 1:
                    rng = range(0, consts[0])
                elif len(consts) == 2:
                    rng = range(consts[0], consts[1])
                elif len(consts) == 3:
                    rng = range(consts[0], consts[1], consts[2])
                else:
                    continue
                values = list(rng)
                if len(values) > 200:
                    values = values[:200]
                for i in values:
                    inner_env = dict(env)
                    inner_env[loopvar] = float(i)
                    inner_tag = f"{iter_tag}#{i}"
                    walk_body(stmt.body, inner_env, inner_tag)
            elif isinstance(stmt, ast.If):
                # Walk BOTH branches. We don't try to evaluate the
                # condition — every primitive that could be added at
                # runtime is collected. False-positive risk: a branch
                # that would never execute still contributes bboxes.
                # In practice LLMs use ifs for checkerboard-style
                # filters, so both branches are reachable across the
                # outer loop, and the union of branches matches the
                # actual generated primitives 1:1.
                walk_body(stmt.body, env, iter_tag)
                walk_body(stmt.orelse, env, iter_tag)

    walk_body(tree.body, {}, "")
    return out


def extract_primitives(code: str) -> list[Primitive]:
    """Top-level entry: walk the AST and return a list of named
    primitives with computed bboxes. Skips any chain that doesn't
    resolve to an XY-plane primitive."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    out: list[Primitive] = []
    for var, rhs, line, env in _walk_assignments(tree):
        bbox = chain_to_bbox(rhs, env=env)
        if bbox is None:
            continue
        out.append(Primitive(var_name=var, bbox=bbox, line=line))
    return out


def _classify_roles(code: str, prims: list[Primitive]) -> None:
    """Walk the AST a second time and tag each primitive's role by
    looking at how its name is used in `.union(...)` / `.cut(...)`
    calls.

    Key behaviour: a primitive is `cut` ONLY if its name appears as
    the argument to `.cut(...)`. Default is `union` (most LLM code
    union-aggregates parts). The largest primitive becomes `root`."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return

    # Build a name -> primitive map (for loop-unrolled vars, strip the
    # "#i" suffix to get the base name)
    name_to_role: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            method = node.func.attr
            if method == "cut":
                for arg in node.args:
                    if isinstance(arg, ast.Name):
                        name_to_role[arg.id] = "cut"
            elif method == "union":
                for arg in node.args:
                    if isinstance(arg, ast.Name):
                        name_to_role.setdefault(arg.id, "union")

    # Apply
    for p in prims:
        base_name = p.var_name.split("#", 1)[0]
        p.role = name_to_role.get(base_name, "unknown")

    # Pick root: the biggest by VOLUME among primitives whose role is
    # `unknown` or `union` (a `cut` primitive is by definition not the
    # root). Tag it.
    candidates = [p for p in prims if p.role != "cut"]
    if not candidates:
        return
    root = max(candidates, key=lambda p: p.bbox.vol)
    root.role = "root"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def check(code: str) -> RaisedPartResult:
    """Run the full gate. Returns a RaisedPartResult.

    Two failure modes:
    1. BURIED — child bbox fully contained in root's bbox. Union has
       no visible effect. Role must be `union` or `unknown` (cuts are
       valid recesses).
    2. TOUCHING-FACE — child Z range starts EXACTLY at root's top
       (within 0.05mm) AND XY footprints overlap. OCC's boolean
       engine often silently drops shapes that share a face exactly,
       leaving the STL with the root only.

    Buried takes precedence over touching: if a child is buried it
    isn't also flagged as touching.
    """
    prims = extract_primitives(code)
    if len(prims) < 2:
        # Single-shape (vase, cup) — no buried-part bug possible
        return RaisedPartResult(passed=True, primitives=prims)

    _classify_roles(code, prims)
    root = next((p for p in prims if p.role == "root"), None)
    if root is None:
        return RaisedPartResult(passed=True, primitives=prims)

    buried: list[Primitive] = []
    touching: list[Primitive] = []
    TOUCH_TOL = 0.05  # |child.zmin - root.zmax| < this counts as touching
    for p in prims:
        if p is root or p.role == "cut":
            continue
        if root.bbox.contains(p.bbox):
            buried.append(p)
            continue  # buried takes precedence
        # touching-face check: child sits exactly ON TOP of root
        # (zmin == root.zmax) with XY overlap. This is the OCC
        # silent-drop trigger.
        if (abs(p.bbox.zmin - root.bbox.zmax) < TOUCH_TOL
                and root.bbox.xy_overlaps(p.bbox)):
            touching.append(p)

    if not buried and not touching:
        return RaisedPartResult(passed=True, primitives=prims, root_var=root.var_name)

    # Build issues list — buried first, then touching
    issues: list[str] = []
    seen_buried: set[str] = set()
    for p in buried:
        base = p.var_name.split("#", 1)[0]
        if base in seen_buried:
            continue
        seen_buried.add(base)
        issues.append(
            f"`{base}` (X[{p.bbox.xmin:.1f},{p.bbox.xmax:.1f}] "
            f"Y[{p.bbox.ymin:.1f},{p.bbox.ymax:.1f}] "
            f"Z[{p.bbox.zmin:.1f},{p.bbox.zmax:.1f}]) is fully inside root "
            f"`{root.var_name}` "
            f"(Z[{root.bbox.zmin:.1f},{root.bbox.zmax:.1f}]) — union has no "
            f"visible effect."
        )

    seen_touching: set[str] = set()
    for p in touching:
        base = p.var_name.split("#", 1)[0]
        if base in seen_touching or base in seen_buried:
            continue
        seen_touching.add(base)
        issues.append(
            f"`{base}` Z={p.bbox.zmin:.2f} touches root `{root.var_name}` "
            f"top Z={root.bbox.zmax:.2f} EXACTLY — OCC's boolean engine "
            f"may silently drop the part on shared faces."
        )

    # Fix suggestion depends on which kind of issue dominates
    if buried:
        fix = (
            f"Each buried part must either (a) sit on top of the root: "
            f"chain `.workplane(offset={root.bbox.zmax - 0.1:.1f})` after "
            f"`.center(x, y)` and add ~0.1mm to its height so it overlaps "
            f"the root's top by 0.1mm (avoids OCC silent-drop on touching "
            f"faces); OR (b) be a recess: change `.union(...)` to "
            f"`.cut(...)` so it removes material from the root."
        )
    else:
        fix = (
            f"Lower each touching part's offset by 0.1mm so it overlaps "
            f"the root rather than just touches it. Replace "
            f"`.workplane(offset={root.bbox.zmax:.0f})` with "
            f"`.workplane(offset={root.bbox.zmax - 0.1:.1f})` and add 0.1 "
            f"to the part's height (e.g. `box(50, 50, 2)` -> "
            f"`box(50, 50, 2.1)`). The 0.1mm overlap is invisible in "
            f"print but makes OCC's boolean robust."
        )

    n_buried = len(seen_buried)
    n_touching = len(seen_touching)
    parts: list[str] = []
    if n_buried:
        parts.append(f"{n_buried} buried")
    if n_touching:
        parts.append(f"{n_touching} touching-face")
    fail_reason = " + ".join(parts) + " issue(s)"

    return RaisedPartResult(
        passed=False,
        fail_reason=fail_reason,
        issues=issues,
        fix_suggestion=fix,
        primitives=prims,
        root_var=root.var_name,
        buried_vars=sorted(seen_buried),
        touching_vars=sorted(seen_touching),
    )


def build_retry_hint(result: RaisedPartResult) -> str:
    """Render a result into an LLM-facing retry message."""
    lines = [
        f"Geometric check failed (raised_part_gate): {result.fail_reason}.",
        "",
        "Issues:",
    ]
    for it in result.issues:
        lines.append(f"  - {it}")
    lines += [
        "",
        f"Fix: {result.fix_suggestion}",
        "",
        "Output ONLY corrected Python code.",
    ]
    return "\n".join(lines)
