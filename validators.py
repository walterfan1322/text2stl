"""AST-based allowlist validator for LLM-generated 3D modeling code.

Parses generated Python code and rejects calls to unknown APIs before exec,
preventing LLM hallucinations (e.g., trimesh.creation.torus that doesn't exist)
and blocking dangerous builtins (exec, eval, __import__, open, subprocess...).

Used as a pre-exec gate: if validation fails, the error messages are fed back
to the LLM so it can regenerate valid code.
"""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Iterable

log = logging.getLogger("text2stl.validators")


# =============================================================================
# Allowlist definitions
# =============================================================================

# Modules that can be imported or referenced
ALLOWED_IMPORTS: set[str] = {
    "trimesh",
    "trimesh.creation",
    "trimesh.transformations",
    "trimesh.util",
    "trimesh.boolean",
    "numpy",
    "math",
    "shapely",
    "shapely.geometry",
    "shapely.affinity",
    "shapely.ops",
    "cadquery",
    "cadquery.exporters",
    "cadquery.selectors",
}

# Function-call targets (dotted names). Resolved from attribute chains.
# NOTE: each backend contributes its own via backend_allowed_calls() below.
# The set here is for common stdlib / numpy / shapely calls.
ALLOWED_CALLS_COMMON: set[str] = {
    # numpy
    "np.array",
    "np.linspace",
    "np.arange",
    "np.zeros",
    "np.ones",
    "np.full",
    "np.column_stack",
    "np.vstack",
    "np.hstack",
    "np.dot",
    "np.cos",
    "np.sin",
    "np.tan",
    "np.deg2rad",
    "np.pi",  # technically an attribute, but harmless
    "numpy.array",
    "numpy.linspace",
    "numpy.arange",
    "numpy.zeros",
    "numpy.ones",
    "numpy.pi",
    "numpy.cos",
    "numpy.sin",
    # math
    "math.pi",
    "math.cos",
    "math.sin",
    "math.sqrt",
    "math.radians",
    "math.degrees",
    # shapely
    "Polygon",
    "shapely.geometry.Polygon",
    "shapely.affinity.rotate",
    "shapely.affinity.translate",
    # builtins commonly used
    "len",
    "range",
    "zip",
    "enumerate",
    "abs",
    "min",
    "max",
    "round",
    "int",
    "float",
    "list",
    "tuple",
    "dict",
    "set",
    "sum",
    "sorted",
    "reversed",
    "map",
    "filter",
    # pre-injected helpers
    "make_frustum",
    "make_solid_revolution",
    "export_stl",
}

# Trimesh backend API surface
ALLOWED_CALLS_TRIMESH: set[str] = {
    "trimesh.Trimesh",
    "trimesh.Scene",
    "trimesh.load",
    "trimesh.creation.box",
    "trimesh.creation.cylinder",
    "trimesh.creation.cone",
    "trimesh.creation.capsule",
    "trimesh.creation.icosphere",
    "trimesh.creation.torus",
    "trimesh.creation.annulus",
    "trimesh.creation.extrude_polygon",
    "trimesh.util.concatenate",
    "trimesh.transformations.rotation_matrix",
    "trimesh.transformations.translation_matrix",
    "trimesh.transformations.scale_matrix",
    "trimesh.boolean.difference",
    "trimesh.boolean.union",
    "trimesh.boolean.intersection",
}

# CadQuery backend API surface (calls that appear at top level — method calls
# on returned objects are handled by the method allowlist below).
ALLOWED_CALLS_CADQUERY: set[str] = {
    "cadquery.Workplane",
    "cadquery.Sketch",
    "cadquery.Vector",
    "cadquery.Plane",
    "cadquery.exporters.export",
    "cq.Workplane",
    "cq.Sketch",
    "cq.Vector",
    "cq.Plane",
    "cq.exporters.export",
    "Workplane",
    "Sketch",
    "Vector",
    "Plane",
    "export_stl",
}

# Method names allowed on arbitrary objects. We don't enforce types — we only
# enforce that the final attribute name is known, which is a reasonable
# compromise between safety and false-positive rate.
ALLOWED_METHODS: set[str] = {
    # trimesh mesh methods
    "apply_translation",
    "apply_transform",
    "apply_scale",
    "export",
    "copy",
    # cadquery workplane methods
    "box",
    "sphere",
    "cylinder",
    "wedge",
    "extrude",
    "revolve",
    "sweep",
    "loft",
    "shell",
    "fillet",
    "chamfer",
    "cut",
    "union",
    "intersect",
    "translate",
    "rotate",
    "rotateAboutCenter",
    "workplane",
    "polyline",
    "close",
    "circle",
    "rect",
    "polygon",
    "ellipse",
    "ellipseArc",
    "line",
    "lineTo",
    "hLine",
    "hLineTo",
    "vLine",
    "vLineTo",
    "moveTo",
    "move",
    "threePointArc",
    "radiusArc",
    "sagittaArc",
    "tangentArcPoint",
    "spline",
    "mirror",
    "mirrorX",
    "mirrorY",
    "offset2D",
    "push",
    "pushPoints",
    "eachpoint",
    "each",
    "tag",
    "end",
    "transformed",
    "copyWorkplane",
    "rarray",
    "cboreHole",
    "cskHole",
    "hole",
    "text",
    "slot2D",
    "faces",
    "edges",
    "vertices",
    "center",
    "first",
    "last",
    "combine",
    "findSolid",
    "val",
    "vals",
    "all",
    "add",
    # shapely polygon methods
    "buffer",
    "simplify",
    "exterior",
    "coords",
    # numpy array methods (safe)
    "reshape",
    "transpose",
    "tolist",
    "flatten",
    "mean",
    "sum",
    "max",
    "min",
    # generic
    "append",
    "extend",
    "pop",
    "insert",
    "remove",
    "update",
    "keys",
    "values",
    "items",
    "get",
    "upper",
    "lower",
    "strip",
    "split",
    "join",
    "format",
    "replace",
    # helpers
    "centroid",
}

# Names that MUST NEVER appear in generated code
FORBIDDEN_NAMES: set[str] = {
    "exec",
    "eval",
    "compile",
    "__import__",
    "open",
    "input",
    "breakpoint",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "hasattr",
    "subprocess",
    "os",
    "sys",
    "socket",
    "urllib",
    "requests",
    "httpx",
    "shutil",
    "pathlib",
    "Path",  # paths are injected (OUTPUT_PATH)
    "pickle",
    "marshal",
    "importlib",
}


# =============================================================================
# Validator
# =============================================================================

@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


def _dotted_name(node: ast.AST) -> str | None:
    """Resolve an Attribute/Name chain into a dotted string.

    e.g. trimesh.creation.box -> "trimesh.creation.box"
    Returns None if the chain contains a non-Name/Attribute node (e.g. a call).
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        if parent is None:
            return None
        return f"{parent}.{node.attr}"
    return None


def _leaf_attr(dotted: str) -> str:
    """Return the last component of a dotted name: a.b.c -> c"""
    return dotted.rsplit(".", 1)[-1]


def _call_name(call: ast.Call) -> str | None:
    """Return the dotted name of a Call's func, or None if dynamic."""
    return _dotted_name(call.func)


def _collect_names(tree: ast.AST) -> tuple[set[str], set[str]]:
    """First pass: collect (module_names, var_names) from imports and assignments.

    - module_names: modules imported (including aliases). e.g. trimesh, np, cq
    - var_names: ordinary variables (assignments, function params, for loops, etc.)
    """
    module_names: set[str] = set()
    var_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # `from x import y` → y is a name from module x, treat as var
            # (it could be a function, class, or submodule; leaf-method logic
            # does not apply to it directly)
            for alias in node.names:
                var_names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            var_names.add(node.name)
            for arg in node.args.args:
                var_names.add(arg.arg)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    var_names.add(tgt.id)
                elif isinstance(tgt, ast.Tuple):
                    for elt in tgt.elts:
                        if isinstance(elt, ast.Name):
                            var_names.add(elt.id)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            var_names.add(node.target.id)
        elif isinstance(node, ast.For):
            if isinstance(node.target, ast.Name):
                var_names.add(node.target.id)
            elif isinstance(node.target, ast.Tuple):
                for elt in node.target.elts:
                    if isinstance(elt, ast.Name):
                        var_names.add(elt.id)
        elif isinstance(node, ast.comprehension) and isinstance(node.target, ast.Name):
            var_names.add(node.target.id)
    return module_names, var_names


def validate_code(
    code: str,
    allowed_calls: Iterable[str] = (),
    allowed_methods: Iterable[str] = (),
) -> ValidationResult:
    """Parse code and reject unknown API calls, forbidden names, and bad imports.

    Rules:
    - Bare function call foo(): must be in calls_ok OR a user-defined function
    - Module call mod.fn(): the full dotted path must be in calls_ok
    - Method on local var.method(): method name must be in methods_ok
    - Chained call x.y().method(): method name must be in methods_ok
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return ValidationResult(ok=False, errors=[f"SyntaxError line {e.lineno}: {e.msg}"])

    calls_ok = ALLOWED_CALLS_COMMON | set(allowed_calls)
    methods_ok = ALLOWED_METHODS | set(allowed_methods)
    allowed_mod_roots = {m.split(".")[0] for m in ALLOWED_IMPORTS}
    errors: list[str] = []

    module_names, var_names = _collect_names(tree)

    for node in ast.walk(tree):
        # Forbidden bare names (e.g. __import__, open, exec, os)
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            errors.append(f"Forbidden name: {node.id}")
            continue

        # Imports must be whitelisted
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in allowed_mod_roots:
                    errors.append(f"Forbidden import: {alias.name}")
            continue
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod and mod.split(".")[0] not in allowed_mod_roots:
                errors.append(f"Forbidden import: from {mod}")
            continue

        # Function calls must be in the allowlist
        if isinstance(node, ast.Call):
            func = node.func

            # Case 1: bare name call  foo()
            if isinstance(func, ast.Name):
                name = func.id
                if name in FORBIDDEN_NAMES:
                    continue  # already flagged
                if name in calls_ok:
                    continue
                if name in var_names:
                    continue  # user-defined function / callable
                errors.append(f"Unknown function: {name}()")
                continue

            # Case 2: attribute call  X.method()
            if isinstance(func, ast.Attribute):
                leaf = func.attr
                full = _dotted_name(func)

                if full is not None:
                    # Pure dotted chain (no intermediate calls)
                    root = full.split(".")[0]
                    # Module-rooted call has highest precedence (handles
                    # aliases like `import cadquery as cq` → cq.Workplane)
                    if root in module_names:
                        if full in calls_ok:
                            continue
                        errors.append(f"Unknown module API: {full}()")
                        continue
                    if root in var_names:
                        # Method on local variable
                        if leaf in methods_ok:
                            continue
                        errors.append(f"Unknown method: {root}.{leaf}()")
                        continue
                    # Unknown root (neither module nor var) — rare
                    errors.append(f"Unknown reference: {full}()")
                    continue
                else:
                    # Chained call like cq.Workplane('XY').box(...).magicShape(...)
                    # We can only check the leaf method name.
                    if leaf in methods_ok:
                        continue
                    errors.append(f"Unknown chained method: .{leaf}()")
                    continue

            # Other call types (Lambda, Subscript, etc.) — allow, very rare
            continue

    # Dedupe while preserving order
    seen = set()
    deduped = []
    for e in errors:
        if e not in seen:
            seen.add(e)
            deduped.append(e)

    return ValidationResult(ok=len(deduped) == 0, errors=deduped)


# =============================================================================
# Convenience: backend-specific validators
# =============================================================================

def validate_trimesh(code: str) -> ValidationResult:
    return validate_code(code, allowed_calls=ALLOWED_CALLS_TRIMESH)


def validate_cadquery(code: str) -> ValidationResult:
    res = validate_code(code, allowed_calls=ALLOWED_CALLS_CADQUERY)
    # Topology pre-check on top of the API allowlist: catches the most common
    # silent-failure mode for shoes/bottles/organic shapes — the LLM emits
    # multiple `.polyline(...)` calls feeding into a single `.loft()` chain
    # but with mismatched point counts. CadQuery's loft requires identical
    # vertex counts across all sections; mismatched counts produce degenerate
    # geometry (or fail at exec with an opaque BRep error). Catching it here
    # gives the LLM a clear, actionable retry message.
    try:
        loft_errs = check_loft_topology(code)
    except Exception:
        loft_errs = []
    if loft_errs:
        merged = list(res.errors) + loft_errs
        return ValidationResult(ok=False, errors=merged)
    return res


def check_loft_topology(code: str) -> list[str]:
    """Return a list of human-readable errors for any `.loft()` call whose
    polyline cross-sections have mismatched point counts.

    Resolves polyline arguments that reference module-level list literals
    (e.g. `sole_pts = [(0,30),(15,10),...]; ...polyline(sole_pts)...`).

    Returns [] if all loft chains are well-formed (or if no loft is present)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    # Step 1: build name -> length map for module-level list literals.
    # Only count entries that look like (x, y) tuples; ignore everything else
    # so we don't flag false positives on unrelated lists.
    name_to_len: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.List):
            elts = node.value.elts
            if not elts:
                continue
            if not all(isinstance(e, ast.Tuple) and len(e.elts) in (2, 3) for e in elts):
                continue
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    name_to_len[tgt.id] = len(elts)

    def polyline_len(arg: ast.AST) -> int | None:
        """Return point count for a polyline argument, or None if unknown."""
        if isinstance(arg, ast.List):
            if all(isinstance(e, ast.Tuple) and len(e.elts) in (2, 3) for e in arg.elts):
                return len(arg.elts)
            return None
        if isinstance(arg, ast.Name):
            return name_to_len.get(arg.id)
        return None

    # Step 2: for every `.loft(...)` call, walk back through the call chain
    # collecting `.polyline(arg)` cross-sections.
    errors: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "loft"):
            continue
        # Walk receiver chain
        sections: list[tuple[str, int | None]] = []  # (label, n_points)
        cursor: ast.AST | None = node.func.value
        while isinstance(cursor, ast.Call) and isinstance(cursor.func, ast.Attribute):
            attr = cursor.func.attr
            if attr == "polyline" and cursor.args:
                arg = cursor.args[0]
                label = arg.id if isinstance(arg, ast.Name) else f"<inline-{getattr(arg, 'lineno', '?')}>"
                sections.append((label, polyline_len(arg)))
            cursor = cursor.func.value
        if len(sections) < 2:
            continue  # not a loft chain we recognize
        # Reverse so they're in source order (we walked outward)
        sections.reverse()
        known = [(lab, n) for lab, n in sections if n is not None]
        if len(known) < 2:
            continue  # can't decide; let exec catch it
        counts = sorted({n for _, n in known})
        if len(counts) > 1:
            detail = ", ".join(f"{lab}={n}" for lab, n in known)
            errors.append(
                f"loft topology mismatch — sections must have IDENTICAL point counts, "
                f"got {detail}. CadQuery's .loft() requires every cross-section "
                f"polyline to have the same number of vertices. "
                f"FIX: replace the entire result with EXACTLY this 2-section shoe "
                f"(both polylines have 15 points), then adjust dimensions if needed:\n"
                f"```python\n"
                f"import cadquery as cq\n"
                f"sole_pts  = [(0,30),(15,10),(50,0),(120,0),(200,0),(250,10),(275,30),"
                f"(280,55),(270,80),(240,95),(180,100),(100,100),(40,95),(10,80),(0,55)]\n"
                f"upper_pts = [(20,40),(40,25),(80,20),(140,20),(200,20),(240,25),(255,40),"
                f"(255,65),(240,80),(200,90),(140,90),(80,90),(40,85),(20,70),(20,55)]\n"
                f"result = (cq.Workplane(\"XY\").polyline(sole_pts).close()"
                f".workplane(offset=70).polyline(upper_pts).close().loft(combine=True))\n"
                f"export_stl(result, OUTPUT_PATH)\n"
                f"```"
            )
    return errors


def format_errors_for_llm(errors: list[str], max_errors: int = 8) -> str:
    """Produce a concise error message suitable for feeding back to the LLM."""
    shown = errors[:max_errors]
    out = "The generated code failed pre-execution validation:\n"
    out += "\n".join(f"- {e}" for e in shown)
    if len(errors) > max_errors:
        out += f"\n(...and {len(errors) - max_errors} more)"
    out += (
        "\n\nPlease regenerate valid code using ONLY the APIs listed in the "
        "system prompt. Do not invent functions that don't exist."
    )
    return out
