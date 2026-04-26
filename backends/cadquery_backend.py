"""CadQuery backend — BREP engine for richer feature support (fillet, sweep, loft).

Generated code contract:
  - Imports: `import cadquery as cq`
  - Produces a variable `result` that is either a Workplane or Shape
  - Does NOT call .export() directly — the backend calls export_stl(result, OUTPUT_PATH)

We also inject `export_stl(result, path)` so simple scripts can optionally call it;
the backend also calls it automatically if `result` is set.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from .base import ModelBackend, BackendError
from . import register

log = logging.getLogger("text2stl.backend.cadquery")


def _make_helpers():
    try:
        import cadquery as _cq
        from cadquery import exporters as _exporters
    except ImportError as e:
        log.warning(f"cadquery not installed: {e}")
        _cq = None
        _exporters = None

    def export_stl(result, path):
        """Export a CadQuery Workplane/Shape to STL."""
        if _exporters is None:
            raise BackendError("cadquery is not installed — cannot export STL")
        _exporters.export(result, str(path), exportType="STL")

    helpers = {"export_stl": export_stl}
    if _cq is not None:
        helpers["cq"] = _cq
        helpers["cadquery"] = _cq
    return helpers


_HELPERS = _make_helpers()


def export_multi_format(result, stl_path: Path) -> dict:
    """Export a CadQuery result to STL + STEP + 3MF + GLB (S5.2).

    Best-effort: each format failure is captured but doesn't block the
    others. Returns a dict {format: path_or_None}.
    The STL itself is written by the caller via execute_and_export;
    this function adds the other three.
    """
    out: dict[str, Path | None] = {}
    base = stl_path.parent
    stem = stl_path.stem  # "model"

    # STEP — true CAD, downstream-editable
    step_path = base / f"{stem}.step"
    try:
        from cadquery import exporters
        exporters.export(result, str(step_path), exportType="STEP")
        out["step"] = step_path
    except Exception as e:
        log.debug(f"STEP export failed: {e}")
        out["step"] = None

    # 3MF — printer-friendly metadata format
    mf3_path = base / f"{stem}.3mf"
    try:
        from cadquery import exporters
        # CadQuery supports 3MF via opencascade
        exporters.export(result, str(mf3_path), exportType="3MF")
        out["3mf"] = mf3_path
    except Exception as e:
        # Fallback: convert STL → 3MF via trimesh
        try:
            import trimesh
            m = trimesh.load(str(stl_path))
            m.export(str(mf3_path), file_type="3mf")
            out["3mf"] = mf3_path
        except Exception as e2:
            log.debug(f"3MF export failed: {e} / fallback: {e2}")
            out["3mf"] = None

    # GLB — web preview
    glb_path = base / f"{stem}.glb"
    try:
        import trimesh
        m = trimesh.load(str(stl_path))
        m.export(str(glb_path), file_type="glb")
        out["glb"] = glb_path
    except Exception as e:
        log.debug(f"GLB export failed: {e}")
        out["glb"] = None

    return out


class CadQueryBackend(ModelBackend):
    name = "cadquery"

    def helper_globals(self) -> dict:
        return dict(_HELPERS)

    def execute_and_export(self, code: str, stl_path: Path,
                           extra_formats: bool = False) -> None:
        # Strip OUTPUT_PATH reassignment
        code = re.sub(r"^OUTPUT_PATH\s*=.*$", "# OUTPUT_PATH is injected", code, flags=re.MULTILINE)

        exec_globals = {
            "OUTPUT_PATH": str(stl_path),
            "__builtins__": __builtins__,
        }
        exec_globals.update(_HELPERS)

        try:
            exec(code, exec_globals)
        except Exception as e:
            raise BackendError(f"{type(e).__name__}: {e}") from e

        result = exec_globals.get("result")

        # If the code didn't export itself, do it now using `result`
        if not stl_path.exists():
            if result is None:
                raise BackendError(
                    "Neither `export_stl(result, OUTPUT_PATH)` was called nor a "
                    "`result` variable was produced. Your code must end with "
                    "`result = <workplane>` or call `export_stl(result, OUTPUT_PATH)`."
                )
            try:
                _HELPERS["export_stl"](result, stl_path)
            except Exception as e:
                raise BackendError(f"Auto-export failed: {type(e).__name__}: {e}") from e

        if not stl_path.exists():
            raise BackendError("STL file was not created after execution.")

        # S5.2: optionally also export STEP / 3MF / GLB
        if extra_formats and result is not None:
            try:
                export_multi_format(result, stl_path)
            except Exception as e:
                log.warning(f"extra_formats export raised: {e}")

    def allowed_calls(self) -> set[str]:
        from validators import ALLOWED_CALLS_CADQUERY
        return set(ALLOWED_CALLS_CADQUERY)

    def system_prompt(self) -> str:
        return _read("system_cadquery.md")

    def enrich_prompt(self) -> str:
        return _read("enrich_cadquery.md")

    def review_prompt(self) -> str:
        return _read("review_generic.md")


def _read(name: str) -> str:
    p = Path(__file__).parent.parent / "prompts" / name
    return p.read_text(encoding="utf-8") if p.exists() else ""


register("cadquery", CadQueryBackend())
