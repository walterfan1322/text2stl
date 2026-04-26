"""Trimesh backend — legacy path, kept for A/B fallback."""
from __future__ import annotations

import logging
import re
from pathlib import Path

from .base import ModelBackend, BackendError
from . import register

log = logging.getLogger("text2stl.backend.trimesh")


# -----------------------------------------------------------------------------
# Helper globals (injected into generated code)
# -----------------------------------------------------------------------------
def _make_helpers():
    try:
        import numpy as _np
        import trimesh as _trimesh
    except ImportError as e:
        log.warning(f"trimesh/numpy not installed: {e}")
        return {}

    def make_frustum(r_bottom, r_top, height, sections=64):
        """Solid frustum (truncated cone) along +Z."""
        angles = _np.linspace(0, 2 * _np.pi, sections, endpoint=False)
        bottom = _np.column_stack([
            r_bottom * _np.cos(angles),
            r_bottom * _np.sin(angles),
            _np.zeros(sections),
        ])
        top = _np.column_stack([
            r_top * _np.cos(angles),
            r_top * _np.sin(angles),
            _np.full(sections, height),
        ])
        vertices = _np.vstack([bottom, top, [[0, 0, 0]], [[0, 0, height]]])
        bc, tc = 2 * sections, 2 * sections + 1
        faces = []
        for i in range(sections):
            j = (i + 1) % sections
            faces.append([i, j, sections + j])
            faces.append([i, sections + j, sections + i])
            faces.append([bc, j, i])
            faces.append([tc, sections + i, sections + j])
        return _trimesh.Trimesh(vertices=_np.array(vertices), faces=_np.array(faces))

    def make_solid_revolution(profile, sections=64):
        """Revolve a CLOSED 2D profile polygon around Z axis into a watertight solid.

        profile: list of (radius, z) points forming a CLOSED polygon.
        """
        angles = _np.linspace(0, 2 * _np.pi, sections, endpoint=False)
        n = len(profile)
        vertices = []
        for r, z in profile:
            for a in angles:
                vertices.append([r * _np.cos(a), r * _np.sin(a), z])
        vertices = _np.array(vertices)
        faces = []
        for i in range(n):
            next_i = (i + 1) % n
            for j in range(sections):
                k = (j + 1) % sections
                a = i * sections + j
                b = i * sections + k
                c = next_i * sections + k
                d = next_i * sections + j
                faces.append([a, b, c])
                faces.append([a, c, d])
        return _trimesh.Trimesh(vertices=vertices, faces=_np.array(faces))

    return {"make_frustum": make_frustum, "make_solid_revolution": make_solid_revolution}


_HELPERS = _make_helpers()


# -----------------------------------------------------------------------------
# Backend
# -----------------------------------------------------------------------------
class TrimeshBackend(ModelBackend):
    name = "trimesh"

    def helper_globals(self) -> dict:
        return dict(_HELPERS)

    def execute_and_export(self, code: str, stl_path: Path) -> None:
        # Strip re-definitions / bad reassigns (already filtered in clean_code,
        # but we double-protect here).
        code = re.sub(r"^OUTPUT_PATH\s*=.*$", "# OUTPUT_PATH is injected", code, flags=re.MULTILINE)
        for fname in ("make_frustum", "make_solid_revolution"):
            pat = rf"^def {fname}\(.*\):[ \t]*\n(?:(?:[ \t]+.*|[ \t]*)\n)*"
            code = re.sub(pat, "", code, flags=re.MULTILINE)
        if "trimesh.creation.revolve" in code:
            log.warning("Fixing LLM mistake: replacing trimesh.creation.revolve with make_solid_revolution")
            code = code.replace("trimesh.creation.revolve", "make_solid_revolution")

        exec_globals = {"OUTPUT_PATH": str(stl_path), "__builtins__": __builtins__}
        exec_globals.update(_HELPERS)
        try:
            exec(code, exec_globals)
        except Exception as e:
            raise BackendError(f"{type(e).__name__}: {e}") from e

        if not stl_path.exists():
            raise BackendError(
                "STL file was not created. Code must call <mesh>.export(OUTPUT_PATH)"
            )

    def allowed_calls(self) -> set[str]:
        from validators import ALLOWED_CALLS_TRIMESH
        return set(ALLOWED_CALLS_TRIMESH)

    def system_prompt(self) -> str:
        return _read("system_trimesh.md")

    def enrich_prompt(self) -> str:
        return _read("enrich_trimesh.md")

    def review_prompt(self) -> str:
        return _read("review_generic.md")


def _read(name: str) -> str:
    p = Path(__file__).parent.parent / "prompts" / name
    return p.read_text(encoding="utf-8") if p.exists() else ""


register("trimesh", TrimeshBackend())
