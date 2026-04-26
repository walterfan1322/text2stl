"""Text2STL - Natural language to 3D model (STL) service.

Uses LLM to generate Python/trimesh code, executes it to produce STL files.
SSH tunnel and Ollama model are loaded on-demand and released after idle.
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx

from validators import (
    validate_trimesh,
    validate_cadquery,
    format_errors_for_llm,
    validate_code,
)
from backends import get_backend, BackendError
from rendering import render_stl_views
from judge import judge_model, build_retry_instruction, JudgeResult

# Directories
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
STATIC_DIR = BASE_DIR / "static"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load config from file (avoids env var encoding issues with wmic)
_config_path = BASE_DIR / "config.json"
_cfg = {}
if _config_path.exists():
    _cfg = json.loads(_config_path.read_text(encoding="utf-8"))


# S4.5: overlay env vars (and .env.local) on top of config.json so
# secrets can stay out of the committed JSON. Priority (highest first):
# real env > .env.local > config.json defaults.
def _load_env_local(p: Path) -> dict:
    env: dict = {}
    if not p.exists():
        return env
    try:
        for line in p.read_text("utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip().strip('"').strip("'")
    except Exception:
        pass
    return env


_env_local = _load_env_local(BASE_DIR / ".env.local")


def _env_or(name: str, default=None):
    v = os.environ.get(name) or _env_local.get(name)
    return v if v else default


# Apply API-key overrides to cloud_providers (keys only, not base URLs).
_env_key_map = {
    "minimax":  "MINIMAX_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "gemini":   "GEMINI_API_KEY",
}
for _pid, _env_name in _env_key_map.items():
    _k = _env_or(_env_name)
    if _k and _pid in _cfg.get("cloud_providers", {}):
        _cfg["cloud_providers"][_pid]["key"] = _k
_cfg["cloud_api_key"] = _env_or("CLOUD_API_KEY", _cfg.get("cloud_api_key", ""))
_cfg["cloud_api_base"] = _env_or("CLOUD_API_BASE", _cfg.get("cloud_api_base", ""))


OLLAMA_URL = _cfg.get("ollama_url", "http://localhost:11434")
OLLAMA_MODEL = _cfg.get("ollama_model", "qwen3:14b")
SSH_TUNNEL_HOST = _cfg.get("ssh_tunnel_host", "user@YOUR_DGX_HOST")
SSH_KEY_PATH = _cfg.get("ssh_key_path", "")
TUNNEL_IDLE_TIMEOUT = int(_cfg.get("tunnel_idle_timeout", 300))

# Cloud API (OpenAI-compatible, multi-provider)
# Legacy single-provider fields (kept for backward compat / vision judge fallback)
CLOUD_API_KEY = _cfg.get("cloud_api_key", "")
CLOUD_API_BASE = _cfg.get("cloud_api_base", "")
CLOUD_MODELS = _cfg.get("cloud_models", [])
CLOUD_VISION_MODEL = _cfg.get("cloud_vision_model", "")
# New: chain of vision models for ensemble judge (S1.2). If empty, falls
# back to [CLOUD_VISION_MODEL] alone.
CLOUD_VISION_MODELS = _cfg.get("cloud_vision_models", [])

# S1.6: per-model failover chain. If the primary returns 429/5xx, retry
# with the next model in the chain. Only triggers for the /api/generate
# LLM (not the judge — the judge has its own cross-provider chain).
# Example: {"MiniMax-M2.7": ["deepseek-chat"]}
MODEL_FAILOVER: dict = _cfg.get("model_failover", {})

# New: map of provider_id -> {"base": "...", "key": "..."}
CLOUD_PROVIDERS: dict = _cfg.get("cloud_providers", {})
# Map of model_name -> provider_id
MODEL_PROVIDER: dict = _cfg.get("model_provider", {})


def _resolve_provider(model: str) -> tuple[str, str]:
    """Return (api_base, api_key) for the given cloud model.

    Uses the model→provider table if present; otherwise falls back to
    the legacy single-provider CLOUD_API_BASE / CLOUD_API_KEY.
    """
    pid = MODEL_PROVIDER.get(model)
    if pid and pid in CLOUD_PROVIDERS:
        p = CLOUD_PROVIDERS[pid]
        return p.get("base", CLOUD_API_BASE), p.get("key", CLOUD_API_KEY)
    return CLOUD_API_BASE, CLOUD_API_KEY

# Feature flags
BACKEND = _cfg.get("backend", "trimesh")  # "trimesh" | "cadquery"
AST_VALIDATE = bool(_cfg.get("ast_validate", True))
JUDGE_ENABLED = bool(_cfg.get("judge_enabled", False))
JUDGE_MAX_RETRIES = int(_cfg.get("judge_max_retries", 2))
JUDGE_MIN_SCORE = int(_cfg.get("judge_min_score", 6))
# S4.1: run mesh_repair.repair_stl after every successful execute_and_export.
MESH_REPAIR_ENABLED = bool(_cfg.get("mesh_repair_enabled", True))
# S4.2: skip VLM judge and retry with watertight hint if STL isn't printable.
WATERTIGHT_GATE_ENABLED = bool(_cfg.get("watertight_gate_enabled", True))
# P3: per-shape routing — override user's model choice when we have
# benchmark evidence that a different model is materially better for
# the inferred shape category. Empty / disabled = honor user's choice.
SHAPE_ROUTING_ENABLED = bool(_cfg.get("shape_routing_enabled", False))
SHAPE_ROUTING: dict = _cfg.get("shape_routing", {})

# Sprint 5-7: feature flags. Each is a single point of rollback. Reading
# from config.feature_flags so all 5/6/7 flags share a single namespace.
_FF: dict = _cfg.get("feature_flags", {})
FEATURE_OUTPUT_CACHE        = bool(_FF.get("output_cache", False))
FEATURE_STRUCTURED_LOG      = bool(_FF.get("structured_log", False))
FEATURE_GEOM_CHECK          = bool(_FF.get("geom_check", False))
FEATURE_PRINT_READINESS     = bool(_FF.get("print_readiness", False))
FEATURE_MULTI_FORMAT_EXPORT = bool(_FF.get("multi_format_export", False))
FEATURE_SANDBOX_STRICT      = bool(_FF.get("sandbox_strict", False))
FEATURE_BEST_OF_N           = bool(_FF.get("best_of_n", False))
FEATURE_SLICER_CHECK        = bool(_FF.get("slicer_check", False))
FEATURE_RENDER_PYVISTA      = bool(_FF.get("render_pyvista", False))
SLICER_PATH: str            = _cfg.get("slicer_path", "")
BEST_OF_PER_CATEGORY: dict  = _cfg.get("best_of_per_category", {})

# S7.1 / S7.3 shadow-mode: contextvar prevents recursive shadow spawning
# (parent generate spawns shadow → shadow re-enters generate → would
# spawn its own shadows ad infinitum without this).
import contextvars as _cvs  # noqa: E402
_BEST_OF_N_SHADOW_CTX: "_cvs.ContextVar[bool]" = _cvs.ContextVar(
    "_best_of_n_shadow_running", default=False
)

# S4.4: rotating file handler so server.log can't grow unbounded.
from logging.handlers import RotatingFileHandler as _Rot  # noqa: E402
_log_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_file_handler = _Rot(
    BASE_DIR / "server.log",
    maxBytes=5 * 1024 * 1024,   # 5 MB per file
    backupCount=5,              # keep 5 rotated copies → ~30 MB cap
    encoding="utf-8",
)
_file_handler.setFormatter(_log_fmt)
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_log_fmt)
logging.basicConfig(level=logging.INFO, handlers=[_stream_handler, _file_handler])
log = logging.getLogger("text2stl")

app = FastAPI(title="Text2STL", description="自然語言 3D 建模服務")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# On-demand SSH tunnel manager
# ---------------------------------------------------------------------------
class TunnelManager:
    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._last_used: float = 0
        self._active_requests: int = 0
        self._lock = asyncio.Lock()
        self._watchdog_task: asyncio.Task | None = None

    @property
    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    async def ensure_tunnel(self):
        """Start tunnel if not running; reset idle timer."""
        async with self._lock:
            if not self.is_alive:
                await self._start()
            self._last_used = time.time()
            self._active_requests += 1

    def release(self):
        """Mark a request as done; reset idle timer."""
        self._active_requests = max(0, self._active_requests - 1)
        self._last_used = time.time()

    async def _start(self):
        log.info(f"Opening SSH tunnel to {SSH_TUNNEL_HOST}...")
        # Split user@host properly
        ssh_user, ssh_host = SSH_TUNNEL_HOST.split("@") if "@" in SSH_TUNNEL_HOST else ("", SSH_TUNNEL_HOST)
        cmd = ["ssh",
               "-o", "StrictHostKeyChecking=no",
               "-o", "ServerAliveInterval=30",
               "-o", "ExitOnForwardFailure=yes",
               "-N", "-L", "11434:localhost:11434"]
        if SSH_KEY_PATH:
            cmd.extend(["-i", SSH_KEY_PATH])
        if ssh_user:
            cmd.extend(["-l", ssh_user])
        cmd.append(ssh_host)
        log_path = BASE_DIR / "tunnel.log"
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=open(log_path, "w"),
        )
        self._last_used = time.time()
        # Wait for tunnel to be ready
        for _ in range(10):
            await asyncio.sleep(0.5)
            try:
                async with httpx.AsyncClient(timeout=2) as c:
                    r = await c.get(f"{OLLAMA_URL}/")
                    if r.status_code == 200:
                        log.info(f"SSH tunnel ready (PID {self._proc.pid})")
                        return
            except Exception:
                pass
        log.warning("SSH tunnel started but Ollama not yet reachable")

    def _stop(self):
        if self._proc:
            log.info("Closing SSH tunnel (idle timeout)")
            self._proc.terminate()
            self._proc = None

    async def watchdog(self):
        """Periodically check if tunnel should be closed due to idle."""
        while True:
            await asyncio.sleep(30)
            async with self._lock:
                if self.is_alive and self._active_requests == 0 and (time.time() - self._last_used > TUNNEL_IDLE_TIMEOUT):
                    self._stop()

    def shutdown(self):
        if self._proc:
            self._proc.terminate()
            self._proc = None


tunnel = TunnelManager()


@app.on_event("startup")
async def on_startup():
    tunnel._watchdog_task = asyncio.create_task(tunnel.watchdog())


@app.on_event("shutdown")
async def on_shutdown():
    if tunnel._watchdog_task:
        tunnel._watchdog_task.cancel()
    tunnel.shutdown()


# ---------------------------------------------------------------------------
# Backend & prompts (dispatched by config.backend)
# ---------------------------------------------------------------------------
_BACKEND = get_backend(BACKEND)
log.info(f"Active backend: {_BACKEND.name}")

SYSTEM_PROMPT = _BACKEND.system_prompt()
ENRICH_PROMPT = _BACKEND.enrich_prompt()
REVIEW_PROMPT = _BACKEND.review_prompt()

# S3.1: pattern cache — retrieval-augmented few-shot store.
from pattern_cache import PatternCache, format_examples_block
_PATTERN_CACHE_PATH = BASE_DIR / "pattern_cache.json"
PATTERN_CACHE = PatternCache(_PATTERN_CACHE_PATH)
PATTERN_CACHE_ENABLED = bool(_cfg.get("pattern_cache_enabled", True))

# S3.4: token / cost monitor — JSONL log.
from token_monitor import TokenMonitor
TOKEN_MONITOR = TokenMonitor(BASE_DIR / "token_usage.jsonl")

# S5.1: exact-match LLM-output cache (sqlite-backed).
from output_cache import OutputCache
OUTPUT_CACHE = OutputCache(
    BASE_DIR / "output_cache.db",
    outputs_root=OUTPUT_DIR,
)

# S7.2: structured per-generation log (JSONL).
from structured_log import StructuredLog
STRUCTURED_LOG = StructuredLog(BASE_DIR / "structured_log.jsonl")


def _record_llm_usage(model: str, usage: dict | None) -> None:
    """Called by call_cloud_llm on each 200 response."""
    TOKEN_MONITOR.record(model, usage)


def _build_system_prompt_for(prompt: str) -> str:
    """SYSTEM_PROMPT optionally appended with retrieval-augmented few-shots."""
    if not PATTERN_CACHE_ENABLED:
        return SYSTEM_PROMPT
    examples = PATTERN_CACHE.examples_for(prompt)
    if not examples:
        return SYSTEM_PROMPT
    # S4.6: log injections so we can eventually answer
    # 'did pattern cache actually help first-attempt score?'
    try:
        from pattern_cache import infer_category
        cat = infer_category(prompt)
        avg = sum(e.get("score", 0) for e in examples) / max(len(examples), 1)
        log.info(
            f"pattern_cache: inject {len(examples)} example(s) "
            f"for category '{cat}' (avg cached score {avg:.1f})"
        )
    except Exception:
        pass
    return SYSTEM_PROMPT + format_examples_block(examples)

# Legacy inline prompts (superseded by prompts/*.md via backend). Retained as
# a safety fallback in case prompt files are missing at deploy time.
_LEGACY_PROMPT = r"""You are an expert Python 3D modeling programmer. The user will describe a 3D object in natural language.
You must generate a Python script that creates the described 3D object and exports it as an STL file.

Available libraries (already installed):
- trimesh (version 4.x): For creating and manipulating 3D meshes
- numpy: For mathematical operations

Coordinate system: X = left/right, Y = front/back, Z = up/down. Objects sit on XY plane (Z=0 is ground).

Rules:
- Output ONLY valid Python code. No explanations, no markdown fences, no thinking.
- The variable `OUTPUT_PATH` is already defined. Do NOT reassign it. Just use it directly: `mesh.export(OUTPUT_PATH)`
- Use `mesh.export(OUTPUT_PATH)` at the end.
- Use millimeters as the unit.
- Center the object at the origin when possible (use `mesh.centroid` to get center, then `mesh.apply_translation(-mesh.centroid)`).
- Do NOT use `mesh.center()` — it does not exist. Use `mesh.apply_translation(-mesh.centroid)` instead.
- Make reasonable assumptions about dimensions if not specified.
- For boolean operations, pass a LIST of meshes as the first argument:
  trimesh.boolean.difference([mesh_a, mesh_b], engine='manifold')
  trimesh.boolean.union([mesh_a, mesh_b], engine='manifold')
  trimesh.boolean.intersection([mesh_a, mesh_b], engine='manifold')
  WRONG: trimesh.boolean.difference(mesh_a, mesh_b) — this will CRASH.
- STRONGLY PREFER building shapes by combining primitives with `trimesh.util.concatenate`. AVOID boolean operations (difference/union/intersection) whenever possible as they frequently crash on non-watertight meshes. Keep designs simple — NO fillets, NO decorative holes, NO engravings.

AVAILABLE trimesh.creation functions (ONLY use these):
- trimesh.creation.box(extents=[w, h, d])
- trimesh.creation.cylinder(radius=r, height=h, sections=64)
- trimesh.creation.cone(radius=r, height=h, sections=64)
- trimesh.creation.capsule(radius=r, height=h)
- trimesh.creation.icosphere(radius=r, subdivisions=3)
- trimesh.creation.torus(major_radius=R, minor_radius=r)
- trimesh.creation.annulus(r_min, r_max, height)
- trimesh.creation.extrude_polygon(polygon, height)  # shapely Polygon

DO NOT USE these functions (they will CRASH):
- trimesh.creation.frustum — does NOT exist
- trimesh.creation.conical_frustum — does NOT exist
- trimesh.creation.revolve — CRASHES, use make_solid_revolution() instead
- trimesh.creation.sphere — does NOT exist, use icosphere instead
- trimesh.creation.rounded_box — does NOT exist

PRE-DEFINED HELPER FUNCTIONS (injected into your execution environment — just call them directly like any built-in function. Do NOT redefine, do NOT import, do NOT wrap in try/except):

1. make_frustum(r_bottom, r_top, height, sections=64)
   Creates a solid frustum (truncated cone). Returns a trimesh.Trimesh.
   Example: frustum = make_frustum(20, 30, 100)

2. make_solid_revolution(profile, sections=64)
   Revolves a CLOSED 2D profile polygon around Z axis to create a watertight solid.
   profile: list of (radius, z) tuples forming a CLOSED polygon (wall cross-section).
   Trace the outer surface from bottom to top, then inner surface from top to bottom.
   The profile auto-closes (last point connects back to first point).

   Example: a cup with 4mm thick walls, 70mm outer diameter, 100mm tall:
   cup = make_solid_revolution([(35,0), (35,100), (31,100), (31,4)])
   # (35,0)→(35,100): outer wall bottom to top
   # (35,100)→(31,100): rim (4mm thick)
   # (31,100)→(31,4): inner wall top to bottom
   # (31,4)→(35,0): bottom floor (auto-closed)

   Example: a vase with curved profile:
   vase = make_solid_revolution([(25,0), (30,30), (35,80), (20,120), (16,120), (31,80), (26,30), (21,0)])
   # Outer: bottom→belly→widest→narrow neck
   # Inner: narrow neck→widest→belly→bottom (4mm thinner)

IMPORTANT for round objects:
- Do NOT use boolean difference to hollow out round objects. Use make_solid_revolution with a closed wall profile instead.
- Do NOT redefine make_frustum or make_solid_revolution. They are already injected — just call them like: mesh = make_solid_revolution(profile)
- Do NOT import them from any module. They are global functions.
- Do NOT wrap them in try/except. They will always work.
- The profile must trace the wall cross-section as a closed loop. Think of it as drawing the outline of the wall if you cut the object in half.

PREFERRED: For stands, brackets, holders — use extrude_polygon for clean single-piece geometry:
```python
from shapely.geometry import Polygon
# Example: Phone/tablet stand side profile (Y=depth front-to-back, Z=height)
# Shape: front lip + flat base + angled back support (like a backward "Z")
profile = Polygon([
    (0, 0),     # front bottom
    (85, 0),    # back bottom
    (85, 5),    # back, base top outer
    (80, 5),    # back plate outer bottom
    (65, 100),  # back plate outer top (angled ~75 deg from horizontal)
    (60, 100),  # back plate inner top
    (75, 5),    # back plate inner bottom
    (5, 5),     # base inner surface
    (5, 15),    # front lip inner top
    (0, 15),    # front lip outer top
])
mesh = trimesh.creation.extrude_polygon(profile, height=75)  # extrude 75mm wide
mesh.apply_translation(-mesh.centroid)
mesh.export(OUTPUT_PATH)
```

COMPOSITE PRIMITIVES approach — for complex shapes like shoes, chairs, animals:
Build the object from multiple simple primitives combined with trimesh.util.concatenate.
CRITICAL RULE for composite objects: All parts MUST physically touch or overlap. NO gaps between parts.
Use trimesh.creation.box/cylinder for parts. box() creates at origin center, cylinder() creates along Z axis centered at origin.
Position each part so it connects to adjacent parts by calculating exact coordinates.

```python
# Example: a simple chair (450mm wide, 400mm deep, 850mm tall)
import trimesh
import numpy as np

# Seat: 450x400x25mm, top surface at Z=450
seat = trimesh.creation.box(extents=[450, 400, 25])
seat.apply_translation([0, 0, 450 - 12.5])  # seat top at Z=450

# 4 Legs: cylinders r=25, from Z=0 to seat bottom (Z=425)
legs = []
leg_h = 425  # ground to seat bottom
# IMPORTANT: legs INSET from seat edges by leg radius so they don't protrude
for x, y in [(-190, -165), (190, -165), (-190, 165), (190, 165)]:
    leg = trimesh.creation.cylinder(radius=25, height=leg_h)
    leg.apply_translation([x, y, leg_h / 2])
    legs.append(leg)

# Backrest: 450x25x400mm, bottom touching seat top at BACK edge (Y=+165)
backrest = trimesh.creation.box(extents=[450, 25, 400])
backrest.apply_translation([0, 165 + 12.5, 450 + 200])  # Y at back edge

chair = trimesh.util.concatenate([seat] + legs + [backrest])
chair.apply_translation(-chair.centroid)
chair.export(OUTPUT_PATH)
```

```python
# Example: a simple shoe (280mm long, 100mm wide, 80mm tall)
from shapely.geometry import Polygon
import trimesh

sole_outline = Polygon([
    (0, 20), (20, 5), (60, 0), (140, 0), (220, 0), (260, 10), (280, 30),
    (280, 50), (270, 65), (260, 75), (220, 90), (140, 100), (60, 100),
    (20, 90), (0, 70)
])
sole = trimesh.creation.extrude_polygon(sole_outline, height=20)

upper_outline = Polygon([
    (10, 25), (30, 12), (70, 8), (140, 8), (200, 10), (230, 25),
    (230, 65), (200, 80), (140, 85), (70, 85), (30, 78), (10, 65)
])
upper = trimesh.creation.extrude_polygon(upper_outline, height=60)
upper.apply_translation([0, 0, 20])  # ON TOP of sole

shoe = trimesh.util.concatenate([sole, upper])
shoe.apply_translation(-shoe.centroid)
shoe.export(OUTPUT_PATH)
```

Common operations:
- mesh.apply_translation([x, y, z])
- R = trimesh.transformations.rotation_matrix(angle_rad, [ax_x, ax_y, ax_z]); mesh.apply_transform(R)
- combined = trimesh.util.concatenate([mesh1, mesh2])
- combined.export(OUTPUT_PATH)

IMPORTANT: Always end with `<your_final_mesh>.export(OUTPUT_PATH)`
"""


_LEGACY_ENRICH = r"""You are a 3D product designer for 3D printing. The user describes an object.
Expand it into a precise design specification in English.

CHOOSE THE RIGHT MODELING METHOD:
1. EXTRUDE PROFILE — for objects with a uniform cross-section along one axis (stands, brackets, bookends): describe as a 2D SIDE PROFILE (polygon) extruded to a width. List polygon vertices as (Y, Z) coordinates.
2. REVOLUTION PROFILE — for ANY round/cylindrical object (cups, vases, bottles, bowls, pen holders, jars, pots, tubes). KEY RULE: if the object is roughly circular when viewed from above, use REVOLUTION: describe as (radius, Z) points for make_solid_revolution.
3. COMPOSITE PRIMITIVES — for complex/organic shapes that are NOT symmetric and NOT a simple extrusion (shoes, animals, furniture, cars, tools): describe as a combination of simple primitives (boxes, cylinders, spheres) with specific positions, dimensions, and how they connect. Use trimesh.util.concatenate to combine parts. Do NOT use boolean operations.

IMPORTANT: A 2D side profile polygon must be a SIMPLE CLOSED LOOP — the vertices trace the OUTLINE of the shape going around ONCE. The path should NEVER cross itself or zigzag back and forth. Think of it as drawing the shape with a single continuous pencil stroke.

DESIGN KNOWLEDGE for common objects:
- PHONE STAND / TABLET STAND: Extrude profile. Backward "Z" shape: (1) flat BASE, (2) ANGLED back support at 65-75 deg, (3) front lip 12-15mm. Walls 4-5mm thick.
- BOOKEND: Extrude profile. L-shaped: flat base + vertical wall.
- VASE: Revolution profile. Wide bottom, belly at 1/3, narrow neck. Walls 3-5mm.
- CUP/MUG: Revolution profile + extruded handle.
- PEN HOLDER / 筆筒: Revolution profile (NOT extrude!). Cylinder, 80-110mm tall, r=30-40mm, walls 3mm.
- SHOE: Composite primitives. Sole = flat box (280x100x20mm, rounded front). Upper = half-cylinder or box on top of sole. Tongue = small box at front-top.
- CHAIR / TABLE: USE THIS EXACT CODE TEMPLATE (only adjust dimensions if user specifies):
```
import trimesh, numpy as np
seat = trimesh.creation.box(extents=[450, 400, 25])
seat.apply_translation([0, 0, 437.5])
legs = []
for x, y in [(-190, -165), (190, -165), (-190, 165), (190, 165)]:
    leg = trimesh.creation.cylinder(radius=25, height=425)
    leg.apply_translation([x, y, 212.5])
    legs.append(leg)
backrest = trimesh.creation.box(extents=[450, 25, 400])
backrest.apply_translation([0, 177.5, 650])
chair = trimesh.util.concatenate([seat] + legs + [backrest])
chair.apply_translation(-chair.centroid)
chair.export(OUTPUT_PATH)
```
KEY RULES: legs INSET from seat edge (not at corners). Backrest at BACK edge Y=+165.
- ANIMAL/FIGURINE: Composite primitives. Body = ellipsoid/box. Head = sphere. Legs = cylinders.

Output ONLY the design specification. No code. No markdown. Under 200 words.
For extrude/revolution methods, provide exact polygon vertices.
For composite primitives, list each part with its shape type, dimensions, and position.

Example 1:
User: "a bookend"
Output: Bookend made by extruding a 2D L-shaped side profile 120mm in the X direction. Side profile polygon vertices (Y=depth, Z=height in mm): (0,0), (80,0), (80,5), (5,5), (5,100), (0,100). This creates a 5mm thick base (80mm deep) connected to a 5mm thick vertical wall (100mm tall). Extrude 120mm wide. Center at origin.

Example 2:
User: "a desk stand for phone"
Output: Phone stand made by extruding a backward-Z side profile 80mm in the X direction. Side profile polygon vertices (Y=depth, Z=height in mm): (0,0), (90,0), (90,5), (82,5), (62,105), (57,105), (77,5), (5,5), (5,14), (0,14). The base is 90mm deep, 5mm thick. An angled back plate rises from Y=82 to Y=62 (leaning 20mm forward over 100mm height, ~78 degrees from horizontal). Front lip is 14mm tall. All walls 5mm thick. Center at origin.

Example 3:
User: "a vase"
Output: Vase made using revolution profile (make_solid_revolution). Closed wall cross-section profile (radius, Z in mm): outer surface from bottom to top: (25,0), (30,40), (35,90), (22,140), (18,160). Inner surface from top to bottom: (15,160), (19,140), (31,90), (26,40), (21,0). Wall thickness ~4mm. Total height 160mm, max diameter 70mm. The bottom closes automatically from (21,0) back to (25,0). Center at origin.

Example 4:
User: "a shoe"
Output: Shoe built using composite primitives. Part 1 (Sole): Shoe-shaped footprint polygon in XY plane, extruded 20mm in Z. Footprint outline (X,Y in mm): (0,20), (20,5), (60,0), (140,0), (220,0), (260,10), (280,30), (280,50), (270,65), (260,75), (220,90), (140,100), (60,100), (20,90), (0,70). Part 2 (Upper): Slightly smaller footprint polygon, extruded 60mm in Z, positioned at Z=20 on top of sole. Upper footprint inset 10mm from sole edge. Part 3 (Heel counter): Box 40x80x40mm at back of shoe at Z=20. Total dimensions: 280mm long, 100mm wide, 80mm tall. Center at origin.
"""


_LEGACY_REVIEW = r"""You are a 3D model code reviewer. You are given:
1. A design specification describing a 3D object
2. Python code that attempts to create it using trimesh

Review the code for FUNCTIONAL correctness:
- Are all parts from the specification present?
- Are dimensions approximately correct?
- Are parts properly positioned and connected (no floating/disconnected parts)?
- Would this object actually work for its intended purpose?
- Is the code using valid trimesh API?
- Does the polygon profile form the correct shape? (e.g., a phone stand needs a backward-Z profile with angled back plate, NOT a rectangular frame or box)

IMPORTANT: Do NOT nitpick rotation axis choices if the result looks structurally correct. Focus on whether the object would actually function and be RECOGNIZABLE as the intended object.

If the code is good enough to be functional, respond with exactly: LGTM
If there are real structural or functional problems, respond with a brief list of specific fixes (under 80 words). Do NOT output code. Do NOT suggest changes that would just swap one valid approach for another.
"""

SELF_REFINE_ROUNDS = 2  # Number of auto-review rounds


# ---------------------------------------------------------------------------
# Web search for object design references
# ---------------------------------------------------------------------------
async def translate_to_english(text: str) -> str:
    """Quick heuristic: if text is mostly non-ASCII, it's likely Chinese/etc. Use LLM to translate."""
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii < len(text) * 0.3:
        return text  # Already mostly English
    # Use a lightweight LLM call to translate
    try:
        result = await call_ollama([
            {"role": "system", "content": "Translate the following to English. Output ONLY the translation, nothing else."},
            {"role": "user", "content": text},
        ])
        translated = result.strip().strip('"').strip("'")
        if translated:
            log.info(f"Translated '{text}' -> '{translated}'")
            return translated
    except Exception as e:
        log.warning(f"Translation failed: {e}")
    return text


VISION_MODEL = "qwen2.5vl:7b"


async def analyze_image_with_vision(image_b64: str, object_name: str) -> str:
    """Use vision model to describe the structural features of an object from an image."""
    try:
        prompt = (
            f"This is an image of a {object_name}. "
            "Describe ONLY the basic geometric shape for simple 3D printing: "
            "What are the 2-4 main structural parts? What primitive shapes are they (box, cylinder, cone)? "
            "IGNORE all surface details, patterns, textures, holes, decorations, color and material. "
            "Focus ONLY on overall silhouette. Reply in under 80 words."
        )
        payload = {
            "model": VISION_MODEL,
            "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
            "stream": False,
            "options": {"num_predict": 200, "temperature": 0.3},
            "keep_alive": "5m",
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("message", {}).get("content", "").strip()
                if content:
                    log.info(f"Vision analysis: {content[:150]}")
                    return content
                err = data.get("error", "")
                if err:
                    log.warning(f"Vision model error: {err[:100]}")
        return ""
    except Exception as e:
        log.warning(f"Vision analysis failed: {e}")
        return ""


async def search_and_download_image(query: str) -> str | None:
    """Search for an image and return it as base64."""
    try:
        from ddgs import DDGS
        ddgs = DDGS()
        hits = list(ddgs.images(query, max_results=3))
        for h in hits:
            url = h.get("image", "")
            if not url:
                continue
            try:
                async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                    resp = await client.get(url)
                    if resp.status_code == 200 and len(resp.content) > 1000:
                        import base64
                        b64 = base64.b64encode(resp.content).decode()
                        log.info(f"Downloaded image ({len(resp.content)} bytes) from: {url[:80]}")
                        return b64
            except Exception as e:
                log.warning(f"Image download failed: {url[:60]} -> {e}")
                continue
        return None
    except Exception as e:
        log.warning(f"Image search failed: {e}")
        return None


async def search_object_references(object_desc: str) -> str:
    """Search the web for structural/design info about the object, including image analysis."""
    try:
        from ddgs import DDGS

        # Translate to English for better search results
        en_desc = await translate_to_english(object_desc)

        # Run text search and image search concurrently
        text_results = []
        image_desc = ""

        # Text search
        ddgs = DDGS()
        queries = [
            f"{en_desc} structure shape dimensions design",
            f"{en_desc} 3D model typical measurements mm",
        ]
        for q in queries:
            try:
                hits = list(ddgs.text(q, max_results=3))
                for h in hits:
                    title = h.get("title", "")
                    body = h.get("body", "")
                    if body:
                        text_results.append(f"- {title}: {body}")
            except Exception as e:
                log.warning(f"Search query failed: {q} -> {e}")

        # Image search + vision analysis
        try:
            img_b64 = await search_and_download_image(f"{en_desc} product photo")
            if img_b64:
                image_desc = await analyze_image_with_vision(img_b64, en_desc)
        except Exception as e:
            log.warning(f"Image analysis pipeline failed: {e}")

        # Combine results
        parts = []
        if image_desc:
            parts.append(f"[Visual analysis of reference image]: {image_desc}")
        if text_results:
            parts.extend(text_results[:4])

        if parts:
            combined = "\n".join(parts)
            log.info(f"Search found {len(text_results)} text + {'1 image' if image_desc else '0 images'} for: {en_desc}")
            return combined
        log.warning(f"Web search returned no results for: {en_desc}")
        return ""
    except ImportError:
        log.warning("ddgs not installed, skipping web search")
        return ""
    except Exception as e:
        log.warning(f"Web search failed: {e}")
        return ""


class GenerateRequest(BaseModel):
    prompt: str
    model: str | None = None


class GenerateResponse(BaseModel):
    id: str
    code: str
    stl_url: str
    enriched_prompt: str = ""
    search_info: str = ""
    thumbnails: list[str] = []  # URLs to rendered view PNGs
    judge: dict | None = None    # VLM judge result, if enabled
    attempts: int = 1
    # Sprint 5-7 additions
    cache_hit: bool = False                  # S5.1
    formats: dict = {}                       # S5.2 — {format: download_url}
    geom_check: dict | None = None           # S6.1 — programmatic gate result
    print_warnings: list[dict] = []          # S6.2 — print-readiness chips
    slicer: dict | None = None               # S6.3 — slicer probe result
    best_of_n_count: int = 1                 # S7.1 — N candidates run


# HTTP statuses that trigger failover to the next model in the chain
_LLM_FAILOVER_STATUSES = {429, 500, 502, 503, 504}

# S4.7: counter so /api/stats can answer 'how often is the primary
# failing?' without grepping the log.
_failover_stats: dict = {
    "total_calls": 0,   # every call to call_cloud_llm
    "failovers":   0,   # completed by a non-primary model
    "errors":      0,   # all chain members failed
    "last_reset_ts": int(time.time()),
}


async def _call_one_cloud(client: httpx.AsyncClient, messages: list[dict],
                          model: str) -> tuple[int, str]:
    """One-shot POST to a cloud LLM. Returns (status, content-or-error-body)."""
    api_base, api_key = _resolve_provider(model)
    if not api_key:
        return 500, f"no API key for {model!r}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.3,
    }
    provider_id = MODEL_PROVIDER.get(model, "(default)")
    log.info(f"Calling cloud LLM: {model} via {api_base} [provider={provider_id}]")
    try:
        resp = await client.post(f"{api_base}/chat/completions",
                                 headers=headers, json=payload)
    except httpx.HTTPError as e:
        return -1, str(e)
    if resp.status_code != 200:
        return resp.status_code, f"Cloud API error ({provider_id}): {resp.text[:500]}"
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    usage = data.get("usage", {})
    log.info(f"Cloud LLM response ({provider_id}): {len(content)} chars, tokens: {usage}")
    # S3.4: record token usage per call for the cost monitor
    try:
        _record_llm_usage(model, usage)
    except Exception as e:
        log.debug(f"Token accounting skipped: {e}")
    return 200, content


async def call_cloud_llm(messages: list[dict], model: str) -> str:
    """Call cloud LLM API with automatic failover (S1.6).

    If the primary model returns a transient error (429/5xx), falls through
    to each entry in MODEL_FAILOVER[model] in order. Auth / bad-request
    errors are NOT retried (won't succeed on other providers either).
    """
    chain: list[str] = [model] + list(MODEL_FAILOVER.get(model, []))
    last_status = 0
    last_body = ""
    _failover_stats["total_calls"] += 1
    async with httpx.AsyncClient(timeout=300) as client:
        for idx, mdl in enumerate(chain):
            status, body = await _call_one_cloud(client, messages, mdl)
            if status == 200:
                if idx > 0:
                    _failover_stats["failovers"] += 1
                    log.warning(f"LLM failover: {chain[0]} → {mdl} succeeded "
                                f"(primary last status={last_status})")
                return body
            last_status, last_body = status, body
            # Don't waste the failover budget on auth/bad-request from primary
            if status not in _LLM_FAILOVER_STATUSES and status != -1:
                break
            if idx < len(chain) - 1:
                log.warning(f"LLM {mdl} failed ({status}); failing over to "
                            f"{chain[idx+1]}")
    _failover_stats["errors"] += 1
    raise HTTPException(status_code=502,
                        detail=f"All LLM providers failed (last={last_status}): "
                               f"{last_body[:400]}")


def is_cloud_model(model: str) -> bool:
    """Check if model name refers to a cloud model."""
    return model in CLOUD_MODELS


async def call_ollama(messages: list[dict], model: str | None = None) -> str:
    """Call LLM API. Routes to cloud or Ollama based on model name."""
    model = model or OLLAMA_MODEL
    if is_cloud_model(model):
        return await call_cloud_llm(messages, model)
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 4096, "num_ctx": 8192},
        "think": False,  # Disable thinking mode for Qwen3
        "keep_alive": "10m",  # Keep model in DGX memory for 10 min (avoids slow reload)
    }
    async with httpx.AsyncClient(timeout=600) as client:
        resp = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Ollama error: {resp.text}")
        data = resp.json()
        msg = data.get("message", {})
        content = msg.get("content", "").strip()
        thinking = msg.get("thinking", "").strip()
        # Qwen3 thinking mode: code may be in 'content' or 'thinking' field
        if content:
            return content
        if thinking:
            log.info("LLM response was in 'thinking' field, extracting code from it")
            return thinking
        return ""


def clean_code(raw: str) -> str:
    """Extract Python code from LLM output (handles thinking tags, markdown, mixed text)."""
    code = raw.strip()

    # If </think> tag present, take everything after it
    if "</think>" in code:
        code = code.split("</think>")[-1].strip()
    # Strip remaining <think> blocks
    code = re.sub(r"<think>.*?</think>", "", code, flags=re.DOTALL).strip()
    if "<think>" in code:
        code = code.split("<think>")[0].strip()

    # If the response starts with import/from, it's already clean code
    if re.match(r"^(import |from )", code):
        pass  # already clean, fall through to post-processing
    # Extract code from markdown fences (prefer the LAST python block)
    elif "```" in code:
        matches = re.findall(r"```(?:python)?\s*\n(.*?)```", code, re.DOTALL)
        if matches:
            code = matches[-1].strip()
    else:
        match = re.search(r"^(import |from )", code, re.MULTILINE)
        if match:
            code = code[match.start():]

    code = code.strip()

    # Remove re-definitions of pre-injected helper functions
    _func_def_pattern = r'^def {}\(.*\):[ \t]*\n(?:(?:[ \t]+.*|[ \t]*)\n)*'
    for fname in ('make_frustum', 'make_solid_revolution'):
        code = re.sub(_func_def_pattern.format(fname), '', code, flags=re.MULTILINE)

    # --- FIX #3: Remove wrong imports of pre-injected helper functions ---
    code = re.sub(r'^from\s+\S+\s+import\s+.*\b(make_frustum|make_solid_revolution)\b.*$',
                  '# (removed: helpers are pre-injected)', code, flags=re.MULTILINE)

    # --- FIX #2: Auto-add missing imports ---
    has_trimesh_import = bool(re.search(r'^import\s+trimesh|^from\s+trimesh', code, re.MULTILINE))
    has_numpy_import = bool(re.search(r'^import\s+numpy|^from\s+numpy', code, re.MULTILINE))
    uses_trimesh = 'trimesh.' in code
    uses_numpy = bool(re.search(r'\bnp\.|numpy\.', code))
    inject_lines = []
    if uses_trimesh and not has_trimesh_import:
        inject_lines.append('import trimesh')
        log.warning("Auto-injected: import trimesh")
    if uses_numpy and not has_numpy_import:
        inject_lines.append('import numpy as np')
        log.warning("Auto-injected: import numpy as np")
    if inject_lines:
        code = '\n'.join(inject_lines) + '\n' + code

    # --- FIX #1: Replace known non-existent trimesh API calls ---
    _api_fixes = {
        'trimesh.creation.frustum':          'make_frustum',
        'trimesh.creation.conical_frustum':  'make_frustum',
        'trimesh.creation.revolve':          'make_solid_revolution',
        'trimesh.creation.sphere':           'trimesh.creation.icosphere',
        'trimesh.creation.rounded_box':      'trimesh.creation.box',
    }
    for bad_api, good_api in _api_fixes.items():
        if bad_api in code:
            log.warning(f"Auto-fix: {bad_api} -> {good_api}")
            code = code.replace(bad_api, good_api)

    # Fix mesh.center() -> mesh.apply_translation(-mesh.centroid)
    code = re.sub(r'(\w+)\.center\(\)', r'\1.apply_translation(-\1.centroid)', code)

    # --- FIX #8: Syntax check with compile() ---
    try:
        compile(code, '<generated>', 'exec')
    except SyntaxError as e:
        log.warning(f"SyntaxError in generated code (line {e.lineno}): {e.msg}")
        lines = code.split('\n')
        if e.lineno and e.lineno <= len(lines):
            fixed_lines = lines[:e.lineno - 1] + lines[e.lineno:]
            fixed_code = '\n'.join(fixed_lines)
            try:
                compile(fixed_code, '<generated>', 'exec')
                log.info(f"Auto-fixed SyntaxError by removing line {e.lineno}")
                code = fixed_code
            except SyntaxError:
                pass

    # Remove blank lines at start
    code = re.sub(r'^\s*\n', '', code)

    return code


def execute_code(code: str, job_id: str) -> Path:
    """Execute generated Python code to create STL file.

    Dispatches to the active backend (trimesh / cadquery). The backend is
    responsible for:
    - injecting backend-specific helpers (make_solid_revolution, export_stl, ...)
    - running the code
    - producing a valid STL at the expected path
    """
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    stl_path = job_dir / "model.stl"
    code_path = job_dir / "code.py"
    code_path.write_text(code, encoding="utf-8")

    try:
        # S5.2: ask backend to also write STEP/3MF/GLB if multi-format flag on.
        # Falls back gracefully if the backend signature is older.
        try:
            _BACKEND.execute_and_export(
                code, stl_path,
                extra_formats=FEATURE_MULTI_FORMAT_EXPORT,
            )
        except TypeError:
            _BACKEND.execute_and_export(code, stl_path)
    except BackendError as e:
        err_msg = str(e)
        hint = ""
        if "not all meshes are volumes" in err_msg.lower():
            hint = "\nHINT: Boolean ops require watertight meshes. Use trimesh.util.concatenate instead."
        elif "manifold" in err_msg.lower():
            hint = "\nHINT: Boolean engine error. Use trimesh.util.concatenate instead."
        raise HTTPException(status_code=500, detail=f"Code execution failed ({BACKEND}):\n{err_msg}{hint}")

    if not stl_path.exists():
        raise HTTPException(
            status_code=500,
            detail="STL file was not created. Ensure your code calls export_stl(result, OUTPUT_PATH) or <mesh>.export(OUTPUT_PATH).",
        )

    # S4.1: try to make the STL watertight/printable. Silent no-op if
    # already watertight or if mesh_repair / pymeshfix can't help.
    if MESH_REPAIR_ENABLED:
        try:
            from mesh_repair import repair_stl
            r = repair_stl(stl_path)
            if r.loaded and (r.method != "none" or not r.before_watertight):
                log.info(f"[{job_id}] mesh_repair: {r.summary()}")
        except Exception as e:
            log.debug(f"[{job_id}] mesh_repair skipped: {e}")

    return stl_path


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
MAX_RETRIES = 2


def _validate_for_backend(code: str):
    """Run AST validation using the active backend's allowlist."""
    if not AST_VALIDATE:
        return None  # disabled
    if BACKEND == "cadquery":
        return validate_cadquery(code)
    return validate_trimesh(code)


def _cadquery_fix_hint(err: str) -> str:
    """Map common CadQuery BRep / fillet errors to concrete repair advice.

    The LLM sees the raw error, but without these hints it tends to
    flail (add more points, add more fillets). These hints nudge it
    toward the simpler fix.
    """
    e = str(err).lower()
    if "brep_api: command not done" in e or "brepoffsetapi_makepipeshell" in e:
        return (
            "\nIMPORTANT: This is almost always a DEGENERATE profile or path. "
            "Try: (1) use FEWER points in the profile (5-6 is enough for a "
            "mug/cup/bowl — do NOT repeat (0,0) at both ends), (2) ensure "
            "the polygon is SIMPLE (no self-intersections — the outer and "
            "inner walls must not cross), (3) for sweep paths, use ONLY a "
            "threePointArc OR a single polyline — do NOT mix lineTo and arc "
            "in the same path. Keep it minimal."
        )
    if "no suitable edges" in e or "fillets requires" in e or "chamfer" in e:
        return (
            "\nIMPORTANT: The fillet/chamfer call failed. Two likely causes: "
            "(a) you used an INVENTED selector string like `\"%Circle\"`, "
            "`\"%Plane Z=...\"`, `\"%Line\"`, or `.edges()` with no argument "
            "— these DO NOT EXIST. Only these selector strings are valid: "
            "`\">Z\"`, `\"<Z\"`, `\">X\"`, `\"<X\"`, `\">Y\"`, `\"<Y\"`, "
            "`\"|Z\"`, `\"|X\"`, `\"|Y\"`. "
            "(b) you filleted AFTER `.union(...)` — the union body's edges "
            "often cannot be re-selected. DELETE every fillet/chamfer call "
            "that appears after a `.union(...)`. For a mug, the CORRECT "
            "structure is exactly: one `body = body.edges(\">Z\").fillet(2)` "
            "BEFORE the handle is created, then `result = body.union(handle)` "
            "with NOTHING after that except `export_stl(...)`."
        )
    # DeepSeek-style: invented type selectors even when no error msg names them
    if "%circle" in e or "%plane" in e or "%line" in e:
        return (
            "\nIMPORTANT: You used a `\"%...\"` type selector that does not "
            "exist. Remove that line entirely. Valid selectors are only: "
            "`\">Z\"`, `\"<Z\"`, `\">X\"`, `\"<X\"`, `\">Y\"`, `\"<Y\"`, "
            "`\"|Z\"`, `\"|X\"`, `\"|Y\"`."
        )
    if "ncollection_sequence" in e or "bnd_box" in e:
        return (
            "\nIMPORTANT: Geometry collection failed — one of the shapes "
            "is empty or malformed. Simplify the profile drastically: use "
            "straightforward primitives (cylinder + hole, or a 6-point "
            "revolve profile)."
        )
    if "loft" in e or "makelolift" in e or "brepfill" in e:
        return (
            "\nIMPORTANT: Loft failed. Check (a) both outlines have AT LEAST "
            "3 points, (b) outlines do NOT self-intersect, (c) both outlines "
            "wind in the SAME direction (both CW or both CCW), (d) the "
            "second workplane offset is > 0 (use `.workplane(offset=N)` "
            "with N>=20). If it keeps failing, use two outlines with "
            "fewer points (6-8) instead of 15+."
        )
    if "could not find valid" in e and "plane" in e:
        return (
            "\nIMPORTANT: A workplane reference was broken. Build each "
            "part on a fresh `cq.Workplane(\"XY\")`, \"XZ\", or \"YZ\" — "
            "don't chain `.workplane()` off a tagged face."
        )
    return ""


# Keywords that indicate the user asked for a container-with-handle.
_MUG_KEYWORDS = (
    "mug", "cup", "coffee", "tea", "teapot", "pitcher",
    "馬克杯", "杯子", "咖啡杯", "茶杯", "水杯", "把手",
)

# Keywords that indicate the user asked for a shoe/footwear/organic shape
_SHOE_KEYWORDS = (
    "shoe", "sneaker", "boot", "sandal", "slipper",
    "鞋", "鞋子", "靴子", "拖鞋",
)

# Keywords for composite furniture
_FURNITURE_KEYWORDS = (
    "chair", "table", "stool", "bench", "desk",
    "椅子", "桌子", "凳子", "板凳", "書桌",
)


def _is_mug_like_prompt(prompt: str) -> bool:
    p = (prompt or "").lower()
    return any(k.lower() in p for k in _MUG_KEYWORDS)


def _is_shoe_like_prompt(prompt: str) -> bool:
    p = (prompt or "").lower()
    return any(k.lower() in p for k in _SHOE_KEYWORDS)


def _is_furniture_like_prompt(prompt: str) -> bool:
    p = (prompt or "").lower()
    return any(k.lower() in p for k in _FURNITURE_KEYWORDS)


# Minimal, known-good mug template. Used as a last-resort hint on the final
# retry when the model keeps failing on a mug-shaped prompt.
_SAFE_MUG_TEMPLATE = """import cadquery as cq

body_profile = [
    (0,   0),
    (40,  0),
    (40, 100),
    (32, 100),
    (32,   8),
    (0,    8),
]
body = cq.Workplane("XZ").polyline(body_profile).close().revolve()

handle_path = (cq.Workplane("XZ")
               .moveTo(40, 25)
               .threePointArc((65, 50), (40, 75)))
handle_profile = cq.Workplane("YZ").circle(5)
handle = handle_profile.sweep(handle_path)

result = body.union(handle)
export_stl(result, OUTPUT_PATH)
"""

# Minimal, known-good shoe template (asymmetric loft — NOT a cylinder).
_SAFE_SHOE_TEMPLATE = """import cadquery as cq

# Sole outline at Z=0: asymmetric foot footprint (X=toe->heel, Y=inside->outside).
sole_pts = [(0,30),(15,10),(50,0),(120,0),(200,0),(250,10),(275,30),
            (280,55),(270,80),(240,95),(180,100),(100,100),(40,95),(10,80),(0,55)]
# Upper outline at Z=70: smaller inset oval.
upper_pts = [(30,40),(60,25),(110,20),(170,20),(220,25),(250,40),
             (250,65),(220,80),(170,85),(110,85),(60,80),(30,65)]
result = (cq.Workplane("XY")
          .polyline(sole_pts).close()
          .workplane(offset=70)
          .polyline(upper_pts).close()
          .loft(combine=True))
export_stl(result, OUTPUT_PATH)
"""

# Minimal, known-good chair template (composite of primitives + union).
_SAFE_CHAIR_TEMPLATE = """import cadquery as cq

# Seat: flat slab at Z = 250 (seat-top 25mm thick)
seat = cq.Workplane("XY").workplane(offset=250).box(450, 450, 25, centered=(True, True, False))

# Four legs: 25x25 square cross-section, 250 tall, inset 30mm from each corner
leg_h = 250
legs = None
for (x, y) in [(-195, -195), (195, -195), (-195, 195), (195, 195)]:
    leg = (cq.Workplane("XY")
           .center(x, y)
           .box(25, 25, leg_h, centered=(True, True, False)))
    legs = leg if legs is None else legs.union(leg)

# Backrest: thin vertical slab rising above the back edge
back = (cq.Workplane("XY")
        .workplane(offset=275)
        .center(0, 210)
        .box(450, 25, 400, centered=(True, True, False)))

result = seat.union(legs).union(back)
export_stl(result, OUTPUT_PATH)
"""


def _final_retry_fallback(prompt: str) -> str:
    """Extra text appended on the LAST retry when we keep failing.

    Provides an explicit minimal template per shape family, rather than
    letting the model keep exploring.
    """
    if _is_mug_like_prompt(prompt):
        return (
            "\n\nFINAL ATTEMPT — you have failed multiple times. STOP "
            "experimenting. Output EXACTLY this minimal template, "
            "adjusting only the body dimensions (40 / 100 / 32 / 8) and "
            "handle position/size to match the user's description. Do "
            "NOT add fillets, chamfers, extra points, or extra curves:\n\n"
            f"```python\n{_SAFE_MUG_TEMPLATE}```"
        )
    if _is_shoe_like_prompt(prompt):
        return (
            "\n\nFINAL ATTEMPT — you have failed multiple times. STOP "
            "experimenting. Output EXACTLY this minimal loft template. "
            "The two outlines MUST be ASYMMETRIC (long in X, short in Y) "
            "and DIFFERENT from each other. Do NOT substitute circles or "
            "squares — loft of two circles is a can, not a shoe:\n\n"
            f"```python\n{_SAFE_SHOE_TEMPLATE}```"
        )
    if _is_furniture_like_prompt(prompt):
        return (
            "\n\nFINAL ATTEMPT — you have failed multiple times. STOP "
            "experimenting. Build from primitives (boxes) + union, like "
            "this chair template. Adjust dimensions/counts for the "
            "user's request (e.g. remove backrest for a stool):\n\n"
            f"```python\n{_SAFE_CHAIR_TEMPLATE}```"
        )
    return ""


def _trimesh_fix_hint(err: str) -> str:
    """Error-specific hints for the trimesh backend."""
    e = str(err).lower()
    if "revolve" in e:
        return ("\nIMPORTANT: Use make_solid_revolution(profile) instead of "
                "trimesh.creation.revolve().")
    if "not all meshes are volumes" in e:
        return ("\nIMPORTANT: Use trimesh.util.concatenate instead of boolean "
                "ops on non-watertight meshes.")
    return ""


# S4.2: pre-judge watertight gate helpers.
def _check_watertight(stl_path: Path) -> bool:
    """Load STL and return whether it is watertight (printable)."""
    try:
        import trimesh
        m = trimesh.load_mesh(str(stl_path))
        # Scene (multi-geom) — flatten
        if not hasattr(m, "is_watertight") and hasattr(m, "geometry"):
            try:
                m = trimesh.util.concatenate(tuple(m.geometry.values()))
            except Exception:
                return False
        return bool(getattr(m, "is_watertight", False))
    except Exception:
        return False


def _watertight_retry_hint(prompt: str) -> str:
    """Directive telling the LLM to regenerate for a printable mesh."""
    return (
        "\n\nThe produced STL is NOT watertight (it has holes or "
        "non-manifold edges) so it cannot be 3D printed. Regenerate "
        "the code following THESE rules (in priority order):\n"
        "1. PREFER closed primitives (`cq.Workplane.box` / `.cylinder` / "
        "`.sphere`) joined with `.union(...)`. Union of closed solids "
        "is always watertight.\n"
        "2. If you must subtract (`.cut(...)`), make the cutting tool "
        "EXTEND BEYOND the body on both ends (e.g. cylinder depth = "
        "body_depth + 4) so there are no co-planar faces.\n"
        "3. For lofts, both outlines MUST be closed loops. Do not "
        "leave the profile 'open'.\n"
        "4. Do NOT use `.shell(...)` for container hollowing unless "
        "the wall thickness is at least 1.5mm and the rim is flat.\n"
        "5. Keep the design SIMPLE. If the current approach keeps "
        "producing non-watertight output, switch to a simpler primitive "
        "composition even if it loses some detail."
    )


async def _render_and_judge(
    job_id: str,
    stl_path: Path,
    user_description: str,
) -> tuple[list[str], JudgeResult | None]:
    """Render STL → VLM judge. Returns (thumbnail_urls, judge_result).

    Returns ([], None) when judging is disabled or fails non-fatally.
    """
    if not JUDGE_ENABLED:
        return [], None
    thumb_dir = OUTPUT_DIR / job_id / "views"
    try:
        pngs = await asyncio.to_thread(render_stl_views, stl_path, thumb_dir, (384, 384))
    except Exception as e:
        log.warning(f"[{job_id}] Rendering failed, skipping judge: {e}")
        return [], None
    thumbnail_urls = [f"/api/thumbnail/{job_id}/{p.name}" for p in pngs]

    # S1.2 — build cross-provider vision spec chain.
    # Priority: explicit `cloud_vision_models` list → legacy single model
    # → per-provider siblings of the primary (Gemini-flash family).
    chain: list[str] = []
    for m in CLOUD_VISION_MODELS:
        if m and m not in chain:
            chain.append(m)
    if CLOUD_VISION_MODEL and CLOUD_VISION_MODEL not in chain:
        chain.append(CLOUD_VISION_MODEL)
    # Same-provider siblings (e.g. other flash variants) as last-resort
    if CLOUD_VISION_MODEL:
        primary_pid = MODEL_PROVIDER.get(CLOUD_VISION_MODEL)
        for m in MODEL_PROVIDER:
            if (MODEL_PROVIDER.get(m) == primary_pid
                    and m != CLOUD_VISION_MODEL
                    and "flash" in m.lower()
                    and m not in chain):
                chain.append(m)

    if not chain:
        log.warning(f"[{job_id}] No vision model configured; skipping judge")
        return thumbnail_urls, None

    vision_specs: list[dict] = []
    for m in chain:
        base, key = _resolve_provider(m)
        if not key:
            continue
        vision_specs.append({"model": m, "api_base": base, "api_key": key})
    if not vision_specs:
        log.warning(f"[{job_id}] No API key for any vision model in {chain!r}; skipping judge")
        return thumbnail_urls, None

    log.info(f"[{job_id}] Vision judge chain: {[s['model'] for s in vision_specs]}")
    try:
        judge = await judge_model(
            user_description=user_description,
            view_paths=pngs,
            vision_specs=vision_specs,
        )
    except Exception as e:
        log.warning(f"[{job_id}] Judge call failed: {e}")
        return thumbnail_urls, None

    log.info(f"[{job_id}] Judge: score={judge.match_score}/10, category={judge.category!r}")
    return thumbnail_urls, judge


def _route_model_for_prompt(prompt: str, user_model: str | None) -> str | None:
    """P3: if shape routing is enabled and the prompt's inferred category
    has a configured override, return that model. Else return user_model.

    Applied before /api/generate starts so the entire call chain uses the
    routed model. Only applies to cloud models — if the user picked a
    local Ollama model, we honor that (they presumably want local).
    """
    if not SHAPE_ROUTING_ENABLED or not SHAPE_ROUTING:
        return user_model
    # Only override if the user is already on a cloud model (don't silently
    # push them off Ollama).
    if user_model and not is_cloud_model(user_model):
        return user_model
    try:
        from pattern_cache import infer_category
        cat = infer_category(prompt)
    except Exception:
        return user_model
    override = SHAPE_ROUTING.get(cat)
    if override and override != user_model:
        log.info(
            f"shape_routing: '{cat}' → {override} "
            f"(user picked: {user_model or 'default'})"
        )
        return override
    return user_model


def _collect_format_urls(job_id: str) -> dict:
    """Build {fmt: /api/download/<job>/<fmt>} for any extra formats present."""
    job_dir = OUTPUT_DIR / job_id
    out: dict[str, str] = {}
    for fmt in ("step", "3mf", "glb"):
        if (job_dir / f"model.{fmt}").exists():
            out[fmt] = f"/api/download/{job_id}?fmt={fmt}"
    return out


def _build_cache_hit_response(req_prompt: str, hit: dict, job_id_log: str
                               ) -> "GenerateResponse":
    """Build a GenerateResponse from a cache hit pointing at hit['job_id']."""
    cached_job = hit["job_id"]
    cached_dir = OUTPUT_DIR / cached_job
    try:
        cached_code = (cached_dir / "code.py").read_text("utf-8")
    except Exception:
        cached_code = ""
    try:
        cached_enrich = (cached_dir / "enriched_prompt.txt").read_text("utf-8")
    except Exception:
        cached_enrich = ""
    thumbs = sorted([
        f"/api/thumbnail/{cached_job}/{p.name}"
        for p in cached_dir.glob("view_*.png")
    ])
    log.info(
        f"[{job_id_log}] OUTPUT CACHE HIT → {cached_job} "
        f"(score={hit.get('judge_score')}, age="
        f"{int(time.time())-int(hit.get('created_ts', 0))}s)"
    )
    return GenerateResponse(
        id=cached_job,
        code=cached_code,
        stl_url=f"/api/download/{cached_job}",
        enriched_prompt=cached_enrich,
        search_info="",
        thumbnails=thumbs,
        judge={
            "category": "cache_hit",
            "match_score": hit.get("judge_score"),
            "geometry_issues": [],
            "fix_suggestion": "",
        },
        attempts=0,
        cache_hit=True,
        formats=_collect_format_urls(cached_job),
        print_warnings=[],
    )


def _run_geom_check(stl_path: Path, prompt: str) -> "object | None":
    """Run programmatic geometric check; return GeomCheckResult or None."""
    if not FEATURE_GEOM_CHECK:
        return None
    try:
        from pattern_cache import infer_category
        from judge_geometric import check as geom_check
        cat = infer_category(prompt)
        return geom_check(stl_path, cat)
    except Exception as e:
        log.debug(f"geom_check skipped: {e}")
        return None


def _run_print_readiness(stl_path: Path) -> list[dict]:
    if not FEATURE_PRINT_READINESS:
        return []
    try:
        from print_readiness import analyse
        return analyse(stl_path)
    except Exception as e:
        log.debug(f"print_readiness skipped: {e}")
        return []


def _run_sandbox_shadow(code: str, job_id: str) -> dict | None:
    """S7.3 shadow-mode: try compiling `code` under RestrictedPython, log
    the verdict but DO NOT execute. Compile-only path is cheap (~ms) and
    answers the key question for shadow rollout: 'would the existing AST
    allowlist still be passing things our strict path would reject?'.

    A future tightening can promote this to actual exec-shadow once the
    compile-shadow data shows < 2pp false-positive rate.
    """
    if not FEATURE_SANDBOX_STRICT or not code or not code.strip():
        return None
    try:
        from sandbox_strict import compile_strict, is_available
        if not is_available():
            return {"available": False}
        code_obj, errors = compile_strict(code, filename=f"<{job_id}>")
        return {
            "available": True,
            "compile_ok": code_obj is not None,
            "errors_n": len(errors),
            "errors_first": [str(e)[:200] for e in errors[:3]],
        }
    except Exception as e:
        log.debug(f"[{job_id}] sandbox shadow skipped: {e}")
        return None


async def _run_best_of_n_shadow(req: "GenerateRequest", category: str,
                                n_extra: int, parent_job_id: str,
                                baseline_score: int | None) -> None:
    """S7.1 shadow-mode: spawn `n_extra` extra independent generations in
    background, log each candidate's score to structured_log, then discard
    the result. Used to gather cost-vs-score data for best_of_n rollout
    decision. Does NOT change the user-facing response."""
    log.info(f"[{parent_job_id}] best_of_n shadow: spawning {n_extra} extra "
             f"candidate(s) for category={category}")
    candidates_scores: list = []
    for i in range(n_extra):
        try:
            tok = _BEST_OF_N_SHADOW_CTX.set(True)  # break recursion
            try:
                from copy import deepcopy
                shadow_resp = await generate(deepcopy(req), no_cache=True)
            finally:
                _BEST_OF_N_SHADOW_CTX.reset(tok)
            score = None
            if shadow_resp is not None and shadow_resp.judge:
                score = shadow_resp.judge.get("match_score")
            candidates_scores.append(score)
            STRUCTURED_LOG.emit(
                "best_of_n_shadow_candidate",
                parent_job_id=parent_job_id, shadow_idx=i,
                shadow_job_id=getattr(shadow_resp, "id", None),
                category=category, shadow_score=score,
                baseline_score=baseline_score,
            )
        except Exception as e:
            log.warning(f"[{parent_job_id}] shadow candidate {i} error: {e}")
            STRUCTURED_LOG.emit(
                "best_of_n_shadow_error",
                parent_job_id=parent_job_id, shadow_idx=i,
                err=str(e)[:200],
            )
            candidates_scores.append(None)
    valid = [s for s in candidates_scores if isinstance(s, int)]
    best_shadow = max(valid) if valid else None
    improvement = None
    if isinstance(best_shadow, int) and isinstance(baseline_score, int):
        improvement = best_shadow - baseline_score
    STRUCTURED_LOG.emit(
        "best_of_n_shadow_round_done",
        parent_job_id=parent_job_id, category=category, n_extra=n_extra,
        baseline_score=baseline_score,
        shadow_scores=candidates_scores,
        best_shadow_score=best_shadow,
        improvement=improvement,
    )


def _maybe_spawn_best_of_n_shadow(req: "GenerateRequest",
                                  response: "GenerateResponse | None",
                                  job_id: str) -> None:
    """Decide whether to fire the shadow round. No-op if flag off, if
    we're already inside a shadow run, or if no judge baseline exists."""
    if not FEATURE_BEST_OF_N:
        return
    if _BEST_OF_N_SHADOW_CTX.get():
        return
    if response is None or response.judge is None:
        return
    try:
        from pattern_cache import infer_category
        cat = infer_category(req.prompt)
    except Exception:
        return
    n = int(BEST_OF_PER_CATEGORY.get(cat, 1))
    if n < 2:
        return
    n_extra = n - 1
    baseline_score = response.judge.get("match_score") if response.judge else None
    try:
        asyncio.create_task(
            _run_best_of_n_shadow(req, cat, n_extra, job_id, baseline_score)
        )
    except Exception as e:
        log.debug(f"[{job_id}] best_of_n shadow spawn skipped: {e}")


def _finalize_generate(req: "GenerateRequest", job_id: str,
                       response: "GenerateResponse | None",
                       code: str, sys_prompt: str, start_ts: float,
                       *, exec_ok: bool, store_cache: bool) -> None:
    """Run output_cache.store + structured_log.emit at end of generate."""
    if (store_cache and FEATURE_OUTPUT_CACHE and response is not None
            and not response.cache_hit):
        try:
            judge_score = None
            if response.judge:
                judge_score = response.judge.get("match_score")
            OUTPUT_CACHE.store(req.prompt,
                               req.model or OLLAMA_MODEL,
                               sys_prompt, job_id, judge_score)
        except Exception as e:
            log.debug(f"[{job_id}] output_cache.store skipped: {e}")
    if FEATURE_STRUCTURED_LOG:
        try:
            judge_score = None
            judge_cat = None
            if response is not None and response.judge:
                judge_score = response.judge.get("match_score")
                judge_cat = response.judge.get("category")
            STRUCTURED_LOG.emit(
                "generate_done",
                job_id=job_id, prompt=req.prompt,
                model=req.model or OLLAMA_MODEL,
                cache_hit=False, exec_ok=exec_ok,
                judge_score=judge_score,
                judge_category=judge_cat,
                attempts=(response.attempts if response else 0),
                geom_passed=(response.geom_check.get("passed")
                             if response and response.geom_check else None),
                print_warning_count=(len(response.print_warnings)
                                     if response else 0),
                latency_ms=int((time.time() - start_ts) * 1000),
                # S7.3 shadow: include compile-only verdict from
                # RestrictedPython. None if flag off / lib missing / empty.
                sandbox_shadow=_run_sandbox_shadow(code, job_id),
            )
        except Exception as e:
            log.debug(f"[{job_id}] structured_log emit skipped: {e}")


def _run_slicer_check(stl_path: Path) -> dict | None:
    if not FEATURE_SLICER_CHECK:
        return None
    try:
        from slicer_check import slice_stl
        r = slice_stl(stl_path, slicer_path=SLICER_PATH or None)
        return {
            "available": r.available,
            "sliced": r.sliced,
            "warnings": r.warnings[:5],
            "errors": r.errors[:5],
            "printable": r.printable,
        }
    except Exception as e:
        log.debug(f"slicer_check skipped: {e}")
        return None


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, no_cache: bool = False):
    """Generate 3D model from natural language description."""
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")

    # P3: shape-based model routing. Mutates req.model in place so the
    # entire pipeline (enrich, generate, refine) consistently uses the
    # routed model.
    req.model = _route_model_for_prompt(req.prompt, req.model)

    job_id = str(uuid.uuid4())[:8]
    _start_ts = time.time()

    # S5.1: output-cache lookup. Cache key includes the system_prompt
    # actually used (with pattern_cache injections), so cache stays
    # invalidated when pattern_cache evolves.
    sys_prompt_for_cache = _build_system_prompt_for(req.prompt)
    if FEATURE_OUTPUT_CACHE and not no_cache:
        try:
            hit = OUTPUT_CACHE.lookup(req.prompt,
                                      req.model or OLLAMA_MODEL,
                                      sys_prompt_for_cache)
        except Exception as e:
            log.debug(f"[{job_id}] output_cache lookup raised: {e}")
            hit = None
        if hit:
            resp = _build_cache_hit_response(req.prompt, hit, job_id)
            if FEATURE_STRUCTURED_LOG:
                STRUCTURED_LOG.emit(
                    "generate_done",
                    job_id=resp.id, prompt=req.prompt,
                    model=req.model or OLLAMA_MODEL,
                    cache_hit=True, exec_ok=True,
                    judge_score=hit.get("judge_score"),
                    latency_ms=int((time.time() - _start_ts) * 1000),
                )
            return resp

    if FEATURE_STRUCTURED_LOG:
        STRUCTURED_LOG.emit(
            "generate_start",
            job_id=job_id, prompt=req.prompt,
            model=req.model or OLLAMA_MODEL,
            system_prompt=sys_prompt_for_cache,
        )

    _using_cloud = is_cloud_model(req.model or OLLAMA_MODEL)
    if not _using_cloud:
        await tunnel.ensure_tunnel()
    try:
        # Step 0: Web search for design references (translate first via LLM)
        log.info(f"[{job_id}] Searching web for design references: {req.prompt[:60]}")
        search_info = await search_object_references(req.prompt)
        if search_info:
            log.info(f"[{job_id}] Search results: {search_info[:200]}")
        # Step 1: Enrich the prompt with detailed 3D design specification
        log.info(f"[{job_id}] Enriching prompt: {req.prompt[:100]}")
        user_content = req.prompt
        if search_info:
            user_content = f"{req.prompt}\n\nReference info from web search (use this to understand the object's structure and typical dimensions):\n{search_info}"
        enrich_messages = [
            {"role": "system", "content": ENRICH_PROMPT},
            {"role": "user", "content": user_content},
        ]
        enriched = await call_ollama(enrich_messages, req.model)
        enriched = enriched.strip()
        log.info(f"[{job_id}] Enriched: {enriched[:200]}")
        # Save enriched prompt
        enrich_dir = OUTPUT_DIR / job_id
        enrich_dir.mkdir(exist_ok=True)
        (enrich_dir / "enriched_prompt.txt").write_text(enriched, encoding="utf-8")

        # Step 2: Generate code from enriched prompt
        # S3.1: inject retrieval-augmented few-shot examples if we have
        # recent high-scoring generations of the same shape category.
        system_prompt_with_cache = _build_system_prompt_for(req.prompt)
        messages = [
            {"role": "system", "content": system_prompt_with_cache},
            {"role": "user", "content": enriched},
        ]
        last_error = None
        code = ""
        best_response: GenerateResponse | None = None  # remembered across judge retries
        total_attempts = 1 + MAX_RETRIES  # exec-fix attempts per generation
        judge_rounds = JUDGE_MAX_RETRIES if JUDGE_ENABLED else 0

        # Outer loop: visual-judge rounds (incl. round 0)
        for judge_round in range(judge_rounds + 1):
            # Inner loop: exec-fix attempts (regen on exec/validation error)
            inner_success = False
            for attempt in range(total_attempts):
                raw_code = await call_ollama(messages, req.model)
                raw_path = OUTPUT_DIR / job_id / f"raw_judge{judge_round}_attempt{attempt+1}.txt"
                raw_path.parent.mkdir(exist_ok=True)
                raw_path.write_text(raw_code, encoding="utf-8")
                code = clean_code(raw_code)
                if not code.strip():
                    log.warning(f"[{job_id}] J{judge_round}A{attempt+1}: empty code")
                    messages.append({"role": "assistant", "content": raw_code})
                    messages.append({"role": "user", "content": "Your response did not contain valid Python code. Please output ONLY a Python script."})
                    continue

                vres = _validate_for_backend(code)
                if vres is not None and not vres.ok:
                    log.warning(f"[{job_id}] J{judge_round}A{attempt+1} AST fail: {vres.errors[:3]}")
                    last_error = "; ".join(vres.errors[:5])
                    if attempt < total_attempts - 1:
                        messages.append({"role": "assistant", "content": code})
                        messages.append({"role": "user", "content": format_errors_for_llm(vres.errors)})
                    continue

                try:
                    await asyncio.to_thread(execute_code, code, job_id)
                    inner_success = True
                    break
                except HTTPException as e:
                    last_error = e.detail
                    log.warning(f"[{job_id}] J{judge_round}A{attempt+1} exec fail: {last_error}")
                    log.warning(f"[{job_id}] Failed code (first 600 chars):\n{code[:600]}")
                    if attempt < total_attempts - 1:
                        messages.append({"role": "assistant", "content": code})
                        fix_hint = _cadquery_fix_hint(last_error) if BACKEND == "cadquery" \
                                   else _trimesh_fix_hint(last_error)
                        # On the last retry before we bail, paste a minimal
                        # known-good template if the prompt is mug-shaped.
                        if BACKEND == "cadquery" and attempt == total_attempts - 2:
                            fix_hint += _final_retry_fallback(req.prompt)
                        messages.append({"role": "user", "content": f"The code failed with:\n{last_error}\nFix the code. Output ONLY valid Python code.{fix_hint}"})

            if not inner_success:
                # Couldn't produce executable code even after MAX_RETRIES; bail
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed after {total_attempts} attempts in judge round {judge_round}. Last error: {last_error or 'empty code'}",
                )

            # We have an executable STL.
            stl_path = OUTPUT_DIR / job_id / "model.stl"

            # S4.2: pre-judge watertight gate. If the STL isn't
            # watertight, skip the (expensive) VLM judge and feed a
            # specific watertight hint into the retry loop. The judge
            # can't see 'hole in the back' reliably anyway; this saves
            # a judge call AND gives the LLM a more actionable fix.
            if (WATERTIGHT_GATE_ENABLED and judge_round < judge_rounds
                    and not _check_watertight(stl_path)):
                log.warning(
                    f"[{job_id}] J{judge_round+1} STL not watertight; "
                    f"skipping judge, retrying with watertight hint"
                )
                best_response = GenerateResponse(
                    id=job_id,
                    code=code,
                    stl_url=f"/api/download/{job_id}",
                    enriched_prompt=enriched,
                    search_info=search_info,
                    thumbnails=[],
                    judge={
                        "category": "not_watertight",
                        "match_score": None,
                        "geometry_issues": [
                            "STL has holes / non-manifold edges (not printable)"
                        ],
                        "fix_suggestion": (
                            "Use union of closed primitives instead of "
                            "boolean subtraction on thin shells"
                        ),
                    },
                    attempts=judge_round + 1,
                )
                messages.append({"role": "assistant", "content": code})
                messages.append({
                    "role": "user",
                    "content": _watertight_retry_hint(req.prompt),
                })
                continue  # next judge round

            # S6.1: programmatic geometric gate (zero-token).
            # Runs after watertight gate, before VLM judge — catches
            # topological errors VLM might miss AND saves a judge call
            # by feeding the LLM an actionable retry hint.
            geom_result = _run_geom_check(stl_path, req.prompt)
            if (geom_result is not None and not geom_result.passed
                    and judge_round < judge_rounds):
                log.warning(
                    f"[{job_id}] J{judge_round+1} geom_check FAIL: "
                    f"{geom_result.fail_reason}"
                )
                best_response = GenerateResponse(
                    id=job_id,
                    code=code,
                    stl_url=f"/api/download/{job_id}",
                    enriched_prompt=enriched,
                    search_info=search_info,
                    thumbnails=[],
                    judge={
                        "category": "geom_check_fail",
                        "match_score": geom_result.score,
                        "geometry_issues": list(geom_result.issues),
                        "fix_suggestion": geom_result.fix_suggestion,
                    },
                    attempts=judge_round + 1,
                    geom_check={
                        "passed": False,
                        "score": geom_result.score,
                        "issues": list(geom_result.issues),
                        "method": geom_result.method,
                    },
                )
                messages.append({"role": "assistant", "content": code})
                messages.append({"role": "user", "content": (
                    f"Geometric check failed: {geom_result.fail_reason}. "
                    f"{geom_result.fix_suggestion} "
                    "Output ONLY corrected Python code."
                )})
                continue  # skip VLM, next judge round

            thumbs, judge = await _render_and_judge(job_id, stl_path, req.prompt)

            response = GenerateResponse(
                id=job_id,
                code=code,
                stl_url=f"/api/download/{job_id}",
                enriched_prompt=enriched,
                search_info=search_info,
                thumbnails=thumbs,
                judge=judge.to_dict() if judge else None,
                attempts=judge_round + 1,
                formats=_collect_format_urls(job_id),
                print_warnings=_run_print_readiness(stl_path),
                slicer=_run_slicer_check(stl_path),
                geom_check=(
                    {
                        "passed": geom_result.passed,
                        "score": geom_result.score,
                        "issues": list(geom_result.issues),
                        "method": geom_result.method,
                    } if geom_result is not None else None
                ),
            )

            # If judging disabled → accept whatever executed.
            if judge is None:
                _finalize_generate(req, job_id, response, code,
                                   sys_prompt_for_cache, _start_ts,
                                   exec_ok=True, store_cache=False)
                return response

            # S1.1: judge returned but couldn't score (API error / disabled
            # mid-flight). Don't retry blindly — the judge is still broken.
            # Accept the current STL but the response carries category=
            # 'judge_api_error' / 'judge_disabled' so benchmark & UI can see.
            if judge.match_score is None:
                log.warning(
                    f"[{job_id}] Judge unavailable ({judge.category}); "
                    f"accepting current STL without retry."
                )
                _finalize_generate(req, job_id, response, code,
                                   sys_prompt_for_cache, _start_ts,
                                   exec_ok=True, store_cache=False)
                return response

            # Normal path: a numeric score came back.
            # 2026-04-26 patch: also require geom_check pass (when available).
            # Was: bottle prompt could still ship 6/10 + geom_passed=False on
            # the final round because retry-on-geom-fail only fires while
            # judge_round < judge_rounds. Now those degraded results stay out
            # of pattern_cache + output_cache and fall to the best-effort path.
            geom_ok = geom_result is None or geom_result.passed
            if judge.match_score >= JUDGE_MIN_SCORE and geom_ok:
                log.info(f"[{job_id}] Judge passed at round {judge_round+1} (score={judge.match_score})")
                # S3.1: cache a high-scoring success for future few-shot use.
                try:
                    PATTERN_CACHE.record_success(req.prompt, code, judge.match_score)
                except Exception as e:
                    log.debug(f"[{job_id}] pattern_cache write skipped: {e}")
                # S5.1: store in output_cache for exact-match future hits.
                _finalize_generate(req, job_id, response, code,
                                   sys_prompt_for_cache, _start_ts,
                                   exec_ok=True, store_cache=True)
                # S7.1 shadow: only spawn shadow round on a TRUE judge-pass,
                # so baseline_score is meaningful. Fire-and-forget.
                _maybe_spawn_best_of_n_shadow(req, response, job_id)
                return response

            # Rejected — remember and prepare retry.
            best_response = response
            if not geom_ok and judge.match_score >= JUDGE_MIN_SCORE:
                log.info(
                    f"[{job_id}] Judge OK ({judge.match_score}) but geom_check FAIL "
                    f"({geom_result.fail_reason if geom_result else '?'}), "
                    f"treating as rejection"
                )
            else:
                log.info(f"[{job_id}] Judge rejected (score={judge.match_score} < {JUDGE_MIN_SCORE}), retrying...")
            if judge_round < judge_rounds:
                messages.append({"role": "assistant", "content": code})
                messages.append({"role": "user", "content": build_retry_instruction(req.prompt, code, judge)})

        # All judge rounds exhausted; return the last (best-effort) response
        log.info(f"[{job_id}] Judge retries exhausted; returning last result")
        _finalize_generate(req, job_id, best_response, code if 'code' in dir() else "",
                           sys_prompt_for_cache, _start_ts,
                           exec_ok=best_response is not None, store_cache=False)
        return best_response  # type: ignore[return-value]
    except HTTPException:
        if FEATURE_STRUCTURED_LOG:
            STRUCTURED_LOG.emit(
                "generate_done",
                job_id=job_id, prompt=req.prompt,
                model=req.model or OLLAMA_MODEL,
                cache_hit=False, exec_ok=False,
                latency_ms=int((time.time() - _start_ts) * 1000),
            )
        raise
    finally:
        if not _using_cloud:
            tunnel.release()


class RefineRequest(BaseModel):
    job_id: str
    feedback: str
    current_code: str
    model: str | None = None


@app.post("/api/refine", response_model=GenerateResponse)
async def refine(req: RefineRequest):
    """Refine an existing 3D model based on user feedback."""
    if not req.feedback.strip():
        raise HTTPException(status_code=400, detail="Feedback is required")

    job_id = req.job_id  # Reuse same job ID to overwrite
    _using_cloud = is_cloud_model(req.model or OLLAMA_MODEL)
    if not _using_cloud:
        await tunnel.ensure_tunnel()
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Here is the current code that generates a 3D model:"},
            {"role": "assistant", "content": req.current_code},
            {"role": "user", "content": f"Please modify the code based on this feedback: {req.feedback}\nOutput ONLY the complete modified Python code."},
        ]
        last_error = None
        code = ""
        for attempt in range(1 + MAX_RETRIES):
            raw_code = await call_ollama(messages, req.model)
            raw_path = OUTPUT_DIR / job_id / f"raw_refine_{attempt+1}.txt"
            raw_path.parent.mkdir(exist_ok=True)
            raw_path.write_text(raw_code, encoding="utf-8")
            code = clean_code(raw_code)
            if not code.strip():
                log.warning(f"[{job_id}] Refine attempt {attempt+1}: empty code")
                messages.append({"role": "assistant", "content": raw_code})
                messages.append({"role": "user", "content": "Your response did not contain valid Python code. Please output ONLY a Python script."})
                continue

            # AST validation
            vres = _validate_for_backend(code)
            if vres is not None and not vres.ok:
                log.warning(f"[{job_id}] Refine attempt {attempt+1} AST validation failed: {vres.errors[:3]}")
                last_error = "; ".join(vres.errors[:5])
                if attempt < MAX_RETRIES:
                    messages.append({"role": "assistant", "content": code})
                    messages.append({"role": "user", "content": format_errors_for_llm(vres.errors)})
                continue

            try:
                await asyncio.to_thread(execute_code, code, job_id)
                return GenerateResponse(id=job_id, code=code, stl_url=f"/api/download/{job_id}", enriched_prompt=req.feedback)
            except HTTPException as e:
                last_error = e.detail
                log.warning(f"[{job_id}] Refine attempt {attempt+1} exec failed: {last_error}")
                if attempt < MAX_RETRIES:
                    messages.append({"role": "assistant", "content": code})
                    messages.append({"role": "user", "content": f"The code failed with this error:\n{last_error}\nPlease fix the code. Output ONLY valid Python code."})
        raise HTTPException(status_code=500, detail=f"Refine failed after {1+MAX_RETRIES} attempts. Last error: {last_error or 'empty code'}")
    finally:
        if not _using_cloud:
            tunnel.release()


class AutoReviewRequest(BaseModel):
    job_id: str
    current_code: str
    enriched_prompt: str
    model: str | None = None


class AutoReviewResponse(BaseModel):
    id: str
    code: str
    stl_url: str
    review: str
    changed: bool


@app.post("/api/auto-review", response_model=AutoReviewResponse)
async def auto_review(req: AutoReviewRequest):
    """One round of self-review: LLM reviews code, then fixes if needed."""
    _using_cloud = is_cloud_model(req.model or OLLAMA_MODEL)
    if not _using_cloud:
        await tunnel.ensure_tunnel()
    try:
        job_id = req.job_id
        # Step 1: Review
        log.info(f"[{job_id}] Auto-review started")
        review_msgs = [
            {"role": "system", "content": REVIEW_PROMPT},
            {"role": "user", "content": f"Design specification:\n{req.enriched_prompt}\n\nCode:\n{req.current_code}"},
        ]
        review = await call_ollama(review_msgs, req.model)
        review = review.strip()
        log.info(f"[{job_id}] Review result: {review[:200]}")

        if "LGTM" in review.upper():
            return AutoReviewResponse(id=job_id, code=req.current_code, stl_url=f"/api/download/{job_id}", review="✅ LGTM — 審查通過，無需修正", changed=False)

        # Step 2: Fix based on review
        refine_msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": req.enriched_prompt},
            {"role": "assistant", "content": req.current_code},
            {"role": "user", "content": f"A reviewer found these issues:\n{review}\nPlease fix the code. Output ONLY the complete fixed Python code."},
        ]
        raw_fix = await call_ollama(refine_msgs, req.model)
        fixed_code = clean_code(raw_fix)
        if not fixed_code.strip():
            return AutoReviewResponse(id=job_id, code=req.current_code, stl_url=f"/api/download/{job_id}", review=f"審查意見: {review}\n\n⚠️ 修正失敗（空程式碼），保留原版", changed=False)

        # AST validation — if fail, fallback to original
        vres = _validate_for_backend(fixed_code)
        if vres is not None and not vres.ok:
            log.warning(f"[{job_id}] Auto-review fix failed validation: {vres.errors[:3]}")
            return AutoReviewResponse(id=job_id, code=req.current_code, stl_url=f"/api/download/{job_id}", review=f"審查意見: {review}\n\n⚠️ 修正後的程式碼含未知 API: {vres.errors[0]}\n保留原版", changed=False)

        try:
            await asyncio.to_thread(execute_code, fixed_code, job_id)
            return AutoReviewResponse(id=job_id, code=fixed_code, stl_url=f"/api/download/{job_id}", review=review, changed=True)
        except HTTPException as e:
            # Fix failed, restore previous
            await asyncio.to_thread(execute_code, req.current_code, job_id)
            return AutoReviewResponse(id=job_id, code=req.current_code, stl_url=f"/api/download/{job_id}", review=f"審查意見: {review}\n\n⚠️ 修正程式碼執行失敗: {e.detail}\n保留原版", changed=False)
    finally:
        if not _using_cloud:
            tunnel.release()


@app.get("/api/download/{job_id}")
async def download(job_id: str, fmt: str = "stl"):
    """Serve generated model file. ?fmt=stl|step|3mf|glb (S5.2)."""
    fmt = fmt.lower().strip()
    fmt_to_mime = {
        "stl":  "application/sla",
        "step": "application/step",
        "3mf":  "model/3mf",
        "glb":  "model/gltf-binary",
    }
    if fmt not in fmt_to_mime:
        raise HTTPException(status_code=400, detail=f"unsupported fmt: {fmt}")
    file_path = OUTPUT_DIR / job_id / f"model.{fmt}"
    if not file_path.exists():
        # If user asks for STL specifically, this is a real 404. For other
        # formats, surface a clearer message indicating the format wasn't
        # produced (multi_format_export disabled or backend export failed).
        detail = (f"{fmt.upper()} file not generated for this job — "
                  "either generation was older than the multi_format_export "
                  "feature or that format failed to export."
                  ) if fmt != "stl" else "STL file not found"
        raise HTTPException(status_code=404, detail=detail)
    return FileResponse(file_path, media_type=fmt_to_mime[fmt],
                        filename=f"{job_id}.{fmt}")


@app.get("/api/thumbnail/{job_id}/{filename}")
async def thumbnail(job_id: str, filename: str):
    """Serve a rendered thumbnail PNG for the VLM judge visualisation."""
    # Sanitize: only allow simple PNG filenames, no traversal
    if "/" in filename or ".." in filename or not filename.endswith(".png"):
        raise HTTPException(status_code=400, detail="invalid thumbnail filename")
    png_path = OUTPUT_DIR / job_id / "views" / filename
    if not png_path.exists():
        raise HTTPException(status_code=404, detail="thumbnail not found")
    return FileResponse(png_path, media_type="image/png")


@app.get("/api/models")
async def list_models():
    """List available models. DGX/Ollama local models hidden from UI by user request
    (2026-04-26); cloud-only. The Ollama path still works server-side for any
    direct /api/generate caller that names a local model — we just don't
    advertise them in the dropdown."""
    return {"models": list(CLOUD_MODELS)}



@app.post("/api/warmup")
async def warmup_model(model: str | None = None):
    """Pre-load model into DGX GPU memory with a tiny request."""
    await tunnel.ensure_tunnel()
    try:
        m = model or OLLAMA_MODEL
        payload = {
            "model": m,
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
            "options": {"num_predict": 1},
            "keep_alive": "10m",
        }
        async with httpx.AsyncClient(timeout=600) as client:
            resp = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
            if resp.status_code == 200:
                return {"status": "ok", "model": m}
            return {"status": "error", "detail": resp.text}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
    finally:
        tunnel.release()



@app.get("/api/tunnel-status")
async def tunnel_status():
    """Check tunnel status."""
    idle = int(time.time() - tunnel._last_used) if tunnel._last_used else -1
    return {
        "alive": tunnel.is_alive,
        "idle_seconds": idle,
        "timeout": TUNNEL_IDLE_TIMEOUT,
        "active_requests": tunnel._active_requests,
    }


@app.get("/api/stats")
async def stats():
    """S4.7: operational stats — failover rate, pattern cache size, etc."""
    cache_size = {
        cat: len(bucket)
        for cat, bucket in getattr(PATTERN_CACHE, "_data", {}).get("categories", {}).items()
    }
    total_calls = _failover_stats["total_calls"]
    failover_pct = (
        100.0 * _failover_stats["failovers"] / total_calls
        if total_calls else 0.0
    )
    output_cache_stats: dict = {}
    try:
        output_cache_stats = OUTPUT_CACHE.stats()
    except Exception as e:
        output_cache_stats = {"error": str(e)}
    log_agg: dict = {}
    try:
        log_agg = STRUCTURED_LOG.aggregate()
    except Exception as e:
        log_agg = {"error": str(e)}
    return {
        "failover": {
            **_failover_stats,
            "failover_pct": round(failover_pct, 2),
        },
        "pattern_cache": {
            "categories": cache_size,
            "total_samples": sum(cache_size.values()),
        },
        "output_cache": output_cache_stats,
        "structured_log": log_agg,
        "feature_flags": {
            # legacy flags (Sprint 1-4)
            "mesh_repair": MESH_REPAIR_ENABLED,
            "watertight_gate": WATERTIGHT_GATE_ENABLED,
            "judge": JUDGE_ENABLED,
            "ast_validate": AST_VALIDATE,
            "pattern_cache": PATTERN_CACHE_ENABLED,
            "shape_routing": SHAPE_ROUTING_ENABLED,
            # Sprint 5-7 flags
            "output_cache": FEATURE_OUTPUT_CACHE,
            "structured_log": FEATURE_STRUCTURED_LOG,
            "geom_check": FEATURE_GEOM_CHECK,
            "print_readiness": FEATURE_PRINT_READINESS,
            "multi_format_export": FEATURE_MULTI_FORMAT_EXPORT,
            "sandbox_strict": FEATURE_SANDBOX_STRICT,
            "best_of_n": FEATURE_BEST_OF_N,
            "slicer_check": FEATURE_SLICER_CHECK,
            "render_pyvista": FEATURE_RENDER_PYVISTA,
        },
        "backend": BACKEND,
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/favicon.ico")
async def favicon():
    """Return empty favicon to prevent 404."""
    return Response(content=b"", media_type="image/x-icon", status_code=204)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
