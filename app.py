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

OLLAMA_URL = _cfg.get("ollama_url", "http://localhost:11434")
OLLAMA_MODEL = _cfg.get("ollama_model", "qwen3:14b")
SSH_TUNNEL_HOST = _cfg.get("ssh_tunnel_host", "user@YOUR_DGX_HOST")
SSH_KEY_PATH = _cfg.get("ssh_key_path", "")
TUNNEL_IDLE_TIMEOUT = int(_cfg.get("tunnel_idle_timeout", 300))

# Cloud API (MiniMax / OpenAI-compatible)
CLOUD_API_KEY = _cfg.get("cloud_api_key", "")
CLOUD_API_BASE = _cfg.get("cloud_api_base", "")
CLOUD_MODELS = _cfg.get("cloud_models", [])

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(BASE_DIR / "server.log", encoding="utf-8"),
    ],
)
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
# LLM prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = r"""You are an expert Python 3D modeling programmer. The user will describe a 3D object in natural language.
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


ENRICH_PROMPT = r"""You are a 3D product designer for 3D printing. The user describes an object.
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


REVIEW_PROMPT = r"""You are a 3D model code reviewer. You are given:
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


async def call_cloud_llm(messages: list[dict], model: str) -> str:
    """Call cloud LLM API (OpenAI-compatible format, e.g. MiniMax)."""
    if not CLOUD_API_KEY:
        raise HTTPException(status_code=500, detail="Cloud API key not configured")
    headers = {
        "Authorization": f"Bearer {CLOUD_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.3,
    }
    log.info(f"Calling cloud LLM: {model} via {CLOUD_API_BASE}")
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(f"{CLOUD_API_BASE}/chat/completions",
                                 headers=headers, json=payload)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Cloud API error: {resp.text[:500]}")
        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        usage = data.get("usage", {})
        log.info(f"Cloud LLM response: {len(content)} chars, tokens: {usage}")
        return content


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


def _make_helper_globals():
    """Create pre-defined helper functions available to generated code."""
    import numpy as _np
    import trimesh as _trimesh

    def make_frustum(r_bottom, r_top, height, sections=64):
        """Create a frustum (truncated cone) mesh."""
        angles = _np.linspace(0, 2 * _np.pi, sections, endpoint=False)
        bottom = _np.column_stack([r_bottom * _np.cos(angles), r_bottom * _np.sin(angles), _np.zeros(sections)])
        top = _np.column_stack([r_top * _np.cos(angles), r_top * _np.sin(angles), _np.full(sections, height)])
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
        """Revolve a CLOSED 2D profile polygon around Z axis to create a watertight solid.
        profile: list of (radius, z) points forming a CLOSED polygon (wall cross-section).
        Trace outer surface top-to-bottom, then inner surface bottom-to-top.
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


HELPER_GLOBALS = _make_helper_globals()


def execute_code(code: str, job_id: str) -> Path:
    """Execute generated Python code to create STL file."""
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    stl_path = job_dir / "model.stl"
    code_path = job_dir / "code.py"
    # Strip any OUTPUT_PATH reassignment from generated code
    code = re.sub(r'^OUTPUT_PATH\s*=.*$', '# OUTPUT_PATH is injected', code, flags=re.MULTILINE)
    # Strip any re-definitions of pre-injected helper functions
    _func_def_pattern = r'^def {}\(.*\):[ \t]*\n(?:(?:[ \t]+.*|[ \t]*)\n)*'
    for fname in ('make_frustum', 'make_solid_revolution'):
        code = re.sub(_func_def_pattern.format(fname), '', code, flags=re.MULTILINE)
    # Fix common LLM mistakes: replace trimesh.creation.revolve with make_solid_revolution
    if 'trimesh.creation.revolve' in code:
        log.warning("Fixing LLM mistake: replacing trimesh.creation.revolve with make_solid_revolution")
        code = code.replace('trimesh.creation.revolve', 'make_solid_revolution')

    code_path.write_text(code, encoding="utf-8")

    exec_globals = {"OUTPUT_PATH": str(stl_path), "__builtins__": __builtins__}
    exec_globals.update(HELPER_GLOBALS)
    try:
        exec(code, exec_globals)
    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        hint = ""
        if "not all meshes are volumes" in str(e).lower():
            hint = "\nHINT: Boolean ops require watertight meshes. Use trimesh.util.concatenate instead."
        elif "manifold" in str(e).lower():
            hint = "\nHINT: Boolean engine error. Use trimesh.util.concatenate instead."
        raise HTTPException(status_code=500, detail=f"Code execution failed:\n{err_msg}{hint}")

    if not stl_path.exists():
        raise HTTPException(status_code=500, detail="STL file was not created. The code may not have called .export(OUTPUT_PATH)")

    return stl_path


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
MAX_RETRIES = 2


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Generate 3D model from natural language description."""
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")

    job_id = str(uuid.uuid4())[:8]
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
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": enriched},
        ]
        last_error = None
        code = ""
        for attempt in range(1 + MAX_RETRIES):
            raw_code = await call_ollama(messages, req.model)
            # Save raw output for debugging
            raw_path = OUTPUT_DIR / job_id / f"raw_attempt{attempt+1}.txt"
            raw_path.parent.mkdir(exist_ok=True)
            raw_path.write_text(raw_code, encoding="utf-8")
            code = clean_code(raw_code)
            if not code.strip():
                log.warning(f"[{job_id}] Attempt {attempt+1}: LLM returned empty code (raw saved to {raw_path})")
                messages.append({"role": "assistant", "content": raw_code})
                messages.append({"role": "user", "content": "Your response did not contain valid Python code. Please output ONLY a Python script that creates the 3D model and exports it with mesh.export(OUTPUT_PATH)."})
                continue
            try:
                await asyncio.to_thread(execute_code, code, job_id)
                return GenerateResponse(id=job_id, code=code, stl_url=f"/api/download/{job_id}", enriched_prompt=enriched, search_info=search_info)
            except HTTPException as e:
                last_error = e.detail
                log.warning(f"[{job_id}] Attempt {attempt+1} exec failed: {last_error}")
                if attempt < MAX_RETRIES:
                    messages.append({"role": "assistant", "content": code})
                    fix_hint = ""
                    if "revolve" in str(last_error).lower():
                        fix_hint = "\nIMPORTANT: Do NOT use trimesh.creation.revolve(). Use the pre-defined make_solid_revolution(profile) function instead."
                    if "not all meshes are volumes" in str(last_error).lower():
                        fix_hint = "\nIMPORTANT: Do NOT use boolean operations with revolution meshes. Use make_solid_revolution with a closed wall profile instead, and use trimesh.util.concatenate to combine parts."
                    messages.append({"role": "user", "content": f"The code failed with this error:\n{last_error}\nPlease fix the code and try again. Output ONLY valid Python code.{fix_hint}"})
        # All retries exhausted
        raise HTTPException(status_code=500, detail=f"Failed after {1+MAX_RETRIES} attempts. Last error: {last_error or 'empty code'}")
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
async def download(job_id: str):
    stl_path = OUTPUT_DIR / job_id / "model.stl"
    if not stl_path.exists():
        raise HTTPException(status_code=404, detail="STL file not found")
    return FileResponse(stl_path, media_type="application/sla", filename=f"{job_id}.stl")


@app.get("/api/models")
async def list_models():
    """List available models (Ollama + cloud)."""
    models = list(CLOUD_MODELS)  # Cloud models first
    await tunnel.ensure_tunnel()
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(f"{OLLAMA_URL}/api/tags")
            data = resp.json()
            models.extend([m["name"] for m in data.get("models", [])])
    except Exception as e:
        log.warning(f"Failed to list Ollama models: {e}")
    finally:
        if not _using_cloud:
            tunnel.release()
    return {"models": models}


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
        if not _using_cloud:
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


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/favicon.ico")
async def favicon():
    """Return empty favicon to prevent 404."""
    return Response(content=b"", media_type="image/x-icon", status_code=204)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
