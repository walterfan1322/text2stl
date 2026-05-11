"""Reference image generation for image-grounded self-correction (P3.1).

Given a user description (e.g. "一張椅子" / "a wooden chair"), generate a
single canonical iso-view reference PNG via fal.ai Flux-schnell. The
candidate STL — once rendered at the same canonical viewpoint — is then
compared to this reference image (silhouette IoU + optional VLM critic)
to drive retries.

Design choices
--------------
1. **General, not per-category.** The prompt template wraps any user
   description; we do not maintain per-shape recipes (per user
   directive). Style hints (white background, flat clay shading, no
   textures) make the reference comparable across categories.
2. **Camera convention matches `rendering.VIEW_ANGLES["iso"]`** —
   upper-front-right ¾ view (~30° elevation, ~45° azimuth). Flux is
   asked for "isometric 3/4 view from upper-front-right" so silhouettes
   line up with the candidate STL's rendered iso view.
3. **Cache by description hash.** Identical prompts reuse the same
   reference, so retry rounds within one job (and re-runs of the same
   prompt) are free after the first call.
4. **Soft-fail.** If FAL_KEY is missing or the API errors, return None;
   callers treat that as "no reference, skip IoU". We never break
   generation because the reference call failed.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger("text2stl.image_gen")

FAL_ENDPOINT = "https://fal.run/fal-ai/flux/schnell"
DEFAULT_TIMEOUT = 60  # fal usually returns in <2s; 60s covers cold-start spikes

# Style suffix matches our trimesh/pyvista candidate render look:
# pale matte color, white background, no textures, single object centered.
# Keeping this CATEGORY-AGNOSTIC is critical (per project directive).
STYLE_PREFIX = (
    "isometric 3/4 view from upper-front-right of a single "
)
STYLE_SUFFIX = (
    ", low-poly 3D model, flat matte gray clay shading, pure white "
    "background, centered, full object visible, blender clay render "
    "style, no textures, no shadows, no scene, no ground plane"
)


def _cache_key(description: str, view: str) -> str:
    """Stable filename-safe hash of (description, view)."""
    h = hashlib.sha256(f"{view}::{description}".encode("utf-8")).hexdigest()
    return h[:16]


def build_flux_prompt(description: str) -> str:
    """Wrap a free-form user description in our canonical-view style hints.

    Kept simple on purpose: same wrapper for every category. If a
    description already contains conflicting style ("photo of...",
    "realistic..."), Flux-schnell is robust enough to bias toward our
    suffix at default guidance.
    """
    return f"{STYLE_PREFIX}{description.strip()}{STYLE_SUFFIX}"


def generate_reference_image(
    description: str,
    out_dir: Path,
    view: str = "iso",
    cache_dir: Path | None = None,
    fal_key: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Path | None:
    """Generate a reference PNG matching the canonical iso view.

    Args:
        description: user prompt (Chinese or English; passed to Flux as-is
            after style wrapping)
        out_dir: per-job directory. The reference is copied here as
            `reference_<view>.png` so it shows up alongside the rendered
            candidate views.
        view: which canonical view to request. Currently only "iso"
            wrapping is implemented; "front"/"side"/"top" callers should
            adjust the wrapper if added later.
        cache_dir: shared cache dir (defaults to `<out_dir>/../_image_cache`).
            Identical descriptions reuse the same PNG.
        fal_key: explicit key, else read FAL_KEY env var.
        timeout: HTTP timeout seconds.

    Returns the per-job reference path on success, None on any failure.
    """
    fal_key = fal_key or os.environ.get("FAL_KEY")
    if not fal_key:
        log.info("image_gen: FAL_KEY not set, skipping reference generation")
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir is None:
        cache_dir = out_dir.parent / "_image_cache"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = _cache_key(description, view)
    cache_png = cache_dir / f"{key}.png"
    out_png = out_dir / f"reference_{view}.png"

    if cache_png.exists() and cache_png.stat().st_size > 1000:
        # Cache hit — copy bytes into per-job dir so the UI/judge sees it.
        out_png.write_bytes(cache_png.read_bytes())
        log.info(f"image_gen: cache hit {key} ({cache_png.stat().st_size} bytes)")
        return out_png

    prompt = build_flux_prompt(description)
    body = {
        "prompt": prompt,
        "image_size": "square_hd",      # 1024×1024
        "num_inference_steps": 4,        # schnell uses 4 steps
        "num_images": 1,
        "enable_safety_checker": False,  # CAD prompts trip false positives
    }
    req = urllib.request.Request(
        FAL_ENDPOINT,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Key {fal_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_txt = e.read().decode("utf-8", "replace")[:300]
        log.warning(f"image_gen: fal.ai HTTP {e.code}: {body_txt}")
        return None
    except Exception as e:
        log.warning(f"image_gen: fal.ai call failed: {e}")
        return None

    images = d.get("images") or []
    if not images or not images[0].get("url"):
        log.warning(f"image_gen: fal.ai returned no image url: {str(d)[:200]}")
        return None

    img_url = images[0]["url"]
    try:
        with urllib.request.urlopen(img_url, timeout=30) as r:
            img_bytes = r.read()
    except Exception as e:
        log.warning(f"image_gen: download {img_url[:60]}... failed: {e}")
        return None

    if len(img_bytes) < 1000:
        log.warning(f"image_gen: suspiciously small image ({len(img_bytes)} bytes)")
        return None

    cache_png.write_bytes(img_bytes)
    out_png.write_bytes(img_bytes)
    dt = time.time() - t0
    log.info(
        f"image_gen: fal.ai ok {dt:.1f}s, {len(img_bytes)} bytes, "
        f"key={key}, prompt_first40={prompt[:40]!r}"
    )
    return out_png


def reference_as_data_url(path: Path) -> str:
    """Encode a reference PNG as a data: URL (for VLM critic prompts)."""
    return f"data:image/png;base64,{base64.b64encode(Path(path).read_bytes()).decode()}"
