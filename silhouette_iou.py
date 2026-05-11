"""Normalized binary silhouette IoU between reference and candidate views.

Used by the image-grounded retry loop (P3.1):
- reference_iso.png  ← Flux-schnell output for the user's prompt
- views/iso.png      ← rendered candidate STL at the same canonical view

We don't expect pixel-perfect alignment (Flux uses its own internal
camera; CadQuery render uses its own bbox-fit camera). Instead we
normalize each silhouette to a centered, aspect-preserved 256² canvas
and compute IoU on those normalized binary masks. This rewards "right
overall shape and proportions" while ignoring framing/zoom differences.

The metric is intentionally LOW-FIDELITY: a 4-legged chair vs a 4-legged
stool both score ~0.9; a chair vs a sphere scores ~0.4. That's what we
want — a hard sanity check on silhouette, with the VLM critic (P3.2) doing
the fine-grained part-counting.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("text2stl.silhouette_iou")

# Foreground = pixel luminance < (bg_luminance - LUM_DELTA). We pick
# bg_luminance from the brightest corner of the image (see
# `_detect_bg_luminance`) instead of a fixed threshold; some Flux
# outputs come back with a soft-shadow bg around RGB(220,220,220) which
# a fixed 240 cut treats as 100% foreground.
LUM_DELTA = 30
# Canvas size for normalized comparison. 256 is plenty for binary IoU
# and keeps the call fast (~5ms total for two masks).
CANVAS = 256
# Smallest valid foreground area as a fraction of the image. Below
# this, we treat the silhouette as "extraction failed" and bail out.
MIN_FG_FRACTION = 0.005
# Upper bound on foreground fraction; above this the extractor is
# almost certainly mis-classifying the bg. Bumped 0.95 → 0.92 because
# realistic clay-render shadows can legitimately fill ~85% of the frame.
MAX_FG_FRACTION = 0.92


def _detect_bg_luminance(arr: np.ndarray) -> float:
    """Estimate background luminance from corner samples.

    Picks the BRIGHTEST of the four corners. The assumption is that
    on an iso 3/4 view at least one of the four corners is clean
    background, even if the object touches one edge. Flux clay
    renders sometimes have a soft shadow biasing the lower corners
    darker — taking the brightest corner avoids that bias.
    """
    h, w = arr.shape[:2]
    box = max(4, min(h, w) // 32)  # ~3% of the smaller dimension
    corners = [
        arr[:box, :box],
        arr[:box, w - box:],
        arr[h - box:, :box],
        arr[h - box:, w - box:],
    ]
    # Per-corner mean luminance (uniform-weight RGB → grayscale)
    lums = [float(c.reshape(-1, 3).mean()) for c in corners]
    return max(lums)


def _load_mask(path: Path) -> np.ndarray | None:
    """Read PNG → binary foreground mask (uint8 0/1).

    Foreground = pixels meaningfully darker than the detected
    background luminance (see `_detect_bg_luminance`). Adaptive: a
    pure-white-bg image and a soft-shadow-bg image are both handled
    correctly.
    """
    try:
        from PIL import Image
    except ImportError:
        log.error("silhouette_iou: PIL not installed")
        return None
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        log.warning(f"silhouette_iou: cannot read {path}: {e}")
        return None

    arr = np.asarray(img)  # H, W, 3
    bg_lum = _detect_bg_luminance(arr)
    pix_lum = arr.mean(axis=2)
    # Foreground: anything darker than bg_lum - LUM_DELTA
    cutoff = max(bg_lum - LUM_DELTA, 0.0)
    fg = (pix_lum < cutoff).astype(np.uint8)

    fraction = float(fg.mean())
    if fraction < MIN_FG_FRACTION:
        log.warning(f"silhouette_iou: {path.name} has only "
                    f"{fraction*100:.2f}% foreground "
                    f"(bg_lum={bg_lum:.1f}, cutoff={cutoff:.1f}); skipping")
        return None
    if fraction > MAX_FG_FRACTION:
        log.warning(f"silhouette_iou: {path.name} is "
                    f"{fraction*100:.2f}% foreground "
                    f"(bg_lum={bg_lum:.1f}, cutoff={cutoff:.1f}); "
                    f"extractor probably wrong")
        return None
    return fg


def _normalize(mask: np.ndarray) -> np.ndarray:
    """Crop to bbox, then aspect-preserved resize into CANVAS×CANVAS.

    The result is centered with letterboxing on the shorter axis. This
    discards absolute scale differences so a tightly-framed Flux output
    and a loosely-framed CadQuery render of the same shape line up.
    """
    from PIL import Image

    ys, xs = np.where(mask > 0)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    cropped = mask[y0:y1, x0:x1]
    h, w = cropped.shape

    # Aspect-preserved fit into CANVAS-2 box (1px border for safety)
    box = CANVAS - 2
    if h >= w:
        new_h = box
        new_w = max(1, int(round(w * box / h)))
    else:
        new_w = box
        new_h = max(1, int(round(h * box / w)))

    pil = Image.fromarray((cropped * 255).astype(np.uint8))
    pil = pil.resize((new_w, new_h), Image.NEAREST)
    resized = (np.asarray(pil) > 127).astype(np.uint8)

    out = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
    y_off = (CANVAS - new_h) // 2
    x_off = (CANVAS - new_w) // 2
    out[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return out


def silhouette_iou(reference: Path, candidate: Path) -> float | None:
    """Compute normalized binary IoU between two view PNGs.

    Returns:
        float in [0, 1], or None if either silhouette extraction failed.

    Caller treats None as "no signal" (don't change the retry decision
    based on silhouette).
    """
    ref_mask = _load_mask(Path(reference))
    if ref_mask is None:
        return None
    cand_mask = _load_mask(Path(candidate))
    if cand_mask is None:
        return None

    ref_n = _normalize(ref_mask)
    cand_n = _normalize(cand_mask)

    inter = int(((ref_n > 0) & (cand_n > 0)).sum())
    union = int(((ref_n > 0) | (cand_n > 0)).sum())
    if union == 0:
        return None
    iou = inter / union
    log.info(
        f"silhouette_iou: ref={Path(reference).name} "
        f"cand={Path(candidate).name} iou={iou:.3f} "
        f"(inter={inter}, union={union})"
    )
    return iou


def silhouette_iou_with_debug(
    reference: Path, candidate: Path, debug_dir: Path | None = None,
) -> tuple[float | None, dict]:
    """Like silhouette_iou but also returns a debug dict and optionally
    writes the normalized masks for inspection.

    Useful during P3.1 bring-up to eyeball alignment quality.
    """
    info: dict = {"reference": str(reference), "candidate": str(candidate)}
    ref_mask = _load_mask(Path(reference))
    if ref_mask is None:
        info["error"] = "reference mask extraction failed"
        return None, info
    cand_mask = _load_mask(Path(candidate))
    if cand_mask is None:
        info["error"] = "candidate mask extraction failed"
        return None, info

    ref_n = _normalize(ref_mask)
    cand_n = _normalize(cand_mask)

    inter = int(((ref_n > 0) & (cand_n > 0)).sum())
    union = int(((ref_n > 0) | (cand_n > 0)).sum())
    iou = inter / union if union else None
    info.update({
        "iou": iou,
        "inter_px": inter,
        "union_px": union,
        "ref_fg_fraction": float(ref_mask.mean()),
        "cand_fg_fraction": float(cand_mask.mean()),
    })

    if debug_dir is not None:
        from PIL import Image
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray((ref_n * 255).astype(np.uint8)).save(
            debug_dir / "ref_normalized.png")
        Image.fromarray((cand_n * 255).astype(np.uint8)).save(
            debug_dir / "cand_normalized.png")
        # Overlay: red=ref-only, green=cand-only, yellow=both
        overlay = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)
        overlay[..., 0] = ref_n * 255   # red channel
        overlay[..., 1] = cand_n * 255  # green channel
        Image.fromarray(overlay).save(debug_dir / "iou_overlay.png")
    return iou, info
