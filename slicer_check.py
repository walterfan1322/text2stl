"""Slicer integration (S6.3).

Run a real 3D slicer (PrusaSlicer / OrcaSlicer / SuperSlicer) headless
against a generated STL to confirm it's actually printable. Captures
warnings (thin walls, manifold issues, support needs) without us having
to reimplement those checks ourselves.

Graceful degradation: if no slicer is found on PATH or at the
configured location, returns a neutral result with `available=False`.
The caller treats that the same as "no slicer info available", not a
failure.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("text2stl.slicer_check")


CANDIDATE_BINARIES = [
    "prusa-slicer",
    "PrusaSlicer",
    "orca-slicer",
    "OrcaSlicer",
    "superslicer",
    "SuperSlicer",
]
CANDIDATE_APP_PATHS = [
    "/Applications/PrusaSlicer.app/Contents/MacOS/PrusaSlicer",
    "/Applications/OrcaSlicer.app/Contents/MacOS/OrcaSlicer",
    "/Applications/SuperSlicer.app/Contents/MacOS/SuperSlicer",
    "C:/Program Files/Prusa3D/PrusaSlicer/prusa-slicer-console.exe",
    "/usr/bin/prusa-slicer",
    "/usr/local/bin/prusa-slicer",
]


@dataclass
class SlicerResult:
    available: bool
    sliced: bool                 # True if slicer ran AND produced gcode
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    print_time_s: Optional[int] = None
    filament_mm: Optional[float] = None
    binary_used: Optional[str] = None
    raw_stderr: str = ""

    @property
    def printable(self) -> bool:
        """We treat a clean slice (no errors) as printable."""
        return self.available and self.sliced and not self.errors


def find_slicer(explicit_path: Optional[str] = None) -> Optional[Path]:
    """Locate a usable slicer binary. Returns None if nothing found."""
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p
    for name in CANDIDATE_BINARIES:
        which = shutil.which(name)
        if which:
            return Path(which)
    for cand in CANDIDATE_APP_PATHS:
        if Path(cand).exists():
            return Path(cand)
    return None


def slice_stl(stl_path: Path, out_dir: Optional[Path] = None,
              slicer_path: Optional[str] = None,
              timeout_s: int = 60) -> SlicerResult:
    """Run slicer headless on `stl_path`, capture output."""
    binary = find_slicer(slicer_path)
    if binary is None:
        return SlicerResult(available=False, sliced=False,
                            errors=["no slicer found on PATH"])

    stl_path = Path(stl_path)
    if not stl_path.exists():
        return SlicerResult(available=True, sliced=False,
                            errors=["STL does not exist"],
                            binary_used=str(binary))

    out_dir = Path(out_dir) if out_dir else stl_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    gcode_path = out_dir / (stl_path.stem + ".gcode")

    # PrusaSlicer / Orca / SuperSlicer share most CLI flags from PrusaSlicer
    cmd = [
        str(binary),
        "--export-gcode",
        "--output", str(gcode_path),
        "--info",
        str(stl_path),
    ]
    log.debug(f"slicer cmd: {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            text=True,
            env={**os.environ, "DISPLAY": ":0"},
        )
    except subprocess.TimeoutExpired:
        return SlicerResult(available=True, sliced=False,
                            errors=[f"slicer timeout after {timeout_s}s"],
                            binary_used=str(binary))
    except Exception as e:
        return SlicerResult(available=True, sliced=False,
                            errors=[f"slicer launch failed: {e}"],
                            binary_used=str(binary))

    raw = proc.stderr or ""
    sliced = proc.returncode == 0 and gcode_path.exists()

    warnings, errors = [], []
    for line in raw.splitlines():
        line = line.strip()
        low = line.lower()
        if not line:
            continue
        if "warning" in low:
            warnings.append(line)
        elif "error" in low or "failed" in low:
            errors.append(line)

    if not sliced and not errors:
        errors.append(f"slicer returncode={proc.returncode}")

    # Extract print time / filament if --info was respected
    print_time = None
    filament_mm = None
    info_text = (proc.stdout or "") + raw
    m = re.search(r"estimated printing time[^=:]*[:=]\s*([0-9hms ]+)",
                  info_text, re.IGNORECASE)
    if m:
        # Convert "2h 5m 33s" -> seconds (rough)
        s_total = 0
        for n, unit in re.findall(r"(\d+)\s*([hms])", m.group(1).lower()):
            mult = {"h": 3600, "m": 60, "s": 1}[unit]
            s_total += int(n) * mult
        if s_total > 0:
            print_time = s_total
    m = re.search(r"filament used[^=:]*[:=]\s*([\d.]+)\s*mm", info_text,
                  re.IGNORECASE)
    if m:
        filament_mm = float(m.group(1))

    # Clean up gcode file — we don't need it, just the diagnostic
    try:
        if gcode_path.exists():
            gcode_path.unlink()
    except Exception:
        pass

    return SlicerResult(
        available=True,
        sliced=sliced,
        warnings=warnings,
        errors=errors,
        print_time_s=print_time,
        filament_mm=filament_mm,
        binary_used=str(binary),
        raw_stderr=raw[:2000],
    )
